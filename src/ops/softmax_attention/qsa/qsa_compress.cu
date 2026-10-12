// QSA preparation: query/gate split, text positions, index query, cache append, the Hadamard
// rotation of quantized-cache queries, and index-key group compression.

#include "ops/softmax_attention/qsa/qsa_common.cuh"

namespace ninfer::ops::flash_next_qsa_detail {
namespace {

struct FlashQsaKVGeometry {
    static constexpr int KVHeads = kKvHeads;
};

__device__ float block_sum(float value, float* partial) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < static_cast<int>(blockDim.x) / 32 ? partial[lane] : 0.0F;
        for (int offset = 16; offset != 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) { partial[0] = value; }
    }
    __syncthreads();
    return partial[0];
}

} // namespace

__global__ void split_query_gate_kernel(const __nv_bfloat16* packed,
                                        __nv_bfloat16* query,
                                        __nv_bfloat16* gate,
                                        int tokens) {
    const std::int64_t count = static_cast<std::int64_t>(kQueryHeads) * kHeadDim * tokens;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const int d = static_cast<int>(index % kHeadDim);
        const std::int64_t row = index / kHeadDim;
        const int head = static_cast<int>(row % kQueryHeads);
        const int token = static_cast<int>(row / kQueryHeads);
        const std::int64_t packed_base = static_cast<std::int64_t>(token) *
                                         (2 * kQueryHeads * kHeadDim) +
                                         static_cast<std::int64_t>(head) * 2 * kHeadDim;
        query[index] = packed[packed_base + d];
        gate[index] = packed[packed_base + kHeadDim + d];
    }
}

__global__ void expand_text_positions_kernel(const int* input, int* output, int tokens) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < 3 * tokens) { output[index] = input[index % tokens]; }
}

__global__ void prepare_index_query_kernel(const __nv_bfloat16* projected,
                                           const __nv_bfloat16* norm_weight,
                                           const int* positions, int tokens,
                                           __nv_bfloat16* output) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(token) * kIndexHeads * kIndexDim +
                              head * kIndexDim;
    const float x = __bfloat162float(projected[base + d]);
    __shared__ float partial[4];
    const float square = block_sum(x * x, partial);
    const float normalized = x * rsqrtf(square / kIndexDim + 1.0e-6F) *
                             (1.0F + __bfloat162float(norm_weight[d]));
    __shared__ float values[kIndexDim];
    values[d] = normalized;
    __syncthreads();
    float result = normalized;
    if (d < 64) {
        const int pair = d & 31;
        const int other = d < 32 ? d + 32 : d - 32;
        const int axis = pair % 3;
        const float angle = static_cast<float>(positions[token + tokens * axis]) *
                            powf(kTheta, -2.0F * pair / 64.0F);
        const float first = d < 32 ? values[d] : values[other];
        const float second = d < 32 ? values[other] : values[d];
        result = d < 32 ? first * cosf(angle) - second * sinf(angle)
                        : second * cosf(angle) + first * sinf(angle);
    }
    output[base + d] = __float2bfloat16_rn(result);
}

__global__ void append_cache_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
                                    const __nv_bfloat16* raw_index_key,
                                    const int* cache_positions, const int* rope_positions,
                                    const int* valid_columns, const int* table_rows,
                                    int width, int batch, int logical_pages,
                                    const int* tables, __nv_bfloat16* key_pages,
                                    __nv_bfloat16* value_pages, __nv_bfloat16* raw_pages,
                                    int* position_pages) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane = token / width;
    const int column = token - lane * width;
    if (lane >= batch || column >= valid_columns[lane]) { return; }
    const int position = cache_positions[token];
    const int page = physical_page(tables, logical_pages, table_rows[lane], position);
    const int page_offset = position % kPagedKVPageSize;
    for (int index = static_cast<int>(threadIdx.x); index < 2 * kKvHeads * kHeadDim;
         index += static_cast<int>(blockDim.x)) {
        const int is_value = index >= kKvHeads * kHeadDim;
        const int local = index - is_value * kKvHeads * kHeadDim;
        const int head = local / kHeadDim;
        const int d = local - head * kHeadDim;
        const std::int64_t cache_offset = d + static_cast<std::int64_t>(kHeadDim) *
            (page_offset + kPagedKVPageSize * (head + kKvHeads * page));
        const std::int64_t source = d + static_cast<std::int64_t>(kHeadDim) *
            (head + kKvHeads * token);
        if (is_value) {
            value_pages[cache_offset] = value[source];
        } else {
            key_pages[cache_offset] = key[source];
        }
    }
    for (int d = static_cast<int>(threadIdx.x); d < kIndexDim;
         d += static_cast<int>(blockDim.x)) {
        raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
                         (page_offset + kPagedKVPageSize * page)] =
            raw_index_key[d + static_cast<std::int64_t>(kIndexDim) * token];
    }
    if (threadIdx.x < 3) {
        position_pages[threadIdx.x + 3LL * (page_offset + kPagedKVPageSize * page)] =
            rope_positions[token + static_cast<std::int64_t>(width * batch) * threadIdx.x];
    }
}

__global__ void append_cache_fp8_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* raw_index_key, const int* cache_positions,
    const int* rope_positions, const int* valid_columns, const int* table_rows,
    int width, int batch, int logical_pages, const int* tables,
    std::uint8_t* key_pages, std::uint8_t* value_pages,
    __half* key_scales, __half* value_scales,
    __nv_bfloat16* raw_pages, int* position_pages) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane_index = token / width;
    const int column = token - lane_index * width;
    if (lane_index >= batch || column >= valid_columns[lane_index]) { return; }
    const int position = cache_positions[token];
    const int page = physical_page(tables, logical_pages, table_rows[lane_index], position);
    const int page_offset = position % kPagedKVPageSize;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp < kKvHeads) {
        kv_cache_append_full_fp8_row<FlashQsaKVGeometry>(
            key, value, key_pages, value_pages, key_scales, value_scales,
            token, warp, page, page_offset, lane);
    }
    for (int d = static_cast<int>(threadIdx.x); d < kIndexDim;
         d += static_cast<int>(blockDim.x)) {
        raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
                         (page_offset + kPagedKVPageSize * page)] =
            raw_index_key[d + static_cast<std::int64_t>(kIndexDim) * token];
    }
    if (threadIdx.x < 3) {
        position_pages[threadIdx.x + 3LL * (page_offset + kPagedKVPageSize * page)] =
            rope_positions[token + static_cast<std::int64_t>(width * batch) * threadIdx.x];
    }
}

__global__ void append_cache_int8_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* raw_index_key, const int* cache_positions,
    const int* rope_positions, const int* valid_columns, const int* table_rows,
    int width, int batch, int logical_pages, const int* tables,
    std::int8_t* key_pages, std::int8_t* value_pages,
    __half* key_scales, __half* value_scales,
    __nv_bfloat16* raw_pages, int* position_pages) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane_index = token / width;
    const int column = token - lane_index * width;
    if (lane_index >= batch || column >= valid_columns[lane_index]) { return; }
    const int position = cache_positions[token];
    const int page = physical_page(tables, logical_pages, table_rows[lane_index], position);
    const int page_offset = position % kPagedKVPageSize;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp < kKvHeads) {
        kv_cache_append_full_int8_row<FlashQsaKVGeometry>(
            key, value, key_pages, value_pages, key_scales, value_scales,
            token, warp, page, page_offset, lane);
    }
    for (int d = static_cast<int>(threadIdx.x); d < kIndexDim;
         d += static_cast<int>(blockDim.x)) {
        raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
                         (page_offset + kPagedKVPageSize * page)] =
            raw_index_key[d + static_cast<std::int64_t>(kIndexDim) * token];
    }
    if (threadIdx.x < 3) {
        position_pages[threadIdx.x + 3LL * (page_offset + kPagedKVPageSize * page)] =
            rope_positions[token + static_cast<std::int64_t>(width * batch) * threadIdx.x];
    }
}

__global__ void hadamard_rows_kernel(__nv_bfloat16* rows, int row_count) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= row_count || lane >= 32) { return; }
    float values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        values[item] = __bfloat162float(rows[static_cast<std::int64_t>(row) * kHeadDim +
                                             lane + 32 * item]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        rows[static_cast<std::int64_t>(row) * kHeadDim + lane + 32 * item] =
            __float2bfloat16_rn(values[item]);
    }
}

__global__ void compress_index_groups_kernel(__nv_bfloat16* raw_pages,
                                             const int* position_pages,
                                             const __nv_bfloat16* key_norm,
                                             const int* cache_positions,
                                             const int* valid_columns,
                                             const int* table_rows, int width, int batch,
                                             int logical_pages, const int* tables) {
    const int token = static_cast<int>(blockIdx.x);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (batch_lane >= batch || column >= valid_columns[batch_lane]) { return; }
    const int end_position = cache_positions[token];
    if ((end_position & (kRatio - 1)) != kRatio - 1) { return; }
    const int d = static_cast<int>(threadIdx.x);
    const int first_position = end_position - (kRatio - 1);
    const int table_row = table_rows[batch_lane];
    float pooled = 0.0F;
#pragma unroll
    for (int member = 0; member < kRatio; ++member) {
        const int position = first_position + member;
        const int page = physical_page(tables, logical_pages, table_row, position);
        pooled += __bfloat162float(raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
            (position % kPagedKVPageSize + kPagedKVPageSize * page)]);
    }
    pooled = __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
    __shared__ float partial[4];
    const float square = block_sum(pooled * pooled, partial);
    pooled *= rsqrtf(square / kIndexDim + 1.0e-6F) *
              (1.0F + __bfloat162float(key_norm[d]));
    __shared__ float values[kIndexDim];
    values[d] = pooled;
    __syncthreads();
    if (d < 64) {
        const int pair = d & 31;
        const int other = d < 32 ? d + 32 : d - 32;
        const int page = physical_page(tables, logical_pages, table_row, first_position);
        const int axis = pair % 3;
        const int rope_position = position_pages[axis + 3LL *
            (first_position % kPagedKVPageSize + kPagedKVPageSize * page)];
        const float angle = static_cast<float>(rope_position) *
                            powf(kTheta, -2.0F * pair / 64.0F);
        const float first = d < 32 ? values[d] : values[other];
        const float second = d < 32 ? values[other] : values[d];
        pooled = d < 32 ? first * cosf(angle) - second * sinf(angle)
                        : second * cosf(angle) + first * sinf(angle);
    }
    const int page = physical_page(tables, logical_pages, table_row, end_position);
    raw_pages[d + static_cast<std::int64_t>(kIndexDim) *
        (end_position % kPagedKVPageSize + kPagedKVPageSize * page)] =
        __float2bfloat16_rn(pooled);
}

} // namespace ninfer::ops::flash_next_qsa_detail
