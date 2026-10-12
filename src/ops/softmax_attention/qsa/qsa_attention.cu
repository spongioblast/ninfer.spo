// QSA attention over the selected keys: the batched and split kernels for each stored K/V code,
// and the split reduction.

#include "ops/softmax_attention/qsa/qsa_common.cuh"

namespace ninfer::ops::flash_next_qsa_detail {
namespace {

// Eight quantized codes of one paged K/V row and the scale that applies to them.
// The scale stays raw FP16 so its load remains in flight until the codes are decoded.
struct KvCodes8 {
    uint2 codes;
    __half scale;
};

template <KvCode Code>
__device__ __forceinline__ KvCodes8 load_kv_codes8(const void* pages, const __half* scales,
                                                   std::int64_t row, int d) {
    static_assert(Code != KvCode::Bf16);
    const std::int64_t offset = d + static_cast<std::int64_t>(kHeadDim) * row;
    const std::int64_t scale  = Code == KvCode::Fp8
                                    ? row
                                    : row * kKVCacheInt8Groups + d / kKVCacheInt8Group;
    return {*reinterpret_cast<const uint2*>(static_cast<const std::uint8_t*>(pages) + offset),
            scales[scale]};
}

// FP8 and INT8 decode every code exactly and apply the row (FP8) or 64-value group (INT8) scale
// in FP32 before the BF16 boundary.
template <KvCode Code>
__device__ __forceinline__ uint4 decode_kv_codes8(const KvCodes8& value) {
    static_assert(Code != KvCode::Bf16);
    const float scale = __half2float(value.scale);
    if constexpr (Code == KvCode::Fp8) {
        const std::uint32_t words[2] = {value.codes.x, value.codes.y};
        std::uint32_t packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __nv_fp8x2_e4m3 pair;
            pair.__x             = static_cast<std::uint16_t>(words[i / 2] >> (16 * (i % 2)));
            const float2 decoded = static_cast<float2>(pair);
            packed[i]            = pack_bf16x2(decoded.x * scale, decoded.y * scale);
        }
        return make_uint4(packed[0], packed[1], packed[2], packed[3]);
    } else {
        // Same arithmetic as kv_cache_int8_dequant_i8x8_from, unpacked from registers.
        const std::uint32_t words[2] = {value.codes.x, value.codes.y};
        std::uint32_t packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const std::uint32_t word = words[i / 2] >> (16 * (i % 2));
            const float x0 = static_cast<float>(static_cast<std::int8_t>(word & 0xFFU)) * scale;
            const float x1 =
                static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xFFU)) * scale;
            packed[i] = pack_bf16x2(x0, x1);
        }
        return make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

// Eight BF16 values of paged K/V row `row` from dimension d.
template <KvCode Code>
__device__ __forceinline__ uint4 load_kv8(const void* pages, const __half* scales,
                                          std::int64_t row, int d) {
    if constexpr (Code == KvCode::Bf16) {
        return *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(pages) + d +
                                               static_cast<std::int64_t>(kHeadDim) * row);
    } else {
        return decode_kv_codes8<Code>(load_kv_codes8<Code>(pages, scales, row, d));
    }
}

// Positions and paged K/V rows of one tile's keys (-1 when absent). Each key's page is resolved
// once here instead of once per staged vector.
__device__ __forceinline__ void stage_selected_qsa_rows(
    int* staged_positions, int* staged_rows, const int* indices, const int* tables,
    int logical_pages, int table_row, int kv_head, int token, int tile_begin, int item_count) {
    for (int row = static_cast<int>(threadIdx.x); row < kPrefillColumns;
         row += static_cast<int>(blockDim.x)) {
        const int item     = tile_begin + row;
        const int position = item < item_count
            ? (indices == nullptr
                   ? item
                   : indices[item + static_cast<std::int64_t>(kOutputWidth) * token])
            : -1;
        staged_positions[row] = position;
        staged_rows[row]      = position >= 0
            ? position % kPagedKVPageSize +
                  kPagedKVPageSize *
                      (kv_head + kKvHeads * physical_page(tables, logical_pages, table_row,
                                                          position))
            : -1;
    }
    __syncthreads();
}

__device__ __forceinline__ void stage_selected_qsa_tile(
    __nv_bfloat16* destination, int* staged_positions, int* staged_rows, const void* cache,
    const int* indices, const int* tables, int logical_pages, int table_row, int kv_head,
    int token, int tile_begin, int item_count) {
    stage_selected_qsa_rows(staged_positions, staged_rows, indices, tables, logical_pages,
                            table_row, kv_head, token, tile_begin, item_count);
    constexpr int kVectorsPerRow = kHeadDim / 8;
    for (int vector = static_cast<int>(threadIdx.x); vector < kPrefillColumns * kVectorsPerRow;
         vector += static_cast<int>(blockDim.x)) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* output = destination + row * kHeadDim + causal_prompt_swz(row, d);
        const int paged_row = staged_rows[row];
        if (paged_row >= 0) {
            cp_async<16, Cache::cg>(output, static_cast<const __nv_bfloat16*>(cache) + d +
                                                static_cast<std::int64_t>(kHeadDim) * paged_row);
        } else {
            store_vec(output, make_int4(0, 0, 0, 0));
        }
    }
}

// Quantized tiles are staged in two steps so their global loads overlap attention work like the
// BF16 cp.async route: issue loads the tile's codes into registers, complete decodes them into
// shared memory before the barrier that publishes the tile.
struct PendingKvTile {
    static constexpr int kVectors = kPrefillColumns * (kHeadDim / 8) / 128;
    KvCodes8 values[kVectors];
    bool present[kVectors];
};

template <KvCode Code>
__device__ __forceinline__ void issue_selected_qsa_tile(
    PendingKvTile& pending, int* staged_positions, int* staged_rows, const void* cache,
    const __half* scales, const int* indices, const int* tables, int logical_pages,
    int table_row, int kv_head, int token, int tile_begin, int item_count) {
    stage_selected_qsa_rows(staged_positions, staged_rows, indices, tables, logical_pages,
                            table_row, kv_head, token, tile_begin, item_count);
    constexpr int kVectorsPerRow = kHeadDim / 8;
#pragma unroll
    for (int i = 0; i < PendingKvTile::kVectors; ++i) {
        const int vector    = static_cast<int>(threadIdx.x) + 128 * i;
        const int paged_row = staged_rows[vector / kVectorsPerRow];
        pending.present[i]  = paged_row >= 0;
        if (paged_row >= 0) {
            pending.values[i] = load_kv_codes8<Code>(cache, scales, paged_row,
                                                     (vector % kVectorsPerRow) * 8);
        }
    }
}

template <KvCode Code>
__device__ __forceinline__ void complete_selected_qsa_tile(const PendingKvTile& pending,
                                                           __nv_bfloat16* destination) {
    constexpr int kVectorsPerRow = kHeadDim / 8;
#pragma unroll
    for (int i = 0; i < PendingKvTile::kVectors; ++i) {
        const int vector = static_cast<int>(threadIdx.x) + 128 * i;
        const int row    = vector / kVectorsPerRow;
        const int d      = (vector % kVectorsPerRow) * 8;
        store_vec(destination + row * kHeadDim + causal_prompt_swz(row, d),
                  pending.present[i] ? decode_kv_codes8<Code>(pending.values[i])
                                     : make_uint4(0, 0, 0, 0));
    }
}

// Four warps share QK/softmax work for the twelve query heads of one KV head and split the
// 256-value output dimension. This preserves the compact 16-key tile while increasing resident
// PV warp-level work at the same dynamic shared-memory footprint.
template <KvCode Code>
__device__ __forceinline__ void selected_attention_batched_body(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits,
    float* partial_maximum, float* partial_denominator, float* partial_numerator,
    __nv_bfloat16* output) {
    constexpr int kHeadsPerKv = kQueryHeads / kKvHeads;
    constexpr int kQkTiles = kPrefillColumns / 8;
    constexpr int kQkSteps = kHeadDim / 16;
    constexpr int kValueWarps = 4;
    constexpr int kPvTiles = kHeadDim / (8 * kValueWarps);
    constexpr int kPvSteps = kPrefillColumns / 16;
    constexpr float kScaleLog2 = (1.0F / 16.0F) * 1.4426950408889634F;
    const int token = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int split = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (column >= valid_columns[batch_lane]) {
        if (num_splits > 1) {
            const int gid = lane >> 2;
            const int lid = lane & 3;
            const std::int64_t split_base = static_cast<std::int64_t>(kQueryHeads) *
                (token + static_cast<std::int64_t>(tokens) * split);
            if (warp == 0 && lid == 0) {
                if (gid < kHeadsPerKv) {
                    partial_maximum[split_base + kv_head * kHeadsPerKv + gid] = -INFINITY;
                    partial_denominator[split_base + kv_head * kHeadsPerKv + gid] = 0.0F;
                }
                if (gid + 8 < kHeadsPerKv) {
                    partial_maximum[split_base + kv_head * kHeadsPerKv + gid + 8] = -INFINITY;
                    partial_denominator[split_base + kv_head * kHeadsPerKv + gid + 8] = 0.0F;
                }
            }
            for (int nt = 0; nt < kPvTiles; ++nt) {
                const int d = warp * (kHeadDim / kValueWarps) + nt * 8 + 2 * lid;
                if (gid < kHeadsPerKv) {
                    const int head = kv_head * kHeadsPerKv + gid;
                    const std::int64_t base = kHeadDim * (split_base + head);
                    partial_numerator[base + d] = 0.0F;
                    partial_numerator[base + d + 1] = 0.0F;
                }
                if (gid + 8 < kHeadsPerKv) {
                    const int head = kv_head * kHeadsPerKv + gid + 8;
                    const std::int64_t base = kHeadDim * (split_base + head);
                    partial_numerator[base + d] = 0.0F;
                    partial_numerator[base + d + 1] = 0.0F;
                }
            }
        }
        return;
    }

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* query_shared = shared;
    __nv_bfloat16* key_shared = query_shared + kPrefillRows * kHeadDim;
    __nv_bfloat16* value_shared = key_shared + kPrefillColumns * kHeadDim;
    int* staged_positions = reinterpret_cast<int*>(
        value_shared + kPrefillColumns * kHeadDim);
    int* staged_rows = staged_positions + kPrefillColumns;

    constexpr int kVectorsPerRow = kHeadDim / 8;
    for (int vector = tid; vector < kPrefillRows * kVectorsPerRow;
         vector += static_cast<int>(blockDim.x)) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* destination = query_shared +
            row * kHeadDim + causal_prompt_swz(row, d);
        if (row < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + row;
            const __nv_bfloat16* source = query + d + kHeadDim *
                (head + static_cast<std::int64_t>(kQueryHeads) * token);
            cp_async<16, Cache::cg>(destination, source);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_row = (lane & 7) + ((a_matrix & 1) << 3);
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) << 3;
    const unsigned q_base = smem_addr(query_shared) +
                            static_cast<unsigned>(a_row * 512);
    const unsigned q_a = static_cast<unsigned>((a_matrix >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>((lane & 7) << 4);
    const unsigned k_base = smem_addr(key_shared) + static_cast<unsigned>(b_row * 512) +
                            (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_a = static_cast<unsigned>((b_k >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_row << 4);
    const unsigned v_base = smem_addr(value_shared) +
                            static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                            static_cast<unsigned>(b_row * 512);
    const unsigned v_a = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r = static_cast<unsigned>(b_row << 4);

    __shared__ unsigned probability_shared[32][kPvSteps][4];
    __shared__ float previous_scale_shared[32][2];
    __shared__ float inverse_shared[32][2];
    float accumulator[kPvTiles][4] = {};
    float maximum0 = -INFINITY;
    float maximum1 = -INFINITY;
    float denominator0 = 0.0F;
    float denominator1 = 0.0F;
    const int sequence_length = cache_positions[token] + 1;
    const int item_count = indices == nullptr ? sequence_length : kOutputWidth;
    const int table_row = table_rows[batch_lane];
    const int tile_count = (item_count + kPrefillColumns - 1) / kPrefillColumns;
    const int tile_begin = split * tile_count / num_splits;
    const int tile_end = (split + 1) * tile_count / num_splits;

    constexpr bool kCodes = Code != KvCode::Bf16;
    PendingKvTile pending_key{};
    PendingKvTile pending_value{};
    cp_commit();
    if constexpr (kCodes) {
        issue_selected_qsa_tile<Code>(pending_key, staged_positions, staged_rows, key_pages,
                                      key_scales, indices, tables, logical_pages, table_row,
                                      kv_head, token, tile_begin * kPrefillColumns, item_count);
    } else {
        stage_selected_qsa_tile(key_shared, staged_positions, staged_rows, key_pages, indices,
                                tables, logical_pages, table_row, kv_head, token,
                                tile_begin * kPrefillColumns, item_count);
    }
    cp_commit();
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        cp_wait<0>();
        if constexpr (kCodes) { complete_selected_qsa_tile<Code>(pending_key, key_shared); }
        __syncthreads();
        if constexpr (kCodes) {
            issue_selected_qsa_tile<Code>(pending_value, staged_positions, staged_rows, value_pages,
                                          value_scales, indices, tables, logical_pages,
                                          table_row, kv_head, token, tile * kPrefillColumns,
                                          item_count);
        } else {
            stage_selected_qsa_tile(value_shared, staged_positions, staged_rows, value_pages,
                                    indices, tables, logical_pages, table_row, kv_head, token,
                                    tile * kPrefillColumns, item_count);
        }
        cp_commit();

        unsigned probabilities[kPvSteps][4] = {};
        float previous_scale0 = 0.0F;
        float previous_scale1 = 0.0F;
        if (warp == 0) {
            float scores[kQkTiles][4] = {};
            unsigned af[2][4];
            unsigned bf[2][kQkTiles][2];
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        causal_prompt_swz_addr(q_base, 0U, q_a, q_r));
#pragma unroll
            for (int nt = 0; nt < kQkTiles; nt += 2) {
                ldmatrix_x4(bf[0][nt][0], bf[0][nt][1], bf[0][nt + 1][0], bf[0][nt + 1][1],
                            causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 4096),
                                                   0U, k_a, k_r));
            }
#pragma unroll
            for (int step = 0; step < kQkSteps; ++step) {
                const int current = step & 1;
                const int next = current ^ 1;
                if (step + 1 < kQkSteps) {
                    const unsigned contraction = static_cast<unsigned>((step + 1) << 5);
                    ldmatrix_x4(af[next][0], af[next][1], af[next][2], af[next][3],
                                causal_prompt_swz_addr(q_base, contraction, q_a, q_r));
#pragma unroll
                    for (int nt = 0; nt < kQkTiles; nt += 2) {
                        ldmatrix_x4(
                            bf[next][nt][0], bf[next][nt][1],
                            bf[next][nt + 1][0], bf[next][nt + 1][1],
                            causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 4096),
                                                   contraction, k_a, k_r));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < kQkTiles; ++nt) {
                    mma_bf16(scores[nt][0], scores[nt][1], scores[nt][2], scores[nt][3],
                             af[current][0], af[current][1], af[current][2], af[current][3],
                             bf[current][nt][0], bf[current][nt][1]);
                }
            }

            float block_maximum0 = -INFINITY;
            float block_maximum1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < kQkTiles; ++nt) {
                const int key0 = nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool valid0 = staged_positions[key0] >= 0;
                const bool valid1 = staged_positions[key1] >= 0;
                scores[nt][0] = gid < kHeadsPerKv && valid0 ? scores[nt][0] : -INFINITY;
                scores[nt][1] = gid < kHeadsPerKv && valid1 ? scores[nt][1] : -INFINITY;
                scores[nt][2] = gid + 8 < kHeadsPerKv && valid0 ? scores[nt][2] : -INFINITY;
                scores[nt][3] = gid + 8 < kHeadsPerKv && valid1 ? scores[nt][3] : -INFINITY;
                block_maximum0 = fmaxf(block_maximum0, fmaxf(scores[nt][0], scores[nt][1]));
                block_maximum1 = fmaxf(block_maximum1, fmaxf(scores[nt][2], scores[nt][3]));
            }
            block_maximum0 = warp_max<4>(block_maximum0, 0xffffffffU);
            block_maximum1 = warp_max<4>(block_maximum1, 0xffffffffU);
            const float next_maximum0 = fmaxf(maximum0, block_maximum0);
            const float next_maximum1 = fmaxf(maximum1, block_maximum1);
            const float scaled_maximum0 = next_maximum0 > -INFINITY
                ? next_maximum0 * kScaleLog2 : 0.0F;
            const float scaled_maximum1 = next_maximum1 > -INFINITY
                ? next_maximum1 * kScaleLog2 : 0.0F;
            previous_scale0 = next_maximum0 > -INFINITY
                ? exp2_approx(__fmaf_rn(maximum0, kScaleLog2, -scaled_maximum0)) : 1.0F;
            previous_scale1 = next_maximum1 > -INFINITY
                ? exp2_approx(__fmaf_rn(maximum1, kScaleLog2, -scaled_maximum1)) : 1.0F;
            float block_sum0 = 0.0F;
            float block_sum1 = 0.0F;
#pragma unroll
            for (int nt = 0; nt < kQkTiles; ++nt) {
                const float p00 = scores[nt][0] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][0], kScaleLog2, -scaled_maximum0)) : 0.0F;
                const float p01 = scores[nt][1] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][1], kScaleLog2, -scaled_maximum0)) : 0.0F;
                const float p10 = scores[nt][2] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][2], kScaleLog2, -scaled_maximum1)) : 0.0F;
                const float p11 = scores[nt][3] > -INFINITY
                    ? exp2_approx(__fmaf_rn(scores[nt][3], kScaleLog2, -scaled_maximum1)) : 0.0F;
                block_sum0 += p00 + p01;
                block_sum1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    probabilities[pk][0] = pack_bf16x2(p00, p01);
                    probabilities[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    probabilities[pk][2] = pack_bf16x2(p00, p01);
                    probabilities[pk][3] = pack_bf16x2(p10, p11);
                }
            }
            denominator0 = __fmaf_rn(denominator0, previous_scale0, block_sum0);
            denominator1 = __fmaf_rn(denominator1, previous_scale1, block_sum1);
            maximum0 = next_maximum0;
            maximum1 = next_maximum1;
#pragma unroll
            for (int step = 0; step < kPvSteps; ++step) {
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    probability_shared[lane][step][item] = probabilities[step][item];
                }
            }
            previous_scale_shared[lane][0] = previous_scale0;
            previous_scale_shared[lane][1] = previous_scale1;
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < kPvTiles; ++nt) {
            accumulator[nt][0] *= previous_scale_shared[lane][0];
            accumulator[nt][1] *= previous_scale_shared[lane][0];
            accumulator[nt][2] *= previous_scale_shared[lane][1];
            accumulator[nt][3] *= previous_scale_shared[lane][1];
        }

        cp_wait<0>();
        if constexpr (kCodes) { complete_selected_qsa_tile<Code>(pending_value, value_shared); }
        __syncthreads();
        if (tile + 1 < tile_end) {
            if constexpr (kCodes) {
                issue_selected_qsa_tile<Code>(pending_key, staged_positions, staged_rows, key_pages,
                                              key_scales, indices, tables, logical_pages,
                                              table_row, kv_head, token,
                                              (tile + 1) * kPrefillColumns, item_count);
            } else {
                stage_selected_qsa_tile(key_shared, staged_positions, staged_rows, key_pages,
                                        indices, tables, logical_pages, table_row, kv_head,
                                        token, (tile + 1) * kPrefillColumns, item_count);
            }
            cp_commit();
        }
        constexpr int kPvHalf = kPvTiles / 2;
        constexpr int kPvLoads = kPvSteps * kPvHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_prompt_swz_addr(
                          v_base, static_cast<unsigned>((warp * kPvTiles) << 4), v_a, v_r));
#pragma unroll
        for (int item = 0; item < kPvLoads; ++item) {
            const int step = item / kPvHalf;
            const int nt = (item % kPvHalf) * 2;
            const int current = item & 1;
            const int next = current ^ 1;
            if (item + 1 < kPvLoads) {
                const int next_step = (item + 1) / kPvHalf;
                const int next_nt = warp * kPvTiles + ((item + 1) % kPvHalf) * 2;
                ldmatrix_x4_t(
                    vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                    causal_prompt_swz_addr(
                        v_base + static_cast<unsigned>(next_step * 8192),
                        static_cast<unsigned>(next_nt << 4), v_a, v_r));
            }
            mma_bf16(accumulator[nt][0], accumulator[nt][1],
                     accumulator[nt][2], accumulator[nt][3],
                     probability_shared[lane][step][0], probability_shared[lane][step][1],
                     probability_shared[lane][step][2], probability_shared[lane][step][3],
                     vf[current][0], vf[current][1]);
            mma_bf16(accumulator[nt + 1][0], accumulator[nt + 1][1],
                     accumulator[nt + 1][2], accumulator[nt + 1][3],
                     probability_shared[lane][step][0], probability_shared[lane][step][1],
                     probability_shared[lane][step][2], probability_shared[lane][step][3],
                     vf[current][2], vf[current][3]);
        }
    }

    if (warp == 0) {
        denominator0 = ninfer::ops::warp_sum<4>(denominator0, 0xffffffffU);
        denominator1 = ninfer::ops::warp_sum<4>(denominator1, 0xffffffffU);
        inverse_shared[lane][0] = denominator0 > 0.0F ? __frcp_rn(denominator0) : 0.0F;
        inverse_shared[lane][1] = denominator1 > 0.0F ? __frcp_rn(denominator1) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < kPvTiles; ++nt) {
        const int d = warp * (kHeadDim / kValueWarps) + nt * 8 + 2 * lid;
        if (gid < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid;
            if (num_splits == 1) {
                *reinterpret_cast<unsigned*>(&output[d + kHeadDim *
                    (head + static_cast<std::int64_t>(kQueryHeads) * token)]) =
                    pack_bf16x2(accumulator[nt][0] * inverse_shared[lane][0],
                                accumulator[nt][1] * inverse_shared[lane][0]);
            } else {
                const std::int64_t stats = head + static_cast<std::int64_t>(kQueryHeads) *
                    (token + static_cast<std::int64_t>(tokens) * split);
                const std::int64_t base = kHeadDim * stats;
                partial_numerator[base + d] = accumulator[nt][0];
                partial_numerator[base + d + 1] = accumulator[nt][1];
            }
        }
        if (gid + 8 < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid + 8;
            if (num_splits == 1) {
                *reinterpret_cast<unsigned*>(&output[d + kHeadDim *
                    (head + static_cast<std::int64_t>(kQueryHeads) * token)]) =
                    pack_bf16x2(accumulator[nt][2] * inverse_shared[lane][1],
                                accumulator[nt][3] * inverse_shared[lane][1]);
            } else {
                const std::int64_t stats = head + static_cast<std::int64_t>(kQueryHeads) *
                    (token + static_cast<std::int64_t>(tokens) * split);
                const std::int64_t base = kHeadDim * stats;
                partial_numerator[base + d] = accumulator[nt][2];
                partial_numerator[base + d + 1] = accumulator[nt][3];
            }
        }
    }
    if (num_splits > 1 && warp == 0 && lid == 0) {
        const std::int64_t split_base = static_cast<std::int64_t>(kQueryHeads) *
            (token + static_cast<std::int64_t>(tokens) * split);
        if (gid < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid;
            partial_maximum[split_base + head] = maximum0 * (1.0F / 16.0F);
            partial_denominator[split_base + head] = denominator0;
        }
        if (gid + 8 < kHeadsPerKv) {
            const int head = kv_head * kHeadsPerKv + gid + 8;
            partial_maximum[split_base + head] = maximum1 * (1.0F / 16.0F);
            partial_denominator[split_base + head] = denominator1;
        }
    }
}

template <KvCode Code>
__device__ __forceinline__ void selected_attention_split_body(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales,
    const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits) {
    constexpr int kTile = 16;
    constexpr float kLog2E = 1.4426950408889634F;
    const int head_group = static_cast<int>(blockIdx.x);
    const int head = head_group * kSplitHeadsPerBlock +
                     (static_cast<int>(threadIdx.x) >> 5);
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int token = static_cast<int>(blockIdx.y);
    const int split = static_cast<int>(blockIdx.z);
    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    const std::int64_t stats_offset = head + static_cast<std::int64_t>(kQueryHeads) *
        (token + static_cast<std::int64_t>(tokens) * split);
    const std::int64_t numerator_base = kHeadDim * stats_offset;
    if (column >= valid_columns[batch_lane]) {
        if (lane_id == 0) {
            partial_maximum[stats_offset] = -INFINITY;
            partial_denominator[stats_offset] = 0.0F;
        }
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            partial_numerator[numerator_base + lane_id + 32 * item] = 0.0F;
        }
        return;
    }

    float q[8];
    float numerator[8] = {};
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int d = lane_id + 32 * item;
        q[item] = __bfloat162float(query[d + kHeadDim *
            (head + static_cast<std::int64_t>(kQueryHeads) * token)]);
    }
    float maximum = -INFINITY;
    float denominator = 0.0F;
    const int sequence_length = cache_positions[token] + 1;
    const int item_count = indices == nullptr ? sequence_length : kOutputWidth;
    // Partition whole 16-key tiles exactly as vLLM does. Dividing the raw item
    // count starts most splits at a different key and changes the reduction tree.
    const int num_tiles = (item_count + kTile - 1) / kTile;
    const int item_begin = (split * num_tiles / num_splits) * kTile;
    const int item_end = min(item_count,
                             ((split + 1) * num_tiles / num_splits) * kTile);
    const int table_row = table_rows[batch_lane];
    __shared__ __align__(16) __nv_bfloat16 staged_key[kTile][kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 staged_value[kTile][kHeadDim];
    __shared__ int staged_positions[kTile];
    __shared__ int staged_rows[kTile];  // paged KV row of each key, -1 when absent
    for (int tile_begin = item_begin; tile_begin < item_end; tile_begin += kTile) {
        const int tile_size = min(kTile, item_end - tile_begin);
        bool present = false;
        if (threadIdx.x < tile_size) {
            const int position = indices == nullptr
                ? tile_begin + static_cast<int>(threadIdx.x)
                : indices[tile_begin + static_cast<int>(threadIdx.x) +
                          static_cast<std::int64_t>(kOutputWidth) * token];
            present = position >= 0;
            staged_positions[threadIdx.x] = position;
            staged_rows[threadIdx.x] = present
                ? position % kPagedKVPageSize +
                      kPagedKVPageSize *
                          (kv_head + kKvHeads * physical_page(tables, logical_pages, table_row,
                                                              position))
                : -1;
        }
        // A tile without selected keys contributes nothing (its maximum stays -inf).
        if (!__syncthreads_or(present)) { continue; }
        {
            // Eight values per load; all loads of a thread are independent.
            constexpr int kVectors = kHeadDim / 8;
#pragma unroll
            for (int i = 0; i < kTile * kVectors / 128; ++i) {
                const int vector = static_cast<int>(threadIdx.x) + 128 * i;
                const int item = vector / kVectors;
                const int d = 8 * (vector % kVectors);
                const int row = item < tile_size ? staged_rows[item] : -1;
                uint4 key = make_uint4(0, 0, 0, 0);
                uint4 value = make_uint4(0, 0, 0, 0);
                if (row >= 0) {
                    key = load_kv8<Code>(key_pages, key_scales, row, d);
                    value = load_kv8<Code>(value_pages, value_scales, row, d);
                }
                *reinterpret_cast<uint4*>(&staged_key[item][d]) = key;
                *reinterpret_cast<uint4*>(&staged_value[item][d]) = value;
            }
        }
        __syncthreads();
        float scores[kTile];
        float tile_maximum = -INFINITY;
        for (int position = 0; position < tile_size; ++position) {
            float score = 0.0F;
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                score += q[item] *
                         __bfloat162float(staged_key[position][lane_id + 32 * item]);
            }
            for (int offset = 16; offset != 0; offset >>= 1) {
                score += __shfl_down_sync(0xffffffffU, score, offset);
            }
            score = staged_positions[position] >= 0
                ? __shfl_sync(0xffffffffU, score, 0) * (1.0F / 16.0F)
                : -INFINITY;
            scores[position] = score;
            tile_maximum = fmaxf(tile_maximum, score);
        }
        if (tile_maximum == -INFINITY) {
            __syncthreads();
            continue;
        }
        const float next_maximum = fmaxf(maximum, tile_maximum);
        const float scaled_next_maximum = next_maximum * kLog2E;
        const float previous_scale = exp2_approx(
            __fmaf_rn(maximum, kLog2E, -scaled_next_maximum));
        denominator *= previous_scale;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            numerator[item] *= previous_scale;
        }
        for (int position = 0; position < tile_size; ++position) {
            if (staged_positions[position] < 0) { continue; }
            const float current_scale = exp2_approx(
                __fmaf_rn(scores[position], kLog2E, -scaled_next_maximum));
            const float current_probability = __bfloat162float(
                __float2bfloat16_rn(current_scale));
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                numerator[item] += current_probability *
                    __bfloat162float(staged_value[position][lane_id + 32 * item]);
            }
            denominator += current_scale;
        }
        maximum = next_maximum;
        __syncthreads();
    }
    if (lane_id == 0) {
        partial_maximum[stats_offset] = maximum;
        partial_denominator[stats_offset] = denominator;
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        partial_numerator[numerator_base + lane_id + 32 * item] = numerator[item];
    }
}

} // namespace

// Three CTAs per SM, as the BF16 route reaches without a bound; the quantized routes otherwise
// spend registers on pending codes and drop to two.
template <KvCode Code>
__launch_bounds__(128, 3) __global__ void selected_attention_batched_kernel(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, __nv_bfloat16* output) {
    selected_attention_batched_body<Code>(
        query, key_pages, value_pages, key_scales, value_scales, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, num_splits, partial_maximum,
        partial_denominator, partial_numerator, output);
}

template <KvCode Code>
__global__ void selected_attention_split_kernel(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits) {
    selected_attention_split_body<Code>(
        query, key_pages, value_pages, key_scales, value_scales, indices, tables, logical_pages,
        cache_positions, valid_columns, table_rows, width, tokens, partial_maximum,
        partial_denominator, partial_numerator, num_splits);
}

// One CTA per (query head, token), one thread per head dimension. Every element combines the
// splits in ascending order, so the result does not depend on the launch shape.
__global__ void __launch_bounds__(kHeadDim) reduce_selected_attention_splits_kernel(
    const float* partial_maximum, const float* partial_denominator,
    const float* partial_numerator, const int* valid_columns, int width, int tokens,
    __nv_bfloat16* output, int num_splits) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    const std::int64_t output_index =
        d + kHeadDim * (head + static_cast<std::int64_t>(kQueryHeads) * token);
    if (column >= valid_columns[batch_lane]) {
        output[output_index] = __float2bfloat16_rn(0.0F);
        return;
    }
    constexpr float kLog2E = 1.4426950408889634F;
    const std::int64_t split_stride = static_cast<std::int64_t>(kQueryHeads) * tokens;
    const std::int64_t stats_base = head + static_cast<std::int64_t>(kQueryHeads) * token;
    // Per-split rescale factors are shared by the head's 256 dimensions; empty splits get 0.
    __shared__ float split_scale[kMaxDecodeAttentionSplits];
    __shared__ float shared_denominator;
    if (d < 32) {
        float maximum = -INFINITY;
        for (int split = d; split < num_splits; split += 32) {
            maximum = fmaxf(maximum, partial_maximum[stats_base + split_stride * split]);
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
        }
        for (int split = d; split < num_splits; split += 32) {
            const std::int64_t stats_offset = stats_base + split_stride * split;
            split_scale[split] = partial_denominator[stats_offset] == 0.0F
                ? 0.0F
                : exp2_approx(__fmaf_rn(partial_maximum[stats_offset], kLog2E,
                                        -maximum * kLog2E));
        }
        __syncwarp();
        if (d == 0) {
            float denominator = 0.0F;
            for (int split = 0; split < num_splits; ++split) {
                const float scale = split_scale[split];
                if (scale != 0.0F) {
                    denominator +=
                        partial_denominator[stats_base + split_stride * split] * scale;
                }
            }
            shared_denominator = denominator;
        }
    }
    __syncthreads();
    float numerator = 0.0F;
#pragma unroll 16
    for (int split = 0; split < num_splits; ++split) {
        const float value = partial_numerator[kHeadDim * (stats_base + split_stride * split) + d];
        const float scale = split_scale[split];
        if (scale != 0.0F) { numerator += value * scale; }
    }
    const float denominator = shared_denominator;
    output[output_index] = __float2bfloat16_rn(numerator / denominator);
}

template __global__ void selected_attention_batched_kernel<KvCode::Bf16>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, __nv_bfloat16* output);
template __global__ void selected_attention_batched_kernel<KvCode::Fp8>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, __nv_bfloat16* output);
template __global__ void selected_attention_batched_kernel<KvCode::Int8>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, __nv_bfloat16* output);
template __global__ void selected_attention_split_kernel<KvCode::Bf16>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits);
template __global__ void selected_attention_split_kernel<KvCode::Fp8>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits);
template __global__ void selected_attention_split_kernel<KvCode::Int8>(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits);

} // namespace ninfer::ops::flash_next_qsa_detail
