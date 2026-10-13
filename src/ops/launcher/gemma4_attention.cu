#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarp = 32;
constexpr int kMaxThreads = 1024;
// Enough resident warps to cover the device several times over before splitting keys.
constexpr int kTargetWarps = 16384;

template <int E>
__device__ __forceinline__ void load_row(const __nv_bfloat16* row, float (&out)[E]) {
    static_assert(E % 8 == 0);
#pragma unroll
    for (int chunk = 0; chunk < E / 8; ++chunk) {
        const uint4 packed = reinterpret_cast<const uint4*>(row)[chunk];
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 value = __bfloat1622float2(pairs[pair]);
            out[chunk * 8 + pair * 2] = value.x;
            out[chunk * 8 + pair * 2 + 1] = value.y;
        }
    }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

// grid (tokens, kv_heads); block = group * splits warps. Warp (h, s) handles query head
// kvh * group + h over keys first + s, first + s + splits, ... with an online softmax, holding
// D / 32 contiguous dimensions per lane. The splits of one head are merged in shared memory.
template <int D>
__global__ void gemma4_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                        const __nv_bfloat16* __restrict__ k_pages,
                                        const __nv_bfloat16* __restrict__ v_pages,
                                        const std::int32_t* __restrict__ block_table,
                                        const std::int32_t* __restrict__ positions,
                                        __nv_bfloat16* __restrict__ out, std::int32_t q_heads,
                                        std::int32_t kv_heads, int group, int splits, int window,
                                        float scale, std::int32_t logical_pages,
                                        std::int32_t physical_pages) {
    constexpr int E = D / kWarp;
    extern __shared__ float shared[];
    float* merged = shared;                     // [group][D]
    float* stats = shared + group * D;          // [group][splits][2]

    const int token = blockIdx.x;
    const int kvh = blockIdx.y;
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int head_in_group = warp / splits;
    const int split = warp - head_in_group * splits;
    const int qh = kvh * group + head_in_group;
    const int qpos = positions[token];
    const int first = window == 0 ? 0 : max(0, qpos + 1 - window);

    for (int index = threadIdx.x; index < group * D; index += blockDim.x) { merged[index] = 0.0f; }

    float qv[E];
    load_row<E>(q + (static_cast<std::int64_t>(token) * q_heads + qh) * D + lane * E, qv);
#pragma unroll
    for (int dim = 0; dim < E; ++dim) { qv[dim] *= scale; }

    float acc[E];
#pragma unroll
    for (int dim = 0; dim < E; ++dim) { acc[dim] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int key = first + split; key <= qpos; key += splits) {
        const int page = key >> kPagedKVPageShift;
        if (page >= logical_pages) { continue; }
        const int physical = block_table[page];
        if (physical < 0 || physical >= physical_pages) { continue; }
        const std::int64_t base =
            static_cast<std::int64_t>(D) * kPagedKVPageSize *
                (static_cast<std::int64_t>(kvh) + static_cast<std::int64_t>(kv_heads) * physical) +
            static_cast<std::int64_t>(D) * (key & kPagedKVPageMask) + lane * E;
        float kv[E];
        load_row<E>(k_pages + base, kv);
        float dot = 0.0f;
#pragma unroll
        for (int dim = 0; dim < E; ++dim) { dot = fmaf(qv[dim], kv[dim], dot); }
        const float score = warp_sum(dot);
        const float next_max = fmaxf(running_max, score);
        const float rescale = expf(running_max - next_max);
        const float weight = expf(score - next_max);
        running_sum = running_sum * rescale + weight;
        running_max = next_max;
        load_row<E>(v_pages + base, kv);
#pragma unroll
        for (int dim = 0; dim < E; ++dim) { acc[dim] = fmaf(acc[dim], rescale, weight * kv[dim]); }
    }

    if (lane == 0) {
        stats[(head_in_group * splits + split) * 2] = running_max;
        stats[(head_in_group * splits + split) * 2 + 1] = running_sum;
    }
    __syncthreads();
    float head_max = -INFINITY;
    for (int other = 0; other < splits; ++other) {
        head_max = fmaxf(head_max, stats[(head_in_group * splits + other) * 2]);
    }
    float head_sum = 0.0f;
    for (int other = 0; other < splits; ++other) {
        const float other_max = stats[(head_in_group * splits + other) * 2];
        if (other_max != -INFINITY) {
            head_sum += stats[(head_in_group * splits + other) * 2 + 1] * expf(other_max - head_max);
        }
    }
    const bool contributes = running_max != -INFINITY && head_sum > 0.0f;
    const float factor = contributes ? expf(running_max - head_max) / head_sum : 0.0f;
    float* row = merged + head_in_group * D + lane * E;
    // Splits add in a fixed order so the result does not depend on scheduling.
    for (int turn = 0; turn < splits; ++turn) {
        if (turn == split && contributes) {
#pragma unroll
            for (int dim = 0; dim < E; ++dim) { row[dim] += acc[dim] * factor; }
        }
        __syncthreads();
    }
    __nv_bfloat16* destination = out + (static_cast<std::int64_t>(token) * q_heads) * D +
                                 static_cast<std::int64_t>(kvh) * group * D;
    for (int index = threadIdx.x; index < group * D; index += blockDim.x) {
        destination[index] = __float2bfloat16_rn(merged[index]);
    }
}

template <int D>
void launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
            const Tensor& block_table, const Tensor& positions, int window, float scale,
            Tensor& out, cudaStream_t stream) {
    const int q_heads = q.ne[1];
    const int tokens = q.ne[2];
    const int kv_heads = k_pages.ne[2];
    const int group = q_heads / kv_heads;
    const int max_splits = std::max(1, kMaxThreads / (kWarp * group));
    int splits = 1;
    while (splits * 2 <= max_splits && tokens * q_heads * splits < kTargetWarps) { splits *= 2; }
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(kv_heads));
    const int threads = kWarp * group * splits;
    const std::size_t shared =
        sizeof(float) * (static_cast<std::size_t>(group) * D + static_cast<std::size_t>(group) * splits * 2);
    if (shared > 48U * 1024U) {
        throw std::invalid_argument("gemma4_attention: query group is too wide for shared memory");
    }
    gemma4_attention_kernel<D><<<grid, threads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k_pages.data),
        static_cast<const __nv_bfloat16*>(v_pages.data),
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        q_heads, kv_heads, group, splits, window, scale, block_table.ne[0], k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gemma4_attention_launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                             const Tensor& block_table, const Tensor& positions, int window,
                             float scale, Tensor& out, cudaStream_t stream) {
    if (q.ne[1] / k_pages.ne[2] * kWarp > kMaxThreads) {
        throw std::invalid_argument("gemma4_attention: at most 32 query heads per KV head");
    }
    if (q.ne[0] == 256) {
        launch<256>(q, k_pages, v_pages, block_table, positions, window, scale, out, stream);
    } else {
        launch<512>(q, k_pages, v_pages, block_table, positions, window, scale, out, stream);
    }
}

} // namespace ninfer::ops::detail
