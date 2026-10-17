#include "ops/launcher/topk_columns.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops::detail {
namespace {

// Same ordering as ops::argmax: a larger value wins, and an equal value keeps the lower index.
__device__ __forceinline__ bool topk_better(float value, std::int32_t index, float best_value,
                                            std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

constexpr int kTopkThreads = 256;

struct TopkCand {
    float value;
    std::int32_t index;
};

template <int K>
__global__ void topk_columns_kernel(const __nv_bfloat16* logits, std::int32_t* ids,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(token) * physical_rows;
    const int tid            = static_cast<int>(threadIdx.x);

    float local_value[K];
    std::int32_t local_index[K];
#pragma unroll
    for (int slot = 0; slot < K; ++slot) {
        local_value[slot] = -CUDART_INF_F;
        local_index[slot] = INT32_MAX;
    }

    auto consider = [&](std::int32_t index, float value) {
        int place = K;
#pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            if (local_index[slot] == INT32_MAX ||
                topk_better(value, index, local_value[slot], local_index[slot])) {
                place = slot;
                break;
            }
        }
        if (place == K) { return; }
#pragma unroll
        for (int slot = K - 1; slot > place; --slot) {
            local_value[slot] = local_value[slot - 1];
            local_index[slot] = local_index[slot - 1];
        }
        local_value[place] = value;
        local_index[place] = index;
    };

    for (std::int32_t row = tid; row < valid_rows; row += kTopkThreads) {
        consider(row, __bfloat162float(logits[base + row]));
    }
    // argmax initializes on row 0, so a NaN there wins and nothing replaces it. A finite
    // row 0 is already in this thread's list when it belongs in the top-k.
    if (tid == 0) {
        const float first = __bfloat162float(logits[base]);
        if (isnan(first)) {
#pragma unroll
            for (int slot = K - 1; slot > 0; --slot) {
                local_value[slot] = local_value[slot - 1];
                local_index[slot] = local_index[slot - 1];
            }
            local_value[0] = first;
            local_index[0] = 0;
        }
    }

    __shared__ TopkCand pooled[kTopkThreads * K];
#pragma unroll
    for (int slot = 0; slot < K; ++slot) {
        pooled[tid * K + slot] = TopkCand{local_value[slot], local_index[slot]};
    }
    __syncthreads();

    if (tid == 0) {
        bool used[kTopkThreads * K] = {};
        for (int rank = 0; rank < K; ++rank) {
            int chosen     = -1;
            float best     = -CUDART_INF_F;
            std::int32_t at = INT32_MAX;
            bool seeded    = false;
            if (rank == 0) {
                for (int cand = 0; cand < kTopkThreads * K; ++cand) {
                    if (pooled[cand].index == 0 && isnan(pooled[cand].value)) {
                        chosen  = cand;
                        best    = pooled[cand].value;
                        at      = 0;
                        seeded  = true;
                        break;
                    }
                }
            }
            for (int cand = 0; cand < kTopkThreads * K; ++cand) {
                if (used[cand] || pooled[cand].index == INT32_MAX) { continue; }
                if (!seeded || topk_better(pooled[cand].value, pooled[cand].index, best, at)) {
                    seeded = true;
                    chosen = cand;
                    best   = pooled[cand].value;
                    at     = pooled[cand].index;
                }
            }
            ids[static_cast<std::int64_t>(token) * K + rank] = at;
            if (chosen >= 0) { used[chosen] = true; }
        }
    }
}

} // namespace

void topk_columns_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                         Tensor& ids, cudaStream_t stream) {
    const auto* values = static_cast<const __nv_bfloat16*>(logits.data);
    auto* out          = static_cast<std::int32_t*>(ids.data);
    const auto tokens  = static_cast<unsigned>(logits.ne[1]);
    const auto launch  = [&](auto kernel) {
        kernel<<<tokens, kTopkThreads, 0, stream>>>(values, out, valid_rows, logits.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    };
    switch (k) {
    case 1: launch(topk_columns_kernel<1>); break;
    case 2: launch(topk_columns_kernel<2>); break;
    case 3: launch(topk_columns_kernel<3>); break;
    case 4: launch(topk_columns_kernel<4>); break;
    case 5: launch(topk_columns_kernel<5>); break;
    case 6: launch(topk_columns_kernel<6>); break;
    case 7: launch(topk_columns_kernel<7>); break;
    case 8: launch(topk_columns_kernel<8>); break;
    default: break;
    }
}

} // namespace ninfer::ops::detail
