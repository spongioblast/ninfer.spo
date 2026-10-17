#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

constexpr int kGemma4VisionHeadDim = 72;
constexpr int kGemma4VisionHeads   = 16;

__device__ __forceinline__ float gemma4_vision_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float gemma4_vision_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
    }
    return value;
}

// scores and weights are col-major [query, key] per head: query + T*key + T*T*head.
// grid (T, 16), block 32. Pad keys get weight 0; pad queries still attend.
// Logits stay fp32. Softmax is fp32; the probabilities written for the value GEMM are bf16.
__global__ void gemma4_vision_softmax_kernel(const float* scores, __nv_bfloat16* weights,
                                             const std::int32_t* mask, int tokens) {
    const int query = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x);
    const float* plane = scores + static_cast<std::int64_t>(head) * tokens * tokens;
    __nv_bfloat16* out = weights + static_cast<std::int64_t>(head) * tokens * tokens;
    auto at = [&](int key) -> float {
        return plane[query + static_cast<std::int64_t>(tokens) * key];
    };
    auto store = [&](int key, __nv_bfloat16 value) {
        out[query + static_cast<std::int64_t>(tokens) * key] = value;
    };
    float maximum = -1.0e30f;
    int valid     = 0;
    for (int key = lane; key < tokens; key += 32) {
        if (mask != nullptr && mask[key] == 0) { continue; }
        maximum = fmaxf(maximum, at(key));
        ++valid;
    }
    valid   = static_cast<int>(gemma4_vision_warp_sum(static_cast<float>(valid)));
    maximum = gemma4_vision_warp_max(maximum);
    if (valid == 0) {
        for (int key = lane; key < tokens; key += 32) { store(key, __nv_bfloat16{}); }
        return;
    }
    float denom = 0.0f;
    for (int key = lane; key < tokens; key += 32) {
        if (mask != nullptr && mask[key] == 0) { continue; }
        denom += expf(at(key) - maximum);
    }
    denom = gemma4_vision_warp_sum(denom);
    for (int key = lane; key < tokens; key += 32) {
        if (mask != nullptr && mask[key] == 0) {
            store(key, __nv_bfloat16{});
            continue;
        }
        store(key, __float2bfloat16(expf(at(key) - maximum) / denom));
    }
}

} // namespace ninfer::ops::detail
