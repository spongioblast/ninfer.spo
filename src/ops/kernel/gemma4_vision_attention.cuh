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

// grid (T, 16), block 32. Lane i holds dims i, i+32, and i+64 when i+64 < 72.
__global__ void gemma4_vision_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                               const __nv_bfloat16* __restrict__ k,
                                               const __nv_bfloat16* __restrict__ v,
                                               const std::int32_t* __restrict__ mask,
                                               __nv_bfloat16* __restrict__ out, int tokens) {
    constexpr int D     = kGemma4VisionHeadDim;
    constexpr int H     = kGemma4VisionHeads;
    const int token     = static_cast<int>(blockIdx.x);
    const int head      = static_cast<int>(blockIdx.y);
    const int lane      = static_cast<int>(threadIdx.x);
    const int parts     = lane + 64 < D ? 3 : 2;
    const int head_base = (token * H + head) * D;
    auto at             = [&](const __nv_bfloat16* src, int t, int part) {
        return src[(t * H + head) * D + lane + part * 32];
    };
    if (mask != nullptr && mask[token] == 0) {
        for (int part = 0; part < parts; ++part) { out[head_base + lane + part * 32] = __nv_bfloat16{}; }
        return;
    }
    float qv[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        if (part < parts) { qv[part] = __bfloat162float(at(q, token, part)); }
    }
    float maximum = -1.0e30f;
    float denom   = 0.0f;
    float acc[3]  = {0.0f, 0.0f, 0.0f};
    int valid     = 0;
    for (int key = 0; key < tokens; ++key) {
        if (mask != nullptr && mask[key] == 0) { continue; }
        float dot = 0.0f;
#pragma unroll
        for (int part = 0; part < 3; ++part) {
            if (part < parts) { dot += qv[part] * __bfloat162float(at(k, key, part)); }
        }
        dot             = gemma4_vision_warp_sum(dot);
        const float next = fmaxf(maximum, dot);
        const float grow = expf(maximum - next);
        const float term = expf(dot - next);
        denom            = denom * grow + term;
#pragma unroll
        for (int part = 0; part < 3; ++part) {
            if (part < parts) {
                acc[part] = acc[part] * grow + term * __bfloat162float(at(v, key, part));
            }
        }
        maximum = next;
        ++valid;
    }
    if (valid == 0) {
        for (int part = 0; part < parts; ++part) { out[head_base + lane + part * 32] = __nv_bfloat16{}; }
        return;
    }
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        if (part < parts) {
            out[head_base + lane + part * 32] = __float2bfloat16(acc[part] / denom);
        }
    }
}

} // namespace ninfer::ops::detail
