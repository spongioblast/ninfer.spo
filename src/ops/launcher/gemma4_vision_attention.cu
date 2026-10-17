#include "ops/launcher/gemma4_vision_attention.h"

#include "core/device.h"
#include "ops/kernel/gemma4_vision_attention.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// score[query, key] = fp32 dot(q, k). One thread per query head. The softmax that follows is the
// same kernel the cuBLAS path used, so only the dot differs from that path.
__global__ void gemma4_vision_qk_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                        const std::int32_t* mask, float* scores, int tokens) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int query = index / kGemma4VisionHeads;
    const int head  = index - query * kGemma4VisionHeads;
    if (query >= tokens) { return; }
    const __nv_bfloat16* qq =
        q + (static_cast<std::int64_t>(query) * kGemma4VisionHeads + head) * kGemma4VisionHeadDim;
    float* plane = scores + static_cast<std::int64_t>(head) * tokens * tokens + query;
    for (int key = 0; key < tokens; ++key) {
        float dot = 0.0f;
        if (mask == nullptr || mask[key] != 0) {
            const __nv_bfloat16* kk =
                k + (static_cast<std::int64_t>(key) * kGemma4VisionHeads + head) *
                        kGemma4VisionHeadDim;
            for (int dim = 0; dim < kGemma4VisionHeadDim; ++dim) {
                dot = fmaf(__bfloat162float(qq[dim]), __bfloat162float(kk[dim]), dot);
            }
        }
        plane[static_cast<std::int64_t>(tokens) * key] = dot;
    }
}

// out = P * V with the bf16 probabilities the softmax stored. One thread per query head.
__global__ void gemma4_vision_pv_kernel(const __nv_bfloat16* v, const __nv_bfloat16* weights,
                                        __nv_bfloat16* out, int tokens) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int query = index / kGemma4VisionHeads;
    const int head  = index - query * kGemma4VisionHeads;
    if (query >= tokens) { return; }
    const __nv_bfloat16* probability =
        weights + static_cast<std::int64_t>(head) * tokens * tokens + query;
    float acc[kGemma4VisionHeadDim] = {};
    for (int key = 0; key < tokens; ++key) {
        const float weight = __bfloat162float(probability[static_cast<std::int64_t>(tokens) * key]);
        if (weight == 0.0f) { continue; }
        const __nv_bfloat16* vv =
            v + (static_cast<std::int64_t>(key) * kGemma4VisionHeads + head) * kGemma4VisionHeadDim;
        for (int dim = 0; dim < kGemma4VisionHeadDim; ++dim) {
            acc[dim] = fmaf(weight, __bfloat162float(vv[dim]), acc[dim]);
        }
    }
    __nv_bfloat16* destination =
        out + (static_cast<std::int64_t>(query) * kGemma4VisionHeads + head) * kGemma4VisionHeadDim;
    for (int dim = 0; dim < kGemma4VisionHeadDim; ++dim) {
        destination[dim] = __float2bfloat16(acc[dim]);
    }
}

} // namespace

void gemma4_vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor* mask, Tensor& scores, Tensor& weights, Tensor& out,
                                    cudaStream_t stream) {
    const int tokens = q.ne[2];
    const int count  = tokens * kGemma4VisionHeads;
    const int block  = 128;
    const int grid   = (count + block - 1) / block;
    const auto* mask_ptr = mask == nullptr ? nullptr : static_cast<const std::int32_t*>(mask->data);
    gemma4_vision_qk_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        mask_ptr, static_cast<float*>(scores.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    gemma4_vision_softmax_kernel<<<dim3(tokens, kGemma4VisionHeads), 32, 0, stream>>>(
        static_cast<const float*>(scores.data), static_cast<__nv_bfloat16*>(weights.data), mask_ptr,
        tokens);
    CUDA_CHECK(cudaGetLastError());
    gemma4_vision_pv_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const __nv_bfloat16*>(weights.data),
        static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
