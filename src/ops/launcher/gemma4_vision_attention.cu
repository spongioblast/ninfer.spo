#include "ops/launcher/gemma4_vision_attention.h"

#include "core/device.h"
#include "ops/kernel/gemma4_vision_attention.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void gemma4_vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor* mask, Tensor& out, cudaStream_t stream) {
    const int tokens = q.ne[2];
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(kGemma4VisionHeads));
    gemma4_vision_attention_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        mask == nullptr ? nullptr : static_cast<const std::int32_t*>(mask->data),
        static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
