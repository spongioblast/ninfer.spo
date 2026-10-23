#include "ops/launcher/gemma4_vision_prep.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ void gemma4_vision_scale_patches_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                   std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16_rn(__fmul_rn(2.0f, __fsub_rn(__bfloat162float(x[i]), 0.5f)));
}

// One thread per (token, head, rotated pair): 36 pairs per head, x then y.
__global__ void gemma4_vision_axial_rope_kernel(const float* table, __nv_bfloat16* q,
                                                __nv_bfloat16* k, int tokens) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= tokens * 16 * 36) { return; }
    const int pair  = index % 36;
    const int head  = (index / 36) % 16;
    const int token = index / (36 * 16);
    const int axis  = pair / 18;
    const int i     = pair % 18;
    const float* angles = table + static_cast<std::int64_t>(token) * 72 + axis * 36;
    const float c       = angles[i];
    const float s       = angles[18 + i];
    const std::int64_t base = (static_cast<std::int64_t>(token) * 16 + head) * 72 + axis * 36 + i;
    const auto rotate = [&](__nv_bfloat16* x) {
        const float a = __bfloat162float(x[base]);
        const float b = __bfloat162float(x[base + 18]);
        x[base]      = __float2bfloat16_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
        x[base + 18] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    };
    rotate(q);
    rotate(k);
}

// One thread per (dimension, bin).
__global__ void gemma4_vision_pool_kernel(const __nv_bfloat16* hidden, const std::int32_t* offsets,
                                          const std::int32_t* members, const __nv_bfloat16* bias,
                                          const __nv_bfloat16* scale, double gain,
                                          __nv_bfloat16* out, int width, int pooled) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= width * pooled) { return; }
    const int d   = index % width;
    const int bin = index / width;
    double acc    = 0.0;
    for (int at = offsets[bin]; at < offsets[bin + 1]; ++at) {
        acc = __dadd_rn(acc, static_cast<double>(__bfloat162float(
                                 hidden[static_cast<std::int64_t>(members[at]) * width + d])));
    }
    const double centered =
        __dsub_rn(__dmul_rn(acc, gain), static_cast<double>(__bfloat162float(bias[d])));
    out[index] = __float2bfloat16_rn(__double2float_rn(
        __dmul_rn(centered, static_cast<double>(__bfloat162float(scale[d])))));
}

unsigned blocks(std::int64_t count) {
    return static_cast<unsigned>((count + kBlock - 1) / kBlock);
}

} // namespace

void gemma4_vision_scale_patches_launch(const Tensor& patches, Tensor& out, cudaStream_t stream) {
    const std::int64_t count = patches.numel();
    gemma4_vision_scale_patches_kernel<<<blocks(count), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(patches.data), static_cast<__nv_bfloat16*>(out.data),
        count);
    CUDA_CHECK(cudaGetLastError());
}

void gemma4_vision_axial_rope_launch(const Tensor& table, Tensor& q, Tensor& k,
                                     cudaStream_t stream) {
    const int tokens = q.ne[2];
    gemma4_vision_axial_rope_kernel<<<blocks(static_cast<std::int64_t>(tokens) * 16 * 36), kBlock,
                                      0, stream>>>(
        static_cast<const float*>(table.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void gemma4_vision_pool_launch(const Tensor& hidden, const Tensor& offsets, const Tensor& members,
                               const Tensor& bias, const Tensor& scale, double gain, Tensor& out,
                               cudaStream_t stream) {
    const int width  = out.ne[0];
    const int pooled = out.ne[1];
    gemma4_vision_pool_kernel<<<blocks(static_cast<std::int64_t>(width) * pooled), kBlock, 0,
                                stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(offsets.data),
        static_cast<const std::int32_t*>(members.data),
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<const __nv_bfloat16*>(scale.data),
        gain, static_cast<__nv_bfloat16*>(out.data), width, pooled);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
