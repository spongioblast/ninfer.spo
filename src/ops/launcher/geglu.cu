#include "ops/launcher/geglu.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/gelu.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

__global__ void geglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float activated = gelu_one<true>(__bfloat162float(gate[i]));
        out[i] = __float2bfloat16_rn(activated * __bfloat162float(up[i]));
    }
}

} // namespace

void geglu_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = out.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    geglu_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                             static_cast<const __nv_bfloat16*>(up.data),
                                             static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
