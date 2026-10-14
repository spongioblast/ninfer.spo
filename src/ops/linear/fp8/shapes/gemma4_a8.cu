#include "ops/linear/fp8/shapes/gemma4_a8.h"

#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail::gemma4_fp8 {
namespace {
using Bulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
constexpr std::int32_t kA8MinTokens = 128;
} // namespace

template <int N, int K>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    launch_fp8_a8_tma<Fp8Geometry<N, K>, Bulk>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= kA8MinTokens; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens >= kA8MinTokens ? Bulk::kPartialBytes : 0;
}

#define NINFER_GEMMA4_FP8_A8(N, K)                                                                 \
    template void launch_a8<N, K>(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace,           \
                                  cudaStream_t);
NINFER_GEMMA4_FP8_A8(16384, 5376)
NINFER_GEMMA4_FP8_A8(18432, 5376)
NINFER_GEMMA4_FP8_A8(43008, 5376)
NINFER_GEMMA4_FP8_A8(5376, 8192)
NINFER_GEMMA4_FP8_A8(5376, 16384)
NINFER_GEMMA4_FP8_A8(5376, 21504)
#undef NINFER_GEMMA4_FP8_A8

} // namespace ninfer::ops::detail::gemma4_fp8
