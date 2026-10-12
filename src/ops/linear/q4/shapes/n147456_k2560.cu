#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q4Launch select_q4_n147456_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<147456, 2560, 4>;
    if (tokens <= 8) return launch_q4_ksplit<147456, 2560, 8>;
    if (tokens <= 16) return launch_q4_ksplit<147456, 2560, 16>;
    if (tokens <= 20) return launch_q4_ksplit<147456, 2560, 20>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 48) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 72) return launch_q4_a16_mma_r64_t72;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
