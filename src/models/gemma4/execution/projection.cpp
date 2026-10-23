#include "models/gemma4/execution/projection.h"

#include "ninfer/ops/fp8_row_cublas.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <algorithm>

namespace ninfer::models::gemma4 {
namespace {

// From this width the per-call code widening is cheaper than the gap between cuBLAS and the FP8
// tensor-core kernels.
constexpr int kFp8CublasMinTokens = 512;

bool fp8_through_cublas(const Weight& weight, ops::LinearPolicy policy, int tokens) {
    // q4_g32 stays in ops::linear (GEMV, wide GEMV, or W4A8). It must not widen through cuBLAS.
    return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && policy == ops::LinearPolicy::A16Only &&
           tokens >= kFp8CublasMinTokens;
}

} // namespace

void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Bf16GemmContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (weight.qtype == QType::BF16) {
        gemm.launch(x, weight, out, stream);
    } else if (fp8_through_cublas(weight, policy, x.ne[1])) {
        ops::fp8_row_cublas(x, weight, out, gemm, workspace, stream);
    } else {
        ops::linear(x, weight, out, policy, workspace, stream);
    }
}

DownA8 prepare_down_a8(WorkspaceArena& workspace, int rows, int tokens) {
    namespace detail = ninfer::ops::detail;
    const auto codes  = workspace.alloc_bytes(detail::fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(rows)), 256);
    const auto scales = workspace.alloc_bytes(detail::fp8_a8_checked_bytes(tokens, sizeof(float)), 256);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data)};
}

void project_down_ready(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
                        DownA8 ready, ops::Bf16GemmContext&, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    ops::linear_a8_ready(x, weight, out, policy, ready.codes, ready.scales, workspace, stream);
}

std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy, int tokens) {
    if (weight.qtype == QType::BF16) { return 0; }
    const std::size_t linear =
        ops::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy, 1, tokens);
    if (!fp8_through_cublas(weight, policy, tokens)) { return linear; }
    return std::max(linear, ops::fp8_row_cublas_workspace_bytes(weight.n, weight.k, tokens));
}

} // namespace ninfer::models::gemma4
