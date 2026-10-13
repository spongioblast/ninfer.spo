#include "ninfer/ops/fp8_row_cublas.h"

#include "ops/launcher/fp8_row_cublas.h"
#include "ops/linear/fp8/fp8_format.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

void require_bf16_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                         const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("fp8_row_cublas: ") + name +
                                    " must be contiguous BF16 [" + std::to_string(rows) + "," +
                                    std::to_string(columns) + "]");
    }
}

} // namespace

std::size_t fp8_row_cublas_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    if (n <= 0 || k <= 0 || tokens <= 0) {
        throw std::invalid_argument("fp8_row_cublas: extents must be positive");
    }
    const auto rows = static_cast<std::size_t>(n);
    return aligned(rows * static_cast<std::size_t>(k) * 2) +
           aligned(rows * static_cast<std::size_t>(tokens) * 4) + kAlignment;
}

void fp8_row_cublas(const Tensor& x, const Weight& w, Tensor& out, Bf16GemmContext& gemm,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (w.qtype != QType::FP8_E4M3FN_ROW_BF16) {
        throw std::invalid_argument("fp8_row_cublas: weight must be FP8_E4M3FN_ROW_BF16");
    }
    (void)detail::validate_fp8_weight(w, "fp8_row_cublas");
    const std::int32_t tokens = x.ne[1];
    require_bf16_matrix(x, w.k, tokens, "x");
    require_bf16_matrix(out, w.n, tokens, "out");
    if (tokens <= 0) { throw std::invalid_argument("fp8_row_cublas: T must be positive"); }

    const std::int64_t codes = static_cast<std::int64_t>(w.n) * w.k;
    const auto scope         = workspace.scope();
    Tensor widened           = workspace.alloc(DType::BF16, {w.k, w.n});
    Tensor partials          = workspace.alloc(DType::FP32, {w.n, tokens});
    detail::fp8_widen_codes_launch(static_cast<const std::uint8_t*>(w.qdata),
                                   static_cast<__nv_bfloat16*>(widened.data), codes, stream);
    Weight dense;
    dense.qtype    = QType::BF16;
    dense.layout   = QuantLayout::Contiguous;
    dense.qdata    = widened.data;
    dense.n        = w.n;
    dense.k        = w.k;
    dense.ndim     = 2;
    dense.shape[0] = w.n;
    dense.shape[1] = w.k;
    gemm.launch(x, dense, partials, stream);
    detail::fp8_scale_rows_launch(static_cast<const float*>(partials.data),
                                  static_cast<const __nv_bfloat16*>(w.scales),
                                  static_cast<__nv_bfloat16*>(out.data), w.n,
                                  static_cast<std::int64_t>(w.n) * tokens, stream);
}

} // namespace ninfer::ops
