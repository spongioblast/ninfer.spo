#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/shapes/gemma4_a8.h"
#include "ops/linear/q4/q4_g32_gemv.cuh"
#include "ops/linear/q4/q4_g32_w4a8.cuh"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::Q4_G32_FP16:
        detail::q4_g32_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::Q4_G32_FP16:
        if (policy == LinearPolicy::A16Only) { return 0; }
        return detail::q4_g32_w4a8_workspace_bytes(output_rows, input_rows, max_tokens);
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
            Bf16GemmContext* bf16_gemm) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    if (bf16_gemm != nullptr && w.qtype == QType::BF16 && x.ne[1] > 16) {
        bf16_gemm->launch(x, w, out, stream);
        return;
    }
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

void linear_a8_ready(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     std::uint8_t* codes, float* scales, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    if (w.qtype == QType::Q4_G32_FP16) {
        if (!allows_a8(policy) || codes == nullptr || scales == nullptr) {
            throw std::invalid_argument("q4_g32 ready: needs the A8 policy, codes, and scales");
        }
        const detail::Q4LinearOperands operands = detail::q4_linear_operands(x, w);
        detail::q4_g32_w4a8_ready(operands, static_cast<__nv_bfloat16*>(out.data), codes, scales, workspace,
                                  stream);
        return;
    }
    detail::fp8_dispatch_ready(x, w, out, policy, codes, scales, &workspace, stream);
}

bool linear_takes_a8_ready(const Weight& w, std::int32_t tokens, LinearPolicy policy) {
    if (!allows_a8(policy)) { return false; }
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        return w.n == 5376 && w.k == 21504 && detail::gemma4_fp8::uses_a8(tokens, tokens);
    }
    if (w.qtype == QType::Q4_G32_FP16) { return detail::q4_g32_takes_ready_codes(w.n, w.k, tokens); }
    return false;
}

void linear_prepare_a8_rows(Weight& w, DeviceBuffer& storage, cudaStream_t stream) {
    if (w.qtype != QType::Q4_G32_FP16 || w.layout != QuantLayout::RowSplit || w.scales == nullptr ||
        w.padded_shape[1] != w.k || !detail::q4_trunk_shape(w.n, w.k)) {
        return;
    }
    DeviceBuffer rows(static_cast<std::size_t>(w.n) * 2);
    detail::q4_g32_row_multipliers(static_cast<const std::uint8_t*>(w.scales), rows.p, w.n, w.k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    storage      = std::move(rows);
    w.row_scales = storage.p;
}

} // namespace ninfer::ops
