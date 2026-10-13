#include "ninfer/ops/gemma4_attention.h"

#include "ops/launcher/gemma4_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kPage = 64;

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_attention: ") + name +
                                    " must be contiguous non-null BF16");
    }
}

constexpr std::size_t kAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace

std::size_t gemma4_attention_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                      std::int32_t kv_heads, std::int32_t tokens) {
    std::size_t widest = 0;
    if (head_dim <= 0 || q_heads <= 0 || kv_heads <= 0) { return widest; }
    for (std::int32_t count = 1; count <= tokens; ++count) {
        const int chunks = detail::gemma4_attention_chunks(count, kv_heads);
        if (chunks == 1) { break; }
        const auto rows = static_cast<std::size_t>(chunks) * q_heads * count;
        widest          = std::max(widest, aligned(rows * head_dim * sizeof(float)) +
                                               aligned(rows * 2 * sizeof(float)));
    }
    return widest;
}

void gemma4_attention(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                      const Tensor& block_table, const Tensor& positions, int window, float scale,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    require_bf16(q, "q");
    require_bf16(k_pages, "k_pages");
    require_bf16(v_pages, "v_pages");
    require_bf16(out, "out");
    if (positions.dtype != DType::I32 || block_table.dtype != DType::I32 ||
        !positions.is_contiguous() || !block_table.is_contiguous() || positions.data == nullptr ||
        block_table.data == nullptr) {
        throw std::invalid_argument("gemma4_attention: positions and block_table must be I32");
    }
    if (window < 0) { throw std::invalid_argument("gemma4_attention: window must be nonnegative"); }
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        throw std::invalid_argument("gemma4_attention: scale must be positive and finite");
    }
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = k_pages.ne[2];
    if ((head_dim != 256 && head_dim != 512) || q_heads <= 0 || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || tokens <= 0 || q.ne[3] != 1) {
        throw std::invalid_argument("gemma4_attention: unsupported q shape");
    }
    if (out.ne[0] != head_dim || out.ne[1] != q_heads || out.ne[2] != tokens || out.ne[3] != 1) {
        throw std::invalid_argument("gemma4_attention: out shape must match q");
    }
    if (positions.ne[0] != tokens || positions.ne[1] != 1) {
        throw std::invalid_argument("gemma4_attention: positions must be [T]");
    }
    if (k_pages.ne[0] != head_dim || k_pages.ne[1] != kPage || k_pages.ne[3] <= 0 ||
        v_pages.ne[0] != k_pages.ne[0] || v_pages.ne[1] != k_pages.ne[1] ||
        v_pages.ne[2] != k_pages.ne[2] || v_pages.ne[3] != k_pages.ne[3]) {
        throw std::invalid_argument("gemma4_attention: k/v pages must be [D,64,Hkv,pages]");
    }
    if (block_table.ne[0] <= 0) {
        throw std::invalid_argument("gemma4_attention: block_table must be nonempty");
    }
    const auto q_bytes   = static_cast<const char*>(q.data);
    const auto out_bytes = static_cast<const char*>(out.data);
    const auto bytes     = static_cast<std::int64_t>(head_dim) * q_heads * tokens * 2;
    if ((out_bytes >= q_bytes && out_bytes < q_bytes + bytes) ||
        (q_bytes >= out_bytes && q_bytes < out_bytes + bytes)) {
        throw std::invalid_argument("gemma4_attention: q and out must not overlap");
    }
    auto scope       = workspace.scope();
    float* partial   = nullptr;
    float* stats     = nullptr;
    const int chunks = detail::gemma4_attention_chunks(tokens, kv_heads);
    if (chunks > 1) {
        partial = static_cast<float*>(
            workspace.alloc(DType::FP32, {head_dim, chunks, q_heads, tokens}).data);
        stats =
            static_cast<float*>(workspace.alloc(DType::FP32, {2, chunks, q_heads, tokens}).data);
    }
    detail::gemma4_attention_launch(q, k_pages, v_pages, block_table, positions, window, scale, out,
                                    partial, stats, stream);
}

} // namespace ninfer::ops
