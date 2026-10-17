#include "ninfer/ops/gemma4_vision_attention.h"

#include "ops/launcher/gemma4_vision_attention.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 72;
constexpr std::int32_t kHeads   = 16;

void require_heads(const Tensor& tensor, std::int32_t tokens, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        tensor.ne[0] != kHeadDim || tensor.ne[1] != kHeads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("gemma4_vision_attention: ") + name +
                                    " must be contiguous BF16 [72,16,T]");
    }
}

} // namespace

void gemma4_vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask,
                             Tensor& scores, Tensor& weights, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) {
        throw std::invalid_argument("gemma4_vision_attention: T must be positive");
    }
    require_heads(q, tokens, "q");
    require_heads(k, tokens, "k");
    require_heads(v, tokens, "v");
    require_heads(out, tokens, "out");
    const auto score_count = static_cast<std::int64_t>(kHeads) * tokens * tokens;
    if (scores.dtype != DType::FP32 || !scores.is_contiguous() || scores.data == nullptr ||
        scores.numel() < score_count) {
        throw std::invalid_argument(
            "gemma4_vision_attention: scores must be contiguous FP32 with 16*T*T elements");
    }
    if (weights.dtype != DType::BF16 || !weights.is_contiguous() || weights.data == nullptr ||
        weights.numel() < score_count) {
        throw std::invalid_argument(
            "gemma4_vision_attention: weights must be contiguous BF16 with 16*T*T elements");
    }
    if (mask != nullptr) {
        if (mask->dtype != DType::I32 || !mask->is_contiguous() || mask->data == nullptr ||
            mask->ne[0] != tokens || mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
            throw std::invalid_argument(
                "gemma4_vision_attention: mask must be contiguous I32 [T]");
        }
    }
    const auto* q_bytes      = static_cast<const char*>(q.data);
    const auto* k_bytes      = static_cast<const char*>(k.data);
    const auto* v_bytes      = static_cast<const char*>(v.data);
    const auto* out_bytes    = static_cast<const char*>(out.data);
    const auto* score_bytes  = static_cast<const char*>(scores.data);
    const auto* weight_bytes = static_cast<const char*>(weights.data);
    const auto qkv_bytes     = static_cast<std::int64_t>(tokens) * kHeads * kHeadDim * 2;
    const auto score_nbytes  = score_count * 4;
    const auto weight_nbytes = score_count * 2;
    auto overlaps            = [&](const char* a, std::int64_t a_bytes, const char* b,
                        std::int64_t b_bytes) {
        return (a >= b && a < b + b_bytes) || (b >= a && b < a + a_bytes);
    };
    if (overlaps(out_bytes, qkv_bytes, q_bytes, qkv_bytes) ||
        overlaps(out_bytes, qkv_bytes, k_bytes, qkv_bytes) ||
        overlaps(out_bytes, qkv_bytes, v_bytes, qkv_bytes) ||
        overlaps(out_bytes, qkv_bytes, score_bytes, score_nbytes) ||
        overlaps(out_bytes, qkv_bytes, weight_bytes, weight_nbytes) ||
        overlaps(score_bytes, score_nbytes, q_bytes, qkv_bytes) ||
        overlaps(score_bytes, score_nbytes, k_bytes, qkv_bytes) ||
        overlaps(score_bytes, score_nbytes, v_bytes, qkv_bytes) ||
        overlaps(score_bytes, score_nbytes, weight_bytes, weight_nbytes) ||
        overlaps(weight_bytes, weight_nbytes, q_bytes, qkv_bytes) ||
        overlaps(weight_bytes, weight_nbytes, k_bytes, qkv_bytes) ||
        overlaps(weight_bytes, weight_nbytes, v_bytes, qkv_bytes)) {
        throw std::invalid_argument(
            "gemma4_vision_attention: out, scores, and weights must not overlap q, k, v, or each other");
    }
    detail::gemma4_vision_attention_launch(q, k, v, mask, scores, weights, out, stream);
}

} // namespace ninfer::ops
