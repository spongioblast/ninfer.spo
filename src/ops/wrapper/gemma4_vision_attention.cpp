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
                             Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) {
        throw std::invalid_argument("gemma4_vision_attention: T must be positive");
    }
    require_heads(q, tokens, "q");
    require_heads(k, tokens, "k");
    require_heads(v, tokens, "v");
    require_heads(out, tokens, "out");
    if (mask != nullptr) {
        if (mask->dtype != DType::I32 || !mask->is_contiguous() || mask->data == nullptr ||
            mask->ne[0] != tokens || mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
            throw std::invalid_argument(
                "gemma4_vision_attention: mask must be contiguous I32 [T]");
        }
    }
    const auto* q_bytes   = static_cast<const char*>(q.data);
    const auto* k_bytes   = static_cast<const char*>(k.data);
    const auto* v_bytes   = static_cast<const char*>(v.data);
    const auto* out_bytes = static_cast<const char*>(out.data);
    const auto bytes      = static_cast<std::int64_t>(tokens) * kHeads * kHeadDim * 2;
    auto overlaps         = [&](const char* other) {
        return (out_bytes >= other && out_bytes < other + bytes) ||
               (other >= out_bytes && other < out_bytes + bytes);
    };
    if (overlaps(q_bytes) || overlaps(k_bytes) || overlaps(v_bytes)) {
        throw std::invalid_argument("gemma4_vision_attention: out must not overlap q, k, or v");
    }
    detail::gemma4_vision_attention_launch(q, k, v, mask, out, stream);
}

} // namespace ninfer::ops
