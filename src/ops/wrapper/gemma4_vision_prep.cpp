#include "ninfer/ops/gemma4_vision_prep.h"

#include "ops/launcher/gemma4_vision_prep.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool ok, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string("gemma4_vision_prep: ") + what); }
}

bool usable(const Tensor& tensor, DType dtype) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr;
}

} // namespace

void gemma4_vision_scale_patches(const Tensor& patches, Tensor& out, cudaStream_t stream) {
    require(usable(patches, DType::BF16) && usable(out, DType::BF16),
            "patches and out must be contiguous BF16");
    for (int axis = 0; axis < 4; ++axis) {
        require(patches.ne[axis] == out.ne[axis], "patches and out must have the same shape");
    }
    if (patches.numel() == 0) { return; }
    detail::gemma4_vision_scale_patches_launch(patches, out, stream);
}

void gemma4_vision_axial_rope(const Tensor& table, Tensor& q, Tensor& k, cudaStream_t stream) {
    require(usable(q, DType::BF16) && usable(k, DType::BF16), "q and k must be contiguous BF16");
    require(usable(table, DType::FP32), "table must be contiguous FP32");
    const int tokens = q.ne[2];
    require(q.ne[0] == 72 && q.ne[1] == 16 && tokens > 0 && q.ne[3] == 1,
            "q must be [72,16,T]");
    require(k.ne[0] == 72 && k.ne[1] == 16 && k.ne[2] == tokens && k.ne[3] == 1,
            "k must match q");
    require(table.ne[0] == 72 && table.ne[1] == tokens && table.ne[2] == 1 && table.ne[3] == 1,
            "table must be [72,T]");
    require(q.data != k.data, "q and k must be different tensors");
    detail::gemma4_vision_axial_rope_launch(table, q, k, stream);
}

void gemma4_vision_pool(const Tensor& hidden, const Tensor& offsets, const Tensor& members,
                        const Tensor& bias, const Tensor& scale, double gain, Tensor& out,
                        cudaStream_t stream) {
    require(usable(hidden, DType::BF16) && usable(out, DType::BF16) &&
                usable(bias, DType::BF16) && usable(scale, DType::BF16),
            "hidden, out, bias, and scale must be contiguous BF16");
    require(usable(offsets, DType::I32) && usable(members, DType::I32),
            "offsets and members must be contiguous I32");
    const int width  = hidden.ne[0];
    const int pooled = out.ne[1];
    require(width > 0 && hidden.ne[1] > 0 && hidden.ne[2] == 1 && out.ne[0] == width &&
                pooled > 0 && out.ne[2] == 1,
            "hidden must be [W,T] and out [W,P]");
    require(bias.numel() == width && scale.numel() == width, "bias and scale must be [W]");
    require(offsets.numel() == static_cast<std::int64_t>(pooled) + 1, "offsets must be [P+1]");
    require(members.numel() >= 1, "members must not be empty");
    detail::gemma4_vision_pool_launch(hidden, offsets, members, bias, scale, gain, out, stream);
}

} // namespace ninfer::ops
