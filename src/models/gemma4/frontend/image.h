#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {

struct Gemma4PreparedImage {
    // NInfer layouts: patches BF16 [patch_pixels, T], positions I32 [T] then [T] y, mask I32 [T].
    std::vector<std::uint16_t> patches;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> mask;
    int tokens = 0;
    int valid  = 0;
};

struct Gemma4ImageGeometry {
    int patch_size          = 16;
    int pooling_kernel_size = 3;
    int max_soft_tokens     = 280;
};

[[nodiscard]] Gemma4ImageGeometry gemma4_image_geometry();

// Aspect-ratio resize so height and width are multiples of patch*pool and the patch
// count is at most max_soft_tokens * pool^2.
[[nodiscard]] std::pair<int, int> gemma4_image_target_size(int height, int width,
                                                           const Gemma4ImageGeometry& geometry);

// Decode RGB bytes, resize, rescale to [0, 1], patchify, and pad to the soft-token budget.
[[nodiscard]] Gemma4PreparedImage gemma4_prepare_image(std::span<const std::uint8_t> bytes,
                                                       const OwnedMedia& media);

} // namespace ninfer::models::gemma4
