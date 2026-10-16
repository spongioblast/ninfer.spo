#include "models/gemma4/frontend/image.h"

#include "media/decode/decode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

constexpr int kChannels = 3;

int round_even(double value) {
    return static_cast<int>(std::nearbyint(value));
}

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

double cubic(double x) {
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            out.weights.push_back(static_cast<float>(cubic((j + begin - center + 0.5) * invscale)));
            sum += out.weights.back();
        }
        if (sum == 0.0) { throw std::runtime_error("gemma4 image resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

void resize_bicubic(media::decode::Image& input, int height, int width) {
    if (input.width == width && input.height == height) { return; }
    const Coefficients horizontal = coefficients(input.width, width);
    const Coefficients vertical   = coefficients(input.height, height);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * width * kChannels);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<float, 3> value{};
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const float weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination = temp.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < kChannels; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    media::decode::Image out;
    out.width  = width;
    out.height = height;
    out.rgb.resize(static_cast<std::size_t>(height) * width * kChannels);
    for (int y = 0; y < height; ++y) {
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            std::array<float, 3> value{};
            const std::uint8_t* source =
                temp.data() + (static_cast<std::size_t>(source_begin) * width + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(width) * 3) {
                const float weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination =
                out.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < kChannels; ++c) {
                destination[c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    input = std::move(out);
}

} // namespace

Gemma4ImageGeometry gemma4_image_geometry() { return {}; }

std::pair<int, int> gemma4_image_target_size(int height, int width,
                                             const Gemma4ImageGeometry& geometry) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("gemma4 image has no pixels");
    }
    const int patch     = geometry.patch_size;
    const int pool      = geometry.pooling_kernel_size;
    const int max_soft  = geometry.max_soft_tokens;
    const int max_patch = max_soft * pool * pool;
    const int side      = pool * patch;
    const double factor =
        std::sqrt(static_cast<double>(max_patch) * patch * patch / (static_cast<double>(height) * width));
    int target_h = static_cast<int>(std::floor(factor * height / side)) * side;
    int target_w = static_cast<int>(std::floor(factor * width / side)) * side;
    const int max_side = (max_soft)*side;
    if (target_h == 0 && target_w == 0) {
        throw std::invalid_argument("gemma4 image resize collapsed to 0x0");
    }
    if (target_h == 0) {
        target_h = side;
        target_w = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side,
                            max_side);
    } else if (target_w == 0) {
        target_w = side;
        target_h = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side,
                            max_side);
    }
    if (target_h / patch * (target_w / patch) > max_patch) {
        throw std::invalid_argument("gemma4 image resize exceeded the patch budget");
    }
    return {target_h, target_w};
}

Gemma4PreparedImage gemma4_prepare_image(std::span<const std::uint8_t> bytes,
                                         const OwnedMedia& media) {
    if (media.kind != MediaKind::Image) {
        throw std::invalid_argument("gemma4 video and audio input is not wired");
    }
    if (bytes.empty()) { throw std::invalid_argument("gemma4 image has no bytes"); }
    const Gemma4ImageGeometry geometry = gemma4_image_geometry();
    media::decode::Policy policy;
    policy.max_decoded_pixels = 64ULL * 1024ULL * 1024ULL;
    media::decode::Image image = media::decode::decode_image(bytes, policy);
    const auto [target_h, target_w] =
        gemma4_image_target_size(image.height, image.width, geometry);
    resize_bicubic(image, target_h, target_w);
    const int patch    = geometry.patch_size;
    const int pool     = geometry.pooling_kernel_size;
    const int max_soft = geometry.max_soft_tokens;
    const int gh       = target_h / patch;
    const int gw       = target_w / patch;
    const int real     = gh * gw;
    const int padded   = max_soft * pool * pool;
    const int pixels   = kChannels * patch * patch;
    if (real <= 0 || real > padded || real % (pool * pool) != 0) {
        throw std::invalid_argument("gemma4 image patch count is not a pool multiple");
    }
    Gemma4PreparedImage out;
    out.tokens = padded;
    out.valid  = real / (pool * pool);
    out.patches.assign(static_cast<std::size_t>(pixels) * padded, 0);
    out.positions.assign(static_cast<std::size_t>(padded) * 2, -1);
    out.mask.assign(static_cast<std::size_t>(padded), 0);
    for (int gy = 0; gy < gh; ++gy) {
        for (int gx = 0; gx < gw; ++gx) {
            const int t = gy * gw + gx;
            out.positions[static_cast<std::size_t>(t)]                            = gx;
            out.positions[static_cast<std::size_t>(t) + padded]                   = gy;
            out.mask[static_cast<std::size_t>(t)]                                 = 1;
            for (int py = 0; py < patch; ++py) {
                for (int px = 0; px < patch; ++px) {
                    const std::uint8_t* source =
                        image.rgb.data() +
                        (static_cast<std::size_t>(gy * patch + py) * image.width + gx * patch + px) *
                            kChannels;
                    for (int c = 0; c < kChannels; ++c) {
                        const int pixel = c + kChannels * (px + patch * py);
                        out.patches[static_cast<std::size_t>(pixel) +
                                    static_cast<std::size_t>(pixels) * t] =
                            to_bf16(static_cast<float>(source[c]) / 255.0f);
                    }
                }
            }
        }
    }
    return out;
}

} // namespace ninfer::models::gemma4
