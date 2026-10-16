#include "models/gemma4/frontend/image.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    int failures = 0;
    const auto geometry = ninfer::models::gemma4::gemma4_image_geometry();
    const auto size = ninfer::models::gemma4::gemma4_image_target_size(320, 480, geometry);
    const int patches = (size.first / geometry.patch_size) * (size.second / geometry.patch_size);
    const int valid   = patches / (geometry.pooling_kernel_size * geometry.pooling_kernel_size);
    if (size.first != 624 || size.second != 960 || patches != 2340 || valid != 260) {
        std::cerr << "FAIL 320x480 geometry " << size.first << "x" << size.second
                  << " patches=" << patches << " valid=" << valid << '\n';
        ++failures;
    }
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n480 320\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    ppm.insert(ppm.end(), static_cast<std::size_t>(480) * 320 * 3, static_cast<std::uint8_t>(40));
    ninfer::OwnedMedia media;
    media.kind        = ninfer::MediaKind::Image;
    media.bytes       = ppm;
    media.media_type  = "image/x-portable-pixmap";
    media.source_name = "solid.ppm";
    const auto image  = ninfer::models::gemma4::gemma4_prepare_image(media.bytes, media);
    if (image.tokens != 2520 || image.valid != 260 || image.mask[0] != 1 ||
        image.positions[0] != 0 || image.positions[2520] != 0 || image.mask[2340] != 0 ||
        image.positions[2340] != -1) {
        std::cerr << "FAIL prepared solid image tokens=" << image.tokens
                  << " valid=" << image.valid << '\n';
        ++failures;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 image\n";
    return failures ? 1 : 0;
}
