#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

[[nodiscard]] Fp8WeightGeometry fp8_row_geometry(std::int32_t n, std::int32_t k,
                                                 const char* operation);
Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
