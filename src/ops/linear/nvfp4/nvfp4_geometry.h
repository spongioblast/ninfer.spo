#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

using Nvfp4N14336K5120 = Nvfp4Geometry<14336, 5120>;
using Nvfp4N16384K5120 = Nvfp4Geometry<16384, 5120>;
using Nvfp4N34816K5120 = Nvfp4Geometry<34816, 5120>;
using Nvfp4N5120K6144  = Nvfp4Geometry<5120, 6144>;
using Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>;
// Gemma 4 31B: sliding qkv, global qk, gate+up, sliding and global o_proj, down.
using Nvfp4N16384K5376 = Nvfp4Geometry<16384, 5376>;
using Nvfp4N18432K5376 = Nvfp4Geometry<18432, 5376>;
using Nvfp4N43008K5376 = Nvfp4Geometry<43008, 5376>;
using Nvfp4N5376K8192  = Nvfp4Geometry<5376, 8192>;
using Nvfp4N5376K16384 = Nvfp4Geometry<5376, 16384>;
using Nvfp4N5376K21504 = Nvfp4Geometry<5376, 21504>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;
using Nvfp4Activation5376Geometry  = Nvfp4ActivationGeometry<5376>;
using Nvfp4Activation8192Geometry  = Nvfp4ActivationGeometry<8192>;
using Nvfp4Activation16384Geometry = Nvfp4ActivationGeometry<16384>;
using Nvfp4Activation21504Geometry = Nvfp4ActivationGeometry<21504>;

enum class Nvfp4GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N5120K6144,
    N5120K17408,
    N16384K5376,
    N18432K5376,
    N43008K5376,
    N5376K8192,
    N5376K16384,
    N5376K21504,
};

template <class Geometry>
constexpr bool nvfp4_geometry_is(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows == Geometry::kOutputRows && input_rows == Geometry::kInputRows;
}

inline Nvfp4GeometryId resolve_nvfp4_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4N14336K5120::kOutputRows &&
        input_rows == Nvfp4N14336K5120::kInputRows) {
        return Nvfp4GeometryId::N14336K5120;
    }
    if (output_rows == Nvfp4N16384K5120::kOutputRows &&
        input_rows == Nvfp4N16384K5120::kInputRows) {
        return Nvfp4GeometryId::N16384K5120;
    }
    if (output_rows == Nvfp4N34816K5120::kOutputRows &&
        input_rows == Nvfp4N34816K5120::kInputRows) {
        return Nvfp4GeometryId::N34816K5120;
    }
    if (output_rows == Nvfp4N5120K6144::kOutputRows && input_rows == Nvfp4N5120K6144::kInputRows) {
        return Nvfp4GeometryId::N5120K6144;
    }
    if (output_rows == Nvfp4N5120K17408::kOutputRows &&
        input_rows == Nvfp4N5120K17408::kInputRows) {
        return Nvfp4GeometryId::N5120K17408;
    }
    if (nvfp4_geometry_is<Nvfp4N16384K5376>(output_rows, input_rows))
        return Nvfp4GeometryId::N16384K5376;
    if (nvfp4_geometry_is<Nvfp4N18432K5376>(output_rows, input_rows))
        return Nvfp4GeometryId::N18432K5376;
    if (nvfp4_geometry_is<Nvfp4N43008K5376>(output_rows, input_rows))
        return Nvfp4GeometryId::N43008K5376;
    if (nvfp4_geometry_is<Nvfp4N5376K8192>(output_rows, input_rows))
        return Nvfp4GeometryId::N5376K8192;
    if (nvfp4_geometry_is<Nvfp4N5376K16384>(output_rows, input_rows))
        return Nvfp4GeometryId::N5376K16384;
    if (nvfp4_geometry_is<Nvfp4N5376K21504>(output_rows, input_rows))
        return Nvfp4GeometryId::N5376K21504;
    throw std::invalid_argument("unsupported NVFP4 problem");
}

} // namespace ninfer::ops::detail
