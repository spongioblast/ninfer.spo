// Gemma 4 31B projections with the 5376-wide hidden input: sliding qkv [16384], global qk
// [18432] and gate+up [43008]. Routes follow n16384_k5120 and n34816_k5120. 5376 is 21 x 256,
// not a multiple of 512, so the GEMV and SIMT lanes take 8 values and sliced K uses four warps.
// Not yet tuned.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 8, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 8, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using T32R64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <int N>
Nvfp4Launch select_a16(int tokens) {
    using Geometry = Nvfp4Geometry<N, 5376>;
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 4, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <int N>
void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16<N>(x.ne[1])(x, w, y, stream);
}

template <int N, Nvfp4GeometryId Id>
Nvfp4A4Route select_a4(std::int32_t tokens) {
    using Geometry = Nvfp4Geometry<N, 5376>;
    if (tokens >= 512) return nvfp4_a4_tma_route<Id>();
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t) { return true; }

template <int N, Nvfp4GeometryId Id>
constexpr Nvfp4LinearShape shape() {
    return {N, 5376, launch_a16<N>, launch_nvfp4_a4<select_a4<N, Id>>, uses_a4};
}
} // namespace

const Nvfp4LinearShape kNvfp4N16384K5376 = shape<16384, Nvfp4GeometryId::N16384K5376>();
const Nvfp4LinearShape kNvfp4N18432K5376 = shape<18432, Nvfp4GeometryId::N18432K5376>();
const Nvfp4LinearShape kNvfp4N43008K5376 = shape<43008, Nvfp4GeometryId::N43008K5376>();
} // namespace ninfer::ops::detail
