// Gemma 4 31B projections back to the 5376-wide hidden state: sliding o_proj [K 8192], global
// o_proj [K 16384] and down [K 21504]. Routes follow n5120_k17408; every K is a multiple of 512.
// Not yet tuned.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using T32R64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <int K>
Nvfp4Launch select_a16(int tokens) {
    using Geometry = Nvfp4Geometry<5376, K>;
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 4, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <int K>
void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16<K>(x.ne[1])(x, w, y, stream);
}

template <int K, Nvfp4GeometryId Id>
Nvfp4A4Route select_a4(std::int32_t tokens) {
    using Geometry = Nvfp4Geometry<5376, K>;
    if (tokens >= 1024) return nvfp4_a4_tma_route<Id>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 8; }

template <int K, Nvfp4GeometryId Id>
constexpr Nvfp4LinearShape shape() {
    return {5376, K, launch_a16<K>, launch_nvfp4_a4<select_a4<K, Id>>, uses_a4};
}
} // namespace

const Nvfp4LinearShape kNvfp4N5376K8192  = shape<8192, Nvfp4GeometryId::N5376K8192>();
const Nvfp4LinearShape kNvfp4N5376K16384 = shape<16384, Nvfp4GeometryId::N5376K16384>();
const Nvfp4LinearShape kNvfp4N5376K21504 = shape<21504, Nvfp4GeometryId::N5376K21504>();
} // namespace ninfer::ops::detail
