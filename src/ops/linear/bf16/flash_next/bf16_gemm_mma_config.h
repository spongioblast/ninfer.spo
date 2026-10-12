#pragma once
#include "core/weight.h"

#include "ops/linear/bf16/flash_next/bf16_gemm_mma.cuh"

#include <array>
#include <cstdlib>

namespace ninfer::ops::detail::flash_next {

// Measured large-T production schedule for the BF16 computation core. Geometry remains a template
// argument so an exact problem can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
template <class Geometry>
using Bf16MmaProductionSchedule =
    Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

template <class Geometry>
struct Bf16MmaScheduleSelector {
    using Type = Bf16MmaProductionSchedule<Geometry>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<10240, 320>> {
    using Type = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<48, 2560>> {
    using Type = Bf16MmaSchedule<16, 128, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<320, 10240>> {
    using Type = Bf16MmaSchedule<32, 64, 64, 16, 32, 4, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaScheduleSelector<Bf16GemvGeometry<2560, 6144>> {
    using Type = Bf16MmaSchedule<128, 128, 64, 32, 32, 2, 1, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong,
                                 Bf16MmaRaster::TokenFast>;
};

template <class Geometry>
using SelectedBf16MmaSchedule = typename Bf16MmaScheduleSelector<Geometry>::Type;

// Cluster K split per token count: the first step whose max_tokens covers T applies, and T beyond
// the last step runs unsplit. Measured on the linear bench (cold weights, RTX PRO 6000, 300 W) at
// T=17..8192 with every split forced. Splits of 8 only pay off while the tile holds few tokens or
// K is long; small grids keep a split up to a few thousand tokens.
struct Bf16MmaSplitStep {
    int max_tokens;
    int split;
};

template <class Geometry>
struct Bf16MmaSplitSelector {
    static constexpr std::array<Bf16MmaSplitStep, 0> kSteps{};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<48, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{2048, 4}, Bf16MmaSplitStep{4096, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<128, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{48, 8}, Bf16MmaSplitStep{4096, 4},
                                       Bf16MmaSplitStep{8192, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<512, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{24, 8}, Bf16MmaSplitStep{512, 4},
                                       Bf16MmaSplitStep{2048, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<640, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{24, 8}, Bf16MmaSplitStep{512, 4},
                                       Bf16MmaSplitStep{2048, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<2560, 640>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{128, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<320, 10240>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{128, 8}, Bf16MmaSplitStep{512, 4},
                                       Bf16MmaSplitStep{1024, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<2560, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{256, 4}, Bf16MmaSplitStep{512, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<2560, 6144>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{128, 8}, Bf16MmaSplitStep{256, 4},
                                       Bf16MmaSplitStep{512, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<6144, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{128, 4}, Bf16MmaSplitStep{1024, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<12288, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{64, 4}, Bf16MmaSplitStep{512, 2}};
};

template <>
struct Bf16MmaSplitSelector<Bf16GemvGeometry<10240, 2560>> {
    static constexpr std::array kSteps{Bf16MmaSplitStep{128, 2}};
};

template <class Geometry>
constexpr bool bf16_mma_split_steps_fit() {
    using Schedule = SelectedBf16MmaSchedule<Geometry>;
    for (const Bf16MmaSplitStep& step : Bf16MmaSplitSelector<Geometry>::kSteps) {
        const bool fits = step.split == 8   ? bf16_mma_k_split_fits<Geometry, Schedule, 8>()
                          : step.split == 4 ? bf16_mma_k_split_fits<Geometry, Schedule, 4>()
                          : step.split == 2 ? bf16_mma_k_split_fits<Geometry, Schedule, 2>()
                                            : false;
        if (!fits) return false;
    }
    return true;
}

template <class Geometry>
int bf16_mma_k_split(int tokens) {
    static_assert(bf16_mma_split_steps_fit<Geometry>());
    for (const Bf16MmaSplitStep& step : Bf16MmaSplitSelector<Geometry>::kSteps) {
        if (tokens <= step.max_tokens) return step.split;
    }
    return 1;
}

} // namespace ninfer::ops::detail::flash_next
