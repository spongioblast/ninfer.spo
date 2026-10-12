#include "ops/linear/bf16/flash_next/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/flash_next/bf16_config.h"
#include "ops/linear/bf16/flash_next/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail::flash_next {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Bf16LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16SmallTContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr int kFlashNextSmallTMax = 16;
template <class Geometry>
constexpr auto make_flash_launchers() {
    return make_launchers<Geometry>(
        std::make_index_sequence<kFlashNextSmallTMax - kBf16SmallTMinTokens + 1>{});
}

#define NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(NAME, N, K) \
    using NAME##Geometry = Bf16GemvGeometry<(N), (K)>;  \
    constexpr auto k##NAME##Launchers = make_flash_launchers<NAME##Geometry>();
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(HcDown, 320, 10240)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(HcUp, 10240, 320)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(QsaQueryGate, 12288, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(NarrowProjection, 512, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(IndexKey, 128, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(AttentionOutput, 2560, 6144)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(SharedGateUp, 640, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(GdnQkv, 10240, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(MtpQuery, 6144, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(SharedDown, 2560, 640)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(HiddenProjection, 2560, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(LmHead, 248320, 2560)
NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS(ProposalTail, 117248, 2560)
#undef NINFER_FLASH_NEXT_SMALL_T_LAUNCHERS

struct HcDownSiluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float value) const {
        // Preserve the explicit BF16 projection boundary owned by HyperConnection before
        // applying its scaled SiLU activation.
        const float represented = __bfloat162float(__float2bfloat16_rn(value)) * 0.25F;
        data[static_cast<std::int64_t>(token) * HcDownGeometry::kOutputRows + row] =
            __float2bfloat16_rn(represented / (1.0F + expf(-represented)));
    }
};

template <int ActiveTokens>
void launch_hc_down_silu_exact(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    using Schedule =
        typename Bf16LinearSmallTProductionSchedule<HcDownGeometry, ActiveTokens>::Type;
    constexpr int kBlocks = HcDownGeometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<HcDownGeometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata),
            HcDownSiluOutput{static_cast<__nv_bfloat16*>(out.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_hc_down_silu_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_hc_down_silu_exact<kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kHcDownSiluLaunchers = make_hc_down_silu_launchers(
    std::make_index_sequence<kFlashNextSmallTMax - kBf16SmallTMinTokens + 1>{});

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kBf16SmallTMinTokens);


#define NINFER_FLASH_NEXT_SMALL_T_DISPATCH(NAME)                                      \
    if (weight.n == NAME##Geometry::kOutputRows && weight.k == NAME##Geometry::kInputRows) { \
        k##NAME##Launchers[index](x, weight, out, stream);                            \
        return;                                                                        \
    }
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(HcDown)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(HcUp)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(QsaQueryGate)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(NarrowProjection)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(IndexKey)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(AttentionOutput)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(SharedGateUp)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(GdnQkv)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(MtpQuery)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(SharedDown)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(HiddenProjection)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(LmHead)
    NINFER_FLASH_NEXT_SMALL_T_DISPATCH(ProposalTail)
#undef NINFER_FLASH_NEXT_SMALL_T_DISPATCH
    throw std::invalid_argument("bf16 linear small-T: unsupported exact problem");
}

void launch_bf16_hc_down_silu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    if (x.ne[1] < kBf16SmallTMinTokens || x.ne[1] > kFlashNextSmallTMax ||
        weight.n != HcDownGeometry::kOutputRows || weight.k != HcDownGeometry::kInputRows) {
        throw std::invalid_argument("BF16 HyperConnection down-SiLU: unsupported exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kBf16SmallTMinTokens);
    kHcDownSiluLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail::flash_next
