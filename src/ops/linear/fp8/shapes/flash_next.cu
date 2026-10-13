// Flash-Next attention/GDN projections, router, shared-expert gate/up, HyperConnection Down, MTP
// input projections and output head in the optional row-scaled FP8 profile. All routes keep BF16
// activations (weight-only FP8) and use the dense routes of fp8_dense_a16.cuh; the MoE entry
// kernels below fuse the router with the shared-expert gate/up for decode and compact batches.
#include "ops/linear/fp8/fp8_dense_a16.cuh"
#include "ops/linear/fp8/flash_next_launch.h"
#include "ops/common/math.cuh"

namespace ninfer::ops::detail {
namespace {

template <int N, int K>
constexpr Fp8LinearShape flash_next_shape() {
    return fp8_dense::shape<N, K>();
}

// CTA-local rows of one shared-expert projection, rounded at the BF16 projection boundary.
struct SharedRows {
    float* rows;
    int base;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t, float value) const {
        rows[parent_row - base] = __bfloat162float(__float2bfloat16_rn(value));
    }
};

constexpr int kMoeHidden       = 2560;
constexpr int kMoeExperts      = 512;
constexpr int kMoeIntermediate = 640;
// Each 256-thread half of a 512-thread CTA runs one eight-row GEMV block. Rows keep the
// production lane/chain/phase accumulation, so results equal the separate GEMVs bit for bit.
using EntryGemv                = Fp8GemvSchedule<8, 1, 8, 4, Fp8CodeCache::Default, 2, 1>;
constexpr int kEntryThreads    = 2 * EntryGemv::kThreads;
constexpr int kRouterBlocks    = kMoeExperts / EntryGemv::kRowsPerCta / 2;
constexpr int kSharedBlocks    = kMoeIntermediate / EntryGemv::kRowsPerCta;

__global__ __launch_bounds__(kEntryThreads, 1) void fp8_moe_entry_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ router_codes,
    const __nv_bfloat16* __restrict__ router_scales, __nv_bfloat16* __restrict__ scores,
    const std::uint8_t* __restrict__ gate_codes, const __nv_bfloat16* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_codes, const __nv_bfloat16* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ activation) {
    const int block  = static_cast<int>(blockIdx.x);
    const int half   = static_cast<int>(threadIdx.x) / EntryGemv::kThreads;
    const int thread = static_cast<int>(threadIdx.x) % EntryGemv::kThreads;
    if (block < kRouterBlocks) {
        fp8_gemv_block<Fp8Geometry<kMoeExperts, kMoeHidden>, EntryGemv>(
            2 * block + half, thread, x, router_codes, router_scales,
            Fp8ContiguousOutput{scores, kMoeExperts});
        return;
    }
    using SharedGeometry = Fp8Geometry<kMoeIntermediate, kMoeHidden>;
    __shared__ float projected[2][EntryGemv::kRowsPerCta];  // gate, up
    const int shared_block = block - kRouterBlocks;
    const int base         = shared_block * EntryGemv::kRowsPerCta;
    fp8_gemv_block<SharedGeometry, EntryGemv>(shared_block, thread, x,
                                              half == 0 ? gate_codes : up_codes,
                                              half == 0 ? gate_scales : up_scales,
                                              SharedRows{projected[half], base});
    __syncthreads();
    if (threadIdx.x < EntryGemv::kRowsPerCta) {
        const int row          = static_cast<int>(threadIdx.x);
        activation[base + row] = __float2bfloat16(silu(projected[0][row]) * projected[1][row]);
    }
}

// Compact batches (2-16 tokens): each 512-thread CTA hosts two production K-split blocks with
// their own dynamic shared storage. Router CTAs run two row blocks; shared CTAs run the gate and
// up blocks of the same rows, then apply SwiGLU at the BF16 projection boundary. The token tile
// matches the production K-split route of the token count (8 up to eight tokens, then 16).
template <int TileTokens>
struct SplitEntry {
    using Schedule = Fp8A16KSplitSchedule<8, TileTokens, 1>;
    using RouterShared = Fp8A16KSplitShared<Fp8Geometry<kMoeExperts, kMoeHidden>, Schedule>;
    using GateShared   = Fp8A16KSplitShared<Fp8Geometry<kMoeIntermediate, kMoeHidden>, Schedule>;
    static_assert(sizeof(RouterShared) == sizeof(GateShared));
    static constexpr int kThreads            = 2 * Schedule::kThreads;
    static constexpr int kRouterBlocks       = kMoeExperts / Schedule::kRowsPerCta / 2;
    static constexpr int kSharedBlocks       = kMoeIntermediate / Schedule::kRowsPerCta;
    static constexpr std::size_t kBytes      = 2 * sizeof(RouterShared);
};

// CTA-local [rows, tokens] values of one shared-expert projection at the BF16 boundary.
template <int TileTokens>
struct SharedTile {
    float (*values)[TileTokens];
    int base;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        values[parent_row - base][token] = __bfloat162float(__float2bfloat16_rn(value));
    }
};

template <int TileTokens>
__global__ __launch_bounds__(SplitEntry<TileTokens>::kThreads, 1) void fp8_moe_entry_small_t_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ router_codes,
    const __nv_bfloat16* __restrict__ router_scales, __nv_bfloat16* __restrict__ scores,
    const std::uint8_t* __restrict__ gate_codes, const __nv_bfloat16* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_codes, const __nv_bfloat16* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ activation, int tokens) {
    using Entry    = SplitEntry<TileTokens>;
    using Schedule = typename Entry::Schedule;
    extern __shared__ float4 entry_split_storage[];
    const int block  = static_cast<int>(blockIdx.x);
    const int half   = static_cast<int>(threadIdx.x) / Schedule::kThreads;
    const int thread = static_cast<int>(threadIdx.x) % Schedule::kThreads;
    if (block < Entry::kRouterBlocks) {
        auto* shared = reinterpret_cast<typename Entry::RouterShared*>(entry_split_storage) + half;
        fp8_a16_ksplit_block<Fp8Geometry<kMoeExperts, kMoeHidden>, TileTokens, Schedule,
                             Fp8ContiguousOutput, true>(
            2 * block + half, thread, *shared, x, router_codes, router_scales,
            Fp8ContiguousOutput{scores, kMoeExperts}, tokens);
        return;
    }
    __shared__ float projected[2][Schedule::kRowsPerCta][TileTokens];  // gate, up
    auto* shared           = reinterpret_cast<typename Entry::GateShared*>(entry_split_storage) + half;
    const int shared_block = block - Entry::kRouterBlocks;
    const int base         = shared_block * Schedule::kRowsPerCta;
    fp8_a16_ksplit_block<Fp8Geometry<kMoeIntermediate, kMoeHidden>, TileTokens, Schedule,
                         SharedTile<TileTokens>, true>(
        shared_block, thread, *shared, x, half == 0 ? gate_codes : up_codes,
        half == 0 ? gate_scales : up_scales, SharedTile<TileTokens>{projected[half], base},
        tokens);
    __syncthreads();
    const int row   = static_cast<int>(threadIdx.x) % Schedule::kRowsPerCta;
    const int token = static_cast<int>(threadIdx.x) / Schedule::kRowsPerCta;
    if (token < tokens) {
        activation[static_cast<std::int64_t>(token) * kMoeIntermediate + base + row] =
            __float2bfloat16(silu(projected[0][row][token]) * projected[1][row][token]);
    }
}

template <int TileTokens>
void launch_entry_small_t(const Tensor& x, const Weight& router, const Weight& shared_gate,
                          const Weight& shared_up, Tensor& scores, Tensor& activation,
                          cudaStream_t stream) {
    using Entry = SplitEntry<TileTokens>;
    static const cudaError_t configured =
        cudaFuncSetAttribute(fp8_moe_entry_small_t_kernel<TileTokens>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Entry::kBytes));
    CUDA_CHECK(configured);
    fp8_moe_entry_small_t_kernel<TileTokens>
        <<<Entry::kRouterBlocks + Entry::kSharedBlocks, Entry::kThreads, Entry::kBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(router.qdata),
            static_cast<const __nv_bfloat16*>(router.scales),
            static_cast<__nv_bfloat16*>(scores.data),
            static_cast<const std::uint8_t*>(shared_gate.qdata),
            static_cast<const __nv_bfloat16*>(shared_gate.scales),
            static_cast<const std::uint8_t*>(shared_up.qdata),
            static_cast<const __nv_bfloat16*>(shared_up.scales),
            static_cast<__nv_bfloat16*>(activation.data), static_cast<int>(x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void flash_next::launch_fp8_moe_entry_decode(const Tensor& x, const Weight& router,
                                             const Weight& shared_gate, const Weight& shared_up,
                                             Tensor& scores, Tensor& activation,
                                             cudaStream_t stream) {
    const auto fp8 = [](const Weight& weight, int n) {
        return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && weight.n == n &&
               weight.k == kMoeHidden && weight.scales != nullptr;
    };
    const int tokens = x.ne[1];
    if (tokens < 1 || tokens > kFp8MoeEntryMaxTokens || x.ne[0] != kMoeHidden ||
        !fp8(router, kMoeExperts) || !fp8(shared_gate, kMoeIntermediate) ||
        !fp8(shared_up, kMoeIntermediate) || scores.numel() != kMoeExperts * tokens ||
        activation.numel() != kMoeIntermediate * tokens) {
        throw std::invalid_argument("FP8 MoE entry: invalid exact problem");
    }
    if (tokens > 8) {
        launch_entry_small_t<16>(x, router, shared_gate, shared_up, scores, activation, stream);
        return;
    }
    if (tokens > 1) {
        launch_entry_small_t<8>(x, router, shared_gate, shared_up, scores, activation, stream);
        return;
    }
    fp8_moe_entry_decode_kernel<<<kRouterBlocks + kSharedBlocks, kEntryThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(router.qdata),
        static_cast<const __nv_bfloat16*>(router.scales), static_cast<__nv_bfloat16*>(scores.data),
        static_cast<const std::uint8_t*>(shared_gate.qdata),
        static_cast<const __nv_bfloat16*>(shared_gate.scales),
        static_cast<const std::uint8_t*>(shared_up.qdata),
        static_cast<const __nv_bfloat16*>(shared_up.scales),
        static_cast<__nv_bfloat16*>(activation.data));
    CUDA_CHECK(cudaGetLastError());
}

const Fp8LinearShape kFp8N12288K2560 = flash_next_shape<12288, 2560>();
const Fp8LinearShape kFp8N10240K2560 = flash_next_shape<10240, 2560>();
const Fp8LinearShape kFp8N6144K2560  = flash_next_shape<6144, 2560>();
const Fp8LinearShape kFp8N2560K6144  = flash_next_shape<2560, 6144>();
const Fp8LinearShape kFp8N512K2560   = flash_next_shape<512, 2560>();
const Fp8LinearShape kFp8N640K2560   = flash_next_shape<640, 2560>();
const Fp8LinearShape kFp8N248320K2560 = flash_next_shape<248320, 2560>();
const Fp8LinearShape kFp8N320K10240  = flash_next_shape<320, 10240>();
const Fp8LinearShape kFp8N2560K2560  = flash_next_shape<2560, 2560>();
} // namespace ninfer::ops::detail
