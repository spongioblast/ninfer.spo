#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gemma4_attention_tile.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kD       = 512;
constexpr int kHalf    = 256; // output columns per warp
constexpr int kRows    = 64;  // query rows per CTA, 16 per warp pair
constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBc      = 16;
constexpr int kQkNt    = kBc / 8;
constexpr int kQkKs    = kD / 16;
constexpr int kPvNt    = kHalf / 8;
constexpr int kPvKs    = kBc / 16;

constexpr std::size_t kSharedBytes = sizeof(__nv_bfloat16) * (kRows + 2 * kBc) * kD;
static_assert(kSharedBytes <= 99 * 1024);
static_assert(kPagedKVPageSize % kBc == 0);

// grid (query tiles, Hq); CTA (tile, h) covers query rows [64 tile, 64 tile + 64) of head h.
// Warps w and w ^ 4 own the same 16 rows and each computes their full QK scores, so both hold
// the same softmax state; warp w writes output columns [256 (w / 4), 256 (w / 4) + 256). One
// staged K and V tile serves both halves. The arithmetic per row and column is the same as a
// 4-warp CTA per output half, so the result does not depend on the split.
__launch_bounds__(kThreads, 1) __global__ void gemma4_attention_d512_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages, const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ out, int tokens,
    int q_heads, int kv_heads, int window, float scale, std::int32_t logical_pages,
    std::int32_t physical_pages) {
    constexpr unsigned FullMask = 0xffffffffu;

    extern __shared__ __align__(16) __nv_bfloat16 gemma4_d512_smem[];
    __nv_bfloat16* q_s = gemma4_d512_smem; // [kRows, D] swizzled
    __nv_bfloat16* k_s = q_s + kRows * kD; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + kBc * kD;   // [Bc, D] swizzled
    __shared__ int range[3];               // first visible key, highest position, lowest position

    // Later tiles see more keys; start them first.
    const int q0       = (static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x)) * kRows;
    const int q_head   = static_cast<int>(blockIdx.y);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int half     = warp >> 2;
    const int row_base = (warp & 3) * 16;
    const int kvh      = q_head / (q_heads / kv_heads);
    const int rows     = min(kRows, tokens - q0);

    if (tid == 0) {
        range[0] = INT_MAX;
        range[1] = -1;
        range[2] = INT_MAX;
    }
    __syncthreads();
    if (tid < rows) {
        const int position = positions[q0 + tid];
        atomicMin(&range[0], window == 0 ? 0 : max(0, position + 1 - window));
        atomicMax(&range[1], position);
        atomicMin(&range[2], position);
    }
    __syncthreads();
    const int first_key       = range[0];
    const int last_key        = range[1];
    const int lowest_position = range[2];

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>((row_base + a_rowoff) * (kD * 2));
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(k_s) + static_cast<unsigned>(b_rin * (kD * 2)) +
                                 (static_cast<unsigned>(lane >> 4) * (8 * kD * 2));
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(v_s) +
                                 static_cast<unsigned>(((lane >> 3) & 1) * (8 * kD * 2)) +
                                 static_cast<unsigned>(b_rin * (kD * 2));
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    {
        constexpr int VecPerRow       = kD / 8;
        const std::int64_t row_stride = static_cast<std::int64_t>(q_heads) * kD;
        const __nv_bfloat16* q_block  = q + (static_cast<std::int64_t>(q0) * q_heads + q_head) * kD;
#pragma unroll
        for (int chunk = tid; chunk < kRows * VecPerRow; chunk += kThreads) {
            const int row    = chunk / VecPerRow;
            const int d      = (chunk % VecPerRow) * 8;
            __nv_bfloat16* p = &q_s[row * kD + causal_swizzle(row, d)];
            if (row < rows) {
                cp_async<16, Cache::cg>(p, &q_block[row * row_stride + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }

    const int row0      = row_base + gid;
    const int row1      = row0 + 8;
    const int position0 = row0 < rows ? positions[q0 + row0] : -1;
    const int position1 = row1 < rows ? positions[q0 + row1] : -1;
    const int low0      = window == 0 ? 0 : position0 + 1 - window;
    const int low1      = window == 0 ? 0 : position1 + 1 - window;

    float acc[kPvNt][4];
#pragma unroll
    for (int n = 0; n < kPvNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    Bf16KvSoftmaxRow state0, state1;

    const float scale_l2   = scale * kLog2E;
    const int block_begin  = first_key / kBc;
    const int block_end    = last_key < 0 ? block_begin : last_key / kBc + 1;
    const auto physical_of = [&](int block) {
        const int page = (block * kBc) >> kPagedKVPageShift;
        if (page >= logical_pages) { return -1; }
        const int physical = block_table[page];
        return physical < physical_pages ? physical : -1;
    };
    const bool all_rows = rows == kRows;

    int physical = block_begin < block_end ? physical_of(block_begin) : -1;
    cp_commit();
    if (block_begin < block_end) {
        gemma4_stage_keys<kD, kD, kBc, kThreads>(k_s, k_pages, kv_heads, kvh, physical,
                                                 block_begin * kBc, last_key, 0, tid);
    }
    cp_commit();

    const auto step = [&]<bool FullTile>(int block) {
        const int k0            = block * kBc;
        const int next_physical = block + 1 < block_end ? physical_of(block + 1) : physical;
        const bool mapped       = physical >= 0;

        cp_wait<0>();
        __syncthreads();
        gemma4_stage_keys<kD, kD, kBc, kThreads>(v_s, v_pages, kv_heads, kvh, physical, k0,
                                                 last_key, 0, tid);
        cp_commit();

        float score[kQkNt][4];
#pragma unroll
        for (int nt = 0; nt < kQkNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][kQkNt][2];
        const auto load = [&](unsigned(&a)[4], unsigned(&b)[kQkNt][2], int k) {
            const unsigned ck = static_cast<unsigned>(k << 5);
            ldmatrix_x4(a[0], a[1], a[2], a[3], causal_swizzle_address(q_lane_base, ck, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < kQkNt; nt2 += 2) {
                ldmatrix_x4(
                    b[nt2][0], b[nt2][1], b[nt2 + 1][0], b[nt2 + 1][1],
                    causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * (8 * kD * 2)),
                                           ck, k_as, k_r));
            }
        };
        load(af[0], bf[0], 0);
#pragma unroll
        for (int k = 0; k < kQkKs; ++k) {
            const int cur = k & 1;
            if (k + 1 < kQkKs) { load(af[cur ^ 1], bf[cur ^ 1], k + 1); }
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        cp_wait<0>();
        __syncthreads();
        if (block + 1 < block_end) {
            physical = next_physical;
            gemma4_stage_keys<kD, kD, kBc, kThreads>(k_s, k_pages, kv_heads, kvh, physical,
                                                     (block + 1) * kBc, last_key, 0, tid);
            cp_commit();
        }

        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
        if constexpr (FullTile) {
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
                const int key0 = mapped ? k0 + nt * 8 + 2 * lid : INT_MAX - 1;
                const int key1 = key0 + 1;
                score[nt][0]   = (key0 <= position0 && key0 >= low0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (key1 <= position0 && key1 >= low0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (key0 <= position1 && key0 >= low1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (key1 <= position1 && key1 >= low1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float alpha0 = state0.update<!FullTile>(bm0, scale_l2);
        const float alpha1 = state1.update<!FullTile>(bm1, scale_l2);
        const float m0     = state0.scaled_maximum;
        const float m1     = state1.scaled_maximum;

        // P goes to the tensor cores as hi + lo BF16 terms (about 16 mantissa bits): a single BF16
        // or FP16 P moved Gemma's greedy tokens and lowered logit cosine below HF bf16's.
        float bl0 = 0.0f;
        float bl1 = 0.0f;
        unsigned p_frag[kPvKs][4];
        unsigned p_low[kPvKs][4];
        const auto split = [](float a, float b, unsigned& high, unsigned& low) {
            const __nv_bfloat162 rounded = __floats2bfloat162_rn(a, b);
            const float2 back            = __bfloat1622float2(rounded);
            high                         = *reinterpret_cast<const unsigned*>(&rounded);
            low                          = pack_bf16x2(a - back.x, b - back.y);
        };
#pragma unroll
        for (int nt = 0; nt < kQkNt; ++nt) {
            float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -m0));
            float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -m0));
            float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -m1));
            float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -m1));
            if constexpr (!FullTile) {
                p00 = score[nt][0] > -CUDART_INF_F ? p00 : 0.0f;
                p01 = score[nt][1] > -CUDART_INF_F ? p01 : 0.0f;
                p10 = score[nt][2] > -CUDART_INF_F ? p10 : 0.0f;
                p11 = score[nt][3] > -CUDART_INF_F ? p11 : 0.0f;
            }
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk   = nt >> 1;
            const int slot = (nt & 1) * 2;
            split(p00, p01, p_frag[pk][slot], p_low[pk][slot]);
            split(p10, p11, p_frag[pk][slot + 1], p_low[pk][slot + 1]);
        }
        state0.accumulate(alpha0, bl0);
        state1.accumulate(alpha1, bl1);
#pragma unroll
        for (int n = 0; n < kPvNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PvHalf   = kPvNt / 2;
        constexpr int PvLoads  = kPvKs * PvHalf;
        const auto load_values = [&](unsigned(&dst)[4], int li) {
            const int k  = li / PvHalf;
            const int n2 = (li % PvHalf) * 2;
            ldmatrix_x4_t(dst[0], dst[1], dst[2], dst[3],
                          causal_swizzle_address(
                              v_lane_base + static_cast<unsigned>(k * (16 * kD * 2)),
                              static_cast<unsigned>((half * (kHalf / 8) + n2) << 4), v_as, v_r));
        };
        unsigned vf[2][4];
        load_values(vf[0], 0);
#pragma unroll
        for (int li = 0; li < PvLoads; ++li) {
            const int k   = li / PvHalf;
            const int n2  = (li % PvHalf) * 2;
            const int cur = li & 1;
            if (li + 1 < PvLoads) { load_values(vf[cur ^ 1], li + 1); }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_low[k][0], p_low[k][1],
                     p_low[k][2], p_low[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_low[k][0],
                     p_low[k][1], p_low[k][2], p_low[k][3], vf[cur][2], vf[cur][3]);
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    };
    for (int block = block_begin; block < block_end; ++block) {
        const int k0     = block * kBc;
        const bool whole = all_rows && physical_of(block) >= 0 && k0 + kBc - 1 <= lowest_position &&
                           (window == 0 || k0 >= last_key + 1 - window);
        if (whole) {
            step.template operator()<true>(block);
        } else {
            step.template operator()<false>(block);
        }
    }
    cp_wait<0>();

    state0.finish();
    state1.finish();
    const float inv0 = state0.sum > 0.0f ? __frcp_rn(state0.sum) : 0.0f;
    const float inv1 = state1.sum > 0.0f ? __frcp_rn(state1.sum) : 0.0f;
    const int column = half * kHalf;
#pragma unroll
    for (int n = 0; n < kPvNt; ++n) {
        const int d = column + n * 8 + 2 * lid;
        if (row0 < rows) {
            *reinterpret_cast<unsigned*>(
                out + (static_cast<std::int64_t>(q0 + row0) * q_heads + q_head) * kD + d) =
                pack_bf16x2(acc[n][0] * inv0, acc[n][1] * inv0);
        }
        if (row1 < rows) {
            *reinterpret_cast<unsigned*>(
                out + (static_cast<std::int64_t>(q0 + row1) * q_heads + q_head) * kD + d) =
                pack_bf16x2(acc[n][2] * inv1, acc[n][3] * inv1);
        }
    }
}

} // namespace

void gemma4_attention_mma_d512_launch(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                                      const Tensor& block_table, const Tensor& positions,
                                      int window, float scale, Tensor& out, cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma4_attention_d512_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSharedBytes)));
        configured = true;
    }
    const int tokens = q.ne[2];
    const dim3 grid(static_cast<unsigned>((tokens + kRows - 1) / kRows),
                    static_cast<unsigned>(q.ne[1]));
    gemma4_attention_d512_kernel<<<grid, kThreads, kSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k_pages.data),
        static_cast<const __nv_bfloat16*>(v_pages.data),
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        tokens, q.ne[1], k_pages.ne[2], window, scale, block_table.ne[0], k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
