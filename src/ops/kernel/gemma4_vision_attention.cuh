#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

constexpr int kGemma4VisionHeadDim = 72;
constexpr int kGemma4VisionHeads   = 16;

constexpr int kVisionRows    = 64;  // queries per CTA, 16 per warp
constexpr int kVisionKeys    = 64;  // keys per tile
constexpr int kVisionThreads = 128;
constexpr int kVisionStride  = 88;  // shared row stride in bf16: 72 dims, zeros to 80, 8 spare
constexpr int kVisionTile    = kVisionKeys * kVisionStride;

// Q is staged in the first K buffer and lives in registers after that, so two CTAs fit per SM.
constexpr std::size_t kVisionSharedBytes =
    sizeof(__nv_bfloat16) * static_cast<std::size_t>(4 * kVisionTile) +
    sizeof(int) * 2 * kVisionKeys;
static_assert(kVisionRows == kVisionKeys);

// A running sum(expf(s - max)) moved to a larger max. An empty sum stays empty.
__device__ __forceinline__ float rescale(float sum, float from, float to) {
    return sum == 0.0f ? 0.0f : sum * expf(from - to);
}

// x = hi + lo with both halves bf16, so an fp32 weight keeps about 16 bits through the mma.
__device__ __forceinline__ void split_bf16x2(float a, float b, unsigned& hi, unsigned& lo) {
    const __nv_bfloat16 ha = __float2bfloat16_rn(a);
    const __nv_bfloat16 hb = __float2bfloat16_rn(b);
    hi = pack_bf16x2(__bfloat162float(ha), __bfloat162float(hb));
    lo = pack_bf16x2(a - __bfloat162float(ha), b - __bfloat162float(hb));
}

// grid (ceil(T/64), 16). One CTA owns 64 queries of one head and sweeps every key tile twice:
// first the row maximum and sum(expf(s - max)) online, then expf(s - max) times V, divided by
// the sum at the end. Logits, the softmax, and the weights stay fp32 (each weight enters the
// bf16 mma as a hi and a lo half), as in an fp32 reference attention, and no T x T buffer is
// stored. Pad keys (mask 0) never enter the max, the sum, or the output; a row without a valid
// key writes zeros.
__launch_bounds__(kVisionThreads) __global__
    void gemma4_vision_flash_kernel(const __nv_bfloat16* __restrict__ q,
                                    const __nv_bfloat16* __restrict__ k,
                                    const __nv_bfloat16* __restrict__ v,
                                    const std::int32_t* __restrict__ mask,
                                    __nv_bfloat16* __restrict__ out, int tokens) {
    constexpr int D        = kGemma4VisionHeadDim;
    constexpr int H        = kGemma4VisionHeads;
    constexpr int S        = kVisionStride;
    constexpr int Chunks   = D / 8; // 16-byte chunks per row
    constexpr int QkSteps  = 5;     // 80 / 16
    constexpr int KeyTiles = kVisionKeys / 8;
    constexpr int PvSteps  = kVisionKeys / 16;
    constexpr int DimTiles = 10; // 80 / 8; tile 9 is the zero pad
    constexpr std::int64_t RowStride = static_cast<std::int64_t>(H) * D;

    extern __shared__ __align__(16) __nv_bfloat16 vision_smem[];
    __nv_bfloat16* k_s = vision_smem;           // [2][keys][S]
    __nv_bfloat16* v_s = k_s + 2 * kVisionTile; // [2][keys][S]
    __nv_bfloat16* q_s = k_s;
    int* valid_s       = reinterpret_cast<int*>(v_s + 2 * kVisionTile); // [2][keys]

    const int q0    = static_cast<int>(blockIdx.x) * kVisionRows;
    const int head  = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int gid   = lane >> 2;
    const int lid   = lane & 3;
    const int tiles = (tokens + kVisionKeys - 1) / kVisionKeys;

    // Dimensions 72..79 stay zero in Q, K, and V; the copies below never touch them.
    for (int row = tid; row < 4 * kVisionKeys; row += kVisionThreads) {
        store_vec(k_s + row * S + D, make_int4(0, 0, 0, 0));
    }

    const auto copy_rows = [&](__nv_bfloat16* dst, const __nv_bfloat16* src, int first) {
        for (int chunk = tid; chunk < kVisionKeys * Chunks; chunk += kVisionThreads) {
            const int row     = chunk / Chunks;
            const int d       = (chunk % Chunks) * 8;
            const int token   = first + row;
            const bool inside = token < tokens;
            const __nv_bfloat16* from =
                src + (static_cast<std::int64_t>(inside ? token : 0) * H + head) * D + d;
            cp_async_zfill<16, Cache::cg>(dst + row * S + d, from, inside ? 16 : 0);
        }
    };
    const auto load_tile = [&](int buffer, int tile, bool values) {
        const int first = tile * kVisionKeys;
        copy_rows(k_s + buffer * kVisionTile, k, first);
        if (values) { copy_rows(v_s + buffer * kVisionTile, v, first); }
        if (tid < kVisionKeys) {
            const int key = first + tid;
            valid_s[buffer * kVisionKeys + tid] =
                key < tokens && (mask == nullptr || mask[key] != 0) ? 1 : 0;
        }
        cp_commit();
    };

    copy_rows(q_s, q, q0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    unsigned qf[QkSteps][4];
    {
        const int row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int col = (lane >> 4) * 8;
#pragma unroll
        for (int step = 0; step < QkSteps; ++step) {
            ldmatrix_x4(qf[step][0], qf[step][1], qf[step][2], qf[step][3],
                        smem_addr(q_s + row * S + col + step * 16));
        }
    }
    __syncthreads();

    const int k_row = (lane & 7) + (lane >> 4) * 8;
    const int k_col = ((lane >> 3) & 1) * 8;
    const int v_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int v_col = (lane >> 4) * 8;

    const auto scores = [&](int buffer, float (&s)[KeyTiles][4]) {
        const __nv_bfloat16* keys = k_s + buffer * kVisionTile;
#pragma unroll
        for (int nt = 0; nt < KeyTiles; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f; }
#pragma unroll
        for (int step = 0; step < QkSteps; ++step) {
#pragma unroll
            for (int nt = 0; nt < KeyTiles; nt += 2) {
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3,
                            smem_addr(keys + (nt * 8 + k_row) * S + k_col + step * 16));
                mma_bf16(s[nt][0], s[nt][1], s[nt][2], s[nt][3], qf[step][0], qf[step][1],
                         qf[step][2], qf[step][3], b0, b1);
                mma_bf16(s[nt + 1][0], s[nt + 1][1], s[nt + 1][2], s[nt + 1][3], qf[step][0],
                         qf[step][1], qf[step][2], qf[step][3], b2, b3);
            }
        }
    };

    float max0 = -CUDART_INF_F;
    float max1 = -CUDART_INF_F;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float acc[DimTiles][4];
#pragma unroll
    for (int n = 0; n < DimTiles; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f; }

#pragma unroll 1
    for (int sweep = 0; sweep < 2; ++sweep) {
        const bool values = sweep == 1;
        load_tile(0, 0, values);
#pragma unroll 1
        for (int tile = 0; tile < tiles; ++tile) {
            const int buffer = tile & 1;
            if (tile + 1 < tiles) {
                load_tile(buffer ^ 1, tile + 1, values);
            } else {
                cp_commit();
            }
            cp_wait<1>();
            __syncthreads();

            float s[KeyTiles][4];
            scores(buffer, s);
            const int* valid = valid_s + buffer * kVisionKeys;
            if (sweep == 0) {
                float top0 = max0;
                float top1 = max1;
#pragma unroll
                for (int nt = 0; nt < KeyTiles; ++nt) {
                    const int key = nt * 8 + 2 * lid;
                    if (valid[key]) {
                        top0 = fmaxf(top0, s[nt][0]);
                        top1 = fmaxf(top1, s[nt][2]);
                    }
                    if (valid[key + 1]) {
                        top0 = fmaxf(top0, s[nt][1]);
                        top1 = fmaxf(top1, s[nt][3]);
                    }
                }
                sum0 = rescale(sum0, max0, top0);
                sum1 = rescale(sum1, max1, top1);
                max0 = top0;
                max1 = top1;
#pragma unroll
                for (int nt = 0; nt < KeyTiles; ++nt) {
                    const int key = nt * 8 + 2 * lid;
                    if (valid[key]) {
                        sum0 += expf(s[nt][0] - max0);
                        sum1 += expf(s[nt][2] - max1);
                    }
                    if (valid[key + 1]) {
                        sum0 += expf(s[nt][1] - max0);
                        sum1 += expf(s[nt][3] - max1);
                    }
                }
            } else {
                unsigned hi[PvSteps][4];
                unsigned lo[PvSteps][4];
#pragma unroll
                for (int nt = 0; nt < KeyTiles; ++nt) {
                    const int key   = nt * 8 + 2 * lid;
                    const bool ok0  = valid[key] != 0;
                    const bool ok1  = valid[key + 1] != 0;
                    const float e00 = ok0 ? expf(s[nt][0] - max0) : 0.0f;
                    const float e01 = ok1 ? expf(s[nt][1] - max0) : 0.0f;
                    const float e10 = ok0 ? expf(s[nt][2] - max1) : 0.0f;
                    const float e11 = ok1 ? expf(s[nt][3] - max1) : 0.0f;
                    const int step  = nt >> 1;
                    const int slot  = (nt & 1) * 2;
                    split_bf16x2(e00, e01, hi[step][slot], lo[step][slot]);
                    split_bf16x2(e10, e11, hi[step][slot + 1], lo[step][slot + 1]);
                }
                const __nv_bfloat16* vals = v_s + buffer * kVisionTile;
#pragma unroll
                for (int step = 0; step < PvSteps; ++step) {
#pragma unroll
                    for (int n = 0; n < DimTiles; n += 2) {
                        unsigned b0, b1, b2, b3;
                        ldmatrix_x4_t(b0, b1, b2, b3,
                                      smem_addr(vals + (step * 16 + v_row) * S + n * 8 + v_col));
                        mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], hi[step][0],
                                 hi[step][1], hi[step][2], hi[step][3], b0, b1);
                        mma_bf16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3],
                                 hi[step][0], hi[step][1], hi[step][2], hi[step][3], b2, b3);
                        mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], lo[step][0],
                                 lo[step][1], lo[step][2], lo[step][3], b0, b1);
                        mma_bf16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3],
                                 lo[step][0], lo[step][1], lo[step][2], lo[step][3], b2, b3);
                    }
                }
            }
            __syncthreads();
        }
        cp_wait<0>();
        if (sweep == 0) {
#pragma unroll
            for (int offset = 1; offset < 4; offset <<= 1) {
                const float other_max0 = __shfl_xor_sync(0xffffffffU, max0, offset);
                const float other_max1 = __shfl_xor_sync(0xffffffffU, max1, offset);
                const float other_sum0 = __shfl_xor_sync(0xffffffffU, sum0, offset);
                const float other_sum1 = __shfl_xor_sync(0xffffffffU, sum1, offset);
                const float top0       = fmaxf(max0, other_max0);
                const float top1       = fmaxf(max1, other_max1);
                sum0 = rescale(sum0, max0, top0) + rescale(other_sum0, other_max0, top0);
                sum1 = rescale(sum1, max1, top1) + rescale(other_sum1, other_max1, top1);
                max0 = top0;
                max1 = top1;
            }
        }
    }

    const float inv0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const int row0   = q0 + warp * 16 + gid;
    const int row1   = row0 + 8;
#pragma unroll
    for (int n = 0; n < DimTiles - 1; ++n) {
        const int d = n * 8 + 2 * lid;
        if (row0 < tokens) {
            store_vec(out + static_cast<std::int64_t>(row0) * RowStride + head * D + d,
                      pack_bf16x2(acc[n][0] * inv0, acc[n][1] * inv0));
        }
        if (row1 < tokens) {
            store_vec(out + static_cast<std::int64_t>(row1) * RowStride + head * D + d,
                      pack_bf16x2(acc[n][2] * inv1, acc[n][3] * inv1));
        }
    }
}

} // namespace ninfer::ops::detail
