// QSA group scoring: the scalar warp scorer and the two tensor-core scorers.

#include "ops/softmax_attention/qsa/qsa_common.cuh"

namespace ninfer::ops::flash_next_qsa_detail {

__global__ void score_groups_batched_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int group_extent, int score_stride,
    float* scores) {
    constexpr int kGroupsPerBlock = 8;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int query_token = static_cast<int>(blockIdx.y);
    const int batch_lane = query_token / width;
    const int column = query_token - batch_lane * width;
    const int sequence_length = cache_positions[query_token] + 1;
    const int groups = sequence_length / kRatio;
    for (int group = static_cast<int>(blockIdx.x) * kGroupsPerBlock + warp;
         group < group_extent;
         group += static_cast<int>(gridDim.x) * kGroupsPerBlock) {
    if (column >= valid_columns[batch_lane] || group >= groups) {
        if (lane_id == 0) {
            scores[group + static_cast<std::int64_t>(score_stride) * query_token] = -INFINITY;
        }
        continue;
    }
    const int table_row = table_rows[batch_lane];
    float compressed[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane_id + 32 * item;
        const int position = group * kRatio + (kRatio - 1);
        const int page = physical_page(tables, logical_pages, table_row, position);
        compressed[item] = __bfloat162float(compressed_pages[
            d + static_cast<std::int64_t>(kIndexDim) *
                (position % kPagedKVPageSize + kPagedKVPageSize * page)]);
    }
    float score = 0.0F;
#pragma unroll
    for (int head = 0; head < kIndexHeads; ++head) {
        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane_id + 32 * item;
            dot = fmaf(compressed[item], __bfloat162float(query[d + kIndexDim *
                (head + static_cast<std::int64_t>(kIndexHeads) * query_token)]), dot);
        }
        dot = warp_sum(dot);
        dot = __shfl_sync(0xffffffffU, dot, 0) * rsqrtf(static_cast<float>(kIndexDim));
        score += fmaxf(dot, 0.0F);
    }
    if (lane_id == 0) {
        scores[group + static_cast<std::int64_t>(score_stride) * query_token] = score;
    }
    }
}

// Score one 64-group tile as Q[16,128] x K[128,64]. Only the first four padded
// query rows are live; the warp reduces their ReLU'd MMA outputs into one score
// per compressed group. This is the prefill geometry used by the fast NVIDIA
// implementation and avoids scalar dot products at every query/group pair.
__launch_bounds__(64, 4) __global__ void score_groups_mma_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages,
    const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width,
    int group_extent, int score_stride, float* scores) {
    constexpr int kNTiles = (kScoreColumns / 2) / 8;
    constexpr int kKSteps = kIndexDim / 16;
    const int token = static_cast<int>(blockIdx.y);
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    if (column >= valid_columns[batch_lane]) { return; }
    const int tile_base = static_cast<int>(blockIdx.x) *
                          (kScoreColumns * kScoreTilesPerBlock);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    if (tile_base >= groups) { return; }

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* query_shared = shared;
    __nv_bfloat16* key_shared = query_shared + kScoreRows * kIndexDim;
    constexpr int kVectorsPerRow = kIndexDim / 8;
    const int tid = static_cast<int>(threadIdx.x);
    for (int vector = tid; vector < kScoreRows * kVectorsPerRow; vector += 64) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* destination = query_shared +
            row * kIndexDim + causal_prompt_swz(row, d);
        if (row < kIndexHeads) {
            const __nv_bfloat16* source = query + d + kIndexDim *
                (row + static_cast<std::int64_t>(kIndexHeads) * token);
            cp_async<16, Cache::cg>(destination, source);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_row = (lane & 7) + ((a_matrix & 1) << 3);
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) << 3;
    const unsigned q_base = smem_addr(query_shared) +
                            static_cast<unsigned>(a_row * 256);
    const unsigned q_a = static_cast<unsigned>((a_matrix >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>((lane & 7) << 4);
    const unsigned k_base = smem_addr(key_shared) + static_cast<unsigned>(warp * 8192) +
                            static_cast<unsigned>(b_row * 256) +
                            (static_cast<unsigned>(lane >> 4) << 11);
    const unsigned k_a = static_cast<unsigned>((b_k >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_row << 4);

    const int table_row = table_rows[batch_lane];
#pragma unroll
    for (int tile = 0; tile < kScoreTilesPerBlock; ++tile) {
        const int tile_begin = tile_base + tile * kScoreColumns;
        if (tile_begin >= groups) { break; }
        for (int vector = tid; vector < kScoreColumns * kVectorsPerRow; vector += 64) {
            const int row = vector / kVectorsPerRow;
            const int d = (vector % kVectorsPerRow) * 8;
            __nv_bfloat16* destination = key_shared +
                row * kIndexDim + causal_prompt_swz(row, d);
            const int group = tile_begin + row;
            if (group < groups) {
                const int position = group * kRatio + (kRatio - 1);
                const int page = physical_page(tables, logical_pages, table_row, position);
                const std::int64_t offset = d + static_cast<std::int64_t>(kIndexDim) *
                    (position % kPagedKVPageSize + kPagedKVPageSize * page);
                cp_async<16, Cache::cg>(destination, compressed_pages + offset);
            } else {
                store_vec(destination, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        float accumulator[kNTiles][4] = {};
        unsigned af[2][4];
        unsigned bf[2][kNTiles][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_prompt_swz_addr(q_base, 0U, q_a, q_r));
#pragma unroll
        for (int nt = 0; nt < kNTiles; nt += 2) {
            ldmatrix_x4(bf[0][nt][0], bf[0][nt][1], bf[0][nt + 1][0], bf[0][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               0U, k_a, k_r));
        }
#pragma unroll
        for (int step = 0; step < kKSteps; ++step) {
            const int current = step & 1;
            const int next = current ^ 1;
            if (step + 1 < kKSteps) {
                const unsigned contraction = static_cast<unsigned>((step + 1) << 5);
                ldmatrix_x4(af[next][0], af[next][1], af[next][2], af[next][3],
                            causal_prompt_swz_addr(q_base, contraction, q_a, q_r));
#pragma unroll
                for (int nt = 0; nt < kNTiles; nt += 2) {
                    ldmatrix_x4(
                        bf[next][nt][0], bf[next][nt][1],
                        bf[next][nt + 1][0], bf[next][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               contraction, k_a, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                mma_bf16(accumulator[nt][0], accumulator[nt][1],
                         accumulator[nt][2], accumulator[nt][3],
                         af[current][0], af[current][1], af[current][2], af[current][3],
                         bf[current][nt][0], bf[current][nt][1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            float score0 = gid < kIndexHeads ? fmaxf(accumulator[nt][0], 0.0F) : 0.0F;
            float score1 = gid < kIndexHeads ? fmaxf(accumulator[nt][1], 0.0F) : 0.0F;
            score0 += __shfl_down_sync(0xffffffffU, score0, 4);
            score1 += __shfl_down_sync(0xffffffffU, score1, 4);
            score0 += __shfl_down_sync(0xffffffffU, score0, 8);
            score1 += __shfl_down_sync(0xffffffffU, score1, 8);
            if (gid == 0) {
                const int group0 = tile_begin + warp * 32 + nt * 8 + 2 * lid;
                const int group1 = group0 + 1;
                if (group0 < groups) {
                    scores[group0 + static_cast<std::int64_t>(score_stride) * token] =
                        score0 * rsqrtf(static_cast<float>(kIndexDim));
                }
                if (group1 < groups) {
                    scores[group1 + static_cast<std::int64_t>(score_stride) * token] =
                        score1 * rsqrtf(static_cast<float>(kIndexDim));
                }
            }
        }
        __syncthreads();
    }
}

// Prompt scorer: 16 consecutive query tokens of one batch lane share each staged key tile, so
// the paged keys are read once per 16 tokens instead of once per token. Row 4*t+h of the query
// tile is head h of token t. MMA rows are independent and each token's four ReLU'd heads are
// summed in the one-token scorer's order, so every score keeps that kernel's bits.
__launch_bounds__(64, 4) __global__ void score_groups_mma_tokens_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages,
    const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width, int tokens,
    int group_extent, int score_stride, float* scores) {
    constexpr int kRows = kScoreRows * kScoreTokenTiles;
    constexpr int kNTiles = (kScoreColumns / 2) / 8;
    constexpr int kKSteps = kIndexDim / 16;
    const int first = static_cast<int>(blockIdx.y) * kScoreTokens;
    const int batch_lane = first / width;
    const int first_column = first - batch_lane * width;
    const int live_tokens = min(min(kScoreTokens, tokens - first),
                                valid_columns[batch_lane] - first_column);
    if (live_tokens <= 0) { return; }
    const int tile_base = static_cast<int>(blockIdx.x) *
                          (kScoreColumns * kScoreTilesPerBlock);
    int block_position = 0;
    for (int t = 0; t < live_tokens; ++t) {
        block_position = max(block_position, cache_positions[first + t]);
    }
    const int block_groups = min((block_position + 1) / kRatio, group_extent);
    if (tile_base >= block_groups) { return; }

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* query_shared = shared;
    __nv_bfloat16* key_shared = query_shared + kRows * kIndexDim;
    constexpr int kVectorsPerRow = kIndexDim / 8;
    const int tid = static_cast<int>(threadIdx.x);
    for (int vector = tid; vector < kRows * kVectorsPerRow; vector += 64) {
        const int row = vector / kVectorsPerRow;
        const int d = (vector % kVectorsPerRow) * 8;
        __nv_bfloat16* destination = query_shared +
            row * kIndexDim + causal_prompt_swz(row, d);
        if (row / kIndexHeads < live_tokens) {
            const __nv_bfloat16* source = query + d + kIndexDim *
                (static_cast<std::int64_t>(kIndexHeads) * first + row);
            cp_async<16, Cache::cg>(destination, source);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_row = (lane & 7) + ((a_matrix & 1) << 3);
    const int b_row = lane & 7;
    const int b_k = ((lane >> 3) & 1) << 3;
    const unsigned q_base = smem_addr(query_shared) +
                            static_cast<unsigned>(a_row * 256);
    const unsigned q_a = static_cast<unsigned>((a_matrix >> 1) << 4);
    const unsigned q_r = static_cast<unsigned>((lane & 7) << 4);
    const unsigned k_base = smem_addr(key_shared) + static_cast<unsigned>(warp * 8192) +
                            static_cast<unsigned>(b_row * 256) +
                            (static_cast<unsigned>(lane >> 4) << 11);
    const unsigned k_a = static_cast<unsigned>((b_k >> 3) << 4);
    const unsigned k_r = static_cast<unsigned>(b_row << 4);

    // Lanes with gid 0 and 4 write: in m-tile m, token 4m+q from row gid and 4m+2+q from gid+8.
    const int q = gid >> 2;
    int write_groups[kScoreTokenTiles][2];
#pragma unroll
    for (int m = 0; m < kScoreTokenTiles; ++m) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int t = 4 * m + 2 * half + q;
            write_groups[m][half] = t < live_tokens
                ? min((cache_positions[first + t] + 1) / kRatio, group_extent) : 0;
        }
    }

    const int table_row = table_rows[batch_lane];
#pragma unroll 1
    for (int tile = 0; tile < kScoreTilesPerBlock; ++tile) {
        const int tile_begin = tile_base + tile * kScoreColumns;
        if (tile_begin >= block_groups) { break; }
        for (int vector = tid; vector < kScoreColumns * kVectorsPerRow; vector += 64) {
            const int row = vector / kVectorsPerRow;
            const int d = (vector % kVectorsPerRow) * 8;
            __nv_bfloat16* destination = key_shared +
                row * kIndexDim + causal_prompt_swz(row, d);
            const int group = tile_begin + row;
            if (group < block_groups) {
                const int position = group * kRatio + (kRatio - 1);
                const int page = physical_page(tables, logical_pages, table_row, position);
                const std::int64_t offset = d + static_cast<std::int64_t>(kIndexDim) *
                    (position % kPagedKVPageSize + kPagedKVPageSize * page);
                cp_async<16, Cache::cg>(destination, compressed_pages + offset);
            } else {
                store_vec(destination, make_int4(0, 0, 0, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        float accumulator[kScoreTokenTiles][kNTiles][4] = {};
        unsigned af[2][kScoreTokenTiles][4];
        unsigned bf[2][kNTiles][2];
#pragma unroll
        for (int m = 0; m < kScoreTokenTiles; ++m) {
            ldmatrix_x4(af[0][m][0], af[0][m][1], af[0][m][2], af[0][m][3],
                        causal_prompt_swz_addr(q_base + static_cast<unsigned>(m * 16 * 256),
                                               0U, q_a, q_r));
        }
#pragma unroll
        for (int nt = 0; nt < kNTiles; nt += 2) {
            ldmatrix_x4(bf[0][nt][0], bf[0][nt][1], bf[0][nt + 1][0], bf[0][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               0U, k_a, k_r));
        }
#pragma unroll
        for (int step = 0; step < kKSteps; ++step) {
            const int current = step & 1;
            const int next = current ^ 1;
            if (step + 1 < kKSteps) {
                const unsigned contraction = static_cast<unsigned>((step + 1) << 5);
#pragma unroll
                for (int m = 0; m < kScoreTokenTiles; ++m) {
                    ldmatrix_x4(af[next][m][0], af[next][m][1], af[next][m][2], af[next][m][3],
                                causal_prompt_swz_addr(
                                    q_base + static_cast<unsigned>(m * 16 * 256),
                                    contraction, q_a, q_r));
                }
#pragma unroll
                for (int nt = 0; nt < kNTiles; nt += 2) {
                    ldmatrix_x4(
                        bf[next][nt][0], bf[next][nt][1],
                        bf[next][nt + 1][0], bf[next][nt + 1][1],
                        causal_prompt_swz_addr(k_base + static_cast<unsigned>(nt * 2048),
                                               contraction, k_a, k_r));
                }
            }
#pragma unroll
            for (int m = 0; m < kScoreTokenTiles; ++m) {
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    mma_bf16(accumulator[m][nt][0], accumulator[m][nt][1],
                             accumulator[m][nt][2], accumulator[m][nt][3],
                             af[current][m][0], af[current][m][1],
                             af[current][m][2], af[current][m][3],
                             bf[current][nt][0], bf[current][nt][1]);
                }
            }
        }

#pragma unroll
        for (int m = 0; m < kScoreTokenTiles; ++m) {
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                float score[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    score[i] = fmaxf(accumulator[m][nt][i], 0.0F);
                    score[i] += __shfl_down_sync(0xffffffffU, score[i], 4);
                    score[i] += __shfl_down_sync(0xffffffffU, score[i], 8);
                }
                if ((gid & 3) == 0) {
                    const int group0 = tile_begin + warp * 32 + nt * 8 + 2 * lid;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int groups = write_groups[m][half];
                        float* row = scores + static_cast<std::int64_t>(score_stride) *
                            (first + 4 * m + 2 * half + q);
                        if (group0 < groups) {
                            row[group0] = score[2 * half] * rsqrtf(static_cast<float>(kIndexDim));
                        }
                        if (group0 + 1 < groups) {
                            row[group0 + 1] =
                                score[2 * half + 1] * rsqrtf(static_cast<float>(kIndexDim));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::flash_next_qsa_detail
