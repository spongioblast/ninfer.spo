#pragma once

// Shared by the Flash-Next QSA translation units: geometry, launch shapes, the stored K/V code,
// two device helpers, and the kernel declarations. The kernels are defined in qsa_compress.cu,
// qsa_score.cu, qsa_select.cu, and qsa_attention.cu and launched from qsa_launch.cu.

#include "ninfer/ops/flash_next_qsa.h"

#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/kv_cache/append/fp8_row.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ninfer::ops::flash_next_qsa_detail {

constexpr int kHeadDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kIndexDim = 128;
constexpr int kIndexHeads = 4;
constexpr int kTopGroups = 512;
constexpr int kRatio = 4;
constexpr int kOutputWidth = kTopGroups * kRatio + kRatio - 1;
static_assert(kOutputWidth == kFlashNextQsaSelectedTokens);
constexpr float kTheta = 1.0e7F;
constexpr int kTopkBlockThreads = 256;
constexpr int kTopkItemsPerThread = 8;
constexpr int kTopkBlockItems = kTopkBlockThreads * kTopkItemsPerThread;

// Stored representation of the paged primary K/V cache. FP8 and INT8 keys are stored after the
// normalized D256 Hadamard rotation; queries are rotated to match.
enum class KvCode : std::uint8_t { Bf16, Fp8, Int8 };

constexpr int kScoreRows = 16;
constexpr int kScoreColumns = 64;
constexpr int kScoreTilesPerBlock = 8;
constexpr int kScoreSharedBytes =
    (kScoreRows + kScoreColumns) * kIndexDim * sizeof(__nv_bfloat16);

constexpr int kScoreTokenTiles = 4;
constexpr int kScoreTokens = kIndexHeads * kScoreTokenTiles;
constexpr int kScoreTokensSharedBytes =
    (kScoreRows * kScoreTokenTiles + kScoreColumns) * kIndexDim * sizeof(__nv_bfloat16);

constexpr int kPrefillRows = 16;
constexpr int kPrefillColumns = 16;
// Q/K/V tiles, then each staged key's selected position and paged K/V row.
constexpr int kPrefillSharedBytes =
    (kPrefillRows + 2 * kPrefillColumns) * kHeadDim * sizeof(__nv_bfloat16) +
    2 * kPrefillColumns * sizeof(int);

constexpr int kMaxDecodeAttentionSplits = 64;
constexpr int kSplitHeadsPerBlock = 4;
static_assert(kSplitHeadsPerBlock * 32 == 128, "split staging assumes 128 threads");

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ int physical_page(const int* tables, int logical_pages, int table_row,
                             int position) {
    return tables[position / kPagedKVPageSize + logical_pages * table_row];
}

} // namespace

// qsa_compress.cu
__global__ void split_query_gate_kernel(const __nv_bfloat16* packed,
                                        __nv_bfloat16* query,
                                        __nv_bfloat16* gate,
                                        int tokens);
__global__ void expand_text_positions_kernel(const int* input, int* output, int tokens);
__global__ void prepare_index_query_kernel(const __nv_bfloat16* projected,
                                           const __nv_bfloat16* norm_weight,
                                           const int* positions, int tokens,
                                           __nv_bfloat16* output);
__global__ void append_cache_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
                                    const __nv_bfloat16* raw_index_key,
                                    const int* cache_positions, const int* rope_positions,
                                    const int* valid_columns, const int* table_rows,
                                    int width, int batch, int logical_pages,
                                    const int* tables, __nv_bfloat16* key_pages,
                                    __nv_bfloat16* value_pages, __nv_bfloat16* raw_pages,
                                    int* position_pages);
__global__ void append_cache_fp8_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* raw_index_key, const int* cache_positions,
    const int* rope_positions, const int* valid_columns, const int* table_rows,
    int width, int batch, int logical_pages, const int* tables,
    std::uint8_t* key_pages, std::uint8_t* value_pages,
    __half* key_scales, __half* value_scales,
    __nv_bfloat16* raw_pages, int* position_pages);
__global__ void append_cache_int8_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* raw_index_key, const int* cache_positions,
    const int* rope_positions, const int* valid_columns, const int* table_rows,
    int width, int batch, int logical_pages, const int* tables,
    std::int8_t* key_pages, std::int8_t* value_pages,
    __half* key_scales, __half* value_scales,
    __nv_bfloat16* raw_pages, int* position_pages);
__global__ void hadamard_rows_kernel(__nv_bfloat16* rows, int row_count);
__global__ void compress_index_groups_kernel(__nv_bfloat16* raw_pages,
                                             const int* position_pages,
                                             const __nv_bfloat16* key_norm,
                                             const int* cache_positions,
                                             const int* valid_columns,
                                             const int* table_rows, int width, int batch,
                                             int logical_pages, const int* tables);

// qsa_score.cu
__global__ void score_groups_batched_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int group_extent, int score_stride,
    float* scores);
__launch_bounds__(64, 4) __global__ void score_groups_mma_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages,
    const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width,
    int group_extent, int score_stride, float* scores);
__launch_bounds__(64, 4) __global__ void score_groups_mma_tokens_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* compressed_pages,
    const int* tables, int logical_pages, const int* cache_positions,
    const int* valid_columns, const int* table_rows, int width, int tokens,
    int group_extent, int score_stride, float* scores);

// qsa_select.cu
__global__ void select_shared_indices_kernel(const int* input, const int* selectors,
                                             int* output, int width, int batch);
__global__ void __launch_bounds__(kTopkBlockThreads) select_top_groups_kernel(
    const float* scores, const int* cache_positions, int score_stride, int group_extent,
    int* selected);
__global__ void hierarchical_top_groups_kernel(
    const float* input_scores, const int* input_indices, int input_count, int input_stride,
    float* output_scores, int* output_indices, int output_stride,
    const int* cache_positions, const int* valid_columns, int width, int reduction_level,
    bool expand_final);
__global__ void expand_indices_batched_kernel(const int* selected_groups,
                                              const int* cache_positions,
                                              const int* valid_columns, int width,
                                              int* output);
__global__ void order_groups_like_persistent_topk_kernel(int* selected_groups,
                                                          const int* cache_positions,
                                                          int group_extent);

// qsa_attention.cu
template <KvCode Code>
__launch_bounds__(128, 3) __global__ void selected_attention_batched_kernel(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, int num_splits, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, __nv_bfloat16* output);
template <KvCode Code>
__global__ void selected_attention_split_kernel(
    const __nv_bfloat16* query, const void* key_pages, const void* value_pages,
    const __half* key_scales, const __half* value_scales, const int* indices, const int* tables,
    int logical_pages, const int* cache_positions, const int* valid_columns,
    const int* table_rows, int width, int tokens, float* partial_maximum,
    float* partial_denominator, float* partial_numerator, int num_splits);
__global__ void __launch_bounds__(kHeadDim) reduce_selected_attention_splits_kernel(
    const float* partial_maximum, const float* partial_denominator,
    const float* partial_numerator, const int* valid_columns, int width, int tokens,
    __nv_bfloat16* output, int num_splits);

} // namespace ninfer::ops::flash_next_qsa_detail
