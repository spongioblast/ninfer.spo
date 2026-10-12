// QSA top-k: the single-CTA and hierarchical group selectors, the persistent-top-k order, index
// expansion, and the shared-index copy for MTP rows.

#include "ops/softmax_attention/qsa/qsa_common.cuh"
#include <cub/block/block_radix_sort.cuh>

namespace ninfer::ops::flash_next_qsa_detail {
namespace {

// Inclusive block scan over the 256 threads of a top-k block; returns the block total.
__device__ __forceinline__ int topk_block_inclusive_scan(int& value, int* warp_totals) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int add = __shfl_up_sync(0xffffffffU, value, offset);
        if (lane >= offset) { value += add; }
    }
    if (lane == 31) { warp_totals[warp] = value; }
    __syncthreads();
    int base = 0;
    int total = 0;
#pragma unroll
    for (int w = 0; w < kTopkBlockThreads / 32; ++w) {
        const int count = warp_totals[w];
        base += w < warp ? count : 0;
        total += count;
    }
    value += base;
    __syncthreads();
    return total;
}

constexpr int kSelectStagedGroups = 8192;
constexpr int kSelectCoarseShift = 20;
constexpr int kSelectCoarseBins = 1 << (32 - kSelectCoarseShift);
constexpr int kSelectCandidates = 2048;
static_assert(kSelectCoarseBins * sizeof(int) + kSelectCandidates * sizeof(unsigned) <=
              kSelectStagedGroups * sizeof(unsigned));

// Radix-select among `count` candidate bit patterns that share `prefix` above the coarse digit:
// returns the bits of the rank-th greatest and leaves in *shared_rank how many groups equal to it
// belong to the selection.
__device__ unsigned select_candidate_threshold(const unsigned* candidates, int count,
                                               unsigned prefix, int rank, int* histogram,
                                               int* warp_totals, unsigned* shared_prefix,
                                               int* shared_rank) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        *shared_prefix = prefix;
        *shared_rank = rank;
    }
    for (int shift = kSelectCoarseShift - 8; shift > -8; shift -= 8) {
        const int digit_shift = max(shift, 0);
        const int width = shift >= 0 ? 8 : 8 + shift;
        histogram[tid] = 0;
        __syncthreads();
        const unsigned mask = 0xffffffffU << (digit_shift + width);
        const unsigned current_prefix = *shared_prefix;
        for (int item = tid; item < count; item += kTopkBlockThreads) {
            const unsigned bits = candidates[item];
            if ((bits & mask) == current_prefix) {
                atomicAdd(histogram + ((bits >> digit_shift) & ((1U << width) - 1U)), 1);
            }
        }
        __syncthreads();
        const int own = histogram[255 - tid];
        int at_or_above = own;
        const int current_rank = *shared_rank;
        topk_block_inclusive_scan(at_or_above, warp_totals);
        if (at_or_above >= current_rank && at_or_above - own < current_rank) {
            *shared_prefix = current_prefix | (static_cast<unsigned>(255 - tid) << digit_shift);
            *shared_rank = current_rank - (at_or_above - own);
        }
        __syncthreads();
    }
    return *shared_prefix;
}

} // namespace

__global__ void select_shared_indices_kernel(const int* input, const int* selectors,
                                             int* output, int width, int batch) {
    const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(blockIdx.y);
    if (item >= kOutputWidth || lane >= batch) { return; }
    const int column = selectors[lane];
    if (column < 0 || column >= width) { return; }
    output[item + static_cast<std::int64_t>(kOutputWidth) * lane] =
        input[item + static_cast<std::int64_t>(kOutputWidth) *
                         (column + width * lane)];
}

// Exact radix selection of the 512 greatest non-negative QSA scores.  Keeping
// only the threshold lets one block select a token without materializing a
// second score buffer or capturing CUB's size-dependent graph topology. Scores are staged in
// shared memory when they fit; digits are located with a block scan. Longer rows are
// bandwidth-bound, so they are read three times instead of six: a 12-bit coarse histogram,
// a gather of the bin holding rank 512 for an in-shared-memory radix select, and the emit pass.
// A boundary bin over kSelectCandidates falls back to the full-row radix passes.
__global__ void __launch_bounds__(kTopkBlockThreads) select_top_groups_kernel(
    const float* scores, const int* cache_positions, int score_stride, int group_extent,
    int* selected) {
    const int token = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    int* output = selected + static_cast<std::int64_t>(kTopGroups) * token;
    // Every visible group is selected before the compressed history exceeds the budget.  Emit
    // those groups directly in causal order.  Routing the degenerate case through the atomic
    // threshold compaction below made their order CTA-scheduler dependent; the selected-attention
    // reduction consequently changed between CUDA Graph replays even though the selected set was
    // identical.
    if (groups <= kTopGroups) {
        for (int item = tid; item < kTopGroups; item += kTopkBlockThreads) {
            output[item] = item < groups ? item : -1;
        }
        return;
    }
    __shared__ __align__(16) unsigned staged[kSelectStagedGroups];
    __shared__ int histogram[256];
    __shared__ int warp_totals[kTopkBlockThreads / 32];
    __shared__ unsigned prefix;
    __shared__ int rank;
    __shared__ int candidate_count;
    __shared__ int coarse_bin;
    const float* row = scores + static_cast<std::int64_t>(score_stride) * token;
    const bool use_staged = groups <= kSelectStagedGroups;
    const auto bits_at = [&](int group) {
        return use_staged ? staged[group] : __float_as_uint(row[group]);
    };

    bool threshold_found = false;
    unsigned threshold = 0;
    if (!use_staged) {
        int* coarse = reinterpret_cast<int*>(staged);
        unsigned* candidates = staged + kSelectCoarseBins;
        for (int bin = tid; bin < kSelectCoarseBins; bin += kTopkBlockThreads) { coarse[bin] = 0; }
        if (tid == 0) { candidate_count = 0; }
        __syncthreads();
        for (int group = tid; group < groups; group += kTopkBlockThreads) {
            atomicAdd(coarse + (__float_as_uint(row[group]) >> kSelectCoarseShift), 1);
        }
        __syncthreads();
        // Thread t owns 16 bins counted from the top; one thread finds the bin holding rank 512.
        constexpr int kBinsPerThread = kSelectCoarseBins / kTopkBlockThreads;
        const int top = kSelectCoarseBins - 1 - tid * kBinsPerThread;
        int own = 0;
#pragma unroll
        for (int i = 0; i < kBinsPerThread; ++i) { own += coarse[top - i]; }
        int at_or_above = own;
        topk_block_inclusive_scan(at_or_above, warp_totals);
        if (at_or_above >= kTopGroups && at_or_above - own < kTopGroups) {
            int above = at_or_above - own;
            for (int i = 0; i < kBinsPerThread; ++i) {
                const int count = coarse[top - i];
                if (above + count >= kTopGroups) {
                    coarse_bin = top - i;
                    rank = kTopGroups - above;
                    break;
                }
                above += count;
            }
        }
        __syncthreads();
        const unsigned bin = static_cast<unsigned>(coarse_bin);
        const int bin_rank = rank;
        for (int group = tid; group < groups; group += kTopkBlockThreads) {
            const unsigned bits = __float_as_uint(row[group]);
            if ((bits >> kSelectCoarseShift) == bin) {
                const int slot = atomicAdd(&candidate_count, 1);
                if (slot < kSelectCandidates) { candidates[slot] = bits; }
            }
        }
        __syncthreads();
        const int count = candidate_count;
        if (count <= kSelectCandidates) {
            threshold = select_candidate_threshold(candidates, count, bin << kSelectCoarseShift,
                                                   bin_rank, histogram, warp_totals, &prefix,
                                                   &rank);
            threshold_found = true;
        }
        __syncthreads();
    } else {
        for (int group = tid; group < groups; group += kTopkBlockThreads) {
            staged[group] = __float_as_uint(row[group]);
        }
    }
    if (!threshold_found) {
        if (tid == 0) {
            prefix = 0;
            rank = kTopGroups;
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            histogram[tid] = 0;
            __syncthreads();
            const unsigned mask = shift == 24 ? 0U : (0xffffffffU << (shift + 8));
            const unsigned current_prefix = prefix;
            for (int group = tid; group < groups; group += kTopkBlockThreads) {
                const unsigned bits = bits_at(group);
                if ((bits & mask) == current_prefix) {
                    atomicAdd(histogram + ((bits >> shift) & 255U), 1);
                }
            }
            __syncthreads();
            // Thread i owns digit 255 - i, so the inclusive scan counts the groups at or above it.
            const int own = histogram[255 - tid];
            int at_or_above = own;
            const int current_rank = rank;
            topk_block_inclusive_scan(at_or_above, warp_totals);
            if (at_or_above >= current_rank && at_or_above - own < current_rank) {
                prefix = current_prefix | (static_cast<unsigned>(255 - tid) << shift);
                rank = current_rank - (at_or_above - own);
            }
            __syncthreads();
        }
        threshold = prefix;
    }
    // Emit every score above the threshold, then the lowest-id equal scores, each in ascending
    // group order: the exact (score descending, group id ascending) top-k contract used by the
    // hierarchical selector as well. `rank` equal groups complete the 512. The whole block steps
    // through the row and places its groups with one scan of packed (above, equal) counts.
    const int equal_taken = rank;
    const int above_total = kTopGroups - equal_taken;
    int above_base = 0;
    int equal_base = 0;
    for (int start = 0; start < groups && (above_base < above_total || equal_base < equal_taken);
         start += kTopkBlockThreads) {
        const int group = start + tid;
        const unsigned bits = group < groups ? bits_at(group) : 0U;
        const int is_above = group < groups && bits > threshold;
        const int is_equal = group < groups && bits == threshold;
        int packed = is_above | (is_equal << 16);
        const int total = topk_block_inclusive_scan(packed, warp_totals);
        if (is_above) { output[above_base + (packed & 0xffff) - 1] = group; }
        if (is_equal) {
            const int slot = equal_base + (packed >> 16) - 1;
            if (slot < equal_taken) { output[above_total + slot] = group; }
        }
        above_base += total & 0xffff;
        equal_base += total >> 16;
    }
}

// Exact hierarchical top-k: every global top-512 item must be in the local top-512 of the
// 2048-item partition that contains it. Repeating that reduction exposes enough parallelism for
// long-context decode, where a single-CTA radix scan otherwise dominates all QSA work.
__global__ void hierarchical_top_groups_kernel(
    const float* input_scores, const int* input_indices, int input_count, int input_stride,
    float* output_scores, int* output_indices, int output_stride,
    const int* cache_positions, const int* valid_columns, int width, int reduction_level,
    bool expand_final) {
    // Scores are non-negative.  Encode the global group id into the low key bits so equal-score
    // selection has one exact ordering at every hierarchy level: greater score first, then lower
    // group id.  Sorting float keys alone lets CUB choose different tied groups between graph
    // replays, which changes the subsequent attention reduction and can alter an MTP draft.
    using Sort = cub::BlockRadixSort<std::uint64_t, kTopkBlockThreads,
                                     kTopkItemsPerThread, int>;
    __shared__ typename Sort::TempStorage storage;
    const int token = static_cast<int>(blockIdx.y);
    const int begin = static_cast<int>(blockIdx.x) * kTopkBlockItems;
    int visible = (cache_positions[token] + 1) / kRatio;
    for (int level = 0; level < reduction_level; ++level) {
        visible = ((visible + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups;
    }
    if (begin >= visible && !expand_final) {
        for (int rank = static_cast<int>(threadIdx.x); rank < kTopGroups;
             rank += static_cast<int>(blockDim.x)) {
            const int output = static_cast<int>(blockIdx.x) * kTopGroups + rank +
                static_cast<std::int64_t>(output_stride) * token;
            output_scores[output] = -INFINITY;
            output_indices[output] = -1;
        }
        return;
    }
    std::uint64_t keys[kTopkItemsPerThread];
    int values[kTopkItemsPerThread];
#pragma unroll
    for (int item = 0; item < kTopkItemsPerThread; ++item) {
        const int local = static_cast<int>(threadIdx.x) * kTopkItemsPerThread + item;
        const int index = begin + local;
        if (index < min(visible, input_count)) {
            values[item] = input_indices == nullptr
                ? index
                : input_indices[index + static_cast<std::int64_t>(input_stride) * token];
            const float score =
                input_scores[index + static_cast<std::int64_t>(input_stride) * token];
            keys[item] = values[item] < 0
                ? 0ULL
                : (static_cast<std::uint64_t>(__float_as_uint(score)) << 32U) |
                      static_cast<std::uint32_t>(INT_MAX - values[item]);
        } else {
            keys[item] = 0ULL;
            values[item] = -1;
        }
    }
    Sort(storage).SortDescending(keys, values);
#pragma unroll
    for (int item = 0; item < kTopkItemsPerThread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * kTopkItemsPerThread + item;
        if (rank < kTopGroups) {
            if (expand_final) {
                const int batch_lane = token / width;
                const int column = token - batch_lane * width;
                const int selected_count = min((cache_positions[token] + 1) / kRatio,
                                               kTopGroups);
                const int group = column < valid_columns[batch_lane] && rank < selected_count
                    ? values[item]
                    : -1;
#pragma unroll
                for (int member = 0; member < kRatio; ++member) {
                    output_indices[rank * kRatio + member +
                                   static_cast<std::int64_t>(kOutputWidth) * token] =
                        group < 0 ? -1 : group * kRatio + member;
                }
            } else {
                const int output = static_cast<int>(blockIdx.x) * kTopGroups + rank +
                    static_cast<std::int64_t>(output_stride) * token;
                output_scores[output] = values[item] < 0
                    ? -INFINITY
                    : __uint_as_float(static_cast<unsigned>(keys[item] >> 32U));
                output_indices[output] = values[item];
            }
        }
    }
    if (expand_final && threadIdx.x < kRatio - 1) {
        const int batch_lane = token / width;
        const int column = token - batch_lane * width;
        const int sequence_length = cache_positions[token] + 1;
        const int tail_begin = (sequence_length / kRatio) * kRatio;
        const int selected_count = min(sequence_length / kRatio, kTopGroups);
        const int tail = static_cast<int>(threadIdx.x);
        output_indices[selected_count * kRatio + tail +
                       static_cast<std::int64_t>(kOutputWidth) * token] =
            column < valid_columns[batch_lane] && tail_begin + tail < sequence_length
                ? tail_begin + tail
                : -1;
    }
}

__global__ void expand_indices_batched_kernel(const int* selected_groups,
                                              const int* cache_positions,
                                              const int* valid_columns, int width,
                                              int* output) {
    const int token = static_cast<int>(blockIdx.y);
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputWidth) { return; }
    const int batch_lane = token / width;
    const int column = token - batch_lane * width;
    int* token_output = output + static_cast<std::int64_t>(kOutputWidth) * token;
    if (column >= valid_columns[batch_lane]) {
        token_output[index] = -1;
        return;
    }
    const int sequence_length = cache_positions[token] + 1;
    const int groups = sequence_length / kRatio;
    const int selected_count = min(groups, kTopGroups);
    const int selected_tokens = selected_count * kRatio;
    if (index < selected_tokens) {
        const int group = selected_groups[index / kRatio +
            static_cast<std::int64_t>(kTopGroups) * token];
        token_output[index] = group < 0 ? -1 : group * kRatio + index % kRatio;
    } else {
        const int tail = index - selected_tokens;
        const int tail_begin = groups * kRatio;
        token_output[index] = tail < sequence_length - tail_begin ? tail_begin + tail : -1;
    }
}

// Match the collection order used by vLLM's short-row persistent top-k: its vectorized
// histogram collector visits one element from each float4 before advancing to the next lane.
// The selected set is unchanged; only the deterministic accumulation order is reproduced.
__global__ void order_groups_like_persistent_topk_kernel(int* selected_groups,
                                                          const int* cache_positions,
                                                          int group_extent) {
    using Sort = cub::BlockRadixSort<std::uint32_t, kTopkBlockThreads, 2, int>;
    __shared__ typename Sort::TempStorage storage;
    const int token = static_cast<int>(blockIdx.x);
    const int groups = min((cache_positions[token] + 1) / kRatio, group_extent);
    if (groups <= kTopGroups) { return; }
    // Keys pack the visitation fields most-significant first in the fewest bits; absent groups
    // take the one value above every packed key. The stable sort over [0, key_bits) orders
    // exactly as the lexicographic field comparison.
    const int key_bits = group_extent <= 8192 ? 14 : (group_extent <= 32768 ? 16 : 17);
    std::uint32_t keys[2];
    int values[2];
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * 2 + item;
        const int group = selected_groups[rank +
            static_cast<std::int64_t>(kTopGroups) * token];
        values[item] = group;
        if (group < 0) {
            keys[item] = 1U << (key_bits - 1);
        } else if (group_extent <= 8192) {
            // Lane (group & 3), then float4 index (group >> 2 < 2048).
            keys[item] = (static_cast<std::uint32_t>(group & 3) << 11U) |
                         static_cast<std::uint32_t>(group >> 2);
        } else if (group_extent <= 32768) {
            // vLLM's medium collector uses one 1024-thread CTA and walks the row at a
            // 1024-element stride.  Preserve its warp/iteration/lane visitation order.
            keys[item] = (static_cast<std::uint32_t>((group & 1023) >> 5) << 10U) |
                         (static_cast<std::uint32_t>(group >> 10) << 5U) |
                         static_cast<std::uint32_t>(group & 31);
        } else {
            // Decode logits have the full 65536-group capacity.  With vLLM's 35968-byte
            // decode shared-memory cap this is eight 8192-element CTAs.  Each CTA reserves
            // one contiguous output span, then its 1024 threads collect at stride 1024.
            constexpr int kRadixChunk = 8192;
            const int local = group & (kRadixChunk - 1);
            keys[item] = (static_cast<std::uint32_t>(group / kRadixChunk) << 13U) |
                         (static_cast<std::uint32_t>((local & 1023) >> 5) << 8U) |
                         (static_cast<std::uint32_t>(local >> 10) << 5U) |
                         static_cast<std::uint32_t>(local & 31);
        }
    }
    Sort(storage).Sort(keys, values, 0, key_bits);
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * 2 + item;
        selected_groups[rank + static_cast<std::int64_t>(kTopGroups) * token] = values[item];
    }
}

} // namespace ninfer::ops::flash_next_qsa_detail
