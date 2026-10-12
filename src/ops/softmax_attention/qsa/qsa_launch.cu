#include "ninfer/ops/flash_next_qsa.h"

#include "ops/flash_next_work.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/linear/bf16/flash_next/bf16_launch.h"
#include "ops/softmax_attention/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {

using namespace flash_next_qsa_detail;

namespace {

// Match the split-K profiles used by the reference vLLM Triton kernel for this
// target's two KV heads. In particular, four-row MTP verification uses 32
// splits rather than the single-token decode profile's 64 splits. Every compact
// verification batch (3 to 16 tokens) keeps 32 splits, so a row's attention
// reduction does not depend on how many rows share the batch.
int decode_attention_splits(int tokens) {
    const int base_programs = tokens * kKvHeads;
    if (base_programs <= 4) { return 64; }
    if (base_programs <= 32) { return 32; }
    if (base_programs <= 256) { return 8; }
    if (base_programs <= 512) { return 4; }
    return 1;
}

void require_weight(const Weight& weight, int n, int k, const char* label) {
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.n != n || weight.k != k) {
        throw std::invalid_argument(label);
    }
}

// Q/K/V/O projections may also be row-scaled FP8 (8-bit projection artifacts); linear dispatches.
void require_projection(const Weight& weight, int n, int k, const char* label) {
    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (weight.layout != QuantLayout::RowScale || weight.qdata == nullptr || weight.n != n ||
            weight.k != k) {
            throw std::invalid_argument(label);
        }
        return;
    }
    require_weight(weight, n, k, label);
}

} // namespace

void flash_next_expand_text_positions(const Tensor& positions, Tensor& mrope_positions,
                                      cudaStream_t stream) {
    const int width = positions.ne[0];
    const int batch = positions.ne[1];
    const int tokens = width * batch;
    if (tokens <= 0 || positions.dtype != DType::I32 || !positions.is_contiguous() ||
        positions.ne[2] != 1 || positions.ne[3] != 1 ||
        mrope_positions.dtype != DType::I32 || !mrope_positions.is_contiguous() ||
        mrope_positions.ne[0] != width || mrope_positions.ne[1] != batch ||
        mrope_positions.ne[2] != 3 || mrope_positions.ne[3] != 1) {
        throw std::invalid_argument("flash_next_expand_text_positions: invalid geometry");
    }
    expand_text_positions_kernel<<<(3 * tokens + 255) / 256, 256, 0, stream>>>(
        static_cast<const int*>(positions.data), static_cast<int*>(mrope_positions.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t flash_next_qsa_workspace_capacity_bytes(std::int32_t tokens,
                                                     std::uint32_t max_context) {
    if (tokens <= 0 || max_context == 0) {
        throw std::invalid_argument("Flash-Next QSA dimensions must be positive");
    }
    const std::uint64_t bf16 = static_cast<std::uint64_t>(tokens) * 39552ULL *
                               sizeof(__nv_bfloat16);
    const std::uint64_t groups = (static_cast<std::uint64_t>(max_context) + kRatio - 1) / kRatio;
    const std::uint64_t hierarchy = tokens <= 16
        ? 2 * ((groups + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups *
              (sizeof(float) + sizeof(std::int32_t))
        : 0;
    const std::uint64_t selection = static_cast<std::uint64_t>(tokens) *
                                    (groups * sizeof(float) +
                                     kTopGroups * sizeof(std::int32_t) + hierarchy);
    const std::uint64_t indices = static_cast<std::uint64_t>(tokens) * kOutputWidth *
                                  sizeof(std::int32_t);
    const std::uint64_t split_partials = tokens <= 16
        ? static_cast<std::uint64_t>(tokens) * kMaxDecodeAttentionSplits * kQueryHeads *
              (kHeadDim + 2) * sizeof(float)
        : 0;
    const std::uint64_t total = bf16 + indices + selection + split_partials + 16 * 256;
    if (total > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Flash-Next QSA workspace size overflow");
    }
    return static_cast<std::size_t>(total);
}

std::size_t flash_next_query_gate_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0) {
        throw std::invalid_argument("Flash-Next query/gate dimensions must be positive");
    }
    return static_cast<std::size_t>(tokens) * 2 * kQueryHeads * kHeadDim *
           sizeof(__nv_bfloat16) + 256;
}

void flash_next_project_query_gate(const Tensor& input, const Weight& query_gate,
                                   Tensor& query, Tensor& gate,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   Bf16GemmContext* bf16_gemm) {
    const int tokens = input.ne[1];
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != 2560 ||
        tokens <= 0 || query.dtype != DType::BF16 || !query.is_contiguous() ||
        query.numel() != static_cast<std::int64_t>(kQueryHeads) * kHeadDim * tokens ||
        gate.dtype != DType::BF16 || !gate.is_contiguous() || gate.numel() != query.numel()) {
        throw std::invalid_argument("flash_next_project_query_gate: invalid tensor geometry");
    }
    require_projection(query_gate, 12288, 2560,
                       "flash_next_project_query_gate: invalid packed weight");
    if (tokens == 1 && query_gate.qtype == QType::BF16) {
        detail::flash_next::launch_bf16_query_gate_decode(input, query_gate, query, gate, stream);
        return;
    }
    auto scope = workspace.scope();
    Tensor packed = workspace.alloc(DType::BF16, {12288, tokens});
    linear(input, query_gate, packed, stream, bf16_gemm);
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        4096, (query.numel() + 255) / 256));
    split_query_gate_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(gate.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_qsa(const Tensor& input, const Tensor& cache_positions,
                    const Tensor& rope_positions, const Tensor& valid_columns,
                    const Tensor& table_rows, const FlashNextQsaWeights& weights,
                    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                    Tensor& destination,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    Bf16GemmContext* bf16_gemm,
                    FlashNextQsaIndexControl index_control) {
    NINFER_PERF_SCOPE(index_control.reused_indices != nullptr ? "qsa.reuse" : "qsa.select",
                       input.ne[1], cache_positions.ne[1], envelope.max_visible_keys,
                       flash_next_work::qsa(input.ne[1], index_control.reused_indices != nullptr));

    const int width = cache_positions.ne[0];
    const int batch = cache_positions.ne[1];
    const int tokens = width * batch;
    if (input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != 2560 ||
        input.ne[1] != tokens || destination.dtype != DType::BF16 || !destination.is_contiguous() ||
        destination.ne[0] != 2560 || destination.ne[1] != tokens) {
        throw std::invalid_argument("flash_next_qsa: invalid input/output geometry");
    }
    if (cache_positions.dtype != DType::I32 || !cache_positions.is_contiguous() ||
        rope_positions.dtype != DType::I32 || !rope_positions.is_contiguous() ||
        rope_positions.ne[0] != width || rope_positions.ne[1] != batch ||
        rope_positions.ne[2] != 3 || valid_columns.dtype != DType::I32 ||
        valid_columns.ne[0] != batch || table_rows.dtype != DType::I32 ||
        table_rows.ne[0] != batch) {
        throw std::invalid_argument("flash_next_qsa: invalid position geometry");
    }
    const auto valid_indices = [tokens](const Tensor* tensor) {
        return tensor == nullptr ||
               (tensor->dtype == DType::I32 && tensor->is_contiguous() &&
                tensor->numel() == static_cast<std::int64_t>(kOutputWidth) * tokens);
    };
    if (!valid_indices(index_control.selected_indices) ||
        !valid_indices(index_control.reused_indices) ||
        (index_control.selected_indices != nullptr && index_control.reused_indices != nullptr)) {
        throw std::invalid_argument("flash_next_qsa: invalid index-sharing control");
    }
    const bool fp8_cache = cache.storage == KvCacheStorage::Fp8E4M3Row256;
    const bool int8_cache = cache.storage == KvCacheStorage::Int8Group64;
    const bool bf16_cache = cache.storage == KvCacheStorage::BFloat16KeyValue;
    const KvCode code = fp8_cache ? KvCode::Fp8 : int8_cache ? KvCode::Int8 : KvCode::Bf16;
    const auto scaled_codes = [&](DType dtype, int groups) {
        return cache.k_pages.dtype == dtype && cache.v_pages.dtype == dtype &&
               cache.k_scale_pages.dtype == DType::FP16 &&
               cache.v_scale_pages.dtype == DType::FP16 &&
               cache.k_scale_pages.ne[0] == groups && cache.v_scale_pages.ne[0] == groups;
    };
    const bool valid_primary =
        (bf16_cache && cache.k_pages.dtype == DType::BF16 &&
         cache.v_pages.dtype == DType::BF16 && cache.k_scale_pages.data == nullptr &&
         cache.v_scale_pages.data == nullptr) ||
        (fp8_cache && scaled_codes(DType::FP8_E4M3FN, 1)) ||
        (int8_cache && scaled_codes(DType::I8, kKVCacheInt8Groups));
    if (!valid_primary || cache.k_pages.ne[0] != kHeadDim ||
        cache.v_pages.ne[0] != kHeadDim ||
        cache.k_pages.ne[2] != kKvHeads || cache.v_pages.ne[2] != kKvHeads) {
        std::ostringstream message;
        message << "flash_next_qsa: invalid primary cache geometry: k dtype="
                << static_cast<int>(cache.k_pages.dtype) << " shape=[" << cache.k_pages.ne[0]
                << ',' << cache.k_pages.ne[1] << ',' << cache.k_pages.ne[2] << ','
                << cache.k_pages.ne[3] << "] v dtype="
                << static_cast<int>(cache.v_pages.dtype) << " shape=[" << cache.v_pages.ne[0]
                << ',' << cache.v_pages.ne[1] << ',' << cache.v_pages.ne[2] << ','
                << cache.v_pages.ne[3] << ']';
        throw std::invalid_argument(message.str());
    }
    if (cache.auxiliary_pages.size() != 2 ||
        cache.auxiliary_pages[0].dtype != DType::BF16 ||
        cache.auxiliary_pages[0].ne[0] != kIndexDim ||
        cache.auxiliary_pages[1].dtype != DType::I32 ||
        cache.auxiliary_pages[1].ne[0] != 3) {
        throw std::invalid_argument("flash_next_qsa: invalid auxiliary cache geometry");
    }
    require_projection(weights.query_gate, 12288, 2560,
                       "flash_next_qsa: invalid packed query/gate weight");
    require_projection(weights.key, 512, 2560, "flash_next_qsa: invalid key weight");
    require_projection(weights.value, 512, 2560, "flash_next_qsa: invalid value weight");
    require_projection(weights.output, 2560, 6144, "flash_next_qsa: invalid output weight");
    require_weight(weights.index_query, 512, 2560, "flash_next_qsa: invalid index query weight");
    require_weight(weights.index_key, 128, 2560, "flash_next_qsa: invalid index key weight");
    auto scope = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor gate = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor key = workspace.alloc(DType::BF16, {512, tokens});
    Tensor value = workspace.alloc(DType::BF16, {512, tokens});
    flash_next_project_query_gate(input, weights.query_gate, query, gate, workspace, stream,
                                  bf16_gemm);
    linear(input, weights.key, key, stream, bf16_gemm);
    linear(input, weights.value, value, stream, bf16_gemm);
    Tensor normalized_query = workspace.alloc(DType::BF16, {6144, tokens});
    Tensor normalized_key = workspace.alloc(DType::BF16, {512, tokens});
    Tensor query_heads = query.view({kHeadDim, kQueryHeads, tokens});
    Tensor key_heads = key.view({kHeadDim, kKvHeads, tokens});
    Tensor normalized_query_heads = normalized_query.view({kHeadDim, kQueryHeads, tokens});
    Tensor normalized_key_heads = normalized_key.view({kHeadDim, kKvHeads, tokens});
    Tensor rope_view = rope_positions.view({tokens, 3});
    rmsnorm(query_heads, weights.query_norm, 1.0e-6F, true, normalized_query_heads, stream);
    rmsnorm(key_heads, weights.key_norm, 1.0e-6F, true, normalized_key_heads, stream);
    int rope_device = 0;
    CUDA_CHECK(cudaGetDevice(&rope_device));
    int rope_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&rope_sms, cudaDevAttrMultiProcessorCount, rope_device));
    rope(rope_view, 64, kTheta, normalized_query_heads, normalized_key_heads,
         DeviceExecutionView{stream, rope_sms});
    if (code != KvCode::Bf16) {
        hadamard_rows_kernel<<<tokens * kQueryHeads, 32, 0, stream>>>(
            static_cast<__nv_bfloat16*>(normalized_query.data), tokens * kQueryHeads);
    }
    Tensor index_key = workspace.alloc(DType::BF16, {128, tokens});
    linear(input, weights.index_key, index_key, stream, bf16_gemm);
    Tensor index_query;
    if (index_control.reused_indices == nullptr) {
        Tensor index_query_projected = workspace.alloc(DType::BF16, {512, tokens});
        linear(input, weights.index_query, index_query_projected, stream, bf16_gemm);
        index_query = workspace.alloc(DType::BF16, {512, tokens});
        prepare_index_query_kernel<<<dim3(kIndexHeads, tokens), kIndexDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(index_query_projected.data),
            static_cast<const __nv_bfloat16*>(weights.index_query_norm.data),
            static_cast<const int*>(rope_view.data), tokens,
            static_cast<__nv_bfloat16*>(index_query.data));
    }
    const int logical_pages = cache.block_tables.ne[0];
    if (fp8_cache) {
        append_cache_fp8_kernel<<<tokens, 64, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(index_key.data),
            static_cast<const int*>(cache_positions.data), static_cast<const int*>(rope_view.data),
            static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
            width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data),
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
            static_cast<int*>(cache.auxiliary_pages[1].data));
    } else if (int8_cache) {
        append_cache_int8_kernel<<<tokens, 64, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(index_key.data),
            static_cast<const int*>(cache_positions.data), static_cast<const int*>(rope_view.data),
            static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
            width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data),
            static_cast<std::int8_t*>(cache.k_pages.data),
            static_cast<std::int8_t*>(cache.v_pages.data),
            static_cast<__half*>(cache.k_scale_pages.data),
            static_cast<__half*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
            static_cast<int*>(cache.auxiliary_pages[1].data));
    } else {
        append_cache_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_key.data),
            static_cast<const __nv_bfloat16*>(value.data),
            static_cast<const __nv_bfloat16*>(index_key.data),
            static_cast<const int*>(cache_positions.data), static_cast<const int*>(rope_view.data),
            static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
            width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data),
            static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__nv_bfloat16*>(cache.v_pages.data),
            static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
            static_cast<int*>(cache.auxiliary_pages[1].data));
    }
    compress_index_groups_kernel<<<tokens, kIndexDim, 0, stream>>>(
        static_cast<__nv_bfloat16*>(cache.auxiliary_pages[0].data),
        static_cast<const int*>(cache.auxiliary_pages[1].data),
        static_cast<const __nv_bfloat16*>(weights.index_key_norm.data),
        static_cast<const int*>(cache_positions.data),
        static_cast<const int*>(valid_columns.data), static_cast<const int*>(table_rows.data),
        width, batch, logical_pages, static_cast<const int*>(cache.block_tables.data));
    if (envelope.max_visible_keys == 0 ||
        envelope.max_visible_keys > static_cast<std::uint32_t>(logical_pages * kPagedKVPageSize)) {
        throw std::invalid_argument("flash_next_qsa: invalid visibility envelope");
    }
    Tensor attention = workspace.alloc(DType::BF16, {6144, tokens});
    const auto* key_scales   = static_cast<const __half*>(cache.k_scale_pages.data);
    const auto* value_scales = static_cast<const __half*>(cache.v_scale_pages.data);
    const auto launch_batched = [&]<KvCode Code>(std::integral_constant<KvCode, Code>, dim3 grid,
                                                 const int* indices, int num_splits,
                                                 float* partial_maximum,
                                                 float* partial_denominator,
                                                 float* partial_numerator,
                                                 __nv_bfloat16* output) {
        static const cudaError_t configured = cudaFuncSetAttribute(
            selected_attention_batched_kernel<Code>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSharedBytes);
        CUDA_CHECK(configured);
        selected_attention_batched_kernel<Code><<<grid, 128, kPrefillSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_query.data), cache.k_pages.data,
            cache.v_pages.data, key_scales, value_scales, indices,
            static_cast<const int*>(cache.block_tables.data), logical_pages,
            static_cast<const int*>(cache_positions.data),
            static_cast<const int*>(valid_columns.data),
            static_cast<const int*>(table_rows.data), width, tokens, num_splits,
            partial_maximum, partial_denominator, partial_numerator, output);
    };
    const auto dispatch_batched = [&](dim3 grid, const int* indices, int num_splits,
                                      float* partial_maximum, float* partial_denominator,
                                      float* partial_numerator, __nv_bfloat16* output) {
        switch (code) {
        case KvCode::Fp8:
            return launch_batched(std::integral_constant<KvCode, KvCode::Fp8>{}, grid, indices,
                                  num_splits, partial_maximum, partial_denominator,
                                  partial_numerator, output);
        case KvCode::Int8:
            return launch_batched(std::integral_constant<KvCode, KvCode::Int8>{}, grid, indices,
                                  num_splits, partial_maximum, partial_denominator,
                                  partial_numerator, output);
        case KvCode::Bf16:
            return launch_batched(std::integral_constant<KvCode, KvCode::Bf16>{}, grid, indices,
                                  num_splits, partial_maximum, partial_denominator,
                                  partial_numerator, output);
        }
    };
    const auto launch_split = [&]<KvCode Code>(std::integral_constant<KvCode, Code>,
                                               const int* indices, float* partial_maximum,
                                               float* partial_denominator,
                                               float* partial_numerator, int num_splits) {
        const dim3 grid(kQueryHeads / kSplitHeadsPerBlock, tokens, num_splits);
        selected_attention_split_kernel<Code><<<grid, kSplitHeadsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized_query.data), cache.k_pages.data,
            cache.v_pages.data, key_scales, value_scales, indices,
            static_cast<const int*>(cache.block_tables.data), logical_pages,
            static_cast<const int*>(cache_positions.data),
            static_cast<const int*>(valid_columns.data),
            static_cast<const int*>(table_rows.data), width, tokens,
            partial_maximum, partial_denominator, partial_numerator, num_splits);
    };
    const auto dispatch_split = [&](const int* indices, float* partial_maximum,
                                    float* partial_denominator, float* partial_numerator,
                                    int num_splits) {
        switch (code) {
        case KvCode::Fp8:
            return launch_split(std::integral_constant<KvCode, KvCode::Fp8>{}, indices,
                                partial_maximum, partial_denominator, partial_numerator,
                                num_splits);
        case KvCode::Int8:
            return launch_split(std::integral_constant<KvCode, KvCode::Int8>{}, indices,
                                partial_maximum, partial_denominator, partial_numerator,
                                num_splits);
        case KvCode::Bf16:
            return launch_split(std::integral_constant<KvCode, KvCode::Bf16>{}, indices,
                                partial_maximum, partial_denominator, partial_numerator,
                                num_splits);
        }
    };
    if (tokens > 16 && envelope.max_visible_keys <= kOutputWidth) {
        dispatch_batched(dim3(tokens, kKvHeads), nullptr, 1, nullptr, nullptr, nullptr,
                         static_cast<__nv_bfloat16*>(attention.data));
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Decode keeps a fixed launch extent across graph profiles.  Prompt
        // work uses only the causally reachable group frontier.
        Tensor indices;
        if (index_control.reused_indices != nullptr) {
            indices = index_control.reused_indices->view({kOutputWidth, tokens});
        } else if (index_control.selected_indices != nullptr) {
            indices = index_control.selected_indices->view({kOutputWidth, tokens});
        } else {
            indices = workspace.alloc(DType::I32, {kOutputWidth, tokens});
        }
        const int score_stride = tokens == 1
            ? std::max(1, logical_pages * kPagedKVPageSize / kRatio)
            : std::max(1, static_cast<int>(envelope.max_visible_keys / kRatio));
        const int group_extent = std::max(1, static_cast<int>(
            envelope.max_visible_keys / kRatio));
        bool hierarchical_selection = false;
        if (index_control.reused_indices == nullptr) {
            Tensor scores = workspace.alloc(DType::FP32, {score_stride, tokens});
            // The scalar warp scorer wins for short decode frontiers. Once the compressed frontier
            // is large, its FP32 dot products dominate token latency; the padded tensor-core scorer
            // amortizes its fixed staging cost and is substantially faster even for one query.
            const bool tensor_score = tokens > 16 || group_extent >= 8192;
            const int score_blocks =
                (group_extent + kScoreColumns * kScoreTilesPerBlock - 1) /
                (kScoreColumns * kScoreTilesPerBlock);
            if (tokens > 16 && (batch == 1 || width % kScoreTokens == 0)) {
                static const cudaError_t tokens_configured = cudaFuncSetAttribute(
                    score_groups_mma_tokens_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kScoreTokensSharedBytes);
                CUDA_CHECK(tokens_configured);
                score_groups_mma_tokens_kernel<<<
                    dim3(score_blocks, (tokens + kScoreTokens - 1) / kScoreTokens), 64,
                    kScoreTokensSharedBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(index_query.data),
                    static_cast<const __nv_bfloat16*>(cache.auxiliary_pages[0].data),
                    static_cast<const int*>(cache.block_tables.data), logical_pages,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data),
                    static_cast<const int*>(table_rows.data), width, tokens, group_extent,
                    score_stride, static_cast<float*>(scores.data));
            } else if (tensor_score) {
                static const cudaError_t score_configured = cudaFuncSetAttribute(
                    score_groups_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kScoreSharedBytes);
                CUDA_CHECK(score_configured);
                score_groups_mma_kernel<<<dim3(score_blocks, tokens), 64,
                                          kScoreSharedBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(index_query.data),
                    static_cast<const __nv_bfloat16*>(cache.auxiliary_pages[0].data),
                    static_cast<const int*>(cache.block_tables.data), logical_pages,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data),
                    static_cast<const int*>(table_rows.data), width, group_extent,
                    score_stride, static_cast<float*>(scores.data));
            } else {
                const int scalar_blocks = std::min(510, (group_extent + 7) / 8);
                score_groups_batched_kernel<<<dim3(scalar_blocks, tokens), 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(index_query.data),
                    static_cast<const __nv_bfloat16*>(cache.auxiliary_pages[0].data),
                    static_cast<const int*>(cache.block_tables.data), logical_pages,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data),
                    static_cast<const int*>(table_rows.data), width, tokens, group_extent,
                    score_stride, static_cast<float*>(scores.data));
            }
            CUDA_CHECK(cudaGetLastError());
            Tensor selected_groups = workspace.alloc(DType::I32, {kTopGroups, tokens});
            hierarchical_selection = tokens <= 16 && group_extent >= 8192;
            // Persistent-top-k collection order is part of the multi-token
            // verification reduction profile. Single-token decode has no MTP
            // acceptance dependency on it and emits final indices directly.
            const bool persistent_collection_order = tokens > 1 && tokens <= 16;
            if (hierarchical_selection) {
            const int candidate_capacity =
                ((group_extent + kTopkBlockItems - 1) / kTopkBlockItems) * kTopGroups;
            Tensor candidate_scores[2] = {
                workspace.alloc(DType::FP32, {candidate_capacity, tokens}),
                workspace.alloc(DType::FP32, {candidate_capacity, tokens}),
            };
            Tensor candidate_indices[2] = {
                workspace.alloc(DType::I32, {candidate_capacity, tokens}),
                workspace.alloc(DType::I32, {candidate_capacity, tokens}),
            };
            const float* current_scores = static_cast<const float*>(scores.data);
            const int* current_indices = nullptr;
            int current_count = group_extent;
            int current_stride = score_stride;
            int stage = 0;
            while (current_count > kTopGroups) {
                const int blocks = (current_count + kTopkBlockItems - 1) / kTopkBlockItems;
                const int output_count = blocks * kTopGroups;
                const bool final = blocks == 1;
                float* next_scores = final
                    ? static_cast<float*>(scores.data)
                    : static_cast<float*>(candidate_scores[stage & 1].data);
                int* next_indices = final
                    ? static_cast<int*>(persistent_collection_order
                                            ? selected_groups.data
                                            : indices.data)
                    : static_cast<int*>(candidate_indices[stage & 1].data);
                const int output_stride = output_count;
                hierarchical_top_groups_kernel<<<dim3(blocks, tokens), kTopkBlockThreads, 0,
                                                  stream>>>(
                    current_scores, current_indices, current_count, current_stride,
                    next_scores, next_indices, output_stride,
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data), width, stage,
                    final && !persistent_collection_order);
                if (final) { break; }
                current_scores = next_scores;
                current_indices = next_indices;
                current_count = output_count;
                current_stride = output_count;
                ++stage;
            }
            } else {
                select_top_groups_kernel<<<tokens, 256, 0, stream>>>(
                    static_cast<const float*>(scores.data),
                    static_cast<const int*>(cache_positions.data), score_stride, group_extent,
                    static_cast<int*>(selected_groups.data));
            }
            CUDA_CHECK(cudaGetLastError());
            if (!hierarchical_selection || persistent_collection_order) {
                if (persistent_collection_order) {
                    order_groups_like_persistent_topk_kernel<<<tokens, kTopkBlockThreads, 0,
                                                               stream>>>(
                        static_cast<int*>(selected_groups.data),
                        static_cast<const int*>(cache_positions.data), group_extent);
                }
                expand_indices_batched_kernel<<<dim3((kOutputWidth + 255) / 256, tokens), 256, 0,
                                                    stream>>>(
                    static_cast<const int*>(selected_groups.data),
                    static_cast<const int*>(cache_positions.data),
                    static_cast<const int*>(valid_columns.data), width,
                    static_cast<int*>(indices.data));
            }
        }
        if (tokens <= 16) {
            const bool dense_within_budget = envelope.max_visible_keys <= kOutputWidth;
            const int num_splits = dense_within_budget ? kMaxDecodeAttentionSplits
                                                       : decode_attention_splits(tokens);
            Tensor partial_maximum = workspace.alloc(
                DType::FP32, {kQueryHeads, tokens, num_splits});
            Tensor partial_denominator = workspace.alloc(
                DType::FP32, {kQueryHeads, tokens, num_splits});
            Tensor partial_numerator = workspace.alloc(
                DType::FP32, {kHeadDim, kQueryHeads, tokens, num_splits});
            if (dense_within_budget) {
                dispatch_split(static_cast<const int*>(indices.data),
                               static_cast<float*>(partial_maximum.data),
                               static_cast<float*>(partial_denominator.data),
                               static_cast<float*>(partial_numerator.data), num_splits);
            } else {
                dispatch_batched(dim3(tokens, kKvHeads, num_splits),
                                 static_cast<const int*>(indices.data), num_splits,
                                 static_cast<float*>(partial_maximum.data),
                                 static_cast<float*>(partial_denominator.data),
                                 static_cast<float*>(partial_numerator.data),
                                 static_cast<__nv_bfloat16*>(attention.data));
            }
            reduce_selected_attention_splits_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0,
                                                      stream>>>(
                static_cast<const float*>(partial_maximum.data),
                static_cast<const float*>(partial_denominator.data),
                static_cast<const float*>(partial_numerator.data),
                static_cast<const int*>(valid_columns.data), width, tokens,
                static_cast<__nv_bfloat16*>(attention.data), num_splits);
        } else {
            dispatch_batched(dim3(tokens, kKvHeads), static_cast<const int*>(indices.data), 1,
                             nullptr, nullptr, nullptr,
                             static_cast<__nv_bfloat16*>(attention.data));
        }
    }
    Tensor attention_heads = attention.view({kHeadDim, kQueryHeads, tokens});
    Tensor gate_heads = gate.view({kHeadDim, kQueryHeads, tokens});
    sigmoid_mul(gate_heads, attention_heads, stream);
    linear(attention, weights.output, destination, stream, bf16_gemm);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_qsa_select_indices(const Tensor& indices, const Tensor& selectors,
                                   Tensor& destination, cudaStream_t stream) {
    const int width = indices.ne[1];
    const int batch = indices.ne[2];
    if (width <= 0 || batch <= 0 || indices.dtype != DType::I32 || !indices.is_contiguous() ||
        indices.ne[0] != kOutputWidth || selectors.dtype != DType::I32 ||
        !selectors.is_contiguous() || selectors.ne[0] != batch ||
        destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.ne[0] != kOutputWidth || destination.ne[1] != batch) {
        throw std::invalid_argument("flash_next_qsa_select_indices: invalid geometry");
    }
    select_shared_indices_kernel<<<dim3((kOutputWidth + 255) / 256, batch), 256, 0, stream>>>(
        static_cast<const int*>(indices.data), static_cast<const int*>(selectors.data),
        static_cast<int*>(destination.data), width, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
