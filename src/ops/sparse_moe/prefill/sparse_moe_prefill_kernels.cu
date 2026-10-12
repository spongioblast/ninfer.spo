#include "core/weight.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = 257;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;


#include "ops/sparse_moe/prefill/sparse_moe_prefill_routing.cuh"
#include "ops/sparse_moe/prefill/sparse_moe_prefill_gemm.cuh"
#include "ops/sparse_moe/prefill/sparse_moe_prefill_combine.cuh"


} // namespace

void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace,
                               DeviceExecutionView execution) {
    if (x.ne[1] != plan.tokens || destination.ne[1] != plan.tokens || plan.slice_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: launch plan does not match tensors");
    }
    const cudaStream_t stream = execution.stream;
    const std::int64_t max_prefill_blocks =
        static_cast<std::int64_t>(execution.multiprocessor_count) * kPrefillQueuedCtasPerSm;

    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* routed_gate_codes = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_gate_scales =
        static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* routed_down_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_down_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_down_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_gate_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_gate_scales =
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    const auto* shared_down_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_down_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);

    auto* ids               = static_cast<int*>(workspace.token_ids.data);
    auto* alpha             = static_cast<float*>(workspace.token_alpha.data);
    auto* local_rank        = static_cast<int*>(workspace.local_rank.data);
    auto* packed_index      = static_cast<int*>(workspace.packed_index.data);
    auto* shared_scale      = static_cast<float*>(workspace.shared_scale.data);
    auto* tile_counts       = static_cast<int*>(workspace.tile_counts.data);
    auto* tile_bases        = static_cast<int*>(workspace.tile_bases.data);
    auto* offsets           = static_cast<int*>(workspace.expert_offsets.data);
    auto* route_job_experts = static_cast<int*>(workspace.route_job_experts.data);
    auto* route_job_columns = static_cast<int*>(workspace.route_job_columns.data);
    auto* route_job_count   = static_cast<int*>(workspace.route_job_count.data);
    auto* scores            = static_cast<float*>(workspace.score_storage.data);
    auto* shared_activation = static_cast<__nv_bfloat16*>(workspace.shared_activation.data);
    auto* grouped_io        = static_cast<__nv_bfloat16*>(workspace.grouped_io.data);
    auto* packed_token      = static_cast<int*>(workspace.packed_token.data);
    auto* routed_activation = static_cast<__nv_bfloat16*>(workspace.routed_storage.data);
    auto* routed_sum        = static_cast<float*>(workspace.routed_sum.data);

    for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
        const std::int32_t tokens =
            std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
        const Tensor input_slice = x.slice(1, token0, tokens);
        Tensor output_slice      = destination.slice(1, token0, tokens);
        const auto* input        = static_cast<const __nv_bfloat16*>(input_slice.data);
        auto* output             = static_cast<__nv_bfloat16*>(output_slice.data);
        const int route_tiles =
            (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;
        const int assignments   = tokens * kTopK;
        const int adaptive_last = weights.routed_down.qtype == QType::Q5_G64_FP16   ? 51
                                  : weights.routed_down.qtype == QType::Q6_G64_FP16 ? 52
                                                                                    : 0;
        const bool adaptive     = tokens >= 47 && tokens <= adaptive_last;

        sparse_moe_prefill_router_mma_kernel<<<dim3((kRouterRows + kRouterBM - 1) / kRouterBM,
                                                    (tokens + kRouterBN - 1) / kRouterBN),
                                               kRouterThreads, 0, stream>>>(input, router, scores,
                                                                            tokens);
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_select_count_kernel<<<route_tiles, kRouterThreads, 0, stream>>>(
            scores, ids, alpha, shared_scale, local_rank, tile_counts, tokens);
        CUDA_CHECK(cudaGetLastError());

        const bool wide_plan   = tokens >= kSparseMoePrefillWideMin;
        const int route_job_bn = wide_plan ? 64 : 32;
        // The scan emits one route job per nonempty column tile of an expert, so it cannot
        // emit more than one job per full tile of assignments plus one tail per expert --
        // the same bound the workspace is sized by. Each job expands into row blocks, and
        // sizing the grid from that product keeps a persistent block on one work item while
        // there are fewer work items than the cap, instead of a fixed count that has to
        // iterate. The exact job count only exists on the device.
        const int max_route_jobs   = assignments / route_job_bn + kExperts;
        const int routed_gate_work = max_route_jobs * (kIntermediate / (kExpertBM / 2));
        const int routed_down_work = max_route_jobs * (kHidden / kExpertBM);
        const int routed_gate_blocks =
            static_cast<int>(std::min<std::int64_t>(routed_gate_work, max_prefill_blocks));
        const int routed_down_blocks =
            static_cast<int>(std::min<std::int64_t>(routed_down_work, max_prefill_blocks));
        sparse_moe_prefill_scan_kernel<<<1, kExpertThreads, 0, stream>>>(
            tile_counts, tile_bases, offsets, route_job_experts, route_job_columns, route_job_count,
            route_tiles, route_job_bn, tokens, adaptive);
        CUDA_CHECK(cudaGetLastError());

        const bool routed_gate_up_q4 = weights.routed_gate_up.qtype == QType::Q4_G64_FP16;
        const int index_blocks       = (assignments + kExpertThreads - 1) / kExpertThreads;
        if (adaptive) {
            auto* adaptive_activations = reinterpret_cast<float*>(grouped_io);
            sparse_moe_decode_launch_d3_small_t(input_slice, weights, ids, adaptive_activations,
                                                tokens, SparseMoeSmallTD3Schedule::Paths3, stream,
                                                route_job_count);
            sparse_moe_decode_launch_d4_small_t(
                weights, output_slice, ids, alpha, shared_scale, adaptive_activations, tokens,
                SparseMoeSmallTD4Schedule::Rows4, stream, route_job_count);
            // adaptive implies a Q5/Q6 routed down, and prefill_min_tokens admits those only
            // with a Q4 routed gate/up, so this path is always the indexed one.
            sparse_moe_prefill_index_kernel<true><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments,
                route_job_count);
        } else if (routed_gate_up_q4) {
            sparse_moe_prefill_index_kernel<false><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments, nullptr);
        } else {
            sparse_moe_prefill_gather_kernel<false><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 routed_gate_grid(kIntermediate / (kExpertBM / 2), kExperts);
        // Same predicate that decided whether packed_token was written above.
        if (routed_gate_up_q4) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_kernel<8, 64>
                    <<<routed_gate_blocks, 8 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_kernel<4, 32, kExpertStages, GateUpRoute::Packed>
                    <<<routed_gate_blocks, 4 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
                sparse_moe_prefill_q4_gate_up_kernel<4, 32, kGateUpNarrowStages,
                                                     GateUpRoute::Spread>
                    <<<routed_gate_blocks, 4 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            }
        } else if (weights.routed_gate_up.qtype == QType::Q8_G32_FP16) {
            sparse_moe_prefill_q8_gate_up_kernel<true>
                <<<routed_gate_grid, kExpertThreads, 0, stream>>>(
                    grouped_io, offsets, routed_gate_codes, routed_gate_scales, routed_activation,
                    tokens, nullptr);
        } else {
            throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_gate_grid(kIntermediate / (kExpertBM / 2),
                                    (tokens + kExpertBN - 1) / kExpertBN);
        if (adaptive) {
            sparse_moe_prefill_q8_gate_up_kernel<false, true>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, route_job_count);
        } else {
            sparse_moe_prefill_q8_gate_up_kernel<false, false>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 routed_down_grid(kHidden / kExpertBM, kExperts);
        switch (weights.routed_down.qtype) {
        case QType::Q5_G64_FP16:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>
                    <<<routed_down_blocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>
                    <<<routed_down_blocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q6_G64_FP16:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 64>
                    <<<routed_down_blocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>
                    <<<routed_down_blocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q8_G32_FP16:
            sparse_moe_prefill_q8_down_kernel<true>
                <<<routed_down_grid, kExpertThreads, 0, stream>>>(
                    routed_activation, offsets, routed_down_codes, routed_down_scales, grouped_io,
                    nullptr, nullptr, nullptr, tokens, nullptr);
            break;
        default:
            throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
        }
        CUDA_CHECK(cudaGetLastError());

        if (adaptive) {
            sparse_moe_prefill_reduce_kernel<true><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, route_job_count);
        } else {
            sparse_moe_prefill_reduce_kernel<false><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_down_grid(kHidden / kExpertBM, (tokens + kExpertBN - 1) / kExpertBN);
        if (adaptive) {
            sparse_moe_prefill_q8_down_kernel<false, true>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, route_job_count);
        } else {
            sparse_moe_prefill_q8_down_kernel<false, false>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
