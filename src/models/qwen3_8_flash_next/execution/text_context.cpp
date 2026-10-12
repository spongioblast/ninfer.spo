#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/execution/text_context.h"
#include "models/qwen3_8_flash_next/execution/diagnostics.h"
#include "models/qwen3_8_flash_next/execution/prefill_graph.h"
#include "runtime/engine/prefill_graph_stats.h"
#include "models/qwen3_8_flash_next/execution/workspace_recipe.h"
#include "core/nvtx.h"
#include "core/performance.h"
#include "program/execution/visual_scatter.h"
#include "models/qwen3_8_flash_next/execution/vision_context.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "models/qwen3_8_flash_next/impl/ple_table.h"
#include "models/qwen3_8_flash_next/execution/text_context_util.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule {

namespace {

// NINFER_DEBUG_LOGITS_DIR only: the MTP head reuses io_.logits after the target sample, so the
// deferred prefill dump reads the target row from this copy. A captured prefill graph keeps the
// address, so the buffer lives for the process instead of one TextContext; it is never freed.
const DeviceBuffer& prefill_logits_dump(std::size_t bytes) {
    static DeviceBuffer* buffer = new DeviceBuffer(bytes);
    if (buffer->bytes < bytes) {
        throw std::logic_error("prefill logits dump buffer is smaller than the target logits");
    }
    return *buffer;
}

} // namespace

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_8_flash_next::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_8_flash_next::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_8_flash_next::PagedKVCacheView mtp_kv,
                         const qwen3_8_flash_next::PagedKVCache* batch_text_kv,
                         const qwen3_8_flash_next::PagedKVCache* batch_mtp_kv, Tensor* ple_state)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv), ple_state_(ple_state) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                                  &source.input_projection,
                                  &source.embedding_norm,
                                  &source.hidden_norm,
                                  &source.input_norm,
                                  &source.query_norm,
                                  &source.key_norm,
                                  &source.output,
                                  &source.post_attention_norm,
                                  &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    std::vector<qwen3_8_flash_next::PleIds> ple_ids;
    [[maybe_unused]] std::size_t ple_begin = 0;
    if constexpr (Variant::flash_next) {
        const std::span<const int> complete_ids = multimodal != nullptr ? multimodal->token_ids
                                                  : text_prefill != nullptr
                                                      ? text_prefill->token_ids
                                                      : ids;
        const std::size_t extent                = static_cast<std::size_t>(base) + ids.size();
        if (complete_ids.size() < extent) {
            throw std::logic_error("Flash-Next PLE history is incomplete");
        }
        ple_begin = qwen3_8_flash_next::ple_history_begin(static_cast<std::size_t>(base));
        std::vector<std::int32_t> represented(extent - ple_begin);
        std::copy_n(complete_ids.data() + ple_begin, represented.size(), represented.data());
        ple_ids.resize(represented.size());
        qwen3_8_flash_next::compute_ple_ids(represented, ple_ids);
    }
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    Tensor dump_ids;
    Tensor dump_positions;
    Tensor dump_logits;
    bool dump_prefill_logits = false;
    const Tensor target_logits = matrix_window(io_.logits, 1);
    const DeviceBuffer* dump_buffer =
        runtime_switches().logits_dir.empty() ? nullptr : &prefill_logits_dump(target_logits.bytes());
    const Tensor logits_dump_copy =
        dump_buffer == nullptr ? Tensor{}
                               : Tensor(dump_buffer->p, target_logits.dtype,
                                        {target_logits.ne[0], target_logits.ne[1]});
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        const auto& graph_settings = detail::prefill_graph_settings();
        const bool small_text = multimodal == nullptr && !graph_settings.state_dir &&
                                graph_settings.max_tokens > 0 && len <= graph_settings.max_tokens &&
                                gdn_state_action_ == GdnStateAction::UpdateInPlace;
        const bool use_ingress = small_text;
        std::uint32_t envelope_tokens = static_cast<std::uint32_t>(base_i + t0 + len);
        if (use_ingress && is_last && mtp_proposal_extent_ != 0) {
            envelope_tokens = static_cast<std::uint32_t>(base_i + T +
                                                         static_cast<int>(mtp_proposal_extent_));
        }
        if (use_ingress) {
            envelope_tokens = detail::prefill_graph_envelope(
                envelope_tokens, kv_.valid() ? kv_.max_context() : 0);
        }
        detail::PrefillGraphKey graph_key;
        detail::PrefillGraphAction graph_action = detail::PrefillGraphAction::Skip;
        detail::PrefillIngress ingress;
        std::size_t ple_upload_bytes = 0;
        if (use_ingress) {
            graph_key.tokens = len;
            graph_key.envelope = envelope_tokens;
            graph_key.autoregressive_steps =
                is_last ? mtp_proposal_extent_ : 0U;
            graph_key.source_slot = linear_state_source_slot_;
            graph_key.destination_slot = linear_state_destination_slot_;
            graph_key.final_chunk = is_last;
            graph_key.ends_at_split =
                split_rel > 0 && t0 + len == split_rel && rewrite_checkpoint_hidden_output_ != nullptr;
            graph_key.mtp = prepare_mtp_prompt;
            graph_key.rope_adjusted = rope_delta_ != 0;
            graph_key.sampled = is_last && sampling_config_ != nullptr;
            graph_key.activation =
                io_.activation != nullptr || io_.activation_capture != nullptr;
            graph_key.checkpoint_hidden =
                graph_key.ends_at_split ? rewrite_checkpoint_hidden_output_->data : nullptr;
            graph_key.checkpoint_mtp_hidden =
                graph_key.ends_at_split && rewrite_checkpoint_mtp_hidden_output_ != nullptr
                    ? rewrite_checkpoint_mtp_hidden_output_->data
                    : nullptr;
            graph_key.sampling = sampling_config_;
            graph_key.text_kv_table = kv_.valid() ? kv_.layer_view(0).block_table.data : nullptr;
            graph_key.mtp_kv_table =
                mtp_kv_.valid() ? mtp_kv_.layer_view(0).block_table.data : nullptr;
            if constexpr (Variant::flash_next) {
                ple_upload_bytes = static_cast<std::size_t>(len) * 2560U *
                                   dtype_size(weights_.ple_table_dtype);
            }
            ingress = detail::prefill_ingress(len, ple_upload_bytes);
            *ingress.position_base = base_i + t0;
            *ingress.sample_position = base_i + T;
            std::copy_n(ids.data() + t0, len, ingress.ids);
            if constexpr (Variant::flash_next) {
                if (prepare_mtp_prompt) {
                    const std::uint32_t alignment_tokens =
                        text_prefill != nullptr ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                                                : static_cast<std::uint32_t>(T);
                    const std::uint32_t alignment_begin =
                        text_prefill != nullptr ? prompt_t0 : static_cast<std::uint32_t>(t0);
                    const ninfer::program::MtpAlignmentWindow mtp_window =
                        ninfer::program::plan_mtp_alignment_window(
                            alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                    const std::span<const int> alignment_ids =
                        text_prefill != nullptr ? text_prefill->token_ids : ids;
                    graph_key.mtp_generated_column = mtp_window.final_column_uses_generated_token;
                    const int prompt_columns =
                        len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                    if (prompt_columns > 0) {
                        std::copy_n(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                                    prompt_columns, ingress.mtp_ids);
                    }
                }
                qwen3_8_flash_next::gather_ple(
                    *weights_.ple_table, weights_.ple_table_dtype,
                    std::span<const qwen3_8_flash_next::PleIds>(ple_ids).subspan(
                        prompt_t0 - ple_begin, static_cast<std::size_t>(len)),
                    std::span<std::byte>(static_cast<std::byte*>(ingress.ple_host), ple_upload_bytes),
                    ple_gather_workers_);
            }
            graph_action = detail::select_prefill_graph(graph_key);
        } else if (multimodal != nullptr) {
            ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::Ineligible,
                                                static_cast<std::uint32_t>(detail::PrefillGraphSkip::Multimodal));
        } else if (graph_settings.state_dir) {
            ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::Ineligible,
                                                static_cast<std::uint32_t>(detail::PrefillGraphSkip::StateDir));
        } else if (gdn_state_action_ != GdnStateAction::UpdateInPlace) {
            ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::Ineligible,
                                                static_cast<std::uint32_t>(detail::PrefillGraphSkip::StateAction));
        } else if (!graph_settings.graphs || len > graph_settings.max_tokens) {
            ninfer::runtime::note_prefill_graph(
                ninfer::runtime::PrefillGraphNote::Ineligible,
                static_cast<std::uint32_t>(graph_settings.graphs ? detail::PrefillGraphSkip::TooLong
                                                                : detail::PrefillGraphSkip::GraphsOff));
        }

        auto run_pass = [&]() {
            if (use_ingress) { detail::prefill_ingress_upload(ple_upload_bytes, s); }
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = use_ingress ? Tensor(ingress.ids_device, DType::I32, {len}) : roots.ids;
            if (!use_ingress) { copy_i32(ids.data() + t0, ids_device, s); }

            Tensor positions = use_ingress ? Tensor(ingress.positions_device, DType::I32, {len})
                                           : roots.positions;
            if (use_ingress) {
                ops::fill_i32_positions_from_device(
                    positions, Tensor(ingress.position_base_device, DType::I32, {1}), s);
            } else {
                ops::fill_i32_positions(positions, base_i + t0, s);
            }

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_ids(active_activation_ids_, ids_device);
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = use_ingress ? envelope_tokens
                                             : static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            Tensor gathered_device = use_ingress
                                         ? Tensor(ingress.ple_device, weights_.ple_table_dtype, {2560, len})
                                         : Tensor();
            std::vector<std::byte> gathered;
            if (!use_ingress) {
                const auto ple_element_bytes = dtype_size(weights_.ple_table_dtype);
                gathered.resize(static_cast<std::size_t>(len) * 2560U * ple_element_bytes);
                qwen3_8_flash_next::gather_ple(
                    *weights_.ple_table, weights_.ple_table_dtype,
                    std::span<const qwen3_8_flash_next::PleIds>(ple_ids).subspan(
                        prompt_t0 - ple_begin, len),
                    gathered, ple_gather_workers_);
                gathered_device = work_.alloc(weights_.ple_table_dtype, {2560, len});
                CUDA_CHECK(cudaMemcpyAsync(gathered_device.data, gathered.data(), gathered.size(),
                                           cudaMemcpyHostToDevice, s));
            }
            ScopedPositions scoped_ple(active_ple_embeddings_, gathered_device);
            run_layers(x, Phase::Prefill);

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            if constexpr (Variant::flash_next) {
                CUDA_CHECK(
                    cudaMemcpyAsync(xf.data, x.data, x.bytes(), cudaMemcpyDeviceToDevice, s));
            } else {
                ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
            }

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s, bf16_gemm_);
                if (use_ingress) {
                    dump_ids = ids_device.slice(0, len - 1, 1);
                    dump_positions = positions.slice(0, len - 1, 1);
                    dump_logits = logits;
                    if (logits_dump_copy.data != nullptr) {
                        CUDA_CHECK(cudaMemcpyAsync(logits_dump_copy.data, logits.data,
                                                   logits.bytes(), cudaMemcpyDeviceToDevice, s));
                        dump_logits = logits_dump_copy;
                    }
                    dump_prefill_logits = true;
                } else {
                    qwen3_8_flash_next::detail::capture_target_logits(
                        "prefill", ids_device.slice(0, len - 1, 1), positions.slice(0, len - 1, 1),
                        nullptr, io_.text_kv_table_row, logits, s);
                }
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                if (use_ingress) {
                    const Tensor sample_position(ingress.sample_position_device, DType::I32, {1});
                    ops::assign_i32_scalar(sample_position, io_.pos, s);
                    // rope_delta was written to its device scalar before this pass. Adding it
                    // here keeps the position out of the captured launch arguments.
                    ops::add_i32_scalars(sample_position, io_.rope_delta, io_.rope_pos, s);
                } else {
                    ops::set_i32_scalar(io_.pos, base_i + T, s);
                    ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                }
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const ninfer::program::MtpAlignmentWindow mtp_window =
                    ninfer::program::plan_mtp_alignment_window(alignment_tokens, alignment_begin,
                                                                  static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = use_ingress ? Tensor(ingress.mtp_ids_device, DType::I32, {len})
                                             : work_.alloc(DType::I32, {len});
                if (!use_ingress && prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const ninfer::program::MtpVisualOverlap overlap =
                            ninfer::program::shifted_visual_overlap(
                                vision_chunk.control->scatter_indices, alignment_tokens,
                                mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            ninfer::program::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits     = matrix_window(io_.logits, 1);
                    Tensor draft0     = io_.mtp->draft_tokens.slice(0, 0, 1);
                    Tensor mtp_source = xf;
                    if constexpr (Variant::flash_next) {
                        if (mtp_prefill_hidden_ == nullptr) {
                            throw std::logic_error(
                                "Flash-Next prefill predictor hidden is unavailable");
                        }
                        mtp_source = matrix_window(*mtp_prefill_hidden_, len);
                    }
                    mtp_prefill_chunk(mtp_ids, mtp_source, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    if (use_ingress) {
                        ops::assign_i32_scalar(
                            Tensor(ingress.sample_position_device, DType::I32, {1}), ar_position, s);
                    } else {
                        ops::set_i32_scalar(ar_position, base_i + T, s);
                    }
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = use_ingress
                                                    ? envelope_tokens
                                                    : static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        if constexpr (Variant::flash_next) {
                            Tensor next_predictor = work_.alloc(DType::BF16, {4 * kCfg.hidden, 1});
                            mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                                ar_envelope, next_hidden, logits, next_token,
                                                &next_predictor);
                            CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_predictor.data,
                                                       next_predictor.bytes(),
                                                       cudaMemcpyDeviceToDevice, s));
                        } else {
                            mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                                ar_envelope, next_hidden, logits, next_token);
                            CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                       io_.mtp->ar_hidden.bytes(),
                                                       cudaMemcpyDeviceToDevice, s));
                        }
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    Tensor mtp_source = xf;
                    if constexpr (Variant::flash_next) {
                        if (mtp_prefill_hidden_ == nullptr) {
                            throw std::logic_error(
                                "Flash-Next prefill predictor hidden is unavailable");
                        }
                        mtp_source = matrix_window(*mtp_prefill_hidden_, len);
                    }
                    mtp_prefill_chunk(mtp_ids, mtp_source, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
                if constexpr (Variant::flash_next) {
                    if (rewrite_checkpoint_mtp_hidden_output_ != nullptr &&
                        mtp_prefill_hidden_ == nullptr) {
                        throw std::logic_error(
                            "Flash-Next rewrite checkpoint has no predictor hidden output");
                    }
                    if (rewrite_checkpoint_mtp_hidden_output_ != nullptr) {
                        require_tensor_shape(*rewrite_checkpoint_mtp_hidden_output_, DType::BF16,
                                             {4 * kCfg.hidden, 1},
                                             "rewrite checkpoint MTP hidden output");
                        const Tensor checkpoint_mtp_hidden =
                            mtp_prefill_hidden_->slice(1, len - 1, 1);
                        CUDA_CHECK(cudaMemcpyAsync(
                            rewrite_checkpoint_mtp_hidden_output_->data, checkpoint_mtp_hidden.data,
                            checkpoint_mtp_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
                    }
                }
            }
        };

        if (graph_action == detail::PrefillGraphAction::Launch && graph_settings.audit &&
            !detail::audit_prefill_graph(graph_key, s, run_pass)) {
            run_pass();
        } else if (graph_action == detail::PrefillGraphAction::Launch) {
            if (is_last) {
                dump_ids = Tensor(ingress.ids_device, DType::I32, {len}).slice(0, len - 1, 1);
                dump_positions =
                    Tensor(ingress.positions_device, DType::I32, {len}).slice(0, len - 1, 1);
                dump_logits = logits_dump_copy.data != nullptr ? logits_dump_copy : target_logits;
                dump_prefill_logits = true;
            }
            detail::launch_prefill_graph(graph_key, s);
            ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::Hit);
        } else if (graph_action == detail::PrefillGraphAction::Capture) {
            try {
                detail::capture_prefill_graph(graph_key, s, run_pass);
                detail::launch_prefill_graph(graph_key, s);
                ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::Capture);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "prefill graph capture failed: %s\n", error.what());
                detail::forget_prefill_graph(graph_key);
                run_pass();
            }
        } else {
            run_pass();
            if (graph_action == detail::PrefillGraphAction::EagerFirst) {
                ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::EagerFirst);
            }
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    if (dump_prefill_logits) {
        qwen3_8_flash_next::detail::capture_target_logits(
            "prefill", dump_ids, dump_positions, nullptr, io_.text_kv_table_row, dump_logits,
            ctx_.stream);
    }
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_8_flash_next::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule
