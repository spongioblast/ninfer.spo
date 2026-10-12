// The text layer loop: attention, GDN, and MLP mixers.

#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/execution/text_context.h"
#include "models/qwen3_8_flash_next/execution/diagnostics.h"
#include "models/qwen3_8_flash_next/execution/workspace_recipe.h"
#include "core/nvtx.h"
#include "core/performance.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/flash_next_gdn.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer/ops/flash_next_ple.h"
#include "ninfer/ops/flash_next_qsa.h"
#include "ninfer/ops/hyperconnection.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"
#include "models/qwen3_8_flash_next/execution/text_context_util.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule {

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(q_batch, k_batch, v_batch, position_batch, valid,
                                      kv_table_rows, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv},
                                      kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
                                      {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                      batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, ctx_.execution_view());

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor conv_states      = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, records.conv, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                query_output, key_output, value_output, gate_output, ph, work_, s);
        }
    } else {
        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, *w.conv1d, conv_state_in, conv_state_out, qc, kc, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_source_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    if constexpr (Variant::flash_next) {
        run_flash_next_layers(x, ph);
        return;
    }
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
            }
        }
    }
}

void TextContext::run_flash_next_layers(Tensor& x, Phase ph) {
    NINFER_PERF_SCOPE(ph == Phase::Prefill ? "ninfer.region/1|target.prefill"
                                         : "ninfer.region/1|target.verify");
    const int tokens = x.ne[1];
    const int batch  = active_sequence_batch_ == 0 ? 1 : active_sequence_batch_;
    const int width  = tokens / batch;
    if (tokens <= 0 || width * batch != tokens) {
        throw std::logic_error("Flash-Next schedule has invalid token geometry");
    }
    if (ph == Phase::Verify && (active_linear_state_source_slots_ == nullptr ||
                                (gdn_state_action_ != GdnStateAction::RecordForReplay &&
                                 active_linear_state_destination_slots_ == nullptr))) {
        throw std::logic_error("Flash-Next verify state selectors are unavailable");
    }
    cudaStream_t stream = ctx_.stream;
    auto schedule_scope = work_.scope();
    Tensor hyper        = work_.alloc(DType::BF16, {10240, tokens});
    Tensor block_input  = work_.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor block_output = work_.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor injection    = work_.alloc(DType::BF16, {4, tokens});
    ops::hyperconnection_repeat(x, hyper, stream);

    Tensor positions = active_cache_positions_->view({width, batch});
    Tensor rope_positions;
    if (active_rope_positions_->numel() == static_cast<std::int64_t>(tokens) * 3) {
        rope_positions = active_rope_positions_->view({width, batch, 3});
    } else {
        if (active_rope_positions_->numel() != tokens) {
            throw std::logic_error("Flash-Next schedule has invalid RoPE geometry");
        }
        rope_positions = work_.alloc(DType::I32, {width, batch, 3});
        ops::flash_next_expand_text_positions(active_rope_positions_->view({width, batch}),
                                              rope_positions, stream);
    }
    Tensor valid;
    if (active_valid_columns_ != nullptr) {
        valid = *active_valid_columns_;
    } else {
        valid = work_.alloc(DType::I32, {batch});
        // Captured schedules must not retain a pointer into a temporary host vector.
        ops::fill_i32(valid, width, stream);
    }
    const Tensor& table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;

    bool pending             = false;
    std::size_t full_index   = 0;
    std::size_t gdn_index    = 0;
    const auto execute_layer = [&](const auto& source) {
        if (source.has_ple) {
            if (pending) {
                ops::hyperconnection_combine(hyper, block_output, injection, stream);
                pending = false;
            }
            if (active_ple_embeddings_ == nullptr) {
                throw std::logic_error("Flash-Next PLE input was not prepared");
            }
            auto ple_scope    = work_.scope();
            Tensor ple_output = work_.alloc(DType::BF16, {10240, tokens});
            if (ple_state_ == nullptr) {
                throw std::logic_error("Flash-Next PLE state is unavailable");
            }
            Tensor gathered = active_ple_embeddings_->view({2560, tokens});
            if (ph == Phase::Verify) {
                if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                    if (ple_replay_records_ == nullptr) {
                        throw std::logic_error("Flash-Next PLE replay records are unavailable");
                    }
                    Tensor records = ple_replay_records_->slice(2, 0, batch).slice(1, 0, width);
                    ops::flash_next_ple_replay_record(hyper, gathered, source.ple, *ple_state_,
                                                      valid, *active_linear_state_source_slots_,
                                                      width, batch, records, ple_output, work_,
                                                      stream, bf16_gemm_);
                } else {
                    ops::flash_next_ple_batch_update(hyper, gathered, source.ple, *ple_state_,
                                                     valid, *active_linear_state_source_slots_,
                                                     *active_linear_state_destination_slots_, width,
                                                     batch, ple_output, work_, stream, bf16_gemm_);
                }
            } else {
                Tensor ple_state =
                    ple_state_->slice(2, linear_state_destination_slot_, 1).view({10240, 9});
                ops::flash_next_ple(hyper, gathered, source.ple, ple_state, ple_output, work_,
                                    stream, bf16_gemm_);
            }
            ops::residual_add(ple_output, hyper, stream);
        }

        const ops::HyperConnectionWeights& attention_hc = source.attention_hc;
        // Steering and capture ride the fused attention-input mix in every phase. Their device
        // controls are fixed addresses, so graphs need no rebuild; inactive rows run the plain mix.
        const bool capturing = io_.activation_capture != nullptr && active_activation_ids_ != nullptr;
        const ops::HyperConnectionActivation activation{
            .steering            = io_.activation,
            .layer               = static_cast<int>(full_index + gdn_index),
            .width               = width,
            .capture             = capturing ? io_.activation_capture : nullptr,
            .positions           = &positions,
            .ids                 = active_activation_ids_,
            .valid               = &valid,
            .speculative_columns = ph == Phase::Verify && width > 1};
        const auto* hooks =
            io_.activation != nullptr || capturing ? &activation : nullptr;
        if (pending) {
            ops::hyperconnection_combine_mix(hyper, block_output, injection, attention_hc,
                                             block_input, &injection, work_, stream, nullptr, hooks);
        } else {
            ops::hyperconnection_mix(hyper, attention_hc, block_input, &injection, work_, stream,
                                     nullptr, hooks);
        }
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(source)>,
                                     typename LoadedModelData::FullLayer>) {
            const auto cache =
                active_sequence_batch_ == 0
                    ? single_row_paged_kv_batch_view(
                          kv_.layer_view(static_cast<std::uint32_t>(full_index)))
                    : batch_text_kv_->batch_layer_view(static_cast<std::uint32_t>(full_index));
            ops::flash_next_qsa(block_input, positions, rope_positions, valid, table_rows,
                                source.projection, cache, *active_causal_attention_envelope_,
                                block_output, work_, stream, bf16_gemm_);
            ++full_index;
        } else {
            if (ph == Phase::Verify) {
                auto layer_state = state_.layer_view(static_cast<std::uint32_t>(gdn_index));
                if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                    if (replay_records_ == nullptr) {
                        throw std::logic_error("Flash-Next GDN replay records are unavailable");
                    }
                    ops::flash_next_gdn_replay_record(
                        block_input, source.projection, layer_state.conv, layer_state.recurrent,
                        valid, *active_linear_state_source_slots_,
                        replay_records_->layer(static_cast<std::int32_t>(gdn_index), batch),
                        block_output, work_, stream);
                } else {
                    if (width != 1) {
                        throw std::logic_error(
                            "Flash-Next direct multi-token state update is unavailable");
                    }
                    ops::flash_next_gdn_batch_update(
                        block_input, source.projection, layer_state.conv, layer_state.recurrent,
                        *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                        block_output, work_, stream);
                }
            } else {
                Tensor conv_in       = state_.conv_slot(static_cast<std::uint32_t>(gdn_index),
                                                        linear_state_source_slot_);
                Tensor conv_out      = state_.conv_slot(static_cast<std::uint32_t>(gdn_index),
                                                        linear_state_destination_slot_);
                Tensor recurrent_in  = state_.recurrent_slot(static_cast<std::uint32_t>(gdn_index),
                                                             linear_state_source_slot_);
                Tensor recurrent_out = state_.recurrent_slot(static_cast<std::uint32_t>(gdn_index),
                                                             linear_state_destination_slot_);
                ops::flash_next_gdn(block_input, source.projection, conv_in, conv_out, recurrent_in,
                                    recurrent_out, block_output, work_, stream, bf16_gemm_);
            }
            ++gdn_index;
        }

        ops::hyperconnection_combine_mix(hyper, block_output, injection, source.mlp_hc, block_input,
                                         &injection, work_, stream);
        ops::flash_next_moe(block_input, source.post_mixer, block_output, work_, stream, bf16_gemm_);
        pending = true;
    };
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            execute_layer(weights_.full_layers.at(full_index));
        } else {
            execute_layer(weights_.gdn_layers.at(gdn_index));
        }
    }
    if (pending) { ops::hyperconnection_combine(hyper, block_output, injection, stream); }
    if (flash_predictor_hidden_output_ != nullptr) {
        Tensor output = flash_predictor_hidden_output_->view({10240, tokens});
        CUDA_CHECK(cudaMemcpyAsync(output.data, hyper.data, hyper.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
    } else if (mtp_prefill_hidden_ != nullptr && ph == Phase::Prefill) {
        Tensor output = matrix_window(*mtp_prefill_hidden_, tokens);
        CUDA_CHECK(cudaMemcpyAsync(output.data, hyper.data, hyper.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
    }
    const ops::HyperConnectionWeights final_weights{
        .norm = weights_.final_hc.norm,
        .down = weights_.final_hc.down,
        .up   = weights_.final_hc.up,
    };
    ops::hyperconnection_mix(hyper, final_weights, x, nullptr, work_, stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule
