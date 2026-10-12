// MTP draft passes: the stem, the tail, prefill chunks, decode batches, and proposals.

#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/execution/text_context.h"
#include "models/qwen3_8_flash_next/execution/diagnostics.h"
#include "models/qwen3_8_flash_next/execution/workspace_recipe.h"
#include "core/nvtx.h"
#include "core/performance.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer/ops/flash_next_qsa.h"
#include "ninfer/ops/hyperconnection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "models/qwen3_8_flash_next/execution/text_context_util.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule {

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s, bf16_gemm_);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
                                      {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                      batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s, bf16_gemm_);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings,
                                   Tensor* predictor_hidden, Tensor* selected_qsa_indices,
                                   const Tensor* reused_qsa_indices) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP requires predictor hidden output");
    }
    mtp_forward_flash_next(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                           input_embeddings, *predictor_hidden, selected_qsa_indices,
                           reused_qsa_indices);
    return;
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_forward_flash_next(const Tensor& ids, const Tensor& hidden,
                                         const Tensor& positions, const Tensor& rope_positions,
                                         ops::CausalAttentionExecutionEnvelope envelope,
                                         Tensor& sample_hidden, const Tensor* input_embeddings,
                                         Tensor& predictor_hidden, Tensor* selected_qsa_indices,
                                         const Tensor* reused_qsa_indices) {
    NINFER_PERF_SCOPE("ninfer.region/1|predictor");
    cudaStream_t stream = ctx_.stream;
    const int tokens    = static_cast<int>(ids.numel());
    const int batch     = active_sequence_batch_ == 0 ? 1 : active_sequence_batch_;
    const int width     = tokens / batch;
    if (tokens <= 0 || width * batch != tokens) {
        throw std::invalid_argument("Flash-Next MTP token geometry is invalid");
    }
    require_tensor_shape(hidden, DType::BF16, {10240, tokens},
                         "Flash-Next MTP input predictor hidden");
    require_tensor_shape(sample_hidden, DType::BF16, {2560, tokens},
                         "Flash-Next MTP sample hidden");
    require_tensor_shape(predictor_hidden, DType::BF16, {10240, tokens},
                         "Flash-Next MTP output predictor hidden");
    Tensor embedding = work_.alloc(DType::BF16, {2560, tokens});
    if (input_embeddings != nullptr) {
        require_tensor_shape(*input_embeddings, DType::BF16, {2560, tokens},
                             "Flash-Next MTP input embeddings");
        CUDA_CHECK(cudaMemcpyAsync(embedding.data, input_embeddings->data, embedding.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        ops::embedding(ids.view({tokens}), *embed_, embedding, stream);
    }
    Tensor normalized_embedding = work_.alloc(DType::BF16, {2560, tokens});
    Tensor projected_embedding  = work_.alloc(DType::BF16, {2560, tokens});
    ops::rmsnorm(embedding, mtp_.payload->embedding_norm, kCfg.rms_eps, true, normalized_embedding,
                 stream);
    ops::linear(normalized_embedding, mtp_.payload->embedding_projection, projected_embedding,
                stream, bf16_gemm_);

    Tensor normalized_hidden = work_.alloc(DType::BF16, {10240, tokens});
    Tensor hyper             = work_.alloc(DType::BF16, {10240, tokens});
    ops::rmsnorm(hidden, mtp_.payload->hidden_norm, kCfg.rms_eps, true, normalized_hidden, stream);
    Tensor normalized_hidden_branches = normalized_hidden.view({2560, 4 * tokens});
    Tensor hyper_branches             = hyper.view({2560, 4 * tokens});
    ops::linear(normalized_hidden_branches, mtp_.payload->hidden_projection, hyper_branches, stream,
                bf16_gemm_);
    ops::hyperconnection_add_repeated(projected_embedding, hyper, stream);
    Tensor block_input  = work_.alloc(DType::BF16, {2560, tokens});
    Tensor block_output = work_.alloc(DType::BF16, {2560, tokens});
    Tensor injection    = work_.alloc(DType::BF16, {4, tokens});
    ops::hyperconnection_mix(hyper, mtp_.payload->attention_hc, block_input, &injection, work_,
                             stream);

    Tensor cache_positions = positions.view({width, batch});
    Tensor mrope_positions;
    if (rope_positions.ne[2] == 3 || rope_positions.ne[1] == 3) {
        mrope_positions = rope_positions.view({width, batch, 3});
    } else {
        mrope_positions = work_.alloc(DType::I32, {width, batch, 3});
        ops::flash_next_expand_text_positions(rope_positions.view({width, batch}), mrope_positions,
                                              stream);
    }
    Tensor valid;
    if (active_valid_columns_ != nullptr) {
        valid = *active_valid_columns_;
    } else {
        valid = work_.alloc(DType::I32, {batch});
        // Captured schedules must not retain a pointer into a temporary host vector.
        ops::fill_i32(valid, width, stream);
    }
    const Tensor& rows = active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                                  : io_.backend_kv_table_row;
    const auto cache   = active_sequence_batch_ == 0
                             ? single_row_paged_kv_batch_view(mtp_kv_.layer_view(0))
                             : batch_mtp_kv_->batch_layer_view(0);
    ops::flash_next_qsa(
        block_input, cache_positions, mrope_positions, valid, rows, mtp_.payload->attention, cache,
        envelope, block_output, work_, stream, bf16_gemm_,
        {.selected_indices = selected_qsa_indices, .reused_indices = reused_qsa_indices});
    ops::hyperconnection_combine_mix(hyper, block_output, injection, mtp_.payload->mlp_hc,
                                     block_input, &injection, work_, stream);
    ops::flash_next_moe(block_input, mtp_.payload->post_mixer, block_output, work_, stream,
                        bf16_gemm_);
    ops::hyperconnection_combine(hyper, block_output, injection, stream);
    CUDA_CHECK(cudaMemcpyAsync(predictor_hidden.data, hyper.data, hyper.bytes(),
                               cudaMemcpyDeviceToDevice, stream));

    const ops::HyperConnectionWeights final_weights{
        .norm = mtp_.payload->final_hc.norm,
        .down = mtp_.payload->final_hc.down,
        .up   = mtp_.payload->final_hc.up,
    };
    ops::hyperconnection_mix(hyper, final_weights, sample_hidden, nullptr, work_, stream);
    (void)envelope;
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, T},
                         "Flash-Next MTP prefill predictor hidden");
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("Flash-Next MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {4 * kCfg.hidden, 1},
                             "Flash-Next MTP final predictor hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1},
                             "Flash-Next MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1},
                             "Flash-Next MTP final prefill draft token");
    }
    auto flash_scope = work_.scope();
    Tensor sample    = work_.alloc(DType::BF16, {kCfg.hidden, T});
    Tensor predictor = work_.alloc(DType::BF16, {4 * kCfg.hidden, T});
    mtp_forward_core(ids, hidden, positions, rope_positions, envelope, sample, input_embeddings,
                     &predictor);
    if (final_chunk) {
        const Tensor last_predictor = predictor.slice(1, T - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(final_hidden->data, last_predictor.data, last_predictor.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        Tensor last_sample = sample.slice(1, T - 1, 1);
        proposal_argmax(last_sample, *logits, *draft_token);
    }
    return;
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream, bf16_gemm_);
        if constexpr (Variant::flash_next) {
            constexpr int kRerankTile        = 512;
            constexpr int kCandidatesPerTile = 2;
            const int candidate_rows =
                ((Variant::draft_head_valid_rows + kRerankTile - 1) / kRerankTile) *
                kCandidatesPerTile;
            Tensor candidate_ids    = work_.alloc(DType::I32, {candidate_rows, T});
            Tensor candidate_scores = work_.alloc(DType::FP32, {candidate_rows, T});
            ops::shortlist_exact_argmax(hidden, proposal_logits, Variant::draft_head_valid_rows,
                                        *lm_head_, proposal_head_ids_, candidate_ids,
                                        candidate_scores, proposal_tokens, ctx_.stream);
        } else {
            ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream, bf16_gemm_);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings, Tensor* predictor_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, T}, "MTP predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP predictor output is required");
    }
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings, predictor_hidden);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token,
                                      Tensor* predictor_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {4 * kCfg.hidden, 1},
                         "MTP AR previous predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP AR predictor output is required");
    }
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden, nullptr,
                     predictor_hidden);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::mtp_forward_decode_batch(
    const Tensor& ids, const Tensor& hidden, const Tensor& cache_positions,
    const Tensor& rope_positions, const Tensor& valid_columns, const Tensor& kv_table_rows,
    ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden, Tensor* predictor_hidden,
    Tensor* selected_qsa_indices, const Tensor* reused_qsa_indices) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, width, batch},
                         "MTP decode batch target predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP decode predictor output is required");
    }
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    Tensor flat_hidden = hidden.view({4 * kCfg.hidden, width * batch});
    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, width * batch});
    Tensor flat_predictor_hidden =
        predictor_hidden->view({4 * kCfg.hidden, width * batch});
    mtp_forward_core(ids, flat_hidden, cache_positions, rope_positions, envelope, flat_mtp_hidden,
                     nullptr, &flat_predictor_hidden, selected_qsa_indices, reused_qsa_indices);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::schedule
