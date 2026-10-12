#include "models/qwen3_8_flash_next/program/p3.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

P3Program::P3Program(const LoadedModelData& model_in, const SequencePlanImpl& plan, DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), prefill_chunk(plan.prefill_chunk),
      proposal_head(plan.proposal_head), workspace_plan(plan.workspace),
      graph_allowance_bytes(plan.graph_allowance_bytes),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      bf16_gemm(std::make_unique<ops::Bf16GemmContext>(device_in.stream)),
      ple_gather_workers(std::make_unique<HostWorkerPool>(16, 16)),
      round_host(sizeof(TokenId)),
      commit_folds_ready_(device_in),
      ordinary_host(sizeof(qwen3_8_flash_next::OrdinaryDecodeIngress) +
                    sizeof(qwen3_8_flash_next::OrdinaryDecodeEgress)) {
    if (plan.speculative_backend == SpeculativeBackend::DFlash) {
        throw std::logic_error("V3 P4 does not support DFlash");
    }
    backend = plan.speculative_backend;
    draft_window = plan.draft_window;
    minimum_draft_window = plan.minimum_draft_window;
    if (model.weight_bytes == 0) {
        throw std::invalid_argument("Flash-Next model view has no owning weight arena");
    }
    if (!plan.persistent.flash_decode_ple ||
        (plan.speculative_backend == SpeculativeBackend::Mtp &&
         !plan.persistent.flash_ple_records)) {
        throw std::logic_error("Flash-Next P3 requires the PLE decode buffers");
    }
    causal_scoring = plan.causal_scoring;
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_8_flash_next::DecoderState>(backing, plan.persistent.decoder);
    page_capacity = decoder->text_kv.execution_tables().logical_page_capacity();
    text_kv_pages = std::make_unique<program::LogicalKVPageStore>(
        decoder->text_kv.page_pool(), decoder->text_kv.page_pool().capacity_pages());
    state_images = std::make_unique<qwen3_8_flash_next::StateImageDevicePool>(
        backing, plan.persistent.state_images);
    state_store = std::make_unique<qwen3_8_flash_next::detail::StateImageStore>(
        *state_images, nullptr, static_cast<std::uint32_t>(state_images->slot_count()));
    cache_slots.resize(static_cast<std::size_t>(state_images->slot_count()));
    text_kv_addresses = std::make_unique<program::KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(),
        std::max(plan.max_concurrency, 1U) + static_cast<std::uint32_t>(cache_slots.size()),
        page_capacity);
    io           = qwen3_8_flash_next::RoundState(backing, plan.persistent.round);
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    flash_decode_ple.emplace(plan.persistent.flash_decode_ple->bind(backing));
    if (plan.persistent.flash_ple_records) {
        flash_ple_records.emplace(plan.persistent.flash_ple_records->bind(backing));
    }
    if (plan.persistent.score_hidden) {
        score_hidden.emplace(plan.persistent.score_hidden->bind(backing));
        score_logprobs_host.emplace(kCausalScoreTile * sizeof(float));
    }
    sampling_config = plan.persistent.sampling_config.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    flash_ple_host.emplace(plan.persistent.flash_decode_ple->region.bytes);
    ordinary_profiles = ordinary_graph_profiles(plan.capacity);
    max_concurrency  = plan.max_concurrency;
    set_i32(io.text_kv_table_row, 0);
    set_i32(io.backend_kv_table_row, 0);
    decode_wait = std::make_unique<LowLatencyStreamWait>(device);
    if (io.ordinary) {
        ordinary_ingress =
            static_cast<qwen3_8_flash_next::OrdinaryDecodeIngress*>(ordinary_host.data());
        ordinary_egress = reinterpret_cast<qwen3_8_flash_next::OrdinaryDecodeEgress*>(
            static_cast<std::byte*>(ordinary_host.data()) +
            sizeof(qwen3_8_flash_next::OrdinaryDecodeIngress));
        if (plan.use_cuda_graph) { prepare_ordinary_graphs(); }
    } else if (backend == SpeculativeBackend::Mtp) {
        setup_mtp(plan, backing);
    } else {
        throw std::logic_error("sequence plan has no ordinary or MTP decode frame");
    }
    // The envelope for short prefill passes stays in force even when capture is off.
    qwen3_8_flash_next::detail::configure_prefill_graphs(plan.use_cuda_graph);
}

P3Program::~P3Program() {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    qwen3_8_flash_next::detail::reset_prefill_graphs();
}

P3Program::Request P3Program::plan_request(const qwen3_8_flash_next::PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) const {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains a token outside the model domain");
        }
    }
    Request out;
    out.tokens                        = prompt.token_ids;
    out.identity.assign(prompt);
    out.rope_delta                    = prompt.rope_delta;
    out.summary.prompt_tokens         = static_cast<std::uint32_t>(prompt.token_ids.size());
    out.summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t room          = capacity - out.summary.prompt_tokens + 1U;
    out.summary.effective_output_tokens = std::min(options.requested_output_tokens, room);
    out.summary.effective_limit_reason  = options.requested_output_tokens <= room
                                              ? FinishReason::OutputLimit
                                              : FinishReason::ContextCapacity;
    out.sampling.temperature            = options.sampling.temperature;
    out.sampling.top_k                  = options.sampling.top_k;
    out.sampling.top_p                  = options.sampling.top_p;
    out.sampling.min_p                  = options.sampling.min_p;
    out.sampling.presence_penalty       = options.sampling.presence_penalty;
    out.sampling.frequency_penalty      = options.sampling.frequency_penalty;
    out.sampling.seed                   = options.sampling.seed;
    if (prompt.has_media()) { out.vision = plan_vision(prompt, out.media); }
    return out;
}

void P3Program::bind(Request request, std::uint32_t row) {
    if (row >= max_concurrency) { throw std::logic_error("V3 lane is outside the plan"); }
    focus_ = row;
    if (lane().bound) { throw std::logic_error("V3 lane is already bound"); }
    const KvEntitlement entitlement = kv_entitlement(request);
    try {
        auto kv = text_kv_addresses->create_active(entitlement.text, static_cast<std::int32_t>(row),
                                                   device.stream);
        if (!kv) { throw std::runtime_error("KV address reservation failed"); }
        lane().kv  = *kv;
        auto state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::runtime_error("state slot reservation failed"); }
        lane().state = *state;
        lane().slot  = state_store->physical_slot(lane().state);
        if (backend == SpeculativeBackend::Mtp) {
            auto mtp = mtp_addresses->create_active(entitlement.mtp,
                                                    static_cast<std::int32_t>(row), device.stream);
            if (!mtp) { throw std::runtime_error("MTP KV address reservation failed"); }
            lane().mtp_kv = *mtp;
        }
        lane().request = std::move(request);
        install_request(lane().request);
        if (lane().request.vision) { attach_vision(); }
    } catch (...) {
        undo_bind();
        throw;
    }
    lane().bound     = true;
    lane().cursor    = 0;
    lane().kv_tokens = 0;
    lane().stats     = fresh_stats();
}

void P3Program::undo_bind() noexcept {
    const auto kv    = lane().kv;
    const auto mtp   = lane().mtp_kv;
    const auto state = lane().state;
    lane()           = {};
    release_frozen(kv, mtp, state);
}

SpeculativeStats P3Program::fresh_stats() const {
    SpeculativeStats out = speculative;
    out.rounds           = 0;
    out.drafted_tokens   = 0;
    out.accepted_tokens  = 0;
    out.fallback_steps   = 0;
    std::fill(out.accepted_per_position.begin(), out.accepted_per_position.end(), 0);
    return out;
}

void P3Program::map_kv(std::uint32_t tokens) {
    const auto grow = [&](program::KVAddressSpaceStore& store,
                          program::KVAddressSpaceHandle handle,
                          std::uint32_t wanted) {
        const std::uint64_t mapped =
            static_cast<std::uint64_t>(store.mapped_pages(handle)) * kPagedKVPageSize;
        if (wanted > mapped) { store.ensure_mapped_to_tokens(handle, wanted, device.stream); }
    };
    grow(*text_kv_addresses, lane().kv, tokens);
    if (lane().mtp_kv.valid() && mtp_addresses) { grow(*mtp_addresses, lane().mtp_kv, tokens); }
}

std::uint32_t P3Program::prompt_chunk_limit() const noexcept {
    std::uint32_t limit = lane().request.summary.prompt_tokens;
    for (const std::uint32_t frontier : lane().request.split_frontiers) {
        if (frontier > lane().cursor && frontier < limit) { limit = frontier; }
    }
    return limit;
}

P3Program::Step P3Program::advance_prefill(runtime::TokenMaskProvider* masks) {
    require_bound();
    Step step;
    step.grammar_dead = apply_mask(masks, 0, focus_, {});
    if (lane().cursor >= lane().request.summary.prompt_tokens) {
        throw std::logic_error("prefill is already complete");
    }
    const std::uint32_t prompt_tokens = lane().request.summary.prompt_tokens;
    const std::uint32_t nominal = std::min(prefill_chunk, prompt_chunk_limit() - lane().cursor);
    std::uint32_t remaining     = nominal;
    std::uint32_t processed     = 0;
    bool finalized              = false;
    std::uint32_t final_tokens  = 0;
    while (remaining != 0) {
        schedule::PrefillContext schedule_state = make_prefill();
        schedule_state.text_kv_base             = lane().cursor;
        const bool final_candidate              = lane().cursor + remaining == prompt_tokens;
        if (final_candidate && backend == SpeculativeBackend::Mtp) {
            schedule_state.mtp_proposal_extent = lane().mtp_extent;
        }
        schedule::PrefillChunkResult result;
        if (lane().vision) {
            result = schedule::prefill_multimodal_chunk(
                schedule_state, *lane().request.media, *lane().vision, remaining, std::nullopt,
                final_candidate);
            lane().vision->release_encoded_media_payloads();
        } else {
            result = schedule::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(lane().request.tokens), remaining,
                std::nullopt, final_candidate);
        }
        if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
            throw std::logic_error("prefill chunk made invalid progress");
        }
        if (lane().fork_pending) {
            state_store->commit_fork(lane().fork_source, lane().state);
            lane().fork_pending = false;
        }
        lane().cursor += result.processed_tokens;
        processed += result.processed_tokens;
        remaining -= result.processed_tokens;
        final_tokens = result.processed_tokens;
        lane().kv_tokens = lane().cursor;
        text_kv_addresses->commit_frontier(lane().kv, lane().kv_tokens);
        if (lane().mtp_kv.valid()) {
            mtp_addresses->commit_frontier(lane().mtp_kv, lane().kv_tokens);
        }
        finalized = result.finalized;
        if (finalized || remaining == 0) { break; }
    }
    step.processed_tokens = processed;
    if (lane().cursor < prompt_tokens && lane().request.summary.publish_continuation &&
        lane().cursor > lane().capture_skipped_through &&
        std::find(lane().request.capture_frontiers.begin(), lane().request.capture_frontiers.end(),
                  lane().cursor) != lane().request.capture_frontiers.end()) {
        step.capture_ready     = true;
        lane().capture_offered = true;
    }
    if (!finalized) { return step; }
    if (lane().cursor != prompt_tokens) {
        throw std::logic_error("prefill sampled before the prompt frontier");
    }
    copy_token();
    if (backend == SpeculativeBackend::Mtp && lane().mtp_extent != 0 && io.mtp) {
        CUDA_CHECK(cudaMemcpyAsync(lane().drafts.data(), io.mtp->draft_tokens.data,
                                   lane().mtp_extent * sizeof(TokenId), cudaMemcpyDeviceToHost,
                                   device.stream));
        lane().draft_count = lane().mtp_extent;
    }
    device.synchronize();
    const TokenId token = *static_cast<TokenId*>(round_host.data());
    if (token < 0 || token >= TextConfig::token_domain) {
        throw std::runtime_error("prefill sampled a token outside the model domain");
    }
    if (lane().pending) { throw std::logic_error("prefill sampled over an uncommitted token"); }
    lane().pending      = token;
    lane().prefill_done = true;
    step.complete     = true;
    step.token        = token;
    return step;
}

void P3Program::select(std::uint32_t row) {
    if (row >= lanes_.size()) { throw std::logic_error("V3 lane is outside the plan"); }
    focus_ = row;
}

void P3Program::note_decode_seconds(std::span<const std::uint32_t> rows, double seconds) noexcept {
    const std::uint32_t saved = focus_;
    for (const std::uint32_t row : rows) {
        select(row);
        lane().decode_seconds += seconds;
    }
    select(saved);
}

P3Program::BatchStep P3Program::decode_rows(std::span<const std::uint32_t> rows,
                                    std::span<const std::uint32_t> budgets,
                                    runtime::TokenMaskProvider* masks) {
    if (rows.empty() || rows.size() != budgets.size()) {
        throw std::logic_error("V3 decode rows and budgets disagree");
    }
    if (backend == SpeculativeBackend::Mtp) { return decode_mtp(rows, budgets, masks); }
    *ordinary_ingress = {};
    *ordinary_egress  = {};
    std::uint32_t maximum = 0;
    const std::size_t ple_bytes = 2560U * dtype_size(model.ple_table_dtype);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        select(rows[index]);
        require_bound();
        if (!lane().prefill_done || lane().pending) {
            throw std::logic_error("decode requires a committed prefill sample");
        }
        const std::uint32_t frontier = lane().kv_tokens;
        if (frontier >= capacity) { throw std::logic_error("decode is past the context capacity"); }
        maximum = std::max(maximum, frontier);
        ordinary_ingress->tokens[index]               = lane().request.tokens.back();
        ordinary_ingress->cache_positions[index]      = static_cast<std::int32_t>(frontier);
        ordinary_ingress->rope_positions[index] =
            checked_i32(frontier, "ordinary batch RoPE position") + lane().request.rope_delta;
        ordinary_ingress->text_kv_table_rows[index]   = text_kv_addresses->bound_row(lane().kv);
        ordinary_ingress->state_source_slots[index]   = lane().slot;
        ordinary_ingress->state_destination_slots[index] = lane().slot;
        (void)apply_mask(masks, index, rows[index], {});
        ordinary_ingress->sampling[index]             = lane().request.sampling;
        gather_last_ple(index * ple_bytes);
        text_kv_addresses->ensure_mapped_to_tokens(lane().kv, frontier + 1, device.stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(flash_decode_ple->data, flash_ple_host->data(),
                               rows.size() * ple_bytes, cudaMemcpyHostToDevice, device.stream));
    const auto planned = std::find_if(
        ordinary_profiles.begin(), ordinary_profiles.end(),
        [&](const GraphExecutionProfile& profile) {
            return maximum >= profile.min && maximum <= profile.max;
        });
    if (planned == ordinary_profiles.end()) {
        throw std::logic_error("ordinary execution profile is unavailable");
    }
    const auto batch = static_cast<std::uint32_t>(rows.size());
    const ops::CausalAttentionExecutionEnvelope envelope{planned->min + 1, planned->max + 1};
    DecodeGraphExecutable* executable = nullptr;
    if (!ordinary_graphs.profiles.empty()) {
        DecodeGraphProfile& graph = ordinary_graphs.select(batch, maximum, "ordinary batch");
        executable                = &ordinary_graphs.install(graph, "ordinary batch");
    }
    schedule::OrdinaryBatchContext schedule_state = make_decode();
    decode_wait->begin(device.stream);
    schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(batch), envelope,
                                    executable);
    decode_wait->mark_launch(device.stream);
    decode_wait->wait(device.stream);
    BatchStep step;
    step.stride = 1;
    step.tokens.resize(rows.size());
    step.dead.resize(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const TokenId token = ordinary_egress->sampled_tokens[index];
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("decode sampled a token outside the model domain");
        }
        select(rows[index]);
        lane().kv_tokens = lane().kv_tokens + 1;
        text_kv_addresses->commit_frontier(lane().kv, lane().kv_tokens);
        lane().pending = token;
        step.tokens[index] = token;
        step.dead[index]   = lane().grammar_dead;
    }
    return step;
}

std::size_t P3Program::cuda_graph_allowance_bytes() const noexcept {
    return graph_allowance_bytes +
           ninfer::models::qwen3_8_flash_next::detail::prefill_graph_reserved_bytes();
}

std::int32_t P3Program::checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

void P3Program::accept_pending(std::optional<std::uint32_t> split) {
    require_bound();
    if (!lane().pending) { throw std::logic_error("V3 P3 has no token to commit"); }
    const TokenId token = *lane().pending;
    lane().pending.reset();
    lane().request.tokens.push_back(token);
    lane().request.identity.append_generated(1, lane().request.rope_delta, split);
}

void P3Program::append_forced(std::span<const TokenId> extra, std::optional<std::uint32_t> split) {
    require_bound();
    if (!lane().prefill_done || lane().pending) {
        throw std::logic_error("forced tokens require a committed prefill sample");
    }
    if (extra.empty()) { return; }
    for (const TokenId id : extra) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("forced token is outside the model domain");
        }
    }
    if (split && (*split == 0 || *split > extra.size())) {
        throw std::logic_error("forced-token execution split is outside its row");
    }
    count_tokens(extra);
    lane().request.tokens.insert(lane().request.tokens.end(), extra.begin(), extra.end());
    lane().request.identity.append_generated(extra.size(), lane().request.rope_delta, split);
    lane().cursor      = lane().kv_tokens;
    lane().draft_count = 0;
    const auto target = static_cast<std::uint32_t>(lane().request.tokens.size() - 1U);
    prefill_gap(target);
    device.synchronize();
}

void P3Program::release_lane() noexcept {
    if (!lane().bound) {
        lane() = {};
        return;
    }
    if (lane().fork_pending) {
        state_store->abort_fork(lane().fork_source, lane().state);
        lane().fork_pending = false;
    }
    const auto kv    = lane().kv;
    const auto mtp   = lane().mtp_kv;
    const auto state = lane().state;
    lane()             = {};
    (void)text_kv_addresses->release_after_deactivate(kv);
    if (mtp.valid() && mtp_addresses) { (void)mtp_addresses->release_after_deactivate(mtp); }
    (void)state_store->release(state);
}

std::pair<runtime::ContextResourceUsage, runtime::ContextResourceUsage> P3Program::pool_usage() const noexcept {
    runtime::ContextResourceUsage capacity_usage;
    runtime::ContextResourceUsage occupied_usage;
    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    capacity_usage.main_kv_pages = text_pool.capacity_pages();
    occupied_usage.main_kv_pages = text_pool.capacity_pages() - text_pool.available_pages();
    if (mtp_kv_pages) {
        const DeviceKVPagePool& mtp_pool = mtp_kv_pages->physical_pool();
        capacity_usage.backend_kv_pages = mtp_pool.capacity_pages();
        occupied_usage.backend_kv_pages = mtp_pool.capacity_pages() - mtp_pool.available_pages();
    }
    capacity_usage.state_slots = state_store->device_capacity();
    occupied_usage.state_slots = state_store->device_occupied();
    return {capacity_usage, occupied_usage};
}

std::optional<P3Program::CachePoint> P3Program::park_point() const noexcept {
    return parked.occupied && parked.device ? std::optional<CachePoint>(parked.point)
                                            : std::nullopt;
}

void P3Program::require_bound() const {
    if (!lane().bound) { throw std::logic_error("V3 P3 lane is not bound"); }
}

void P3Program::prefill_gap(std::uint32_t target) {
    if (target > lane().request.tokens.size() || lane().kv_tokens > target) {
        throw std::logic_error("forced-token KV gap is invalid");
    }
    map_kv(target);
    while (lane().kv_tokens < target) {
        const std::uint32_t left = std::min(prefill_chunk, target - lane().kv_tokens);
        schedule::PrefillContext schedule_state = make_prefill();
        schedule_state.text_kv_base             = lane().kv_tokens;
        const schedule::PrefillChunkResult result = schedule::prefill_text_chunk(
            schedule_state, std::span<const TokenId>(lane().request.tokens), left, std::nullopt,
            false);
        if (result.finalized || result.processed_tokens == 0 || result.processed_tokens > left) {
            throw std::logic_error("forced-token prefill made invalid progress");
        }
        if (lane().fork_pending) {
            state_store->commit_fork(lane().fork_source, lane().state);
            lane().fork_pending = false;
        }
        lane().cursor += result.processed_tokens;
        lane().kv_tokens += result.processed_tokens;
        text_kv_addresses->commit_frontier(lane().kv, lane().kv_tokens);
        if (lane().mtp_kv.valid()) {
            mtp_addresses->commit_frontier(lane().mtp_kv, lane().kv_tokens);
        }
    }
}

void P3Program::set_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice,
                               device.stream));
}

void P3Program::copy_token() {
    CUDA_CHECK(cudaMemcpyAsync(round_host.data(), io.token.data, sizeof(TokenId),
                               cudaMemcpyDeviceToHost, device.stream));
}

void P3Program::prepare_ordinary_graphs() {
    if (max_concurrency == 0 || max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("ordinary CUDA Graph concurrency is invalid");
    }
    validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
    std::vector<qwen3_8_flash_next::detail::StateImageHandle> capture_states;
    capture_states.reserve(max_concurrency);
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        auto state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        capture_states.push_back(*state);
    }
    const auto capture_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };
    std::vector<program::KVAddressSpaceHandle> capture_rows;
    capture_rows.reserve(max_concurrency);
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        auto allocation =
            text_kv_addresses->create_active(1, static_cast<std::int32_t>(row), device.stream);
        if (!allocation) { throw std::bad_alloc(); }
        capture_rows.push_back(*allocation);
        text_kv_addresses->ensure_mapped_to_tokens(*allocation, 1, device.stream);
        decoder->text_kv.execution_tables().publish_repeated(
            text_kv_addresses->execution_row(*allocation).handle(),
            text_kv_addresses->physical_page(*allocation, 0),
            decoder->text_kv.execution_tables().logical_page_capacity(), device.stream);
    }
    device.synchronize();

    const auto clear_controls = [&] {
        for (Tensor* tensor : {&io.token, &io.pos, &io.rope_pos, &io.rope_delta}) {
            CUDA_CHECK(cudaMemsetAsync(tensor->data, 0, tensor->bytes(), device.stream));
        }
    };
    const auto zero_capture_pages = [&](std::uint32_t batch_size) {
        std::vector<DeviceKVPageHandle> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(text_kv_addresses->physical_page(capture_rows[row], 0));
        }
        decoder->text_kv.page_pool().zero_pages(pages, device.stream);
    };
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        work.reset();
        clear_controls();
        zero_capture_pages(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            state_images->zero_slot(capture_slot(row), device.stream);
        }
        set_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        *ordinary_ingress = {};
        *ordinary_egress  = {};
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            ordinary_ingress->cache_positions[row] =
                checked_i32(frontier, "graph representative ordinary position");
            ordinary_ingress->rope_positions[row] =
                checked_i32(frontier, "graph representative ordinary RoPE position");
            ordinary_ingress->text_kv_table_rows[row]        = static_cast<std::int32_t>(row);
            ordinary_ingress->state_source_slots[row]        = capture_slot(row);
            ordinary_ingress->state_destination_slots[row]   = capture_slot(row);
        }
    };

    schedule::OrdinaryBatchContext schedule_state = make_decode();
    const GraphExecutionProfile code_warm         = ordinary_profiles.front();
    prepare_representative(code_warm.min, 1);
    device.synchronize();
    schedule::ordinary_decode_batch(schedule_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                    nullptr);
    device.synchronize();

    ordinary_graphs.profiles.reserve(ordinary_profiles.size() * max_concurrency);
    for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
        for (const GraphExecutionProfile planned : ordinary_profiles) {
            DecodeGraphProfile& profile    = ordinary_graphs.profiles.emplace_back();
            profile.batch_size             = batch_size;
            profile.min_execution_frontier = planned.min;
            profile.max_execution_frontier = planned.max;
            profile.topology_class = planned.topology_class * max_concurrency + (batch_size - 1U);
            const ops::CausalAttentionExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
            schedule::capture_ordinary_decode_batch(schedule_state,
                                                    static_cast<std::int32_t>(batch_size),
                                                    envelope, profile.definition);
        }
    }
    instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    state_images->zero_all(device.stream);
    device.synchronize();
    for (const auto state : capture_states) {
        if (!state_store->release(state)) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }
    for (const auto row : capture_rows) {
        text_kv_addresses->deactivate(row);
        if (!text_kv_addresses->release(row)) {
            throw std::logic_error("CUDA Graph capture KV address space could not be released");
        }
    }
}

void P3Program::gather_last_ple(std::size_t byte_offset) {
    const auto& tokens = lane().request.tokens;
    const std::size_t suffix_size = std::min<std::size_t>(3, tokens.size());
    std::vector<std::int32_t> suffix(suffix_size);
    for (std::size_t index = 0; index < suffix_size; ++index) {
        suffix[index] = tokens[tokens.size() - suffix_size + index];
    }
    std::vector<qwen3_8_flash_next::PleIds> ids(suffix_size);
    qwen3_8_flash_next::compute_ple_ids(suffix, ids);
    const std::size_t bytes = 2560U * dtype_size(model.ple_table_dtype);
    auto* destination = static_cast<std::byte*>(flash_ple_host->data()) + byte_offset;
    qwen3_8_flash_next::gather_ple(
        *model.ple_table, model.ple_table_dtype,
        std::span<const qwen3_8_flash_next::PleIds>(ids).subspan(suffix_size - 1U, 1U),
        std::span<std::byte>(destination, bytes));
}

schedule::ExecutionCore P3Program::execution_core() {
    return {device,
            model,
            work,
            state_images->linear(),
            state_images->ple_store(),
            flash_ple_records ? &*flash_ple_records : nullptr,
            replay_records ? &*replay_records : nullptr,
            io,
            prefill_hidden,
            prefill_chunk,
            proposal_head,
            mtp_prefill_hidden ? &*mtp_prefill_hidden : nullptr,
            state_images->mtp_hidden_store(),
            bf16_gemm.get(),
            ple_gather_workers.get()};
}

schedule::PrefillContext P3Program::make_prefill() {
    Tensor lane_sampling = sampling_config.ne[1] > 1 ? sampling_config.slice(1, 0, 1)
                                                     : sampling_config;
    CUDA_CHECK(cudaMemcpyAsync(lane_sampling.data, &lane().request.sampling,
                               sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               device.stream));
    qwen3_8_flash_next::PagedKVCacheView mtp_view{};
    const qwen3_8_flash_next::PagedKVCache* mtp_cache = nullptr;
    if (lane().mtp_kv.valid() && decoder->mtp_cache() != nullptr) {
        mtp_cache = decoder->mtp_cache();
        mtp_view  = mtp_cache->execution_view(mtp_addresses->execution_row(lane().mtp_kv));
    }
    schedule::PrefillContext out{
        .execution = execution_core(),
        .text_kv = decoder->text_kv.execution_view(text_kv_addresses->execution_row(lane().kv)),
        .mtp_kv                        = mtp_view,
        .text_cache                    = decoder->text_kv,
        .mtp_cache                     = mtp_cache,
        .text_kv_base                  = lane().cursor,
        .sampling                      = static_cast<const ops::SamplingConfig*>(lane_sampling.data),
        .rewrite_checkpoint_hidden     = nullptr,
        .rewrite_checkpoint_mtp_hidden = nullptr,
        .state_source_slot             = lane().fork_pending
                                             ? state_store->physical_slot(lane().fork_source)
                                             : lane().slot,
        .state_destination_slot        = lane().slot,
        .mtp_proposal_extent           = 0,
        .rope_delta                    = lane().request.rope_delta};
    return out;
}

void P3Program::configure_activation(const EngineOptions& options) {
    if (options.capture_path.empty() && options.steering_pack.empty()) { return; }
    activation = std::make_unique<qwen3_8_flash_next::ActivationControl>(device.stream);
    activation->configure(options, {});
    if (!options.capture_path.empty()) { io.activation_capture = activation->device(); }
}

void P3Program::activate_steering(const qwen3_8_flash_next::SteeringPack* pack) {
    if (activation == nullptr) {
        activation = std::make_unique<qwen3_8_flash_next::ActivationControl>(device.stream);
    }
    activation->activate(pack);
    active_steering_ = pack != nullptr ? pack->state : SteeringState{};
    io.activation = pack != nullptr ? activation->device() : nullptr;
}

SteeringState P3Program::steering_state() const { return active_steering_; }

void P3Program::begin_activation(std::uint32_t lane_index, std::uint64_t request_id,
                                 std::span<const TokenId>,
                                 const runtime::ResolvedExecutionOptions& options) {
    if (activation == nullptr) { return; }
    if (lane_index >= lanes_.size()) { throw std::out_of_range("activation lane"); }
    // Admission has already moved the engine prompt into this lane.
    const auto& tokens = lanes_[lane_index].request.tokens;
    activation->begin(static_cast<int>(lane_index), request_id, tokens, options.capture,
                      options.steering);
}

void P3Program::select_activation(std::span<const std::uint32_t> lanes) {
    if (activation == nullptr) { return; }
    activation->select(lanes);
}

std::string P3Program::flush_activation(std::uint32_t lane_index) {
    if (activation == nullptr || lane_index >= lanes_.size()) { return {}; }
    const auto& tokens = lanes_[lane_index].request.tokens;
    if (tokens.empty()) { return {}; }
    return activation->flush(static_cast<int>(lane_index), tokens,
                             static_cast<std::uint32_t>(tokens.size()));
}

schedule::OrdinaryBatchContext P3Program::make_decode() {
    return {execution_core(),
            decoder->text_kv,
            *io.ordinary,
            *ordinary_ingress,
            *ordinary_egress,
            state_images->continuation_hidden_store(),
            &*flash_decode_ple};
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
