// One-lane eager MTP round.
// Decode graphs for MTP are not captured here.

#include "models/qwen3_8_flash_next/program/p3.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

void P3Program::setup_mtp(const SequencePlanImpl& plan, DeviceSpan backing) {
    if (!io.mtp_decode || decoder->mtp_cache() == nullptr || !plan.persistent.replay_records ||
        !plan.persistent.round.mtp_decode) {
        throw std::logic_error("MTP sequence plan is missing its decode frame");
    }
    if (minimum_draft_window == 0 || minimum_draft_window > draft_window) {
        throw std::logic_error("MTP draft count range is invalid");
    }
    speculative.backend               = SpeculativeBackend::Mtp;
    speculative.enabled               = true;
    speculative.draft_window          = draft_window;
    speculative.accepted_per_position.assign(draft_window, 0);
    qwen3_8_flash_next::PagedKVCache* cache = decoder->mtp_cache();
    const std::uint32_t pages = cache->execution_tables().logical_page_capacity();
    mtp_kv_pages = std::make_unique<program::LogicalKVPageStore>(
        cache->page_pool(), cache->page_pool().capacity_pages());
    mtp_addresses = std::make_unique<program::KVAddressSpaceStore>(
        *mtp_kv_pages, cache->execution_tables(),
        std::max(max_concurrency, 1U) + static_cast<std::uint32_t>(cache_slots.size()), pages);
    replay_records.emplace(backing, *plan.persistent.replay_records);
    if (plan.persistent.mtp_prefill_hidden) {
        mtp_prefill_hidden.emplace(plan.persistent.mtp_prefill_hidden->bind(backing));
    }
    for (std::uint32_t drafts = minimum_draft_window; drafts <= draft_window; ++drafts) {
        mtp_frames.emplace_back(backing, *plan.persistent.round.mtp_decode,
                                plan.persistent.round.spec.batch_capacity, drafts);
        const auto width = static_cast<std::int32_t>(drafts + 1U);
        mtp_replay_records.push_back(replay_records->narrowed(width));
        mtp_replay_folds.emplace_back(mtp_replay_records.back(),
                                      state_images->linear().all_layers_view());
        if (flash_ple_records) {
            mtp_ple_records.emplace_back(
                flash_ple_records->data, flash_ple_records->dtype,
                std::initializer_list<std::int32_t>{flash_ple_records->ne[0], width,
                                                    flash_ple_records->ne[2]});
        }
    }
    mtp_host.emplace(sizeof(qwen3_8_flash_next::MtpDecodeIngress) +
                     sizeof(qwen3_8_flash_next::MtpDecodeEgress));
    mtp_ingress = static_cast<qwen3_8_flash_next::MtpDecodeIngress*>(mtp_host->data());
    mtp_egress  = reinterpret_cast<qwen3_8_flash_next::MtpDecodeEgress*>(
        static_cast<std::byte*>(mtp_host->data()) + sizeof(qwen3_8_flash_next::MtpDecodeIngress));
    if (plan.use_cuda_graph) { prepare_mtp_graphs(plan.capacity); }
}

void P3Program::prepare_mtp_graphs(std::uint32_t graph_capacity) {
    const auto profiles = mtp_graph_profiles(graph_capacity, draft_window);
    validate_graph_profiles(profiles, graph_capacity - 1U, "MTP");
    auto state = state_store->reserve_reset(device.stream);
    if (!state) { throw std::bad_alloc(); }
    const std::int32_t slot = state_store->physical_slot(*state);
    auto text = text_kv_addresses->create_active(1, 0, device.stream);
    auto mtp  = mtp_addresses->create_active(1, 0, device.stream);
    if (!text || !mtp) { throw std::bad_alloc(); }
    text_kv_addresses->ensure_mapped_to_tokens(*text, 1, device.stream);
    mtp_addresses->ensure_mapped_to_tokens(*mtp, 1, device.stream);
    decoder->text_kv.execution_tables().publish_repeated(
        text_kv_addresses->execution_row(*text).handle(), text_kv_addresses->physical_page(*text, 0),
        decoder->text_kv.execution_tables().logical_page_capacity(), device.stream);
    decoder->mtp_cache()->execution_tables().publish_repeated(
        mtp_addresses->execution_row(*mtp).handle(), mtp_addresses->physical_page(*mtp, 0),
        decoder->mtp_cache()->execution_tables().logical_page_capacity(), device.stream);
    device.synchronize();

    const auto prepare = [&](std::uint32_t frontier, std::uint32_t drafts) {
        work.reset();
        for (Tensor* tensor : {&io.token, &io.pos, &io.rope_pos, &io.rope_delta}) {
            CUDA_CHECK(cudaMemsetAsync(tensor->data, 0, tensor->bytes(), device.stream));
        }
        if (io.mtp) {
            CUDA_CHECK(cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(),
                                       device.stream));
        }
        state_images->zero_slot(slot, device.stream);
        decoder->text_kv.page_pool().zero_pages(
            std::array<DeviceKVPageHandle, 1>{text_kv_addresses->physical_page(*text, 0)},
            device.stream);
        decoder->mtp_cache()->page_pool().zero_pages(
            std::array<DeviceKVPageHandle, 1>{mtp_addresses->physical_page(*mtp, 0)}, device.stream);
        set_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        *mtp_ingress = {};
        *mtp_egress  = {};
        const std::uint32_t extent = std::min(drafts, graph_capacity - frontier - 1U);
        const std::uint32_t width  = drafts + 1U;
        mtp_ingress->base_frontiers[0] = checked_i32(frontier, "graph representative MTP frontier");
        mtp_ingress->remaining_budgets[0] =
            checked_i32(graph_capacity, "graph representative MTP budget");
        mtp_ingress->current_extents[0]      = static_cast<std::int32_t>(extent);
        mtp_ingress->target_valid_columns[0] = static_cast<std::int32_t>(extent + 1U);
        for (std::uint32_t column = 0; column < width; ++column) {
            mtp_ingress->target_rope_positions[column] = checked_i32(
                frontier + std::min(column, extent), "graph representative MTP RoPE position");
        }
        mtp_ingress->state_source_slots[0]      = slot;
        mtp_ingress->state_destination_slots[0] = slot;
    };

    mtp_graphs.resize(draft_window - minimum_draft_window + 1U);
    const GraphExecutionProfile code_warm = profiles.front();
    for (std::uint32_t drafts = minimum_draft_window; drafts <= draft_window; ++drafts) {
        const std::size_t frame = drafts - minimum_draft_window;
        schedule::ExecutionCore core = execution_core();
        core.replay_records          = &mtp_replay_records[frame];
        core.ple_records = frame < mtp_ple_records.size() ? &mtp_ple_records[frame] : nullptr;
        schedule::MtpBatchContext schedule_state{core,
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 mtp_frames[frame],
                                                 *mtp_ingress,
                                                 *mtp_egress,
                                                 state_images->continuation_hidden_store(),
                                                 &*flash_decode_ple};
        prepare(code_warm.min, drafts);
        device.synchronize();
        schedule::mtp_decode_batch(
            schedule_state, 1, drafts,
            mtp_causal_attention_envelopes(code_warm.max, drafts, graph_capacity), nullptr);
        device.synchronize();
        DecodeGraphFamily& graphs = mtp_graphs[frame];
        for (const GraphExecutionProfile planned : profiles) {
            DecodeGraphProfile& profile    = graphs.profiles.emplace_back();
            profile.batch_size             = 1;
            profile.min_execution_frontier = planned.min;
            profile.max_execution_frontier = planned.max;
            profile.topology_class         = planned.topology_class;
            schedule::capture_mtp_decode_batch(
                schedule_state, 1, drafts,
                mtp_causal_attention_envelopes(planned.max, drafts, graph_capacity),
                profile.definition);
        }
        instantiate_graph_family(graphs, "MTP", device,
                                 [&](std::uint32_t frontier, std::uint32_t) {
                                     prepare(frontier, drafts);
                                 });
    }
    state_images->zero_all(device.stream);
    device.synchronize();
    if (!state_store->release(*state)) {
        throw std::logic_error("MTP CUDA Graph capture StateImage could not be released");
    }
    text_kv_addresses->deactivate(*text);
    mtp_addresses->deactivate(*mtp);
    if (!text_kv_addresses->release(*text) || !mtp_addresses->release(*mtp)) {
        throw std::logic_error("MTP CUDA Graph capture KV address space could not be released");
    }
}

std::uint32_t P3Program::select_mtp_drafts() const {
    if (minimum_draft_window == draft_window) { return draft_window; }
    constexpr std::uint32_t kFusedVerifyTokens = 16;
    const std::uint32_t widest =
        std::clamp(kFusedVerifyTokens / 1U, minimum_draft_window + 1U, draft_window + 1U) - 1U;
    const auto shape = [](std::uint32_t count) { return 6.75e-3 + 0.85e-3 * count; };
    const auto seconds = [&](std::uint32_t drafts) {
        if (measured_seconds[drafts] > 0.0) { return measured_seconds[drafts]; }
        double scale    = 0.0;
        std::uint32_t n = 0;
        for (std::uint32_t count = minimum_draft_window; count <= draft_window; ++count) {
            if (measured_seconds[count] > 0.0) {
                scale += measured_seconds[count] / shape(count);
                ++n;
            }
        }
        return (n == 0 ? 1.0 : scale / n) * shape(drafts);
    };
    std::uint32_t best = minimum_draft_window;
    double best_rate   = -1.0;
    for (std::uint32_t drafts = minimum_draft_window; drafts <= widest; ++drafts) {
        const double rate = draft_statistics.expected_tokens(drafts) / seconds(drafts);
        if (rate > best_rate) {
            best      = drafts;
            best_rate = rate;
        }
    }
    return best;
}

P3Program::BatchStep P3Program::decode_mtp(std::span<const std::uint32_t> rows,
                                           std::span<const std::uint32_t> budgets,
                                           runtime::TokenMaskProvider* masks) {
    const std::uint32_t drafts = select_mtp_drafts();
    const std::uint32_t width  = drafts + 1U;
    const std::size_t frame    = drafts - minimum_draft_window;
    *mtp_ingress               = {};
    *mtp_egress                = {};
    std::uint32_t maximum      = 0;
    std::vector<std::uint32_t> extents(rows.size());
    const std::size_t ple_bytes = 2560U * dtype_size(model.ple_table_dtype);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        select(rows[index]);
        require_bound();
        if (!lane().prefill_done || lane().pending) {
            throw std::logic_error("MTP decode requires a committed prefill sample");
        }
        const std::uint32_t frontier = lane().kv_tokens;
        const std::uint32_t max_by_budget = budgets[index] > 1 ? budgets[index] - 1U : 0U;
        const std::uint32_t extent =
            std::min({lane().draft_count, drafts, max_by_budget,
                      capacity > frontier ? capacity - frontier - 1U : 0U});
        extents[index] = extent;
        maximum        = std::max(maximum, frontier);
        mtp_ingress->anchors[index]              = lane().request.tokens.back();
        mtp_ingress->base_frontiers[index]       = checked_i32(frontier, "MTP batch frontier");
        mtp_ingress->remaining_budgets[index]    = checked_i32(budgets[index], "MTP batch remaining budget");
        mtp_ingress->current_extents[index]      = static_cast<std::int32_t>(extent);
        mtp_ingress->target_valid_columns[index] = static_cast<std::int32_t>(extent + 1U);
        for (std::uint32_t column = 0; column < drafts; ++column) {
            mtp_ingress->current_drafts[index * drafts + column] =
                column < extent ? lane().drafts[column] : lane().request.tokens.back();
        }
        for (std::uint32_t column = 0; column < width; ++column) {
            mtp_ingress->target_rope_positions[index * width + column] =
                checked_i32(frontier + std::min(column, extent), "MTP batch RoPE position") +
                lane().request.rope_delta;
        }
        mtp_ingress->rope_deltas[index] = lane().request.rope_delta;
        mtp_ingress->text_kv_table_rows[index]      = text_kv_addresses->bound_row(lane().kv);
        mtp_ingress->mtp_kv_table_rows[index]       = mtp_addresses->bound_row(lane().mtp_kv);
        mtp_ingress->state_source_slots[index]      = lane().slot;
        mtp_ingress->state_destination_slots[index] = lane().slot;
        (void)apply_mask(masks, index, rows[index],
                         std::span<const TokenId>(
                             mtp_ingress->current_drafts.data() + index * drafts, extent));
        mtp_ingress->sampling[index]                = lane().request.sampling;

        const std::size_t history = std::min<std::size_t>(3, lane().request.tokens.size());
        std::vector<std::int32_t> ple_tokens(history + drafts);
        for (std::size_t token = 0; token < history; ++token) {
            ple_tokens[token] = lane().request.tokens[lane().request.tokens.size() - history + token];
        }
        for (std::uint32_t column = 1; column < width; ++column) {
            ple_tokens[history + column - 1U] =
                mtp_ingress->current_drafts[index * drafts + column - 1U];
        }
        std::vector<qwen3_8_flash_next::PleIds> ple_ids(ple_tokens.size());
        qwen3_8_flash_next::compute_ple_ids(ple_tokens, ple_ids);
        auto* ple_destination = static_cast<std::byte*>(flash_ple_host->data()) +
                                index * width * ple_bytes;
        qwen3_8_flash_next::gather_ple(
            *model.ple_table, model.ple_table_dtype,
            std::span<const qwen3_8_flash_next::PleIds>(ple_ids).subspan(history - 1U, width),
            std::span<std::byte>(ple_destination, width * ple_bytes));
        text_kv_addresses->ensure_mapped_to_tokens(lane().kv, frontier + extent + 1U, device.stream);
        mtp_addresses->ensure_mapped_to_tokens(
            lane().mtp_kv, std::min(capacity, frontier + extent + drafts), device.stream);
        lane().batch_index = static_cast<std::uint32_t>(index);
    }
    CUDA_CHECK(cudaMemcpyAsync(flash_decode_ple->data, flash_ple_host->data(),
                               rows.size() * width * ple_bytes, cudaMemcpyHostToDevice,
                               device.stream));

    schedule::ExecutionCore core = execution_core();
    core.replay_records = &mtp_replay_records[frame];
    core.ple_records    = frame < mtp_ple_records.size() ? &mtp_ple_records[frame] : nullptr;
    schedule::MtpBatchContext schedule_state{core,
                                             decoder->text_kv,
                                             *decoder->mtp_cache(),
                                             mtp_frames[frame],
                                             *mtp_ingress,
                                             *mtp_egress,
                                             state_images->continuation_hidden_store(),
                                             &*flash_decode_ple};
    DecodeGraphExecutable* executable = nullptr;
    auto envelopes = mtp_causal_attention_envelopes(maximum, drafts, capacity);
    if (rows.size() == 1 && frame < mtp_graphs.size() && !mtp_graphs[frame].profiles.empty()) {
        DecodeGraphProfile& profile = mtp_graphs[frame].select(1, maximum, "MTP batch");
        executable                  = &mtp_graphs[frame].install(profile, "MTP batch");
        envelopes  = mtp_causal_attention_envelopes(profile.max_execution_frontier, drafts, capacity);
    }
    const auto started = std::chrono::steady_clock::now();
    decode_wait->begin(device.stream);
    schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(rows.size()), drafts,
                               envelopes, executable);
    decode_wait->mark_launch(device.stream);
    decode_wait->wait(device.stream);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (measured_samples[drafts]++ > 0) {
        measured_seconds[drafts] =
            measured_seconds[drafts] == 0.0
                ? seconds
                : 0.9 * measured_seconds[drafts] + 0.1 * std::min(seconds, 1.5 * measured_seconds[drafts]);
    }

    BatchStep step;
    step.stride = width;
    step.tokens.assign(rows.size() * width, 0);
    step.counts.resize(rows.size());
    step.dead.resize(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const std::int32_t count    = mtp_egress->licensed_counts[index];
        const std::int32_t accepted = mtp_egress->accepted_drafts[index];
        const std::int32_t next     = mtp_egress->next_extents[index];
        if (count <= 0 || count > static_cast<std::int32_t>(width) || accepted < 0 ||
            accepted + 1 != count || next < 0 || next > static_cast<std::int32_t>(drafts)) {
            throw std::runtime_error("MTP batch returned invalid row metadata");
        }
        select(rows[index]);
        if (extents[index] != 0) {
            draft_statistics.record(extents[index], static_cast<std::uint32_t>(accepted));
        }
        for (SpeculativeStats* stats : {&speculative, &lane().stats}) {
            if (extents[index] == 0) {
                stats->fallback_steps += 1;
                continue;
            }
            stats->rounds += 1;
            stats->drafted_tokens += extents[index];
            stats->accepted_tokens += static_cast<std::uint32_t>(accepted);
            for (std::int32_t token = 0; token < accepted; ++token) {
                stats->accepted_per_position[static_cast<std::size_t>(token)] += 1;
            }
        }
        lane().pending_base     = lane().kv_tokens;
        lane().pending_produced = static_cast<std::uint32_t>(count);
        const TokenId* licensed = mtp_egress->licensed_tokens.data() + index * width;
        std::copy(licensed, licensed + width, step.tokens.begin() + static_cast<std::ptrdiff_t>(index * width));
        step.counts[index] = count;
        step.dead[index]   = lane().grammar_dead;
    }
    last_draft_window = drafts;
    return step;
}

void P3Program::accept_speculative(std::uint32_t accepted, bool terminal,
                                   std::optional<std::uint32_t> split) {
    require_bound();
    if (accepted > lane().pending_produced) {
        throw std::logic_error("MTP commit accepted more tokens than the round produced");
    }
    const std::size_t frame = last_draft_window - minimum_draft_window;
    // Both folds read record row b for rows[b], so the lane's row sits at its batch index and
    // the rows before it are zero-column no-ops on distinct spare slots.
    const std::uint32_t batch_row = lane().batch_index;
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<ops::FlashNextPleFoldRow, kMaximumConcurrency> ple_rows{};
    std::int32_t spare = 0;
    for (std::uint32_t row = 0; row <= batch_row; ++row) {
        if (row == batch_row) {
            fold_rows[row] = {lane().slot, lane().slot, static_cast<std::int32_t>(accepted)};
        } else {
            spare += spare == lane().slot ? 1 : 0;
            fold_rows[row] = {spare, spare, 0};
            ++spare;
        }
        ple_rows[row] = {fold_rows[row].source_state_slot, fold_rows[row].destination_state_slot,
                         fold_rows[row].commit_columns};
    }
    mtp_replay_folds[frame].execute(
        std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), batch_row + 1U), device.stream);
    if (!mtp_ple_records.empty() && state_images->ple_store() != nullptr) {
        ops::flash_next_ple_replay_fold(
            mtp_ple_records[frame], *state_images->ple_store(),
            std::span<const ops::FlashNextPleFoldRow>(ple_rows.data(), batch_row + 1U),
            device.stream);
    }
    const bool partial = terminal && accepted != 0 && accepted < lane().pending_produced;
    if (partial) {
        qwen3_8_flash_next::MtpDecodeState& mtp_frame = mtp_frames[frame];
        const std::int32_t selector = static_cast<std::int32_t>(accepted - 1U);
        const auto batch_row        = static_cast<std::int32_t>(lane().batch_index);
        Tensor selector_tensor      = mtp_frame.current_extents.slice(0, batch_row, 1);
        CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, &selector, sizeof(selector),
                                   cudaMemcpyHostToDevice, device.stream));
        Tensor hidden   = mtp_frame.target_hidden.slice(2, batch_row, 1);
        Tensor selected = mtp_frame.target_continuation_hidden.slice(1, batch_row, 1);
        Tensor destinations = mtp_frame.state_destination_slots.slice(0, batch_row, 1);
        ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected, device.stream);
        ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                     device.stream);
        if (mtp_frame.target_mtp_hidden && mtp_frame.target_continuation_mtp_hidden &&
            state_images->mtp_hidden_store() != nullptr) {
            Tensor mtp_hidden   = mtp_frame.target_mtp_hidden->slice(2, batch_row, 1);
            Tensor mtp_selected = mtp_frame.target_continuation_mtp_hidden->slice(1, batch_row, 1);
            ops::speculative_select_accepted_hidden(mtp_hidden, selector_tensor, mtp_selected,
                                                    device.stream);
        ops::scatter(mtp_selected, destinations, *state_images->mtp_hidden_store(),
                     device.stream);
        }
    }
    commit_folds_ready_.record(device.stream);
    if (partial) { device.synchronize(); }
    const std::uint32_t width = last_draft_window + 1U;
    const TokenId* licensed   = mtp_egress->licensed_tokens.data() + lane().batch_index * width;
    if (accepted != 0) {
        lane().request.tokens.insert(lane().request.tokens.end(), licensed,
                                     licensed + static_cast<std::ptrdiff_t>(accepted));
        lane().request.identity.append_generated(accepted, lane().request.rope_delta, split);
    }
    const std::uint32_t frontier = lane().pending_base + accepted;
    lane().kv_tokens               = frontier;
    text_kv_addresses->commit_frontier(lane().kv, frontier);
    mtp_addresses->commit_frontier(lane().mtp_kv, frontier);
    text_kv_addresses->destructive_truncate(lane().kv, frontier);
    mtp_addresses->destructive_truncate(lane().mtp_kv, frontier);
    if (terminal || accepted == 0) {
        lane().draft_count = 0;
    } else {
        const std::uint32_t next = static_cast<std::uint32_t>(mtp_egress->next_extents[lane().batch_index]);
        lane().draft_count       = next;
        for (std::uint32_t step = 0; step < next; ++step) {
            lane().drafts[step] = mtp_egress->next_drafts[step * max_concurrency + lane().batch_index];
        }
    }
    lane().pending.reset();
    lane().pending_produced = 0;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
