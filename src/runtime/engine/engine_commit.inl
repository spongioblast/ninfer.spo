#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                    bool decode_round,
                    const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start)
{
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    const std::size_t row_count = lane_indices.size();
    if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
        (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
        pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
        (void)instance_.program->abort_pending(std::move(pending));
        throw std::logic_error("pending batch returned an invalid ragged layout");
    }

    std::array<CommitDecision, kMaximumConcurrency> decisions{};
    std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
    std::array<ContinuationAction, kMaximumConcurrency> continuations{};
    std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
    std::array<bool, kMaximumConcurrency> cancelled{};
    std::array<bool, kMaximumConcurrency> failed{};
    bool generated_staged = false;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
    std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
    std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
    std::size_t terminal_count    = 0;
    const auto rollback_generated = [&]() {
        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = slots_[lane_indices[row]];
            if (request != nullptr) {
                try {
                    request->output.discard_preview();
                } catch (...) {}
            }
            if (generated_staged && request != nullptr &&
                request->generated.size() >= generated_sizes[row]) {
                request->generated.resize(generated_sizes[row]);
            }
        }
        generated_staged = false;
    };
    try {
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane = lane_indices[row];
            const auto& request      = slots_[lane];
            if (request == nullptr || !request->sequence || !request->lane ||
                request->lane->value != lane || !request->budget) {
                throw std::logic_error("pending row has no active Engine request");
            }
            if (decode_round) { ++request->host_timing.decode_rounds; }
            cancelled[row] = cancelled_at_unit_start[lane];
            failed[row]    = !cancelled[row] && pending.constraint_failed(row);
            const std::int32_t raw_count =
                pending.row_counts().empty() ? 1 : pending.row_counts()[row];
            if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                throw std::logic_error("pending row has an invalid licensed extent");
            }
            const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
            const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                 static_cast<std::size_t>(count));
            generated_sizes[row]      = request->generated.size();
            if (failed[row]) {
                decisions[row] = CommitDecision{.terminal = true, .failed = true};
                continue;
            }
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                decisions[row] = CommitDecision{
                    .accepted_tokens = 0,
                    .terminal        = true,
                    .cancelled       = true,
                };
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview_model(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count) ||
                (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            decisions[row] = CommitDecision{
                .accepted_tokens              = decision.accepted_tokens,
                .terminal                     = decision.finished(),
                .cancelled                    = false,
                .prefix_execution_split_after = decision.prefix_execution_split_after,
            };
            finish_reasons[row] = decision.finish_reason;
            continuations[row]  = decision.continuation;
        }
        generated_staged = true;
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t accepted = decisions[row].accepted_tokens;
            if (accepted == 0) { continue; }
            const auto& request = slots_[lane_indices[row]];
            if (request->generated.size() > request->generated.capacity() ||
                accepted > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error("admission did not reserve generated-token capacity");
            }
            const auto first = pending.tokens().begin() +
                               static_cast<std::ptrdiff_t>(row * pending.row_stride());
            request->generated.insert(request->generated.end(), first,
                                      first + static_cast<std::ptrdiff_t>(accepted));
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        rollback_generated();
        const auto discarded = instance_.program->abort_pending(std::move(pending));
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("failed pending batch could not be discarded");
        }
        std::rethrow_exception(error);
    }

    std::optional<typename ModelContract::CommitResult> committed_storage;
    try {
        phase.pause_range();
        ProgramCallScope program_call(*this);
        auto committed = instance_.program->commit(
            std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
            CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
        program_call.finish(committed.timing);
        committed_storage.emplace(std::move(committed));
        phase.resume_range();
    } catch (...) {
        rollback_generated();
        throw;
    }
    generated_staged = false;
    auto& committed  = *committed_storage;
    if (committed.row_count != row_count) {
        throw std::logic_error("Runtime commit result is not row aligned");
    }
    for (std::size_t row = 0; row < row_count; ++row) {
        const CommitDisposition expected =
            failed[row]               ? CommitDisposition::FailedReleased
            : cancelled[row]          ? CommitDisposition::CancelledReleased
            : decisions[row].terminal ? CommitDisposition::Finishable
                                      : CommitDisposition::Active;
        if (committed.rows[row].disposition != expected) {
            throw std::logic_error("Runtime commit row disposition is invalid");
        }
        if (committed.capture_ready[row] &&
            (decode_round || expected != CommitDisposition::Active)) {
            throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
        }
    }
    const bool terminal_in_batch = std::any_of(
        decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
        [](const CommitDecision& decision) { return decision.terminal; });

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto& request  = slots_[lane_indices[row]];
        auto& observed       = request->speculative_stats;
        const auto& counters = committed.rows[row].speculative_counters;
        cumulative_stats_.speculative_rounds += counters.rounds - observed.rounds;
        cumulative_stats_.speculative_draft_tokens +=
            counters.drafted_tokens - observed.drafted_tokens;
        cumulative_stats_.speculative_accepted_tokens +=
            counters.accepted_tokens - observed.accepted_tokens;
        cumulative_stats_.speculative_fallback_steps +=
            counters.fallback_steps - observed.fallback_steps;
        observed.rounds          = counters.rounds;
        observed.drafted_tokens  = counters.drafted_tokens;
        observed.accepted_tokens = counters.accepted_tokens;
        observed.fallback_steps  = counters.fallback_steps;
        if (cancelled[row] || failed[row]) {
            request->generation_timings = committed.rows[row].timings;
            request->speculative_stats  = std::move(committed.rows[row].speculative);
        }
    }

    if (decode_round) {
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += row_count;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (!cancelled[row]) {
                cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                slots_[lane_indices[row]]->committed_decode_tokens +=
                    decisions[row].accepted_tokens;
            }
        }
    }

    try {
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane     = lane_indices[row];
            const auto& request          = slots_[lane];
            const std::uint32_t accepted = decisions[row].accepted_tokens;
            if (failed[row]) {
                complete_error(request, std::make_exception_ptr(
                                            RequestError(RequestErrorKind::ConstraintDeadEnd,
                                                         "grammar has no valid next token")));
                remove_completed_slot(lane);
                continue;
            }
            if (!cancelled[row]) { request->budget->commit(accepted); }
            update_recovery(request, cancelled[row]);
            auto published = request->output.commit_preview();
            auto timing    = record_committed_output(request, accepted);
            append_output(request, std::move(published), std::move(timing), &phase);
            if (decisions[row].terminal) {
                if (cancelled[row]) {
                    terminal_requests[terminal_count] = request;
                    terminal_lanes[terminal_count]    = lane;
                    terminal_reasons[terminal_count]  = finish_reasons[row];
                    ++terminal_count;
                } else {
                    request->model_state     = EngineRequestState::ModelFinished;
                    request->terminal_reason = finish_reasons[row];
                }
            } else if (committed.capture_ready[row]) {
                if (!request->is_prefilling()) {
                    throw std::logic_error("prompt-frontier capture lost its prefill owner");
                }
                const EngineRequestState post_capture_state =
                    continuations[row] == ContinuationAction::ApplyTargetControl
                        ? EngineRequestState::ControlReady
                        : EngineRequestState::DecodeReady;
                if (terminal_in_batch) {
                    instance_.program->skip_capture(*request->sequence);
                    request->model_state = post_capture_state;
                } else {
                    reserve_active_capture(request, post_capture_state);
                }
                committed.capture_ready[row] = false;
                if (!request->capture_pending) {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            } else {
                request->model_state =
                    continuations[row] == ContinuationAction::ApplyTargetControl
                        ? EngineRequestState::ControlReady
                        : EngineRequestState::DecodeReady;
            }
        }
    } catch (...) {
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
        throw;
    }
    phase.finish();
    for (std::size_t index = 0; index < terminal_count; ++index) {
        complete_success(terminal_requests[index], terminal_reasons[index]);
        remove_completed_slot(terminal_lanes[index]);
    }
}

template <class Instance>
void EngineCore<Instance>::reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next)
{
    request->model_state        = next;
    request->capture_pending    = true;
    request->post_capture_state = next;
    if (instance_.program->has_context_transaction()) { return; }
    bool started = instance_.program->start_capture(*request->sequence);
    if (!started) {
        instance_.program->skip_capture(*request->sequence);
        request->capture_pending = false;
        return;
    }
    context_owner_ = request;
}

template <class Instance>
bool EngineCore<Instance>::prepare_semantic_capture(const std::shared_ptr<Request>& request)
{
    auto& decision = capture_decisions_[request->lane->value];
    if (decision && decision->sequence != *request->sequence) { decision.reset(); }
    if (instance_.program->capture_is_input(*request->sequence) && !decision) {
        (void)resources_.recycle_input(*instance_.program, request->continuation_owner);
    }
    // A reclaim that reports a change but leaves the same shortage would spin here and
    // hold the engine lock, so later requests never start. Skipping the capture loses
    // one reuse point and lets the request continue.
    int reclaims = 0;
    while (const auto capture = instance_.program->prepare_capture(*request->sequence)) {
        if (capture->reserved) {
            decision.reset();
            return true;
        }
        if (instance_.program->has_context_transaction()) { return false; }
        if (++reclaims > 32) {
            instance_.program->skip_capture(*request->sequence);
            decision.reset();
            return true;
        }
        const auto admission = resources_.capture_admission(
            *instance_.program, request->continuation_owner, *request->base_plan,
            capture->frontier,
            instance_.program->capture_is_input(*request->sequence) &&
                !capture->shortage.main_kv_pages && !capture->shortage.backend_kv_pages);
        if (!decision || decision->frontier != capture->frontier) {
            decision.emplace(
                CaptureDecision{*request->sequence, capture->frontier,
                                resources_.begin_reclaim(*instance_.program, admission)});
        }
        const auto usage = instance_.program->physical_usage();
        if (capture->host_bytes && capture->host_bytes <= usage.capacity.host_bytes) {
            if (capture->host_bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
                (void)instance_.program->release_redundant_host({}, request->sequence);
            }
            if (const auto victims = resources_.host_victims(
                    *instance_.program, capture->host_bytes, std::nullopt, {}, {}, admission,
                    &decision->reclaim)) {
                resources_.commit_host_victims(*instance_.program, *victims, decision->reclaim);
                if (!victims->empty()) { scheduler_.capacity_released(); }
                if (const auto retry = instance_.program->prepare_capture(*request->sequence);
                    retry && retry->reserved) {
                    decision.reset();
                    return true;
                }
            }
        }
        const auto progress =
            resources_.reclaim(*instance_.program, capture->shortage, {}, decision->reclaim);
        if (progress == ReclaimProgress::Transferring) { return false; }
        if (progress == ReclaimProgress::Changed) { continue; }
        instance_.program->skip_capture(*request->sequence);
        decision.reset();
    }
    decision.reset();
    return true;
}

template <class Instance>
void
EngineCore<Instance>::resolve_prefill_progress(const std::shared_ptr<Request>& request,
                         typename ModelContract::PrefillProgress&& progress,
                         const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start)
{
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    ++cumulative_stats_.host_work.prefill_units;
    ++request->host_timing.prefill_units;
    const PrefillGraphTally graph_tally = consume_prefill_graph_tally();
    request->host_timing.prefill_graph_hits += graph_tally.hits;
    request->host_timing.prefill_graph_captures += graph_tally.captures;
    request->host_timing.prefill_graph_eager_first += graph_tally.eager_first;
    request->host_timing.prefill_graph_ineligible += graph_tally.ineligible;
    request->host_timing.prefill_graph_audited += graph_tally.audited;
    request->host_timing.prefill_graph_audit_mismatches += graph_tally.audit_mismatches;
    if (graph_tally.last_reason != 0) {
        request->host_timing.prefill_graph_skip_reason = graph_tally.last_reason;
    }
    cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
    if (!request->admitted_begin) {
        throw std::logic_error("prefill progress has no committed admission summary");
    }
    const BeginSummary& begin = *request->admitted_begin;
    if (begin.reused_prompt_tokens > begin.prompt_tokens) {
        throw std::logic_error("admitted prefix exceeds its prompt");
    }
    const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
    if (request->computed_prompt_tokens > suffix_tokens ||
        progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
        throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
    }
    const auto computed_begin = begin.reused_prompt_tokens + request->computed_prompt_tokens;
    request->computed_prompt_tokens += progress.processed_prompt_tokens;
    if (!request->cancelled.load(std::memory_order_acquire)) {
        resources_.observe_committed(*request->base_plan, request->id, computed_begin,
                                     computed_begin + progress.processed_prompt_tokens);
    }
    if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
        throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
    }
    if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
    if (progress.capture_ready) {
        if (progress.complete || progress.pending) {
            throw std::logic_error("prefill capture offer overlaps prompt completion");
        }
        reserve_active_capture(request, EngineRequestState::Prefill);
        return;
    }
    if (!progress.complete) { return; }
    if (!request->lane || !progress.pending) {
        throw std::logic_error("completed prefill has no lane or pending token");
    }
    if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
        throw std::logic_error("runtime Begin summary differs from committed admission");
    }
    const std::uint32_t lane = request->lane->value;
    request->begin           = progress.summary;
    const std::array<std::uint32_t, 1> lanes{lane};
    phase.finish();
    commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
    progress.pending.reset();
}

} // namespace ninfer::runtime
