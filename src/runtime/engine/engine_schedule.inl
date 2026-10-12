#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::observe_scheduling(const std::shared_ptr<Request>& request,
                        GenerationSchedulingTransition transition, Clock::time_point now )
{
    if (!request->observation.scheduling) { return; }
    if (now == Clock::time_point{}) { now = Clock::now(); }
    const GenerationSchedulingObservation observation{
        .transition              = transition,
        .route                   = request->recovery_route,
        .engine_request_id       = request->id,
        .steady_ns               = elapsed_ns(Clock::time_point{}, now),
        .elapsed_ns              = elapsed_ns(request->submitted, now),
        .preemption_index        = request->preemption_count,
        .global_prefill_tokens   = cumulative_stats_.computed_prefill_tokens,
        .global_decode_tokens    = cumulative_stats_.committed_decode_tokens,
        .global_replayed_tokens  = cumulative_stats_.replayed_tokens,
        .request_prefill_tokens  = request->computed_prompt_tokens,
        .request_decode_tokens   = request->committed_decode_tokens,
        .request_replayed_tokens = request->replayed_tokens,
    };
    {
        std::lock_guard lock(request->mutex);
        request->events.emplace_back(
            std::make_unique<GenerationSchedulingObservation>(observation));
    }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::update_recovery(const std::shared_ptr<Request>& request, bool released )
{
    if (!request->recovery_pending) { return; }
    request->recovery_pending = !released && request->sequence &&
                                instance_.program->recovery_pending(*request->sequence);
    if (!request->recovery_pending) {
        if (!released) {
            observe_scheduling(request, GenerationSchedulingTransition::RecoveryComplete);
        }
        scheduler_.capacity_released();
        request_admission_check();
    }
}

template <class Instance>
void EngineCore<Instance>::reserve_resident_units(ReservationScope scope )
{
    runnable_units_.fill(false);
    std::optional<std::uint32_t> recovering;
    const auto oldest_resident = [&] {
        std::optional<std::uint32_t> found;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || !request->sequence || request->is_model_finished()) { continue; }
            if (!found || request->id < slots_[*found]->id) { found = lane; }
        }
        return found;
    };
    std::optional<std::uint32_t> oldest = oldest_resident();
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (!request || !request->sequence || request->is_model_finished()) { continue; }
        if (request->recovery_pending) {
            if (recovering) { throw std::logic_error("multiple protected recoveries"); }
            recovering = lane;
        }
    }
    const auto prefill = recovering && (slots_[*recovering]->is_prefilling() ||
                                        slots_[*recovering]->is_replaying())
                             ? recovering
                             : scheduler_.next_prefill(slots_, max_concurrency_);
    std::vector<std::uint32_t> candidates;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (!request || request->capture_pending || !request->sequence ||
            request->cancelled.load(std::memory_order_acquire) ||
            instance_.program->context_blocks(*request->sequence)) {
            continue;
        }
        if (prefill == lane || (scope == ReservationScope::Round &&
                                (request->is_control_ready() || request->is_decode_ready()))) {
            candidates.push_back(lane);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
        if (a == b) { return false; }
        if (recovering == a || recovering == b) { return recovering == a; }
        return slots_[a]->id < slots_[b]->id;
    });
    std::optional<std::uint32_t> blocked;
    for (const auto lane : candidates) {
        const auto request = slots_[lane];
        // Reclaiming for an earlier row may already have paused this candidate.
        if (!request || !request->sequence || request->capture_pending ||
            request->cancelled.load(std::memory_order_acquire) ||
            instance_.program->context_blocks(*request->sequence) ||
            !(request->is_control_ready() || request->is_decode_ready() ||
              request->is_prefilling() || request->is_replaying())) {
            continue;
        }
        const ExecutionUnit unit{
            *request->sequence,
            request->is_control_ready()  ? UnitKind::Control
            : request->is_decode_ready() ? UnitKind::Decode
            : request->is_replaying()    ? UnitKind::Replay
                                         : UnitKind::Prefill,
            request->is_control_ready()
                ? static_cast<std::uint32_t>(request->output.pending_control_tokens().size())
            : request->is_decode_ready()
                ? (request->recovery_pending ? 1U
                                             : request->output.model_token_budget_remaining(
                                                   request->budget->remaining()))
                : 0U};
        std::optional<typename ResourceManagement::ReclaimCursor> cursor;
        for (;;) {
            const auto result = instance_.program->reserve_units({&unit, 1});
            if (result) {
                runnable_units_[lane] = true;
                break;
            }
            if (instance_.program->has_context_transaction()) { break; }
            if (!cursor) { cursor.emplace(resources_.begin_reclaim(*instance_.program)); }
            if (reclaim(result.shortage, false, {}, &*cursor)) {
                if (!instance_.program->has_context_transaction()) { continue; }
                break;
            }
            if (recovering == lane) {
                throw std::logic_error("protected recovery exceeded its Native reservation");
            }
            if (!recovering && oldest == lane) {
                // Only the oldest required unit can displace younger residents. Other
                // failed rows wait or yield their own lane; they cannot block ready rows.
                // The same shortage has already exhausted cache and paused snapshots.
                if (pause_reclaim_victim(request->id)) { break; }
                // Failing this request must not latch the engine. A throw here reaches
                // fail_all_locked and every later request gets unavailable until restart.
                if (request->sequence && request->lane && request->lane->value == lane) {
                    auto aborted                = instance_.program->abort(*request->sequence);
                    request->generation_timings = aborted.timings;
                    request->speculative_stats  = std::move(aborted.speculative);
                }
                // Not Overloaded (429): nothing younger is left to pause, so a client retry
                // would prefill again and fail the same way.
                complete_error(request, std::make_exception_ptr(RequestError(
                                            RequestErrorKind::ContextLengthExceeded,
                                            "request needs more context memory than the engine "
                                            "can free for it")));
                remove_completed_slot(lane);
                // The right to displace younger rows passes to the oldest resident still here.
                oldest = oldest_resident();
                break;
            }
            if (oldest != lane && (!blocked || request->id > slots_[*blocked]->id)) {
                blocked = lane;
            }
            break;
        }
    }
    if (blocked && !instance_.program->has_context_transaction()) {
        (void)pause_resident(*blocked);
    }
    if (prefill && runnable_units_[*prefill] && !prepare_semantic_capture(slots_[*prefill])) {
        runnable_units_[*prefill] = false;
    }
    // A pause may have selected an already licensed younger row. Its permit belongs
    // to Native cleanup, and it must not enter this cycle's compact execution batch.
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (!slots_[lane] || slots_[lane]->model_state == EngineRequestState::Pausing ||
            (slots_[lane]->sequence &&
             instance_.program->context_blocks(*slots_[lane]->sequence))) {
            runnable_units_[lane] = false;
        }
    }
}

template <class Instance>
void EngineCore<Instance>::run_decode_round(const RoundMembership& membership,
                      const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start)
{
    nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                   static_cast<std::uint64_t>(membership.size));
    ProgramCallScope program_call(*this);
    RoundMasks masks;
    bool constrained = false;
    for (std::size_t row = 0; row < membership.size; ++row) {
        masks.outputs[row] = &slots_[membership.lane_span()[row]]->output;
        constrained |= masks.outputs[row]->constrained();
    }
    if constexpr (requires { instance_.program->select_activation(membership.lane_span()); }) {
        instance_.program->select_activation(membership.lane_span());
    }
    auto pending = instance_.program->decode(
        membership.sequence_span(), membership.budget_span(), &program_call.failed_timing(),
        constrained ? &masks : nullptr);
    program_call.finish(pending.execution_timing());
    commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
    publish_runtime_stats();
}

template <class Instance>
void EngineCore<Instance>::run_control_batch(const ControlMembership& membership)
{
    nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                    static_cast<std::uint64_t>(membership.size));
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    if (membership.empty() || membership.row_stride == 0 ||
        membership.tokens.size() !=
            static_cast<std::size_t>(membership.row_stride) * membership.size) {
        throw std::logic_error("thinking control membership is invalid");
    }

    std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
    std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
    bool generated_staged         = false;
    const auto rollback_generated = [&]() {
        if (!generated_staged) { return; }
        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request != nullptr) {
                try {
                    request->output.discard_preview();
                } catch (...) {}
            }
            if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                request->generated.resize(generated_sizes[row]);
            }
        }
        generated_staged = false;
    };

    for (std::size_t row = 0; row < membership.size; ++row) {
        const auto& request = slots_[membership.lanes[row]];
        if (request == nullptr) {
            throw std::logic_error("thinking control membership lost its request");
        }
        generated_sizes[row] = request->generated.size();
    }
    generated_staged = true;
    try {
        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            if (request == nullptr || !request->is_control_ready() ||
                request->capture_pending || !request->budget || !request->sequence ||
                !request->lane || request->lane->value != lane) {
                throw std::logic_error("thinking control row lost its active request");
            }
            const std::span<const TokenId> tokens =
                std::span<const TokenId>(membership.tokens)
                    .subspan(row * membership.row_stride, membership.row_stride);
            const OutputDecision decision =
                request->output.preview_control(tokens, request->budget->remaining());
            if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                decision.continuation != ContinuationAction::Decode ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("target control preview returned an invalid decision");
            }
            prefix_execution_splits[row] = decision.prefix_execution_split_after;
            if (request->generated.size() > request->generated.capacity() ||
                tokens.size() > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error(
                    "admission did not reserve thinking-control token capacity");
            }
            request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
        }
        phase.pause_range();
        ProgramCallScope program_call(*this);
        const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
            membership.sequence_span(), membership.tokens, membership.row_stride,
            std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                          membership.size),
            &program_call.failed_timing());
        program_call.finish(timing);
        phase.resume_range();
    } catch (...) {
        rollback_generated();
        throw;
    }
    generated_staged = false;

    ++cumulative_stats_.host_work.control_units;
    for (const std::uint32_t lane : membership.lane_span()) {
        ++slots_[lane]->host_timing.control_units;
        slots_[lane]->committed_decode_tokens += membership.row_stride;
    }
    cumulative_stats_.committed_decode_tokens += membership.size * membership.row_stride;

    for (std::size_t row = 0; row < membership.size; ++row) {
        const std::uint32_t lane = membership.lanes[row];
        const auto& request      = slots_[lane];
        request->budget->commit(membership.row_stride);
        auto timing = record_committed_output(request, membership.row_stride);
        append_output(request, request->output.commit_preview(), std::move(timing), &phase);
        update_recovery(request);
        request->model_state = EngineRequestState::DecodeReady;
    }
    publish_runtime_stats();
}

} // namespace ninfer::runtime
