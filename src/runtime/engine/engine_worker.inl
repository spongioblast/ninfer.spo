#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::fail_all_locked(std::exception_ptr error) noexcept
{
    std::deque<std::shared_ptr<Request>> pending;
    {
        std::lock_guard lock(queue_mutex_);
        failed_ = true;
        pending.swap(pending_);
    }
    scheduler_ = Scheduling{};
    const std::shared_ptr<Request> materializing_request =
        materializing_ ? materializing_->request : nullptr;
    const std::shared_ptr<Request> context_request = context_owner_;
    const std::shared_ptr<Request> admission_request =
        admission_decision_ ? admission_decision_->request : nullptr;
    // Native retires all transfer readers before any external checkpoint owner is destroyed.
    instance_.program->fail_all_cleanup();
    materializing_.reset();
    admission_decision_.reset();
    for (auto& decision : capture_decisions_) { decision.reset(); }
    context_owner_.reset();
    for (auto& request : paused_) {
        request->suspended.reset();
        complete_error(request, error);
    }
    paused_.clear();
    resources_.release_all(*instance_.program);
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] != nullptr) {
            complete_error(slots_[lane], error);
            slots_[lane].reset();
        }
    }
    if (materializing_request != nullptr) { complete_error(materializing_request, error); }
    if (context_request != nullptr) { complete_error(context_request, error); }
    if (admission_request != nullptr) { complete_error(admission_request, error); }
    for (const auto& request : pending) { complete_error(request, error); }
    publish_runtime_stats();
}

template <class Instance>
void EngineCore<Instance>::worker_loop() noexcept
{
    for (;;) {
        std::uint64_t cycle_epoch = 0;
        {
            std::unique_lock lock(queue_mutex_);
            if (!stopping_ && pending_.empty() && resident_empty() && paused_.empty() &&
                !instance_.program->has_context_transaction()) {
                queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
            }
            cycle_epoch = work_epoch_;
            if (stopping_) {
                lock.unlock();
                std::scoped_lock execution_lock(execution_mutex_);
                fail_all_locked(std::make_exception_ptr(RequestError(
                    RequestErrorKind::Unavailable, "inference engine is shutting down")));
                return;
            }
        }
        std::unique_lock execution_lock(execution_mutex_);
        bool executed = false;
        try {
            set_host_work_class(HostWorkClass::Control);
            auto boundary = begin_host_phase();
            (void)expire_pending_requests();
            executed = progress_context_transaction(boundary);
            if (!instance_.program->has_context_transaction()) {
                for (const auto& request : slots_) {
                    if (request && request->capture_pending) {
                        reserve_active_capture(request, request->post_capture_state);
                        break;
                    }
                }
            }
            (void)settle_terminal_requests(boundary);
            cancel_active_requests(snapshot_cancellations(), boundary);
            for (auto it = paused_.begin();
                 !instance_.program->has_context_transaction() && it != paused_.end();) {
                const auto request = *it;
                if (!request->cancelled.load(std::memory_order_acquire)) {
                    ++it;
                    continue;
                }
                request->suspended.reset();
                it = paused_.erase(it);
                complete_detached_cancelled(request);
                scheduler_.capacity_released();
            }
            const bool recovering =
                std::any_of(slots_.begin(), slots_.end(),
                            [](const auto& r) { return r && r->recovery_pending; }) ||
                (materializing_ && materializing_->resumed);
            if (paused_.empty() && !context_owner_ && !recovering) {
                scheduler_.paused_queue_empty();
            }
            if (!recovering &&
                ((admission_decision_ && admission_decision_->restoring) ||
                 scheduler_.should_restore(!paused_.empty(), resident_empty()))) {
                (void)try_admit_one(true);
                executed |= progress_context_transaction(boundary);
            }
            reserve_resident_units();
            // A failed restore leaves spare resources available to fresh requests.
            // An in-flight restoration/reclaim keeps its own candidate until settled.
            if (!admission_decision_ || !admission_decision_->restoring) {
                if (!admission_decision_ && consume_admission_check()) {
                    ++admission_generation_;
                    scheduler_.begin_admission_scan();
                }
                if (admission_decision_ || scheduler_.admission_scan_pending()) {
                    (void)try_admit_one(false);
                }
            }
            if (instance_.program->has_context_transaction()) {
                executed |= progress_context_transaction(boundary);
                if (!instance_.program->has_context_transaction()) { reserve_resident_units(); }
            }
            const auto cancelled = snapshot_cancellations();
            auto controls = scheduler_.build_control_membership(slots_, max_concurrency_);
            ControlMembership selected_control;
            selected_control.row_stride = controls.row_stride;
            for (std::size_t row = 0; row < controls.size; ++row) {
                if (!runnable_units_[controls.lanes[row]] ||
                    slots_[controls.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                    continue;
                }
                selected_control.lanes[selected_control.size]       = controls.lanes[row];
                selected_control.sequences[selected_control.size++] = controls.sequences[row];
                const auto begin = controls.tokens.begin() + row * controls.row_stride;
                selected_control.tokens.insert(selected_control.tokens.end(), begin,
                                               begin + controls.row_stride);
            }
            controls = std::move(selected_control);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            if (!controls.empty()) {
                set_host_work_class(HostWorkClass::Control);
                run_control_batch(controls);
                executed = true;
            }
            // Control rows become decode-ready next cycle, after reserving their new unit.
            auto decode = scheduler_.build_round_membership(slots_, max_concurrency_);
            {
                RoundMembership selected;
                for (std::size_t row = 0; row < decode.size; ++row) {
                    if (!runnable_units_[decode.lanes[row]] ||
                        slots_[decode.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                        continue;
                    }
                    if (std::find(controls.lane_span().begin(), controls.lane_span().end(),
                                  decode.lanes[row]) != controls.lane_span().end()) {
                        continue;
                    }
                    selected.lanes[selected.size]     = decode.lanes[row];
                    selected.sequences[selected.size] = decode.sequences[row];
                    selected.budgets[selected.size++] = decode.budgets[row];
                }
                decode = selected;
            }
            std::int64_t decode_ns = 0;
            if (!decode.empty()) {
                set_host_work_class(HostWorkClass::Decode, decode.lane_span());
                const auto decode_started = Clock::now();
                run_decode_round(decode, cancelled);
                decode_ns = static_cast<std::int64_t>(elapsed_ns(decode_started, Clock::now()));
                executed  = true;
            }
            if (instance_.program->has_context_transaction()) {
                set_host_work_class(HostWorkClass::Control);
                auto progress_boundary = begin_host_phase();
                executed |= progress_context_transaction(progress_boundary);
                if (!instance_.program->has_context_transaction()) {
                    // Control/Decode permits were consumed. Only the still-unexecuted
                    // Prefill/Replay turn can receive a new permit in this round.
                    reserve_resident_units(ReservationScope::Prefill);
                }
                finish_engine_phase(progress_boundary, EngineHostPhase::Boundary);
            }
            auto prefill_slots = slots_;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (!runnable_units_[lane] ||
                    (prefill_slots[lane] &&
                     prefill_slots[lane]->cancelled.load(std::memory_order_acquire))) {
                    prefill_slots[lane].reset();
                }
            }
            const auto lane = scheduler_.next_prefill(prefill_slots, max_concurrency_);
            if (!decode.empty()) { scheduler_.decode_round_ran(decode_ns, lane.has_value()); }
            if (lane && !scheduler_.prefill_yields_to_decode(
                            !decode.empty(), slots_[*lane]->prefill_remaining_tokens())) {
                const auto request = slots_[*lane];
                const auto unit_started = Clock::now();
                set_host_work_class(HostWorkClass::Prefill);
                if (request->is_replaying()) {
                    ProgramCallScope call(*this);
                    auto progress = instance_.program->advance_replay(*request->sequence,
                                                                      &call.failed_timing());
                    call.finish(progress.timing);
                    if (!request->first_output_timing) {
                        record_execution_work(request->replay_work, progress.timing);
                    }
                    request->replayed_tokens += progress.processed_tokens;
                    cumulative_stats_.replayed_tokens += progress.processed_tokens;
                    if (progress.capture_ready) {
                        reserve_active_capture(request, EngineRequestState::Replay);
                    }
                    if (progress.complete) {
                        ++cumulative_stats_.replay_restores;
                        ++request->replay_restores;
                        request->model_state = request->resume_phase;
                        observe_scheduling(request,
                                           GenerationSchedulingTransition::ReplayComplete);
                    }
                } else {
                    run_prefill_step(*lane, snapshot_cancellations());
                }
                scheduler_.prefill_executed(*lane, max_concurrency_);
                scheduler_.prefill_unit_ran(
                    static_cast<std::int64_t>(elapsed_ns(unit_started, Clock::now())));
                update_recovery(request);
                executed = true;
            }
            publish_runtime_stats();
        } catch (...) {
            auto error = std::current_exception();
            fail_all_locked(error);
            return;
        }
        const bool transaction_in_flight = instance_.program->has_context_transaction();
        execution_lock.unlock();
        if (!executed && transaction_in_flight) {
            // A context transfer finishes on the device. A condition wait on Windows
            // lasts a scheduler tick, so poll the transfer at a fine interval instead.
            precise_sleep_ns(100'000);
            continue;
        }
        if (!executed) {
            // A bare timed wait drops a submit or cancel that arrived after the cycle started.
            // The epoch taken at the top of the cycle makes that wakeup stick. The timeout stays
            // as the backstop for pending deadlines and device transfers, which do not notify.
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return stopping_ || work_epoch_ != cycle_epoch;
            });
        }
    }
}

} // namespace ninfer::runtime
