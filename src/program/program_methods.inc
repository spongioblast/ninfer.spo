// The engine-facing methods of a model's Program, declared once for every model family.
// Include it in the public part of `class Program`, in a namespace that defines the model's
// RequestBasePlan, ResumeState, ContextReclaimPlan, ContextRelease, ContextDemotion,
// ContextProgress, and PreparedPrompt, and imports the shared names (program_names.inc).
// The engine checks the result through GenerationContract in model_instance.cpp.

Program(const Program&)            = delete;
Program& operator=(const Program&) = delete;
Program(Program&&)                 = delete;
Program& operator=(Program&&)      = delete;

[[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                           const runtime::ResolvedExecutionOptions& options);
[[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt, std::uint32_t first_target);
[[nodiscard]] std::optional<SourceCandidate>
inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
               bool consume_source                              = false,
               std::span<const CheckpointHandle> private_points = {},
               std::span<const CheckpointHandle> retired_points = {}) const;
[[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
[[nodiscard]] runtime::ContextResourceUsage
checkpoint_footprint(std::span<const CheckpointHandle>) const;
[[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
[[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
[[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                      const RequestBasePlan& base) const;
[[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                         const RequestBasePlan& base,
                                                         std::uint32_t target) const;
[[nodiscard]] std::uint64_t
checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                         std::span<const CheckpointHandle> surviving) const;
[[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
[[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
[[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
[[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState& paused) const;
// Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
[[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
[[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
[[nodiscard]] std::size_t
release_redundant_host(std::span<const CheckpointHandle> excluded,
                       std::optional<SequenceHandle> pending_backup = std::nullopt);
// Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
[[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
[[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
void release_units(std::span<const SequenceHandle> sequences) noexcept;
[[nodiscard]] BindingReservation
start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
              ResumeState* resume           = nullptr,
              ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
              std::uint32_t resume_tokens   = 1);
[[nodiscard]] bool start_capture(SequenceHandle sequence);
[[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
[[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence);
void skip_capture(SequenceHandle sequence);
// Finite physical actions, each including every optional holder needed for its release.
[[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                              std::span<const CheckpointHandle> excluded,
                                              runtime::ContextResourceUsage shortage) const;
[[nodiscard]] std::vector<ContextRelease>
plan_releases(std::span<const CheckpointHandle> allowed,
              std::span<const CheckpointHandle> excluded,
              runtime::ContextResourceUsage shortage) const;
[[nodiscard]] bool start_demote(const ContextDemotion& plan);
[[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot,
                               runtime::ExecutionTiming* timing = nullptr);
[[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
[[nodiscard]] bool has_context_transaction() const noexcept;
[[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
// A resumed binding retains its complete recovery capacity until committed new progress.
[[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
[[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                              runtime::ExecutionTiming* failed_timing = nullptr,
                                              runtime::TokenMaskProvider* masks       = nullptr);
[[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence,
                                            runtime::ExecutionTiming* failed_timing = nullptr);
[[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                  std::span<const runtime::RoundBudget> budgets,
                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                  runtime::TokenMaskProvider* masks       = nullptr);
// Forced control contributes to counts once. Replay does not call this operation.
[[nodiscard]] runtime::ExecutionTiming
append_forced_tokens(std::span<const SequenceHandle> sequences,
                     std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                     std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                     runtime::ExecutionTiming* failed_timing = nullptr);
[[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                  std::span<const runtime::CommitDecision> decisions,
                                  runtime::CommitObservation observation = runtime::CommitObservation::AllRows,
                                  runtime::ExecutionTiming* failed_timing = nullptr);
[[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
[[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
[[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
void fail_all_cleanup() noexcept;
[[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
[[nodiscard]] MemorySummary memory_summary() const noexcept;
void reset_memory_peaks() noexcept;
