#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ninfer/types/error.h"

namespace ninfer {

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

enum class GenerationSchedulingTransition : std::uint8_t {
    PauseStarted,
    Paused,
    RestoreStarted,
    Restored,
    ReplayComplete,
    RecoveryComplete,
    SnapshotRevoked,
    Terminal,
};

enum class GenerationRecoveryRoute : std::uint8_t { None, Snapshot, Replay };

// Sparse lifecycle observations captured by the Engine worker and delivered by wait() on the
// consumer thread. Counter deltas between boundaries, less the request's own deltas, measure
// other requests' completed work during recovery. Decode includes forced control tokens.
struct GenerationSchedulingObservation {
    GenerationSchedulingTransition transition = GenerationSchedulingTransition::PauseStarted;
    GenerationRecoveryRoute route             = GenerationRecoveryRoute::None;
    std::uint64_t engine_request_id           = 0;
    std::uint64_t steady_ns                   = 0;
    std::uint64_t elapsed_ns                  = 0;
    std::uint64_t preemption_index            = 0;
    std::uint64_t global_prefill_tokens       = 0;
    std::uint64_t global_decode_tokens        = 0;
    std::uint64_t global_replayed_tokens      = 0;
    std::uint64_t request_prefill_tokens      = 0;
    std::uint64_t request_decode_tokens       = 0;
    std::uint64_t request_replayed_tokens     = 0;
};

using GenerationSchedulingObserver = std::function<void(const GenerationSchedulingObservation&)>;

// Emitted once on the first accepted model/control token, delivered by wait() in either consumer
// mode. Submission elapsed includes queueing and binding; preparation is reported separately.
struct GenerationFirstTokenObservation {
    double prepare_seconds              = 0.0;
    double elapsed_since_submit_seconds = 0.0;
    double queue_wait_seconds           = 0.0;
};

using GenerationFirstTokenObserver = std::function<void(const GenerationFirstTokenObservation&)>;

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live output timings and prompt progress require a
// Streaming consumer; phase timings and scheduling observations also support Aggregate consumers.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
    // Optional for either consumer mode. No scheduling events are retained when unset.
    GenerationSchedulingObserver scheduling;
    GenerationFirstTokenObserver first_token;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // Prompt wall time begins at the successful initial binding attempt and ends at the first
    // accepted output token. Generation spans the first through last accepted output token and
    // therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                    = 0.0;
    double engine_boundary_exposed_seconds       = 0.0;
    double program_submit_exposed_seconds        = 0.0;
    double program_post_exposed_seconds          = 0.0;
    double engine_commit_output_exposed_seconds  = 0.0;
    double engine_maintenance_exposed_seconds    = 0.0;
    double device_wait_exposed_seconds           = 0.0;
    double constraint_draft_wait_exposed_seconds = 0.0;
    double decode_host_exposed_seconds           = 0.0;
    double decode_device_wait_exposed_seconds    = 0.0;
    std::uint64_t prefill_units                  = 0;
    std::uint64_t decode_rounds                  = 0;
    std::uint64_t control_units                  = 0;
    std::uint64_t prefill_graph_hits             = 0;
    std::uint64_t prefill_graph_captures         = 0;
    std::uint64_t prefill_graph_eager_first      = 0;
    std::uint64_t prefill_graph_ineligible       = 0;
    std::uint64_t prefill_graph_audited          = 0;
    std::uint64_t prefill_graph_audit_mismatches = 0;
    std::uint32_t prefill_graph_skip_reason      = 0;
};

// Request-owned scheduling observations. Restore counters count completed restorations;
// replayed_tokens counts tokens actually recomputed, including previously generated output;
// these tokens are separate from initial prompt prefill and delivered output. paused_ns covers
// pause preparation, waiting and binding restoration, excluding replay computation.
// Transfer bytes count request-owned payload submitted for binding, capture,
// pause and restore; background cache maintenance remains in the Engine-wide counters.
struct GenerationSchedulingStats {
    std::uint64_t preemptions          = 0;
    std::uint64_t snapshot_restores    = 0;
    std::uint64_t replay_restores      = 0;
    std::uint64_t replayed_tokens      = 0;
    std::uint64_t paused_ns            = 0;
    std::uint64_t device_to_host_bytes = 0;
    std::uint64_t host_to_device_bytes = 0;
};

// Request-owned execution work. GPU stream intervals overlap Host submission and completion
// waits; they are evidence about Device work, not an additional wall-time component.
struct GenerationWorkTiming {
    double submit_seconds = 0.0;
    double wait_seconds   = 0.0;
    double post_seconds   = 0.0;
    double gpu_seconds    = 0.0;
};

struct GenerationTransferTiming {
    std::uint64_t bytes = 0;
    double seconds      = 0.0;
};

// Frozen immediately before the first nonempty public output delta is published. This boundary
// differs from the first accepted model token and from the client's first HTTP output. Elapsed
// starts at Engine submission, excluding Frontend preparation. Initial queue, initial binding,
// paused time and the remaining resident interval partition this Engine wall time.
struct GenerationFirstOutputTiming {
    double elapsed_seconds         = 0.0;
    double initial_binding_seconds = 0.0;
    GenerationEngineTiming engine;
    GenerationWorkTiming prefill;
    GenerationWorkTiming replay;
    GenerationSchedulingStats scheduling;
    std::uint32_t computed_prefill_tokens = 0;
    // Resources: State, Main KV, Backend KV. Directions: D2H, H2D, D2D.
    // Completed request-owned transfers only; background reclamation remains Engine-wide.
    std::array<std::array<GenerationTransferTiming, 3>, 3> context_transfers{};
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class ConstraintCacheAccess : std::uint8_t { Hit, Built, Waited };
enum class ConstraintOutputBranch : std::uint8_t { Undecided, Content, Tools };

// State describes the committed output language, including an assistant continuation prefix.
// Work includes speculative lookahead that was subsequently rolled back. Times are subintervals
// of existing request timings, not extra latency to add to them.
struct ConstraintObservation {
    ConstraintOutputBranch branch   = ConstraintOutputBranch::Undecided;
    bool complete                   = false;
    bool terminated                 = false;
    ConstraintCacheAccess cache     = ConstraintCacheAccess::Hit;
    bool timings_collected          = false;
    double prepare_seconds          = 0.0;
    double mask_seconds             = 0.0;
    double matcher_seconds          = 0.0;
    std::uint64_t mask_positions    = 0;
    std::uint64_t mask_upload_bytes = 0;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    Checkpoint,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

enum class AdmissionFallbackReason : std::uint8_t {
    None,
    SourceInvalid,
    SourceRevoked,
    CostChanged,
    CapacityLimit,
    IsolatedCapacity,
};

struct GenerationAdmissionStats {
    std::uint32_t preferred_reused_tokens = 0;
    // Subset of initial queue wait, not additional TTFT. Includes waiting for a lane
    // after a useful source has been selected.
    double source_wait_seconds              = 0.0;
    std::uint32_t revoked_checkpoints       = 0;
    AdmissionFallbackReason fallback_reason = AdmissionFallbackReason::None;
};

} // namespace ninfer
