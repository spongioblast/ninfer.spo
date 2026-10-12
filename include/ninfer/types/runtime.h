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

#include "ninfer/types/output.h"

namespace ninfer {

struct GenerationResult {
    // Unique within this Engine instance; diagnostic correlation only.
    std::uint64_t engine_request_id = 0;
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    // Completed initial prefill, excluding cached tokens and separately counted replay work.
    std::uint32_t computed_prefill_tokens = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    GenerationSchedulingStats scheduling;
    GenerationAdmissionStats admission;
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
    std::optional<ConstraintObservation> constraint;
    std::string capture_record;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

struct MemorySummary {
    std::size_t activation_capacity_bytes     = 0;
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t kv_payload_bytes                  = 0;
    // One shared physical Host context backing. Reserved bytes are included in occupied bytes;
    // State/KV occupancy below is a breakdown and must not be added to this ledger again.
    std::uint32_t host_state_capacity_slots = 0;
    std::size_t host_kv_capacity_bytes      = 0;
    std::size_t host_context_capacity_bytes = 0;
    std::size_t host_context_occupied_bytes = 0;
    std::size_t host_context_reserved_bytes = 0;
    std::uint32_t host_state_occupied_slots = 0;
    std::size_t host_kv_occupied_bytes      = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns       = 0;
    std::uint64_t program_submit_ns        = 0;
    std::uint64_t program_post_ns          = 0;
    std::uint64_t engine_commit_output_ns  = 0;
    std::uint64_t engine_maintenance_ns    = 0;
    std::uint64_t device_wait_ns           = 0;
    std::uint64_t constraint_draft_wait_ns = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Full prompt counted once when initial binding succeeds; includes reused tokens.
    std::uint64_t prompt_tokens = 0;
    // All committed output tokens, including the first token and Engine-injected control tokens.
    std::uint64_t generated_tokens = 0;
    // Native verification work, using the same acceptance semantics as SpeculativeStats.
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Initial prompt tokens evaluated by prefill. Reused checkpoint-prefix tokens and replay
    // recomputation are excluded; replayed_tokens separately counts that additional model work.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t paused_requests           = 0;
    std::uint32_t replaying_requests        = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    std::uint64_t preemptions               = 0;
    std::uint64_t snapshot_restores         = 0;
    std::uint64_t replay_restores           = 0;
    std::uint64_t replayed_tokens           = 0;

    std::uint64_t root_selections               = 0;
    std::uint64_t checkpoint_selections         = 0;
    std::uint64_t reused_prompt_tokens          = 0;
    std::uint32_t last_selected_frontier_tokens = 0;

    std::uint64_t state_moves                 = 0;
    std::uint64_t state_forks                 = 0;
    std::uint64_t materialization_state_forks = 0;
    std::uint64_t state_restores              = 0;
    std::uint64_t state_d2h_count             = 0;
    std::uint64_t state_h2d_count             = 0;
    std::uint64_t state_d2d_count             = 0;
    std::uint64_t state_d2h_bytes             = 0;
    std::uint64_t state_h2d_bytes             = 0;
    std::uint64_t state_d2d_bytes             = 0;
    double state_d2h_seconds                  = 0.0;
    double state_h2d_seconds                  = 0.0;
    double state_d2d_seconds                  = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages             = 0;
    std::uint64_t partial_tail_cow_pages           = 0;
    std::uint32_t device_state_occupied_slots      = 0;
    std::uint32_t host_state_occupied_slots        = 0;
    std::uint32_t device_main_kv_occupied_pages    = 0;
    std::uint32_t device_backend_kv_occupied_pages = 0;
    std::size_t host_kv_occupied_bytes             = 0;
    // Unified physical Host context occupancy. Reserved destinations are included in occupied.
    std::size_t host_context_occupied_bytes = 0;
    std::size_t host_context_reserved_bytes = 0;
    // Allocator lifetime high-water mark, including reserved transfer destinations.
    std::size_t host_context_peak_occupied_bytes = 0;
    double actual_context_transfer_seconds       = 0.0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;
};


} // namespace ninfer
