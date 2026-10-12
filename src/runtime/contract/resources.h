#pragma once

#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include "core/uint128.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };

// Physical quantities are sampled from stores, never a scheduler-owned occupancy ledger.
struct ContextResourceUsage {
    std::uint32_t state_slots                                                   = 0;
    std::uint32_t main_kv_pages                                                 = 0;
    std::uint32_t backend_kv_pages                                              = 0;
    std::size_t host_bytes                                                      = 0;
    friend bool operator==(ContextResourceUsage, ContextResourceUsage) noexcept = default;
};

struct ResourceReservation {
    bool reserved = false;
    ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    const Uint128 suffix      = suffix_tokens;
    const Uint128 linear      = Uint128{prefix_tokens} * suffix;
    const Uint128 triangular  = suffix * (suffix + Uint128{1}) / Uint128{2};
    constexpr Uint128 maximum = ~Uint128{0};
    const Uint128 attention =
        triangular > maximum - linear ? maximum : linear + triangular;
    result.attention_pairs = attention > std::numeric_limits<std::uint64_t>::max()
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : static_cast<std::uint64_t>(attention);
    return result;
}

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
    std::uint64_t historical_fork_hits   = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

// Facts the Flash-Next program still names. The V3 engine does not schedule with them.
enum class ContextTransactionStatus : std::uint8_t { InProgress, Published, Aborted };

struct ContextTransactionInProgress {};

enum class ContextTransactionReserveStatus : std::uint8_t { Reserved, Aborted };

enum class FinishDisposition : std::uint8_t { Catalogued, Released };

enum class PreflightStatus : std::uint8_t { Ready, StalePolicyState, InvariantFailure };

enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
    SharedStablePrefix,
    LongAnchor,
};

enum class CheckpointScope : std::uint8_t { Private, Shared };

enum class ReplicaResidency : std::uint8_t { DeviceOnly, HostOnly, Both };

enum class RetentionClass : std::uint8_t { SharedStable, LiveSession, RecentPrivate, Disposable };

struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

struct PlanningOwnerId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningOwnerId, PlanningOwnerId) noexcept =
        default;
};

struct PlanningCandidateId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningCandidateId,
                                                   PlanningCandidateId) noexcept = default;
};

struct FinalScheduleIntent {
    std::span<const std::uint32_t> shared_capture_frontiers;
};

enum class PrivateSourceMode : std::uint8_t { Retain, ConsumeToActive };

enum class VictimDisposition : std::uint8_t { Retained, Evicted };

struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

enum class MaterializationPhysicalStatus : std::uint8_t { Feasible, Infeasible, StructuralInvalid };

inline constexpr std::size_t kContextTransferDirectionCount = 3;
using CoalescedTransferWork = std::array<TransferWork, kContextTransferDirectionCount>;

struct MaterializationMachineWork {
    CoalescedTransferWork pressure_transfers{};
    CoalescedTransferWork candidate_transfers{};
    CoalescedTransferWork optimistic_candidate_transfers{};
    PrefillWork remaining_prefill_work;
    std::uint32_t reused_prompt_tokens = 0;

    [[nodiscard]] friend constexpr bool operator==(const MaterializationMachineWork&,
                                                   const MaterializationMachineWork&) noexcept =
        default;
};

struct CheckpointRecoveryAlternativeWork {
    CoalescedTransferWork transfers{};
    PrefillWork prefill;

    [[nodiscard]] friend constexpr bool operator==(const CheckpointRecoveryAlternativeWork&,
                                                   const CheckpointRecoveryAlternativeWork&) noexcept =
        default;
};

struct IdentityMaterializationAssessment {
    MaterializationPhysicalStatus physical_status = MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode                 = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    bool pressure_may_change_machine_work = false;
    bool expandable                       = false;
    std::uint64_t projection_work         = 0;
    std::uint64_t assessment_digest       = 0;

    [[nodiscard]] friend constexpr bool operator==(const IdentityMaterializationAssessment&,
                                                   const IdentityMaterializationAssessment&) noexcept =
        default;
};

struct PressureConstructionOptionId {
    std::uint32_t cursor_generation = 0;
    std::uint32_t scan_generation   = 0;
    std::uint32_t index             = 0;
};

struct PressureOwnerOutcome {
    PlanningOwnerId owner;
    VictimDisposition disposition     = VictimDisposition::Retained;
    std::uint32_t degradation_units   = 0;
    std::uint32_t dropped_checkpoints = 0;
};

struct PressureCheckpointOutcome {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    bool survives = true;
};

struct PressureOwnerRecoveryGuidance {
    PlanningOwnerId owner;
    CoalescedTransferWork additional_restore{};
};

struct PressurePhysicalGuidance {
    std::uint32_t unsatisfied_constraints   = 0;
    std::uint32_t estimated_remaining_steps = 0;
    std::uint64_t normalized_residual_q20   = 0;
    bool requires_exact_feedback            = false;
};

struct PressureTargetGuidance {
    PressurePhysicalGuidance physical;
    MaterializationMachineWork estimated_machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    PrivateSourceMode source_mode       = PrivateSourceMode::ConsumeToActive;
    std::span<const PressureCheckpointOutcome> checkpoint_changes;
    std::span<const PressureOwnerRecoveryGuidance> recovery_estimates;
    bool recovery_estimate_complete = false;
};

struct PressureConstructionStep {
    std::optional<PressureTargetGuidance> guidance;
    PressureConstructionOptionId option;
    bool exhausted = false;
};

struct PressureCheckpointRecoveryImpact {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    std::span<const CheckpointRecoveryAlternativeWork> target_recovery_work;
    bool survives = true;
};

struct PressureTargetAssessment {
    MaterializationPhysicalStatus physical_status = MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode                 = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    std::span<const PressureCheckpointRecoveryImpact> checkpoint_impacts;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    std::uint64_t projection_work       = 0;
    std::uint64_t assessment_digest     = 0;
    bool expandable                     = false;
    bool root_maximal                   = false;
};

} // namespace ninfer::runtime
