#pragma once

// Public types the V3 engine and resource manager read. The shared value types come from
// src/program; this file adds the ones that carry Flash-Next plans.

#include "ninfer/types.h"
#include "program/program_types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::v3 {
namespace detail {
struct RequestBasePlanImpl;
struct DemotionPlan {};
struct ReleasePlan {};
struct ReclaimPlan {};
struct ProgramAccess;
} // namespace detail

#include "program/program_names.inc"

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::RequestBasePlanImpl> impl_;
    friend struct detail::ProgramAccess;
    friend class Program;
};

class ContextDemotionBatch {
public:
    ContextDemotionBatch()                                             = default;
    ContextDemotionBatch(ContextDemotionBatch&&) noexcept              = default;
    ContextDemotionBatch& operator=(ContextDemotionBatch&&) noexcept   = default;
    ~ContextDemotionBatch()                                            = default;
    ContextDemotionBatch(const ContextDemotionBatch&)                  = delete;
    ContextDemotionBatch& operator=(const ContextDemotionBatch&)       = delete;

    [[nodiscard]] bool append(const ContextDemotion&) { return false; }
    [[nodiscard]] bool covers(const ContextRelease&) const { return false; }
    [[nodiscard]] std::optional<ContextDemotion> finish() const { return std::nullopt; }
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle>,
                                              std::span<const CheckpointHandle>) const {
        return 0;
    }
};

class ResumeState {
public:
    ResumeState() noexcept = default;
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;

    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Program;
};

using ContextProgress = program::ContextProgressOf<ResumeState>;

} // namespace ninfer::models::qwen3_8_flash_next::v3
