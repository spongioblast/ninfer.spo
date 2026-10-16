#pragma once

#include "models/gemma4/frontend/prompt.h"
#include "program/program_types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {
namespace detail {
struct DemotionPlan {};
struct ReleasePlan {};
struct RequestPlan {
    runtime::RequestPlanSummary summary;
    PreparedContextCache cache;
    std::vector<TokenId> tokens;
    std::vector<Gemma4PreparedImage> images;
    std::vector<std::int32_t> image_scatter;
    ResolvedSamplingParameters sampling;
    bool allow_prefix_reuse = false;
};
} // namespace detail

#include "program/program_names.inc"

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept = default;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept = default;
    ~RequestBasePlan() = default;
    RequestBasePlan(const RequestBasePlan&) = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return impl_->summary;
    }
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept { return impl_->cache; }
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const { return {}; }
    [[nodiscard]] std::optional<PrefixShortlistKey> prefix_shortlist_key(std::uint32_t) const noexcept {
        return std::nullopt;
    }

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestPlan> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<detail::RequestPlan> impl_;
    friend class Program;
};

class ContextDemotionBatch {
public:
    [[nodiscard]] bool append(const ContextDemotion&) { return false; }
    [[nodiscard]] bool covers(const ContextRelease&) const { return false; }
    [[nodiscard]] std::optional<ContextDemotion> finish() const { return std::nullopt; }
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage) const { return {}; }
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle>,
                                              std::span<const CheckpointHandle>) const {
        return 0;
    }
};

class ResumeState {
public:
    ResumeState() noexcept = default;
    ResumeState(ResumeState&&) noexcept = default;
    ResumeState& operator=(ResumeState&&) noexcept = default;
    ~ResumeState() = default;
    ResumeState(const ResumeState&) = delete;
    ResumeState& operator=(const ResumeState&) = delete;

    [[nodiscard]] bool has_snapshot() const noexcept { return false; }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t frontier() const noexcept { return 0; }
};

using ContextProgress = program::ContextProgressOf<ResumeState>;

} // namespace ninfer::models::gemma4
