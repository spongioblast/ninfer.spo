#pragma once

#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include "models/qwen3_8_flash_next/program/v3_contract.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_8_flash_next::v3 {

class Program {
public:
    ~Program();

#include "program/program_methods.inc"

    void configure_activation(const EngineOptions& options);
    void activate_steering(const qwen3_8_flash_next::SteeringPack* pack);
    void begin_activation(std::uint32_t lane, std::uint64_t request_id,
                          const qwen3_8_flash_next::PreparedPrompt& prompt,
                          const runtime::ResolvedExecutionOptions& options);
    void select_activation(std::span<const std::uint32_t> lanes);
    [[nodiscard]] std::string flush_activation(std::uint32_t lane);
    [[nodiscard]] ninfer::SteeringState steering_state() const;

    struct Impl;
    [[nodiscard]] static std::unique_ptr<Program> from_parts(std::unique_ptr<Impl> impl);

private:
    explicit Program(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

struct RuntimeTypes {
    using Frontend          = qwen3_8_flash_next::Frontend;
    using PreparedPrompt    = qwen3_8_flash_next::PreparedPrompt;
    using OutputSession     = qwen3_8_flash_next::OutputSession;
    using PublishedOutput   = qwen3_8_flash_next::PublishedOutput;
    using RequestBasePlan   = v3::RequestBasePlan;
    using SequenceHandle    = v3::SequenceHandle;
    using CheckpointHandle  = v3::CheckpointHandle;
    using CheckpointSummary = v3::CheckpointSummary;
    using SourceCandidate   = v3::SourceCandidate;
    using ResumeState       = v3::ResumeState;
    using ExecutionUnit     = v3::ExecutionUnit;
    using ExecutionUnitKind = v3::ExecutionUnitKind;
    using ContextProgress   = v3::ContextProgress;
    using PendingBatch      = v3::PendingBatch;
    using PrefillProgress   = v3::PrefillProgress;
    using ReplayProgress    = v3::ReplayProgress;
    using CommitResult      = v3::CommitResult;
    using DiscardResult     = v3::DiscardResult;
    using FinishResult      = v3::FinishResult;
    using AbortResult       = v3::AbortResult;
    using Program           = v3::Program;
    using CacheSessionKey   = qwen3_8_flash_next::PreparedSessionKey;
};

} // namespace ninfer::models::qwen3_8_flash_next::v3
