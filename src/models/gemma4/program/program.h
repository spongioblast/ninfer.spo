#pragma once

#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/contract.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::gemma4 {

class Program {
public:
    // One lane with `capacity` KV positions; prefill runs in chunks of `prefill_chunk` tokens.
    Program(LoadedText& model, std::uint32_t capacity, std::uint32_t prefill_chunk,
            cudaStream_t stream);
    ~Program();

    // KV bytes per position of the global layers; the sliding layers' ring is in fixed_bytes.
    [[nodiscard]] static std::size_t kv_bytes_per_token(const LoadedText& model);
    // Device bytes the Program needs besides the global layers' KV.
    [[nodiscard]] static std::size_t fixed_bytes(const LoadedText& model,
                                                 std::uint32_t prefill_chunk);

#include "program/program_methods.inc"

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct RuntimeTypes {
    using Frontend          = gemma4::Frontend;
    using PreparedPrompt    = gemma4::PreparedPrompt;
    using OutputSession     = gemma4::OutputSession;
    using PublishedOutput   = gemma4::PublishedOutput;
    using RequestBasePlan   = gemma4::RequestBasePlan;
    using SequenceHandle    = gemma4::SequenceHandle;
    using CheckpointHandle  = gemma4::CheckpointHandle;
    using CheckpointSummary = gemma4::CheckpointSummary;
    using SourceCandidate   = gemma4::SourceCandidate;
    using ResumeState       = gemma4::ResumeState;
    using ExecutionUnit     = gemma4::ExecutionUnit;
    using ExecutionUnitKind = gemma4::ExecutionUnitKind;
    using ContextProgress   = gemma4::ContextProgress;
    using PendingBatch      = gemma4::PendingBatch;
    using PrefillProgress   = gemma4::PrefillProgress;
    using ReplayProgress    = gemma4::ReplayProgress;
    using CommitResult      = gemma4::CommitResult;
    using DiscardResult     = gemma4::DiscardResult;
    using FinishResult      = gemma4::FinishResult;
    using AbortResult       = gemma4::AbortResult;
    using Program           = gemma4::Program;
    using CacheSessionKey   = std::uint64_t;
};

} // namespace ninfer::models::gemma4
