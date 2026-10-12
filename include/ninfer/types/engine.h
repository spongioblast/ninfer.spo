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

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
    BFloat16KeyValue,
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP 1..5; DFlash and DFlash2 1..15 (query width K+1).
    // Zero with adaptive_draft_tokens lets an MTP target choose K for every round.
    std::uint32_t draft_tokens = 0;
    bool adaptive_draft_tokens = false;
    ProposalHead proposal_head = ProposalHead::Full;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostContextPin,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

struct ContextCacheOptions {
    // Controls cross-request history reads and writes. Request pause/replay resources remain
    // available when history is disabled.
    bool enabled = true;
    // Extra Device StateImage slots beyond max_concurrency. Defaults to max_concurrency.
    std::optional<std::uint32_t> device_state_slots;
    // Shared Host quota for StateImages, KV, pause snapshots and in-flight destinations. Native
    // startup defaults to 8 GiB plus eight Host StateImages using the model's actual layout.
    // This does not bound total process RAM.
    // Engine::options() returns both resolved capacities after construction.
    std::optional<std::size_t> host_capacity_bytes;
    // Flash-Next host stores. Zero leaves them unallocated until a caller sets them.
    std::size_t host_kv_capacity_bytes = 0;
    std::uint32_t host_state_slots     = 0;
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

enum class PleResidency : std::uint8_t {
    // Resident on Windows, where a mapped PLE row read is a hard page fault unless the file
    // happens to be in the standby cache. Mapped elsewhere, where the page cache serves it.
    PlatformDefault,
    Mapped,
    Resident,
};

inline constexpr std::size_t kDefaultHostRamBudgetBytes = 200ULL << 30;

struct HostResidencyOptions {
    PleResidency ple     = PleResidency::PlatformDefault;
    bool ple_large_pages = false;
    std::optional<int> numa_node;
    std::size_t host_ram_budget_bytes = kDefaultHostRamBudgetBytes;
};

struct SteeringState {
    std::string pack_sha;
    double strength          = 0;
    std::uint64_t generation = 0;
    int rank                 = 0;
    std::vector<int> layers;
};

struct EngineOptions {
    std::filesystem::path capture_path;
    std::vector<std::string> capture_sites = {"prompt_last", "completion_last"};
    std::filesystem::path steering_pack;
    HostResidencyOptions host_residency;
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes    = 256ULL * 1024 * 1024;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
};

} // namespace ninfer
