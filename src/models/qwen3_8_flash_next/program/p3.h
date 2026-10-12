#pragma once

// Flash-Next lanes: prefill, decode, prefix checkpoints, pause, and ledger replay. A finished
// request can keep its KV and state as a prefix checkpoint.

#include "core/device.h"
#include "models/qwen3_8_flash_next/execution/schedule.h"
#include "models/qwen3_8_flash_next/execution/text_context.h"
#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include "models/qwen3_8_flash_next/impl/runtime/layouts.h"
#include "models/qwen3_8_flash_next/impl/ple_table.h"
#include "models/qwen3_8_flash_next/impl/runtime/prefix_identity.h"
#include "models/qwen3_8_flash_next/execution/prefill_graph.h"
#include "models/qwen3_8_flash_next/impl/runtime/state_image_store.h"
#include "models/qwen3_8_flash_next/execution/vision_context.h"
#include "models/qwen3_8_flash_next/program/graph_family.h"
#include "program/storage/kv_address_space.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "ninfer/ops/flash_next_ple.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

class P3Program {
public:
    struct Request {
        runtime::RequestPlanSummary summary;
        std::vector<TokenId> tokens;
        ops::SamplingConfig sampling;
        std::optional<qwen3_8_flash_next::PreparedPromptData> media;
        std::optional<VisionPrefillPlan> vision;
        // Prompt positions where a unit must stop, below the prompt end. Cold and resumed
        // prefills then use the same chunk boundaries.
        std::vector<std::uint32_t> split_frontiers;
        // Subset of the splits that become a shared checkpoint when the unit lands on them.
        std::vector<std::uint32_t> capture_frontiers;
        // Media, positions, and steering behind tokens; grows with every committed token.
        ResidentPrefixIdentity identity;
        // M-RoPE offset of every token after a multimodal prompt.
        std::int32_t rope_delta = 0;
    };

    struct Step {
        bool complete                  = false;
        bool capture_ready             = false;
        TokenId token                  = 0;
        std::uint32_t processed_tokens = 0;
        std::vector<TokenId> licensed;
        std::int32_t licensed_count = 0;
        std::uint32_t stride        = 0;
        std::uint32_t grammar_dead  = 0;
    };

    struct BatchStep {
        std::uint32_t stride = 1;
        std::vector<TokenId> tokens;
        std::vector<std::int32_t> counts;
        std::vector<std::uint32_t> dead;
    };

    // One catalogued prefix. The generation changes when the slot is consumed or released.
    struct CachePoint {
        std::uint32_t index      = 0;
        std::uint64_t generation = 0;
    };

    P3Program(const LoadedModelData& model_in, const SequencePlanImpl& plan, DeviceContext& device_in);

    ~P3Program();

    [[nodiscard]] Request plan_request(const qwen3_8_flash_next::PreparedPromptData& prompt,
                                       const runtime::ResolvedExecutionOptions& options) const;

    void bind(Request request, std::uint32_t row);

    [[nodiscard]] Step advance_prefill(runtime::TokenMaskProvider* masks = nullptr);

    void select(std::uint32_t row);

    void note_prefill_seconds(double seconds) noexcept { lane().prefill_seconds += seconds; }

    // Each lane records the whole round, matching the batch-1 rate the bench reports.
    void note_decode_seconds(std::span<const std::uint32_t> rows, double seconds) noexcept;

    [[nodiscard]] double noted_prefill_seconds() const noexcept { return lane().prefill_seconds; }
    [[nodiscard]] double noted_decode_seconds() const noexcept { return lane().decode_seconds; }

    [[nodiscard]] BatchStep decode_rows(std::span<const std::uint32_t> rows,
                                        std::span<const std::uint32_t> budgets,
                                        runtime::TokenMaskProvider* masks = nullptr);

    // split is the output policy's execution split inside the accepted span (a reasoning close).
    // It goes into the prefix identity so the next turn's rendered history matches.
    void accept_speculative(std::uint32_t accepted, bool terminal,
                            std::optional<std::uint32_t> split = std::nullopt);
    [[nodiscard]] const SpeculativeStats& speculative_stats() const noexcept { return speculative; }

    // Counters of the selected lane's request. They survive a pause and its resume.
    [[nodiscard]] const SpeculativeStats& request_speculative_stats() const noexcept {
        return lane().stats;
    }
    [[nodiscard]] std::size_t cuda_graph_allowance_bytes() const noexcept;
    // Zero is a valid position. The namespace-level checked_i32 in layouts.cpp rejects it.
    [[nodiscard]] static std::int32_t checked_i32(std::uint32_t value, const char* label);
    void setup_mtp(const SequencePlanImpl& plan, DeviceSpan backing);
    void prepare_mtp_graphs(std::uint32_t graph_capacity);
    [[nodiscard]] BatchStep decode_mtp(std::span<const std::uint32_t> rows,
                                      std::span<const std::uint32_t> budgets,
                                      runtime::TokenMaskProvider* masks);
    [[nodiscard]] std::uint32_t apply_mask(runtime::TokenMaskProvider* masks, std::size_t compact_row,
                                           std::uint32_t lane, std::span<const TokenId> drafts);
    [[nodiscard]] std::vector<float> causal_score(qwen3_8_flash_next::PreparedPromptData prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::uint32_t select_mtp_drafts() const;

    void accept_pending(std::optional<std::uint32_t> split = std::nullopt);

    void drop_pending() noexcept { lane().pending.reset(); }

    [[nodiscard]] bool has_pending() const noexcept { return lane().pending.has_value(); }

    // Forced control tokens join the ledger. KV covers every token except the last one, which
    // the next decode consumes. A split marks where a later prompt may diverge from them.
    void append_forced(std::span<const TokenId> extra, std::optional<std::uint32_t> split);

    void release_lane() noexcept;

    // Moves the live lane into a prefix checkpoint when the request asked to publish one.
    // The checkpoint covers every token except the last, which the next prefill consumes.
    [[nodiscard]] std::optional<CachePoint> catalog_lane() noexcept;
    [[nodiscard]] bool cache_valid(CachePoint point) const noexcept;
    [[nodiscard]] std::uint32_t cache_frontier(CachePoint point) const;
    [[nodiscard]] std::span<const TokenId> cache_prefix(CachePoint point,
                                                        std::uint32_t frontier) const;
    // A match must leave at least one prompt token to prefill, since sampling needs its logits.
    [[nodiscard]] bool cache_matches(CachePoint point,
                                     const qwen3_8_flash_next::PreparedPromptData& prompt) const;
    [[nodiscard]] bool release_cache(CachePoint point) noexcept;
    void resume_lane(Request request, CachePoint point);
    // Copies a shared checkpoint into this lane and leaves the checkpoint in the catalog.
    void fork_lane(Request request, CachePoint point);
    // Freezes the live prefix at the cursor into a shared checkpoint and keeps the lane going.
    [[nodiscard]] std::optional<CachePoint> capture_prefix();
    // The next shared frontier this lane will reach, if it still has one.
    [[nodiscard]] std::optional<std::uint32_t> next_capture_frontier() const;
    // Drops the offered frontier, or the next one when nothing is offered, so the engine's
    // capture loop moves on instead of asking for the same shortage again.
    void skip_capture() noexcept;
    // A new checkpoint needs one free device state slot besides the lane's own.
    [[nodiscard]] bool state_capture_ready() const noexcept;
    [[nodiscard]] runtime::CheckpointRole cache_role(CachePoint point) const;
    // KV pages a request can grow into: prompt, output budget, and one MTP window past it.
    // A full-width reservation per lane would leave no room for a second lane while a long
    // prompt prefills, so a short request would wait for the whole prompt.
    struct KvEntitlement {
        std::uint32_t text = 0;
        std::uint32_t mtp  = 0;
    };
    [[nodiscard]] KvEntitlement kv_entitlement(const Request& request) const noexcept;
    // Pages and state slots a lane bind lacks for this request.
    [[nodiscard]] runtime::ContextResourceUsage
    bind_shortage(const Request& request, std::optional<CachePoint> resume = std::nullopt,
                  bool copy = false) const;
    [[nodiscard]] runtime::ContextResourceUsage cache_footprint(CachePoint point) const;
    // What releasing the record frees: pages no other address holds, and its state slot.
    [[nodiscard]] runtime::ContextResourceUsage cache_release_quote(CachePoint point) const;
    // capacity and occupied, with reservations counted as occupied.
    [[nodiscard]] std::pair<runtime::ContextResourceUsage, runtime::ContextResourceUsage>
    pool_usage() const noexcept;
    [[nodiscard]] std::optional<CachePoint> park_point() const noexcept;

    struct ReplayStep {
        std::uint32_t processed = 0;
        bool complete           = false;
    };

    struct Park {
        bool occupied = false;
        bool device   = false;
        std::uint64_t generation = 0;
        CachePoint point;
        Request request;
        bool prefill_done = false;
        std::uint32_t cursor = 0;
        std::uint32_t kv_tokens = 0;
        std::array<TokenId, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> drafts{};
        std::uint32_t draft_count = 0;
        std::uint32_t mtp_extent  = 0;
        SpeculativeStats stats;
    };

    // Parks the live lane on device. The caller restores it or releases it.
    [[nodiscard]] std::optional<std::uint64_t> park_lane();
    [[nodiscard]] std::optional<std::uint64_t> park_as_ledger();
    [[nodiscard]] bool park_valid(std::uint64_t generation) const noexcept;
    [[nodiscard]] std::uint32_t park_frontier(std::uint64_t generation) const noexcept;
    void restore_park(std::uint64_t generation);
    void begin_replay(std::uint64_t generation);
    [[nodiscard]] ReplayStep advance_replay();
    [[nodiscard]] bool park_ledger(std::uint64_t generation) const noexcept;
    [[nodiscard]] SpeculativeStats park_stats(std::uint64_t generation) const;
    void release_park(std::uint64_t generation) noexcept;

    // No-ops unless a capture path or a steering pack was configured. Ordinary mixes keep a null
    // activation pointer.
    void configure_activation(const EngineOptions& options);
    void activate_steering(const qwen3_8_flash_next::SteeringPack* pack);
    void begin_activation(std::uint32_t lane, std::uint64_t request_id, std::span<const TokenId> prompt,
                          const runtime::ResolvedExecutionOptions& options);
    void select_activation(std::span<const std::uint32_t> lanes);
    [[nodiscard]] std::string flush_activation(std::uint32_t lane);
    [[nodiscard]] SteeringState steering_state() const;

private:
    struct Lane {
        bool bound = false;
        bool prefill_done = false;
        bool replaying = false;
        bool replay_resume_prefill = false;
        std::uint32_t replay_target = 0;
        Request request;
        std::optional<TokenId> pending;
        program::KVAddressSpaceHandle mtp_kv;
        std::array<TokenId, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> drafts{};
        std::uint32_t draft_count = 0;
        std::uint32_t mtp_extent  = 0;
        std::uint32_t pending_base = 0;
        std::uint32_t pending_produced = 0;
        std::uint32_t batch_index = 0;
        std::uint32_t grammar_dead = 0;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        std::size_t vision_handoff_peak = 0;
        program::KVAddressSpaceHandle kv;
        qwen3_8_flash_next::detail::StateImageHandle state;
        // Set after a capture or a shared bind, until the next prefill chunk commits the fork.
        bool fork_pending = false;
        qwen3_8_flash_next::detail::StateImageHandle fork_source;
        // Shared frontiers at or below this are never offered or prepared again.
        std::uint32_t capture_skipped_through = 0;
        // A unit stopped on a shared frontier and offered it; cleared by the capture or a skip.
        bool capture_offered = false;
        std::int32_t slot      = 0;
        std::uint32_t cursor   = 0;
        std::uint32_t kv_tokens = 0;
        double prefill_seconds = 0;
        double decode_seconds  = 0;
        SpeculativeStats stats;
    };

    struct CacheRecord {
        bool occupied = false;
        std::uint64_t generation = 0;
        std::vector<TokenId> tokens;
        program::KVAddressSpaceHandle kv;
        program::KVAddressSpaceHandle mtp;
        qwen3_8_flash_next::detail::StateImageHandle state;
        std::uint32_t frontier = 0;
        runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
        ResidentPrefixIdentity identity;
    };

    void require_bound() const;

    [[nodiscard]] VisionPrefillPlan
    plan_vision(const qwen3_8_flash_next::PreparedPromptData& prompt,
                std::optional<qwen3_8_flash_next::PreparedPromptData>& media) const;
    void attach_vision();

    [[nodiscard]] CacheRecord* find_cache(CachePoint point) noexcept;
    [[nodiscard]] const CacheRecord* find_cache(CachePoint point) const noexcept;
    [[nodiscard]] CachePoint seal_lane();
    void install_request(Request& request);
    // Releases whatever a failed bind, resume, or copy already holds and empties the lane.
    void undo_bind() noexcept;
    // Program counters with the totals zeroed: the start of a new request's counters.
    [[nodiscard]] SpeculativeStats fresh_stats() const;
    // Adds tokens to the lane's penalty counts when its request uses penalties.
    void count_tokens(std::span<const TokenId> tokens);
    // Maps both KV pools far enough for the lane to write tokens below this frontier.
    void map_kv(std::uint32_t tokens);
    // Where the next prompt chunk must stop: the prompt end or the next split above the cursor.
    [[nodiscard]] std::uint32_t prompt_chunk_limit() const noexcept;
    void release_frozen(program::KVAddressSpaceHandle kv,
                        program::KVAddressSpaceHandle mtp,
                        qwen3_8_flash_next::detail::StateImageHandle state) noexcept;

    void prefill_gap(std::uint32_t target);

    void set_i32(Tensor& tensor, std::int32_t value);

    void copy_token();

    void prepare_ordinary_graphs();

    void gather_last_ple(std::size_t byte_offset);

    [[nodiscard]] schedule::ExecutionCore execution_core();

    [[nodiscard]] schedule::PrefillContext make_prefill();

    [[nodiscard]] schedule::OrdinaryBatchContext make_decode();

    const LoadedModelData& model;
    DeviceContext& device;
    CudaCompletionEvent commit_folds_ready_;
    std::uint32_t capacity;
    std::uint32_t kv_capacity;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_window = 0;
    std::uint32_t minimum_draft_window = 0;
    WorkspacePlan workspace_plan;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<ops::Bf16GemmContext> bf16_gemm;
    std::unique_ptr<HostWorkerPool> ple_gather_workers;
    PinnedHostBuffer round_host;
    PinnedHostBuffer ordinary_host;
    std::optional<PinnedHostBuffer> flash_ple_host;
    std::unique_ptr<qwen3_8_flash_next::DecoderState> decoder;
    std::unique_ptr<program::LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<program::KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<program::LogicalKVPageStore> mtp_kv_pages;
    std::unique_ptr<program::KVAddressSpaceStore> mtp_addresses;
    std::unique_ptr<qwen3_8_flash_next::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_8_flash_next::detail::StateImageStore> state_store;
    qwen3_8_flash_next::RoundState io;
    std::unique_ptr<qwen3_8_flash_next::ActivationControl> activation;
    SteeringState active_steering_;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    std::optional<Tensor> flash_decode_ple;
    std::optional<Tensor> flash_ple_records;
    std::optional<Tensor> score_hidden;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    bool causal_scoring = false;
    std::optional<Tensor> mtp_prefill_hidden;
    std::optional<GdnReplayRecords> replay_records;
    std::vector<GdnReplayRecords> mtp_replay_records;
    std::vector<ops::GdnReplayFoldPlan> mtp_replay_folds;
    std::vector<Tensor> mtp_ple_records;
    std::vector<qwen3_8_flash_next::MtpDecodeState> mtp_frames;
    std::vector<DecodeGraphFamily> mtp_graphs;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_8_flash_next::MtpDecodeIngress* mtp_ingress = nullptr;
    qwen3_8_flash_next::MtpDecodeEgress* mtp_egress   = nullptr;
    std::array<TokenId, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t last_draft_window = 0;
    MtpDraftStatistics draft_statistics;
    SpeculativeStats speculative;
    std::array<double, qwen3_8_flash_next::kMtpDecodeMaximumDrafts + 1> measured_seconds{};
    std::array<std::uint32_t, qwen3_8_flash_next::kMtpDecodeMaximumDrafts + 1> measured_samples{};
    qwen3_8_flash_next::OrdinaryDecodeIngress* ordinary_ingress = nullptr;
    qwen3_8_flash_next::OrdinaryDecodeEgress* ordinary_egress   = nullptr;
    std::uint32_t max_concurrency = 1;
    std::uint32_t page_capacity = 0;
    std::size_t graph_allowance_bytes = 0;
    std::uint32_t grammar_mask_words = 0;
    std::optional<DeviceBuffer> grammar_masks_device;
    std::optional<PinnedHostBuffer> grammar_masks_host;
    std::vector<GraphExecutionProfile> ordinary_profiles;
    DecodeGraphFamily ordinary_graphs;
    std::unique_ptr<LowLatencyStreamWait> decode_wait;
    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::uint32_t focus_ = 0;

    Lane& lane() noexcept { return lanes_[focus_]; }
    const Lane& lane() const noexcept { return lanes_[focus_]; }
    // One record per state image: every record freezes one image, so the catalog is never full
    // while the context cache still sees a free state slot it could reclaim for.
    std::vector<CacheRecord> cache_slots;
    std::uint64_t cache_clock = 0;
    Park parked;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
