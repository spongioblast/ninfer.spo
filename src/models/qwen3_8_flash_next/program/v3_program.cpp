#include "models/qwen3_8_flash_next/program/v3_program.h"
#include "models/qwen3_8_flash_next/program/p3.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::v3 {
namespace {

using Core = ::ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime::
    P3Program;

[[noreturn]] void not_yet(const char* what) {
    throw std::logic_error(std::string("V3 P3: not yet: ") + what);
}

[[nodiscard]] runtime::ExecutionTiming elapsed_since(std::chrono::steady_clock::time_point start) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    runtime::ExecutionTiming timing;
    timing.device_wait_ns = ns > 0 ? static_cast<std::uint64_t>(ns) : 0;
    return timing;
}

// Same digest on a checkpoint and on a new prompt, so the prefix index can walk them.
[[nodiscard]] std::array<std::uint64_t, 2> prefix_digest(std::span<const TokenId> tokens) noexcept {
    std::uint64_t high = 14695981039346656037ull;
    std::uint64_t low  = 0x6c62272e07bb0142ull;
    for (const TokenId id : tokens) {
        const auto word = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id));
        high ^= word;
        high *= 1099511628211ull;
        low ^= word + 0x9e3779b97f4a7c15ull;
        low *= 0x100000001b3ull;
    }
    return {high, low};
}

constexpr std::uint32_t kPrefixIdentityTag = 1;

[[nodiscard]] PrefixShortlistKey prefix_key(std::span<const TokenId> tokens,
                                            std::uint32_t frontier) noexcept {
    return {.digests      = prefix_digest(tokens),
            .frontier     = frontier,
            .identity_tag = kPrefixIdentityTag};
}

[[nodiscard]] Core::CachePoint cache_point(CheckpointHandle handle) noexcept {
    return {.index = handle.index, .generation = handle.generation};
}

[[nodiscard]] bool frontier_splits_vision(const PreparedPromptData& prompt,
                                          std::uint32_t frontier) noexcept {
    for (const VisionItem& item : prompt.vision_items) {
        for (const TokenSpan& span : item.token_spans) {
            const auto begin = static_cast<std::uint32_t>(span.begin);
            const auto end   = static_cast<std::uint32_t>(span.begin + span.count);
            if (begin < frontier && frontier < end) { return true; }
        }
    }
    return false;
}

void add_frontier(std::vector<std::uint32_t>& out, const PreparedPromptData& prompt,
                  std::uint32_t frontier, std::uint32_t prompt_tokens) {
    if (frontier == 0 || frontier >= prompt_tokens) { return; }
    if (frontier_splits_vision(prompt, frontier)) { return; }
    if (std::find(out.begin(), out.end(), frontier) != out.end()) { return; }
    out.push_back(frontier);
}

} // namespace

struct detail::RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    PreparedContextCache cache;
    PreparedPromptData prompt;
    Core::Request request;
    bool allow_prefix_reuse = false;
    std::vector<std::uint32_t> capture_frontiers;
};

struct detail::ProgramAccess {
    [[nodiscard]] static RequestBasePlan make_plan(std::shared_ptr<RequestBasePlanImpl> impl) {
        return RequestBasePlan(std::move(impl));
    }

    [[nodiscard]] static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                                      std::uint64_t epoch) {
        return program::HandleAccess::make_sequence(owner, lane, epoch);
    }

    [[nodiscard]] static bool same(SequenceHandle handle, const void* owner,
                                   std::uint64_t epoch) noexcept {
        return program::HandleAccess::owner(handle) == owner &&
               program::HandleAccess::epoch(handle) == epoch;
    }

    [[nodiscard]] static std::uint32_t lane_of(SequenceHandle handle) noexcept {
        return program::HandleAccess::lane(handle).value;
    }

    [[nodiscard]] static PendingBatch make_pending(const void* owner, std::uint64_t transaction,
                                                   std::span<const TokenId> tokens,
                                                   runtime::ExecutionTiming timing,
                                                   std::span<const std::int32_t> counts = {},
                                                   std::uint32_t stride                 = 0,
                                                   std::size_t rows                     = 1,
                                                   std::span<const std::uint8_t> failed = {}) {
        PendingBatch batch = program::HandleAccess::make_pending(
            owner, transaction, {}, rows, tokens, counts,
            stride != 0 ? stride : static_cast<std::uint32_t>(tokens.size()), timing);
        for (std::size_t index = 0; index < failed.size() && index < kMaximumConcurrency; ++index) {
            program::HandleAccess::constraint_failed(batch, index, failed[index] != 0);
        }
        return batch;
    }

    [[nodiscard]] static bool owns(const PendingBatch& batch, const void* owner,
                                   std::uint64_t transaction) noexcept {
        return program::HandleAccess::owner(batch) == owner &&
               program::HandleAccess::transaction(batch) == transaction && batch.row_count() >= 1;
    }
};

struct Program::Impl {
    Core core;
    std::uint32_t capacity      = 0;
    std::uint32_t kv_capacity   = 0;
    std::uint32_t prefill_chunk = 1;
    int device_index            = 0;
    KvCacheStorage kv_cache     = KvCacheStorage::BFloat16;
    struct Live {
        bool occupied = false;
        SequenceHandle handle;
        std::uint64_t epoch = 0;
        runtime::BeginSummary begin;
    };

    bool binding                  = false;
    bool pausing                  = false;
    bool capturing                = false;
    CheckpointHandle captured{};
    bool replay_bind              = false;
    std::uint64_t park_generation = 0;
    std::uint32_t binding_row     = 0;
    std::uint64_t next_epoch      = 1;
    runtime::BeginSummary begin;
    std::array<Live, kMaximumConcurrency> live{};
    std::vector<std::uint32_t> pending_rows;
    std::vector<TokenId> pending_tokens;
    std::vector<std::int32_t> pending_counts;
    std::uint64_t pending_transaction = 0;
    runtime::ExecutionTiming pending_timing;

    Impl(const typename ::ninfer::models::qwen3_8_flash_next_125b_a6b::detail::Variant::ModelView& model,
         const qwen3_8_flash_next::detail::SequencePlanImpl<::ninfer::models::qwen3_8_flash_next_125b_a6b::detail::Variant>& plan,
         DeviceContext& device)
        : core(model, plan, device), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
          prefill_chunk(plan.prefill_chunk == 0 ? 1 : plan.prefill_chunk),
          device_index(device.device),
          // The plan stores BF16 as its key-and-value layout; report the option that chose it.
          kv_cache(plan.kv_storage == KvCacheStorage::BFloat16KeyValue ? KvCacheStorage::BFloat16
                                                                       : plan.kv_storage) {}

    [[nodiscard]] bool owns(SequenceHandle sequence) const noexcept {
        const std::uint32_t row = detail::ProgramAccess::lane_of(sequence);
        if (row >= live.size()) { return false; }
        const Live& slot = live[row];
        return slot.occupied && detail::ProgramAccess::same(sequence, this, slot.epoch);
    }

    void clear_slot(std::uint32_t row) noexcept {
        if (row < live.size()) { live[row] = {}; }
    }
};

struct ResumeState::Impl {
    Program::Impl* owner     = nullptr;
    std::uint64_t generation = 0;

    ~Impl() {
        if (owner != nullptr) { owner->core.release_park(generation); }
    }
};

ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;

bool ResumeState::has_snapshot() const noexcept {
    return impl_ != nullptr && impl_->owner != nullptr &&
           impl_->owner->core.park_valid(impl_->generation);
}

std::optional<CheckpointHandle> ResumeState::snapshot_handle() const noexcept {
    return std::nullopt;
}

std::uint32_t ResumeState::frontier() const noexcept {
    if (!has_snapshot()) { return 0; }
    return impl_->owner->core.park_frontier(impl_->generation);
}

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    return impl_->summary;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept { return impl_->cache; }

std::vector<std::uint32_t> RequestBasePlan::capture_frontiers() const {
    return impl_ == nullptr ? std::vector<std::uint32_t>{} : impl_->capture_frontiers;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prompt.token_ids.size()) {
        return std::nullopt;
    }
    return prefix_key(std::span<const TokenId>(impl_->prompt.token_ids.data(), frontier), frontier);
}

Program::Program(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() = default;

std::unique_ptr<Program> Program::from_parts(std::unique_ptr<Impl> impl) {
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    if (!prompt) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data    = PreparedPromptAccess::take(std::move(prompt));
    auto request = impl_->core.plan_request(data, options);
    const bool reuse = options.allow_prefix_reuse && data.identity.reusable;
    request.summary.publish_continuation = reuse;
    auto plan                            = std::make_shared<detail::RequestBasePlanImpl>();
    plan->summary                        = request.summary;
    plan->allow_prefix_reuse             = reuse;
    plan->cache                          = data.context_cache;
    const std::uint32_t prompt_tokens = request.summary.prompt_tokens;
    std::vector<std::uint32_t> splits;
    std::vector<std::uint32_t> captures;
    for (const std::uint32_t frontier : data.identity.rewrite_execution_frontiers) {
        add_frontier(splits, data, frontier, prompt_tokens);
    }
    if (data.identity.rewrite_checkpoint) {
        add_frontier(splits, data, data.identity.rewrite_checkpoint->frontier, prompt_tokens);
        add_frontier(captures, data, data.identity.rewrite_checkpoint->frontier, prompt_tokens);
    }
    for (const PreparedCacheOpportunity& opportunity : data.context_cache.opportunities) {
        add_frontier(splits, data, opportunity.frontier, prompt_tokens);
        add_frontier(captures, data, opportunity.frontier, prompt_tokens);
    }
    std::sort(splits.begin(), splits.end());
    std::sort(captures.begin(), captures.end());
    request.split_frontiers = splits;
    if (request.summary.publish_continuation) { request.capture_frontiers = captures; }
    plan->prompt            = std::move(data);
    plan->capture_frontiers = std::move(captures);
    plan->request           = std::move(request);
    return detail::ProgramAccess::make_plan(std::move(plan));
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    if (!prompt) { throw std::invalid_argument("prepared prompt is empty"); }
    return impl_->core.causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target);
}

std::optional<SourceCandidate>
Program::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                        bool consume, std::span<const CheckpointHandle>,
                        std::span<const CheckpointHandle>) const {
    if (!base.impl_) { return std::nullopt; }
    std::uint32_t reused = 0;
    if (checkpoint) {
        if (!base.impl_->allow_prefix_reuse || !valid_checkpoint(*checkpoint) ||
            !impl_->core.cache_matches(cache_point(*checkpoint), base.impl_->prompt)) {
            return std::nullopt;
        }
        reused = impl_->core.cache_frontier(cache_point(*checkpoint));
    }
    SourceCandidate candidate;
    candidate.checkpoint     = checkpoint;
    candidate.reused_tokens  = reused;
    candidate.consume_source = checkpoint && consume;
    const std::uint32_t suffix = base.summary().prompt_tokens - reused;
    candidate.remaining_work = runtime::make_prefill_work(reused, suffix, 0, 0, impl_->prefill_chunk);
    return candidate;
}

PrefixShortlistKey Program::checkpoint_key(CheckpointHandle handle, std::uint32_t frontier) const {
    if (!valid_checkpoint(handle)) { throw std::logic_error("V3 P5 checkpoint is stale"); }
    return prefix_key(impl_->core.cache_prefix(cache_point(handle), frontier), frontier);
}

runtime::ContextResourceUsage
Program::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    runtime::ContextResourceUsage total;
    for (const CheckpointHandle handle : handles) {
        if (!valid_checkpoint(handle)) { continue; }
        const runtime::ContextResourceUsage usage = impl_->core.cache_footprint(cache_point(handle));
        total.state_slots += usage.state_slots;
        total.main_kv_pages += usage.main_kv_pages;
        total.backend_kv_pages += usage.backend_kv_pages;
    }
    return total;
}

CheckpointSummary Program::checkpoint_summary(CheckpointHandle handle) const {
    const CheckpointMetadata meta = checkpoint_metadata(handle);
    return {.key = checkpoint_key(handle, meta.frontier), .frontier = meta.frontier, .role = meta.role};
}

CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle handle) const {
    if (!valid_checkpoint(handle)) { throw std::logic_error("V3 P5 checkpoint is stale"); }
    return {.frontier = impl_->core.cache_frontier(cache_point(handle)),
            .role     = impl_->core.cache_role(cache_point(handle))};
}

bool Program::checkpoint_matches(CheckpointHandle handle, const RequestBasePlan& base) const {
    return base.impl_ && base.impl_->allow_prefix_reuse && valid_checkpoint(handle) &&
           impl_->core.cache_matches(cache_point(handle), base.impl_->prompt);
}

std::uint32_t Program::checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&,
                                                   std::uint32_t) const {
    return 0;
}

std::uint64_t Program::checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                               std::span<const CheckpointHandle>) const {
    return 0;
}

bool Program::valid_checkpoint(CheckpointHandle handle) const noexcept {
    return handle.owner == impl_.get() && impl_->core.cache_valid(cache_point(handle));
}

bool Program::release_checkpoint(CheckpointHandle handle) noexcept {
    if (handle.owner != impl_.get()) { return false; }
    return impl_->core.release_cache(cache_point(handle));
}
bool Program::revoke_snapshot(ResumeState&) noexcept { return false; }

runtime::ContextResourceUsage Program::snapshot_resources(const ResumeState&) const { return {}; }

std::size_t Program::host_bytes_released(std::span<const CheckpointHandle>) const { return 0; }

std::optional<std::size_t> Program::pause_host_bytes(SequenceHandle) const { return std::nullopt; }

std::size_t Program::release_redundant_host(std::span<const CheckpointHandle>,
                                            std::optional<SequenceHandle>) {
    return 0;
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    if (units.size() != 1 || !impl_->owns(units[0].sequence)) {
        throw std::logic_error("V3 P3 reservation does not match the active sequence");
    }
    if (units[0].kind == ExecutionUnitKind::Normalize) { not_yet("normalize"); }
    return {.reserved = true};
}

bool Program::reclaim_capture_reservation(runtime::ContextResourceUsage) { return false; }

void Program::release_units(std::span<const SequenceHandle>) noexcept {}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind, std::uint32_t) {
    if (lane.value >= kMaximumConcurrency) { throw std::logic_error("V3 lane is outside the plan"); }
    if (impl_->binding || impl_->pausing || impl_->live[lane.value].occupied) {
        return {.reserved = false, .source_valid = true, .capacity_possible = true};
    }
    impl_->core.select(lane.value);
    const auto lacking = [](const runtime::ContextResourceUsage& shortage) {
        return shortage.state_slots != 0 || shortage.main_kv_pages != 0 ||
               shortage.backend_kv_pages != 0;
    };
    const auto short_of = [](runtime::ContextResourceUsage shortage) {
        return BindingReservation{.reserved = false, .capacity_possible = true, .shortage = shortage};
    };
    if (resume != nullptr) {
        if (!base.impl_) { throw std::logic_error("request plan is empty"); }
        const std::uint64_t generation = resume->impl_->generation;
        const auto shortage = impl_->core.bind_shortage(
            base.impl_->request, resume->has_snapshot() ? impl_->core.park_point() : std::nullopt);
        if (lacking(shortage)) { return short_of(shortage); }
        if (resume->has_snapshot()) {
            impl_->core.restore_park(generation);
            impl_->replay_bind = false;
        } else if (impl_->core.park_ledger(generation)) {
            impl_->core.begin_replay(generation);
            impl_->replay_bind = true;
        } else {
            throw std::logic_error("V3 P6 resume has no snapshot or ledger");
        }
        resume->impl_->owner = nullptr;
        impl_->binding     = true;
        impl_->binding_row = lane.value;
        impl_->begin       = {.prompt_tokens        = base.summary().prompt_tokens,
                              .reused_prompt_tokens = 0,
                              .prefix_reuse_path    = PrefixReusePath::Root};
        return {.reserved = true};
    }
    if (!base.impl_) { throw std::logic_error("request plan is empty"); }
    if (source.checkpoint) {
        if (!valid_checkpoint(*source.checkpoint) ||
            source.reused_tokens != impl_->core.cache_frontier(cache_point(*source.checkpoint))) {
            return {.source_valid = false};
        }
        if (!source.consume_source) {
            const auto shortage = impl_->core.bind_shortage(
                base.impl_->request, cache_point(*source.checkpoint), true);
            if (lacking(shortage)) { return short_of(shortage); }
            impl_->core.fork_lane(base.impl_->request, cache_point(*source.checkpoint));
            impl_->binding     = true;
            impl_->binding_row = lane.value;
            impl_->begin       = {.prompt_tokens        = base.summary().prompt_tokens,
                                  .reused_prompt_tokens = source.reused_tokens,
                                  .prefix_reuse_path    = PrefixReusePath::Checkpoint};
            return {.reserved = true};
        }
        const auto shortage =
            impl_->core.bind_shortage(base.impl_->request, cache_point(*source.checkpoint));
        if (lacking(shortage)) { return short_of(shortage); }
        impl_->core.resume_lane(base.impl_->request, cache_point(*source.checkpoint));
        impl_->binding     = true;
        impl_->binding_row = lane.value;
        impl_->begin       = {.prompt_tokens        = base.summary().prompt_tokens,
                              .reused_prompt_tokens = source.reused_tokens,
                              .prefix_reuse_path    = PrefixReusePath::Checkpoint};
        return {.reserved = true, .consumed_source = source.checkpoint};
    }
    if (source.reused_tokens != 0) { return {.source_valid = false}; }
    if (const auto shortage = impl_->core.bind_shortage(base.impl_->request); lacking(shortage)) {
        return short_of(shortage);
    }
    impl_->core.bind(base.impl_->request, lane.value);
    impl_->binding     = true;
    impl_->binding_row = lane.value;
    impl_->begin       = {.prompt_tokens        = base.summary().prompt_tokens,
                          .reused_prompt_tokens = 0,
                          .prefix_reuse_path    = PrefixReusePath::Root};
    return {.reserved = true};
}

bool Program::start_capture(SequenceHandle sequence) {
    if (!impl_->owns(sequence) || impl_->binding || impl_->pausing || impl_->capturing) {
        return false;
    }
    const std::uint32_t row = detail::ProgramAccess::lane_of(sequence);
    impl_->core.select(row);
    const auto point = impl_->core.capture_prefix();
    if (!point) { return false; }
    impl_->captured   = CheckpointHandle{.owner = impl_.get(), .index = point->index, .generation = point->generation};
    impl_->capturing  = true;
    impl_->binding_row = row;
    return true;
}
bool Program::capture_is_input(SequenceHandle) const { return false; }
std::optional<CapturePreparation> Program::prepare_capture(SequenceHandle sequence) {
    if (!impl_->owns(sequence)) { return std::nullopt; }
    impl_->core.select(detail::ProgramAccess::lane_of(sequence));
    const auto frontier = impl_->core.next_capture_frontier();
    if (!frontier || impl_->core.state_capture_ready()) { return std::nullopt; }
    CapturePreparation preparation;
    preparation.frontier             = *frontier;
    preparation.shortage.state_slots = 1;
    return preparation;
}
void Program::skip_capture(SequenceHandle sequence) {
    if (!impl_->owns(sequence)) { return; }
    impl_->core.select(detail::ProgramAccess::lane_of(sequence));
    impl_->core.skip_capture();
}

ContextReclaimPlan Program::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         runtime::ContextResourceUsage shortage) const {
    ContextReclaimPlan plan;
    plan.releases = plan_releases(allowed, excluded, shortage);
    return plan;
}

// A release frees the record's state slot and the pages no other address holds. Full pages a
// shared prefix still shares with a lane or a copy stay. Only the queried pools are quoted.
std::vector<ContextRelease> Program::plan_releases(std::span<const CheckpointHandle> allowed,
                                                   std::span<const CheckpointHandle> excluded,
                                                   runtime::ContextResourceUsage shortage) const {
    std::vector<ContextRelease> releases;
    if (impl_->binding || impl_->pausing) { return releases; }
    for (const CheckpointHandle handle : allowed) {
        if (!valid_checkpoint(handle) ||
            std::find(excluded.begin(), excluded.end(), handle) != excluded.end()) {
            continue;
        }
        const runtime::ContextResourceUsage usage =
            impl_->core.cache_release_quote(cache_point(handle));
        ContextRelease release;
        release.sources = {handle};
        release.released.state_slots      = shortage.state_slots ? usage.state_slots : 0U;
        release.released.main_kv_pages    = shortage.main_kv_pages ? usage.main_kv_pages : 0U;
        release.released.backend_kv_pages = shortage.backend_kv_pages ? usage.backend_kv_pages : 0U;
        releases.push_back(std::move(release));
    }
    return releases;
}

// No host checkpoint tier yet: plan_reclaim quotes no demotion, and a stray quote is declined.
bool Program::start_demote(const ContextDemotion&) { return false; }
bool Program::start_pause(SequenceHandle sequence, bool, runtime::ExecutionTiming*) {
    if (!impl_->owns(sequence) || impl_->binding || impl_->pausing) { return false; }
    impl_->core.select(detail::ProgramAccess::lane_of(sequence));
    if (impl_->core.has_pending()) { return false; }
    const auto generation = impl_->core.park_lane();
    if (!generation) { return false; }
    impl_->park_generation = *generation;
    impl_->binding_row     = detail::ProgramAccess::lane_of(sequence);
    impl_->pausing         = true;
    return true;
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    if (impl_->capturing) {
        ContextProgress progress;
        progress.kind     = ContextOperationKind::Capture;
        progress.advanced = true;
        progress.complete = true;
        progress.published = true;
        progress.captured_checkpoints = {impl_->captured};
        progress.sequence = impl_->live[impl_->binding_row].handle;
        impl_->capturing  = false;
        return progress;
    }
    if (impl_->pausing) {
        ContextProgress progress;
        progress.kind     = ContextOperationKind::Pause;
        progress.advanced = true;
        progress.complete = true;
        if (cancellation.requested()) {
            impl_->core.release_park(impl_->park_generation);
            impl_->clear_slot(impl_->binding_row);
            impl_->pausing = false;
            return progress;
        }
        ResumeState parked;
        parked.impl_              = std::make_unique<ResumeState::Impl>();
        parked.impl_->owner       = impl_.get();
        parked.impl_->generation  = impl_->park_generation;
        progress.published        = true;
        progress.paused           = std::move(parked);
        progress.request_speculative = impl_->core.park_stats(impl_->park_generation);
        impl_->clear_slot(impl_->binding_row);
        impl_->pausing = false;
        return progress;
    }
    if (!impl_->binding) { not_yet("context transaction"); }
    if (cancellation.requested()) {
        impl_->core.select(impl_->binding_row);
        impl_->core.release_lane();
        impl_->binding = false;
        return {.kind = ContextOperationKind::Bind, .advanced = true, .complete = true};
    }
    const std::uint64_t epoch = impl_->next_epoch++;
    const runtime::LaneId lane{impl_->binding_row};
    auto& slot   = impl_->live[impl_->binding_row];
    slot.occupied = true;
    slot.epoch    = epoch;
    slot.begin    = impl_->begin;
    slot.handle   = detail::ProgramAccess::make_sequence(impl_.get(), lane, epoch);
    impl_->binding = false;
    ContextProgress progress;
    progress.kind      = ContextOperationKind::Bind;
    progress.advanced  = true;
    progress.complete  = true;
    progress.published = true;
    progress.replaying = impl_->replay_bind;
    progress.sequence  = slot.handle;
    impl_->replay_bind = false;
    return progress;
}

bool Program::has_context_transaction() const noexcept {
    return impl_->binding || impl_->pausing || impl_->capturing;
}
bool Program::context_blocks(SequenceHandle) const noexcept { return false; }
bool Program::recovery_pending(SequenceHandle) const noexcept { return false; }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider* masks) {
    if (!impl_->owns(sequence)) { throw std::logic_error("prefill sequence is not active"); }
    const std::uint32_t row = detail::ProgramAccess::lane_of(sequence);
    impl_->core.select(row);
    const auto start = std::chrono::steady_clock::now();
    const auto step  = impl_->core.advance_prefill(masks);
    const auto timing = elapsed_since(start);
    impl_->core.note_prefill_seconds(static_cast<double>(timing.elapsed_ns()) / 1.0e9);
    PrefillProgress progress;
    progress.summary                  = impl_->live[row].begin;
    progress.processed_prompt_tokens  = step.processed_tokens;
    progress.timing                   = timing;
    if (step.capture_ready && !step.complete) { progress.capture_ready = true; }
    if (!step.complete) { return progress; }
    impl_->pending_rows   = {row};
    impl_->pending_tokens = {step.token};
    impl_->pending_counts.clear();
    impl_->pending_timing = timing;
    ++impl_->pending_transaction;
    progress.complete = true;
    const std::uint8_t failed = (step.grammar_dead & 1U) != 0 ? 1U : 0U;
    progress.pending.emplace(detail::ProgramAccess::make_pending(
        impl_.get(), impl_->pending_transaction, impl_->pending_tokens, timing, {}, 0, 1,
        std::span<const std::uint8_t>(&failed, failed)));
    return progress;
}

ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    if (!impl_->owns(sequence)) { throw std::logic_error("replay sequence is not active"); }
    impl_->core.select(detail::ProgramAccess::lane_of(sequence));
    const auto step = impl_->core.advance_replay();
    ReplayProgress progress;
    progress.processed_tokens = step.processed;
    progress.complete         = step.complete;
    return progress;
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    if (sequences.empty() || sequences.size() != budgets.size()) {
        throw std::logic_error("V3 decode rows and budgets disagree");
    }
    std::vector<std::uint32_t> rows;
    std::vector<std::uint32_t> remaining;
    rows.reserve(sequences.size());
    remaining.reserve(sequences.size());
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        if (!impl_->owns(sequences[index])) {
            throw std::logic_error("V3 decode sequence is not active");
        }
        rows.push_back(detail::ProgramAccess::lane_of(sequences[index]));
        remaining.push_back(budgets[index].generated_tokens_remaining);
    }
    const auto start  = std::chrono::steady_clock::now();
    const auto step   = impl_->core.decode_rows(rows, remaining, masks);
    const auto timing = elapsed_since(start);
    impl_->core.note_decode_seconds(rows, static_cast<double>(timing.elapsed_ns()) / 1.0e9);
    std::vector<std::uint8_t> failed(step.dead.size());
    for (std::size_t index = 0; index < step.dead.size(); ++index) {
        const std::uint32_t licensed = step.counts.empty() ? 1U
                                       : step.counts[index] > 0
                                           ? static_cast<std::uint32_t>(step.counts[index])
                                           : 0U;
        const std::uint32_t bits = licensed >= 32U ? 0xffffffffU : ((1U << licensed) - 1U);
        failed[index]            = (step.dead[index] & bits) != 0 ? 1U : 0U;
    }
    impl_->pending_rows   = std::move(rows);
    impl_->pending_tokens = step.tokens;
    impl_->pending_counts = step.counts;
    impl_->pending_timing = timing;
    ++impl_->pending_transaction;
    return detail::ProgramAccess::make_pending(
        impl_.get(), impl_->pending_transaction, impl_->pending_tokens, timing,
        impl_->pending_counts, step.stride, impl_->pending_rows.size(), failed);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming*) {
    if (sequences.size() != 1 || !impl_->owns(sequences[0]) || row_stride == 0 ||
        row_major_tokens.size() != row_stride || prefix_execution_splits.size() > 1) {
        throw std::logic_error("V3 forced tokens require one active row");
    }
    impl_->core.select(detail::ProgramAccess::lane_of(sequences[0]));
    const auto start = std::chrono::steady_clock::now();
    impl_->core.append_forced(row_major_tokens, prefix_execution_splits.empty()
                                                    ? std::nullopt
                                                    : prefix_execution_splits[0]);
    return elapsed_since(start);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    if (!detail::ProgramAccess::owns(pending, impl_.get(), impl_->pending_transaction) ||
        decisions.size() != impl_->pending_rows.size()) {
        throw std::logic_error("V3 commit does not match the pending batch");
    }
    for (std::size_t index = 0; index < decisions.size(); ++index) {
        const runtime::CommitDecision& decision = decisions[index];
        impl_->core.select(impl_->pending_rows[index]);
        if (!impl_->pending_counts.empty()) {
            const std::uint32_t accepted =
                decision.failed || decision.cancelled ? 0U : decision.accepted_tokens;
            impl_->core.accept_speculative(
                accepted, decision.terminal || decision.failed || decision.cancelled,
                accepted == 0 ? std::nullopt : decision.prefix_execution_split_after);
        } else if (!decision.failed && !decision.cancelled && decision.accepted_tokens == 1) {
            impl_->core.accept_pending(decision.prefix_execution_split_after);
        } else if (decision.accepted_tokens == 0) {
            impl_->core.drop_pending();
        } else {
            throw std::logic_error("V3 commit accepted an unexpected token count");
        }
    }
    const auto timing = impl_->pending_timing;
    const std::vector<std::uint32_t> rows = std::move(impl_->pending_rows);
    impl_->pending_tokens.clear();
    impl_->pending_counts.clear();
    impl_->pending_rows.clear();
    CommitResult result;
    result.row_count = decisions.size();
    result.timing    = timing;
    for (std::size_t index = 0; index < decisions.size(); ++index) {
        const runtime::CommitDecision& decision = decisions[index];
        impl_->core.select(rows[index]);
        const SpeculativeStats& stats = impl_->core.request_speculative_stats();
        result.rows[index].disposition =
            decision.failed      ? runtime::CommitDisposition::FailedReleased
            : decision.cancelled ? runtime::CommitDisposition::CancelledReleased
            : decision.terminal  ? runtime::CommitDisposition::Finishable
                                 : runtime::CommitDisposition::Active;
        result.rows[index].speculative                           = stats;
        result.rows[index].speculative_counters.rounds           = stats.rounds;
        result.rows[index].speculative_counters.drafted_tokens   = stats.drafted_tokens;
        result.rows[index].speculative_counters.accepted_tokens  = stats.accepted_tokens;
        result.rows[index].speculative_counters.fallback_steps   = stats.fallback_steps;
    }
    return result;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    if (!detail::ProgramAccess::owns(pending, impl_.get(), impl_->pending_transaction)) {
        return {};
    }
    for (const std::uint32_t row : impl_->pending_rows) {
        impl_->core.select(row);
        impl_->core.drop_pending();
    }
    impl_->pending_rows.clear();
    impl_->pending_tokens.clear();
    impl_->pending_counts.clear();
    return {.status = runtime::ConsumeStatus::Consumed, .row_count = 1};
}

void Program::configure_activation(const EngineOptions& options) {
    impl_->core.configure_activation(options);
}

void Program::activate_steering(const qwen3_8_flash_next::SteeringPack* pack) {
    impl_->core.activate_steering(pack);
}

void Program::begin_activation(std::uint32_t lane, std::uint64_t request_id,
                               const qwen3_8_flash_next::PreparedPrompt& prompt,
                               const runtime::ResolvedExecutionOptions& options) {
    // plan_request already took the prompt. The lane holds its tokens.
    (void)prompt;
    impl_->core.begin_activation(lane, request_id, {}, options);
}

void Program::select_activation(std::span<const std::uint32_t> lanes) {
    impl_->core.select_activation(lanes);
}

std::string Program::flush_activation(std::uint32_t lane) {
    return impl_->core.flush_activation(lane);
}

ninfer::SteeringState Program::steering_state() const { return impl_->core.steering_state(); }

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    if (!impl_->owns(sequence)) { return {}; }
    const std::uint32_t row = detail::ProgramAccess::lane_of(sequence);
    impl_->core.select(row);
    SpeculativeStats stats = impl_->core.request_speculative_stats();
    FinishResult out{.status = runtime::ConsumeStatus::Consumed, .speculative = std::move(stats)};
    out.timings.prefill_seconds = impl_->core.noted_prefill_seconds();
    out.timings.decode_seconds  = impl_->core.noted_decode_seconds();
    if (const auto point = impl_->core.catalog_lane()) {
        out.checkpoint = CheckpointHandle{.owner      = impl_.get(),
                                          .index      = point->index,
                                          .generation = point->generation};
    }
    impl_->clear_slot(row);
    return out;
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (!impl_->owns(sequence)) { return {}; }
    const std::uint32_t row = detail::ProgramAccess::lane_of(sequence);
    impl_->core.select(row);
    impl_->core.drop_pending();
    impl_->core.release_lane();
    impl_->clear_slot(row);
    impl_->pending_tokens.clear();
    return {.status = runtime::ConsumeStatus::Consumed};
}

void Program::fail_all_cleanup() noexcept {
    for (std::uint32_t row = 0; row < kMaximumConcurrency; ++row) {
        impl_->core.select(row);
        impl_->core.release_lane();
        impl_->clear_slot(row);
    }
    impl_->core.release_park(impl_->park_generation);
    impl_->binding   = false;
    impl_->pausing   = false;
    impl_->capturing = false;
    impl_->pending_rows.clear();
    impl_->pending_tokens.clear();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    PhysicalUsageSnapshot usage;
    const auto [capacity, occupied] = impl_->core.pool_usage();
    usage.capacity.state_slots      = capacity.state_slots;
    usage.capacity.main_kv_pages    = capacity.main_kv_pages;
    usage.capacity.backend_kv_pages = capacity.backend_kv_pages;
    usage.occupied.state_slots      = occupied.state_slots;
    usage.occupied.main_kv_pages    = occupied.main_kv_pages;
    usage.occupied.backend_kv_pages = occupied.backend_kv_pages;
    return usage;
}

MemorySummary Program::memory_summary() const noexcept {
    MemorySummary summary;
    summary.device       = impl_->device_index;
    summary.max_context  = impl_->capacity;
    summary.kv_capacity  = impl_->kv_capacity;
    summary.kv_cache                   = impl_->kv_cache;
    summary.cuda_graph_allowance_bytes = impl_->core.cuda_graph_allowance_bytes();
    return summary;
}

void Program::reset_memory_peaks() noexcept {}

} // namespace ninfer::models::qwen3_8_flash_next::v3

namespace ninfer::models::qwen3_8_flash_next {

using detail::qwen3_8_flash_next_125b_a6b_runtime::Variant;

template <>
std::unique_ptr<v3::Program> create_v3_program<Variant>(
    const Variant::ModelView& model, Variant::WeightsProfile weights_profile,
    SequencePlan<Variant>&& plan, DeviceContext& device, const StartupObserver&) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->weights_profile != weights_profile) {
        throw std::invalid_argument("loaded model weights profile does not match the sequence plan");
    }
    if (plan.impl_->speculative_backend == SpeculativeBackend::DFlash) {
        throw std::logic_error("V3 P4 does not support DFlash");
    }
    auto impl = std::make_unique<v3::Program::Impl>(model, *plan.impl_, device);
    plan.impl_.reset();
    return v3::Program::from_parts(std::move(impl));
}

} // namespace ninfer::models::qwen3_8_flash_next
