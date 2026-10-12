// Prefix checkpoints for one lane().
// A finished request freezes its state and deactivates KV at the committed frontier, which is
// one token behind the ledger. The next prompt resumes that prefix and prefills only the suffix.

#include "models/qwen3_8_flash_next/program/p3.h"

#include "ninfer/ops/sampling.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

namespace {

std::uint32_t kv_pages(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// KvEntitlement counts a lane's total pages; the store reserves growth beyond the mapped ones.
std::uint32_t growth_beyond(std::uint32_t entitlement, std::uint32_t mapped) noexcept {
    return entitlement > mapped ? entitlement - mapped : 0U;
}

} // namespace

void P3Program::install_request(Request& request) {
    const bool penalties = request.sampling.presence_penalty != 0.0F ||
                           request.sampling.frequency_penalty != 0.0F;
    request.sampling.token_counts = nullptr;
    const std::uint32_t prompt_tokens = request.summary.prompt_tokens;
    if (penalties) {
        Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(focus_), 1)
                            .view({TextConfig::token_domain});
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
        request.sampling.token_counts = static_cast<std::int32_t*>(counts.data);
        // A resumed request counts what it produced before the pause again.
        if (request.tokens.size() > prompt_tokens) {
            count_tokens(std::span<const TokenId>(request.tokens).subspan(prompt_tokens));
        }
    }
    const std::uint32_t mapped_text = std::min(capacity, prompt_tokens);
    if (backend != SpeculativeBackend::Mtp) {
        map_kv(mapped_text);
        return;
    }
    const std::uint32_t room = capacity > prompt_tokens ? capacity - prompt_tokens - 1U : 0U;
    const std::uint32_t budget = request.summary.effective_output_tokens > 1
                                     ? request.summary.effective_output_tokens - 2U
                                     : 0U;
    lane().mtp_extent = std::min({minimum_draft_window, budget, room});
    const std::uint32_t extent = lane().mtp_extent;
    map_kv(mapped_text);
    const std::uint32_t mtp_tokens = std::min(capacity, prompt_tokens + (extent == 0 ? 0U : extent - 1U));
    if (static_cast<std::uint64_t>(mtp_addresses->mapped_pages(lane().mtp_kv)) * kPagedKVPageSize <
        mtp_tokens) {
        mtp_addresses->ensure_mapped_to_tokens(lane().mtp_kv, mtp_tokens, device.stream);
    }
}

void P3Program::count_tokens(std::span<const TokenId> tokens) {
    if (tokens.empty() || lane().request.sampling.token_counts == nullptr) { return; }
    work.reset();
    Tensor ids = work.alloc(DType::I32, {checked_i32(static_cast<std::uint32_t>(tokens.size()),
                                                     "penalty token count")});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data(), ids.bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(focus_), 1)
                        .view({TextConfig::token_domain});
    ops::increment_token_counts(ids, counts, device.stream);
    work.reset();
}

P3Program::CacheRecord* P3Program::find_cache(CachePoint point) noexcept {
    if (point.index >= cache_slots.size()) { return nullptr; }
    CacheRecord& record = cache_slots[point.index];
    if (!record.occupied || record.generation != point.generation) { return nullptr; }
    return &record;
}

const P3Program::CacheRecord* P3Program::find_cache(CachePoint point) const noexcept {
    if (point.index >= cache_slots.size()) { return nullptr; }
    const CacheRecord& record = cache_slots[point.index];
    if (!record.occupied || record.generation != point.generation) { return nullptr; }
    return &record;
}

void P3Program::release_frozen(program::KVAddressSpaceHandle kv,
                               program::KVAddressSpaceHandle mtp,
                               qwen3_8_flash_next::detail::StateImageHandle state) noexcept {
    try {
        if (state_store->valid(state) &&
            state_store->role(state) == qwen3_8_flash_next::detail::StateImageRole::CheckpointImmutable &&
            state_store->checkpoint_references(state) > 0) {
            state_store->release_checkpoint_reference(state);
        }
    } catch (...) {}
    if (text_kv_addresses->valid(kv)) {
        if (text_kv_addresses->active(kv)) {
            (void)text_kv_addresses->release_after_deactivate(kv);
        } else {
            (void)text_kv_addresses->release(kv);
        }
    }
    if (mtp.valid() && mtp_addresses && mtp_addresses->valid(mtp)) {
        if (mtp_addresses->active(mtp)) {
            (void)mtp_addresses->release_after_deactivate(mtp);
        } else {
            (void)mtp_addresses->release(mtp);
        }
    }
    if (state_store->valid(state)) { (void)state_store->release(state); }
}

P3Program::CachePoint P3Program::seal_lane() {
    if (!lane().bound || lane().kv_tokens == 0 || lane().request.tokens.size() < lane().kv_tokens) {
        throw std::logic_error("V3 P5 lane cannot be sealed");
    }
    CacheRecord* slot = nullptr;
    for (auto& record : cache_slots) {
        if (!record.occupied) {
            slot = &record;
            break;
        }
    }
    if (slot == nullptr) { throw std::logic_error("V3 P5 prefix catalog is full"); }
    const auto kv       = lane().kv;
    const auto mtp      = lane().mtp_kv;
    const auto state    = lane().state;
    const auto frontier = lane().kv_tokens;
    // Throws before anything changes when the frontier splits a Vision item.
    ResidentPrefixIdentity identity = lane().request.identity;
    identity.truncate(frontier);
    std::vector<TokenId> prefix(lane().request.tokens.begin(),
                                lane().request.tokens.begin() + static_cast<std::ptrdiff_t>(frontier));
    state_store->freeze(state);
    state_store->retain_checkpoint_reference(state);
    text_kv_addresses->set_checkpoint_requirement(kv, frontier);
    text_kv_addresses->deactivate(kv);
    if (mtp.valid()) {
        mtp_addresses->set_checkpoint_requirement(mtp, frontier);
        mtp_addresses->deactivate(mtp);
    }
    slot->tokens     = std::move(prefix);
    slot->identity   = std::move(identity);
    slot->kv         = kv;
    slot->mtp        = mtp;
    slot->state      = state;
    slot->frontier   = frontier;
    slot->role       = runtime::CheckpointRole::Continuation;
    slot->occupied   = true;
    slot->generation = ++cache_clock;
    if (slot->generation == 0) { slot->generation = ++cache_clock; }
    const CachePoint point{static_cast<std::uint32_t>(slot - cache_slots.data()), slot->generation};
    lane() = {};
    return point;
}

std::optional<P3Program::CachePoint> P3Program::catalog_lane() noexcept {
    if (!lane().bound) { return std::nullopt; }
    if (!lane().request.summary.publish_continuation || lane().kv_tokens == 0 || lane().pending ||
        lane().request.tokens.size() < lane().kv_tokens) {
        release_lane();
        return std::nullopt;
    }
    const auto kv    = lane().kv;
    const auto mtp   = lane().mtp_kv;
    const auto state = lane().state;
    try {
        return seal_lane();
    } catch (...) {
        release_frozen(kv, mtp, state);
        lane() = {};
        return std::nullopt;
    }
}

std::optional<std::uint64_t> P3Program::park_lane() {
    if (!lane().bound || lane().pending || parked.occupied || lane().kv_tokens == 0 ||
        lane().request.tokens.size() < lane().kv_tokens) {
        return std::nullopt;
    }
    const bool free_slot =
        std::any_of(cache_slots.begin(), cache_slots.end(),
                    [](const CacheRecord& record) { return !record.occupied; });
    if (!free_slot) { return park_as_ledger(); }
    Park next;
    next.device       = true;
    next.request      = lane().request;
    next.prefill_done = lane().prefill_done;
    next.cursor       = lane().cursor;
    next.kv_tokens    = lane().kv_tokens;
    next.drafts       = lane().drafts;
    next.draft_count  = lane().draft_count;
    next.mtp_extent   = lane().mtp_extent;
    next.stats        = lane().stats;
    try {
        next.point = seal_lane();
    } catch (...) {
        if (lane().bound) { return park_as_ledger(); }
        return std::nullopt;
    }
    next.generation = next.point.generation;
    next.occupied   = true;
    const std::uint64_t generation = next.generation;
    parked                         = std::move(next);
    return generation;
}

std::optional<std::uint64_t> P3Program::park_as_ledger() {
    if (!lane().bound || lane().pending || parked.occupied || lane().kv_tokens == 0 ||
        lane().request.tokens.size() < lane().kv_tokens) {
        return std::nullopt;
    }
    Park next;
    next.device       = false;
    next.request      = lane().request;
    next.prefill_done = lane().prefill_done;
    next.cursor       = lane().cursor;
    next.kv_tokens    = lane().kv_tokens;
    next.stats        = lane().stats;
    next.generation   = ++cache_clock;
    if (next.generation == 0) { next.generation = ++cache_clock; }
    next.occupied = true;
    const std::uint64_t generation = next.generation;
    release_lane();
    parked = std::move(next);
    return generation;
}

bool P3Program::park_valid(std::uint64_t generation) const noexcept {
    return parked.occupied && parked.generation == generation && cache_valid(parked.point);
}

std::uint32_t P3Program::park_frontier(std::uint64_t generation) const noexcept {
    if (!parked.occupied || parked.generation != generation) { return 0; }
    if (!parked.device) { return parked.kv_tokens; }
    const CacheRecord* record = find_cache(parked.point);
    return record == nullptr ? 0U : record->frontier;
}

bool P3Program::park_ledger(std::uint64_t generation) const noexcept {
    return parked.occupied && !parked.device && parked.generation == generation &&
           parked.kv_tokens != 0;
}

SpeculativeStats P3Program::park_stats(std::uint64_t generation) const {
    if (!parked.occupied || parked.generation != generation) { return fresh_stats(); }
    return parked.stats;
}

void P3Program::begin_replay(std::uint64_t generation) {
    if (!park_ledger(generation) || lane().bound) {
        throw std::logic_error("V3 P6 replay has no ledger");
    }
    // The ledger stays parked until the bind holds, so a failed bind can be retried.
    bind(parked.request, focus_);
    lane().replaying             = true;
    lane().replay_target         = parked.kv_tokens;
    lane().replay_resume_prefill = parked.prefill_done;
    lane().prefill_done          = false;
    lane().cursor                = 0;
    lane().kv_tokens             = 0;
    lane().draft_count           = 0;
    lane().stats                 = parked.stats;
    parked                       = {};
}

P3Program::ReplayStep P3Program::advance_replay() {
    require_bound();
    if (!lane().replaying) { throw std::logic_error("V3 P6 lane is not replaying"); }
    if (lane().cursor >= lane().replay_target) {
        lane().replaying    = false;
        lane().prefill_done = lane().replay_resume_prefill;
        device.synchronize();
        return {.complete = true};
    }
    // Prompt chunks stop where a cold prefill stops; the ledger after the prompt is plain text.
    const bool in_prompt = lane().cursor < lane().request.summary.prompt_tokens;
    const std::uint32_t limit =
        in_prompt ? std::min(prompt_chunk_limit(), lane().replay_target) : lane().replay_target;
    const std::uint32_t remaining = std::min(prefill_chunk, limit - lane().cursor);
    map_kv(lane().cursor + remaining);
    schedule::PrefillContext schedule_state = make_prefill();
    schedule_state.text_kv_base             = lane().cursor;
    schedule::PrefillChunkResult result;
    if (in_prompt && lane().vision) {
        result = schedule::prefill_multimodal_chunk(schedule_state, *lane().request.media,
                                                    *lane().vision, remaining, std::nullopt, false);
        lane().vision->release_encoded_media_payloads();
    } else {
        result = schedule::prefill_text_chunk(schedule_state,
                                              std::span<const TokenId>(lane().request.tokens),
                                              remaining, std::nullopt, false);
    }
    if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
        throw std::logic_error("replay chunk made invalid progress");
    }
    lane().cursor += result.processed_tokens;
    lane().kv_tokens = lane().cursor;
    text_kv_addresses->commit_frontier(lane().kv, lane().kv_tokens);
    if (lane().mtp_kv.valid()) { mtp_addresses->commit_frontier(lane().mtp_kv, lane().kv_tokens); }
    ReplayStep step;
    step.processed = result.processed_tokens;
    if (lane().cursor >= lane().replay_target) {
        lane().replaying    = false;
        lane().prefill_done = lane().replay_resume_prefill;
        device.synchronize();
        step.complete = true;
    }
    return step;
}

void P3Program::restore_park(std::uint64_t generation) {
    if (!park_valid(generation)) { throw std::logic_error("V3 P6 pause snapshot is stale"); }
    resume_lane(parked.request, parked.point);
    lane().prefill_done = parked.prefill_done;
    lane().cursor       = parked.cursor;
    lane().drafts       = parked.drafts;
    lane().draft_count  = parked.draft_count;
    lane().mtp_extent   = parked.mtp_extent;
    lane().stats        = parked.stats;
    parked            = {};
}

void P3Program::release_park(std::uint64_t generation) noexcept {
    if (!parked.occupied || parked.generation != generation) { return; }
    const bool device      = parked.device;
    const CachePoint point = parked.point;
    parked                 = {};
    if (device) { (void)release_cache(point); }
}

bool P3Program::cache_valid(CachePoint point) const noexcept { return find_cache(point) != nullptr; }

std::uint32_t P3Program::cache_frontier(CachePoint point) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr) { throw std::logic_error("V3 P5 checkpoint is stale"); }
    return record->frontier;
}

std::span<const TokenId> P3Program::cache_prefix(CachePoint point,
                                                 std::uint32_t frontier) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr || frontier == 0 || frontier > record->tokens.size()) {
        throw std::logic_error("checkpoint key is outside coverage");
    }
    return std::span<const TokenId>(record->tokens.data(), frontier);
}

std::optional<std::uint32_t> P3Program::next_capture_frontier() const {
    if (!lane().bound || !lane().request.summary.publish_continuation) { return std::nullopt; }
    std::optional<std::uint32_t> next;
    for (const std::uint32_t frontier : lane().request.capture_frontiers) {
        if (frontier <= lane().cursor || frontier <= lane().capture_skipped_through ||
            frontier >= lane().request.summary.prompt_tokens) {
            continue;
        }
        if (!next || frontier < *next) { next = frontier; }
    }
    return next;
}

void P3Program::skip_capture() noexcept {
    if (!lane().bound) { return; }
    if (lane().capture_offered) {
        lane().capture_offered         = false;
        lane().capture_skipped_through = std::max(lane().capture_skipped_through, lane().cursor);
        return;
    }
    if (const auto next = next_capture_frontier()) { lane().capture_skipped_through = *next; }
}

bool P3Program::state_capture_ready() const noexcept {
    return state_store->device_occupied() < state_store->device_capacity();
}

bool P3Program::cache_matches(CachePoint point,
                              const qwen3_8_flash_next::PreparedPromptData& prompt) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr || prompt.token_ids.size() <= record->frontier ||
        record->tokens.size() != record->frontier) {
        return false;
    }
    return prefix_matches(prompt, record->tokens, record->identity, record->frontier);
}

bool P3Program::release_cache(CachePoint point) noexcept {
    CacheRecord* record = find_cache(point);
    if (record == nullptr) { return false; }
    const auto kv    = record->kv;
    const auto mtp   = record->mtp;
    const auto state = record->state;
    record->occupied = false;
    record->tokens.clear();
    record->identity.clear();
    record->kv    = {};
    record->mtp   = {};
    record->state = {};
    if (++record->generation == 0) { ++record->generation; }
    release_frozen(kv, mtp, state);
    return !text_kv_addresses->valid(kv) &&
           !(mtp.valid() && mtp_addresses && mtp_addresses->valid(mtp)) &&
           !state_store->valid(state);
}

P3Program::KvEntitlement P3Program::kv_entitlement(const Request& request) const noexcept {
    const std::uint64_t reserved = static_cast<std::uint64_t>(request.summary.prompt_tokens) +
                                   request.summary.effective_output_tokens;
    const auto pages = [&](std::uint64_t tokens) {
        const std::uint64_t needed = 1U + (tokens - 1U) / kPagedKVPageSize;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(needed, page_capacity));
    };
    KvEntitlement out;
    out.text = pages(reserved + draft_window + 1U);
    if (backend == SpeculativeBackend::Mtp) { out.mtp = pages(reserved + 2ULL * draft_window); }
    return out;
}

runtime::ContextResourceUsage P3Program::bind_shortage(const Request& request,
                                                       std::optional<CachePoint> resume,
                                                       bool copy) const {
    const CacheRecord* record = resume ? find_cache(*resume) : nullptr;
    const KvEntitlement entitlement = kv_entitlement(request);
    const auto lacking = [&](const program::KVAddressSpaceStore& addresses,
                             const program::LogicalKVPageStore& pages,
                             program::KVAddressSpaceHandle frozen,
                             std::uint32_t wanted) {
        std::uint32_t reused = 0;
        if (record != nullptr && addresses.valid(frozen)) {
            // A copy shares the full pages and allocates the tail plus the suffix. A resume
            // takes the frozen address, so every mapped page is reused.
            reused = copy ? record->frontier / static_cast<std::uint32_t>(kPagedKVPageSize)
                          : addresses.mapped_pages(frozen);
        }
        const std::uint32_t required  = wanted > reused ? wanted - reused : 0U;
        const std::uint32_t available = pages.physical_pool().available_pages();
        return required > available ? required - available : 0U;
    };
    runtime::ContextResourceUsage shortage;
    shortage.main_kv_pages =
        lacking(*text_kv_addresses, *text_kv_pages,
                record != nullptr ? record->kv : program::KVAddressSpaceHandle{},
                entitlement.text);
    if (backend == SpeculativeBackend::Mtp && mtp_addresses && mtp_kv_pages) {
        shortage.backend_kv_pages = lacking(
            *mtp_addresses, *mtp_kv_pages,
            record != nullptr ? record->mtp : program::KVAddressSpaceHandle{},
            entitlement.mtp);
    }
    if ((record == nullptr || copy) &&
        state_store->device_occupied() >= state_store->device_capacity()) {
        shortage.state_slots = 1;
    }
    return shortage;
}

runtime::CheckpointRole P3Program::cache_role(CachePoint point) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr) { throw std::logic_error("V3 P5 checkpoint is stale"); }
    return record->role;
}

runtime::ContextResourceUsage P3Program::cache_footprint(CachePoint point) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr) { return {}; }
    runtime::ContextResourceUsage usage;
    usage.state_slots = 1;
    if (text_kv_addresses->valid(record->kv)) {
        usage.main_kv_pages = text_kv_addresses->mapped_pages(record->kv);
    }
    if (record->mtp.valid() && mtp_addresses && mtp_addresses->valid(record->mtp)) {
        usage.backend_kv_pages = mtp_addresses->mapped_pages(record->mtp);
    }
    return usage;
}

runtime::ContextResourceUsage P3Program::cache_release_quote(CachePoint point) const {
    const CacheRecord* record = find_cache(point);
    if (record == nullptr) { return {}; }
    const auto unique = [](const program::KVAddressSpaceStore& addresses,
                           const program::LogicalKVPageStore& pages,
                           program::KVAddressSpaceHandle handle) {
        std::uint32_t count = 0;
        const std::uint32_t mapped = addresses.mapped_pages(handle);
        for (std::uint32_t page = 0; page < mapped; ++page) {
            if (pages.address_references(addresses.logical_page(handle, page)) == 1) { ++count; }
        }
        return count;
    };
    runtime::ContextResourceUsage usage;
    usage.state_slots = 1;
    if (text_kv_addresses->valid(record->kv)) {
        usage.main_kv_pages = unique(*text_kv_addresses, *text_kv_pages, record->kv);
    }
    if (record->mtp.valid() && mtp_addresses && mtp_addresses->valid(record->mtp)) {
        usage.backend_kv_pages = unique(*mtp_addresses, *mtp_kv_pages, record->mtp);
    }
    return usage;
}

void P3Program::resume_lane(Request request, CachePoint point) {
    CacheRecord* record = find_cache(point);
    if (record == nullptr || lane().bound) {
        throw std::logic_error("V3 P5 checkpoint cannot be resumed");
    }
    if (request.tokens.size() < record->frontier) {
        throw std::logic_error("V3 P5 resume prompt is shorter than the checkpoint");
    }
    const auto kv       = record->kv;
    const auto mtp      = record->mtp;
    const auto state    = record->state;
    const auto frontier = record->frontier;
    const KvEntitlement entitlement = kv_entitlement(request);
    state_store->release_checkpoint_reference(state);
    state_store->thaw(state);
    text_kv_addresses->activate(kv, growth_beyond(entitlement.text, text_kv_addresses->mapped_pages(kv)),
                                static_cast<std::int32_t>(focus_), device.stream);
    if (mtp.valid()) {
        mtp_addresses->activate(mtp, growth_beyond(entitlement.mtp, mtp_addresses->mapped_pages(mtp)),
                                static_cast<std::int32_t>(focus_), device.stream);
    }
    record->occupied = false;
    record->tokens.clear();
    record->identity.clear();
    record->kv    = {};
    record->mtp   = {};
    record->state = {};
    if (++record->generation == 0) { ++record->generation; }

    lane().request   = std::move(request);
    lane().kv        = kv;
    lane().mtp_kv    = mtp;
    lane().state     = state;
    lane().slot      = state_store->physical_slot(state);
    lane().cursor    = frontier;
    lane().kv_tokens = frontier;
    try {
        install_request(lane().request);
        if (lane().request.vision) { attach_vision(); }
    } catch (...) {
        undo_bind();
        throw;
    }
    lane().bound = true;
    lane().stats = fresh_stats();
}

void P3Program::fork_lane(Request request, CachePoint point) {
    CacheRecord* record = find_cache(point);
    if (record == nullptr || record->role != runtime::CheckpointRole::SharedPrefix || lane().bound) {
        throw std::logic_error("V3 P5 shared checkpoint cannot be copied");
    }
    if (request.tokens.size() < record->frontier) {
        throw std::logic_error("V3 P5 copy prompt is shorter than the checkpoint");
    }
    const std::uint32_t frontier    = record->frontier;
    const KvEntitlement entitlement = kv_entitlement(request);
    // The lane owns each piece as soon as it exists, so a failure releases all of them.
    try {
        auto text_dest = text_kv_addresses->create_inactive();
        if (!text_dest) { throw std::runtime_error("KV address reservation failed"); }
        lane().kv      = *text_dest;
        auto text_fork = text_kv_addresses->prepare_prefix_fork(
            record->kv, *text_dest, frontier, growth_beyond(entitlement.text, kv_pages(frontier)),
            static_cast<std::int32_t>(focus_));
        if (text_fork.needs_tail_copy()) {
            (void)decoder->text_kv.page_pool().copy_page(
                text_kv_addresses->prefix_fork_tail_source(text_fork),
                text_kv_addresses->prefix_fork_tail_destination(text_fork), device.stream);
        }
        text_kv_addresses->commit_prefix_fork(std::move(text_fork), device.stream);
        if (record->mtp.valid() && mtp_addresses && decoder->mtp_cache() != nullptr) {
            auto created = mtp_addresses->create_inactive();
            if (!created) { throw std::runtime_error("MTP KV address reservation failed"); }
            lane().mtp_kv = *created;
            auto mtp_fork = mtp_addresses->prepare_prefix_fork(
                record->mtp, *created, frontier, growth_beyond(entitlement.mtp, kv_pages(frontier)),
                static_cast<std::int32_t>(focus_));
            if (mtp_fork.needs_tail_copy()) {
                (void)decoder->mtp_cache()->page_pool().copy_page(
                    mtp_addresses->prefix_fork_tail_source(mtp_fork),
                    mtp_addresses->prefix_fork_tail_destination(mtp_fork), device.stream);
            }
            mtp_addresses->commit_prefix_fork(std::move(mtp_fork), device.stream);
        }
        auto state_dest = state_store->reserve_destination();
        if (!state_dest) { throw std::runtime_error("state slot reservation failed"); }
        lane().state           = *state_dest;
        const auto source_slot = state_store->physical_slot(record->state);
        const auto dest_slot   = state_store->physical_slot(*state_dest);
        state_images->copy_slot(source_slot, dest_slot, device.stream);
        (void)state_store->begin_fork(record->state, *state_dest);
        state_store->commit_fork(record->state, *state_dest);
        lane().request      = std::move(request);
        lane().slot         = dest_slot;
        lane().fork_pending = false;
        lane().cursor       = frontier;
        lane().kv_tokens    = frontier;
        install_request(lane().request);
        if (lane().request.vision) { attach_vision(); }
    } catch (...) {
        undo_bind();
        throw;
    }
    lane().bound = true;
    lane().stats = fresh_stats();
}

std::optional<P3Program::CachePoint> P3Program::capture_prefix() {
    if (!lane().bound || lane().pending || lane().fork_pending ||
        !lane().request.summary.publish_continuation) {
        return std::nullopt;
    }
    const std::uint32_t frontier = lane().kv_tokens;
    if (frontier == 0 || frontier != lane().cursor ||
        frontier >= lane().request.summary.prompt_tokens ||
        std::find(lane().request.capture_frontiers.begin(), lane().request.capture_frontiers.end(),
                  frontier) == lane().request.capture_frontiers.end()) {
        return std::nullopt;
    }
    CacheRecord* slot = nullptr;
    for (auto& record : cache_slots) {
        if (!record.occupied) {
            slot = &record;
            break;
        }
    }
    if (slot == nullptr) { return std::nullopt; }
    ResidentPrefixIdentity identity = lane().request.identity;
    try {
        identity.truncate(frontier);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    const bool with_mtp = lane().mtp_kv.valid() && mtp_addresses && decoder->mtp_cache() != nullptr;
    auto state_dest     = state_store->reserve_destination();
    auto text_dest      = text_kv_addresses->create_inactive();
    std::optional<program::KVAddressSpaceHandle> mtp_dest;
    if (with_mtp) { mtp_dest = mtp_addresses->create_inactive(); }
    const auto drop_reservations = [&]() noexcept {
        if (state_dest && state_store->valid(*state_dest)) { (void)state_store->release(*state_dest); }
        if (text_dest && text_kv_addresses->valid(*text_dest)) {
            (void)text_kv_addresses->release(*text_dest);
        }
        if (mtp_dest && mtp_addresses->valid(*mtp_dest)) { (void)mtp_addresses->release(*mtp_dest); }
    };
    // The capture is optional: without a slot or an address the lane keeps going uncaptured.
    if (!state_dest || !text_dest || (with_mtp && !mtp_dest)) {
        drop_reservations();
        return std::nullopt;
    }
    // The record gets an immutable view of the lane's prefix: full pages are shared and a partial
    // tail is copied. The lane keeps its address, row, and mapped suffix. The protected range
    // stays on the lane, which never truncates below a frontier it has passed.
    bool published = false;
    try {
        text_kv_addresses->set_checkpoint_requirement(lane().kv, frontier);
        auto text_view =
            text_kv_addresses->prepare_active_prefix_view(lane().kv, *text_dest, frontier);
        if (text_view.needs_tail_copy()) {
            (void)decoder->text_kv.page_pool().copy_page(
                text_kv_addresses->active_prefix_view_tail_source(text_view),
                text_kv_addresses->active_prefix_view_tail_destination(text_view), device.stream);
        }
        std::optional<program::KVActivePrefixViewReservation> mtp_view;
        if (with_mtp) {
            mtp_addresses->set_checkpoint_requirement(lane().mtp_kv, frontier);
            mtp_view.emplace(
                mtp_addresses->prepare_active_prefix_view(lane().mtp_kv, *mtp_dest, frontier));
            if (mtp_view->needs_tail_copy()) {
                (void)decoder->mtp_cache()->page_pool().copy_page(
                    mtp_addresses->active_prefix_view_tail_source(*mtp_view),
                    mtp_addresses->active_prefix_view_tail_destination(*mtp_view), device.stream);
            }
        }
        text_kv_addresses->commit_active_prefix_view(std::move(text_view));
        published = true;
        if (mtp_view) { mtp_addresses->commit_active_prefix_view(std::move(*mtp_view)); }
        const auto frozen_text = *text_dest;
        const program::KVAddressSpaceHandle frozen_mtp =
            with_mtp ? *mtp_dest : program::KVAddressSpaceHandle{};
        const auto frozen_state = lane().state;
        state_store->freeze(frozen_state);
        state_store->retain_checkpoint_reference(frozen_state);
        const auto source_slot = state_store->physical_slot(frozen_state);
        const auto dest_slot   = state_store->physical_slot(*state_dest);
        state_images->copy_slot(source_slot, dest_slot, device.stream);
        (void)state_store->begin_fork(frozen_state, *state_dest);
        state_store->commit_fork(frozen_state, *state_dest);
        lane().state        = *state_dest;
        lane().slot         = dest_slot;
        lane().fork_pending = false;
        slot->tokens.assign(lane().request.tokens.begin(),
                            lane().request.tokens.begin() + static_cast<std::ptrdiff_t>(frontier));
        slot->identity   = std::move(identity);
        slot->kv         = frozen_text;
        slot->mtp        = frozen_mtp;
        slot->state      = frozen_state;
        slot->frontier   = frontier;
        slot->role       = runtime::CheckpointRole::SharedPrefix;
        slot->occupied   = true;
        slot->generation = ++cache_clock;
        if (slot->generation == 0) { slot->generation = ++cache_clock; }
        lane().capture_offered = false;
        return CachePoint{static_cast<std::uint32_t>(slot - cache_slots.data()), slot->generation};
    } catch (const std::exception& error) {
        if (published) { throw; }
        drop_reservations();
        throw std::runtime_error(std::string("capture_prefix: ") + error.what());
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
