#include "models/gemma4/program/prefix_cache.h"

#include "models/gemma4/program/sliding_ring.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

constexpr PositionKey kImageBit = PositionKey{1} << 63U;

std::uint64_t mix(std::uint64_t value) noexcept {
    value ^= value >> 33U;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33U;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33U;
    return value;
}

std::uint64_t image_digest(const Gemma4PreparedImage& image) noexcept {
    std::uint64_t digest =
        mix(image.patches.size() ^ (static_cast<std::uint64_t>(image.valid) << 32U));
    const auto* bytes    = reinterpret_cast<const unsigned char*>(image.patches.data());
    const std::size_t size = image.patches.size() * sizeof(image.patches[0]);
    std::size_t at         = 0;
    for (; at + 8 <= size; at += 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes + at, 8);
        digest = mix(digest ^ word) + at;
    }
    for (; at < size; ++at) { digest = mix(digest ^ bytes[at]); }
    return digest;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 prefix cache: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<PositionKey> prompt_keys(std::span<const TokenId> tokens,
                                     std::span<const Gemma4PreparedImage> images,
                                     std::span<const std::int32_t> image_scatter) {
    std::vector<PositionKey> keys(tokens.size());
    std::transform(tokens.begin(), tokens.end(), keys.begin(), token_key);
    std::size_t column = 0;
    for (const Gemma4PreparedImage& image : images) {
        const std::uint64_t digest = image_digest(image);
        for (int index = 0; index < image.valid && column < image_scatter.size(); ++index) {
            const auto position = static_cast<std::size_t>(image_scatter[column++]);
            if (position >= keys.size()) {
                throw std::invalid_argument("gemma4 image scatter is outside the prompt");
            }
            keys[position] = kImageBit | (mix(digest + static_cast<std::uint64_t>(index)) >> 1U);
        }
    }
    return keys;
}

PrefixCache::PrefixCache(Config config)
    : store_(config.store), config_(std::move(config)),
      table_host_(static_cast<std::size_t>(config_.table_pages) * sizeof(std::int32_t)) {
    if (config_.table == nullptr || config_.table_pages <= 0) {
        throw std::invalid_argument("gemma4 prefix cache needs a page table");
    }
}

PrefixCache::~PrefixCache() = default;

void PrefixCache::prepare_write(std::uint32_t first, std::uint32_t count) {
    if (count == 0) { return; }
    const auto low  = static_cast<std::size_t>(first / kKvPageTokens);
    const auto high = static_cast<std::size_t>((first + count - 1U) / kKvPageTokens);
    if (high >= static_cast<std::size_t>(config_.table_pages)) {
        throw std::logic_error("gemma4 write is past the page table");
    }
    for (std::size_t page = low; page <= high; ++page) {
        if (page < live_pages_.size()) {
            const int held = live_pages_[page];
            if (!store_.shared(held)) { continue; }
            const int owned = store_.allocate_page();
            if (page == low && first % kKvPageTokens != 0) { store_.copy_page(held, owned); }
            store_.release(held);
            live_pages_[page] = owned;
        } else {
            while (live_pages_.size() < page) { live_pages_.push_back(store_.allocate_page()); }
            live_pages_.push_back(store_.allocate_page());
        }
        table_dirty_ = true;
    }
    if (table_dirty_) { upload_table(); }
}

void PrefixCache::upload_table() {
    cudaStream_t stream = config_.store.stream;
    check(cudaStreamSynchronize(stream), "table sync");
    auto* host = static_cast<std::int32_t*>(table_host_.data());
    std::fill_n(host, config_.table_pages, 0);
    std::copy(live_pages_.begin(), live_pages_.end(), host);
    check(cudaMemcpyAsync(config_.table, host, table_host_.size(), cudaMemcpyHostToDevice, stream),
          "table upload");
    table_dirty_ = false;
}

void PrefixCache::wrote(std::uint32_t first, std::span<const PositionKey> keys) {
    if (first > resident_.size()) { throw std::logic_error("gemma4 write left a gap"); }
    resident_.resize(first);
    resident_.insert(resident_.end(), keys.begin(), keys.end());
    if (keys.empty()) { return; }
    const std::int64_t last =
        static_cast<std::int64_t>(first) + static_cast<std::int64_t>(keys.size()) - 1;
    if (first == 0) {
        ring_low_  = 0;
        ring_high_ = last;
    } else {
        ring_high_ = std::max(ring_high_, last);
    }
}

void PrefixCache::wrote_tokens(std::uint32_t first, std::span<const TokenId> tokens) {
    std::vector<PositionKey> keys(tokens.size());
    std::transform(tokens.begin(), tokens.end(), keys.begin(), token_key);
    wrote(first, keys);
}

void PrefixCache::truncate(std::uint32_t length) {
    if (length < resident_.size()) { resident_.resize(length); }
}

std::uint32_t PrefixCache::live_reuse(std::span<const PositionKey> keys) const {
    if (keys.size() < 2) { return 0; }
    const std::size_t limit = std::min(resident_.size(), keys.size() - 1);
    std::size_t length      = 0;
    while (length < limit && resident_[length] == keys[length]) { ++length; }
    if (length == 0 || config_.store.ring.empty()) { return static_cast<std::uint32_t>(length); }
    // The sliding window before the reuse point must still be in the ring.
    const std::int64_t oldest =
        std::max<std::int64_t>(0, static_cast<std::int64_t>(length) - (config_.store.window - 1));
    if (oldest < ring_low_ ||
        oldest + std::int64_t{config_.store.ring_pages} * kKvPageTokens <= ring_high_) {
        return 0;
    }
    return static_cast<std::uint32_t>(length);
}

PrefixCache::Source PrefixCache::best(std::span<const PositionKey> keys) const {
    Source source{.reuse = live_reuse(keys)};
    if (!config_.history || keys.size() < 2) { return source; }
    const auto match = store_.best_match(keys, static_cast<std::uint32_t>(keys.size() - 1));
    if (!match || match->reuse < kMinCheckpointTokens) { return source; }
    // A restore (and the capture of the state it displaces) costs tens of milliseconds on the
    // device and more from the host, so it must save that much prefill over the live state.
    const std::uint32_t gain = match->on_host ? kMinHostRestoreGain : kMinCheckpointTokens;
    if (match->reuse >= source.reuse + gain) {
        source = {.reuse = match->reuse, .match = match};
    }
    return source;
}

void PrefixCache::bind(const Source& source, std::span<const PositionKey> keys) {
    store_.touch(keys);
    const std::uint32_t kept = source.match ? 0U : source.reuse;
    const auto held          = static_cast<std::uint32_t>(resident_.size());
    // A displaced state less than kMinCheckpointTokens past the kept prefix (a repeated prompt,
    // a cut reply) could never win over the live state in best(), and its capture can demote a
    // device checkpoint to the host before this request's prefill.
    if (kept + kMinCheckpointTokens <= held) {
        capture(held, source.match ? std::optional{source.match->id} : std::nullopt);
    }
    if (!source.match) {
        truncate(kept);
        const auto pages = static_cast<std::size_t>((kept + kKvPageTokens - 1U) / kKvPageTokens);
        while (live_pages_.size() > pages) {
            store_.release(live_pages_.back());
            live_pages_.pop_back();
            table_dirty_ = true;
        }
        return;
    }
    clear();
    ring_low_  = store_.restore(*source.match, live_pages_);
    ring_high_ = static_cast<std::int64_t>(source.reuse) - 1;
    resident_.assign(keys.begin(), keys.begin() + source.reuse);
    table_dirty_ = true;
    upload_table();
}

void PrefixCache::capture(std::uint32_t length, std::optional<std::uint64_t> keep) {
    if (!config_.history || length < kMinCheckpointTokens || length > resident_.size()) { return; }
    std::vector<PositionKey> keys(resident_.begin(), resident_.begin() + length);
    if (store_.covers(keys)) { return; }
    store_.capture(std::move(keys), live_pages_, ring_low_, ring_high_, keep);
}

void PrefixCache::clear() noexcept {
    for (const int page : live_pages_) { store_.release(page); }
    live_pages_.clear();
    resident_.clear();
    ring_low_    = 0;
    ring_high_   = -1;
    table_dirty_ = true;
}

} // namespace ninfer::models::gemma4
