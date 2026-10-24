#include "models/gemma4/program/prefix_store.h"

#include "models/gemma4/program/sliding_ring.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::gemma4 {
namespace {

// Positions before the window that a snapshot also keeps, so a prompt that diverges up to this
// far before the frontier (a re-rendered reply, a trimmed tail) still resumes from it.
constexpr int kSnapshotSlack = 256;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 prefix store: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

// First-fit ranges of the host arena, sorted by offset.
class Ranges {
public:
    explicit Ranges(std::size_t size) {
        if (size > 0) { free_.push_back({0, size}); }
    }

    std::optional<std::size_t> take(std::size_t bytes) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->second < bytes) { continue; }
            const std::size_t at = it->first;
            it->first += bytes;
            it->second -= bytes;
            if (it->second == 0) { free_.erase(it); }
            return at;
        }
        return std::nullopt;
    }

    void give(std::size_t at, std::size_t bytes) {
        auto it = std::lower_bound(free_.begin(), free_.end(), std::pair{at, std::size_t{0}});
        it      = free_.insert(it, {at, bytes});
        if (const auto next = std::next(it);
            next != free_.end() && it->first + it->second == next->first) {
            it->second += next->second;
            free_.erase(next);
        }
        if (it != free_.begin()) {
            const auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                free_.erase(it);
            }
        }
    }

private:
    std::vector<std::pair<std::size_t, std::size_t>> free_;
};

struct Point {
    std::uint64_t id = 0;
    std::vector<PositionKey> keys;
    std::uint32_t min_reuse  = 1;
    int first_page           = 0;
    int snapshot_count       = 0;
    std::int64_t valid_from  = 0;
    std::uint64_t last_use   = 0;
    int slot                 = -1;
    std::vector<int> pages;
    std::optional<std::size_t> host_at;
    std::size_t host_bytes = 0;
};

} // namespace

struct PrefixStore::Impl {
    Config config;
    int snapshot_pages     = 0;
    std::size_t slot_bytes = 0;
    std::vector<std::size_t> ring_offset;
    std::size_t page_set_bytes = 0;
    std::vector<std::size_t> global_offset;
    std::vector<int> refs;
    std::vector<int> free_pages;
    DeviceBuffer slots;
    std::vector<bool> slot_used;
    std::unique_ptr<PinnedHostBuffer> host;
    Ranges host_ranges{0};
    std::vector<Point> points;
    std::uint64_t next_id = 1;
    std::uint64_t clock   = 0;
    std::optional<std::uint64_t> pinned;
    Stats stats;

    [[nodiscard]] std::uint8_t* slot_base(int slot) const {
        return static_cast<std::uint8_t*>(slots.p) + static_cast<std::size_t>(slot) * slot_bytes;
    }
    [[nodiscard]] std::uint8_t* host_base(const Point& point) const {
        return static_cast<std::uint8_t*>(host->data()) + *point.host_at;
    }

    [[nodiscard]] std::size_t index_of(std::uint64_t id) const {
        for (std::size_t index = 0; index < points.size(); ++index) {
            if (points[index].id == id) { return index; }
        }
        throw std::logic_error("gemma4 prefix store lost a checkpoint");
    }

    // Copies `count` ring pages from logical page `first` between the ring and a snapshot laid
    // out like a device slot (per ring tensor, `snapshot_pages` pages).
    void ring_transfer(std::uint8_t* snapshot, int first, int count, bool save,
                       cudaMemcpyKind kind) {
        for (std::size_t tensor = 0; tensor < config.ring.size(); ++tensor) {
            const PagedTensor& ring = config.ring[tensor];
            for (int done = 0; done < count;) {
                const int physical = (first + done) % config.ring_pages;
                const int run      = std::min(count - done, config.ring_pages - physical);
                std::uint8_t* live = ring.base + static_cast<std::size_t>(physical) * ring.page_bytes;
                std::uint8_t* kept = snapshot + ring_offset[tensor] +
                                     static_cast<std::size_t>(done) * ring.page_bytes;
                const std::size_t bytes = static_cast<std::size_t>(run) * ring.page_bytes;
                check(cudaMemcpyAsync(save ? kept : live, save ? live : kept, bytes, kind,
                                      config.stream),
                      "ring copy");
                done += run;
            }
        }
    }

    // Copies global pages between the pool (`pages`) and host memory laid out per tensor.
    void global_transfer(std::uint8_t* host_pages, std::span<const int> pages, bool save) {
        const auto count = pages.size();
        for (std::size_t tensor = 0; tensor < config.global.size(); ++tensor) {
            const PagedTensor& global = config.global[tensor];
            std::uint8_t* region      = host_pages + global_offset[tensor] * count;
            for (std::size_t done = 0; done < count;) {
                std::size_t run = 1;
                while (done + run < count && pages[done + run] == pages[done] + static_cast<int>(run)) {
                    ++run;
                }
                std::uint8_t* device =
                    global.base + static_cast<std::size_t>(pages[done]) * global.page_bytes;
                std::uint8_t* kept = region + done * global.page_bytes;
                check(cudaMemcpyAsync(save ? kept : device, save ? device : kept,
                                      run * global.page_bytes,
                                      save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                                      config.stream),
                      "page copy");
                done += run;
            }
        }
    }

    void release_pages(Point& point) {
        for (const int page : point.pages) { release(page); }
        point.pages.clear();
    }

    void release(int page) noexcept {
        if (--refs[static_cast<std::size_t>(page)] == 0) { free_pages.push_back(page); }
    }

    void drop(std::size_t index) {
        Point& point = points[index];
        release_pages(point);
        if (point.slot >= 0) { slot_used[static_cast<std::size_t>(point.slot)] = false; }
        if (point.host_at) { host_ranges.give(*point.host_at, point.host_bytes); }
        points.erase(points.begin() + static_cast<std::ptrdiff_t>(index));
        ++stats.drops;
    }

    // Host bytes for `bytes`, dropping the least recently used host checkpoints to make room.
    std::optional<std::size_t> host_take(std::size_t bytes) {
        if (!host || bytes > host->size()) { return std::nullopt; }
        while (true) {
            if (const auto at = host_ranges.take(bytes)) { return at; }
            std::optional<std::size_t> oldest;
            for (std::size_t index = 0; index < points.size(); ++index) {
                const Point& point = points[index];
                if (!point.host_at || point.id == pinned) { continue; }
                if (!oldest || point.last_use < points[*oldest].last_use) { oldest = index; }
            }
            if (!oldest) { return std::nullopt; }
            drop(*oldest);
        }
    }

    // Moves a device checkpoint to the host arena, or drops it when the arena has no room.
    void demote(std::size_t index) {
        const std::size_t bytes = slot_bytes + points[index].pages.size() * page_set_bytes;
        const std::uint64_t id  = points[index].id;
        const auto at           = host_take(bytes);
        index                   = index_of(id);
        if (!at) {
            drop(index);
            return;
        }
        Point& point     = points[index];
        point.host_at    = at;
        point.host_bytes = bytes;
        std::uint8_t* to = host_base(point);
        if (slot_bytes > 0) {
            check(cudaMemcpyAsync(to, slot_base(point.slot), slot_bytes, cudaMemcpyDeviceToHost,
                                  config.stream),
                  "snapshot demotion");
        }
        global_transfer(to + slot_bytes, point.pages, true);
        slot_used[static_cast<std::size_t>(point.slot)] = false;
        point.slot                                      = -1;
        release_pages(point);
        ++stats.demotions;
    }

    // The least recently used device checkpoint (one that owns a page alone when `pages`).
    [[nodiscard]] std::optional<std::size_t> oldest_device(bool pages) const {
        std::optional<std::size_t> oldest;
        for (std::size_t index = 0; index < points.size(); ++index) {
            const Point& point = points[index];
            if (point.slot < 0 || point.id == pinned) { continue; }
            if (pages && std::none_of(point.pages.begin(), point.pages.end(), [&](int page) {
                    return refs[static_cast<std::size_t>(page)] == 1;
                })) {
                continue;
            }
            if (!oldest || point.last_use < points[*oldest].last_use) { oldest = index; }
        }
        return oldest;
    }
};

PrefixStore::PrefixStore(Config config) : impl_(std::make_unique<Impl>()) {
    Impl& impl = *impl_;
    if (config.pool_pages <= 0 || config.global.empty()) {
        throw std::invalid_argument("gemma4 prefix store needs a page pool");
    }
    impl.config         = std::move(config);
    const Config& setup = impl.config;
    impl.snapshot_pages = setup.ring.empty() ? 0 : snapshot_pages(setup.ring_pages, setup.window);
    for (const PagedTensor& ring : setup.ring) {
        impl.ring_offset.push_back(impl.slot_bytes);
        impl.slot_bytes += ring.page_bytes * static_cast<std::size_t>(impl.snapshot_pages);
    }
    for (const PagedTensor& global : setup.global) {
        impl.global_offset.push_back(impl.page_set_bytes);
        impl.page_set_bytes += global.page_bytes;
    }
    impl.refs.assign(static_cast<std::size_t>(setup.pool_pages), 0);
    for (int page = setup.pool_pages - 1; page >= 0; --page) { impl.free_pages.push_back(page); }
    if (setup.device_slots > 0 && impl.slot_bytes > 0) {
        impl.slots = DeviceBuffer(impl.slot_bytes * setup.device_slots);
    }
    impl.slot_used.assign(setup.device_slots, false);
    if (setup.host_bytes > 0) {
        impl.host        = std::make_unique<PinnedHostBuffer>(setup.host_bytes);
        impl.host_ranges = Ranges(setup.host_bytes);
    }
}

PrefixStore::~PrefixStore() = default;

int PrefixStore::snapshot_pages(int ring_pages, int window) {
    const int wanted = (window - 1 + kSnapshotSlack + kKvPageTokens - 1) / kKvPageTokens + 1;
    return std::min(ring_pages, wanted);
}

std::size_t PrefixStore::snapshot_bytes(std::span<const std::size_t> ring_page_bytes,
                                        int ring_pages, int window) {
    std::size_t bytes = 0;
    for (const std::size_t page : ring_page_bytes) { bytes += page; }
    return bytes * static_cast<std::size_t>(snapshot_pages(ring_pages, window));
}

int PrefixStore::allocate_page() {
    Impl& impl = *impl_;
    while (impl.free_pages.empty()) {
        const auto oldest = impl.oldest_device(true);
        if (!oldest) { throw std::runtime_error("gemma4 KV page pool is exhausted"); }
        impl.demote(*oldest);
    }
    const int page = impl.free_pages.back();
    impl.free_pages.pop_back();
    impl.refs[static_cast<std::size_t>(page)] = 1;
    return page;
}

void PrefixStore::retain(int page) noexcept { ++impl_->refs[static_cast<std::size_t>(page)]; }

void PrefixStore::release(int page) noexcept { impl_->release(page); }

bool PrefixStore::shared(int page) const noexcept {
    return impl_->refs[static_cast<std::size_t>(page)] > 1;
}

void PrefixStore::copy_page(int from, int to) {
    for (const PagedTensor& global : impl_->config.global) {
        check(cudaMemcpyAsync(global.base + static_cast<std::size_t>(to) * global.page_bytes,
                              global.base + static_cast<std::size_t>(from) * global.page_bytes,
                              global.page_bytes, cudaMemcpyDeviceToDevice, impl_->config.stream),
              "page copy-on-write");
    }
}

std::optional<PrefixStore::Match> PrefixStore::best_match(std::span<const PositionKey> prompt,
                                                          std::uint32_t limit) const {
    std::optional<Match> best;
    std::uint64_t best_use = 0;
    for (const Point& point : impl_->points) {
        const std::size_t bound = std::min<std::size_t>({limit, prompt.size(), point.keys.size()});
        std::size_t length      = 0;
        while (length < bound && prompt[length] == point.keys[length]) { ++length; }
        if (length < point.min_reuse) { continue; }
        const Match match{.id      = point.id,
                          .reuse   = static_cast<std::uint32_t>(length),
                          .on_host = point.slot < 0};
        const bool better = !best || match.reuse > best->reuse ||
                            (match.reuse == best->reuse &&
                             (best->on_host && !match.on_host ||
                              (best->on_host == match.on_host && point.last_use > best_use)));
        if (better) {
            best     = match;
            best_use = point.last_use;
        }
    }
    return best;
}

bool PrefixStore::covers(std::span<const PositionKey> keys) const {
    return std::any_of(impl_->points.begin(), impl_->points.end(), [&](const Point& point) {
        return std::equal(point.keys.begin(), point.keys.end(), keys.begin(), keys.end());
    });
}

void PrefixStore::touch(std::span<const PositionKey> prompt) {
    for (Point& point : impl_->points) {
        if (point.keys.size() <= prompt.size() &&
            std::equal(point.keys.begin(), point.keys.end(), prompt.begin())) {
            point.last_use = ++impl_->clock;
        }
    }
}

bool PrefixStore::capture(std::vector<PositionKey> keys, std::span<const int> live_pages,
                          std::int64_t ring_low, std::int64_t ring_high,
                          std::optional<std::uint64_t> keep) {
    Impl& impl                = *impl_;
    const Config& setup       = impl.config;
    struct Unpin {
        std::optional<std::uint64_t>& pinned;
        ~Unpin() { pinned.reset(); }
    } unpin{impl.pinned};
    impl.pinned = keep;
    const auto frontier       = static_cast<std::int64_t>(keys.size());
    const auto pages          = static_cast<std::size_t>((frontier + kKvPageTokens - 1) / kKvPageTokens);
    if (frontier == 0 || pages > live_pages.size() || ring_high < frontier - 1) { return false; }
    Point point;
    point.id = impl.next_id;
    if (!setup.ring.empty()) {
        const int last = static_cast<int>((frontier - 1) / kKvPageTokens);
        point.first_page     = std::max(0, last - impl.snapshot_pages + 1);
        point.snapshot_count = last - point.first_page + 1;
        point.valid_from     = std::max<std::int64_t>(
            {std::int64_t{point.first_page} * kKvPageTokens, ring_low,
             ring_high - std::int64_t{setup.ring_pages} * kKvPageTokens + 1, 0});
        point.min_reuse = point.valid_from == 0
                              ? 1U
                              : static_cast<std::uint32_t>(point.valid_from + setup.window - 1);
        if (point.min_reuse > frontier) { return false; }
    }
    auto slot = std::find(impl.slot_used.begin(), impl.slot_used.end(), false);
    if (slot == impl.slot_used.end() && !impl.slot_used.empty()) {
        if (const auto oldest = impl.oldest_device(false)) { impl.demote(*oldest); }
        slot = std::find(impl.slot_used.begin(), impl.slot_used.end(), false);
    }
    point.pages.assign(live_pages.begin(), live_pages.begin() + static_cast<std::ptrdiff_t>(pages));
    for (const int page : point.pages) { retain(page); }
    if (slot != impl.slot_used.end()) {
        point.slot = static_cast<int>(slot - impl.slot_used.begin());
        *slot      = true;
        impl.ring_transfer(impl.slot_base(point.slot), point.first_page, point.snapshot_count,
                           true, cudaMemcpyDeviceToDevice);
    } else {
        const std::size_t bytes = impl.slot_bytes + pages * impl.page_set_bytes;
        const auto at           = impl.host_take(bytes);
        if (!at) {
            impl.release_pages(point);
            return false;
        }
        point.host_at    = at;
        point.host_bytes = bytes;
        std::uint8_t* to = impl.host_base(point);
        impl.ring_transfer(to, point.first_page, point.snapshot_count, true,
                           cudaMemcpyDeviceToHost);
        impl.global_transfer(to + impl.slot_bytes, point.pages, true);
        impl.release_pages(point);
    }
    point.keys     = std::move(keys);
    point.last_use = ++impl.clock;
    ++impl.next_id;
    impl.points.push_back(std::move(point));
    ++impl.stats.captures;
    return true;
}

std::int64_t PrefixStore::restore(const Match& match, std::vector<int>& live_pages) {
    Impl& impl = *impl_;
    if (!live_pages.empty()) { throw std::logic_error("gemma4 restore needs an empty live state"); }
    const auto pages = static_cast<std::size_t>((match.reuse + kKvPageTokens - 1) / kKvPageTokens);
    impl.pinned      = match.id;
    std::size_t index = impl.index_of(match.id);
    if (impl.points[index].slot >= 0) {
        Point& point = impl.points[index];
        for (std::size_t page = 0; page < pages; ++page) {
            retain(point.pages[page]);
            live_pages.push_back(point.pages[page]);
        }
        impl.ring_transfer(impl.slot_base(point.slot), point.first_page, point.snapshot_count,
                           false, cudaMemcpyDeviceToDevice);
        ++impl.stats.device_hits;
    } else {
        for (std::size_t page = 0; page < pages; ++page) { live_pages.push_back(allocate_page()); }
        index        = impl.index_of(match.id);
        Point& point = impl.points[index];
        std::uint8_t* from = impl.host_base(point);
        impl.ring_transfer(from, point.first_page, point.snapshot_count, false,
                           cudaMemcpyHostToDevice);
        const std::size_t stored = (point.host_bytes - impl.slot_bytes) / impl.page_set_bytes;
        // Host pages are laid out per tensor for all `stored` pages; restore the first `pages`.
        for (std::size_t tensor = 0; tensor < impl.config.global.size(); ++tensor) {
            const PagedTensor& global = impl.config.global[tensor];
            std::uint8_t* region      = from + impl.slot_bytes + impl.global_offset[tensor] * stored;
            for (std::size_t page = 0; page < pages; ++page) {
                check(cudaMemcpyAsync(
                          global.base + static_cast<std::size_t>(live_pages[page]) * global.page_bytes,
                          region + page * global.page_bytes, global.page_bytes,
                          cudaMemcpyHostToDevice, impl.config.stream),
                      "page restore");
            }
        }
        ++impl.stats.host_hits;
    }
    Point& point   = impl.points[impl.index_of(match.id)];
    point.last_use = ++impl.clock;
    impl.pinned.reset();
    return point.valid_from;
}

PrefixStore::Stats PrefixStore::stats() const noexcept {
    Stats stats = impl_->stats;
    for (const Point& point : impl_->points) {
        if (point.slot >= 0) {
            ++stats.device_points;
        } else {
            ++stats.host_points;
        }
    }
    return stats;
}

} // namespace ninfer::models::gemma4
