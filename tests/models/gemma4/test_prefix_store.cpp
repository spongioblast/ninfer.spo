// PrefixStore on tiny device buffers: a demoted checkpoint restores its global pages and ring
// snapshot byte for byte, a capture that makes room for the displaced state never drops the
// checkpoint about to be restored, and touch() keeps a checkpoint the prompts extend from being
// the first evicted.

#include "models/gemma4/program/prefix_store.h"
#include "models/gemma4/program/sliding_ring.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::gemma4::kKvPageTokens;
using ninfer::models::gemma4::PositionKey;
using ninfer::models::gemma4::PrefixStore;

constexpr int kPoolPages        = 32;
constexpr int kRingPages        = 8;
constexpr int kWindow           = 64;
constexpr std::size_t kPage     = 256;
constexpr std::uint32_t kTokens = 300;

int failures = 0;

void check(bool ok, const std::string& label) {
    std::cout << (ok ? "ok   " : "FAIL ") << label << '\n';
    if (!ok) { ++failures; }
}

void cuda(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

struct Buffers {
    std::uint8_t* global = nullptr;
    std::uint8_t* ring   = nullptr;
    cudaStream_t stream  = nullptr;
    Buffers() {
        cuda(cudaMalloc(&global, kPoolPages * kPage));
        cuda(cudaMalloc(&ring, kRingPages * kPage));
        cuda(cudaStreamCreate(&stream));
    }
    ~Buffers() {
        cudaFree(global);
        cudaFree(ring);
        cudaStreamDestroy(stream);
    }
    Buffers(const Buffers&)            = delete;
    Buffers& operator=(const Buffers&) = delete;
};

// Device slot of one snapshot plus five global pages, so the host arena holds two checkpoints.
std::size_t point_bytes() {
    const int snapshot = PrefixStore::snapshot_pages(kRingPages, kWindow);
    const auto pages   = (kTokens + kKvPageTokens - 1) / kKvPageTokens;
    return static_cast<std::size_t>(snapshot) * kPage + pages * kPage;
}

PrefixStore make_store(const Buffers& buffers) {
    PrefixStore::Config config;
    config.global       = {{buffers.global, kPage}};
    config.ring         = {{buffers.ring, kPage}};
    config.pool_pages   = kPoolPages;
    config.ring_pages   = kRingPages;
    config.window       = kWindow;
    config.device_slots = 1;
    config.host_bytes   = 2 * point_bytes();
    config.stream       = buffers.stream;
    return PrefixStore(std::move(config));
}

std::vector<PositionKey> keys_from(PositionKey first, std::uint32_t count) {
    std::vector<PositionKey> keys(count);
    std::iota(keys.begin(), keys.end(), first);
    return keys;
}

// Writes a sequence of kTokens positions tagged `tag` into fresh pool pages and the ring,
// captures it, and gives the pages back as the live sequence would on the next request.
void capture_sequence(PrefixStore& store, const Buffers& buffers, PositionKey first,
                      std::uint8_t tag, std::optional<std::uint64_t> keep = std::nullopt) {
    std::vector<int> live;
    for (std::uint32_t page = 0; page * kKvPageTokens < kTokens; ++page) {
        live.push_back(store.allocate_page());
        cuda(cudaMemsetAsync(buffers.global + live.back() * kPage, tag + page, kPage, buffers.stream));
    }
    for (int page = 0; page < kRingPages; ++page) {
        cuda(cudaMemsetAsync(buffers.ring + page * kPage, tag + 0x40 + page, kPage, buffers.stream));
    }
    store.capture(keys_from(first, kTokens), live, 0, kTokens - 1, keep);
    for (const int page : live) { store.release(page); }
}

std::vector<PositionKey> prompt_after(PositionKey first) {
    auto prompt = keys_from(first, kTokens);
    prompt.push_back(999999);
    return prompt;
}

// A, B, C captured in turn with one device slot: A and B are on the host (full), C on the device.
void fill(PrefixStore& store, const Buffers& buffers) {
    capture_sequence(store, buffers, 1000, 0x10);
    capture_sequence(store, buffers, 2000, 0x20);
    capture_sequence(store, buffers, 3000, 0x30);
}

void restores_kept_checkpoint() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    const auto match = store.best_match(prompt_after(1000), kTokens);
    check(match && match->on_host && match->reuse == kTokens, "A is matched on the host");
    if (!match) { return; }
    capture_sequence(store, buffers, 4000, 0x40, match->id);
    check(store.stats().drops == 1, "making room for D drops one host checkpoint");
    check(store.best_match(prompt_after(2000), kTokens) == std::nullopt, "B, not A, was dropped");
    std::vector<int> live;
    const std::int64_t valid_from = store.restore(*match, live);
    cuda(cudaStreamSynchronize(buffers.stream));
    bool same = valid_from == 0 && live.size() == 5;
    std::vector<std::uint8_t> host(kPage);
    for (std::size_t page = 0; same && page < live.size(); ++page) {
        cuda(cudaMemcpy(host.data(), buffers.global + live[page] * kPage, kPage, cudaMemcpyDeviceToHost));
        const auto tag = static_cast<std::uint8_t>(0x10 + page);
        same           = host[0] == tag && host[kPage - 1] == tag;
    }
    for (int page = 0; same && page < 5; ++page) {
        cuda(cudaMemcpy(host.data(), buffers.ring + page * kPage, kPage, cudaMemcpyDeviceToHost));
        same = host[0] == static_cast<std::uint8_t>(0x50 + page);
    }
    check(same, "A restores its global pages and ring snapshot");
    for (const int page : live) { store.release(page); }
}

void drops_without_keep() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    capture_sequence(store, buffers, 4000, 0x40);
    check(store.best_match(prompt_after(1000), kTokens) == std::nullopt,
          "without keep, the oldest host checkpoint A is dropped");
}

void touch_keeps_used_checkpoint() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    store.touch(prompt_after(1000));
    capture_sequence(store, buffers, 4000, 0x40);
    check(store.best_match(prompt_after(1000), kTokens).has_value() &&
              !store.best_match(prompt_after(2000), kTokens).has_value(),
          "a touched checkpoint outlives an untouched older one");
}

} // namespace

int main() {
    try {
        restores_kept_checkpoint();
        drops_without_keep();
        touch_keeps_used_checkpoint();
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 prefix store\n";
    return failures ? 1 : 0;
}
