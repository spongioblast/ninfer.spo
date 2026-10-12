// An attention-only model (no recurrent state) keeps checkpoints made only of KV pages, in two
// pools with different page geometry. This drives the shared program storage through what a
// Program does: bind, prefill, capture a prefix view, quote and release it, fork a lane from a
// checkpoint, pause and resume without a snapshot, and replay past the checkpoint. No state store
// is involved, so the shared layer must not assume one.

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "program/storage/kv_address_space.h"
#include "program/storage/logical_kv_store.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace shared = ninfer::program;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr std::uint32_t kPageCapacity = 4;
constexpr std::uint32_t kPageGroups   = 12;
constexpr std::uint32_t kFrontier     = 100; // one full page and a 36-column tail

struct Pool {
    std::unique_ptr<ninfer::DeviceArena> arena;
    std::unique_ptr<ninfer::DeviceKVPagePool> physical;
    std::unique_ptr<ninfer::KVExecutionTablePool> tables;
    std::unique_ptr<shared::LogicalKVPageStore> pages;
    std::unique_ptr<shared::KVAddressSpaceStore> addresses;
};

Pool make_pool(ninfer::KVPageGeometry geometry) {
    ninfer::LayoutBuilder builder;
    const auto page_layout = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = kPageGroups, .geometry = std::move(geometry)});
    const auto table_layout = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = kPageCapacity, .table_rows = 2});
    Pool pool;
    pool.arena = std::make_unique<ninfer::DeviceArena>(builder.finish(256));
    const ninfer::DeviceSpan backing{pool.arena->base(), pool.arena->capacity()};
    pool.physical  = std::make_unique<ninfer::DeviceKVPagePool>(backing, page_layout);
    pool.tables    = std::make_unique<ninfer::KVExecutionTablePool>(backing, table_layout,
                                                                    *pool.physical);
    pool.pages     = std::make_unique<shared::LogicalKVPageStore>(
        *pool.physical, pool.physical->capacity_pages() + 8U);
    pool.addresses = std::make_unique<shared::KVAddressSpaceStore>(*pool.pages, *pool.tables, 4,
                                                                   kPageCapacity);
    return pool;
}

// Pages that releasing this address frees: the ones no other address holds.
std::uint32_t release_quote(const Pool& pool, shared::KVAddressSpaceHandle address) {
    std::uint32_t pages = 0;
    for (std::uint32_t page = 0; page < pool.addresses->mapped_pages(address); ++page) {
        if (pool.pages->address_references(pool.addresses->logical_page(address, page)) == 1) {
            ++pages;
        }
    }
    return pages;
}

void exercise(Pool& pool, ninfer::DeviceContext& device, std::string_view name) {
    const std::string label(name);
    auto& addresses = *pool.addresses;
    auto& physical  = *pool.physical;

    // Bind and prefill lane A.
    const auto lane_a = addresses.create_active(3, 0, device.stream);
    expect(lane_a.has_value(), label + ": bind lane A");
    addresses.ensure_mapped_to_tokens(*lane_a, kFrontier, device.stream);
    addresses.commit_frontier(*lane_a, kFrontier);
    expect(addresses.mapped_pages(*lane_a) == 2, label + ": prefill maps two pages");

    // Capture a checkpoint at the frontier: full pages shared, the partial tail copied.
    addresses.set_checkpoint_requirement(*lane_a, kFrontier);
    const auto checkpoint = addresses.create_inactive();
    expect(checkpoint.has_value(), label + ": checkpoint address");
    auto view = addresses.prepare_active_prefix_view(*lane_a, *checkpoint, kFrontier);
    expect(view.needs_tail_copy(), label + ": unaligned capture copies its tail");
    (void)physical.copy_page(addresses.active_prefix_view_tail_source(view),
                             addresses.active_prefix_view_tail_destination(view), device.stream);
    device.synchronize();
    addresses.commit_active_prefix_view(std::move(view));
    expect(!addresses.active(*checkpoint) && addresses.committed_frontier(*checkpoint) == kFrontier,
           label + ": checkpoint is an inactive view at the frontier");
    expect(addresses.logical_page(*checkpoint, 0) == addresses.logical_page(*lane_a, 0) &&
               addresses.logical_page(*checkpoint, 1) != addresses.logical_page(*lane_a, 1),
           label + ": checkpoint shares the full page and owns its tail");

    // The lane keeps decoding on its own pages; the checkpoint protects the captured prefix.
    addresses.ensure_mapped_to_tokens(*lane_a, kFrontier + 40, device.stream);
    addresses.commit_frontier(*lane_a, kFrontier + 40);
    bool truncate_rejected = false;
    try {
        addresses.destructive_truncate(*lane_a, 50);
    } catch (const std::logic_error&) { truncate_rejected = true; }
    expect(truncate_rejected, label + ": lane cannot truncate below its checkpoint");
    expect(pool.pages->committed_columns(addresses.logical_page(*checkpoint, 1)) == kFrontier - 64,
           label + ": decoding past the frontier leaves the checkpoint tail unchanged");

    // Fork lane B from the checkpoint into execution row 1.
    const auto lane_b = addresses.create_inactive();
    expect(lane_b.has_value(), label + ": lane B address");
    auto fork = addresses.prepare_prefix_fork(*checkpoint, *lane_b, kFrontier, 2, 1);
    expect(fork.needs_tail_copy(), label + ": unaligned fork copies its tail");
    (void)physical.copy_page(addresses.prefix_fork_tail_source(fork),
                             addresses.prefix_fork_tail_destination(fork), device.stream);
    addresses.commit_prefix_fork(std::move(fork), device.stream);
    device.synchronize();
    expect(addresses.active(*lane_b) && addresses.bound_row(*lane_b) == 1 &&
               addresses.committed_frontier(*lane_b) == kFrontier,
           label + ": forked lane is active at the checkpoint frontier");
    expect(pool.pages->address_references(addresses.logical_page(*lane_a, 0)) == 3,
           label + ": the full page is shared by lane A, the checkpoint, and lane B");

    // Pause lane B without a snapshot, then resume it.
    addresses.deactivate(*lane_b);
    expect(addresses.bound_row(*lane_b) == -1, label + ": paused lane has no execution row");
    auto activation = addresses.prepare_activation(*lane_b, 2, 1);
    addresses.commit_activation(std::move(activation), device.stream);
    expect(addresses.active(*lane_b) && addresses.mapped_pages(*lane_b) == 2 &&
               addresses.committed_frontier(*lane_b) == kFrontier,
           label + ": resumed lane keeps its pages and frontier");

    // Replay past the checkpoint on lane B.
    addresses.ensure_mapped_to_tokens(*lane_b, kFrontier + 64, device.stream);
    addresses.commit_frontier(*lane_b, kFrontier + 64);
    expect(addresses.mapped_pages(*lane_b) == 3 &&
               pool.pages->committed_columns(addresses.logical_page(*checkpoint, 1)) ==
                   kFrontier - 64,
           label + ": replay on the fork leaves the checkpoint unchanged");

    // Reclaim the checkpoint: the quote is exactly what the release frees.
    const std::uint32_t quote  = release_quote(pool, *checkpoint);
    const std::uint32_t before = physical.allocated_pages();
    expect(quote == 1, label + ": checkpoint quote counts only its own tail page");
    expect(addresses.release(*checkpoint), label + ": checkpoint releases");
    expect(physical.allocated_pages() == before - quote,
           label + ": release frees exactly the quoted pages");

    addresses.deactivate(*lane_a);
    addresses.deactivate(*lane_b);
    expect(addresses.release(*lane_a) && addresses.release(*lane_b), label + ": lanes release");
    device.synchronize();
    expect(pool.pages->occupied() == 0 && physical.allocated_pages() == 0 &&
               physical.reserved_pages() == 0,
           label + ": the pool is empty after teardown");
}

} // namespace

int main() {
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        ninfer::DeviceContext device(0);
        const auto page_tokens = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
        ninfer::KVPageGeometry main_geometry;
        main_geometry.page_tokens        = page_tokens;
        main_geometry.device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor;
        main_geometry.planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}};
        ninfer::KVPageGeometry second_geometry;
        second_geometry.page_tokens        = page_tokens;
        second_geometry.device_plane_order = ninfer::PagedKVPlaneOrder::HeadMajor;
        second_geometry.planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 4, .head_extent = 1},
                                  {.dtype = ninfer::DType::BF16, .leading_extent = 2, .head_extent = 4}};
        Pool main_pool   = make_pool(std::move(main_geometry));
        Pool second_pool = make_pool(std::move(second_geometry));
        exercise(main_pool, device, "main pool");
        exercise(second_pool, device, "second pool");
        device.synchronize();
    } catch (const std::exception& failure) {
        std::cerr << "FAIL: unexpected exception: " << failure.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
