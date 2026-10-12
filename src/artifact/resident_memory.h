#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ninfer::artifact {

struct ResidentReport {
    std::uint64_t bytes = 0;
    int numa_node       = -1;
    bool large_pages    = false;
    bool locked         = false;
    double fill_seconds = 0.0;
    // Why large pages or locking were not used; empty when everything asked for was applied.
    std::string note;
};

// Page-aligned host memory for a table that is filled once at load and read for the life of the
// process. Windows: VirtualAllocExNuma on `numa_node` (-1: no preference), with large pages when
// asked and the "Lock pages in memory" right is held. Linux: anonymous mmap.
[[nodiscard]] std::shared_ptr<std::byte> allocate_resident(std::uint64_t bytes, int numa_node,
                                                           bool large_pages,
                                                           ResidentReport& report);

// Keeps a filled allocation in RAM (Windows: raise the working-set minimum, then VirtualLock).
// Large-page memory is already non-pageable. Failure is recorded in `report`, not thrown.
void lock_resident(std::byte* data, std::uint64_t bytes, ResidentReport& report);

[[nodiscard]] std::string describe(const ResidentReport& report);

} // namespace ninfer::artifact
