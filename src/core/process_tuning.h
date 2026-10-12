#pragma once

#include <optional>
#include <string>

namespace ninfer {

enum class ThreadRole { Engine, Worker };

// Host NUMA node for a CUDA device: an explicit request, then NINFER_NUMA_NODE, then the node the
// OS reports for the GPU's PCI location (Windows: PnP DEVPKEY_Device_Numa_Node; Linux:
// cudaDevAttrHostNumaId). -1 means no pinning. `source` names the rule that decided.
int resolve_numa_node(std::optional<int> requested, int cuda_device, std::string* source = nullptr);

// Windows: opt out of power throttling, raise the priority class, record `numa_node` for
// tune_current_thread, and pin the calling (main) thread so startup host allocations land on that
// node. Call before any host-memory-heavy startup work. No-op on Linux.
void apply_process_tuning(int numa_node);

// The node recorded by apply_process_tuning, or -1.
[[nodiscard]] int process_numa_node() noexcept;

// Pin the current thread to the process NUMA node and raise the engine thread priority.
// NINFER_THREAD_TUNING=0 disables it.
void tune_current_thread(ThreadRole role);

} // namespace ninfer
