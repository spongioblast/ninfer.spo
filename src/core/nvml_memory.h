#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

struct NvmlDeviceMemory {
    std::uint64_t total_bytes = 0;
    std::uint64_t used_bytes  = 0;
    std::uint64_t free_bytes  = 0;
};

// Memory of the GPU behind CUDA device `cuda_device` as NVML reports it, matched by PCI bus id.
// NVML is loaded at run time (nvml.dll / libnvidia-ml.so.1); empty if it is unavailable. Unlike
// cudaMemGetInfo under WDDM, NVML counts allocations made by other processes and WSL guests.
[[nodiscard]] std::optional<NvmlDeviceMemory> nvml_device_memory(int cuda_device);

} // namespace ninfer
