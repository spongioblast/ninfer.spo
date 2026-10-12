#include "core/nvml_memory.h"

#include <cuda_runtime.h>

#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer {
namespace {

using NvmlReturn = int;
using NvmlDevice = void*;

struct NvmlMemoryV1 {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct NvmlApi {
    NvmlReturn (*init)()                                  = nullptr;
    NvmlReturn (*by_pci_bus_id)(const char*, NvmlDevice*) = nullptr;
    NvmlReturn (*memory_info)(NvmlDevice, NvmlMemoryV1*)  = nullptr;
    bool ready                                            = false;
};

template <class Function> bool bind(void* library, const char* name, Function& out) {
#ifdef _WIN32
    out = reinterpret_cast<Function>(
        reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name)));
#else
    out = reinterpret_cast<Function>(dlsym(library, name));
#endif
    return out != nullptr;
}

const NvmlApi& nvml() {
    static NvmlApi api;
    static std::once_flag once;
    std::call_once(once, [] {
#ifdef _WIN32
        void* library = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
        void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        if (library == nullptr) { return; }
        if (!bind(library, "nvmlInit_v2", api.init) ||
            !bind(library, "nvmlDeviceGetHandleByPciBusId_v2", api.by_pci_bus_id) ||
            !bind(library, "nvmlDeviceGetMemoryInfo", api.memory_info)) {
            return;
        }
        api.ready = api.init() == 0;
    });
    return api;
}

} // namespace

std::optional<NvmlDeviceMemory> nvml_device_memory(int cuda_device) {
    const NvmlApi& api = nvml();
    if (!api.ready) { return std::nullopt; }
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), cuda_device) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    NvmlDevice device = nullptr;
    if (api.by_pci_bus_id(bus_id, &device) != 0) {
        // CUDA prints a 4-digit domain, NVML's canonical form has 8.
        const std::string wide = std::string("0000") + bus_id;
        if (api.by_pci_bus_id(wide.c_str(), &device) != 0) { return std::nullopt; }
    }
    NvmlMemoryV1 memory{};
    if (api.memory_info(device, &memory) != 0) { return std::nullopt; }
    return NvmlDeviceMemory{memory.total, memory.used, memory.free};
}

} // namespace ninfer
