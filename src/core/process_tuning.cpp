#include "core/process_tuning.h"
#include "core/runtime_options.h"

#include <cuda_runtime.h>

#include <atomic>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// clang-format off
#include <initguid.h>
#include <devguid.h>
#include <devpkey.h>
#include <setupapi.h>
// clang-format on
#endif

namespace ninfer {
namespace {

std::atomic<int> g_numa_node{-1};

#ifdef _WIN32
// cudaDevAttrHostNumaId is not stable under WDDM (it read 0 and 1 in two processes for the same
// GPU), so ask PnP for the node of the PCI function the CUDA device reports.
int pnp_numa_node(unsigned bus, unsigned device, unsigned function) {
    HDEVINFO set = SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr, DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) { return -1; }
    int node = -1;
    SP_DEVINFO_DATA info{};
    info.cbSize = sizeof(info);
    for (DWORD index = 0; SetupDiEnumDeviceInfo(set, index, &info); ++index) {
        const auto read_u32 = [&](const DEVPROPKEY& key, UINT32& out) {
            DEVPROPTYPE type = 0;
            DWORD size       = 0;
            return SetupDiGetDevicePropertyW(set, &info, &key, &type, reinterpret_cast<PBYTE>(&out),
                                             sizeof(out), &size, 0) != FALSE &&
                   type == DEVPROP_TYPE_UINT32;
        };
        UINT32 device_bus = 0;
        UINT32 address    = 0;
        if (!read_u32(DEVPKEY_Device_BusNumber, device_bus) ||
            !read_u32(DEVPKEY_Device_Address, address)) {
            continue;
        }
        if (device_bus != bus || (address >> 16U) != device || (address & 0xFFFFU) != function) {
            continue;
        }
        UINT32 numa = 0;
        if (read_u32(DEVPKEY_Device_Numa_Node, numa)) { node = static_cast<int>(numa); }
        break;
    }
    SetupDiDestroyDeviceInfoList(set);
    return node;
}
#endif

} // namespace

int resolve_numa_node(std::optional<int> requested, int cuda_device, std::string* source) {
    const auto decided = [&](int node, const char* rule) {
        if (source != nullptr) { *source = rule; }
        return node;
    };
    if (requested) { return decided(*requested, "option"); }
    if (const auto from_env = runtime_switches().numa_node) {
        return decided(*from_env, "NINFER_NUMA_NODE");
    }
#ifdef _WIN32
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, cuda_device) == cudaSuccess) {
        const int node = pnp_numa_node(static_cast<unsigned>(props.pciBusID),
                                       static_cast<unsigned>(props.pciDeviceID), 0);
        if (node >= 0) { return decided(node, "PnP device NUMA node"); }
    } else {
        (void)cudaGetLastError();
    }
#else
    int node = -1;
    if (cudaDeviceGetAttribute(&node, cudaDevAttrHostNumaId, cuda_device) == cudaSuccess &&
        node >= 0) {
        return decided(node, "cudaDevAttrHostNumaId");
    }
    (void)cudaGetLastError();
#endif
    return decided(-1, "unknown");
}

void apply_process_tuning(int numa_node) {
    g_numa_node.store(numa_node);
#ifdef _WIN32
    PROCESS_POWER_THROTTLING_STATE state{};
    state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
#ifdef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
    state.ControlMask |= PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
#endif
    state.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
    if (runtime_switches().high_priority) {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    tune_current_thread(ThreadRole::Worker);
#endif
}

int process_numa_node() noexcept { return g_numa_node.load(); }

void tune_current_thread(ThreadRole role) {
#ifdef _WIN32
    if (!runtime_switches().thread_tuning) { return; }
    const int node = g_numa_node.load();
    GROUP_AFFINITY affinity{};
    if (node >= 0 && GetNumaNodeProcessorMaskEx(static_cast<USHORT>(node), &affinity)) {
        SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
    }
    if (role == ThreadRole::Engine) { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); }
#else
    (void)role;
#endif
}

} // namespace ninfer
