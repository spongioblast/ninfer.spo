#include "artifact/resident_memory.h"

#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32
std::string win32_error(const char* what) {
    return std::string(what) + " failed, Win32 error " + std::to_string(GetLastError());
}

bool enable_lock_memory_privilege(std::string& note) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        note = win32_error("OpenProcessToken");
        return false;
    }
    TOKEN_PRIVILEGES privileges{};
    privileges.PrivilegeCount           = 1;
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &privileges.Privileges[0].Luid) &&
              AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr);
    // AdjustTokenPrivileges succeeds without assigning a right the account does not hold.
    if (ok && GetLastError() != ERROR_SUCCESS) {
        note = "large pages need the \"Lock pages in memory\" right";
        ok   = false;
    } else if (!ok) {
        note = win32_error("AdjustTokenPrivileges");
    }
    CloseHandle(token);
    return ok;
}
#endif

} // namespace

std::shared_ptr<std::byte> allocate_resident(std::uint64_t bytes, int numa_node, bool large_pages,
                                             ResidentReport& report) {
    report.bytes     = bytes;
    report.numa_node = numa_node;
#ifdef _WIN32
    const DWORD node   = numa_node >= 0 ? static_cast<DWORD>(numa_node) : NUMA_NO_PREFERRED_NODE;
    void* memory       = nullptr;
    if (large_pages && enable_lock_memory_privilege(report.note)) {
        const SIZE_T granule = GetLargePageMinimum();
        const SIZE_T size    = static_cast<SIZE_T>((bytes + granule - 1) / granule * granule);
        memory               = VirtualAllocExNuma(GetCurrentProcess(), nullptr, size,
                                                  MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                  PAGE_READWRITE, node);
        if (memory == nullptr) {
            report.note = win32_error("large-page VirtualAllocExNuma");
        } else {
            report.large_pages = true;
            report.locked      = true;
        }
    }
    if (memory == nullptr) {
        memory = VirtualAllocExNuma(GetCurrentProcess(), nullptr, static_cast<SIZE_T>(bytes),
                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node);
    }
    if (memory == nullptr) { throw std::runtime_error(win32_error("resident VirtualAllocExNuma")); }
    return std::shared_ptr<std::byte>(static_cast<std::byte*>(memory), [](std::byte* pointer) {
        if (pointer != nullptr) { VirtualFree(pointer, 0, MEM_RELEASE); }
    });
#else
    (void)large_pages;
    void* memory = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) { throw std::runtime_error("resident mmap failed"); }
    const auto size = static_cast<std::size_t>(bytes);
    return std::shared_ptr<std::byte>(static_cast<std::byte*>(memory), [size](std::byte* pointer) {
        if (pointer != nullptr) { munmap(pointer, size); }
    });
#endif
}

void lock_resident(std::byte* data, std::uint64_t bytes, ResidentReport& report) {
#ifdef _WIN32
    if (report.large_pages) { return; }
    constexpr SIZE_T kSlack = SIZE_T{1} << 30U;
    SIZE_T minimum          = 0;
    SIZE_T maximum          = 0;
    DWORD flags             = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &minimum, &maximum, &flags) ||
        !SetProcessWorkingSetSizeEx(GetCurrentProcess(),
                                    minimum + static_cast<SIZE_T>(bytes) + kSlack,
                                    maximum + static_cast<SIZE_T>(bytes) + kSlack, 0)) {
        report.note = win32_error("SetProcessWorkingSetSizeEx");
        return;
    }
    if (!VirtualLock(data, static_cast<SIZE_T>(bytes))) {
        report.note = win32_error("VirtualLock");
        return;
    }
    report.locked = true;
#else
    (void)data;
    (void)bytes;
    (void)report;
#endif
}

std::string describe(const ResidentReport& report) {
    char text[256];
    std::snprintf(text, sizeof(text),
                  "%.2f GiB, NUMA node %d, large pages %s, locked %s, filled in %.1f s",
                  static_cast<double>(report.bytes) / (1ULL << 30U), report.numa_node,
                  report.large_pages ? "yes" : "no", report.locked ? "yes" : "no",
                  report.fill_seconds);
    std::string out = text;
    if (!report.note.empty()) { out += " (" + report.note + ")"; }
    return out;
}

} // namespace ninfer::artifact
