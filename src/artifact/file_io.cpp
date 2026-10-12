#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    const auto error = ::GetLastError();
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(static_cast<unsigned long>(error)));
}

HANDLE open_read(const std::filesystem::path& path, DWORD extra_flags) {
    constexpr DWORD kShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ, kShareMode, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL | extra_flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { fail(path, "CreateFileW"); }
    return handle;
}

std::uint64_t regular_file_bytes(HANDLE handle, const std::filesystem::path& path) {
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle, &size)) { fail(path, "GetFileSizeEx"); }
    if (size.QuadPart < 0) { throw ArtifactError(path.string() + ": expected a regular file"); }
    BY_HANDLE_FILE_INFORMATION info{};
    if (::GetFileInformationByHandle(handle, &info) &&
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        throw ArtifactError(path.string() + ": expected a regular file");
    }
    return static_cast<std::uint64_t>(size.QuadPart);
}

// Both handles are opened with FILE_FLAG_OVERLAPPED: every read carries its own offset and
// event, so concurrent readers share no file pointer and the I/O manager does not serialize
// them. lpNumberOfBytesRead stays null; the count comes only from GetOverlappedResult.
bool read_at(HANDLE handle, std::uint64_t offset, std::byte* data, std::size_t count,
             DWORD& transferred) {
    transferred = 0;
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFu);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    overlapped.hEvent     = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (overlapped.hEvent == nullptr) { return false; }
    bool ok = ::ReadFile(handle, data, static_cast<DWORD>(count), nullptr, &overlapped) != 0;
    if (!ok && ::GetLastError() == ERROR_IO_PENDING) { ok = true; }
    if (ok) { ok = ::GetOverlappedResult(handle, &overlapped, &transferred, TRUE) != 0; }
    if (!ok && ::GetLastError() == ERROR_HANDLE_EOF) {
        transferred = 0;
        ok          = true;
    }
    const DWORD error = ::GetLastError();
    ::CloseHandle(overlapped.hEvent);
    ::SetLastError(error);
    return ok && transferred <= count;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    fd_    = open_read(path_, FILE_FLAG_OVERLAPPED);
    bytes_ = regular_file_bytes(static_cast<HANDLE>(fd_), path_);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
#ifdef _WIN32
    if (direct_fd_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_fd_)); }
    if (fd_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(fd_)); }
#else
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
#ifdef _WIN32
        DWORD transferred = 0;
        if (!read_at(static_cast<HANDLE>(fd_), offset, destination.data(), count, transferred)) {
            fail(path_, "ReadFile");
        }
        if (!transferred) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        const auto read = static_cast<std::size_t>(transferred);
#else
        const auto read = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
#endif
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment
#ifndef _WIN32
        || destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())
#endif
    ) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#ifdef _WIN32
    std::call_once(direct_once_, [&] {
        direct_fd_ = open_read(path_, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
    });
    if (destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
        throw ArtifactError(path_.string() + ": direct read exceeds one Win32 ReadFile");
    }
    DWORD transferred = 0;
    if (!read_at(static_cast<HANDLE>(direct_fd_), offset, destination.data(), destination.size(),
                 transferred)) {
        fail(path_, "direct ReadFile");
    }
    return static_cast<std::size_t>(transferred);
#else
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

std::shared_ptr<const std::byte> InputFile::map(std::uint64_t offset, std::size_t bytes,
                                                MappingAccess access) const {
    if (!bytes || offset > bytes_ || bytes > bytes_ - offset) {
        throw std::invalid_argument("mapped range exceeds input file");
    }
#ifdef _WIN32
    (void)access; // MappingAccess::Random has no Win32 equivalent.
    SYSTEM_INFO info{};
    ::GetSystemInfo(&info);
    const std::uint64_t granularity = info.dwAllocationGranularity;
    const std::uint64_t start       = offset / granularity * granularity;
    const std::uint64_t delta       = offset - start;
    const std::uint64_t length      = delta + bytes;
    HANDLE mapping = ::CreateFileMappingW(static_cast<HANDLE>(fd_), nullptr, PAGE_READONLY, 0, 0,
                                          nullptr);
    if (mapping == nullptr) { fail(path_, "CreateFileMappingW"); }
    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, static_cast<DWORD>(start >> 32),
                                 static_cast<DWORD>(start & 0xFFFFFFFFu),
                                 static_cast<SIZE_T>(length));
    const DWORD error = ::GetLastError();
    ::CloseHandle(mapping);
    if (view == nullptr) {
        ::SetLastError(error);
        fail(path_, "MapViewOfFile");
    }
    return {static_cast<const std::byte*>(view) + delta,
            [view](const std::byte*) { ::UnmapViewOfFile(view); }};
#else
    const auto page   = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    const auto start  = offset / page * page;
    const auto delta  = offset - start;
    const auto length = bytes + delta;
    void* ptr         = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd_, start);
    if (ptr == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "map artifact range");
    }
    if (access == MappingAccess::Random && ::madvise(ptr, length, MADV_RANDOM) != 0) {
        const int error = errno;
        ::munmap(ptr, length);
        throw std::system_error(error, std::generic_category(), "advise random artifact mapping");
    }
    return {static_cast<const std::byte*>(ptr) + delta,
            [ptr, length](const std::byte*) { ::munmap(ptr, length); }};
#endif
}

} // namespace ninfer::artifact
