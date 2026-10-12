#include "artifact/reader.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::artifact {

struct Reader::Impl {
    std::filesystem::path entry;
    Directory directory;
    ArtifactId id{};
    std::uint64_t entry_payload_start = 0;
    std::uint64_t file_bytes          = 0;
    mutable std::vector<std::unique_ptr<InputFile>> files;
    // Parallel readers (read_range workers, the weight upload pipeline) open parts lazily.
    mutable std::mutex files_mutex;
    mutable std::vector<std::optional<WeightGeometry>> geometries;

    explicit Impl(const std::filesystem::path& path) : entry(path) {
        auto file = std::make_unique<InputFile>(entry);
        std::array<std::byte, kHeaderBytes> header{};
        file->read_exact(0, header);
        if (!std::equal(kEntryMagic.begin(), kEntryMagic.end(), header.begin())) {
            if (std::equal(kEntryMagic.begin(), kEntryMagic.end() - 1, header.begin()) &&
                header[kEntryMagic.size() - 1] == std::byte{2}) {
                throw ArtifactError(
                    entry.string() +
                    ": NInfer v2 artifact is not supported. Upgrade to v3 with: "
                    "python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer");
            }
            throw ArtifactError(entry.string() + ": expected NInfer v3 entry magic");
        }
        const auto json_bytes = read_u64_le(header.data() + 8);
        if (!json_bytes || json_bytes > file->bytes() - kHeaderBytes ||
            json_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError(entry.string() + ": invalid directory length");
        }
        std::copy_n(header.begin() + 16, id.size(), id.begin());
        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        file->read_exact(kHeaderBytes, std::as_writable_bytes(std::span(text.data(), text.size())));
        directory =
            parse_directory(parse_json(text, "artifact directory"), entry.filename().string());
        entry_payload_start = align_up(checked_add(kHeaderBytes, json_bytes, "metadata end"),
                                       kPayloadAlignment, "entry payload start");
        const auto expected =
            checked_add(entry_payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != file->bytes()) {
            throw ArtifactError(entry.string() + ": entry length differs from directory");
        }
        file_bytes = checked_add(entry_payload_start, directory.payload_bytes, "file bytes");
        file_bytes = checked_add(
            file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");
        files.resize(directory.files.size());
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
    }

    InputFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        std::lock_guard lock(files_mutex);
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<InputFile>(entry.parent_path() / *record.path);
            std::array<std::byte, kHeaderBytes> header{};
            input->read_exact(0, header);
            if (!std::equal(kPartMagic.begin(), kPartMagic.end(), header.begin()) ||
                read_u64_le(header.data() + 8) != index ||
                !std::equal(id.begin(), id.end(), header.begin() + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->bytes() !=
                checked_add(kPayloadAlignment, record.payload_bytes, "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file_bytes; }

ObjectHandle Reader::find(std::string_view id) const {
    const auto found = directory().object_index.find(id);
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& Reader::geometry(ObjectHandle handle) const {
    const auto& tensor = directory().tensor(handle);
    auto& result       = impl_->geometries[handle.index];
    if (!result) { result = describe_tensor(tensor); }
    return *result;
}

void Reader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<TensorObject>(object)) {
        (void)geometry(handle);
    } else if (std::get<ResourceObject>(object).encoding != kRawBytesEncoding) {
        throw ArtifactError(object_id(object) + ": unsupported resource encoding");
    }
}

std::vector<ReadSegment> Reader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto& data = directory();
    if (offset > data.payload_bytes || bytes > data.payload_bytes - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    const auto found     = std::upper_bound(data.files.begin(), data.files.end(), offset,
                                            [](std::uint64_t position, const FileRecord& file) {
                                            return position < file.logical_begin;
                                        });
    std::size_t index    = static_cast<std::size_t>(found - data.files.begin() - 1);
    std::uint64_t copied = 0;
    while (copied < bytes) {
        const auto& file = data.files[index];
        const auto local = offset + copied - file.logical_begin;
        const auto count = std::min(bytes - copied, file.payload_bytes - local);
        const auto start = index == 0 ? impl_->entry_payload_start : kPayloadAlignment;
        result.push_back({index, checked_add(start, local, "file offset"), copied, count});
        copied += count;
        ++index;
    }
    return result;
}

void Reader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    for (const auto& segment : segments(offset, destination.size())) {
        impl_->file(segment.file_index)
            .read_exact(segment.file_offset,
                        destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                            static_cast<std::size_t>(segment.bytes)));
    }
}

std::vector<std::byte> Reader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    (void)segments(offset, bytes);
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> Reader::read_object(ObjectHandle handle) const {
    validate_object(handle);
    const auto& object = directory().object(handle);
    return read_range(object_offset(object), object_bytes(object));
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

MappedRange Reader::map_range(std::uint64_t offset, std::uint64_t bytes,
                             MappingAccess access) const {
    MappedRange result;
    for (const auto& segment : segments(offset, bytes)) {
        result.segments.push_back(
            {impl_->file(segment.file_index).map(segment.file_offset, segment.bytes, access),
             segment.destination_offset, segment.bytes});
    }
    return result;
}

void MappedRange::copy(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > size() || destination.size() > size() - offset) {
        throw ArtifactError("mapped read exceeds range");
    }
    if (destination.empty()) { return; }
    auto it = std::upper_bound(
        segments.begin(), segments.end(), offset,
        [](std::uint64_t value, const MappedSegment& segment) { return value < segment.begin; });
    --it;
    while (!destination.empty()) {
        const auto local = offset - it->begin;
        const auto count = std::min<std::uint64_t>(destination.size(), it->bytes - local);
        std::copy_n(it->storage.get() + local, count, destination.data());
        offset += count;
        destination = destination.subspan(count);
        ++it;
    }
}

namespace {

struct PageSpan {
    void* address = nullptr;
    std::size_t bytes = 0;
};

void collect_pages(const MappedRange& range, std::uint64_t offset, std::uint64_t bytes,
                   std::uintptr_t page, std::vector<PageSpan>& out) {
    if (bytes == 0 || offset > range.size() || bytes > range.size() - offset) { return; }
    auto it = std::upper_bound(
        range.segments.begin(), range.segments.end(), offset,
        [](std::uint64_t value, const MappedSegment& segment) { return value < segment.begin; });
    --it;
    while (bytes != 0) {
        const auto local = offset - it->begin;
        const auto count = std::min<std::uint64_t>(bytes, it->bytes - local);
        const auto first = reinterpret_cast<std::uintptr_t>(it->storage.get() + local);
        const auto begin = first / page * page;
        out.push_back({reinterpret_cast<void*>(begin),
                       static_cast<std::size_t>(first + count - begin)});
        offset += count;
        bytes -= count;
        ++it;
    }
}

} // namespace

void MappedRange::will_need(std::uint64_t offset, std::uint64_t bytes) const {
    will_need_many(std::array{ByteRange{offset, bytes}});
}

void MappedRange::will_need_many(std::span<const ByteRange> ranges) const {
    if (resident || ranges.empty()) { return; }
#ifdef _WIN32
    SYSTEM_INFO info{};
    ::GetSystemInfo(&info);
    const auto page = static_cast<std::uintptr_t>(info.dwPageSize);
#else
    const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
#endif
    std::vector<PageSpan> pages;
    pages.reserve(ranges.size());
    for (const auto& range : ranges) { collect_pages(*this, range.offset, range.bytes, page, pages); }
    if (pages.empty()) { return; }
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> entries(pages.size());
    for (std::size_t i = 0; i < pages.size(); ++i) {
        entries[i].VirtualAddress = pages[i].address;
        entries[i].NumberOfBytes  = pages[i].bytes;
    }
    // Hint only, like madvise(MADV_WILLNEED). No completion signal.
    (void)::PrefetchVirtualMemory(::GetCurrentProcess(), static_cast<ULONG_PTR>(entries.size()),
                                  entries.data(), 0);
#else
    for (const auto& page_span : pages) {
        (void)::madvise(page_span.address, page_span.bytes, MADV_WILLNEED);
    }
#endif
}

void Reader::read_range(std::uint64_t offset, std::uint64_t bytes,
                        std::span<std::byte> destination, unsigned threads,
                        bool unbuffered) const {
    if (destination.size() != bytes) {
        throw ArtifactError("parallel read destination size does not match the requested range");
    }
    if (bytes == 0) { return; }
    const auto parts = segments(offset, bytes);
    if (threads == 0) { threads = 1; }
    struct Job {
        std::size_t file_index = 0;
        std::uint64_t file_offset = 0;
        std::uint64_t destination_offset = 0;
        std::uint64_t bytes = 0;
    };
    std::vector<Job> jobs;
    for (const auto& part : parts) {
        std::uint64_t done = 0;
        while (done < part.bytes) {
            const auto count = std::min<std::uint64_t>(part.bytes - done, 64ULL << 20);
            jobs.push_back({part.file_index, part.file_offset + done,
                            part.destination_offset + done, count});
            done += count;
        }
    }
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    const unsigned workers = std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::size_t>(jobs.size(), 1)));
    auto run = [&] {
        try {
            std::vector<std::byte> bounce;
            while (!failed.load(std::memory_order_relaxed)) {
                const auto index = next.fetch_add(1);
                if (index >= jobs.size()) { return; }
                const auto& job = jobs[index];
                auto* dest = destination.data() + job.destination_offset;
                if (!unbuffered) {
                    impl_->file(job.file_index)
                        .read_exact(job.file_offset, std::span(dest, static_cast<std::size_t>(job.bytes)));
                    continue;
                }
                constexpr std::uint64_t kAlign = 4096;
                const auto aligned = job.file_offset / kAlign * kAlign;
                const auto lead    = job.file_offset - aligned;
                const auto tail    = (lead + job.bytes + kAlign - 1) / kAlign * kAlign;
                bounce.resize(static_cast<std::size_t>(tail) + kAlign);
                auto* aligned_ptr = reinterpret_cast<std::byte*>(
                    (reinterpret_cast<std::uintptr_t>(bounce.data()) + kAlign - 1) / kAlign * kAlign);
                const auto got = impl_->file(job.file_index)
                                     .read_direct(aligned, std::span(aligned_ptr, static_cast<std::size_t>(tail)));
                if (got < lead + job.bytes) {
                    throw ArtifactError("short unbuffered artifact read");
                }
                std::copy_n(aligned_ptr + lead, static_cast<std::size_t>(job.bytes), dest);
            }
        } catch (...) {
            std::lock_guard lock(error_mutex);
            if (error == nullptr) { error = std::current_exception(); }
            failed.store(true, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers > 0 ? workers - 1 : 0);
    for (unsigned i = 1; i < workers; ++i) { pool.emplace_back(run); }
    run();
    for (auto& thread : pool) { thread.join(); }
    if (error != nullptr) { std::rethrow_exception(error); }
}
} // namespace ninfer::artifact
