#pragma once

#include "artifact/file_io.h"
#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {

struct MappedSegment {
    std::shared_ptr<const std::byte> storage;
    std::uint64_t begin = 0;
    std::uint64_t bytes = 0;
};

// Owns mappings independently of Reader; no payload is copied or pinned in host RAM.
class MappedRange {
public:
    std::vector<MappedSegment> segments;
    // True when the bytes are an ordinary allocation, not a file mapping. will_need is then a no-op.
    bool resident = false;
    struct ByteRange {
        std::uint64_t offset = 0;
        std::uint64_t bytes  = 0;
    };
    [[nodiscard]] std::uint64_t size() const noexcept {
        return segments.empty() ? 0 : segments.back().begin + segments.back().bytes;
    }
    void copy(std::uint64_t offset, std::span<std::byte> destination) const;
    // Starts asynchronous reads of the pages covering [offset, offset + bytes) without waiting,
    // so scattered lookups that are issued together share one round of storage latency.
    void will_need(std::uint64_t offset, std::uint64_t bytes) const;
    void will_need_many(std::span<const ByteRange> ranges) const;
};

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

// Owns the cold directory and lazily opened files. Runtime views borrow materialized storage,
// never this Reader. Encodings are interpreted only for requested objects.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    [[nodiscard]] const Directory& directory() const noexcept;
    [[nodiscard]] const ArtifactId& artifact_id() const noexcept;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
    [[nodiscard]] ObjectHandle find(std::string_view id) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    void validate_object(ObjectHandle handle) const;

    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    // Fills destination (size must equal bytes). unbuffered uses read_direct and keeps the
    // file cache from holding a second copy; the caller buffer need not be aligned.
    void read_range(std::uint64_t offset, std::uint64_t bytes, std::span<std::byte> destination,
                    unsigned threads, bool unbuffered = false) const;
    [[nodiscard]] std::vector<std::byte> read_object(ObjectHandle handle) const;
    [[nodiscard]] std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                          std::span<std::byte> destination) const;
    [[nodiscard]] MappedRange map_range(std::uint64_t offset, std::uint64_t bytes,
                                        MappingAccess access) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
