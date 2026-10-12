#include "artifact/file_io.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer::artifact;
using ninfer::test::artifact_fixture::require;

unsigned char pattern_byte(std::uint64_t index) {
    return static_cast<unsigned char>((index * 2654435761u) >> 24);
}

std::filesystem::path write_pattern(std::uint64_t bytes) {
    const auto path = std::filesystem::temp_directory_path() / "ninfer-file-mapping.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    for (std::uint64_t i = 0; i < bytes; ++i) {
        out.put(static_cast<char>(pattern_byte(i)));
    }
    return path;
}

void check_map(const InputFile& file, std::uint64_t offset, std::size_t bytes) {
    if (offset >= file.bytes()) { return; }
    if (bytes > file.bytes() - offset) { bytes = static_cast<std::size_t>(file.bytes() - offset); }
    const auto mapped = file.map(offset, bytes, MappingAccess::Random);
    std::vector<std::byte> exact(bytes);
    file.read_exact(offset, exact);
    for (std::size_t i = 0; i < bytes; ++i) {
        if (mapped.get()[i] != exact[i] ||
            static_cast<unsigned char>(mapped.get()[i]) != pattern_byte(offset + i)) {
            throw std::runtime_error("mapped bytes differ from the file");
        }
    }
}

} // namespace

int main() {
    try {
        constexpr std::uint64_t kBytes = (3ull << 20) + 123;
        const auto path                = write_pattern(kBytes);
        std::shared_ptr<const std::byte> survived;
        {
            InputFile file(path);
            const std::uint64_t offsets[] = {0, 1, 4095, 4096, 65535, 65536, 65537, kBytes - 1};
            for (const std::uint64_t offset : offsets) {
                for (const std::size_t length : {std::size_t{1}, std::size_t{160},
                                                 std::size_t{70000}}) {
                    check_map(file, offset, length);
                }
            }
            survived = file.map(100, 32, MappingAccess::Normal);
            MappedRange range;
            range.segments.push_back({file.map(0, file.bytes(), MappingAccess::Normal), 0,
                                      file.bytes()});
            range.will_need(0, 0);
            range.will_need(0, 1);
            std::vector<MappedRange::ByteRange> many;
            std::mt19937 rng(1);
            std::uniform_int_distribution<std::uint64_t> dist(0, file.bytes() - 160);
            for (int i = 0; i < 2000; ++i) { many.push_back({dist(rng), 160}); }
            range.will_need_many(many);

            std::vector<std::byte> unaligned(16);
            require(true, "setup");
            bool threw = false;
            try {
                (void)file.read_direct(1, unaligned);
            } catch (const ArtifactError&) { threw = true; }
            require(threw, "unaligned direct read was accepted");
        }
        require(static_cast<unsigned char>(survived.get()[0]) == pattern_byte(100),
                "mapping died with the file");
        std::filesystem::remove(path);

        ninfer::test::artifact_fixture::Fixture fixture;
        fixture.write();
        Reader reader(fixture.entry);
        const auto bytes = reader.read_range(0, 5);
        std::vector<std::byte> parallel(bytes.size());
        for (const unsigned threads : {1u, 4u, 8u}) {
            reader.read_range(0, bytes.size(), parallel, threads, false);
            require(parallel == bytes, "parallel read did not match the serial read");
        }
        std::cout << "file mapping checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
