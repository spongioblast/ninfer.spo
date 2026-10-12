#include "models/qwen3_8_flash_next/impl/runtime/rebuild_work.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::optional<std::uint32_t> rewrite_split_after(const std::vector<std::uint32_t>& frontiers,
                                                 std::uint32_t cursor, std::uint32_t limit) {
    return ninfer::models::qwen3_8_flash_next::runtime_support::rewrite_split_after(
        frontiers, cursor, limit);
}

void expect(std::optional<std::uint32_t> actual, std::optional<std::uint32_t> wanted,
            const std::string& label) {
    if (actual != wanted) {
        const auto text = [](std::optional<std::uint32_t> value) {
            return value ? std::to_string(*value) : std::string("none");
        };
        throw std::runtime_error(label + ": got " + text(actual) + ", want " + text(wanted));
    }
}

// Pass ends a prompt [cursor, limit) is split into, applying the rule pass by pass.
std::vector<std::uint32_t> passes(const std::vector<std::uint32_t>& frontiers,
                                  std::uint32_t cursor, std::uint32_t limit) {
    std::vector<std::uint32_t> ends;
    while (cursor < limit) {
        cursor = rewrite_split_after(frontiers, cursor, limit).value_or(limit);
        ends.push_back(cursor);
    }
    return ends;
}

} // namespace

int main() {
    try {
        // Thinking off: the generation prompt is the opener (3 tokens) and the closed think block
        // (4 tokens). Live, it ends the prompt; replayed, the reply follows.
        const std::vector<std::uint32_t> live_off{103, 107};
        const std::vector<std::uint32_t> replay_off{103, 107, 150};
        expect(rewrite_split_after(live_off, 100, 107), std::nullopt, "live, thinking off");
        expect(rewrite_split_after(replay_off, 100, 300), 107, "replay, thinking off");
        if (passes(replay_off, 100, 300) != std::vector<std::uint32_t>{107, 150, 300}) {
            throw std::runtime_error("replay, thinking off: wrong pass ends");
        }

        // Thinking on: the opener (3) and "<think>\n" (2).
        expect(rewrite_split_after({103, 105}, 100, 105), std::nullopt, "live, thinking on");
        expect(rewrite_split_after({103, 105, 400}, 100, 500), 105, "replay, thinking on");

        // A 1-token piece takes different kernels: split before and after it.
        expect(rewrite_split_after({101, 104}, 100, 110), 101, "1-token first piece");
        expect(rewrite_split_after({103, 107}, 100, 108), 107, "1-token last piece");

        // A group never exceeds 8 tokens.
        expect(rewrite_split_after({103, 107, 110}, 100, 200), 107, "group over 8 tokens");
        expect(rewrite_split_after({104}, 100, 109), 104, "two pieces of 4 and 5");

        // A long piece splits as before; the short pieces after it still group.
        expect(rewrite_split_after({500, 503}, 100, 507), 500, "long first piece");
        expect(rewrite_split_after({500, 503}, 500, 507), std::nullopt, "short tail after long");

        // The capture frontier is the limit: the group ends there.
        expect(rewrite_split_after({103, 107}, 100, 105), std::nullopt, "capture limit");
        expect(rewrite_split_after({120}, 100, 110), std::nullopt, "frontier beyond limit");
        expect(rewrite_split_after({}, 100, 110), std::nullopt, "no frontiers");
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << "\n";
        return 1;
    }
    std::cout << "rewrite grouping ok\n";
    return 0;
}
