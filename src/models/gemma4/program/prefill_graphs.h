#pragma once

#include "core/decode_graph.h"

#include <array>
#include <cstdint>
#include <utility>

namespace ninfer::models::gemma4 {

// Text prefill chunks of these widths run as one captured CUDA graph each. Vision chunks and
// other widths stay eager.
class PrefillGraphs {
public:
    static constexpr std::array<int, 6> kWidths{32, 64, 128, 256, 512, 1024};

    // Captures `body` on the first use of `width`, then replays it. Returns false, and does
    // nothing, when `width` has no bucket.
    template <class Body>
    bool run(std::uint32_t width, cudaStream_t stream, Body&& body) {
        const int index = slot(width);
        if (index < 0) { return false; }
        Entry& entry = entries_[static_cast<std::size_t>(index)];
        if (!entry.executable.ready()) {
            entry.definition.capture(stream, std::forward<Body>(body));
            entry.executable.instantiate(entry.definition);
        }
        entry.executable.launch(stream);
        return true;
    }

private:
    struct Entry {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
    };

    static int slot(std::uint32_t width) {
        for (std::size_t index = 0; index < kWidths.size(); ++index) {
            if (kWidths[index] == static_cast<int>(width)) { return static_cast<int>(index); }
        }
        return -1;
    }

    std::array<Entry, kWidths.size()> entries_{};
};

} // namespace ninfer::models::gemma4
