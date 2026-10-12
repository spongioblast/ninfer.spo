#include "models/qwen3_8_flash_next/execution/prefill_graph.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& label) {
    if (!condition) { throw std::runtime_error(label); }
}

} // namespace

int main() {
    using ninfer::models::qwen3_8_flash_next::detail::prefill_graph_envelope;
    using ninfer::models::qwen3_8_flash_next::detail::PrefillGraphKey;
    expect(prefill_graph_envelope(0, 262144) == 0, "empty");
    expect(prefill_graph_envelope(1, 262144) == 128, "first bucket");
    expect(prefill_graph_envelope(128, 262144) == 128, "bucket edge");
    expect(prefill_graph_envelope(129, 262144) == 512, "next bucket");
    // Same visible-key count as decode at the QSA profile edges (dense up to 2051 keys,
    // hierarchical selection from 8192 groups, persistent order layout up to 32768 groups).
    expect(prefill_graph_envelope(2048, 262144) == 2048, "decode bucket 2048");
    expect(prefill_graph_envelope(2049, 262144) == 8192, "decode bucket 8192");
    expect(prefill_graph_envelope(32768, 262144) == 32768, "decode bucket 32768");
    expect(prefill_graph_envelope(131072, 262144) == 131072, "decode bucket 131072");
    expect(prefill_graph_envelope(200000, 262144) == 262144, "high bucket");
    expect(prefill_graph_envelope(99000, 100000) == 100000, "capacity is the last decode bucket");
    expect(prefill_graph_envelope(100, 50) == 100, "capacity cannot hide tokens");
    expect(prefill_graph_envelope(100, 200) == 128, "capacity above the bucket");
    expect(prefill_graph_envelope(300000, 0) == 300000, "past the last bucket");

    PrefillGraphKey lhs;
    PrefillGraphKey rhs;
    lhs.tokens = 8;
    rhs.tokens = 8;
    lhs.envelope = 511;
    rhs.envelope = 511;
    expect(lhs == rhs, "equal keys");
    rhs.destination_slot = 1;
    expect(!(lhs == rhs), "slot is part of the key");
    rhs = lhs;
    rhs.sampling = &rhs;
    expect(!(lhs == rhs), "sampling pointer is part of the key");
    rhs = lhs;
    rhs.text_kv_table = &rhs;
    expect(!(lhs == rhs), "text block-table row is part of the key");
    rhs = lhs;
    rhs.mtp_kv_table = &rhs;
    expect(!(lhs == rhs), "MTP block-table row is part of the key");
    rhs = lhs;
    rhs.mtp_generated_column = true;
    expect(!(lhs == rhs), "generated MTP column is part of the key");
    std::cout << "OK prefill graph envelope and key\n";
    return EXIT_SUCCESS;
}
