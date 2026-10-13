#include "models/gemma4/program/top_tokens.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::uint16_t to_bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float from_bf16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

} // namespace

int main() {
    constexpr std::int32_t kVocab   = 5000;
    constexpr std::int32_t kColumns = 37;
    constexpr std::uint32_t kTop    = 64;
    std::mt19937 generator(11);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kVocab) * kColumns);
    for (auto& value : logits) { value = to_bf16(normal(generator)); }
    // Column 3 has a tie-free spike so its top-1 probability is near one.
    logits[3 * kVocab + 1234] = to_bf16(60.0f);

    ninfer::TopTokenScores top{.k = kTop};
    top.ids.assign(kTop, -1);
    top.logprobs.assign(kTop, 0.0f);
    ninfer::models::gemma4::append_top_tokens(logits, kVocab, kColumns, top);
    if (top.ids.size() != kTop * (kColumns + 1) || top.logprobs.size() != top.ids.size()) {
        std::cerr << "FAIL top tokens size " << top.ids.size() << '\n';
        return 1;
    }
    int failures = 0;
    for (std::int32_t c = 0; c < kColumns; ++c) {
        const std::uint16_t* column = logits.data() + static_cast<std::size_t>(c) * kVocab;
        double peak = -INFINITY;
        for (std::int32_t i = 0; i < kVocab; ++i) { peak = std::max<double>(peak, from_bf16(column[i])); }
        double sum = 0.0;
        for (std::int32_t i = 0; i < kVocab; ++i) { sum += std::exp(from_bf16(column[i]) - peak); }
        const double log_total = peak + std::log(sum);
        std::vector<std::int32_t> order(kVocab);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return from_bf16(column[a]) > from_bf16(column[b]);
        });
        const std::size_t base = kTop * static_cast<std::size_t>(c + 1);
        for (std::uint32_t j = 0; j < kTop; ++j) {
            const std::int32_t id = top.ids[base + j];
            const double expect   = from_bf16(column[order[j]]) - log_total;
            // Equal logits may come in either order; the value at each rank must match.
            if (id < 0 || id >= kVocab ||
                from_bf16(column[id]) != from_bf16(column[order[j]]) ||
                std::fabs(top.logprobs[base + j] - expect) > 1.0e-5 * (1.0 + std::fabs(expect))) {
                if (failures < 4) {
                    std::cerr << "column " << c << " rank " << j << " id " << id << " logprob "
                              << top.logprobs[base + j] << " expect " << expect << '\n';
                }
                ++failures;
            }
        }
    }
    if (top.ids[0] != -1 || top.ids[kTop * 4] != 1234 || top.logprobs[kTop * 4] < -1.0e-6f) {
        std::cerr << "FAIL prefix kept or spike " << top.ids[kTop * 4] << ' ' << top.logprobs[kTop * 4]
                  << '\n';
        ++failures;
    }
    try {
        ninfer::TopTokenScores bad{.k = 0};
        ninfer::models::gemma4::append_top_tokens(logits, kVocab, kColumns, bad);
        std::cerr << "FAIL k=0 accepted\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gemma4 top tokens\n";
    return failures == 0 ? 0 : 1;
}
