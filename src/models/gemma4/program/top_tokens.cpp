#include "models/gemma4/program/top_tokens.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

float bf16(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }

void column_top(const std::uint16_t* column, std::int32_t vocab, std::uint32_t k, TokenId* ids,
                float* logprobs) {
    float peak = -INFINITY;
    for (std::int32_t i = 0; i < vocab; ++i) { peak = std::max(peak, bf16(column[i])); }
    // Min-heap of the k largest (value, id) pairs seen so far.
    std::vector<std::pair<float, TokenId>> heap;
    heap.reserve(k);
    const auto greater = std::greater<std::pair<float, TokenId>>();
    double sum = 0.0;
    for (std::int32_t i = 0; i < vocab; ++i) {
        const float value = bf16(column[i]);
        sum += std::exp(static_cast<double>(value) - peak);
        if (heap.size() < k) {
            heap.emplace_back(value, i);
            std::push_heap(heap.begin(), heap.end(), greater);
        } else if (value > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), greater);
            heap.back() = {value, i};
            std::push_heap(heap.begin(), heap.end(), greater);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), greater);
    const double log_total = static_cast<double>(peak) + std::log(sum);
    for (std::uint32_t j = 0; j < k; ++j) {
        ids[j]      = heap[j].second;
        logprobs[j] = static_cast<float>(static_cast<double>(heap[j].first) - log_total);
    }
}

} // namespace

void append_top_tokens(std::span<const std::uint16_t> logits, std::int32_t vocab,
                       std::int32_t columns, TopTokenScores& out) {
    if (out.k == 0 || out.k > static_cast<std::uint32_t>(vocab) || columns < 0 ||
        logits.size() != static_cast<std::size_t>(vocab) * static_cast<std::size_t>(columns)) {
        throw std::invalid_argument("top tokens: k must be in [1,vocab] over [vocab,columns]");
    }
    const std::size_t first = out.ids.size();
    out.ids.resize(first + static_cast<std::size_t>(columns) * out.k);
    out.logprobs.resize(out.ids.size());
    const auto run = [&](std::int32_t begin, std::int32_t end) {
        for (std::int32_t c = begin; c < end; ++c) {
            const std::size_t at = first + static_cast<std::size_t>(c) * out.k;
            column_top(logits.data() + static_cast<std::size_t>(c) * vocab, vocab, out.k,
                       out.ids.data() + at, out.logprobs.data() + at);
        }
    };
    const auto workers = static_cast<std::int32_t>(
        std::clamp<unsigned>(std::thread::hardware_concurrency(), 1U, 32U));
    const std::int32_t per = (columns + workers - 1) / workers;
    std::vector<std::jthread> threads;
    for (std::int32_t begin = per; begin < columns; begin += per) {
        threads.emplace_back(run, begin, std::min(columns, begin + per));
    }
    run(0, std::min(columns, per));
}

} // namespace ninfer::models::gemma4
