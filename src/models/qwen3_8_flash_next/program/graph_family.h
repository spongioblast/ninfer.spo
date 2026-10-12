#pragma once

// MTP attention envelopes for the decode CUDA Graphs and the adaptive MTP draft statistics. The
// graph families themselves are shared (program/execution/graph_family.h). Included from p3.h.

#include "models/qwen3_8_flash_next/execution/schedule.h"
#include "program/execution/graph_family.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

using program::DecodeGraphFamily;
using program::DecodeGraphProfile;
using program::DecodeGraphTopology;
using program::instantiate_graph_family;
using program::validate_graph_profiles;

inline schedule::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                               std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpCausalAttentionEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

// Decayed evidence for adaptive MTP: for every draft position, how often it was accepted or was
// the first rejected draft. Expected round tokens follow the product of the per-position
// conditional acceptance rates, so early rejections do not hide a high acceptance run later.
struct MtpDraftStatistics {
    static constexpr float kDecay          = 0.92F;
    static constexpr float kPriorAccepted  = 1.5F;
    static constexpr float kPriorRejected  = 0.5F;
    std::array<float, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> accepted = filled(kPriorAccepted);
    std::array<float, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> rejected = filled(kPriorRejected);

    void record(std::uint32_t extent, std::uint32_t accepted_drafts) {
        for (std::uint32_t position = 0; position < extent && position <= accepted_drafts;
             ++position) {
            accepted[position] = kDecay * accepted[position] + (position < accepted_drafts ? 1.0F : 0.0F);
            rejected[position] = kDecay * rejected[position] + (position == accepted_drafts ? 1.0F : 0.0F);
        }
    }

    // Expected committed tokens of one round that verifies `drafts` drafts (bonus included).
    [[nodiscard]] double expected_tokens(std::uint32_t drafts) const {
        double expected = 1.0;
        double run      = 1.0;
        for (std::uint32_t position = 0; position < drafts; ++position) {
            run *= accepted[position] / (accepted[position] + rejected[position]);
            expected += run;
        }
        return expected;
    }

private:
    static constexpr std::array<float, qwen3_8_flash_next::kMtpDecodeMaximumDrafts>
    filled(float value) {
        std::array<float, qwen3_8_flash_next::kMtpDecodeMaximumDrafts> values{};
        values.fill(value);
        return values;
    }
};

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
