#pragma once

// Turns generated token bytes into published reasoning and content text, holding back
// partial UTF-8, the think-close marker, and partial stop strings.

#include "models/qwen3_8_flash_next/frontend/chat_template.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_8_flash_next {
namespace frontend_internal {
class Tokenizer;
} // namespace frontend_internal

namespace output_decoder {
namespace fi = frontend_internal;

inline constexpr std::string_view kThinkClose      = "</think>";

template <std::size_t Size>
consteval std::array<std::size_t, Size> make_prefix_failure_table(std::string_view pattern) {
    std::array<std::size_t, Size> failure{};
    for (std::size_t index = 1; index < Size; ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) {
            matched = failure[matched - 1U];
        }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

struct PrefixExecutionTracker {
    static constexpr std::string_view kBoundary = fi::kCanonicalReasoningCloseSerialization;
    static constexpr auto kFailure = make_prefix_failure_table<kBoundary.size()>(kBoundary);

    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes) noexcept {
        if (!tracking) { return std::nullopt; }
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != kBoundary[matched]) { matched = kFailure[matched - 1U]; }
            if (byte == kBoundary[matched]) { ++matched; }
            if (matched != kBoundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning              = false;
    bool strip_content_leading     = false;
    bool exact_reasoning_framing   = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
};

struct SemanticThinkingState {
    std::optional<std::uint32_t> budget;
    std::string close_pending;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool in_reasoning                   = false;
    bool control_pending                = false;
    bool applied                        = false;
    bool exact_reasoning_framing        = false;
};

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller);
void feed_semantic_thinking(SemanticThinkingState& state, std::string_view bytes);
void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match);
void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens);
DecoderState terminal_state(DecoderState state);

} // namespace output_decoder
} // namespace ninfer::models::qwen3_8_flash_next
