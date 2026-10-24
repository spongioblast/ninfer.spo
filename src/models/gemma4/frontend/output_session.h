#pragma once

#include "models/gemma4/frontend/tokenizer.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {

class PublishedOutput {
public:
    using iterator = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
    [[nodiscard]] iterator end() noexcept { return values_.begin() + static_cast<std::ptrdiff_t>(size_); }
    [[nodiscard]] const_iterator end() const noexcept {
        return values_.begin() + static_cast<std::ptrdiff_t>(size_);
    }

    void push_back(OutputDelta value) {
        values_.at(size_) = std::move(value);
        ++size_;
    }

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

// Turns committed model tokens into content and reasoning text. `<|channel>thought\n` ...
// `<channel|>` is reasoning, `<|tool_call>` ... `<tool_call|>` is one structured tool call, and
// other special tokens are hidden. Text is published only at complete UTF-8 boundaries. Stop
// strings match the decoded channel text; a suffix that may still begin one is held back. A
// forced tool choice starts inside the call the prompt opened.
class OutputSession {
public:
    OutputSession() noexcept = default;
    OutputSession(OutputSession&&) noexcept = default;
    OutputSession& operator=(OutputSession&&) noexcept = default;
    OutputSession(const OutputSession&) = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
        return total_budget_remaining;
    }
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept { return {}; }
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId>,
                                                          std::uint32_t);
    void validate_generation_capacity(std::uint32_t) const noexcept {}
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    [[nodiscard]] bool constrained() const noexcept { return false; }
    void observe_constraint(bool, double) noexcept {}
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
        return 0;
    }
    void constraint_uploaded(std::size_t) noexcept {}
    void discard_preview() noexcept {
        staged_tokens_.clear();
        staged_terminal_ = false;
    }
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept {
        return std::move(state_.tool_calls);
    }
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept {
        return state_.diagnostics;
    }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept {
        return state_.reasoning_tokens;
    }
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept { return thinking_; }
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    enum class Mode : std::uint8_t { Content, ReasoningHeader, Reasoning, ToolCall };

    // Everything committed tokens change. content and reasoning hold bytes not yet published.
    struct State {
        Mode mode = Mode::Content;
        std::string content;
        std::string reasoning;
        std::string header;
        std::string tool_text;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
        std::uint32_t reasoning_tokens = 0;
        std::optional<std::size_t> matched_stop;
    };

    struct Setup {
        std::vector<TokenId> stop_ids;
        std::vector<StopString> stop_strings;
        bool publish_stop = false;
        ThinkingBudgetStats thinking;
        bool starts_in_reasoning = false;
        // "call:" or "call:<name>" when the prompt ends inside a forced tool call.
        std::optional<std::string> forced_call;
        std::optional<std::vector<std::string>> allowed_names;
    };

    OutputSession(std::shared_ptr<const Tokenizer> tokenizer, Setup setup);

    [[nodiscard]] bool is_stop(TokenId token) const noexcept;
    void consume(State& state, TokenId token) const;
    static void append_text(State& state, const std::string& bytes);
    static void close_reasoning(State& state);
    void close_tool_call(State& state) const;
    static void flush_terminal(State& state);
    [[nodiscard]] bool match_stop_string(State& state) const;
    [[nodiscard]] std::size_t stop_hold(const std::string& text, OutputChannel channel) const;

    std::shared_ptr<const Tokenizer> tokenizer_;
    std::vector<TokenId> stop_ids_;
    std::vector<StopString> stop_strings_;
    std::optional<std::vector<std::string>> allowed_names_;
    bool publish_stop_ = false;
    ThinkingBudgetStats thinking_;
    State state_;
    State staged_;
    std::vector<TokenId> staged_tokens_;
    bool staged_terminal_ = false;

    friend class Frontend;
};

} // namespace ninfer::models::gemma4
