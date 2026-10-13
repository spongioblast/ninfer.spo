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
// other special tokens are hidden. Text is published only at complete UTF-8 boundaries.
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
        staged_token_.reset();
        staged_terminal_ = false;
    }
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept {
        return std::move(tool_calls_);
    }
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept {
        return diagnostics_;
    }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return reasoning_tokens_; }
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept { return thinking_; }
    [[nodiscard]] std::optional<std::string> matched_stop_string() const { return std::nullopt; }

private:
    enum class Mode : std::uint8_t { Content, ReasoningHeader, Reasoning, ToolCall };

    OutputSession(std::shared_ptr<const Tokenizer> tokenizer, std::vector<TokenId> stop_ids,
                  bool publish_stop, ThinkingBudgetStats thinking, bool starts_in_reasoning)
        : tokenizer_(std::move(tokenizer)), stop_ids_(std::move(stop_ids)),
          publish_stop_(publish_stop), thinking_(thinking),
          mode_(starts_in_reasoning ? Mode::Reasoning : Mode::Content) {}

    [[nodiscard]] bool is_stop(TokenId token) const noexcept;
    void consume(TokenId token);
    void append_text(const std::string& bytes);
    void close_reasoning();
    void close_tool_call();
    void flush_terminal();

    std::shared_ptr<const Tokenizer> tokenizer_;
    std::vector<TokenId> stop_ids_;
    bool publish_stop_ = false;
    ThinkingBudgetStats thinking_;
    Mode mode_ = Mode::Content;
    std::optional<TokenId> staged_token_;
    bool staged_terminal_ = false;
    std::string content_;
    std::string reasoning_;
    std::string header_;
    std::string tool_text_;
    std::vector<GeneratedToolCall> tool_calls_;
    ToolCallParseDiagnostics diagnostics_;
    std::uint32_t reasoning_tokens_ = 0;

    friend class Frontend;
};

} // namespace ninfer::models::gemma4
