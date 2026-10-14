#include "models/gemma4/frontend/output_session.h"

#include "models/gemma4/frontend/tool_parser.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace ninfer::models::gemma4 {
namespace {

constexpr TokenId kToolCallOpen  = 48;
constexpr TokenId kToolCallClose = 49;
constexpr TokenId kChannelOpen   = 100;
constexpr TokenId kChannelClose  = 101;
constexpr std::string_view kThoughtHeader = "thought\n";
constexpr std::string_view kReplacement   = "\xEF\xBF\xBD";

std::size_t sequence_width(unsigned char lead) {
    if (lead < 0x80U) { return 1; }
    if (lead >= 0xc2U && lead <= 0xdfU) { return 2; }
    if (lead >= 0xe0U && lead <= 0xefU) { return 3; }
    if (lead >= 0xf0U && lead <= 0xf4U) { return 4; }
    return 0;
}

bool continuation(unsigned char byte) { return (byte & 0xc0U) == 0x80U; }

// Moves the valid UTF-8 prefix of pending into out, replacing invalid bytes. A trailing
// incomplete sequence stays in pending unless `final` is set.
void drain_utf8(std::string& pending, std::string& out, bool final) {
    std::size_t index = 0;
    while (index < pending.size()) {
        const auto lead   = static_cast<unsigned char>(pending[index]);
        const auto width  = sequence_width(lead);
        if (width == 0) {
            out += kReplacement;
            ++index;
            continue;
        }
        if (index + width > pending.size()) {
            bool prefix_ok = true;
            for (std::size_t at = index + 1; at < pending.size(); ++at) {
                prefix_ok = prefix_ok && continuation(static_cast<unsigned char>(pending[at]));
            }
            if (prefix_ok && !final) { break; }
            out += kReplacement;
            ++index;
            continue;
        }
        bool valid = true;
        for (std::size_t at = index + 1; at < index + width; ++at) {
            valid = valid && continuation(static_cast<unsigned char>(pending[at]));
        }
        if (!valid) {
            out += kReplacement;
            ++index;
            continue;
        }
        out.append(pending, index, width);
        index += width;
    }
    pending.erase(0, index);
}

} // namespace

bool OutputSession::is_stop(TokenId token) const noexcept {
    return std::find(stop_ids_.begin(), stop_ids_.end(), token) != stop_ids_.end();
}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (tokens.empty() || total_budget_remaining == 0) {
        throw std::logic_error("gemma4 output preview expects tokens inside the budget");
    }
    runtime::OutputDecision decision;
    decision.continuation = runtime::ContinuationAction::Decode;
    staged_tokens_.clear();
    for (const TokenId token : tokens) {
        if (staged_tokens_.size() == total_budget_remaining) { break; }
        staged_tokens_.push_back(token);
        if (is_stop(token)) {
            decision.finish_reason = FinishReason::StopToken;
            break;
        }
    }
    if (!decision.finished() && staged_tokens_.size() == total_budget_remaining) {
        decision.finish_reason = limit_reason;
    }
    decision.accepted_tokens = static_cast<std::uint32_t>(staged_tokens_.size());
    staged_terminal_         = decision.finished();
    return decision;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId>, std::uint32_t) {
    throw std::logic_error("gemma4 has no target-control tokens");
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    runtime::OutputDecision decision;
    decision.finish_reason   = reason;
    decision.accepted_tokens = 0;
    staged_tokens_.clear();
    staged_terminal_ = true;
    return decision;
}

void OutputSession::append_text(const std::string& bytes) {
    switch (mode_) {
    case Mode::Content: content_ += bytes; return;
    case Mode::Reasoning: reasoning_ += bytes; return;
    case Mode::ToolCall: tool_text_ += bytes; return;
    case Mode::ReasoningHeader: break;
    }
    header_ += bytes;
    if (header_.size() < kThoughtHeader.size() &&
        kThoughtHeader.substr(0, header_.size()) == header_) {
        return;
    }
    mode_ = Mode::Reasoning;
    if (std::string_view(header_).starts_with(kThoughtHeader)) {
        reasoning_.append(header_, kThoughtHeader.size());
    } else {
        reasoning_ += header_;
    }
    header_.clear();
}

void OutputSession::close_reasoning() {
    if (mode_ == Mode::ReasoningHeader) {
        if (header_ != kThoughtHeader.substr(0, header_.size())) { reasoning_ += header_; }
        header_.clear();
    }
    mode_ = Mode::Content;
}

void OutputSession::close_tool_call() {
    auto call = parse_tool_call_body(tool_text_);
    if (call) {
        tool_calls_.push_back(std::move(*call));
        ++diagnostics_.structured_call_count;
    } else {
        content_ += "<|tool_call>";
        content_ += tool_text_;
        content_ += "<tool_call|>";
        if (diagnostics_.fallback_reason == ToolCallParseFallbackReason::None) {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        }
    }
    tool_text_.clear();
    mode_ = Mode::Content;
}

void OutputSession::consume(TokenId token) {
    if (is_stop(token)) {
        if (publish_stop_) { content_ += tokenizer_->token_bytes(token); }
        return;
    }
    const bool in_reasoning = mode_ == Mode::Reasoning || mode_ == Mode::ReasoningHeader;
    if (in_reasoning) { ++reasoning_tokens_; }
    if (mode_ == Mode::ToolCall) {
        if (token == kToolCallClose) {
            close_tool_call();
        } else {
            tool_text_ += tokenizer_->token_bytes(token);
        }
        return;
    }
    switch (token) {
    case kChannelOpen:
        if (mode_ == Mode::Content) {
            mode_ = Mode::ReasoningHeader;
            ++reasoning_tokens_;
        }
        return;
    case kChannelClose:
        if (in_reasoning) { close_reasoning(); }
        return;
    case kToolCallOpen:
        if (in_reasoning) { close_reasoning(); }
        diagnostics_.marker_seen = true;
        mode_                    = Mode::ToolCall;
        tool_text_.clear();
        return;
    default: break;
    }
    if (tokenizer_->is_special(token)) { return; }
    append_text(tokenizer_->token_bytes(token));
}

void OutputSession::flush_terminal() {
    if (mode_ == Mode::ToolCall) {
        content_ += "<|tool_call>";
        content_ += tool_text_;
        tool_text_.clear();
        if (diagnostics_.fallback_reason == ToolCallParseFallbackReason::None) {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        }
        mode_ = Mode::Content;
    } else if (mode_ == Mode::ReasoningHeader) {
        close_reasoning();
    }
}

PublishedOutput OutputSession::commit_preview() {
    for (const TokenId token : staged_tokens_) { consume(token); }
    const bool final = staged_terminal_;
    if (final) { flush_terminal(); }
    staged_tokens_.clear();
    staged_terminal_ = false;

    PublishedOutput published;
    OutputDelta reasoning{.channel = OutputChannel::Reasoning, .text = {}};
    drain_utf8(reasoning_, reasoning.text, final);
    if (!reasoning.text.empty()) { published.push_back(std::move(reasoning)); }
    OutputDelta content{.channel = OutputChannel::Content, .text = {}};
    drain_utf8(content_, content.text, final);
    if (!content.text.empty()) { published.push_back(std::move(content)); }
    if (thinking_.configured_budget) { thinking_.model_thinking_tokens = reasoning_tokens_; }
    return published;
}

} // namespace ninfer::models::gemma4
