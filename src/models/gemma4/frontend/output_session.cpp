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

// The longest proper prefix of `stop` that `text` ends with.
std::size_t suffix_prefix(std::string_view text, std::string_view stop) {
    for (std::size_t length = std::min(text.size(), stop.size() - 1); length > 0; --length) {
        if (text.ends_with(stop.substr(0, length))) { return length; }
    }
    return 0;
}

// Publishes the complete UTF-8 prefix of `pending` except the last `hold` bytes.
std::string drain_held(std::string& pending, std::size_t hold, bool final) {
    std::string head = pending.substr(0, pending.size() - hold);
    std::string out;
    drain_utf8(head, out, final);
    pending = head + pending.substr(pending.size() - hold);
    return out;
}

} // namespace

OutputSession::OutputSession(std::shared_ptr<const Tokenizer> tokenizer, Setup setup)
    : tokenizer_(std::move(tokenizer)), stop_ids_(std::move(setup.stop_ids)),
      stop_strings_(std::move(setup.stop_strings)),
      allowed_names_(std::move(setup.allowed_names)), publish_stop_(setup.publish_stop),
      thinking_(setup.thinking) {
    for (const StopString& stop : stop_strings_) {
        if (stop.text.empty()) { throw std::invalid_argument("gemma4 stop strings must be non-empty"); }
    }
    state_.mode = setup.starts_in_reasoning ? Mode::Reasoning : Mode::Content;
    if (setup.forced_call) {
        state_.mode                    = Mode::ToolCall;
        state_.tool_text               = *setup.forced_call;
        state_.diagnostics.marker_seen = true;
    }
}

bool OutputSession::is_stop(TokenId token) const noexcept {
    return std::find(stop_ids_.begin(), stop_ids_.end(), token) != stop_ids_.end();
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (!state_.matched_stop) { return std::nullopt; }
    return stop_strings_.at(*state_.matched_stop).text;
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
    staged_ = state_;
    for (const TokenId token : tokens) {
        if (staged_tokens_.size() == total_budget_remaining) { break; }
        staged_tokens_.push_back(token);
        consume(staged_, token);
        if (is_stop(token)) {
            decision.finish_reason = FinishReason::StopToken;
            break;
        }
        if (match_stop_string(staged_)) {
            decision.finish_reason = FinishReason::StopString;
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
    staged_          = state_;
    staged_terminal_ = true;
    return decision;
}

void OutputSession::append_text(State& state, const std::string& bytes) {
    switch (state.mode) {
    case Mode::Content: state.content += bytes; return;
    case Mode::Reasoning: state.reasoning += bytes; return;
    case Mode::ToolCall: state.tool_text += bytes; return;
    case Mode::ReasoningHeader: break;
    }
    state.header += bytes;
    if (state.header.size() < kThoughtHeader.size() &&
        kThoughtHeader.substr(0, state.header.size()) == state.header) {
        return;
    }
    state.mode = Mode::Reasoning;
    if (std::string_view(state.header).starts_with(kThoughtHeader)) {
        state.reasoning.append(state.header, kThoughtHeader.size());
    } else {
        state.reasoning += state.header;
    }
    state.header.clear();
}

void OutputSession::close_reasoning(State& state) {
    if (state.mode == Mode::ReasoningHeader) {
        if (state.header != kThoughtHeader.substr(0, state.header.size())) {
            state.reasoning += state.header;
        }
        state.header.clear();
    }
    state.mode = Mode::Content;
}

void OutputSession::close_tool_call(State& state) const {
    auto call = parse_tool_call_body(state.tool_text);
    auto reason = ToolCallParseFallbackReason::MalformedStructure;
    if (call && allowed_names_ &&
        std::find(allowed_names_->begin(), allowed_names_->end(), call->name) ==
            allowed_names_->end()) {
        call.reset();
        reason = ToolCallParseFallbackReason::UndeclaredTool;
    }
    if (call) {
        state.tool_calls.push_back(std::move(*call));
        ++state.diagnostics.structured_call_count;
    } else {
        state.content += "<|tool_call>";
        state.content += state.tool_text;
        state.content += "<tool_call|>";
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = reason;
        }
    }
    state.tool_text.clear();
    state.mode = Mode::Content;
}

void OutputSession::consume(State& state, TokenId token) const {
    if (is_stop(token)) {
        if (publish_stop_) { state.content += tokenizer_->token_bytes(token); }
        return;
    }
    const bool in_reasoning = state.mode == Mode::Reasoning || state.mode == Mode::ReasoningHeader;
    if (in_reasoning) { ++state.reasoning_tokens; }
    if (state.mode == Mode::ToolCall) {
        if (token == kToolCallClose) {
            close_tool_call(state);
        } else {
            state.tool_text += tokenizer_->token_bytes(token);
        }
        return;
    }
    switch (token) {
    case kChannelOpen:
        if (state.mode == Mode::Content) {
            state.mode = Mode::ReasoningHeader;
            ++state.reasoning_tokens;
        }
        return;
    case kChannelClose:
        if (in_reasoning) { close_reasoning(state); }
        return;
    case kToolCallOpen:
        if (in_reasoning) { close_reasoning(state); }
        state.diagnostics.marker_seen = true;
        state.mode                    = Mode::ToolCall;
        state.tool_text.clear();
        return;
    default: break;
    }
    if (tokenizer_->is_special(token)) { return; }
    append_text(state, tokenizer_->token_bytes(token));
}

void OutputSession::flush_terminal(State& state) {
    if (state.mode == Mode::ToolCall) {
        state.content += "<|tool_call>";
        state.content += state.tool_text;
        state.tool_text.clear();
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        }
        state.mode = Mode::Content;
    } else if (state.mode == Mode::ReasoningHeader) {
        close_reasoning(state);
    }
}

// The earliest stop string in the unpublished channel text (declaration order breaks a tie at
// the same byte) cuts that channel there.
bool OutputSession::match_stop_string(State& state) const {
    std::optional<std::size_t> best;
    std::size_t best_at = 0;
    for (std::size_t index = 0; index < stop_strings_.size(); ++index) {
        const StopString& stop = stop_strings_[index];
        const std::string& text =
            stop.channel == OutputChannel::Content ? state.content : state.reasoning;
        const std::size_t at = text.find(stop.text);
        if (at == std::string::npos) { continue; }
        const bool same_channel = best && stop_strings_[*best].channel == stop.channel;
        if (!best || (same_channel && at < best_at)) {
            best    = index;
            best_at = at;
        }
    }
    if (!best) { return false; }
    const StopString& stop = stop_strings_[*best];
    std::string& text = stop.channel == OutputChannel::Content ? state.content : state.reasoning;
    text.resize(best_at);
    if (stop.include_in_output) { text += stop.text; }
    state.matched_stop = *best;
    return true;
}

std::size_t OutputSession::stop_hold(const std::string& text, OutputChannel channel) const {
    std::size_t hold = 0;
    for (const StopString& stop : stop_strings_) {
        if (stop.channel == channel) { hold = std::max(hold, suffix_prefix(text, stop.text)); }
    }
    return hold;
}

PublishedOutput OutputSession::commit_preview() {
    state_           = std::move(staged_);
    staged_          = {};
    const bool final = staged_terminal_;
    if (final) { flush_terminal(state_); }
    staged_tokens_.clear();
    staged_terminal_ = false;

    PublishedOutput published;
    const auto held = [&](const std::string& text, OutputChannel channel) {
        return final ? std::size_t{0} : stop_hold(text, channel);
    };
    OutputDelta reasoning{
        .channel = OutputChannel::Reasoning,
        .text    = drain_held(state_.reasoning, held(state_.reasoning, OutputChannel::Reasoning),
                              final)};
    if (!reasoning.text.empty()) { published.push_back(std::move(reasoning)); }
    OutputDelta content{
        .channel = OutputChannel::Content,
        .text = drain_held(state_.content, held(state_.content, OutputChannel::Content), final)};
    if (!content.text.empty()) { published.push_back(std::move(content)); }
    if (thinking_.configured_budget) {
        thinking_.model_thinking_tokens = state_.reasoning_tokens;
    }
    return published;
}

} // namespace ninfer::models::gemma4
