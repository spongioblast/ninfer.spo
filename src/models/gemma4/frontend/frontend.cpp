#include "models/gemma4/frontend/frontend.h"

#include "models/gemma4/frontend/image.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "text/jinja.h"

#include "ninfer/types/error.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::gemma4 {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kTurnEnd        = "<turn|>\n";
constexpr std::string_view kReasoningStart = "<|channel>thought\n";
constexpr std::string_view kImagePlaceholder = "<|image|>";
constexpr std::string_view kBoiToken         = "<|image>";
constexpr std::string_view kEoiToken         = "<image|>";
constexpr std::string_view kCallPrefix       = "call:";
constexpr TokenId kToolCallOpen              = 48;
constexpr TokenId kChannelClose              = 101;

void check_length(std::size_t tokens, std::uint32_t max_context) {
    if (tokens > max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prepared prompt has " + std::to_string(tokens) +
                               " tokens, beyond Engine max_context " + std::to_string(max_context));
    }
}

// The function name of each declared tool, in declaration order.
std::vector<std::string> tool_names(const std::vector<std::string>& tool_jsons) {
    std::vector<std::string> names;
    for (const std::string& tool : tool_jsons) {
        const Json parsed       = Json::parse(tool);
        const Json& function    = parsed.contains("function") ? parsed["function"] : parsed;
        if (function.contains("name") && function["name"].is_string()) {
            names.push_back(function["name"].get<std::string>());
        }
    }
    return names;
}

int single_token(const Tokenizer& tokenizer, std::string_view text, const char* name) {
    const std::vector<int> ids = tokenizer.encode(text);
    if (ids.size() != 1) {
        throw std::invalid_argument(std::string("gemma4 ") + name + " is not one tokenizer token");
    }
    return ids[0];
}

std::string message_text(const ChatMessage& message, std::vector<Gemma4PreparedImage>* images) {
    std::string text;
    for (const MessagePart& part : message.parts) {
        if (part.kind == MessagePartKind::Text) {
            text += part.text;
            continue;
        }
        if (part.kind != MessagePartKind::Media || part.media.kind != MediaKind::Image) {
            throw std::invalid_argument("gemma4 video and audio input is not wired");
        }
        if (images != nullptr) {
            images->push_back(gemma4_prepare_image(part.media.bytes, part.media));
        }
        text += kImagePlaceholder;
    }
    return text;
}

const char* role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::Developer: return "developer";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("gemma4 chat role is unknown");
}

Json tool_call_json(const ToolCall& call) {
    Json arguments = call.arguments_json.empty() ? Json::object() : Json::parse(call.arguments_json);
    if (!arguments.is_object()) {
        throw std::invalid_argument("gemma4 tool call arguments must be a JSON object");
    }
    Json function;
    function["name"]      = call.name;
    function["arguments"] = std::move(arguments);
    Json item;
    item["id"]       = call.id;
    item["type"]     = "function";
    item["function"] = std::move(function);
    return item;
}

Json template_context(const PromptInput& input, std::vector<Gemma4PreparedImage>* images) {
    const PromptOptions& options = input.options;
    Json context = options.chat_template_kwargs_json.empty()
                       ? Json::object()
                       : Json::parse(options.chat_template_kwargs_json);
    if (!context.is_object()) {
        throw std::invalid_argument("chat_template_kwargs must be a JSON object");
    }
    for (const char* reserved : {"messages", "tools", "add_generation_prompt", "bos_token"}) {
        if (context.contains(reserved)) {
            throw std::invalid_argument(std::string("chat_template_kwargs cannot override ") +
                                        reserved);
        }
    }
    if (options.enable_thinking) {
        context["enable_thinking"] = *options.enable_thinking;
    } else if (options.reasoning_effort && !context.contains("enable_thinking")) {
        context["enable_thinking"] = *options.reasoning_effort != ReasoningEffort::None;
    }
    if (options.preserve_thinking) { context["preserve_thinking"] = *options.preserve_thinking; }

    Json messages = Json::array();
    for (const ChatMessage& message : input.messages) {
        Json item;
        item["role"]    = role_name(message.role);
        item["content"] = message_text(message, images);
        if (!message.reasoning_content.empty()) {
            item["reasoning_content"] = message.reasoning_content;
        }
        if (!message.tool_calls.empty()) {
            Json calls = Json::array();
            for (const ToolCall& call : message.tool_calls) { calls.push_back(tool_call_json(call)); }
            item["tool_calls"] = std::move(calls);
        }
        if (!message.tool_call_id.empty()) { item["tool_call_id"] = message.tool_call_id; }
        messages.push_back(std::move(item));
    }
    context["messages"]  = std::move(messages);
    context["bos_token"] = "<bos>";
    context["add_generation_prompt"] =
        options.continuation == PromptContinuationMode::NewAssistantTurn;
    if (!options.tool_jsons.empty()) {
        Json tools = Json::array();
        for (const std::string& tool : options.tool_jsons) {
            Json parsed = Json::parse(tool);
            if (!parsed.is_object()) { throw std::invalid_argument("gemma4 tool must be an object"); }
            if (!parsed.contains("function")) {
                Json wrapped;
                wrapped["type"]     = "function";
                wrapped["function"] = std::move(parsed);
                parsed              = std::move(wrapped);
            }
            tools.push_back(std::move(parsed));
        }
        context["tools"] = std::move(tools);
    }
    return context;
}

// Token counts (before image expansion) of the prompt prefixes worth a checkpoint: the end of a
// leading system or developer turn, which holds the instructions and tools, and each explicit
// prompt_cache_breakpoint marker. A marker's prefix is the messages before it rendered on their
// own; its frontier is where those tokens stop matching the prompt.
std::vector<std::size_t> frontier_tokens(const text::JinjaTemplate& chat, const Tokenizer& tokenizer,
                                         int turn_close, const PromptInput& input,
                                         const std::vector<TokenId>& tokens) {
    std::vector<std::size_t> ends;
    const std::vector<ChatMessage>& messages = input.messages;
    const ChatRole lead                      = messages.front().role;
    if (lead == ChatRole::System || lead == ChatRole::Developer) {
        const auto close = std::find(tokens.begin(), tokens.end(), turn_close);
        if (close != tokens.end()) {
            ends.push_back(std::min<std::size_t>(tokens.size(), close - tokens.begin() + 2));
        }
    }
    const std::vector<PromptCacheMarker>& markers = input.context_cache.markers;
    for (std::size_t index = 0;
         index < markers.size() && index < kMaximumExplicitPromptCacheMarkers; ++index) {
        const PromptCacheMarker marker = markers[index];
        // The serve layer's automatic boundary after the last message is what the live state
        // continues anyway; capturing it every turn would only evict the useful checkpoints.
        if (marker.location == PromptCacheMarkerLocation::ToolBoundary ||
            !has_shared_candidate_evidence(marker.evidence, SharedCandidateEvidence::ExplicitBoundary)) {
            continue;
        }
        const bool leading = marker.location == PromptCacheMarkerLocation::LeadingInstructionBoundary;
        const std::size_t count = leading ? 1U : marker.after_message_count;
        if (count == 0 || count > messages.size()) { continue; }
        PromptInput partial;
        partial.options              = input.options;
        partial.options.continuation = PromptContinuationMode::ContinueFinalAssistant;
        partial.messages.assign(messages.begin(), messages.begin() + static_cast<std::ptrdiff_t>(count));
        std::vector<MessagePart>& parts = partial.messages.back().parts;
        if (marker.location == PromptCacheMarkerLocation::MessagePartBoundary &&
            marker.after_message_part_count < parts.size()) {
            parts.resize(marker.after_message_part_count);
        }
        if (leading) {
            std::size_t left = marker.leading_instruction_bytes;
            for (MessagePart& part : parts) {
                if (part.kind != MessagePartKind::Text) { continue; }
                part.text.resize(std::min(part.text.size(), left));
                left -= part.text.size();
            }
        }
        const std::vector<int> ids = tokenizer.encode(chat.render(template_context(partial, nullptr)).text);
        std::size_t length = 0;
        while (length < ids.size() && length < tokens.size() && ids[length] == tokens[length]) {
            ++length;
        }
        ends.push_back(length);
    }
    return ends;
}

} // namespace

struct Frontend::Impl {
    std::shared_ptr<const Tokenizer> tokenizer;
    text::JinjaTemplate chat;
    int boi_token_id   = 0;
    int eoi_token_id   = 0;
    int image_token_id = 0;
    int turn_close_id  = 0;
    std::uint32_t max_context = 0;
    Impl(const std::string& tokenizer_json, std::string chat_template, std::uint32_t max_context)
        : tokenizer(std::make_shared<const Tokenizer>(tokenizer_json)),
          chat(std::move(chat_template), "gemma4-chat"), max_context(max_context) {
        boi_token_id   = single_token(*tokenizer, kBoiToken, "boi");
        eoi_token_id   = single_token(*tokenizer, kEoiToken, "eoi");
        image_token_id = single_token(*tokenizer, kImagePlaceholder, "image");
        turn_close_id  = single_token(*tokenizer, kTurnEnd.substr(0, kTurnEnd.size() - 1), "turn end");
    }
};

Frontend::Frontend(std::string tokenizer_json, std::string chat_template, std::uint32_t max_context)
    : impl_(std::make_unique<Impl>(tokenizer_json, std::move(chat_template), max_context)) {}
Frontend::Frontend(Frontend&&) noexcept = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend() = default;

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const std::vector<int> ids = impl_->tokenizer->encode(text);
    return {ids.begin(), ids.end()};
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    if (token_ids.empty()) { throw std::invalid_argument("gemma4 prompt is empty"); }
    check_length(token_ids.size(), impl_->max_context);
    PromptSummary summary;
    summary.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    return PreparedPrompt(std::move(token_ids), summary, {});
}

std::string Frontend::render(const PromptInput& input) const {
    if (input.messages.empty()) { throw std::invalid_argument("gemma4 prompt has no messages"); }
    const bool continue_final =
        input.options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (continue_final && input.messages.back().role != ChatRole::Assistant) {
        throw std::invalid_argument("continuing a reply needs a final assistant message");
    }
    std::string rendered = impl_->chat.render(template_context(input, nullptr)).text;
    if (continue_final && rendered.ends_with(kTurnEnd)) {
        rendered.resize(rendered.size() - kTurnEnd.size());
    }
    return rendered;
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl&) const {
    std::vector<Gemma4PreparedImage> images;
    const bool continue_final =
        input.options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (input.messages.empty()) { throw std::invalid_argument("gemma4 prompt has no messages"); }
    if (continue_final && input.messages.back().role != ChatRole::Assistant) {
        throw std::invalid_argument("continuing a reply needs a final assistant message");
    }
    std::string rendered = impl_->chat.render(template_context(input, &images)).text;
    if (continue_final && rendered.ends_with(kTurnEnd)) {
        rendered.resize(rendered.size() - kTurnEnd.size());
    }
    std::vector<TokenId> tokens = tokenize_text(rendered);
    std::vector<TokenId> expanded;
    std::vector<std::int32_t> scatter;
    std::vector<std::uint32_t> expanded_at;
    expanded.reserve(tokens.size());
    expanded_at.reserve(tokens.size() + 1);
    std::size_t image_index = 0;
    for (const TokenId id : tokens) {
        expanded_at.push_back(static_cast<std::uint32_t>(expanded.size()));
        if (id == impl_->image_token_id && image_index < images.size()) {
            const Gemma4PreparedImage& image = images[image_index++];
            expanded.push_back(impl_->boi_token_id);
            for (int token = 0; token < image.valid; ++token) {
                scatter.push_back(static_cast<std::int32_t>(expanded.size()));
                expanded.push_back(impl_->image_token_id);
            }
            expanded.push_back(impl_->eoi_token_id);
            continue;
        }
        expanded.push_back(id);
    }
    if (image_index != images.size()) {
        throw std::invalid_argument("gemma4 image count does not match the chat template");
    }
    check_length(expanded.size(), impl_->max_context);
    expanded_at.push_back(static_cast<std::uint32_t>(expanded.size()));
    std::vector<std::uint32_t> frontiers;
    for (const std::size_t end : frontier_tokens(impl_->chat, *impl_->tokenizer,
                                                 impl_->turn_close_id, input, tokens)) {
        const std::uint32_t frontier = expanded_at[end];
        if (frontier > 0 && frontier < expanded.size()) { frontiers.push_back(frontier); }
    }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
    PromptSummary summary;
    summary.prompt_tokens       = static_cast<std::uint32_t>(expanded.size());
    summary.starts_in_reasoning = rendered.ends_with(kReasoningStart);
    summary.has_media           = !images.empty();
    PromptPreparationStats stats;
    stats.media_items     = images.size();
    stats.vision_tokens   = scatter.size();
    stats.raw_patches     = 0;
    for (const Gemma4PreparedImage& image : images) {
        stats.raw_patches += static_cast<std::uint64_t>(image.tokens);
        stats.media_bytes += image.patches.size() * 2;
    }
    PreparedPrompt prompt(std::move(expanded), summary, stats, std::move(images),
                          std::move(scatter));
    prompt.cache_frontiers_ = std::move(frontiers);
    if (!input.options.tool_jsons.empty()) {
        prompt.tool_openers_.push_back({"", tokenize_text(kCallPrefix)});
        for (std::string& name : tool_names(input.options.tool_jsons)) {
            std::vector<TokenId> tokens = tokenize_text(std::string(kCallPrefix) + name);
            prompt.tool_openers_.push_back({std::move(name), std::move(tokens)});
        }
    }
    return prompt;
}

void Frontend::apply_tool_choice(PreparedPrompt& prompt, const ToolChoice& tool_choice) {
    if (tool_choice.mode != ToolChoiceMode::Required || prompt.forced_call_) { return; }
    if (prompt.tool_openers_.empty()) {
        throw std::invalid_argument("gemma4 required tool choice needs declared tools");
    }
    std::string name;
    if (tool_choice.allowed_names && tool_choice.allowed_names->size() == 1) {
        name = tool_choice.allowed_names->front();
    }
    const auto opener =
        std::find_if(prompt.tool_openers_.begin(), prompt.tool_openers_.end(),
                     [&](const PreparedPrompt::ToolOpener& item) { return item.name == name; });
    if (opener == prompt.tool_openers_.end()) {
        throw std::invalid_argument("gemma4 tool choice names an undeclared tool: " + name);
    }
    if (prompt.summary_.starts_in_reasoning) {
        prompt.tokens_.push_back(kChannelClose);
        prompt.summary_.starts_in_reasoning = false;
    }
    prompt.tokens_.push_back(kToolCallOpen);
    prompt.tokens_.insert(prompt.tokens_.end(), opener->tokens.begin(), opener->tokens.end());
    prompt.summary_.prompt_tokens = static_cast<std::uint32_t>(prompt.tokens_.size());
    prompt.forced_call_           = std::string(kCallPrefix) + name;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}

ModelSamplingDefaults Frontend::sampling_defaults() const noexcept {
    ModelSamplingDefaults defaults;
    const SamplingPreset preset{.temperature = 1.0F, .top_k = 20, .top_p = 0.95F};
    defaults.thinking     = preset;
    defaults.non_thinking = preset;
    return defaults;
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions&,
                                            const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    if (constraint) { throw std::invalid_argument("gemma4 grammar constraints are not wired"); }
    if (tool_choice.mode == ToolChoiceMode::Required && !prompt.forced_call_) {
        throw std::invalid_argument("gemma4 required tool choice was not applied to the prompt");
    }
    OutputSession::Setup setup;
    setup.stop_ids = caller_stop.token_ids;
    if (caller_stop.include_model_defaults) {
        for (const TokenId id : {TokenId{1}, TokenId{106}, TokenId{50}}) {
            if (std::find(setup.stop_ids.begin(), setup.stop_ids.end(), id) ==
                setup.stop_ids.end()) {
                setup.stop_ids.push_back(id);
            }
        }
    }
    setup.stop_strings               = caller_stop.strings;
    setup.publish_stop               = caller_stop.publish_stop_token;
    setup.thinking.configured_budget = thinking.budget;
    setup.starts_in_reasoning        = prompt.summary().starts_in_reasoning;
    setup.forced_call                = prompt.forced_call_;
    setup.allowed_names              = tool_choice.allowed_names;
    return OutputSession(impl_->tokenizer, std::move(setup));
}

} // namespace ninfer::models::gemma4
