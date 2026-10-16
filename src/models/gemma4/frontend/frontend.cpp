#include "models/gemma4/frontend/frontend.h"

#include "models/gemma4/frontend/image.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "text/jinja.h"

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

} // namespace

struct Frontend::Impl {
    std::shared_ptr<const Tokenizer> tokenizer;
    text::JinjaTemplate chat;
    int boi_token_id   = 0;
    int eoi_token_id   = 0;
    int image_token_id = 0;
    explicit Impl(const std::string& tokenizer_json, std::string chat_template)
        : tokenizer(std::make_shared<const Tokenizer>(tokenizer_json)),
          chat(std::move(chat_template), "gemma4-chat") {
        boi_token_id   = single_token(*tokenizer, kBoiToken, "boi");
        eoi_token_id   = single_token(*tokenizer, kEoiToken, "eoi");
        image_token_id = single_token(*tokenizer, kImagePlaceholder, "image");
    }
};

Frontend::Frontend(std::string tokenizer_json, std::string chat_template)
    : impl_(std::make_unique<Impl>(tokenizer_json, std::move(chat_template))) {}
Frontend::Frontend(Frontend&&) noexcept = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend() = default;

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const std::vector<int> ids = impl_->tokenizer->encode(text);
    return {ids.begin(), ids.end()};
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    if (token_ids.empty()) { throw std::invalid_argument("gemma4 prompt is empty"); }
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
    expanded.reserve(tokens.size());
    std::size_t image_index = 0;
    for (const TokenId id : tokens) {
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
    return PreparedPrompt(std::move(expanded), summary, stats, std::move(images),
                          std::move(scatter));
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
    if (tool_choice.mode == ToolChoiceMode::Required) {
        throw std::invalid_argument("gemma4 required tool choice is not wired");
    }
    if (!caller_stop.strings.empty()) {
        throw std::invalid_argument("gemma4 stop strings are not wired");
    }
    std::vector<TokenId> stops = caller_stop.token_ids;
    if (caller_stop.include_model_defaults) {
        for (const TokenId id : {TokenId{1}, TokenId{106}, TokenId{50}}) {
            if (std::find(stops.begin(), stops.end(), id) == stops.end()) { stops.push_back(id); }
        }
    }
    ThinkingBudgetStats stats;
    stats.configured_budget = thinking.budget;
    return OutputSession(impl_->tokenizer, std::move(stops), caller_stop.publish_stop_token, stats,
                         prompt.summary().starts_in_reasoning);
}

} // namespace ninfer::models::gemma4
