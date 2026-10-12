#include "serve/openai_responses_request_internal.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::serve {

using Json = RequestJson;

void reject_nonnull_unknown_members(const Json& object,
                                    const std::unordered_set<std::string>& allowed,
                                    const char* param) {
    for (auto iterator = object.begin(); iterator != object.end(); ++iterator) {
        if (!allowed.contains(iterator.key()) && !iterator.value().is_null()) {
            bad_request("unsupported " + std::string(param) + " member: " + iterator.key(), param,
                        "parameter_not_supported");
        }
    }
}

namespace {

std::string item_id(const Json& item, const char* prefix) {
    if (!item.contains("id") || item.at("id").is_null()) {
        return new_openai_response_item_id(prefix);
    }
    if (!item.at("id").is_string() || item.at("id").get_ref<const std::string&>().empty()) {
        bad_request("input Item id must be a non-empty string", "input");
    }
    return item.at("id").get<std::string>();
}

ninfer::product::media_acquire::Source parse_image_source(const Json& part) {
    if (part.contains("file_id") && !part.at("file_id").is_null()) {
        bad_request("input_image.file_id requires a Files API, which NInfer does not provide",
                    "input", "file_inputs_not_supported");
    }
    if (!part.contains("image_url") || !part.at("image_url").is_string() ||
        part.at("image_url").get_ref<const std::string&>().empty()) {
        bad_request("input_image must contain a non-empty image_url", "input");
    }
    if (part.contains("detail") && !part.at("detail").is_null()) {
        if (!part.at("detail").is_string()) {
            bad_request("input_image.detail must be a string", "input");
        }
        if (part.at("detail").get<std::string>() != "auto") {
            bad_request("only input_image detail 'auto' is supported", "input",
                        "image_detail_not_supported");
        }
    }

    ninfer::product::media_acquire::Source source;
    source.value = part.at("image_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_image.image_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

ninfer::product::media_acquire::Source parse_video_source(const Json& part) {
    if (!part.contains("video_url") || !part.at("video_url").is_string() ||
        part.at("video_url").get_ref<const std::string&>().empty()) {
        bad_request("input_video must contain a non-empty video_url", "input");
    }
    ninfer::product::media_acquire::Source source;
    source.value = part.at("video_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_video.video_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

void apply_shared_breakpoint(ContentPart& part, const Json& wire, std::size_t& breakpoint_count) {
    if (!parse_openai_prompt_cache_breakpoint(wire, "input")) { return; }
    ++breakpoint_count;
    part.cache_boundary_after = CacheBoundary{};
}

struct ParsedMessage {
    ChatTurn turn;
    Json canonical;
};

void append_message_text(ParsedMessage& parsed, Json& content, const std::string& text,
                         const std::string& wire_type, const Json* wire,
                         std::size_t& breakpoint_count) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.text     = text;
    part.type_raw = wire_type;
    if (wire != nullptr) { apply_shared_breakpoint(part, *wire, breakpoint_count); }
    parsed.turn.content.push_back(std::move(part));

    Json canonical;
    if (wire_type == "refusal") {
        canonical = Json{{"type", "refusal"}, {"refusal", text}};
    } else {
        canonical = Json{{"type", wire_type}, {"text", text}};
        if (wire_type == "output_text") {
            canonical["annotations"] = wire != nullptr && wire->contains("annotations")
                                           ? wire->at("annotations")
                                           : Json::array();
            if (wire != nullptr && wire->contains("logprobs")) {
                canonical["logprobs"] = wire->at("logprobs");
            }
        }
    }
    if (wire != nullptr && wire->contains("prompt_cache_breakpoint")) {
        canonical["prompt_cache_breakpoint"] = wire->at("prompt_cache_breakpoint");
    }
    content.push_back(std::move(canonical));
}

void parse_message_content_part(const Json& value, ChatRole role, ParsedMessage& parsed,
                                Json& content, std::size_t& breakpoint_count) {
    if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
        bad_request("input message content parts must have a string type", "input");
    }
    const std::string type = value.at("type").get<std::string>();
    if (type == "input_text") {
        if (!value.contains("text") || !value.at("text").is_string()) {
            bad_request("input_text must contain a string text", "input");
        }
        append_message_text(parsed, content, value.at("text").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "output_text") {
        if (role != ChatRole::Assistant) {
            bad_request("output_text is only valid on assistant messages", "input");
        }
        if (!value.contains("text") || !value.at("text").is_string()) {
            bad_request("output_text must contain a string text", "input");
        }
        if (value.contains("annotations") && !value.at("annotations").is_null() &&
            !value.at("annotations").is_array()) {
            bad_request("output_text.annotations must be an array", "input");
        }
        if (value.contains("logprobs") && !value.at("logprobs").is_null() &&
            !value.at("logprobs").is_array()) {
            bad_request("output_text.logprobs must be an array", "input");
        }
        append_message_text(parsed, content, value.at("text").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "refusal") {
        if (role != ChatRole::Assistant) {
            bad_request("refusal is only valid on assistant messages", "input");
        }
        if (!value.contains("refusal") || !value.at("refusal").is_string()) {
            bad_request("refusal must contain a string refusal", "input");
        }
        append_message_text(parsed, content, value.at("refusal").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "input_image") {
        if (role != ChatRole::User && role != ChatRole::Assistant) {
            bad_request("input_image is only supported on user or assistant messages", "input");
        }
        ContentPart part;
        part.kind     = ContentKind::Image;
        part.type_raw = type;
        part.source   = parse_image_source(value);
        apply_shared_breakpoint(part, value, breakpoint_count);
        parsed.turn.content.push_back(std::move(part));
        Json canonical{
            {"type", "input_image"}, {"image_url", value.at("image_url")}, {"detail", "auto"}};
        if (value.contains("prompt_cache_breakpoint")) {
            canonical["prompt_cache_breakpoint"] = value.at("prompt_cache_breakpoint");
        }
        content.push_back(std::move(canonical));
        return;
    }
    if (type == "input_video") {
        if (role != ChatRole::User) {
            bad_request("input_video is only supported on user messages", "input");
        }
        ContentPart part;
        part.kind     = ContentKind::Video;
        part.type_raw = type;
        part.source   = parse_video_source(value);
        parsed.turn.content.push_back(std::move(part));
        content.push_back(Json{{"type", "input_video"}, {"video_url", value.at("video_url")}});
        return;
    }
    if (type == "input_file") {
        bad_request("input_file requires a Files API, which NInfer does not provide", "input",
                    "file_inputs_not_supported");
    }
    if (type == "input_audio") {
        bad_request("input_audio is not supported by the Engine", "input",
                    "audio_inputs_not_supported");
    }
    bad_request("unsupported message content type: " + type, "input", "modality_not_supported");
}

ParsedMessage parse_message_item(const Json& item, std::size_t index,
                                 std::size_t& breakpoint_count) {
    if (!item.contains("role") || !item.at("role").is_string()) {
        bad_request("input message " + std::to_string(index) + " must contain a string role",
                    "input");
    }
    const std::string role = item.at("role").get<std::string>();
    ChatRole parsed_role;
    if (role == "user") {
        parsed_role = ChatRole::User;
    } else if (role == "assistant") {
        parsed_role = ChatRole::Assistant;
    } else if (role == "system") {
        parsed_role = ChatRole::System;
    } else if (role == "developer") {
        parsed_role = ChatRole::Developer;
    } else {
        bad_request("unsupported input message role: " + role, "input", "unsupported_role");
    }

    if (item.contains("status") && !item.at("status").is_null() && !item.at("status").is_string()) {
        bad_request("input message status must be a string", "input");
    }
    if (item.contains("phase") && !item.at("phase").is_null() && !item.at("phase").is_string()) {
        bad_request("input message phase must be a string", "input");
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("input message " + std::to_string(index) + " must contain content", "input");
    }

    ParsedMessage parsed;
    parsed.turn.role = parsed_role;
    Json content     = Json::array();

    if (item.at("content").is_string()) {
        append_message_text(parsed, content, item.at("content").get<std::string>(),
                            parsed_role == ChatRole::Assistant ? "output_text" : "input_text",
                            nullptr, breakpoint_count);
    } else if (item.at("content").is_array()) {
        for (const Json& value : item.at("content")) {
            parse_message_content_part(value, parsed_role, parsed, content, breakpoint_count);
        }
    } else {
        bad_request("input message content must be a string or array", "input");
    }
    if (parsed.turn.content.empty()) {
        bad_request("input message content must not be empty", "input");
    }

    parsed.canonical = {{"id", item_id(item, "msg")},
                        {"type", "message"},
                        {"role", role},
                        {"content", std::move(content)}};
    if (item.contains("status") && !item.at("status").is_null()) {
        parsed.canonical["status"] = item.at("status");
    }
    if (item.contains("phase") && !item.at("phase").is_null()) {
        parsed.canonical["phase"] = item.at("phase");
    }
    return parsed;
}

std::string parse_reasoning_item(const Json& item, Json& canonical) {
    const bool has_encrypted =
        item.contains("encrypted_content") && !item.at("encrypted_content").is_null();
    if (has_encrypted && !item.at("encrypted_content").is_string()) {
        bad_request("reasoning encrypted_content must be a string", "input");
    }
    if (item.contains("summary") && !item.at("summary").is_null() &&
        !item.at("summary").is_array()) {
        bad_request("reasoning summary must be an array", "input");
    }
    if (!item.contains("content") || !item.at("content").is_array()) {
        bad_request("reasoning Item must contain a content array", "input");
    }

    std::string text;
    Json content = Json::array();
    for (const Json& part : item.at("content")) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string() ||
            part.at("type").get<std::string>() != "reasoning_text" || !part.contains("text") ||
            !part.at("text").is_string()) {
            bad_request("reasoning content only supports reasoning_text parts", "input");
        }
        text += part.at("text").get<std::string>();
        content.push_back(Json{{"type", "reasoning_text"}, {"text", part.at("text")}});
    }
    if (text.empty() &&
        (has_encrypted || (item.contains("summary") && !item.at("summary").is_null() &&
                           !item.at("summary").empty()))) {
        bad_request("reasoning Items require raw reasoning_text; summary or encrypted content "
                    "cannot reconstruct the model context",
                    "input", "reasoning_content_not_supported");
    }

    canonical = {{"id", item_id(item, "rs")},
                 {"type", "reasoning"},
                 {"summary", item.contains("summary") && !item.at("summary").is_null()
                                 ? item.at("summary")
                                 : Json::array()},
                 {"content", std::move(content)}};
    if (has_encrypted) { canonical["encrypted_content"] = item.at("encrypted_content"); }
    return text;
}

ToolCall parse_function_call_item(
    const Json& item, Json& canonical,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    static const std::unordered_set<std::string> allowed = {
        "id", "type", "call_id", "name", "arguments", "status", "caller", "namespace"};
    reject_nonnull_unknown_members(item, allowed, "input");
    if (item.contains("caller") && !item.at("caller").is_null()) {
        bad_request("function_call.caller is not supported", "input",
                    "tool_relationship_not_supported");
    }

    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get_ref<const std::string&>().empty()) {
        bad_request("function_call must contain a non-empty call_id", "input");
    }
    call.id                                        = item.at("call_id").get<std::string>();
    const OpenAIResponsesFunctionIdentity identity = function_identity(item, "input");
    call.name = lower_function_identity(identity, identities, "input");
    if (!item.contains("arguments") || !item.at("arguments").is_string()) {
        bad_request("function_call arguments must be a JSON string", "input");
    }
    call.arguments_json  = item.at("arguments").get<std::string>();
    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        bad_request("function_call arguments must encode a JSON object", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("partial function_call Items cannot be represented in model history", "input",
                    "partial_tool_call_not_supported");
    }
    canonical = {{"id", item_id(item, "fc")},
                 {"type", "function_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"arguments", call.arguments_json}};
    add_wire_function_identity(canonical, identity);
    return call;
}

ContentPart tool_output_text(std::string text, const Json* wire, std::size_t& breakpoint_count) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    if (wire != nullptr) { apply_shared_breakpoint(part, *wire, breakpoint_count); }
    return part;
}

ChatTurn parse_function_call_output_item(
    const Json& item, Json& canonical, std::size_t& breakpoint_count,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    static const std::unordered_set<std::string> allowed = {
        "id", "type", "call_id", "output", "status", "caller", "name", "namespace"};
    reject_nonnull_unknown_members(item, allowed, "input");
    if (item.contains("caller") && !item.at("caller").is_null()) {
        bad_request("function_call_output.caller is not supported", "input",
                    "tool_relationship_not_supported");
    }
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get_ref<const std::string&>().empty()) {
        bad_request("function_call_output must contain a non-empty call_id", "input");
    }
    if (!item.contains("output")) {
        bad_request("function_call_output must contain output", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("partial function_call_output Items cannot be represented in model history",
                    "input", "partial_tool_result_not_supported");
    }

    ChatTurn turn;
    turn.role                = ChatRole::Tool;
    turn.tool_call_id        = item.at("call_id").get<std::string>();
    const bool has_name      = item.contains("name") && !item.at("name").is_null();
    const bool has_namespace = item.contains("namespace") && !item.at("namespace").is_null();
    if (has_namespace && !has_name) {
        bad_request("function_call_output.namespace requires function_call_output.name", "input",
                    "invalid_tool_history");
    }
    std::optional<OpenAIResponsesFunctionIdentity> asserted_identity;
    if (has_name) {
        asserted_identity     = function_identity(item, "input");
        turn.tool_result_name = lower_function_identity(*asserted_identity, identities, "input");
    }
    if (item.at("output").is_string()) {
        turn.content.push_back(
            tool_output_text(item.at("output").get<std::string>(), nullptr, breakpoint_count));
    } else if (item.at("output").is_array()) {
        for (const Json& value : item.at("output")) {
            if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
                bad_request("function_call_output content parts must have a string type", "input");
            }
            const std::string type = value.at("type").get<std::string>();
            if (type == "input_text") {
                if (!value.contains("text") || !value.at("text").is_string()) {
                    bad_request("tool result input_text must contain a string text", "input");
                }
                turn.content.push_back(tool_output_text(value.at("text").get<std::string>(), &value,
                                                        breakpoint_count));
            } else if (type == "input_image") {
                ContentPart part;
                part.kind     = ContentKind::Image;
                part.type_raw = type;
                part.source   = parse_image_source(value);
                apply_shared_breakpoint(part, value, breakpoint_count);
                turn.content.push_back(std::move(part));
            } else if (type == "input_file") {
                bad_request("tool result input_file requires a Files API", "input",
                            "file_inputs_not_supported");
            } else {
                bad_request("unsupported function_call_output content type: " + type, "input",
                            "modality_not_supported");
            }
        }
        if (turn.content.empty()) {
            bad_request("function_call_output content must not be empty", "input");
        }
    } else {
        bad_request("function_call_output output must be a string or content array", "input");
    }

    canonical = {{"id", item_id(item, "fco")},
                 {"type", "function_call_output"},
                 {"status", "completed"},
                 {"call_id", turn.tool_call_id},
                 {"output", item.at("output")}};
    if (asserted_identity) { add_wire_function_identity(canonical, *asserted_identity); }
    return turn;
}

enum class AssistantInputPhase {
    Empty,
    Reasoning,
    Content,
    Calls,
};

[[noreturn]] void invalid_assistant_history(std::size_t index, std::string message) {
    bad_request("input Item " + std::to_string(index) + " " + std::move(message) +
                    "; supported assistant Item order is reasoning, message content, then "
                    "function calls",
                "input", "invalid_assistant_history");
}

struct AssistantInputRun {
    AssistantInputRun() { reset(); }

    void append_reasoning(std::string reasoning, std::size_t index) {
        switch (phase) {
        case AssistantInputPhase::Empty:
            turn.reasoning_content = std::move(reasoning);
            phase                  = AssistantInputPhase::Reasoning;
            return;
        case AssistantInputPhase::Reasoning:
            invalid_assistant_history(index, "reasoning cannot follow another reasoning Item");
        case AssistantInputPhase::Content:
            invalid_assistant_history(index, "reasoning cannot follow assistant message content");
        case AssistantInputPhase::Calls:
            invalid_assistant_history(index, "reasoning cannot follow function_call Items");
        }
        throw std::logic_error("unreachable assistant input phase");
    }

    void append_message(ChatTurn message, std::size_t index) {
        if (message.role != ChatRole::Assistant) {
            throw std::logic_error("assistant input run received a non-assistant message");
        }
        if (phase == AssistantInputPhase::Calls) {
            invalid_assistant_history(
                index, "assistant message content cannot follow function_call Items");
        }
        turn.content.insert(turn.content.end(), std::make_move_iterator(message.content.begin()),
                            std::make_move_iterator(message.content.end()));
        phase = AssistantInputPhase::Content;
    }

    void append_call(ToolCall call) {
        turn.tool_calls.push_back(std::move(call));
        phase = AssistantInputPhase::Calls;
    }

    void flush(std::vector<ChatTurn>& turns) {
        if (phase == AssistantInputPhase::Empty) { return; }
        if (!turn.reasoning_content.empty() || !turn.content.empty() || !turn.tool_calls.empty()) {
            turns.push_back(std::move(turn));
        }
        reset();
    }

private:
    void reset() {
        turn      = ChatTurn{};
        turn.role = ChatRole::Assistant;
        phase     = AssistantInputPhase::Empty;
    }

    ChatTurn turn;
    AssistantInputPhase phase = AssistantInputPhase::Empty;
};

} // namespace

void parse_input(const Json& input, OpenAIResponsesPromptRequest& out,
                 std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    Json values;
    if (input.is_string()) {
        values = Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", input}}});
    } else if (input.is_array()) {
        values = input;
    } else {
        bad_request("input must be a string or an array of Items", "input");
    }

    AssistantInputRun assistant;
    std::size_t breakpoint_count = 0;
    std::unordered_set<std::string> item_ids;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const Json& item = values.at(index);
        if (!item.is_object()) {
            bad_request("input Item " + std::to_string(index) + " must be an object", "input");
        }
        std::string type;
        if (item.contains("type") && !item.at("type").is_null()) {
            if (!item.at("type").is_string()) {
                bad_request("input Item type must be a string", "input");
            }
            type = item.at("type").get<std::string>();
        } else if (item.contains("role")) {
            type = "message";
        } else {
            bad_request("input Item must contain type", "input");
        }

        Json canonical;
        if (type == "message") {
            ParsedMessage message = parse_message_item(item, index, breakpoint_count);
            canonical             = std::move(message.canonical);
            if (message.turn.role == ChatRole::Assistant) {
                assistant.append_message(std::move(message.turn), index);
            } else {
                assistant.flush(out.input_turns);
                out.input_turns.push_back(std::move(message.turn));
            }
        } else if (type == "reasoning") {
            assistant.append_reasoning(parse_reasoning_item(item, canonical), index);
        } else if (type == "function_call") {
            assistant.append_call(parse_function_call_item(item, canonical, identities));
        } else if (type == "function_call_output") {
            ChatTurn result =
                parse_function_call_output_item(item, canonical, breakpoint_count, identities);
            assistant.flush(out.input_turns);
            out.input_turns.push_back(std::move(result));
        } else if (type == "input_file") {
            bad_request("input_file requires a Files API, which NInfer does not provide", "input",
                        "file_inputs_not_supported");
        } else {
            bad_request("unsupported input Item type: " + type, "input", "item_type_not_supported");
        }

        const std::string id = canonical.at("id").get<std::string>();
        if (!item_ids.insert(id).second) { bad_request("duplicate input Item id: " + id, "input"); }
        out.input_items.push_back(std::move(canonical));
    }
    assistant.flush(out.input_turns);
}

} // namespace ninfer::serve
