#include "serve/openai_responses.h"
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


namespace ninfer::serve {

using Json = RequestJson;

namespace {

void require_object(const Json& value, std::string_view name = "request body") {
    if (!value.is_object()) { bad_request(std::string(name) + " must be a JSON object"); }
}

std::string require_function_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    const std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

std::optional<std::string> optional_namespace_name(const Json& object, const char* param) {
    if (!object.contains("namespace") || object.at("namespace").is_null()) { return std::nullopt; }
    if (!object.at("namespace").is_string()) {
        bad_request("function namespace must be a string or null", param);
    }
    const std::string name = object.at("namespace").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("function namespace must match [A-Za-z0-9_-]{1,64}", param,
                    "invalid_tool_name");
    }
    return name;
}

std::string require_namespace_tool_name(const Json& object) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("namespace name must be a string", "tools");
    }
    const std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("namespace name must match [A-Za-z0-9_-]{1,64}", "tools", "invalid_tool_name");
    }
    return name;
}

} // namespace

OpenAIResponsesFunctionIdentity function_identity(const Json& object, const char* param) {
    return OpenAIResponsesFunctionIdentity{.name = require_function_name(object, param),
                                           .wire_namespace =
                                               optional_namespace_name(object, param)};
}

std::string lower_function_identity(
    const OpenAIResponsesFunctionIdentity& identity,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities,
    const char* param) {
    const std::string engine_name =
        identity.wire_namespace ? *identity.wire_namespace + "__" + identity.name : identity.name;
    if (!valid_tool_name(engine_name, 64)) {
        bad_request("flattened function identity '" + engine_name +
                        "' exceeds the Engine tool-name contract [A-Za-z0-9_-]{1,64}",
                    param, "invalid_tool_name");
    }
    const auto [position, inserted] = identities.emplace(engine_name, identity);
    if (!inserted && position->second != identity) {
        bad_request("function identity collision after namespace translation: '" + engine_name +
                        "'",
                    param, "duplicate_tool_name");
    }
    return engine_name;
}

void add_wire_function_identity(Json& object, const OpenAIResponsesFunctionIdentity& identity) {
    object["name"] = identity.name;
    if (identity.wire_namespace) { object["namespace"] = *identity.wire_namespace; }
}

namespace {


struct ParsedPromptFields {
    OpenAIResponsesPromptRequest prompt;
    Json wire_tools          = Json::array();
    Json wire_tool_choice    = "auto";
    bool parallel_tool_calls = true;
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity> tool_identities;
};

struct ParsedFunctionTool {
    ToolDefinition definition;
    Json canonical;
    std::string engine_name;
};

ParsedFunctionTool
parse_function_tool(const Json& item, std::optional<std::string> wire_namespace,
                    std::string_view namespace_description,
                    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities,
                    std::string schema_param) {
    static const std::unordered_set<std::string> allowed_members = {
        "type",          "name",         "description", "parameters", "strict", "allowed_callers",
        "defer_loading", "output_schema"};
    reject_nonnull_unknown_members(item, allowed_members, "tools");

    const OpenAIResponsesFunctionIdentity identity{.name = require_function_name(item, "tools"),
                                                   .wire_namespace = std::move(wire_namespace)};
    ParsedFunctionTool parsed;
    parsed.definition.schema_param = std::move(schema_param);
    parsed.engine_name     = lower_function_identity(identity, identities, "tools");
    parsed.definition.name = parsed.engine_name;

    std::string function_description;
    if (item.contains("description") && !item.at("description").is_null()) {
        if (!item.at("description").is_string()) {
            bad_request("function description must be a string", "tools");
        }
        function_description = item.at("description").get<std::string>();
    }
    if (!namespace_description.empty()) {
        parsed.definition.description = std::string(namespace_description);
        if (!function_description.empty()) {
            parsed.definition.description += "\n\n" + function_description;
        }
    } else {
        parsed.definition.description = function_description;
    }

    Json parameters = Json{{"type", "object"}, {"properties", Json::object()}};
    if (item.contains("parameters") && !item.at("parameters").is_null()) {
        if (!item.at("parameters").is_object()) {
            bad_request("function parameters must be a JSON object", "tools");
        }
        parameters = item.at("parameters");
    }
    if (item.contains("strict") && !item.at("strict").is_null()) {
        if (!item.at("strict").is_boolean()) {
            bad_request("function strict must be a boolean", "tools");
        }
        parsed.definition.strict = item.at("strict").get<bool>();
    }
    if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
        if (!item.at("defer_loading").is_boolean()) {
            bad_request("function defer_loading must be a boolean", "tools");
        }
        if (item.at("defer_loading").get<bool>()) {
            bad_request("deferred tool loading is not supported", "tools",
                        "deferred_tools_not_supported");
        }
    }
    if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
        const Json& callers = item.at("allowed_callers");
        if (!callers.is_array()) {
            bad_request("function allowed_callers must be an array", "tools");
        }
        bool direct = false;
        for (const Json& caller : callers) {
            if (!caller.is_string()) {
                bad_request("function allowed_callers entries must be strings", "tools");
            }
            direct = direct || caller.get<std::string>() == "direct";
        }
        if (!direct) {
            bad_request("function allowed_callers must permit direct invocation", "tools",
                        "tool_caller_not_supported");
        }
    }
    if (item.contains("output_schema") && !item.at("output_schema").is_null()) {
        bad_request("function output_schema cannot be enforced", "tools",
                    "tool_output_schema_not_supported");
    }

    parsed.definition.input_schema_json = parameters.dump();
    parsed.canonical                    = {{"type", "function"},
                                           {"name", identity.name},
                                           {"parameters", parameters},
                                           {"strict", parsed.definition.strict}};
    if (!function_description.empty()) {
        parsed.canonical["description"] = std::move(function_description);
    }
    if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
        parsed.canonical["allowed_callers"] = item.at("allowed_callers");
    }
    if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
        parsed.canonical["defer_loading"] = false;
    }
    return parsed;
}

void parse_tools(const Json& body, ParsedPromptFields& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }

    static const std::unordered_set<std::string> namespace_members = {"type", "name", "description",
                                                                      "tools"};
    std::unordered_set<std::string> declared_names;
    std::unordered_set<std::string> namespace_names;
    const auto append_function = [&](ParsedFunctionTool parsed) {
        if (!declared_names.insert(parsed.engine_name).second) {
            bad_request("duplicate function tool identity: " + parsed.engine_name, "tools",
                        "duplicate_tool_name");
        }
        out.prompt.generation.tools.push_back(std::move(parsed.definition));
        return std::move(parsed.canonical);
    };

    for (const Json& item : body.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must be objects with a string type", "tools");
        }
        const std::string type = item.at("type").get<std::string>();
        if (type == "function") {
            out.wire_tools.push_back(append_function(parse_function_tool(
                item, std::nullopt, {}, out.tool_identities,
                "tools/" + std::to_string(out.wire_tools.size()) + "/parameters")));
            continue;
        }
        if (type != "namespace") {
            bad_request("tool type '" + type +
                            "' requires an executor that NInfer does not provide",
                        "tools", "tool_type_not_supported");
        }

        // OpenAI Responses beta groups functions/custom tools under a namespace. NInfer lowers
        // only nested functions because custom tools require unsupported free-form decoding.
        reject_nonnull_unknown_members(item, namespace_members, "tools");
        const std::string namespace_name = require_namespace_tool_name(item);
        if (!namespace_names.insert(namespace_name).second) {
            bad_request("duplicate namespace tool name: " + namespace_name, "tools",
                        "duplicate_tool_name");
        }
        std::string namespace_description;
        if (item.contains("description") && !item.at("description").is_null()) {
            if (!item.at("description").is_string()) {
                bad_request("namespace description must be a string", "tools");
            }
            namespace_description = item.at("description").get<std::string>();
        }
        if (!item.contains("tools") || !item.at("tools").is_array()) {
            bad_request("namespace tools must contain a tools array", "tools");
        }
        Json canonical = {
            {"type", "namespace"}, {"name", namespace_name}, {"tools", Json::array()}};
        if (!namespace_description.empty()) { canonical["description"] = namespace_description; }
        for (const Json& nested : item.at("tools")) {
            if (!nested.is_object() || !nested.contains("type") || !nested.at("type").is_string()) {
                bad_request("namespace tool entries must be objects with a string type", "tools");
            }
            const std::string nested_type = nested.at("type").get<std::string>();
            if (nested_type != "function") {
                bad_request("nested tool type '" + nested_type +
                                "' cannot be represented by the Engine",
                            "tools", "tool_type_not_supported");
            }
            canonical["tools"].push_back(append_function(parse_function_tool(
                nested, namespace_name, namespace_description, out.tool_identities,
                "tools/" + std::to_string(out.wire_tools.size()) + "/tools/" +
                    std::to_string(canonical["tools"].size()) + "/parameters")));
        }
        out.wire_tools.push_back(std::move(canonical));
    }
}

void filter_allowed_tools(const Json& choice, ParsedPromptFields& out) {
    static const std::unordered_set<std::string> allowed_choice = {"type", "mode", "tools"};
    reject_nonnull_unknown_members(choice, allowed_choice, "tool_choice");
    if (!choice.contains("mode") || !choice.at("mode").is_string()) {
        bad_request("allowed_tools tool_choice must contain a string mode", "tool_choice");
    }
    const auto mode = choice.at("mode").get<std::string>();
    if (mode != "auto" && mode != "required")
        bad_request("allowed_tools mode must be auto or required", "tool_choice");
    out.prompt.generation.tool_choice.mode =
        mode == "required" ? ToolChoiceMode::Required : ToolChoiceMode::Auto;
    if (!choice.contains("tools") || !choice.at("tools").is_array()) {
        bad_request("allowed_tools tool_choice must contain a tools array", "tool_choice");
    }

    std::unordered_set<std::string> declared;
    for (const ToolDefinition& tool : out.prompt.generation.tools) { declared.insert(tool.name); }
    std::unordered_set<std::string> selected;
    for (const Json& item : choice.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string() ||
            item.at("type").get<std::string>() != "function") {
            bad_request("allowed_tools only supports function entries", "tool_choice",
                        "tool_choice_not_supported");
        }
        static const std::unordered_set<std::string> allowed_entry = {"type", "name", "namespace"};
        reject_nonnull_unknown_members(item, allowed_entry, "tool_choice");
        const OpenAIResponsesFunctionIdentity identity = function_identity(item, "tool_choice");
        const std::string name =
            lower_function_identity(identity, out.tool_identities, "tool_choice");
        if (!declared.contains(name)) {
            const std::string wire_name = identity.wire_namespace
                                              ? *identity.wire_namespace + "." + identity.name
                                              : identity.name;
            bad_request("allowed_tools references undeclared function '" + wire_name + "'",
                        "tool_choice", "invalid_tool_choice");
        }
        selected.insert(name);
    }

    out.prompt.generation.tool_choice.allowed_names.emplace();
    for (const ToolDefinition& tool : out.prompt.generation.tools)
        if (selected.contains(tool.name))
            out.prompt.generation.tool_choice.allowed_names->push_back(tool.name);
}

void parse_tool_choice(const Json& body, ParsedPromptFields& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) {
        out.wire_tool_choice = "auto";
        return;
    }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            out.prompt.generation.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            out.prompt.generation.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            out.prompt.generation.tool_choice.mode = ToolChoiceMode::Required;
        } else {
            bad_request("tool_choice must be 'auto', 'none', 'required', or a supported object",
                        "tool_choice");
        }
        out.wire_tool_choice = value;
        return;
    }
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string()) {
        bad_request("tool_choice must be a string or typed object", "tool_choice");
    }
    const auto type = choice.at("type").get<std::string>();
    if (type == "function") {
        const auto identity = function_identity(choice, "tool_choice");
        const auto name     = lower_function_identity(identity, out.tool_identities, "tool_choice");
        out.prompt.generation.tool_choice.mode          = ToolChoiceMode::Required;
        out.prompt.generation.tool_choice.allowed_names = std::vector<std::string>{name};
        out.prompt.generation.tool_choice.parallel      = false;
    } else if (type == "allowed_tools")
        filter_allowed_tools(choice, out);
    else
        bad_request("unsupported tool_choice type", "tool_choice");
    out.wire_tool_choice = choice;
}

void parse_reasoning(const Json& body, OpenAIResponsesPromptRequest& out) {
    if (!body.contains("reasoning") || body.at("reasoning").is_null()) { return; }
    const Json& reasoning = body.at("reasoning");
    if (!reasoning.is_object()) { bad_request("reasoning must be an object", "reasoning"); }
    static const std::unordered_set<std::string> allowed = {"effort", "context", "summary",
                                                            "generate_summary", "mode"};
    reject_nonnull_unknown_members(reasoning, allowed, "reasoning");
    for (const char* key : {"context", "summary", "generate_summary", "mode"}) {
        if (reasoning.contains(key) && !reasoning.at(key).is_null()) {
            bad_request("reasoning." + std::string(key) +
                            " changes reasoning input or output and is not supported",
                        "reasoning", "reasoning_option_not_supported");
        }
    }
    if (!reasoning.contains("effort") || reasoning.at("effort").is_null()) { return; }
    if (!reasoning.at("effort").is_string()) {
        bad_request("reasoning.effort must be a string", "reasoning");
    }
    const std::string value = reasoning.at("effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> effort = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("reasoning.effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning");
    }
    out.generation.reasoning_effort = *effort;
}

void parse_text(const Json& body, GenerationRequest& request) {
    if (!body.contains("text") || body.at("text").is_null()) { return; }
    const Json& text = body.at("text");
    if (!text.is_object()) { bad_request("text must be an object", "text"); }
    static const std::unordered_set<std::string> allowed = {"format", "verbosity"};
    reject_nonnull_unknown_members(text, allowed, "text");
    if (text.contains("format") && !text.at("format").is_null()) {
        parse_json_output_format(text["format"], request, "text.format",
                                 JsonFormatProtocol::Responses);
    }

    if (text.contains("verbosity") && !text.at("verbosity").is_null()) {
        if (!text.at("verbosity").is_string()) {
            bad_request("text.verbosity must be a string", "text");
        }
        const std::string verbosity = text.at("verbosity").get<std::string>();
        if (verbosity != "medium") {
            bad_request("text.verbosity '" + verbosity + "' cannot be enforced by the Engine",
                        "text", "verbosity_not_supported");
        }
    }
}

void parse_preserve_thinking(const Json& body, OpenAIResponsesPromptRequest& out) {
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean or null", "preserve_thinking");
        }
        out.generation.preserve_thinking = body.at("preserve_thinking").get<bool>();
    }
    if (!body.contains("chat_template_kwargs") || body.at("chat_template_kwargs").is_null()) {
        return;
    }
    const Json& kwargs = body.at("chat_template_kwargs");
    if (!kwargs.is_object()) {
        bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
    }
    out.generation.chat_template_kwargs_json = kwargs.dump();
    if (!kwargs.contains("preserve_thinking") || kwargs.at("preserve_thinking").is_null()) {
        return;
    }
    if (!kwargs.at("preserve_thinking").is_boolean()) {
        bad_request("chat_template_kwargs.preserve_thinking must be a boolean or null",
                    "chat_template_kwargs");
    }
    const bool nested = kwargs.at("preserve_thinking").get<bool>();
    if (out.generation.preserve_thinking && *out.generation.preserve_thinking != nested) {
        bad_request("conflicting preserve_thinking values", "preserve_thinking",
                    "conflicting_template_option");
    }
    out.generation.preserve_thinking = nested;
}

void parse_truncation(const Json& body) {
    if (!body.contains("truncation") || body.at("truncation").is_null()) { return; }
    if (!body.at("truncation").is_string()) {
        bad_request("truncation must be a string", "truncation");
    }
    if (body.at("truncation").get<std::string>() != "disabled") {
        bad_request("truncation 'auto' would discard input Items and is not supported",
                    "truncation", "truncation_not_supported");
    }
}

ParsedPromptFields parse_prompt_fields(const Json& body, const RequestLimits& limits) {
    ParsedPromptFields out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get_ref<const std::string&>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.prompt.model = body.at("model").get<std::string>();
    if (body.contains("input") && !body.at("input").is_null()) {
        parse_input(body.at("input"), out.prompt, out.tool_identities);
    }
    if (body.contains("instructions") && !body.at("instructions").is_null()) {
        if (!body.at("instructions").is_string()) {
            bad_request("instructions must be a string", "instructions");
        }
        out.prompt.instructions = body.at("instructions").get<std::string>();
    }
    if (body.contains("previous_response_id") && !body.at("previous_response_id").is_null()) {
        if (!body.at("previous_response_id").is_string() ||
            body.at("previous_response_id").get_ref<const std::string&>().empty()) {
            bad_request("previous_response_id must be a non-empty string", "previous_response_id");
        }
        out.prompt.previous_response_id = body.at("previous_response_id").get<std::string>();
    }

    parse_tools(body, out);
    parse_tool_choice(body, out);
    out.parallel_tool_calls = optional_bool(body, "parallel_tool_calls", true);
    out.prompt.generation.tool_choice.parallel &= out.parallel_tool_calls;
    parse_reasoning(body, out.prompt);
    parse_text(body, out.prompt.generation);
    parse_truncation(body);
    parse_preserve_thinking(body, out.prompt);
    out.prompt.generation.max_tokens = limits.default_max_tokens;
    return out;
}

void validate_metadata(const Json& body, nlohmann::json& metadata) {
    if (!body.contains("metadata") || body.at("metadata").is_null()) { return; }
    if (!body.at("metadata").is_object()) { bad_request("metadata must be an object", "metadata"); }
    if (body.at("metadata").size() > 16) {
        bad_request("metadata supports at most 16 entries", "metadata");
    }
    for (auto iterator = body.at("metadata").begin(); iterator != body.at("metadata").end();
         ++iterator) {
        if (iterator.key().size() > 64 || !iterator.value().is_string() ||
            iterator.value().get_ref<const std::string&>().size() > 512) {
            bad_request("metadata keys must be at most 64 characters and string values at most "
                        "512 characters",
                        "metadata");
        }
    }
    metadata = body.at("metadata");
}

void reject_unsupported_platform_fields(const Json& body) {
    const struct {
        const char* field;
        const char* code;
        const char* reason;
    } unsupported[] = {
        {"conversation", "conversations_not_supported",
         "conversation requires an OpenAI Conversations resource"},
        {"prompt", "prompt_templates_not_supported",
         "prompt requires an OpenAI prompt-template resource"},
        {"context_management", "context_management_not_supported",
         "context_management requires an API compaction pipeline"},
        {"moderation", "moderation_not_supported",
         "moderation changes request acceptance and output but no moderator is configured"},
    };

    for (const auto& entry : unsupported) {
        if (body.contains(entry.field) && !body.at(entry.field).is_null()) {
            bad_request(entry.reason, entry.field, entry.code);
        }
    }
}

void validate_common_top_level(const Json& body, bool create) {
    static const std::unordered_set<std::string> create_fields = {"background",
                                                                  "capture",
                                                                  "steering",
                                                                  "structured_outputs",
                                                                  "chat_template_kwargs",
                                                                  "client_metadata",
                                                                  "context_management",
                                                                  "conversation",
                                                                  "include",
                                                                  "input",
                                                                  "instructions",
                                                                  "max_output_tokens",
                                                                  "max_tool_calls",
                                                                  "metadata",
                                                                  "model",
                                                                  "moderation",
                                                                  "parallel_tool_calls",
                                                                  "previous_response_id",
                                                                  "preserve_thinking",
                                                                  "prompt",
                                                                  "prompt_cache_key",
                                                                  "prompt_cache_options",
                                                                  "prompt_cache_retention",
                                                                  "reasoning",
                                                                  "safety_identifier",
                                                                  "service_tier",
                                                                  "store",
                                                                  "stream",
                                                                  "stream_options",
                                                                  "temperature",
                                                                  "text",
                                                                  "tool_choice",
                                                                  "tools",
                                                                  "top_logprobs",
                                                                  "top_p",
                                                                  "truncation",
                                                                  "user"};
    static const std::unordered_set<std::string> count_fields  = {"chat_template_kwargs",
                                                                  "conversation",
                                                                  "input",
                                                                  "instructions",
                                                                  "model",
                                                                  "parallel_tool_calls",
                                                                  "personality",
                                                                  "previous_response_id",
                                                                  "preserve_thinking",
                                                                  "reasoning",
                                                                  "text",
                                                                  "tool_choice",
                                                                  "tools",
                                                                  "truncation"};
    const auto& allowed = create ? create_fields : count_fields;
    for (auto iterator = body.begin(); iterator != body.end(); ++iterator) {
        if (!allowed.contains(iterator.key())) {
            bad_request("unknown parameter: " + iterator.key(), iterator.key(),
                        "unknown_parameter");
        }
    }
}

} // namespace

OpenAIResponsesCreateRequest parse_openai_responses_create_request(const Json& body,
                                                                   const RequestLimits& limits) {
    require_object(body);
    validate_common_top_level(body, true);
    reject_unsupported_platform_fields(body);
    const OpenAIPromptCachePolicy cache_policy = parse_openai_prompt_cache_policy(body);

    ParsedPromptFields parsed  = parse_prompt_fields(body, limits);
    parsed.prompt.cache_policy = cache_policy;
    OpenAIResponsesCreateRequest out;
    out.prompt = std::move(parsed.prompt);
    if (body.contains("text") && body["text"].is_object() && body["text"].contains("format") &&
        !body["text"]["format"].is_null())
        out.text_format = body["text"]["format"];
    out.tools               = std::move(parsed.wire_tools);
    out.tool_choice         = std::move(parsed.wire_tool_choice);
    out.tool_identities     = std::move(parsed.tool_identities);
    out.parallel_tool_calls = parsed.parallel_tool_calls;
    out.store               = optional_bool(body, "store", true);
    out.stream              = optional_bool(body, "stream", false);
    validate_metadata(body, out.metadata);

    // Codex attaches per-request tracing information here. It is an opaque client hint and has no
    // Engine, cache-identity, response-store, or response-body semantics.
    if (body.contains("client_metadata") && !body.at("client_metadata").is_null() &&
        !body.at("client_metadata").is_object()) {
        bad_request("client_metadata must be an object or null", "client_metadata", "invalid_type");
    }

    if (body.contains("background") && !body.at("background").is_null()) {
        if (!body.at("background").is_boolean()) {
            bad_request("background must be a boolean", "background");
        }
        if (body.at("background").get<bool>()) {
            bad_request("background execution is not supported", "background",
                        "background_not_supported");
        }
    }
    if (body.contains("include") && !body.at("include").is_null()) {
        if (!body.at("include").is_array()) { bad_request("include must be an array", "include"); }
        if (!body.at("include").empty()) {
            bad_request("the requested additional response fields have no available response "
                        "representation",
                        "include", "include_not_supported");
        }
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        const Json& options = body.at("stream_options");
        if (!options.is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        static const std::unordered_set<std::string> allowed = {"include_obfuscation"};
        reject_nonnull_unknown_members(options, allowed, "stream_options");
        if (options.contains("include_obfuscation") &&
            !options.at("include_obfuscation").is_null() &&
            !options.at("include_obfuscation").is_boolean()) {
            bad_request("stream_options.include_obfuscation must be a boolean", "stream_options");
        }
    }
    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        if (!body.at("service_tier").is_string()) {
            bad_request("service_tier must be a string", "service_tier");
        }
        const std::string tier = body.at("service_tier").get<std::string>();
        if (tier != "auto" && tier != "default") {
            bad_request("the requested service tier is not provided by this local server",
                        "service_tier", "service_tier_not_supported");
        }
    }
    if (const std::optional<int> top_logprobs = optional_int(body, "top_logprobs")) {
        if (*top_logprobs < 0 || *top_logprobs > 20) {
            bad_request("top_logprobs must be in [0,20]", "top_logprobs");
        }
        if (*top_logprobs != 0) {
            bad_request("the Engine does not return token log probabilities", "top_logprobs",
                        "logprobs_not_supported");
        }
    }
    if (const std::optional<int> max_tool_calls = optional_int(body, "max_tool_calls")) {
        if (*max_tool_calls < 0) {
            bad_request("max_tool_calls must be non-negative", "max_tool_calls");
        }
        out.max_tool_calls = *max_tool_calls;
    }

    if (const std::optional<double> temperature = optional_number(body, "temperature")) {
        if (*temperature < 0.0 || *temperature > 2.0) {
            bad_request("temperature must be in [0,2]", "temperature");
        }
        out.prompt.generation.sampling.temperature = *temperature;
    }
    if (const std::optional<double> top_p = optional_number(body, "top_p")) {
        if (*top_p < 0.0 || *top_p > 1.0) { bad_request("top_p must be in [0,1]", "top_p"); }
        out.prompt.generation.sampling.top_p = *top_p;
    }
    if (const std::optional<int> max_output = optional_int(body, "max_output_tokens")) {
        if (*max_output < 0) {
            bad_request("max_output_tokens must be non-negative", "max_output_tokens");
        }
        out.requested_max_output_tokens  = *max_output;
        out.prompt.generation.max_tokens = *max_output;
    }
    parse_structured_outputs(body, out.prompt.generation);
    parse_activation_fields(body, out.prompt.generation);
    return out;
}

OpenAIResponsesPromptRequest
parse_openai_responses_input_tokens_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    validate_common_top_level(body, false);
    reject_unsupported_platform_fields(body);
    if (body.contains("personality") && !body.at("personality").is_null()) {
        bad_request("personality changes prompt construction and is not supported", "personality",
                    "personality_not_supported");
    }
    auto parsed                          = parse_prompt_fields(body, limits);
    parsed.prompt.cache_policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
    return std::move(parsed.prompt);
}

} // namespace ninfer::serve
