// Gemma 4 tokenizer and chat template against Hugging Face fixtures.
// The fixture directory (tests/models/gemma4/fixtures, or NINFER_GEMMA4_FIXTURES) holds
// tok_fixtures.json (text, ids, decoded) and chat_fixtures20.json (messages, tools,
// enable_thinking, continue_final, text, ids).
// NINFER_GEMMA4_HF names the Hugging Face model directory with tokenizer.json and
// chat_template.jinja.

#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "ninfer/types/error.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

ninfer::ChatRole role_of(const std::string& name) {
    if (name == "system") { return ninfer::ChatRole::System; }
    if (name == "developer") { return ninfer::ChatRole::Developer; }
    if (name == "user") { return ninfer::ChatRole::User; }
    if (name == "assistant") { return ninfer::ChatRole::Assistant; }
    if (name == "tool") { return ninfer::ChatRole::Tool; }
    throw std::runtime_error("unknown role " + name);
}

ninfer::PromptInput prompt_of(const Json& item) {
    ninfer::PromptInput input;
    for (const Json& source : item.at("messages")) {
        ninfer::ChatMessage message;
        message.role = role_of(source.at("role").get<std::string>());
        ninfer::MessagePart part;
        part.text = source.value("content", std::string());
        message.parts.push_back(std::move(part));
        message.reasoning_content = source.value("reasoning_content", std::string());
        message.tool_call_id = source.value("tool_call_id", std::string());
        if (source.contains("tool_calls")) {
            for (const Json& call : source.at("tool_calls")) {
                ninfer::ToolCall tool_call;
                tool_call.id = call.value("id", std::string());
                tool_call.name = call.at("function").at("name").get<std::string>();
                tool_call.arguments_json = call.at("function").at("arguments").dump();
                message.tool_calls.push_back(std::move(tool_call));
            }
        }
        input.messages.push_back(std::move(message));
    }
    if (item.contains("tools")) {
        for (const Json& tool : item.at("tools")) { input.options.tool_jsons.push_back(tool.dump()); }
    }
    input.options.enable_thinking = item.value("enable_thinking", false);
    if (item.value("continue_final", false)) {
        input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    }
    return input;
}

struct Run {
    std::string content;
    ninfer::FinishReason finish = ninfer::FinishReason::None;
    std::uint32_t accepted      = 0;
};

// Feeds `tokens` in previews of `step` tokens and collects the published content.
Run feed(ninfer::models::gemma4::OutputSession& session, const std::vector<ninfer::TokenId>& tokens,
         std::size_t step) {
    Run run;
    for (std::size_t at = 0; at < tokens.size() && run.finish == ninfer::FinishReason::None;
         at += step) {
        const std::size_t count = std::min(step, tokens.size() - at);
        const auto decision     = session.preview_model(
            std::span<const ninfer::TokenId>(tokens).subspan(at, count), 1000,
            ninfer::FinishReason::OutputLimit);
        run.finish = decision.finish_reason;
        run.accepted += decision.accepted_tokens;
        for (const auto& delta : session.commit_preview()) {
            if (delta.channel == ninfer::OutputChannel::Content) { run.content += delta.text; }
        }
    }
    if (run.finish == ninfer::FinishReason::None) {
        (void)session.preview_terminal(ninfer::FinishReason::OutputLimit);
        for (const auto& delta : session.commit_preview()) {
            if (delta.channel == ninfer::OutputChannel::Content) { run.content += delta.text; }
        }
    }
    return run;
}

int output_checks(const ninfer::models::gemma4::Frontend& frontend) {
    using ninfer::models::gemma4::Frontend;
    int failures    = 0;
    const auto fail = [&](bool ok, const std::string& label) {
        if (!ok) {
            std::cerr << "FAIL " << label << '\n';
            ++failures;
        }
    };
    const auto tools_input = [] {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "What time is it?",
                              .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        for (const char* name : {"get_weather", "get_time"}) {
            input.options.tool_jsons.push_back(
                Json{{"type", "function"},
                     {"function",
                      {{"name", name},
                       {"parameters", {{"type", "object"}, {"properties", Json::object()}}}}}}
                    .dump());
        }
        return input;
    };
    const auto ids = [&](std::string_view text) { return frontend.tokenize_text(text); };
    const auto with = [](std::vector<ninfer::TokenId> tokens,
                         std::initializer_list<ninfer::TokenId> tail) {
        tokens.insert(tokens.end(), tail);
        return tokens;
    };
    constexpr ninfer::TokenId kOpen = 48, kClose = 49, kTurnEnd = 106;

    {
        auto prompt          = frontend.prepare(tools_input(), {});
        const auto plain     = prompt.tokens();
        ninfer::ToolChoice choice;
        choice.mode          = ninfer::ToolChoiceMode::Required;
        choice.allowed_names = std::vector<std::string>{"get_time"};
        Frontend::apply_tool_choice(prompt, choice);
        auto expect = plain;
        expect.push_back(kOpen);
        for (const auto id : ids("call:get_time")) { expect.push_back(id); }
        fail(prompt.tokens() == expect && prompt.summary().prompt_tokens == expect.size(),
             "named tool choice appends <|tool_call>call:get_time");
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, choice);
        const Run run =
            feed(session, with(ids("{zone:<|\"|>Berlin<|\"|>}"), {kClose, kTurnEnd}), 3);
        const auto calls = session.take_tool_calls();
        fail(run.finish == ninfer::FinishReason::StopToken && run.content.empty() &&
                 calls.size() == 1 && calls[0].name == "get_time" &&
                 calls[0].arguments_json == "{\"zone\":\"Berlin\"}",
             "named tool choice parses the forced call");
    }
    {
        auto prompt = frontend.prepare(tools_input(), {});
        ninfer::ToolChoice choice;
        choice.mode = ninfer::ToolChoiceMode::Required;
        Frontend::apply_tool_choice(prompt, choice);
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, choice);
        (void)feed(session, with(ids("get_weather{city:<|\"|>Paris<|\"|>}"), {kClose, kTurnEnd}),
                   1);
        const auto calls = session.take_tool_calls();
        fail(calls.size() == 1 && calls[0].name == "get_weather" &&
                 calls[0].arguments_json == "{\"city\":\"Paris\"}",
             "required tool choice opens the call without a name");
    }
    {
        auto prompt = frontend.prepare(tools_input(), {});
        ninfer::ToolChoice choice;
        choice.mode          = ninfer::ToolChoiceMode::Required;
        choice.allowed_names = std::vector<std::string>{"get_weather", "get_time"};
        Frontend::apply_tool_choice(prompt, choice);
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, choice);
        (void)feed(session, with(ids("send_mail{to:1}"), {kClose, kTurnEnd}), 4);
        fail(session.take_tool_calls().empty() &&
                 session.tool_call_parse_diagnostics().fallback_reason ==
                     ninfer::ToolCallParseFallbackReason::UndeclaredTool,
             "a call outside allowed_names is rejected");
        ninfer::ToolChoice missing;
        missing.mode          = ninfer::ToolChoiceMode::Required;
        missing.allowed_names = std::vector<std::string>{"send_mail"};
        auto other            = frontend.prepare(tools_input(), {});
        bool threw            = false;
        try {
            Frontend::apply_tool_choice(other, missing);
        } catch (const std::invalid_argument&) { threw = true; }
        fail(threw, "a named tool choice outside the declared tools throws");
    }
    for (const std::size_t step : {std::size_t{1}, std::size_t{8}}) {
        auto prompt = frontend.prepare(tools_input(), {});
        ninfer::StopPolicy stop;
        stop.strings.push_back({.text = "lo wor", .channel = ninfer::OutputChannel::Content,
                                .include_in_output = false});
        auto session = frontend.make_output_session(prompt, stop, {}, {}, std::nullopt, {});
        const auto tokens = ids("Hello world, again");
        const Run run     = feed(session, tokens, step);
        fail(tokens.size() >= 3 && run.finish == ninfer::FinishReason::StopString &&
                 run.content == "Hel" && run.accepted == 2 &&
                 session.matched_stop_string() == std::optional<std::string>("lo wor"),
             "a stop string split across tokens cuts the content (step " +
                 std::to_string(step) + ", got \"" + run.content + "\")");
    }
    std::cout << "output session: " << (failures ? "FAIL" : "ok") << '\n';
    return failures;
}

std::string first_difference(const std::string& got, const std::string& want) {
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
    const std::size_t from = at > 40 ? at - 40 : 0;
    return "at byte " + std::to_string(at) + "\n got:  " + got.substr(from, 120) +
           "\n want: " + want.substr(from, 120);
}

} // namespace

int main() {
    const char* override_dir = std::getenv("NINFER_GEMMA4_FIXTURES");
    const char* fixtures = override_dir ? override_dir : NINFER_GEMMA4_FIXTURE_DIR;
    const char* hf = std::getenv("NINFER_GEMMA4_HF");
    if (hf == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA4_HF is not set\n";
        return 77;
    }
    const std::filesystem::path model = hf;
    try {
        const std::string tokenizer_json = read_file(model / "tokenizer.json");
        const ninfer::models::gemma4::Tokenizer tokenizer(tokenizer_json);
        const ninfer::models::gemma4::Frontend frontend(
            tokenizer_json, read_file(model / "chat_template.jinja"), 262144);

        int token_failures = 0;
        const Json strings = Json::parse(read_file(std::filesystem::path(fixtures) / "tok_fixtures.json"));
        for (std::size_t index = 0; index < strings.size(); ++index) {
            const Json& record = strings[index];
            const auto text = record.at("text").get<std::string>();
            const auto want = record.at("ids").get<std::vector<int>>();
            const auto got = tokenizer.encode(text);
            const auto decoded = tokenizer.decode(want, false);
            const bool ids_ok = got == want;
            const bool text_ok = decoded == record.at("decoded").get<std::string>();
            if (!ids_ok || !text_ok) {
                if (++token_failures <= 5) {
                    std::cerr << "FAIL string " << index << (ids_ok ? "" : " ids")
                              << (text_ok ? "" : " decode") << ": " << Json(text).dump() << '\n';
                }
            }
        }
        std::cout << "tokenizer: " << strings.size() - token_failures << '/' << strings.size()
                  << " strings match\n";

        int chat_failures = 0;
        const Json chats =
            Json::parse(read_file(std::filesystem::path(fixtures) / "chat_fixtures20.json"));
        for (const Json& item : chats) {
            const std::string name = item.at("name").get<std::string>();
            const std::string want = item.at("text").get<std::string>();
            std::string got;
            try {
                got = frontend.render(prompt_of(item));
            } catch (const std::exception& error) {
                ++chat_failures;
                std::cerr << "FAIL " << name << " threw: " << error.what() << '\n';
                continue;
            }
            if (got != want) {
                ++chat_failures;
                std::cerr << "FAIL " << name << " text " << first_difference(got, want) << '\n';
                continue;
            }
            const auto prepared = frontend.prepare(prompt_of(item), {});
            const auto want_ids = item.at("ids").get<std::vector<ninfer::TokenId>>();
            if (prepared.tokens() != want_ids) {
                ++chat_failures;
                std::cerr << "FAIL " << name << " ids\n";
            }
        }
        std::cout << "chat template: " << chats.size() - chat_failures << '/' << chats.size()
                  << " conversations match\n";
        const int output_failures = output_checks(frontend);

        // A prompt past max_context is a request error, which serve answers with HTTP 400.
        const ninfer::models::gemma4::Frontend tight(
            tokenizer_json, read_file(model / "chat_template.jinja"), 16);
        ninfer::PromptInput long_input;
        ninfer::ChatMessage long_user;
        long_user.role = ninfer::ChatRole::User;
        std::string words;
        for (int index = 0; index < 40; ++index) { words += "word "; }
        long_user.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = words, .media = {}});
        long_input.messages.push_back(std::move(long_user));
        bool length_ok = false;
        try {
            (void)tight.prepare(std::move(long_input), {});
        } catch (const ninfer::RequestError& error) {
            length_ok = error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
        }
        std::cout << "context length: " << (length_ok ? "rejected" : "FAIL not rejected") << '\n';

        const bool ok =
            token_failures == 0 && chat_failures == 0 && output_failures == 0 && length_ok;
        std::cout << (ok ? "OK" : "FAIL") << " gemma4 frontend fixtures\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
