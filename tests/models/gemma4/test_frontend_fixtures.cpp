// Gemma 4 tokenizer and chat template against Hugging Face fixtures.
// The fixture directory (tests/models/gemma4/fixtures, or NINFER_GEMMA4_FIXTURES) holds
// tok_fixtures.json (text, ids, decoded) and chat_fixtures20.json (messages, tools,
// enable_thinking, continue_final, text, ids).
// NINFER_GEMMA4_HF names the Hugging Face model directory with tokenizer.json and
// chat_template.jinja.

#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
        const ninfer::models::gemma4::Frontend frontend(tokenizer_json,
                                                        read_file(model / "chat_template.jinja"));

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
        const bool ok = token_failures == 0 && chat_failures == 0;
        std::cout << (ok ? "OK" : "FAIL") << " gemma4 frontend fixtures\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
