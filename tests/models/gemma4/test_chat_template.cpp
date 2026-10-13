#include "text/jinja.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool check(const char* name, const std::string& got, const std::string& want) {
    if (got == want) { return true; }
    std::cerr << "FAIL " << name << "\n got: " << got << "\n want: " << want << '\n';
    return false;
}

std::string render(const ninfer::text::JinjaTemplate& tpl, const nlohmann::ordered_json& ctx) {
    return tpl.render(ctx).text;
}

} // namespace

int main() {
    const char* hf_dir = std::getenv("NINFER_GEMMA4_HF");
    std::ifstream input(hf_dir ? std::string(hf_dir) + "/chat_template.jinja" : std::string(),
                        std::ios::binary);
    if (!input) {
        std::cout << "SKIP: NINFER_GEMMA4_HF does not hold chat_template.jinja\n";
        return 77;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    try {
        const ninfer::text::JinjaTemplate tpl(buffer.str(), "gemma4-chat");
        int failures = 0;
        nlohmann::ordered_json user;
        user["messages"] = nlohmann::ordered_json::array(
            {{{"role", "user"}, {"content", "Hello"}}});
        user["bos_token"] = "<bos>";
        user["add_generation_prompt"] = true;
        user["enable_thinking"] = false;
        if (!check("user", render(tpl, user),
                   "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")) {
            ++failures;
        }
        nlohmann::ordered_json system;
        system["messages"] = nlohmann::ordered_json::array(
            {{{"role", "system"}, {"content", "Be brief."}},
             {{"role", "user"}, {"content", "Hi"}}});
        system["bos_token"] = "<bos>";
        system["add_generation_prompt"] = true;
        system["enable_thinking"] = false;
        if (!check("system", render(tpl, system),
                   "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n"
                   "<|turn>model\n<|channel>thought\n<channel|>")) {
            ++failures;
        }
        nlohmann::ordered_json thinking = user;
        thinking["messages"][0]["content"] = "2+2";
        thinking["enable_thinking"] = true;
        if (!check("thinking", render(tpl, thinking),
                   "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n2+2<turn|>\n"
                   "<|turn>model\n")) {
            ++failures;
        }
        nlohmann::ordered_json multi;
        multi["messages"] = nlohmann::ordered_json::array(
            {{{"role", "user"}, {"content", "One"}},
             {{"role", "assistant"}, {"content", "Two"}},
             {{"role", "user"}, {"content", "Three"}}});
        multi["bos_token"] = "<bos>";
        multi["add_generation_prompt"] = false;
        multi["enable_thinking"] = false;
        if (!check("multi", render(tpl, multi),
                   "<bos><|turn>user\nOne<turn|>\n<|turn>model\nTwo<turn|>\n"
                   "<|turn>user\nThree<turn|>\n")) {
            ++failures;
        }
        nlohmann::ordered_json tool;
        tool["messages"] = nlohmann::ordered_json::array(
            {{{"role", "user"}, {"content", "Weather?"}}});
        tool["bos_token"] = "<bos>";
        tool["add_generation_prompt"] = true;
        tool["enable_thinking"] = false;
        tool["tools"] = nlohmann::ordered_json::array({
            {{"type", "function"},
             {"function",
              {{"name", "get_weather"},
               {"description", "Look up weather"},
               {"parameters",
                {{"type", "object"},
                 {"properties",
                  {{"city", {{"type", "string"}, {"description", "City"}}}}},
                 {"required", nlohmann::ordered_json::array({"city"})}}}}}},
        });
        if (!check(
                "tool", render(tpl, tool),
                "<bos><|turn>system\n<|tool>declaration:get_weather{description:<|\"|>Look up "
                "weather<|\"|>,parameters:{properties:{city:{description:<|\"|>City<|\"|>,type:<|"
                "\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
                "<turn|>\n<|turn>user\nWeather?<turn|>\n<|turn>model\n<|channel>thought\n"
                "<channel|>")) {
            ++failures;
        }
        std::cout << (failures ? "FAIL" : "OK") << " gemma4 chat template\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
