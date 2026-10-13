#include "models/gemma4/frontend/tool_parser.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* label, const std::string& detail = {}) {
    if (ok) { return; }
    std::cerr << "FAIL " << label << ' ' << detail << '\n';
    ++failures;
}

} // namespace

int main() {
    using ninfer::models::gemma4::parse_tool_call_body;
    using ninfer::models::gemma4::parse_tool_calls;

    const auto calls = parse_tool_calls(
        "prefix <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|> tail");
    check(calls.size() == 1 && calls[0].name == "get_weather" &&
              calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":2}",
          "flat", calls.empty() ? "" : calls[0].arguments_json);

    const auto empty = parse_tool_calls("<|tool_call>call:ping{}<tool_call|>");
    check(empty.size() == 1 && empty[0].name == "ping" && empty[0].arguments_json == "{}", "empty");

    const auto nested = parse_tool_call_body(
        "call:edit{path:<|\"|>a \"b\"\n{c}<|\"|>,opts:{deep:true,list:[1,-2.5,<|\"|>x<|\"|>,null]}}");
    check(nested && nested->name == "edit" &&
              nested->arguments_json ==
                  "{\"path\":\"a \\\"b\\\"\\n{c}\",\"opts\":{\"deep\":true,\"list\":[1,-2.5,\"x\",null]}}",
          "nested", nested ? nested->arguments_json : "nullopt");

    const auto two = parse_tool_calls("<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{}"
                                      "<tool_call|>");
    check(two.size() == 2 && two[0].name == "a" && two[1].name == "b", "two calls");

    check(!parse_tool_call_body("call:bad{x:<|\"|>open}"), "unterminated string");
    check(!parse_tool_call_body("call:bad{x:1} trailing"), "trailing text");
    check(!parse_tool_call_body("call:dup{x:1,x:2}"), "duplicate key");
    check(!parse_tool_call_body("nocall{}"), "missing call prefix");

    std::cout << (failures ? "FAIL" : "OK") << " gemma4 tool parser\n";
    return failures ? 1 : 0;
}
