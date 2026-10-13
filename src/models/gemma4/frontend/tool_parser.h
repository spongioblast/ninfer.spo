#pragma once

#include "ninfer/types.h"

#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::gemma4 {

// Parse one tool-call body, the text between `<|tool_call>` and `<tool_call|>`:
// `call:NAME{key:value,...}`. Keys are bare words, strings are wrapped in `<|"|>`, and values
// may nest objects and arrays. Returns nullopt when the body is malformed.
[[nodiscard]] std::optional<GeneratedToolCall> parse_tool_call_body(std::string_view body);

// Parse every `<|tool_call>...<tool_call|>` block in text. Throws on a malformed block.
[[nodiscard]] std::vector<GeneratedToolCall> parse_tool_calls(std::string_view text);

} // namespace ninfer::models::gemma4
