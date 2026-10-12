#pragma once

// Shared by the Responses request parsers. Not part of the public serve API.

#include "serve/openai_responses.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::serve {

OpenAIResponsesFunctionIdentity function_identity(const RequestJson& object, const char* param);

std::string lower_function_identity(
    const OpenAIResponsesFunctionIdentity& identity,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities, const char* param);

void add_wire_function_identity(RequestJson& object, const OpenAIResponsesFunctionIdentity& identity);

void reject_nonnull_unknown_members(const RequestJson& object,
                                    const std::unordered_set<std::string>& allowed,
                                    const char* param);

void parse_input(const RequestJson& input, OpenAIResponsesPromptRequest& out,
                 std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities);

} // namespace ninfer::serve
