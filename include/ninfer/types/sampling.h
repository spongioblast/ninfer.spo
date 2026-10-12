#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ninfer/types/engine.h"

namespace ninfer {

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    bool capture                          = false;
    std::optional<double> steering_strength;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

enum class OutputConstraintKind : std::uint8_t { Grammar, JsonObject, JsonSchema, Choice, Regex };

// Constrains generated content; Chat reasoning retains the model's framing. Source is owning
// GBNF, JSON Schema or regex text. Choice owns literal alternatives; JsonObject has no payload.
struct OutputConstraint {
    OutputConstraintKind kind = OutputConstraintKind::Grammar;
    std::string source;
    std::vector<std::string> choices;

    [[nodiscard]] static OutputConstraint grammar(std::string source) {
        return {OutputConstraintKind::Grammar, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint json_object() {
        return {OutputConstraintKind::JsonObject, {}, {}};
    }

    [[nodiscard]] static OutputConstraint json_schema(std::string source) {
        return {OutputConstraintKind::JsonSchema, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint choice(std::vector<std::string> values) {
        return {OutputConstraintKind::Choice, {}, std::move(values)};
    }

    [[nodiscard]] static OutputConstraint regex(std::string pattern) {
        return {OutputConstraintKind::Regex, std::move(pattern), {}};
    }

    bool operator==(const OutputConstraint&) const = default;
};

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
// Basic protects every tool call. Automatic applies constraints only for strict tools or an
// explicit selection/count policy.
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };

// Declarations belong to PromptOptions. Selection and cardinality affect this generation only.
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel                  = true;
    ToolConstraintMode constraints = ToolConstraintMode::Basic;
};

} // namespace ninfer
