#pragma once

#include "models/gemma4/frontend/output_session.h"
#include "models/gemma4/frontend/prompt.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::gemma4 {

class Frontend {
public:
    Frontend(std::string tokenizer_json, std::string chat_template);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();
    Frontend(const Frontend&) = delete;
    Frontend& operator=(const Frontend&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input, const PreparationControl& control) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    // The chat template output for input, before tokenization.
    [[nodiscard]] std::string render(const PromptInput& input) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const noexcept { return {}; }
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const noexcept;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output,
                                                    const ThinkingControlOptions& thinking,
                                                    const std::optional<OutputConstraint>& constraint,
                                                    const ToolChoice& tool_choice) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::gemma4
