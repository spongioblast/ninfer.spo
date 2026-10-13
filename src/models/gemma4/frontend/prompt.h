#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::gemma4 {

struct PreparedContextCache {
    std::optional<std::uint64_t> session_key;
    bool update_session_index = false;
};

class PreparedPrompt {
public:
    PreparedPrompt() noexcept = default;
    PreparedPrompt(PreparedPrompt&&) noexcept = default;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept = default;
    PreparedPrompt(const PreparedPrompt&) = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const noexcept { return summary_; }
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept { return stats_; }
    [[nodiscard]] explicit operator bool() const noexcept { return !tokens_.empty(); }
    [[nodiscard]] const std::vector<TokenId>& tokens() const noexcept { return tokens_; }

private:
    PreparedPrompt(std::vector<TokenId> tokens, PromptSummary summary, PromptPreparationStats stats)
        : tokens_(std::move(tokens)), summary_(summary), stats_(stats) {}

    std::vector<TokenId> tokens_;
    PromptSummary summary_;
    PromptPreparationStats stats_;

    friend class Frontend;
};

} // namespace ninfer::models::gemma4
