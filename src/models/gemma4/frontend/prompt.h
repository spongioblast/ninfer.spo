#pragma once

#include "models/gemma4/frontend/image.h"
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
    [[nodiscard]] const std::vector<Gemma4PreparedImage>& images() const noexcept { return images_; }
    [[nodiscard]] const std::vector<std::int32_t>& image_scatter() const noexcept {
        return image_scatter_;
    }

private:
    PreparedPrompt(std::vector<TokenId> tokens, PromptSummary summary, PromptPreparationStats stats,
                   std::vector<Gemma4PreparedImage> images = {},
                   std::vector<std::int32_t> image_scatter = {})
        : tokens_(std::move(tokens)), summary_(summary), stats_(stats), images_(std::move(images)),
          image_scatter_(std::move(image_scatter)) {}

    std::vector<TokenId> tokens_;
    PromptSummary summary_;
    PromptPreparationStats stats_;
    std::vector<Gemma4PreparedImage> images_;
    std::vector<std::int32_t> image_scatter_;

    friend class Frontend;
    friend class Program;
};

} // namespace ninfer::models::gemma4
