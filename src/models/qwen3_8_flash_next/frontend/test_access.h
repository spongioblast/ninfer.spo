#pragma once

#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>
#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

namespace ninfer::models::qwen3_8_flash_next {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   FrontendOptions options);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::models::qwen3_8_flash_next
