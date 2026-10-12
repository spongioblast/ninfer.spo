#pragma once


#include <ninfer/models/qwen3_8_flash_next/frontend.h>

#include <string>

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::models::qwen3_8_flash_next {

struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
};

} // namespace ninfer::models::qwen3_8_flash_next
