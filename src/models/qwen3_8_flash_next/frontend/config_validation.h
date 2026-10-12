#pragma once

// Checks the checkpoint's frontend resources against what this build was compiled for.

#include "models/qwen3_8_flash_next/frontend/chat_template.h"
#include "models/qwen3_8_flash_next/frontend/processor.h"
#include "models/qwen3_8_flash_next/frontend/tokenizer.h"
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>



namespace ninfer::models::qwen3_8_flash_next::config_validation {

frontend_internal::ProcessorOptions processor_options(const FrontendResources& resources);
void validate_registered_processor(const frontend_internal::ProcessorOptions& options);
frontend_internal::CompiledChatTemplate compile_chat_template(const FrontendResources& resources);
void validate_registered_tokenizer(const frontend_internal::Tokenizer& tokenizer);

} // namespace ninfer::models::qwen3_8_flash_next::config_validation
