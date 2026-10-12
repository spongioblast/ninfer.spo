#pragma once

// Frontend::Impl and OutputSession::Impl, shared by the frontend translation units.

#include "models/qwen3_8_flash_next/frontend/chat_template.h"
#include "models/qwen3_8_flash_next/frontend/media_cache.h"
#include "models/qwen3_8_flash_next/frontend/output_decoder.h"
#include "models/qwen3_8_flash_next/frontend/processor.h"
#include "models/qwen3_8_flash_next/frontend/tokenizer.h"
#include "models/qwen3_8_flash_next/frontend/tool_call_parser.h"
#include "text/grammar.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {

namespace fi = frontend_internal;

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, bool registered_checkpoint, FrontendOptions options);

    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    fi::ProcessorOptions processor;
    std::shared_ptr<fi::MediaPreprocessCache> media_cache;
    StopPolicy defaults;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool vision_enabled       = true;
    std::uint32_t max_context = 0;
    mutable std::unique_ptr<text::GrammarCompiler> grammar;
};

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const fi::ToolCallOutputContract> tool_call_output_,
         std::unique_ptr<text::GrammarSession> grammar_);

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool preserve_special = false;
    bool split_reasoning  = false;
    output_decoder::DecoderState state;
    output_decoder::DecoderState preview_state;
    output_decoder::SemanticThinkingState semantic;
    output_decoder::SemanticThinkingState preview_semantic;
    output_decoder::PrefixExecutionTracker prefix_execution;
    output_decoder::PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    fi::ToolCallOutputDecoder tool_call_output;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    bool preview_ready = false;
    std::unique_ptr<text::GrammarSession> grammar;

    void accept_grammar(std::span<const TokenId> tokens);
};

} // namespace ninfer::models::qwen3_8_flash_next
