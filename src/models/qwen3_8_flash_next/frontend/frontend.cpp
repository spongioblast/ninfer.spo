#include "models/qwen3_8_flash_next/frontend/frontend_impl.h"
#include "models/qwen3_8_flash_next/frontend/config_validation.h"
#include "models/qwen3_8_flash_next/frontend/test_access.h"
#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

using config_validation::compile_chat_template;
using config_validation::processor_options;
using config_validation::validate_registered_processor;
using config_validation::validate_registered_tokenizer;
using output_decoder::kThinkClose;
using output_decoder::merge_stop_policy;

constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
static_assert(kThinkingControl.ends_with(fi::kCanonicalReasoningCloseSerialization));

} // namespace

Frontend::Impl::Impl(const FrontendResources& resources, bool registered_checkpoint, FrontendOptions options)
    : chat_template(compile_chat_template(resources)),
      tokenizer(std::make_shared<const fi::Tokenizer>(
          fi::TokenizerResources{.tokenizer_json         = resources.tokenizer_json,
                                 .tokenizer_config_json  = resources.tokenizer_config_json,
                                 .generation_config_json = resources.generation_config_json})),
      processor(options.vision_enabled ? processor_options(resources) : fi::ProcessorOptions{}),
      vision_enabled(options.vision_enabled),
      max_context(options.max_context) {
    if (options.max_context == 0) {
        throw std::invalid_argument("frontend max_context must be nonzero");
    }
    std::vector<std::int32_t> eos_ids;
    eos_ids.reserve(tokenizer->default_stop_token_ids().size());
    for (const int id : tokenizer->default_stop_token_ids()) { eos_ids.push_back(id); }
    std::vector<std::string> vocab(std::max(options.token_domain, tokenizer->token_bytes().size()));
    for (std::size_t index = 0; index < tokenizer->token_bytes().size(); ++index) {
        vocab[index] = tokenizer->token_bytes()[index];
    }
    grammar = std::make_unique<text::GrammarCompiler>(std::move(vocab), std::move(eos_ids),
                                                      options.grammar_cache_bytes);
    const std::uint64_t vision_tokens =
        std::min<std::uint64_t>(options.max_context, kMaximumPromptVisionTokens);
    processor.max_vision_tokens = vision_tokens;
    processor.max_raw_patches   = vision_tokens * kRawPatchesPerVisionToken;
    if (vision_enabled) {
        const std::uint64_t minimum_live =
            processor.max_raw_patches * kPreparedVisionPatchFeatures * sizeof(std::uint16_t);
        if (minimum_live > options.media_live_bytes) {
            throw std::invalid_argument(
                "media live-byte capacity cannot hold the maximum supported Vision prompt");
        }
        media_cache = std::make_shared<fi::MediaPreprocessCache>(
            options.media_cache_bytes, options.media_live_bytes,
            options.media_preprocess_threads, static_cast<std::size_t>(minimum_live));
    }
    if (registered_checkpoint) {
        if (vision_enabled) { validate_registered_processor(processor); }
        validate_registered_tokenizer(*tokenizer);
    }
    for (const int token : tokenizer->default_stop_token_ids()) {
        if (!tokenizer->is_valid_token(token)) {
            throw std::invalid_argument(
                "generation_config.json contains a stop token outside the vocabulary");
        }
        defaults.token_ids.push_back(token);
    }
    std::vector<TokenId> encoded = tokenizer->encode(kThinkingControl);
    if (encoded.empty()) {
        throw std::invalid_argument(
            "Qwen tokenizer cannot encode the canonical thinking control suffix");
    }
    const std::string exact =
        tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = false});
    const std::string presented =
        tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = true});
    if (exact != kThinkingControl || presented.find(kThinkClose) == std::string::npos) {
        throw std::invalid_argument(
            "Qwen tokenizer cannot present the canonical thinking control suffix");
    }
    for (const TokenId token : encoded) {
        if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
            defaults.token_ids.end()) {
            throw std::invalid_argument(
                "canonical thinking control suffix contains a default terminal token");
        }
    }
    thinking_control_tokens = std::make_shared<const std::vector<TokenId>>(std::move(encoded));
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, true, options));
}

Frontend FrontendTestAccess::create_component(const FrontendResources& resources,
                                              bool vision_enabled) {
    FrontendOptions options;
    options.vision_enabled = vision_enabled;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return create_component(resources, options);
}

Frontend FrontendTestAccess::create_component(const FrontendResources& resources,
                                              FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, false, options));
}

MediaCacheSummary Frontend::media_cache_summary() const {
    if (impl_ == nullptr || !impl_->media_cache) { return {}; }
    const fi::MediaCacheStats stats = impl_->media_cache->stats();
    return MediaCacheSummary{
        .capacity_bytes      = stats.capacity_bytes,
        .live_capacity_bytes = stats.live_capacity_bytes,
        .retained_bytes      = stats.retained_bytes,
        .live_bytes          = stats.live_bytes,
        .entries             = stats.entries,
        .inflight            = stats.inflight,
        .queued_tasks        = stats.queued_tasks,
        .active_tasks        = stats.active_tasks,
        .preprocess_threads  = stats.preprocess_threads,
        .hits                = stats.hits,
        .misses              = stats.misses,
        .singleflight_waits  = stats.singleflight_waits,
        .evictions           = stats.evictions,
        .oversize_bypasses   = stats.oversize_bypasses,
    };
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    (void)tool_choice;
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    std::unique_ptr<text::GrammarSession> session;
    if (constraint.has_value()) {
        if (impl_->grammar == nullptr) {
            throw std::logic_error("grammar compiler is unavailable");
        }
        if (prompt.data_->tool_call_output && !prompt.data_->tool_call_output->tools.empty()) {
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "output constraints cannot be combined with active tools yet",
                               {}, RequestErrorSource::Tools);
        }
        const RequestErrorKind kind = text::constraint_error_kind(constraint->kind);
        if (impl_->defaults.token_ids.empty() || !caller_stop.token_ids.empty() ||
            !caller_stop.strings.empty() || !caller_stop.include_model_defaults ||
            caller_stop.publish_stop_token || output.raw || output.preserve_special_tokens) {
            throw RequestError(kind, "constraints require default EOS, text output, no custom "
                                     "stops, and one output language");
        }
        const std::string_view close = prompt.data_->starts_in_reasoning
                                           ? fi::kCanonicalReasoningCloseSerialization
                                           : std::string_view{};
        try {
            session = impl_->grammar->compile(*constraint, close, {});
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
            throw RequestError(kind, error.what());
        }
    }
    return OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, prompt.data_->starts_in_reasoning, thinking,
        impl_->thinking_control_tokens, prompt.data_->tool_call_output, std::move(session)));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

} // namespace ninfer::models::qwen3_8_flash_next
