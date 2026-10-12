#include "artifact/reader.h"
#include "models/qwen3_8_flash_next/frontend/test_access.h"
#include "models/qwen3_8_flash_next/frontend/tokenizer.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace flash = ninfer::models::qwen3_8_flash_next;
namespace fi    = ninfer::models::qwen3_8_flash_next::frontend_internal;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool same_results(const fi::BoundaryEncodedText& lhs, const fi::BoundaryEncodedText& rhs) {
    return lhs.input_ids == rhs.input_ids && lhs.boundaries.size() == rhs.boundaries.size() &&
           std::equal(lhs.boundaries.begin(), lhs.boundaries.end(), rhs.boundaries.begin(),
                      [](const fi::TokenBoundaryResult& a, const fi::TokenBoundaryResult& b) {
                          return a.exact_frontier == b.exact_frontier &&
                                 a.stable_frontier == b.stable_frontier;
                      });
}

// The encode cache must answer an extension and an exact repeat with what a tokenizer that never
// saw the earlier text produces, and must actually fire.
void check_encode_cache(const fi::TokenizerResources& resources) {
    const fi::Tokenizer warm(resources);
    const fi::Tokenizer cold(resources);
    const std::string first  = "<|im_start|>user\nthe resolver reads each pending entry<|im_end|>\n";
    const std::string second = "<|im_start|>assistant\n<think>\n\n</think>\n\nnoted<|im_end|>\n";
    const std::string short_text = first + second;
    const std::string long_text  = short_text + "<|im_start|>user\nthen the arena hands it back";
    const std::vector<std::size_t> short_boundaries{first.size(), short_text.size()};
    const std::vector<std::size_t> long_boundaries{first.size(), short_text.size(),
                                                   long_text.size()};

    const fi::BoundaryEncodedText before =
        warm.encode_with_boundaries(short_text, short_boundaries);
    require(warm.encode_cache_splices() == 0, "a first encode reported a splice");
    const fi::BoundaryEncodedText spliced = warm.encode_with_boundaries(long_text, long_boundaries);
    require(warm.encode_cache_splices() == 1, "the extending encode did not splice");
    require(warm.encode_cache_fallbacks() == 0, "the splice threw and fell back");
    require(same_results(spliced, cold.encode_with_boundaries(long_text, long_boundaries)),
            "an incremental encode did not reproduce the full encode");
    require(std::equal(before.input_ids.begin(), before.input_ids.end(), spliced.input_ids.begin()),
            "the extended prompt did not extend the shorter one's ids");

    const fi::BoundaryEncodedText repeated = warm.encode_with_boundaries(long_text, long_boundaries);
    require(warm.encode_cache_splices() == 2, "a repeated encode did not answer from the cache");
    require(same_results(repeated, spliced), "a repeated encode changed the result");
    const std::vector<std::size_t> narrowed{first.size()};
    require(same_results(warm.encode_with_boundaries(long_text, narrowed),
                         cold.encode_with_boundaries(long_text, narrowed)),
            "a boundary set the cache never answered changed the result");

    // A seam inside a BPE stretch must be refused: "pending" must not split into "pend" + "ing".
    const std::string mid       = "<|im_start|>user\nthe pend";
    const std::string mid_long  = mid + "ing entry";
    const fi::Tokenizer warm_mid(resources);
    (void)warm_mid.encode_with_boundaries(mid, std::vector<std::size_t>{mid.size()});
    const fi::BoundaryEncodedText mid_result =
        warm_mid.encode_with_boundaries(mid_long, std::vector<std::size_t>{mid_long.size()});
    require(mid_result.input_ids == cold.encode(mid_long),
            "a seam inside a BPE stretch changed the ids");
    require(warm_mid.encode_cache_splices() == 0, "a seam inside a BPE stretch was spliced");

    // The bytes below the splice window are copied from the cache. A different literal span there
    // changes those ids, so the splice must decline and the result must match a full encode.
    const std::string marked = "<|im_start|>";
    const std::string middle = "user\nhello ";
    const std::string grown  = marked + middle + "world and more";
    const std::string base   = marked + middle;
    const fi::ByteSpan marked_span{.begin = 0, .end = marked.size()};
    const std::span<const fi::ByteSpan> marked_spans(&marked_span, 1);
    const std::vector<std::size_t> base_bounds{marked.size(), base.size()};
    const std::vector<std::size_t> grown_bounds{marked.size(), base.size(), grown.size()};

    const fi::Tokenizer different(resources);
    (void)different.encode_with_boundaries(base, base_bounds, {}, marked_spans);
    const std::size_t splices_before = different.encode_cache_splices();
    const fi::BoundaryEncodedText without_span = different.encode_with_boundaries(grown, grown_bounds);
    require(same_results(without_span, cold.encode_with_boundaries(grown, grown_bounds)),
            "a different literal span below the window changed the encoded ids");
    require(different.encode_cache_splices() == splices_before,
            "a different literal span below the window was spliced");

    const fi::Tokenizer same_warm(resources);
    const fi::Tokenizer same_cold(resources);
    (void)same_warm.encode_with_boundaries(base, base_bounds, {}, marked_spans);
    const fi::BoundaryEncodedText with_span =
        same_warm.encode_with_boundaries(grown, grown_bounds, {}, marked_spans);
    require(same_warm.encode_cache_splices() == 1,
            "matching literal spans below the window did not splice");
    require(same_results(with_span, same_cold.encode_with_boundaries(grown, grown_bounds, {},
                                                                     marked_spans)),
            "a splice with matching literal spans changed the ids");
}

ninfer::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    return message;
}

// A Pi-shaped conversation that grows by a turn per request must prepare to the same ids and
// frontiers whether or not the frontend has seen the shorter conversation.
void check_growing_conversation(const flash::FrontendResources& resources) {
    auto warm = flash::make_frontend(resources, {.vision_enabled = false});
    const std::string tool =
        R"({"type":"function","function":{"name":"read","description":"Read a file",)"
        R"("parameters":{"type":"object","properties":{"path":{"type":"string"}},)"
        R"("required":["path"]}}})";
    std::vector<ninfer::ChatMessage> history{
        text_message(ninfer::ChatRole::System, "You are a coding agent."),
        text_message(ninfer::ChatRole::User, "Summarise how decode waits work."),
    };
    for (int turn = 0; turn < 4; ++turn) {
        ninfer::ChatMessage call = text_message(ninfer::ChatRole::Assistant, "Reading a file.");
        const std::string id     = "call_" + std::to_string(turn);
        call.tool_calls.push_back(
            {.id = id, .name = "read", .arguments_json = R"({"path":"src/core/device.cu"})"});
        history.push_back(std::move(call));
        ninfer::ChatMessage result = text_message(
            ninfer::ChatRole::Tool, "int wait() { return spin ? 1 : 0; } // turn " +
                                        std::to_string(turn) + " \xC3\xA9t\xC3\xA9");
        result.tool_call_id = id;
        history.push_back(std::move(result));
    }
    const auto prepare = [&](const flash::Frontend& frontend,
                             const std::vector<ninfer::ChatMessage>& messages) {
        ninfer::PromptInput input;
        input.messages                  = messages;
        input.options.enable_thinking   = false;
        input.options.preserve_thinking = true;
        input.options.tool_jsons        = {tool};
        return frontend.prepare(std::move(input));
    };
    for (int request = 0; request < 4; ++request) {
        history.push_back(
            text_message(ninfer::ChatRole::User, "Reply with the number " +
                                                     std::to_string(request) + " only."));
        const auto warm_prompt = prepare(warm, history);
        auto cold              = flash::make_frontend(resources, {.vision_enabled = false});
        const auto cold_prompt = prepare(cold, history);
        const auto& a          = flash::FrontendTestAccess::inspect(warm_prompt);
        const auto& b          = flash::FrontendTestAccess::inspect(cold_prompt);
        require(a.token_ids == b.token_ids, "a growing conversation changed its token ids");
        require(a.identity.rewrite_execution_frontiers == b.identity.rewrite_execution_frontiers,
                "a growing conversation changed its rewrite execution frontiers");
        require(a.identity.rewrite_checkpoint.has_value() ==
                        b.identity.rewrite_checkpoint.has_value() &&
                    (!a.identity.rewrite_checkpoint ||
                     a.identity.rewrite_checkpoint->frontier ==
                         b.identity.rewrite_checkpoint->frontier),
                "a growing conversation changed its rewrite checkpoint");
        const auto repeat = prepare(warm, history);
        require(flash::FrontendTestAccess::inspect(repeat).token_ids == b.token_ids,
                "a repeated conversation changed its token ids");
        history.push_back(text_message(ninfer::ChatRole::Assistant, std::to_string(request)));
    }
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (!path || !*path) { return 77; }
    try {
        ninfer::artifact::Reader reader(path);
        const auto& resources = reader.directory().component("text").resources;
        auto read             = [&](const char* role) {
            const auto bytes = reader.read_object(resources.at(role));
            return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        namespace flash = ninfer::models::qwen3_8_flash_next;
        const flash::FrontendResources frontend_resources{
            .tokenizer_json         = read("tokenizer.json"),
            .tokenizer_config_json  = read("tokenizer_config.json"),
            .chat_template_jinja    = read("chat_template.jinja"),
            .generation_config_json = read("generation_config.json"),
        };
        auto frontend = flash::make_frontend(frontend_resources, {.vision_enabled = false});
        const std::array<std::optional<bool>, 3> settings{std::nullopt, false, true};
        for (auto thinking : settings) {
            ninfer::PromptInput input;
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(
                {.kind = ninfer::MessagePartKind::Text, .text = "Reply READY."});
            input.messages.push_back(std::move(message));
            input.options.enable_thinking = thinking;
            const auto prompt             = frontend.prepare(std::move(input));
            const bool expected_reasoning = thinking.value_or(true);
            if (prompt.summary().starts_in_reasoning != expected_reasoning) {
                throw std::runtime_error("thinking option selected the wrong output channel");
            }
            auto output       = frontend.make_output_session(prompt, {});
            const auto tokens = frontend.tokenize_text("READY 1");
            (void)output.preview_model(tokens, tokens.size(), ninfer::FinishReason::OutputLimit);
            const auto deltas = output.commit_preview();
            std::string text;
            for (const auto& delta : deltas) {
                const auto expected = expected_reasoning ? ninfer::OutputChannel::Reasoning
                                                         : ninfer::OutputChannel::Content;
                if (!delta.text.empty() && delta.channel != expected) {
                    throw std::runtime_error("output was published in the wrong channel");
                }
                text += delta.text;
            }
            if (text != "READY 1") { throw std::runtime_error("output text was not preserved"); }
        }
        std::cout << "OK Flash-Next default and explicit thinking output channels\n";
        check_encode_cache({.tokenizer_json         = frontend_resources.tokenizer_json,
                            .tokenizer_config_json  = frontend_resources.tokenizer_config_json,
                            .generation_config_json = frontend_resources.generation_config_json});
        std::cout << "OK Flash-Next encode cache splice, repeat and refusal\n";
        check_growing_conversation(frontend_resources);
        std::cout << "OK Flash-Next growing conversation matches a cold frontend\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
