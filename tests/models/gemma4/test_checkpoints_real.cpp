// Gemma 4 prompt checkpoints through the public Engine, shaped like a voice session: a long
// system turn with tools, a memory message, and turns interleaved with a summary request, an
// image request and a manual chat. Checks that the next voice turn resumes from its displaced
// state, that a memory change resumes from the system-plus-tools frontier, that a breakpoint
// marker adds a frontier, and that a turn whose checkpoints were evicted still matches a fresh
// prefill. Needs NINFER_GEMMA4_CHECKPOINT_WEIGHTS (an MTP artifact; NINFER_GEMMA4_CHECKPOINT_VISION=1
// also sends an image).

#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& label) {
    std::cout << (ok ? "ok   " : "FAIL ") << label << '\n';
    if (!ok) { ++failures; }
}

void add_message(ninfer::PromptInput& input, ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    ninfer::MessagePart part;
    part.text = std::move(text);
    message.parts.push_back(std::move(part));
    input.messages.push_back(std::move(message));
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool reuse) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.top_k          = 1;
    request.execution.allow_prefix_reuse      = reuse;
    return request;
}

std::string numbered(const std::string& topic, int lines, int salt) {
    std::string text;
    for (int line = 0; line < lines; ++line) {
        text += topic + " note " + std::to_string(salt * 1000 + line) + ": keep answers short, " +
                "friendly and spoken; never read out lists longer than three items.\n";
    }
    return text;
}

ninfer::PromptInput voice(const std::string& memory) {
    ninfer::PromptInput input;
    add_message(input, ninfer::ChatRole::System,
                "You are Juno, a voice assistant. Reply in one or two spoken sentences.\n" +
                    numbered("Style", 90, 1));
    add_message(input, ninfer::ChatRole::System, "Memory: " + memory);
    add_message(input, ninfer::ChatRole::User, "Hi Juno, what should I cook tonight?");
    input.options.enable_thinking = false;
    input.options.tool_jsons      = {
        R"({"type":"function","function":{"name":"get_time","description":"Current local time.","parameters":{"type":"object","properties":{}}}})",
        R"({"type":"function","function":{"name":"set_timer","description":"Start a kitchen timer.","parameters":{"type":"object","properties":{"minutes":{"type":"integer"}},"required":["minutes"]}}})"};
    return input;
}

ninfer::PromptInput side(const std::string& system, const std::string& user) {
    ninfer::PromptInput input;
    add_message(input, ninfer::ChatRole::System, system);
    add_message(input, ninfer::ChatRole::User, user);
    input.options.enable_thinking = false;
    return input;
}

std::vector<std::uint8_t> solid_ppm(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        ppm.push_back(r);
        ppm.push_back(g);
        ppm.push_back(b);
    }
    return ppm;
}

struct Turn {
    ninfer::GenerationResult result;
    double seconds = 0.0;
};

Turn run(ninfer::Engine& engine, ninfer::PromptInput input, const ninfer::RequestOptions& options,
         const char* label) {
    const auto start = std::chrono::steady_clock::now();
    Turn turn{engine.generate(engine.prepare(std::move(input)), options)};
    turn.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << label << ": prompt " << turn.result.prompt.prompt_tokens << ", reused "
              << turn.result.reused_prompt_tokens << ", " << turn.seconds << " s, \""
              << turn.result.content << "\"\n";
    return turn;
}

// Same tokens with and without the reused prefix.
void same_as_fresh(ninfer::Engine& engine, const ninfer::PromptInput& input, const Turn& reused,
                   const std::string& label) {
    const Turn fresh = run(engine, input, greedy(32, false), (label + " fresh").c_str());
    check(fresh.result.reused_prompt_tokens == 0 &&
              fresh.result.generated_token_ids == reused.result.generated_token_ids,
          label + " matches a fresh prefill");
}

void session(const char* artifact, bool vision) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 16384;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::automatic();
    options.enable_vision                    = vision;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 5;
    options.context_cache.device_state_slots = 1;
    ninfer::Engine engine(options);
    std::cout << "capacity " << engine.memory_summary().max_context << ", host "
              << (engine.options().context_cache.host_capacity_bytes.value_or(0) >> 20U)
              << " MiB\n";

    // Prompt tokens after a resume that kept everything before `text` (plus turn markers).
    const auto after = [&](const std::string& text) {
        return static_cast<std::uint32_t>(engine.tokenize_text(text).size()) + 16U;
    };
    const std::string greeting = "Hi Juno, what should I cook tonight?";
    auto first                 = voice("The user is vegetarian and likes spicy food.");
    const Turn v1              = run(engine, first, greedy(32, true), "voice 1");
    check(v1.result.reused_prompt_tokens == 0, "first voice turn is cold");

    const std::string article = numbered("Report", 40, 2);
    const Turn summary =
        run(engine, side("You summarize documents in one sentence.", article), greedy(32, true),
            "summary");
    if (vision) {
        const auto picture = [](std::vector<std::uint8_t> bytes, const std::string& text) {
            ninfer::PromptInput input;
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            ninfer::MessagePart image;
            image.kind              = ninfer::MessagePartKind::Media;
            image.media.kind        = ninfer::MediaKind::Image;
            image.media.bytes       = std::move(bytes);
            image.media.media_type  = "image/x-portable-pixmap";
            image.media.source_name = "solid.ppm";
            message.parts.push_back(std::move(image));
            message.parts.push_back(
                ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
            input.messages.push_back(std::move(message));
            input.options.enable_thinking = false;
            return input;
        };
        const std::string color = "What is the main color of this image? Answer with one word.";
        run(engine, picture(solid_ppm(220, 20, 20), "Read any text in this image."), greedy(16, true),
            "image");
        const auto red     = picture(solid_ppm(220, 20, 20), color);
        const Turn again   = run(engine, red, greedy(32, true), "same image");
        const auto& answer = again.result.content;
        check(again.result.reused_prompt_tokens + after(color) >= again.result.prompt.prompt_tokens &&
                  (answer.find("red") != std::string::npos || answer.find("Red") != std::string::npos),
              "same image resumes past its patches by digest");
        same_as_fresh(engine, red, again, "same image");
        const Turn blue = run(engine, picture(solid_ppm(20, 20, 220), color), greedy(16, true),
                              "other image");
        check(blue.result.reused_prompt_tokens < 16 &&
                  (blue.result.content.find("lue") != std::string::npos),
              "another image is not reused");
    }
    run(engine, side("You are a helpful assistant.", "Give me one tip for sleeping better."),
        greedy(24, true), "manual chat");

    auto second = first;
    add_message(second, ninfer::ChatRole::Assistant, v1.result.content);
    add_message(second, ninfer::ChatRole::User, "Something quicker, please.");
    const Turn v2 = run(engine, second, greedy(32, true), "voice 2");
    check(v2.result.reused_prompt_tokens + 16 >= v1.result.prompt.prompt_tokens,
          "voice turn after side requests resumes from its continuation");
    same_as_fresh(engine, second, v2, "voice 2");

    const Turn again =
        run(engine, side("You summarize documents in one sentence.", article), greedy(32, true),
            "summary again");
    check(again.result.reused_prompt_tokens + 1 == again.result.prompt.prompt_tokens &&
              again.result.generated_token_ids == summary.result.generated_token_ids,
          "repeated side request resumes its own checkpoint");

    const std::string memory3 = "The user is vegetarian, likes spicy food, and is cooking for two.";
    auto changed              = voice(memory3);
    const Turn v3             = run(engine, changed, greedy(32, true), "memory changed");
    const std::uint32_t tail  = after("Memory: " + memory3) + after(greeting);
    check(v3.result.reused_prompt_tokens + tail >= v3.result.prompt.prompt_tokens &&
              v3.result.reused_prompt_tokens + after(greeting) < v3.result.prompt.prompt_tokens,
          "memory change resumes from the system-plus-tools frontier");
    same_as_fresh(engine, changed, v3, "memory changed");

    // A breakpoint after a long memory message adds a frontier there. The user turn after it
    // is long enough that the displaced state's window cannot reach back to the marker.
    const std::string memory4 = "The user is vegan.\n" + numbered("Memory", 15, 4);
    auto marked               = voice(memory4);
    marked.messages.back().parts.back().text = numbered("Question", 16, 5);
    marked.context_cache.markers.push_back(
        {.after_message_count = 2, .location = ninfer::PromptCacheMarkerLocation::MessageBoundary});
    run(engine, marked, greedy(8, true), "marked");
    run(engine, side("You summarize documents in one sentence.", numbered("Memo", 30, 3)),
        greedy(8, true), "displace");
    auto marked_next = voice(memory4);
    marked_next.messages.back().parts.back().text = "And a dessert?";
    const Turn v4 = run(engine, marked_next, greedy(32, true), "after marker");
    check(v4.result.reused_prompt_tokens + after("And a dessert?") >= v4.result.prompt.prompt_tokens,
          "breakpoint after the memory message is a frontier");
    same_as_fresh(engine, marked_next, v4, "after marker");

    // Fill the host tier until the voice checkpoints are evicted; the turn must still match.
    for (int index = 0; index < 9; ++index) {
        run(engine, side("You summarize documents in one sentence.", numbered("Batch", 25, 10 + index)),
            greedy(4, true), "flood");
    }
    auto third = second;
    add_message(third, ninfer::ChatRole::Assistant, v2.result.content);
    add_message(third, ninfer::ChatRole::User, "Great, set a timer for that.");
    const Turn v5 = run(engine, third, greedy(32, true), "after eviction");
    check(v5.result.reused_prompt_tokens < v2.result.prompt.prompt_tokens,
          "evicted continuation is not reused");
    same_as_fresh(engine, third, v5, "after eviction");
}

bool usable(const char* path) {
    return path != nullptr && path[0] != '\0' && std::ifstream(path).good();
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_CHECKPOINT_WEIGHTS");
    if (!usable(artifact)) {
        std::cout << "SKIP: NINFER_GEMMA4_CHECKPOINT_WEIGHTS names no gemma4 artifact\n";
        return 77;
    }
    const char* vision = std::getenv("NINFER_GEMMA4_CHECKPOINT_VISION");
    try {
        session(artifact, vision != nullptr && std::string(vision) == "1");
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 checkpoints\n";
    return failures ? 1 : 0;
}
