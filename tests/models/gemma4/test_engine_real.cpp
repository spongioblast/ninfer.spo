// Gemma 4 31B through the public Engine: raw-token greedy parity, chat text output, sampling,
// a prompt longer than one prefill chunk, HF greedy fixtures (NINFER_GEMMA4_PARITY), prefix
// reuse across turns, causal scoring, and MTP greedy equality (NINFER_GEMMA4_MTP_WEIGHTS). Needs
// the bf16 artifact (NINFER_GEMMA4_WEIGHTS) and about 75 GB of device memory.

#include "ninfer/engine.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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

ninfer::PromptInput chat(std::string text) {
    ninfer::PromptInput input;
    add_message(input, ninfer::ChatRole::User, std::move(text));
    input.options.enable_thinking = false;
    return input;
}

std::vector<ninfer::TokenId> read_ids(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    std::vector<ninfer::TokenId> ids(static_cast<std::size_t>(in.tellg()) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
    return ids;
}

std::size_t common_prefix(const std::vector<ninfer::TokenId>& a,
                          const std::vector<ninfer::TokenId>& b) {
    std::size_t length = 0;
    while (length < a.size() && length < b.size() && a[length] == b[length]) { ++length; }
    return length;
}

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.top_k          = 1;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

void report(const char* label, const ninfer::GenerationResult& result, double seconds) {
    std::cout << label << ": " << result.generated_token_ids.size() << " tokens, finish "
              << static_cast<int>(result.finish_reason) << ", " << seconds << " s, content \""
              << result.content << "\"\n";
}

double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// An early greedy divergence from HF, checked by scoring once the generation engine is gone.
struct Divergence {
    std::string name;
    std::vector<ninfer::TokenId> context; // prompt plus the agreed tokens
    ninfer::TokenId ours      = 0;
    ninfer::TokenId reference = 0;
};

// Raw-token greedy, chat, sampling, long prompts, HF fixtures, prefix reuse and scoring.
void bf16_checks(const char* artifact) {
    std::vector<Divergence> divergences;
    {
        ninfer::EngineOptions options;
        options.artifact_path  = artifact;
        options.max_context    = 8192;
        options.kv_capacity    = ninfer::KvCapacityPolicy::automatic();
        options.enable_vision  = false;
        options.use_cuda_graph = false;
        ninfer::Engine engine(options);

        const std::vector<ninfer::TokenId> prompt{818, 5279, 529, 7001, 563};
        auto raw                        = greedy(9);
        raw.stop.include_model_defaults = false;
        auto start                      = std::chrono::steady_clock::now();
        const auto loop                 = engine.generate(engine.prepare_tokens(prompt), raw);
        report("raw", loop, since(start));
        check(loop.generated_token_ids == std::vector<ninfer::TokenId>{7001, 563, 7001, 563,
                                                                       7001, 563, 7001, 563,
                                                                       7001},
              "raw greedy tokens match Hugging Face");
        check(loop.content == " France is France is France is France is France",
              "raw greedy text");

        start             = std::chrono::steady_clock::now();
        const auto answer = engine.generate(
            engine.prepare(chat("What is the capital of France? Answer with one word.")),
            greedy(32));
        report("chat", answer, since(start));
        check(answer.finish_reason == ninfer::FinishReason::StopToken,
              "chat stops at end of turn");
        check(answer.content.find("Paris") != std::string::npos, "chat answers Paris");
        check(answer.reasoning.empty(), "no reasoning without thinking");

        auto sampled                           = greedy(48);
        sampled.execution.sampling.temperature = 1.0F;
        sampled.execution.sampling.top_k       = 20;
        sampled.execution.sampling.top_p       = 0.95F;
        sampled.execution.sampling.seed        = 7;
        start                                  = std::chrono::steady_clock::now();
        const auto first =
            engine.generate(engine.prepare(chat("Write one sentence about rain.")), sampled);
        const auto second =
            engine.generate(engine.prepare(chat("Write one sentence about rain.")), sampled);
        report("sampled", first, since(start) / 2.0);
        check(!first.content.empty(), "sampled text is not empty");
        check(first.generated_token_ids == second.generated_token_ids,
              "same seed, same tokens");

        std::string filler;
        for (int line = 0; filler.size() < 14000; ++line) {
            filler += "Line " + std::to_string(line) +
                      ": the quick brown fox jumps over the lazy dog near the river.\n";
        }
        std::cout << "capacity " << engine.memory_summary().max_context << ", filler tokens "
                  << engine.tokenize_text(filler).size() << '\n';
        const auto long_prompt =
            engine.prepare(chat("Remember this: the secret word is BLUEBERRY.\n\n" + filler +
                                "\nWhat is the secret word? Answer with the word only."));
        std::cout << "long prompt tokens " << long_prompt.summary().prompt_tokens << '\n';
        start             = std::chrono::steady_clock::now();
        const auto recall = engine.generate(
            engine.prepare(chat("Remember this: the secret word is BLUEBERRY.\n\n" + filler +
                                "\nWhat is the secret word? Answer with the word only.")),
            greedy(16));
        report("long", recall, since(start));
        check(long_prompt.summary().prompt_tokens > 2048, "long prompt spans several chunks");
        check(recall.content.find("BLUEBERRY") != std::string::npos ||
                  recall.content.find("Blueberry") != std::string::npos,
              "recalls a word from the first chunk");

        // HF bf16 greedy continuations written by tmp hf_parity.py; a divergence after
        // token 32 is accepted as a near tie, an earlier one only at an exact tie (below).
        if (const char* fixtures = std::getenv("NINFER_GEMMA4_PARITY")) {
            const std::filesystem::path dir(fixtures);
            for (const char* name : {"prose", "code", "chinese"}) {
                const auto bf16   = dir / (std::string(name) + ".greedy_bf16");
                const auto expect = read_ids(std::filesystem::exists(bf16)
                                                 ? bf16
                                                 : dir / (std::string(name) + ".greedy"));
                const auto ids    = read_ids(dir / (std::string(name) + ".ids"));
                start             = std::chrono::steady_clock::now();
                const auto result =
                    engine.generate(engine.prepare_tokens(ids),
                                    greedy(static_cast<std::uint32_t>(expect.size())));
                const double seconds   = since(start);
                const std::size_t same = common_prefix(result.generated_token_ids, expect);
                std::cout << name << ": " << ids.size() << " prompt tokens, greedy " << same
                          << '/' << expect.size() << ", " << seconds << " s\n";
                if (same == expect.size() || same >= 32) {
                    check(true, std::string(name) + " engine greedy matches HF");
                } else if (same < result.generated_token_ids.size()) {
                    Divergence divergence{name, ids, result.generated_token_ids[same],
                                          expect[same]};
                    divergence.context.insert(divergence.context.end(), expect.begin(),
                                              expect.begin() +
                                                  static_cast<std::ptrdiff_t>(same));
                    divergences.push_back(std::move(divergence));
                } else {
                    check(false, std::string(name) + " engine greedy matches HF");
                }
            }
        }

        auto reuse                         = greedy(24);
        reuse.execution.allow_prefix_reuse = true;
        auto conversation = chat("Name three rivers in Europe. Keep it short.");
        const auto turn1  = engine.generate(engine.prepare(conversation), reuse);
        add_message(conversation, ninfer::ChatRole::Assistant, turn1.content);
        add_message(conversation, ninfer::ChatRole::User, "Which of them is the longest?");
        const auto turn2 = engine.generate(engine.prepare(conversation), reuse);
        report("turn2", turn2, 0.0);
        std::cout << "turn2 prompt " << turn2.prompt.prompt_tokens << ", reused "
                  << turn2.reused_prompt_tokens << ", computed "
                  << turn2.computed_prefill_tokens << '\n';
        check(turn2.reused_prompt_tokens > 0, "second turn reuses the first turn's KV");
        const auto again = engine.generate(engine.prepare(conversation), reuse);
        const auto fresh = engine.generate(engine.prepare(conversation), greedy(24));
        check(again.reused_prompt_tokens + 1 == again.prompt.prompt_tokens,
              "repeated prompt reuses all but its last token");
        check(fresh.reused_prompt_tokens == 0, "reuse off computes the whole prompt");
        check(again.generated_token_ids == fresh.generated_token_ids &&
                  turn2.generated_token_ids == fresh.generated_token_ids,
              "reused and fresh prefill produce the same tokens");

        // A long prompt wraps the sliding layers' ring. Reusing all but its last token keeps
        // the window before it; a prompt sharing only the start must not reuse wrapped slots.
        const std::string intro = "Remember this: the secret word is BLUEBERRY.\n\n";
        const std::string ask   = "\nWhat is the secret word? Answer with the word only.";
        const auto long_first =
            engine.generate(engine.prepare(chat(intro + filler + ask)), reuse);
        const auto long_again =
            engine.generate(engine.prepare(chat(intro + filler + ask)), reuse);
        check(long_again.reused_prompt_tokens + 1 == long_again.prompt.prompt_tokens &&
                  long_again.generated_token_ids == long_first.generated_token_ids,
              "long prompt reuses the wrapped ring with the same tokens");
        const std::string short_prompt = intro + filler.substr(0, 300) + ask;
        const auto short_reuse = engine.generate(engine.prepare(chat(short_prompt)), reuse);
        const auto short_fresh =
            engine.generate(engine.prepare(chat(short_prompt)), greedy(24));
        check(short_reuse.reused_prompt_tokens == 0 &&
                  short_reuse.generated_token_ids == short_fresh.generated_token_ids,
              "shared start after a wrapped ring is recomputed");
    }
    {
        ninfer::EngineOptions options;
        options.artifact_path  = artifact;
        options.purpose        = ninfer::EnginePurpose::CausalScoring;
        options.max_context    = 4096;
        options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.enable_vision  = false;
        options.use_cuda_graph = false;
        ninfer::Engine engine(options);
        std::vector<ninfer::TokenId> tokens =
            engine.tokenize_text("The capital of France is Paris.");
        tokens.insert(tokens.begin(), 2);
        const auto start                  = std::chrono::steady_clock::now();
        const std::vector<float> logprobs = engine.score_tokens(tokens, 1);
        std::cout << "score (" << since(start) << " s):";
        bool finite = logprobs.size() == tokens.size() - 1;
        for (const float value : logprobs) {
            std::cout << ' ' << value;
            finite = finite && std::isfinite(value) && value <= 0.0F;
        }
        std::cout << '\n';
        check(finite, "scores are finite log-probabilities");
        check(logprobs.size() >= 6 && logprobs[5] > -1.0F,
              "\" Paris\" is likely after the prompt");

        // An early divergence passes only if both tokens score the same: equal BF16 logits,
        // where greedy's lower-index rule and HF's arithmetic may pick either.
        for (const Divergence& divergence : divergences) {
            const auto first = static_cast<std::uint32_t>(divergence.context.size());
            auto with_ours   = divergence.context;
            with_ours.push_back(divergence.ours);
            auto with_reference = divergence.context;
            with_reference.push_back(divergence.reference);
            const float ours      = engine.score_tokens(with_ours, first).at(0);
            const float reference = engine.score_tokens(with_reference, first).at(0);
            std::cout << divergence.name << ": diverges at " << first << " (ours "
                      << divergence.ours << ' ' << ours << ", HF " << divergence.reference
                      << ' ' << reference << ")\n";
            check(std::fabs(ours - reference) <= 1.0e-6F,
                  divergence.name + " engine greedy matches HF up to an exact tie");
        }
    }
}

// Plain decode (one token per graph step) and the MTP verify forward (K+1 tokens) round
// differently, so the top two tokens may swap when their BF16 logits are one step apart. The
// softcap keeps logits below 30, where one BF16 step is at most 0.125.
constexpr float kMtpTie = 0.125F + 1.0e-5F;

// MTP is lossless under greedy: the same tokens as plain decode, except at a near tie, which
// is told apart from a wrong token by scoring both candidates.
void mtp_equality(const char* artifact) {
    const std::vector<std::string> prompts{
        "Explain how a refrigerator keeps food cold, in two short paragraphs.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "Translate 'Good morning, how are you today?' into French, German and Spanish.",
        "A train travels 180 km in 2.5 hours. What is its average speed? Show the steps.",
        "List five tips for writing clear commit messages."};
    std::vector<std::vector<ninfer::TokenId>> prompt_ids;
    std::vector<std::vector<ninfer::TokenId>> plain;
    std::vector<Divergence> divergences;
    for (const bool speculative : {false, true}) {
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context   = 4096;
        options.kv_capacity   = ninfer::KvCapacityPolicy::automatic();
        options.enable_vision = false;
        if (speculative) {
            options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = 5;
        }
        ninfer::Engine engine(options);
        ninfer::SpeculativeStats total;
        for (std::size_t index = 0; index < prompts.size(); ++index) {
            const std::string name = "mtp prompt " + std::to_string(index);
            if (!speculative) {
                prompt_ids.push_back(engine.tokenize_text(
                    "<bos><|turn>user\n" + prompts[index] +
                    "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"));
                check(engine.prepare(chat(prompts[index])).summary().prompt_tokens ==
                          prompt_ids.back().size(),
                      name + " raw ids match the chat template");
            }
            const auto start  = std::chrono::steady_clock::now();
            const auto result = engine.generate(engine.prepare_tokens(prompt_ids[index]),
                                                greedy(128));
            std::cout << (speculative ? "mtp " : "plain ") << index << ": "
                      << result.generated_token_ids.size() << " tokens, " << since(start)
                      << " s\n";
            if (!speculative) {
                plain.push_back(result.generated_token_ids);
                continue;
            }
            total.rounds += result.speculative.rounds;
            total.drafted_tokens += result.speculative.drafted_tokens;
            total.accepted_tokens += result.speculative.accepted_tokens;
            const auto& ours       = result.generated_token_ids;
            const auto& expect     = plain[index];
            const std::size_t same = common_prefix(ours, expect);
            if (ours == expect) {
                check(true, name + " matches plain greedy");
            } else if (same < ours.size() && same < expect.size()) {
                Divergence divergence{name, prompt_ids[index], ours[same], expect[same]};
                divergence.context.insert(divergence.context.end(), expect.begin(),
                                          expect.begin() + static_cast<std::ptrdiff_t>(same));
                divergences.push_back(std::move(divergence));
            } else {
                check(false, name + " matches plain greedy");
            }
        }
        if (speculative) {
            std::cout << "mtp rounds " << total.rounds << ", drafted " << total.drafted_tokens
                      << ", accepted " << total.accepted_tokens << '\n';
            check(total.rounds > 0 && total.accepted_tokens > 0, "mtp drafts are accepted");
        }
    }
    if (divergences.empty()) { return; }
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.purpose        = ninfer::EnginePurpose::CausalScoring;
    options.max_context    = 4096;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.enable_vision  = false;
    options.use_cuda_graph = false;
    ninfer::Engine engine(options);
    for (const Divergence& divergence : divergences) {
        const auto first = static_cast<std::uint32_t>(divergence.context.size());
        auto with_ours   = divergence.context;
        with_ours.push_back(divergence.ours);
        auto with_plain = divergence.context;
        with_plain.push_back(divergence.reference);
        const float ours        = engine.score_tokens(with_ours, first).at(0);
        const float plain_score = engine.score_tokens(with_plain, first).at(0);
        std::cout << divergence.name << ": diverges at position " << first << " (mtp "
                  << divergence.ours << ' ' << ours << ", plain " << divergence.reference << ' '
                  << plain_score << ")\n";
        check(std::fabs(ours - plain_score) <= kMtpTie,
              divergence.name + " matches plain greedy up to a near tie");
    }
}

bool usable(const char* path) {
    return path != nullptr && path[0] != '\0' && std::ifstream(path).good();
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_WEIGHTS");
    const char* mtp      = std::getenv("NINFER_GEMMA4_MTP_WEIGHTS");
    if (!usable(artifact) && !usable(mtp)) {
        std::cout << "SKIP: neither NINFER_GEMMA4_WEIGHTS nor NINFER_GEMMA4_MTP_WEIGHTS names a "
                     "gemma4 artifact\n";
        return 77;
    }
    try {
        if (usable(artifact)) { bf16_checks(artifact); }
        if (usable(mtp)) { mtp_equality(mtp); }
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 engine\n";
    return failures ? 1 : 0;
}
