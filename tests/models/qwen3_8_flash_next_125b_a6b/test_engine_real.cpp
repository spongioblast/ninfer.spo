#include "ninfer/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Full;
    options.enable_vision                    = true;
    options.use_cuda_graph                   = true;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 4;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes       = 1;
    return options;
}

const std::vector<ninfer::TokenId>& canonical_prompt() {
    static const std::vector<ninfer::TokenId> prompt{
        248045, 846, 198,  814, 20139,  303, 2250,   2716,  22157, 3069,   279, 12515,  7701, 6105,
        2261,   279, 1834, 13,  248046, 198, 248045, 74455, 198,   248068, 271, 248069, 271};
    return prompt;
}

// The third output token is a near-tie in the quantized artifacts (27891 and 5435 within about
// 0.5 logit), so a change of BF16 or FP8 summation order can select either continuation. Both are
// coherent text; the second is also what the fp8proj artifact produces.
const std::vector<std::vector<ninfer::TokenId>>& canonical_outputs() {
    static const std::vector<std::vector<ninfer::TokenId>> outputs{
        {29108, 4009, 27891, 8964, 579, 16078, 321, 1100, 9872, 303, 660, 17425},
        {29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 864, 43000}};
    return outputs;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

int exercise_mtp_and_prefix(ninfer::Engine& engine) {
    // Greedy output from the source checkpoint for the canonical non-thinking chat template.
    // Checking semantic text would require duplicating the tokenizer in this C++ integration
    // test, so protect the exact token prefix instead. This catches numerically plausible but
    // language-corrupt model execution, including routed-MoE row-layout regressions.
    const auto& prompt                   = canonical_prompt();
    const auto& expected_prefixes        = canonical_outputs();
    const ninfer::GenerationResult first = engine.generate(
        engine.prepare_tokens(prompt), greedy_options(expected_prefixes.front().size(), true));
    const bool known_prefix =
        std::find(expected_prefixes.begin(), expected_prefixes.end(),
                  first.generated_token_ids) != expected_prefixes.end();
    if (!known_prefix || first.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        first.speculative.rounds == 0) {
        std::cerr << "Flash-Next greedy text prefix is corrupt or did not complete through MTP\n";
        std::cerr << "actual:";
        for (auto token : first.generated_token_ids) { std::cerr << ' ' << token; }
        for (const auto& expected_prefix : expected_prefixes) {
            std::cerr << "\nexpected:";
            for (auto token : expected_prefix) { std::cerr << ' ' << token; }
        }
        std::cerr << "\nbackend=" << static_cast<int>(first.speculative.backend)
                  << " rounds=" << first.speculative.rounds << '\n';
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, false));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2 ||
        cold.reused_prompt_tokens != 0 || cold.generated_token_ids != reused.generated_token_ids) {
        std::cerr << "Flash-Next prefix reuse is incorrect: reused=" << reused.reused_prompt_tokens
                  << " expected=" << expected_reuse << " cold/reused output match="
                  << (cold.generated_token_ids == reused.generated_token_ids) << '\n';
        return 1;
    }

    if (first.generated_token_ids[0] == first.generated_token_ids[1]) {
        std::cerr << "Flash-Next partial-terminal fixture repeats its first token\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = greedy_options(6, true);
    stop_options.stop.token_ids.push_back(first.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        stopped.generated_token_ids[0] != first.generated_token_ids[0] ||
        stopped.generated_token_ids[1] != first.generated_token_ids[1]) {
        std::cerr << "Flash-Next custom stop did not terminate inside the MTP round\n";
        return 1;
    }

    // Reused and cold prefill run different schedules, so the continuation follows the greedy
    // text: after a newline here the next token is an exact tie in the cold run.
    std::vector<ninfer::TokenId> stopped_continuation = prompt;
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(first.generated_token_ids[2]);
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, true));
    const ninfer::GenerationResult stopped_cold =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, false));
    const std::uint32_t expected_stopped_reuse =
        static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
    if (stopped_reuse.reused_prompt_tokens != expected_stopped_reuse ||
        stopped_cold.reused_prompt_tokens != 0 ||
        stopped_cold.generated_token_ids != stopped_reuse.generated_token_ids) {
        std::cerr << "Flash-Next partial MTP terminal reused " << stopped_reuse.reused_prompt_tokens
                  << ", expected " << expected_stopped_reuse << ", cold/reused output match="
                  << (stopped_cold.generated_token_ids == stopped_reuse.generated_token_ids)
                  << '\n';
        return 1;
    }
    return 0;
}

int exercise_ordinary_greedy(const char* artifact) {
    ninfer::EngineOptions options    = engine_options(artifact);
    options.speculative.backend      = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = 0;
    options.enable_vision            = false;
    ninfer::Engine engine(std::move(options));
    // On the 5435 branch the one-token decode kernel and the MTP verify kernel part at token 11,
    // another near-tie, so the ordinary route is held to the first 10 tokens.
    constexpr std::size_t kOrdinaryCheckedTokens = 10;
    std::vector<std::vector<ninfer::TokenId>> expected_prefixes;
    for (const auto& output : canonical_outputs()) {
        expected_prefixes.emplace_back(output.begin(), output.begin() + kOrdinaryCheckedTokens);
    }
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(canonical_prompt()),
                        greedy_options(kOrdinaryCheckedTokens, false));
    const bool known_prefix =
        std::find(expected_prefixes.begin(), expected_prefixes.end(),
                  result.generated_token_ids) != expected_prefixes.end();
    if (result.speculative.backend != ninfer::SpeculativeBackend::None || !known_prefix) {
        std::cerr << "Flash-Next ordinary greedy output disagrees with the checkpoint/MTP path\n";
        std::cerr << "ordinary:";
        for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
        for (const auto& expected_prefix : expected_prefixes) {
            std::cerr << "\nexpected:";
            for (const auto token : expected_prefix) { std::cerr << ' ' << token; }
        }
        std::cerr << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_state(ninfer::Engine& engine) {
    // Different frontiers exercise local prefill rows and shared decode rows. Repeat in
    // reversed admission order to reuse both physical lanes and recurrent state slots.
    const auto& expected_prefixes = canonical_outputs();
    const auto solo               = engine.generate(engine.prepare_tokens(canonical_prompt()),
                                                    greedy_options(12, false));
    const auto& fixture           = solo.generated_token_ids;
    if (std::find(expected_prefixes.begin(), expected_prefixes.end(), fixture) ==
        expected_prefixes.end()) {
        std::cerr << "Flash-Next solo greedy output is not a known canonical prefix\n";
        return 1;
    }
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), fixture.begin(), fixture.begin() + 4);
    const auto cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(8, false));
    const auto before = engine.runtime_stats();
    for (bool reverse : {false, true}) {
        auto first =
            engine.submit(engine.prepare_tokens(reverse ? continuation : canonical_prompt()),
                          greedy_options(reverse ? 8 : 12, false));
        auto second =
            engine.submit(engine.prepare_tokens(reverse ? canonical_prompt() : continuation),
                          greedy_options(reverse ? 12 : 8, false));
        const auto a        = first.wait();
        const auto b        = second.wait();
        const auto& root    = reverse ? b : a;
        const auto& resumed = reverse ? a : b;
        if (root.generated_token_ids != fixture ||
            resumed.generated_token_ids != cold.generated_token_ids) {
            std::cerr << "Flash-Next concurrent lane reuse changed the canonical fixture\n";
            return 1;
        }
    }
    const auto after = engine.runtime_stats();
    if (after.decode_row_rounds - before.decode_row_rounds <=
        after.decode_rounds - before.decode_rounds) {
        std::cerr << "Flash-Next concurrent fixture never executed a two-row decode\n";
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), greedy_options(1, false));
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "Flash-Next Vision did not complete through the public Engine\n";
        return 1;
    }
    return 0;
}

// New chats share a system prompt. Once the engine has seen it twice, a chat captures that prefix
// and keeps decoding on its own lane, and later chats copy the checkpoint. Each chat must match
// the same prompt prefilled cold, so neither the capturing lane nor a copy reads changed KV.
int exercise_shared_prefix(ninfer::Engine& engine) {
    const auto chat = [](const char* question) {
        const auto text = [](std::string value) {
            return ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(value), .media = {}};
        };
        ninfer::ChatMessage system;
        system.role = ninfer::ChatRole::System;
        system.parts.push_back(text("You are a careful assistant. Answer in one short sentence "
                                    "and give the unit of every number you mention."));
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(text(question));
        ninfer::PromptInput input;
        input.messages.push_back(std::move(system));
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        return input;
    };
    std::uint32_t reused_chats = 0;
    for (const char* question : {"How tall is Mount Everest?", "How far away is the Moon?",
                                 "How long is a marathon?", "How heavy is a litre of water?"}) {
        const auto warm = engine.generate(engine.prepare(chat(question)), greedy_options(6, true));
        const auto cold = engine.generate(engine.prepare(chat(question)), greedy_options(6, false));
        if (warm.generated_token_ids.size() != 6 ||
            warm.generated_token_ids != cold.generated_token_ids) {
            std::cerr << "Flash-Next shared-prefix chat differs from cold: reused="
                      << warm.reused_prompt_tokens << " question=" << question << '\n';
            return 1;
        }
        if (warm.reused_prompt_tokens != 0) { ++reused_chats; }
    }
    if (reused_chats == 0) {
        std::cerr << "Flash-Next chats never reused their shared system prompt\n";
        return 1;
    }
    return 0;
}

// Compare graph replay with the same eager arithmetic schedules on the supplied artifact.
// Projection and selected-slot state mathematics have independent FP64 Op qualification.
int exercise_graph_equivalence(const char* artifact) {
    std::vector<std::vector<ninfer::TokenId>> reference;
    std::vector<std::vector<ninfer::TokenId>> continuation_reference;
    std::vector<std::vector<ninfer::TokenId>> batch_reference;
    for (const bool graphs : {false, true}) {
        auto options                     = engine_options(artifact);
        options.speculative.backend      = ninfer::SpeculativeBackend::None;
        options.speculative.draft_tokens = 0;
        options.enable_vision            = false;
        options.use_cuda_graph           = graphs;
        options.max_context              = 1024;
        options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        ninfer::Engine engine(options);
        if ((engine.memory_summary().cuda_graph_allowance_bytes != 0) != graphs) {
            throw std::runtime_error("ordinary graphs were silently disabled");
        }
        std::size_t case_index = 0;
        const std::size_t allowance_before = engine.memory_summary().cuda_graph_allowance_bytes;
        for (const std::size_t length : {27U, 124U, 508U}) {
            auto prompt = canonical_prompt();
            prompt.resize(length, 198);
            const auto first =
                engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
            if (first.generated_token_ids.size() != 16) {
                throw std::runtime_error("graph fixture did not execute all requested steps");
            }
            if (!graphs) {
                reference.push_back(first.generated_token_ids);
            } else if (reference.at(case_index) != first.generated_token_ids) {
                throw std::runtime_error("graph/eager output mismatch at prompt length " +
                                         std::to_string(length));
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto repeated =
                    engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
                if (repeated.generated_token_ids != first.generated_token_ids) {
                    throw std::runtime_error("ordinary repeated output mismatch");
                }
            }
            // 27 tokens is inside the small-pass graph cap, so the repeat replays a captured
            // prefill. A longer prompt stays eager. The allowance grows by the reserved graph.
            if (graphs && length == 27U &&
                engine.memory_summary().cuda_graph_allowance_bytes <= allowance_before) {
                throw std::runtime_error("small prefill pass was not captured");
            }
            engine.generate(engine.prepare_tokens(prompt), greedy_options(16, true));
            auto history = prompt;
            history.insert(history.end(), first.generated_token_ids.begin(),
                           first.generated_token_ids.end());
            history.push_back(198);
            const auto reused =
                engine.generate(engine.prepare_tokens(history), greedy_options(8, true));
            const auto cold =
                engine.generate(engine.prepare_tokens(history), greedy_options(8, false));
            if (reused.reused_prompt_tokens == 0) {
                throw std::runtime_error("ordinary continuation fixture never reused a prefix");
            }
            // Prefill and incremental schedules have their own qualified floating-point profiles.
            // Verify that graph replay preserves each schedule, including cached state, rather
            // than imposing bitwise equality between different arithmetic schedules.
            if (!graphs) {
                continuation_reference.push_back(reused.generated_token_ids);
                continuation_reference.push_back(cold.generated_token_ids);
            } else if (continuation_reference.at(case_index * 2) != reused.generated_token_ids ||
                       continuation_reference.at(case_index * 2 + 1) != cold.generated_token_ids) {
                throw std::runtime_error("graph/eager continuation output mismatch");
            }
            ++case_index;
        }
        auto second_prompt = canonical_prompt();
        second_prompt.insert(second_prompt.end(), reference[0].begin(), reference[0].begin() + 4);
        const auto before = engine.runtime_stats();
        for (bool reverse : {false, true}) {
            auto a =
                engine.submit(engine.prepare_tokens(reverse ? second_prompt : canonical_prompt()),
                              greedy_options(12, false));
            auto b =
                engine.submit(engine.prepare_tokens(reverse ? canonical_prompt() : second_prompt),
                              greedy_options(12, false));
            const auto first    = a.wait();
            const auto second   = b.wait();
            const auto& root    = reverse ? second : first;
            const auto& resumed = reverse ? first : second;
            // Batch size selects projection/reduction routes. Compare the same compact-batch
            // schedule across launch modes; selected-slot mathematics is independently checked
            // by the GDN FP64 oracle, rather than treating single-row output as an oracle.
            if (!graphs) {
                batch_reference.push_back(root.generated_token_ids);
                batch_reference.push_back(resumed.generated_token_ids);
            } else {
                const std::size_t index = reverse ? 2 : 0;
                if (root.generated_token_ids != batch_reference.at(index) ||
                    resumed.generated_token_ids != batch_reference.at(index + 1)) {
                    throw std::runtime_error("ordinary graph/eager compact batch mismatch");
                }
            }
        }
        const auto after = engine.runtime_stats();
        if (after.decode_row_rounds - before.decode_row_rounds <=
            after.decode_rounds - before.decode_rounds) {
            throw std::runtime_error("graph fixture never executed a two-row batch");
        }
        std::cout << "OK " << (graphs ? "graph" : "eager")
                  << " equivalence: repeats, continuations, batch/lane "
                     "reuse, 128/512 profile boundaries; decode_rounds="
                  << after.decode_rounds << '\n';
    }
    return 0;
}

void set_process_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// A four-turn MTP chat, played three times so small prefill passes run eager, are captured,
// then replay. With the audit on, every third use re-captures and compares launch arguments.
// The graph engine must reproduce the eager engine's tokens and accepted counts exactly,
// including two lanes prefilling side by side (each lane has its own block-table row).
int exercise_prefill_graph_mtp(const char* artifact) {
    set_process_env("NINFER_DEBUG_PREFILL_GRAPH_AUDIT", "1");
    const std::vector<ninfer::TokenId> user_turn{198,   248045, 846, 198, 814,   20139,
                                                 248046, 198,   248045, 74455, 198};
    std::vector<std::vector<ninfer::TokenId>> reference_tokens;
    std::vector<std::uint64_t> reference_accepted;
    for (const bool graphs : {false, true}) {
        auto options           = engine_options(artifact);
        options.enable_vision  = false;
        options.use_cuda_graph = graphs;
        options.max_context    = 1024;
        options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        ninfer::Engine engine(options);
        std::uint64_t hits       = 0;
        std::uint64_t audited    = 0;
        std::uint64_t mismatches = 0;
        std::size_t index        = 0;
        const auto check = [&](const ninfer::GenerationResult& result, const char* label) {
            hits += result.engine_timing.prefill_graph_hits;
            audited += result.engine_timing.prefill_graph_audited;
            mismatches += result.engine_timing.prefill_graph_audit_mismatches;
            if (result.speculative.backend != ninfer::SpeculativeBackend::Mtp) {
                throw std::runtime_error(std::string(label) + " did not run MTP");
            }
            if (!graphs) {
                reference_tokens.push_back(result.generated_token_ids);
                reference_accepted.push_back(result.speculative.accepted_tokens);
            } else if (reference_tokens.at(index) != result.generated_token_ids ||
                       reference_accepted.at(index) != result.speculative.accepted_tokens) {
                throw std::runtime_error(std::string("prefill graph changed MTP output: ") +
                                         label + " #" + std::to_string(index));
            }
            ++index;
        };
        std::vector<std::vector<ninfer::TokenId>> histories;
        for (int play = 0; play < 3; ++play) {
            auto history = canonical_prompt();
            for (int turn = 0; turn < 4; ++turn) {
                const auto result =
                    engine.generate(engine.prepare_tokens(history), greedy_options(12, true));
                check(result, "chat turn");
                history.insert(history.end(), result.generated_token_ids.begin(),
                               result.generated_token_ids.end());
                history.insert(history.end(), user_turn.begin(), user_turn.end());
                if (play == 0) { histories.push_back(history); }
            }
        }
        for (bool reverse : {false, true}) {
            auto a = engine.submit(engine.prepare_tokens(histories[reverse ? 2 : 1]),
                                   greedy_options(12, true));
            auto b = engine.submit(engine.prepare_tokens(histories[reverse ? 1 : 2]),
                                   greedy_options(12, true));
            const auto first  = a.wait();
            const auto second = b.wait();
            check(first, "concurrent first");
            check(second, "concurrent second");
        }
        if (graphs && (hits == 0 || audited == 0 || mismatches != 0)) {
            throw std::runtime_error("prefill graph MTP fixture: hits=" + std::to_string(hits) +
                                     " audited=" + std::to_string(audited) +
                                     " mismatches=" + std::to_string(mismatches));
        }
        std::cout << "OK " << (graphs ? "graph" : "eager")
                  << " MTP multi-turn prefill: requests=" << index << " graph_hits=" << hits
                  << " audited=" << audited << '\n';
    }
    return 0;
}

// Two requests whose KV entitlements cannot both fit (6 + 6 pages in an 8-page pool). A lane
// reserves its whole entitlement at bind, so the second must wait for the first, not fail or
// pause it, and each must produce its solo tokens. A third request afterwards must still run.
int exercise_kv_pressure(const char* artifact) {
    auto options                  = engine_options(artifact);
    options.enable_vision         = false;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.max_pending_requests  = 4;
    options.context_cache.enabled = false;
    ninfer::Engine engine(options);
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), {29108, 4009});
    const std::vector<std::vector<ninfer::TokenId>> prompts{canonical_prompt(), continuation};
    std::vector<std::vector<ninfer::TokenId>> solo;
    for (const auto& prompt : prompts) {
        solo.push_back(
            engine.generate(engine.prepare_tokens(prompt), greedy_options(300, false))
                .generated_token_ids);
    }
    const auto before = engine.runtime_stats();
    auto first        = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(300, false));
    auto second       = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(300, false));
    const auto a      = first.wait();
    const auto b      = second.wait();
    const auto after  = engine.runtime_stats();
    if (a.generated_token_ids != solo[0] || b.generated_token_ids != solo[1]) {
        std::cerr << "Flash-Next KV pressure changed a request's tokens\n";
        return 1;
    }
    if (after.decode_row_rounds - before.decode_row_rounds !=
        after.decode_rounds - before.decode_rounds) {
        std::cerr << "Flash-Next KV pressure fixture ran both requests at once; the pool is too "
                     "large to test waiting\n";
        return 1;
    }
    if (after.preemptions != before.preemptions) {
        std::cerr << "Flash-Next paused a lane that holds its whole KV entitlement\n";
        return 1;
    }
    const auto third =
        engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(300, false));
    if (third.generated_token_ids != solo[0]) {
        std::cerr << "Flash-Next request after KV pressure changed its tokens\n";
        return 1;
    }
    std::cout << "OK Flash-Next KV pressure: second request waited, tokens equal solo\n";
    return 0;
}

// More finished chats than the old fixed catalog of eight records. With twelve state images
// every chat keeps its continuation and resumes its whole history, matching a cold prefill.
int exercise_many_checkpoints(const char* artifact) {
    constexpr std::uint32_t kChats = 10;
    auto options                             = engine_options(artifact);
    options.enable_vision                    = false;
    options.max_context                      = 1024;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.context_cache.device_state_slots = kChats;
    ninfer::Engine engine(options);
    std::vector<std::vector<ninfer::TokenId>> histories;
    for (std::uint32_t chat = 0; chat < kChats; ++chat) {
        auto prompt = canonical_prompt();
        prompt[5]   = static_cast<ninfer::TokenId>(1000 + 37 * chat);
        const auto first =
            engine.generate(engine.prepare_tokens(prompt), greedy_options(4, true));
        prompt.insert(prompt.end(), first.generated_token_ids.begin(),
                      first.generated_token_ids.end());
        prompt.push_back(198);
        histories.push_back(std::move(prompt));
    }
    for (std::uint32_t chat = 0; chat < kChats; ++chat) {
        const auto& history = histories[chat];
        const auto expected = static_cast<std::uint32_t>(history.size() - 2);
        const auto warm = engine.generate(engine.prepare_tokens(history), greedy_options(2, true));
        if (warm.reused_prompt_tokens != expected) {
            std::cerr << "Flash-Next chat " << chat << " lost its continuation: reused="
                      << warm.reused_prompt_tokens << " expected=" << expected << '\n';
            return 1;
        }
        if (chat + 1 == kChats) {
            const auto cold =
                engine.generate(engine.prepare_tokens(history), greedy_options(2, false));
            if (cold.generated_token_ids != warm.generated_token_ids) {
                std::cerr << "Flash-Next resumed continuation differs from cold prefill\n";
                return 1;
            }
        }
    }
    std::cout << "OK Flash-Next " << kChats << " continuations kept and resumed\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        if (argc == 2 && std::string(argv[1]) == "--graph-equivalence") {
            return exercise_graph_equivalence(artifact);
        }
        if (argc == 2 && std::string(argv[1]) == "--prefill-graph-mtp") {
            return exercise_prefill_graph_mtp(artifact);
        }
        if (argc == 2 && std::string(argv[1]) == "--kv-pressure") {
            return exercise_kv_pressure(artifact);
        }
        if (argc == 2 && std::string(argv[1]) == "--many-checkpoints") {
            return exercise_many_checkpoints(artifact);
        }
        for (const auto head : {ninfer::ProposalHead::Full, ninfer::ProposalHead::Optimized}) {
            auto options                      = engine_options(artifact);
            options.speculative.proposal_head = head;
            ninfer::Engine engine(options);
            const ninfer::LoadSummary load = engine.load_summary();
            if (load.architecture != "Qwen3_8FlashNextForCausalLM" ||
                load.host_to_device_bytes == 0) {
                std::cerr << "Flash-Next Engine construction has an invalid load summary\n";
                return 1;
            }
            if (exercise_mtp_and_prefix(engine) != 0) { return 1; }
            if (exercise_concurrent_state(engine) != 0) { return 1; }
            if (exercise_vision(engine) != 0) { return 1; }
            if (exercise_shared_prefix(engine) != 0) { return 1; }
        }
        {
            // Adaptive MTP selects each round's draft count; greedy output and prefix/concurrent
            // state transactions must match the fixed-window contract.
            auto options                              = engine_options(artifact);
            options.speculative.draft_tokens          = 0;
            options.speculative.adaptive_draft_tokens = true;
            options.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
            ninfer::Engine engine(options);
            if (exercise_mtp_and_prefix(engine) != 0) { return 1; }
            if (exercise_concurrent_state(engine) != 0) { return 1; }
        }
        if (exercise_ordinary_greedy(artifact) != 0) { return 1; }
        std::cout << "OK Qwen3.8 Flash Next real Engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 Flash Next real Engine: " << error.what() << '\n';
        return 1;
    }
}
