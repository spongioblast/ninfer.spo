// MTP assistant on real weights. NINFER_GEMMA4_MTP_WEIGHTS names an artifact with the mtp
// component and NINFER_GEMMA4_PARITY the prompt fixtures (<name>.ids per index.json entry); the
// test skips without them. After each prompt it runs greedy rounds of kDrafts drafts verified by
// one target forward and reports how many drafts the target agrees with. With
// NINFER_GEMMA4_DRAFT_DUMP it also writes the first round's drafter inputs and outputs:
// <name>.draft.json, <name>.carry.bin (BF16 [kDrafts+1][hidden], the carried state before every
// step), <name>.sliding.bin and <name>.global.bin (BF16 K then V as [kv][positions][D] of the
// assistant's KV source layers, the sliding one over its last window positions).
// NINFER_GEMMA4_MARGIN_CORPUS names a whitespace token-id file. With the MTP artifact set, the
// test then skips the fixtures and scores that prompt: for each round it prints the first draft,
// the target argmax, that draft's rank in the target distribution, and the softcapped logit gap.
#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/execution/draft_forward.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/sliding_ring.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace g4 = ninfer::models::gemma4;

constexpr int kPage   = 64;
constexpr int kDrafts = 5;
constexpr int kRounds = 16;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::int32_t> read_ids(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(in.tellg()) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
    return ids;
}

std::vector<std::int32_t> read_text_ids(const fs::path& path, std::size_t limit) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    std::vector<std::int32_t> ids;
    ids.reserve(limit);
    long long value = 0;
    while (ids.size() < limit && (in >> value)) {
        ids.push_back(static_cast<std::int32_t>(value));
    }
    if (ids.size() < limit) {
        throw std::runtime_error("corpus is shorter than the requested prompt");
    }
    return ids;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

void write_raw(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

// One sequence's paged KV with identity block tables.
struct Cache {
    std::vector<ninfer::DeviceBuffer> k_store;
    std::vector<ninfer::DeviceBuffer> v_store;
    std::vector<ninfer::Tensor> k_pages;
    std::vector<ninfer::Tensor> v_pages;
    ninfer::DeviceBuffer table_buf;
    ninfer::Tensor table;
    std::vector<g4::Gemma4CacheLayer> layers;

    Cache(const g4::LoadedText& loaded, int pages)
        : table_buf(static_cast<std::size_t>(pages) * 4),
          table(nullptr, ninfer::DType::I32, {pages}) {
        const auto ids = g4::ring_block_table(pages, pages);
        table_buf.copy_from_host(ids.data(), table_buf.bytes);
        table = ninfer::Tensor(table_buf.p, ninfer::DType::I32, {pages});
        const std::size_t count = loaded.layers.size();
        k_store.reserve(count);
        v_store.reserve(count);
        for (const auto& layer : loaded.layers) {
            const std::size_t bytes = static_cast<std::size_t>(layer.head_dim) * kPage *
                                      static_cast<std::size_t>(layer.kv_heads) * pages * 2;
            k_store.emplace_back(bytes);
            v_store.emplace_back(bytes);
            k_store.back().fill(0);
            v_store.back().fill(0);
            const std::initializer_list<std::int32_t> shape{layer.head_dim, kPage, layer.kv_heads,
                                                            pages};
            k_pages.emplace_back(k_store.back().p, ninfer::DType::BF16, shape);
            v_pages.emplace_back(v_store.back().p, ninfer::DType::BF16, shape);
        }
        for (std::size_t i = 0; i < count; ++i) { layers.push_back({&k_pages[i], &v_pages[i], &table}); }
    }

    // K then V of `layer` for positions [first, last] as BF16 [kv][positions][D].
    std::vector<std::uint16_t> export_layer(const g4::Gemma4LayerWeights& layer, std::size_t index,
                                            int first, int last) const {
        const std::size_t dim = layer.head_dim, kv = layer.kv_heads;
        const std::size_t count = static_cast<std::size_t>(last - first + 1);
        std::vector<std::uint16_t> out(2 * kv * count * dim);
        for (int side = 0; side < 2; ++side) {
            const auto& store = side == 0 ? k_store[index] : v_store[index];
            std::vector<std::uint16_t> host(store.bytes / 2);
            store.copy_to_host(host.data(), store.bytes);
            for (std::size_t h = 0; h < kv; ++h) {
                for (std::size_t t = 0; t < count; ++t) {
                    const std::size_t position = static_cast<std::size_t>(first) + t;
                    const std::size_t page = position / kPage, slot = position % kPage;
                    const std::size_t from = dim * (slot + kPage * (h + kv * page));
                    std::copy_n(host.data() + from, dim,
                                out.data() + ((side * kv + h) * count + t) * dim);
                }
            }
        }
        return out;
    }
};

struct Runner {
    g4::LoadedText& loaded;
    cudaStream_t stream;
    ninfer::ops::Bf16GemmContext gemm;
    ninfer::DeviceArena workspace;
    ninfer::DeviceBuffer hidden_buf;
    ninfer::DeviceBuffer carry_buf;
    ninfer::DeviceBuffer slots_buf;

    Runner(g4::LoadedText& model, cudaStream_t s, int max_tokens, std::size_t workspace_bytes)
        : loaded(model), stream(s), gemm(s), workspace(workspace_bytes),
          hidden_buf(static_cast<std::size_t>(model.hidden) * max_tokens * 2),
          carry_buf(static_cast<std::size_t>(model.hidden) * 2), slots_buf((2 + kDrafts) * 4) {}

    // Runs `ids` at positions first.. and returns the argmax of every scored column. With
    // `carry_column` the final-normed hidden state of that column lands in carry_buf. When
    // `first_column` is set and the forward is at most kDrafts+1 tokens, it receives the
    // softcapped logits of column 0 (the anchor).
    std::vector<std::int32_t> forward(Cache& cache, const std::vector<std::int32_t>& ids,
                                      int first, int carry_column,
                                      std::vector<float>* first_column = nullptr) {
        const int count = static_cast<int>(ids.size());
        std::vector<std::int32_t> positions(ids.size());
        for (int i = 0; i < count; ++i) { positions[static_cast<std::size_t>(i)] = first + i; }
        workspace.reset();
        auto id_tensor  = workspace.alloc(ninfer::DType::I32, {count});
        auto pos_tensor = workspace.alloc(ninfer::DType::I32, {count});
        check(cudaMemcpyAsync(id_tensor.data, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice,
                              stream), "ids");
        check(cudaMemcpyAsync(pos_tensor.data, positions.data(), ids.size() * 4,
                              cudaMemcpyHostToDevice, stream), "positions");
        ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::BF16, {loaded.hidden, count});
        g4::gemma4_layers(id_tensor, hidden, loaded.token_embedding, loaded.layers, cache.layers,
                          pos_tensor, gemm, workspace, stream);
        // Only the last kDrafts + 1 columns are ever scored.
        const int scored = std::min(count, kDrafts + 1);
        const ninfer::Tensor tail = hidden.slice(1, count - scored, scored);
        auto logits = workspace.alloc(ninfer::DType::BF16, {loaded.vocab, scored});
        auto out    = workspace.alloc(ninfer::DType::I32, {scored});
        g4::gemma4_logits(tail, logits, loaded.final_norm, loaded.output_head, loaded.softcap,
                          loaded.rms_eps, gemm, workspace, stream);
        ninfer::ops::argmax(logits, out, loaded.vocab, stream);
        if (carry_column >= 0) {
            ninfer::Tensor carry(carry_buf.p, ninfer::DType::BF16, {loaded.hidden, 1});
            ninfer::ops::rmsnorm(hidden.slice(1, carry_column, 1),
                                 ninfer::Tensor(const_cast<void*>(loaded.final_norm),
                                                ninfer::DType::BF16, {loaded.hidden}),
                                 loaded.rms_eps, false, carry, stream);
        }
        std::vector<std::uint16_t> raw_column;
        if (first_column != nullptr) {
            if (count != scored) {
                throw std::logic_error("first-column logits are the anchor only for a short forward");
            }
            raw_column.resize(static_cast<std::size_t>(loaded.vocab));
            check(cudaMemcpyAsync(raw_column.data(), logits.data, raw_column.size() * 2,
                                  cudaMemcpyDeviceToHost, stream),
                  "column logits");
        }
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(scored));
        check(cudaMemcpyAsync(tokens.data(), out.data, tokens.size() * 4, cudaMemcpyDeviceToHost,
                              stream), "argmax");
        check(cudaStreamSynchronize(stream), "forward");
        if (first_column != nullptr) {
            first_column->resize(raw_column.size());
            for (std::size_t i = 0; i < raw_column.size(); ++i) {
                (*first_column)[i] = bf16_to_float(raw_column[i]);
            }
        }
        return tokens;
    }

    // kDrafts draft steps from carry_buf at `position`; `states` collects carry before each step.
    std::vector<std::int32_t> draft(Cache& cache, std::int32_t anchor, std::int32_t position,
                                    std::vector<std::uint16_t>* states) {
        const std::int32_t head[2] = {anchor, position};
        check(cudaMemcpyAsync(slots_buf.p, head, sizeof(head), cudaMemcpyHostToDevice, stream),
              "draft head");
        auto* slots = static_cast<std::int32_t*>(slots_buf.p);
        const ninfer::Tensor pos(slots + 1, ninfer::DType::I32, {1});
        ninfer::Tensor carry(carry_buf.p, ninfer::DType::BF16, {loaded.hidden, 1});
        const std::size_t plane = static_cast<std::size_t>(loaded.hidden);
        for (int step = 0; step < kDrafts; ++step) {
            if (states != nullptr) {
                states->resize(states->size() + plane);
                check(cudaMemcpyAsync(states->data() + states->size() - plane, carry.data,
                                      plane * 2, cudaMemcpyDeviceToHost, stream), "state");
            }
            workspace.reset();
            const ninfer::Tensor token(step == 0 ? slots : slots + 1 + step, ninfer::DType::I32,
                                       {1});
            ninfer::Tensor next(slots + 2 + step, ninfer::DType::I32, {1});
            g4::gemma4_draft_step(token, pos, carry, next, loaded.token_embedding, *loaded.draft,
                                  cache.layers, gemm, workspace, stream);
        }
        if (states != nullptr) {
            states->resize(states->size() + plane);
            check(cudaMemcpyAsync(states->data() + states->size() - plane, carry.data, plane * 2,
                                  cudaMemcpyDeviceToHost, stream), "state");
        }
        std::vector<std::int32_t> drafts(kDrafts);
        check(cudaMemcpyAsync(drafts.data(), slots + 2, kDrafts * 4, cudaMemcpyDeviceToHost,
                              stream), "drafts");
        check(cudaStreamSynchronize(stream), "draft");
        return drafts;
    }
};

// Agreed drafts over kRounds rounds after the prompt.
int run_prompt(Runner& runner, const std::vector<std::int32_t>& prompt, const std::string& name,
               const char* dump) {
    const auto& loaded = runner.loaded;
    const int total    = static_cast<int>(prompt.size()) + kRounds * (kDrafts + 1);
    Cache cache(loaded, (total + kPage - 1) / kPage);
    int fed            = static_cast<int>(prompt.size());
    std::int32_t anchor = runner.forward(cache, prompt, 0, fed - 1).back();
    int agreed_total   = 0;
    for (int round = 0; round < kRounds; ++round) {
        std::vector<std::uint16_t> states;
        const bool first = round == 0 && dump != nullptr;
        const auto drafts = runner.draft(cache, anchor, fed - 1, first ? &states : nullptr);
        if (first) {
            const fs::path dir(dump);
            fs::create_directories(dir);
            write_raw(dir / (name + ".carry.bin"), states.data(), states.size() * 2);
            for (std::size_t index = 0; index < loaded.draft->layers.size(); ++index) {
                const int source   = loaded.draft->kv_source[index];
                const auto& layer  = loaded.layers[static_cast<std::size_t>(source)];
                const int begin    = layer.global_layer ? 0 : std::max(0, fed - layer.window);
                const auto values  = cache.export_layer(layer, static_cast<std::size_t>(source),
                                                        begin, fed - 1);
                write_raw(dir / (name + (layer.global_layer ? ".global.bin" : ".sliding.bin")),
                          values.data(), values.size() * 2);
            }
            const nlohmann::json meta = {{"anchor", anchor},      {"position", fed - 1},
                                         {"drafts", drafts},      {"window", loaded.layers[0].window},
                                         {"hidden", loaded.hidden}};
            std::ofstream(dir / (name + ".draft.json")) << meta.dump();
        }
        std::vector<std::int32_t> verify{anchor};
        verify.insert(verify.end(), drafts.begin(), drafts.end());
        int agreed = 0;
        const auto targets = runner.forward(cache, verify, fed, -1);
        while (agreed < kDrafts && drafts[static_cast<std::size_t>(agreed)] ==
                                       targets[static_cast<std::size_t>(agreed)]) {
            ++agreed;
        }
        if (first) {
            std::cout << name << " round 0 drafts";
            for (const auto token : drafts) { std::cout << ' ' << token; }
            std::cout << " targets";
            for (const auto token : targets) { std::cout << ' ' << token; }
            std::cout << '\n';
        }
        agreed_total += agreed;
        // Recompute the accepted column's state: the verify forward ran with carry_column -1.
        runner.forward(cache, std::vector<std::int32_t>(verify.begin(), verify.begin() + agreed + 1),
                       fed, agreed);
        fed += agreed + 1;
        anchor = targets[static_cast<std::size_t>(agreed)];
    }
    std::cout << name << ": " << prompt.size() << " prompt tokens, " << agreed_total << " of "
              << kRounds * kDrafts << " drafts agreed ("
              << static_cast<double>(agreed_total) / kRounds << " per round)\n";
    return agreed_total;
}

std::size_t workspace_for(const g4::LoadedText& loaded, int tokens) {
    std::size_t bytes = 0;
    for (const g4::Gemma4LayerWeights& layer : loaded.layers) {
        bytes = std::max(bytes, g4::gemma4_layer_workspace_bytes(layer, tokens));
    }
    if (loaded.draft != nullptr) {
        bytes = std::max(bytes, g4::gemma4_draft_workspace_bytes(*loaded.draft));
    }
    bytes += static_cast<std::size_t>(loaded.hidden) * static_cast<std::size_t>(tokens) * 2U;
    bytes += static_cast<std::size_t>(loaded.vocab) * static_cast<std::size_t>(kDrafts + 1) * 2U;
    return bytes + 64U * 1024U * 1024U;
}

// One prompt, printing the target-distribution gap of every first draft. Stops after `rounds`
// or once `token_budget` generated tokens have been emitted (0 means no token cap).
double run_margin(Runner& runner, const std::vector<std::int32_t>& prompt, int rounds,
                  int token_budget) {
    const int total = static_cast<int>(prompt.size()) + rounds * (kDrafts + 1);
    Cache cache(runner.loaded, (total + kPage - 1) / kPage);
    int fed = static_cast<int>(prompt.size());
    std::cout << "prefill " << prompt.size() << std::endl;
    std::int32_t anchor = runner.forward(cache, prompt, 0, fed - 1).back();
    std::cout << "anchor " << anchor << std::endl;
    int agreed_total    = 0;
    int first_hits      = 0;
    int generated       = 0;
    int per_pos[kDrafts]{};
    std::vector<float> column;
    const int vocab = runner.loaded.vocab;
    int round       = 0;
    for (; round < rounds && (token_budget <= 0 || generated < token_budget); ++round) {
        const auto drafts = runner.draft(cache, anchor, fed - 1, nullptr);
        std::vector<std::int32_t> verify{anchor};
        verify.insert(verify.end(), drafts.begin(), drafts.end());
        const auto targets = runner.forward(cache, verify, fed, -1, &column);
        int agreed = 0;
        while (agreed < kDrafts && drafts[static_cast<std::size_t>(agreed)] ==
                                       targets[static_cast<std::size_t>(agreed)]) {
            ++agreed;
        }
        const int draft0  = drafts[0];
        const int target0 = targets[0];
        if (draft0 < 0 || draft0 >= vocab || target0 < 0 || target0 >= vocab) {
            throw std::runtime_error("draft or target id is outside the vocab");
        }
        const float draft_logit  = column[static_cast<std::size_t>(draft0)];
        const float target_logit = column[static_cast<std::size_t>(target0)];
        int rank = 1;
        for (const float value : column) {
            if (value > draft_logit) { ++rank; }
        }
        std::cout << "round " << round << " pos " << (fed - 1) << " draft0 " << draft0
                  << " target0 " << target0 << " rank " << rank << " logit_draft " << draft_logit
                  << " logit_target " << target_logit                   << " gap " << (target_logit - draft_logit) << " agreed " << agreed << std::endl;
        if (draft0 == target0) { ++first_hits; }
        agreed_total += agreed;
        generated += agreed + 1;
        for (int index = 0; index < agreed; ++index) { ++per_pos[index]; }
        runner.forward(cache,
                       std::vector<std::int32_t>(verify.begin(), verify.begin() + agreed + 1), fed,
                       agreed);
        fed += agreed + 1;
        anchor = targets[static_cast<std::size_t>(agreed)];
    }
    const double accept = round == 0 ? 0.0 : static_cast<double>(generated) / round;
    std::cout << "margin: " << prompt.size() << " prompt tokens, " << round << " rounds, generated "
              << generated << ", first " << first_hits << '/' << round << ", agreed " << agreed_total
              << " of " << round * kDrafts << " (" << static_cast<double>(agreed_total) / round
              << " per round, accept " << accept << ") positions";
    for (const int count : per_pos) { std::cout << ' ' << count; }
    std::cout << std::endl;
    return accept;
}

// Teacher-forced drafter ranks along the greedy continuation. One line per start: six ranks of
// the true next token in the drafter's top 8 (8 means it was not there). chain5 accept is the
// mean tokens per round of the rank-0 chain, which should match a K=5 MTP run.
void run_rank(Runner& runner, const std::vector<std::int32_t>& prompt, int generate, int depth,
              int k, const fs::path& out_path) {
    if (prompt.size() < 2 || generate < 1) {
        throw std::runtime_error("rank prompt does not leave room for the draft depth");
    }
    const auto& loaded = runner.loaded;
    const int total    = static_cast<int>(prompt.size()) + generate + depth;
    Cache growing(loaded, (total + kPage - 1) / kPage);
    int fed = static_cast<int>(prompt.size());
    std::int32_t anchor = runner.forward(growing, prompt, 0, fed - 1).back();
    std::vector<std::int32_t> ids = prompt;
    for (int index = 0; index < generate + depth; ++index) {
        ids.push_back(anchor);
        anchor = runner.forward(growing, {anchor}, fed, 0).back();
        ++fed;
    }
    Cache filled(loaded, (total + kPage - 1) / kPage);
    runner.forward(filled, ids, 0, -1);
    const int prompt_len = static_cast<int>(prompt.size());
    const int hidden = loaded.hidden;
    ninfer::DeviceBuffer token_buf(4);
    ninfer::DeviceBuffer pos_buf(4);
    ninfer::DeviceBuffer top_buf(static_cast<std::size_t>(k) * 4);
    std::ofstream out(out_path);
    if (!out) { throw std::runtime_error("cannot write " + out_path.string()); }
    int starts = 0;
    int accepted = 0;
    std::vector<std::int32_t> top(static_cast<std::size_t>(k));
    for (int start = prompt_len; start + depth < static_cast<int>(ids.size()); ++start) {
        ninfer::Tensor carry(runner.carry_buf.p, ninfer::DType::BF16, {hidden, 1});
        ninfer::Tensor column(static_cast<std::byte*>(runner.hidden_buf.p) +
                                  static_cast<std::size_t>(start - 1) * hidden * 2,
                              ninfer::DType::BF16, {hidden, 1});
        ninfer::ops::rmsnorm(column,
                             ninfer::Tensor(const_cast<void*>(loaded.final_norm),
                                            ninfer::DType::BF16, {hidden}),
                             loaded.rms_eps, false, carry, runner.stream);
        int leading = 0;
        bool open    = true;
        for (int step = 0; step < depth; ++step) {
            const std::int32_t token = ids[static_cast<std::size_t>(start + step)];
            const std::int32_t position = start - 1;
            check(cudaMemcpyAsync(token_buf.p, &token, 4, cudaMemcpyHostToDevice, runner.stream),
                  "rank token");
            check(cudaMemcpyAsync(pos_buf.p, &position, 4, cudaMemcpyHostToDevice, runner.stream),
                  "rank position");
            runner.workspace.reset();
            const ninfer::Tensor token_tensor(token_buf.p, ninfer::DType::I32, {1});
            const ninfer::Tensor pos_tensor(pos_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor top_tensor(top_buf.p, ninfer::DType::I32, {k, 1});
            g4::gemma4_draft_steps(token_tensor, pos_tensor, carry, carry, top_tensor, k,
                                   loaded.token_embedding, *loaded.draft, filled.layers, runner.gemm,
                                   runner.workspace, runner.stream);
            check(cudaMemcpyAsync(top.data(), top_buf.p, top.size() * 4, cudaMemcpyDeviceToHost,
                                  runner.stream),
                  "rank top");
            check(cudaStreamSynchronize(runner.stream), "rank");
            const std::int32_t truth = ids[static_cast<std::size_t>(start + step + 1)];
            int rank = k;
            for (int index = 0; index < k; ++index) {
                if (top[static_cast<std::size_t>(index)] == truth) {
                    rank = index;
                    break;
                }
            }
            if (step != 0) { out << ' '; }
            out << rank;
            if (open && rank == 0 && step < kDrafts) { ++leading; }
            else { open = false; }
        }
        out << '\n';
        accepted += leading + 1;
        ++starts;
    }
    std::cout << "rank starts " << starts << " chain5 accept "
              << (starts == 0 ? 0.0 : static_cast<double>(accepted) / starts) << " file "
              << out_path << std::endl;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_MTP_WEIGHTS");
    const char* margin   = std::getenv("NINFER_GEMMA4_MARGIN_CORPUS");
    if (artifact == nullptr || !std::ifstream(artifact).good()) {
        std::cout << "SKIP: NINFER_GEMMA4_MTP_WEIGHTS is unset\n";
        return 77;
    }
    const char* rank_out = std::getenv("NINFER_GEMMA4_RANK_OUT");
    const char* rank_ids = std::getenv("NINFER_GEMMA4_RANK_IDS");
    if (rank_out != nullptr && rank_ids != nullptr) {
        try {
            const int prompt = std::max(2, std::atoi(std::getenv("NINFER_GEMMA4_RANK_PROMPT") != nullptr
                                                         ? std::getenv("NINFER_GEMMA4_RANK_PROMPT")
                                                         : "512"));
            const int gen = std::max(8, std::atoi(std::getenv("NINFER_GEMMA4_RANK_GEN") != nullptr
                                                      ? std::getenv("NINFER_GEMMA4_RANK_GEN")
                                                      : "64"));
            constexpr int kDepth = 6;
            constexpr int kTop   = 8;
            const auto ids =
                read_text_ids(rank_ids, static_cast<std::size_t>(prompt));
            std::cout << "rank load " << artifact << " prompt " << prompt << " gen " << gen
                      << std::endl;
            ninfer::DeviceContext device;
            auto loaded         = g4::load_text(artifact, device, true);
            cudaStream_t stream = nullptr;
            check(cudaStreamCreate(&stream), "stream");
            const int total = prompt + gen + kDepth;
            Runner runner(loaded, stream, total, workspace_for(loaded, total));
            run_rank(runner, ids, gen, kDepth, kTop, rank_out);
            cudaStreamDestroy(stream);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "EXC " << error.what() << '\n';
            return 1;
        }
    }
    if (margin != nullptr) {
        try {
            const int length = std::atoi(
                std::getenv("NINFER_GEMMA4_MARGIN_LEN") != nullptr
                    ? std::getenv("NINFER_GEMMA4_MARGIN_LEN")
                    : "4096");
            const int rounds = std::atoi(
                std::getenv("NINFER_GEMMA4_MARGIN_ROUNDS") != nullptr
                    ? std::getenv("NINFER_GEMMA4_MARGIN_ROUNDS")
                    : "32");
            if (length < 2 || rounds < 1) {
                throw std::runtime_error("margin length and rounds must be positive");
            }
            const int prompt_count = std::max(
                1, std::atoi(std::getenv("NINFER_GEMMA4_MARGIN_PROMPTS") != nullptr
                                 ? std::getenv("NINFER_GEMMA4_MARGIN_PROMPTS")
                                 : "1"));
            const int token_budget = std::atoi(
                std::getenv("NINFER_GEMMA4_MARGIN_TOKENS") != nullptr
                    ? std::getenv("NINFER_GEMMA4_MARGIN_TOKENS")
                    : "0");
            std::cout << "margin load " << artifact << " length " << length << " rounds " << rounds
                      << " prompts " << prompt_count << " tokens " << token_budget << std::endl;
            const auto all = read_text_ids(margin, static_cast<std::size_t>(length) * prompt_count);
            ninfer::DeviceContext device;
            auto loaded         = g4::load_text(artifact, device, true);
            cudaStream_t stream = nullptr;
            check(cudaStreamCreate(&stream), "stream");
            Runner runner(loaded, stream, length + rounds * (kDrafts + 1),
                          workspace_for(loaded, length));
            double accept_sum = 0.0;
            for (int index = 0; index < prompt_count; ++index) {
                const auto begin = all.begin() + static_cast<std::ptrdiff_t>(index) * length;
                std::cout << "prompt " << index << std::endl;
                accept_sum += run_margin(runner, {begin, begin + length}, rounds, token_budget);
            }
            std::cout << "mean accept " << accept_sum / prompt_count << std::endl;
            cudaStreamDestroy(stream);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "EXC " << error.what() << '\n';
            return 1;
        }
    }
    const char* fixtures = std::getenv("NINFER_GEMMA4_PARITY");
    if (fixtures == nullptr) {
        std::cout << "SKIP: NINFER_GEMMA4_PARITY or NINFER_GEMMA4_MTP_WEIGHTS is unset\n";
        return 77;
    }
    const char* dump = std::getenv("NINFER_GEMMA4_DRAFT_DUMP");
    try {
        const fs::path dir(fixtures);
        std::ifstream index_file(dir / "index.json");
        const auto index = nlohmann::json::parse(index_file);
        ninfer::DeviceContext device;
        auto loaded = g4::load_text(artifact, device, true);
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "stream");
        int agreed = 0, rounds = 0;
        {
            std::vector<std::pair<std::string, std::vector<std::int32_t>>> prompts;
            std::vector<std::int32_t> joined;
            for (const auto& entry : index) {
                const std::string name = entry.at("name").get<std::string>();
                prompts.emplace_back(name, read_ids(dir / (name + ".ids")));
                joined.insert(joined.end(), prompts.back().second.begin(),
                              prompts.back().second.end());
            }
            // Longer than the sliding window, so the sliding source layer is windowed.
            while (joined.size() < 1500) { joined.insert(joined.end(), joined.begin(), joined.end()); }
            joined.resize(1500);
            prompts.emplace_back("long", joined);
            int widest = 0;
            for (const auto& prompt : prompts) {
                widest = std::max(widest, static_cast<int>(prompt.second.size()));
            }
            Runner runner(loaded, stream, widest, 1024ull * 1024ull * 1024ull);
            for (const auto& [name, ids] : prompts) {
                agreed += run_prompt(runner, ids, name, dump);
                rounds += kRounds;
            }
        }
        cudaStreamDestroy(stream);
        const double per_round = static_cast<double>(agreed) / rounds;
        // A broken drafter agrees with almost nothing; the published assistant averages well
        // above one accepted draft per round on prose and code.
        const bool ok = per_round >= 1.0;
        std::cout << (ok ? "OK" : "FAIL") << " gemma4 draft: " << per_round
                  << " agreed drafts per round\n";
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
