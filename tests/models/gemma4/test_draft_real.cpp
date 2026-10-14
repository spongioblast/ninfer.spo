// MTP assistant on real weights. NINFER_GEMMA4_MTP_WEIGHTS names an artifact with the mtp
// component and NINFER_GEMMA4_PARITY the prompt fixtures (<name>.ids per index.json entry); the
// test skips without them. After each prompt it runs greedy rounds of kDrafts drafts verified by
// one target forward and reports how many drafts the target agrees with. With
// NINFER_GEMMA4_DRAFT_DUMP it also writes the first round's drafter inputs and outputs:
// <name>.draft.json, <name>.carry.bin (BF16 [kDrafts+1][hidden], the carried state before every
// step), <name>.sliding.bin and <name>.global.bin (BF16 K then V as [kv][positions][D] of the
// assistant's KV source layers, the sliding one over its last window positions).
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

    Runner(g4::LoadedText& model, cudaStream_t s, int max_tokens)
        : loaded(model), stream(s), gemm(s), workspace(1024u * 1024u * 1024u),
          hidden_buf(static_cast<std::size_t>(model.hidden) * max_tokens * 2),
          carry_buf(static_cast<std::size_t>(model.hidden) * 2), slots_buf((2 + kDrafts) * 4) {}

    // Runs `ids` at positions first.. and returns the argmax of every column. With `carry_column`
    // the final-normed hidden state of that column lands in carry_buf.
    std::vector<std::int32_t> forward(Cache& cache, const std::vector<std::int32_t>& ids,
                                      int first, int carry_column) {
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
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(scored));
        check(cudaMemcpyAsync(tokens.data(), out.data, tokens.size() * 4, cudaMemcpyDeviceToHost,
                              stream), "argmax");
        check(cudaStreamSynchronize(stream), "forward");
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

} // namespace

int main() {
    const char* fixtures = std::getenv("NINFER_GEMMA4_PARITY");
    const char* artifact = std::getenv("NINFER_GEMMA4_MTP_WEIGHTS");
    if (fixtures == nullptr || artifact == nullptr || !std::ifstream(artifact).good()) {
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
            Runner runner(loaded, stream, widest);
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
