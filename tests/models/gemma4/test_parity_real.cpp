// Logits, layer, and greedy parity against Hugging Face dumps. NINFER_GEMMA4_PARITY names the
// fixture directory (<name>.ids/.logits/.greedy per index.json entry); the test skips without it.
#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/load.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

constexpr int kPage = 64;

template <typename T>
std::vector<T> read_raw(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    const auto bytes = static_cast<std::size_t>(in.tellg());
    std::vector<T> values(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    return values;
}

float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fU;
    std::uint32_t mantissa = bits & 0x3ffU;
    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            out = sign | (exponent << 23) | ((mantissa & 0x3ffU) << 13);
        } else {
            out = sign;
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000U | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, 4);
    return value;
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

// Per-layer paged KV for one sequence with an identity block table.
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
        std::vector<std::int32_t> ids(static_cast<std::size_t>(pages));
        for (int i = 0; i < pages; ++i) { ids[static_cast<std::size_t>(i)] = i; }
        table_buf.copy_from_host(ids.data(), table_buf.bytes);
        table = ninfer::Tensor(table_buf.p, ninfer::DType::I32, {pages});
        const std::size_t count = loaded.layers.size();
        k_store.reserve(count);
        v_store.reserve(count);
        k_pages.reserve(count);
        v_pages.reserve(count);
        for (const auto& layer : loaded.layers) {
            const std::size_t bytes = static_cast<std::size_t>(layer.head_dim) * kPage *
                                      static_cast<std::size_t>(layer.kv_heads) * pages * 2;
            k_store.emplace_back(bytes);
            v_store.emplace_back(bytes);
            const std::initializer_list<std::int32_t> shape{layer.head_dim, kPage, layer.kv_heads,
                                                            pages};
            k_pages.emplace_back(k_store.back().p, ninfer::DType::BF16, shape);
            v_pages.emplace_back(v_store.back().p, ninfer::DType::BF16, shape);
        }
        for (std::size_t i = 0; i < count; ++i) {
            layers.push_back({&k_pages[i], &v_pages[i], &table});
        }
    }
};

struct Runner {
    g4::LoadedText& loaded;
    cudaStream_t stream;
    ninfer::ops::Bf16GemmContext gemm;
    ninfer::DeviceArena workspace;
    ninfer::DeviceBuffer ones;
    ninfer::Tensor v_ones;

    Runner(g4::LoadedText& model, cudaStream_t s)
        : loaded(model), stream(s), gemm(s), workspace(1024u * 1024u * 1024u), ones(1024),
          v_ones(nullptr, ninfer::DType::BF16, {512}) {
        std::vector<std::uint16_t> bits(512, 0x3f80);
        ones.copy_from_host(bits.data(), ones.bytes);
        v_ones = ninfer::Tensor(ones.p, ninfer::DType::BF16, {512});
    }

    // Runs `ids` at positions first.. and returns float logits [count][vocab].
    std::vector<float> forward(Cache& cache, const std::vector<int>& ids, int first) {
        const int count = static_cast<int>(ids.size());
        std::vector<std::int32_t> positions(ids.size());
        for (int i = 0; i < count; ++i) { positions[static_cast<std::size_t>(i)] = first + i; }
        ninfer::DeviceBuffer id_buf(ids.size() * 4);
        ninfer::DeviceBuffer pos_buf(ids.size() * 4);
        ninfer::DeviceBuffer hidden_buf(static_cast<std::size_t>(loaded.hidden) * count * 2);
        ninfer::DeviceBuffer logit_buf(static_cast<std::size_t>(loaded.vocab) * count * 2);
        id_buf.copy_from_host(ids.data(), id_buf.bytes);
        pos_buf.copy_from_host(positions.data(), pos_buf.bytes);
        ninfer::Tensor id_tensor(id_buf.p, ninfer::DType::I32, {count});
        ninfer::Tensor pos_tensor(pos_buf.p, ninfer::DType::I32, {count});
        ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::BF16, {loaded.hidden, count});
        ninfer::Tensor logits(logit_buf.p, ninfer::DType::BF16, {loaded.vocab, count});
        g4::gemma4_layers(id_tensor, hidden, loaded.token_embedding, loaded.vocab, loaded.layers,
                          cache.layers, pos_tensor, v_ones, gemm, workspace, stream);
        g4::gemma4_logits(hidden, logits, loaded.final_norm, loaded.output_head, loaded.softcap,
                          loaded.rms_eps, gemm, workspace, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            throw std::runtime_error("gemma4 parity: forward failed");
        }
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(loaded.vocab) * count);
        logit_buf.copy_to_host(bits.data(), bits.size() * 2);
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = ninfer::test::bf16_to_f32(bits[i]); }
        return out;
    }
};

bool is_stop(int id) { return id == 1 || id == 106 || id == 50; }

double relative_l2(const float* reference, const float* other, std::size_t count) {
    double diff = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double d = static_cast<double>(other[i]) - reference[i];
        diff += d * d;
        norm += static_cast<double>(reference[i]) * reference[i];
    }
    return std::sqrt(diff / norm);
}

// Residual stream after every layer against `<name>.hidden` ([layers+1][tokens][hidden]: the
// embedding, then each layer's output with the last one after the final norm). Prints the
// relative L2 error per layer, next to `<name>.hidden_bf16` when that file exists. Fails on a
// 10x jump between consecutive layers, or an error above 1.25x HF bf16's.
bool bisect(Runner& runner, const fs::path& dir, const std::string& name,
            const std::vector<int>& ids) {
    const fs::path path = dir / (name + ".hidden");
    if (!fs::exists(path)) { return true; }
    const auto reference = read_raw<float>(path);
    const fs::path other_path = dir / (name + ".hidden_bf16");
    const auto hf_bf16 = fs::exists(other_path) ? read_raw<float>(other_path) : std::vector<float>{};
    auto& loaded = runner.loaded;
    const int tokens = static_cast<int>(ids.size());
    const std::size_t plane = static_cast<std::size_t>(loaded.hidden) * tokens;
    if (reference.size() != plane * (loaded.layers.size() + 1)) {
        throw std::runtime_error(name + ": hidden dump size does not match");
    }
    Cache cache(loaded, (tokens + kPage - 1) / kPage);
    std::vector<std::int32_t> positions(ids.size());
    for (int i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = i; }
    ninfer::DeviceBuffer id_buf(ids.size() * 4);
    ninfer::DeviceBuffer pos_buf(ids.size() * 4);
    ninfer::DeviceBuffer hidden_buf(plane * 2);
    ninfer::DeviceBuffer normed_buf(plane * 2);
    id_buf.copy_from_host(ids.data(), id_buf.bytes);
    pos_buf.copy_from_host(positions.data(), pos_buf.bytes);
    ninfer::Tensor id_tensor(id_buf.p, ninfer::DType::I32, {tokens});
    ninfer::Tensor pos_tensor(pos_buf.p, ninfer::DType::I32, {tokens});
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::BF16, {loaded.hidden, tokens});
    ninfer::Tensor normed(normed_buf.p, ninfer::DType::BF16, {loaded.hidden, tokens});
    std::vector<std::uint16_t> bits(plane);
    std::vector<float> ours(plane);
    bool ok = true;
    double previous = 0.0;
    auto report = [&](std::size_t index, const char* label, const ninfer::DeviceBuffer& buffer) {
        if (cudaStreamSynchronize(runner.stream) != cudaSuccess) {
            throw std::runtime_error("gemma4 parity: bisect sync failed");
        }
        buffer.copy_to_host(bits.data(), plane * 2);
        for (std::size_t i = 0; i < plane; ++i) { ours[i] = ninfer::test::bf16_to_f32(bits[i]); }
        const double error = relative_l2(reference.data() + index * plane, ours.data(), plane);
        std::cout << "  " << label << ' ' << index << " ninfer " << error;
        if (previous > 0.0 && error > 10.0 * previous) { ok = false; }
        previous = error;
        if (!hf_bf16.empty()) {
            const double base = relative_l2(reference.data() + index * plane,
                                            hf_bf16.data() + index * plane, plane);
            std::cout << " hf_bf16 " << base;
            if (error > 1.25 * base + 1.0e-4) {
                std::cout << " WORSE";
                ok = false;
            }
        }
        std::cout << '\n';
    };
    for (std::size_t layer = 0; layer < loaded.layers.size(); ++layer) {
        const std::span<const g4::Gemma4LayerWeights> one(&loaded.layers[layer], 1);
        const std::span<const g4::Gemma4CacheLayer> one_cache(&cache.layers[layer], 1);
        if (layer == 0) {
            g4::gemma4_layers(id_tensor, hidden, loaded.token_embedding, loaded.vocab, {},
                              {}, pos_tensor, runner.v_ones, runner.gemm, runner.workspace,
                              runner.stream);
            report(0, "embed", hidden_buf);
        }
        auto scope = runner.workspace.scope();
        g4::gemma4_layer(hidden, one[0], pos_tensor, *one_cache[0].k_pages, *one_cache[0].v_pages,
                         *one_cache[0].block_table, runner.v_ones, runner.gemm, runner.workspace,
                         runner.stream);
        if (layer + 1 < loaded.layers.size()) { report(layer + 1, "layer", hidden_buf); }
    }
    {
        auto scope = runner.workspace.scope();
        ninfer::ops::rmsnorm(hidden, ninfer::Tensor(const_cast<void*>(loaded.final_norm),
                                                    ninfer::DType::BF16, {loaded.hidden}),
                             loaded.rms_eps, false, normed, runner.stream);
        report(loaded.layers.size(), "final", normed_buf);
    }
    return ok;
}

// Logits [tokens][vocab] stored as float32 or float16, chosen by file size.
std::vector<float> read_logits(const fs::path& path, std::size_t elements) {
    const auto bits = read_raw<std::uint16_t>(path);
    std::vector<float> values(elements);
    if (bits.size() == 2 * elements) {
        std::memcpy(values.data(), bits.data(), elements * 4);
    } else if (bits.size() == elements) {
        for (std::size_t i = 0; i < elements; ++i) { values[i] = f16_to_f32(bits[i]); }
    } else {
        throw std::runtime_error(path.string() + ": logits size does not match ids and vocab");
    }
    return values;
}

struct Agreement {
    double cosine_mean = 0.0;
    double cosine_min = 1.0;
    double top1 = 0.0;
};

Agreement agree(const std::vector<float>& reference, const std::vector<float>& other, int tokens,
                std::size_t vocab) {
    Agreement result;
    int top1 = 0;
    for (int t = 0; t < tokens; ++t) {
        const float* r = reference.data() + static_cast<std::size_t>(t) * vocab;
        const float* o = other.data() + static_cast<std::size_t>(t) * vocab;
        double dot = 0.0;
        double r_sq = 0.0;
        double o_sq = 0.0;
        for (std::size_t v = 0; v < vocab; ++v) {
            dot += static_cast<double>(r[v]) * o[v];
            r_sq += static_cast<double>(r[v]) * r[v];
            o_sq += static_cast<double>(o[v]) * o[v];
        }
        const double cosine = dot / std::sqrt(r_sq * o_sq);
        result.cosine_mean += cosine;
        result.cosine_min = std::min(result.cosine_min, cosine);
        if (std::max_element(r, r + vocab) - r == std::max_element(o, o + vocab) - o) { ++top1; }
    }
    result.cosine_mean /= tokens;
    result.top1 = static_cast<double>(top1) / tokens;
    return result;
}

std::size_t common_prefix(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
    std::size_t length = 0;
    while (length < a.size() && length < b.size() && a[length] == b[length]) { ++length; }
    return length;
}

// Gates NInfer against the reference in `dir` (ideally fp32 compute). When the HF bf16 baseline
// files (<name>.logits_bf16, .greedy_bf16, .hidden_bf16) are present, NInfer must be at least
// as close to the reference as HF bf16 is; otherwise the absolute plan thresholds apply.
int run_prompt(Runner& runner, const fs::path& dir, const std::string& name) {
    const auto ids = read_raw<std::int32_t>(dir / (name + ".ids"));
    const auto ref_greedy = read_raw<std::int32_t>(dir / (name + ".greedy"));
    const std::size_t vocab = static_cast<std::size_t>(runner.loaded.vocab);
    const int tokens = static_cast<int>(ids.size());
    const auto reference = read_logits(dir / (name + ".logits"), vocab * ids.size());
    const int pages = (tokens + static_cast<int>(ref_greedy.size()) + kPage) / kPage + 1;
    Cache cache(runner.loaded, pages);
    const std::vector<int> prompt(ids.begin(), ids.end());
    const bool layers_ok = bisect(runner, dir, name, prompt);
    const auto ours = runner.forward(cache, prompt, 0);
    const Agreement mine = agree(reference, ours, tokens, vocab);

    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) { hash = (hash ^ p[i]) * 1099511628211ULL; }
    };
    mix(ours.data(), ours.size() * 4);

    std::vector<std::int32_t> generated;
    std::vector<float> margins;
    std::vector<float> last(ours.end() - static_cast<std::ptrdiff_t>(vocab), ours.end());
    while (true) {
        const int token = argmax(last);
        float second = -INFINITY;
        for (std::size_t v = 0; v < vocab; ++v) {
            if (static_cast<int>(v) != token) { second = std::max(second, last[v]); }
        }
        margins.push_back(last[static_cast<std::size_t>(token)] - second);
        generated.push_back(token);
        if (is_stop(token) || generated.size() >= ref_greedy.size()) { break; }
        last = runner.forward(cache, {token}, tokens + static_cast<int>(generated.size()) - 1);
    }
    mix(generated.data(), generated.size() * 4);
    const std::size_t diverge = common_prefix(generated, ref_greedy);
    const bool greedy_equal = generated.size() == ref_greedy.size() && diverge == generated.size();

    std::cout << name << " tokens " << tokens << " ninfer cosine mean " << mine.cosine_mean
              << " min " << mine.cosine_min << " top1 " << mine.top1 << " greedy " << diverge
              << '/' << ref_greedy.size() << " hash " << std::hex << hash << std::dec << '\n';
    if (!greedy_equal && diverge < generated.size()) {
        std::cout << "  greedy diverges at " << diverge << ": ours " << generated[diverge]
                  << " reference " << ref_greedy[diverge] << ", our top-2 margin "
                  << margins[diverge] << '\n';
    }

    bool ok = layers_ok;
    const fs::path bf16_logits = dir / (name + ".logits_bf16");
    if (fs::exists(bf16_logits)) {
        const Agreement base = agree(reference, read_logits(bf16_logits, vocab * ids.size()),
                                     tokens, vocab);
        std::cout << "  hf_bf16 cosine mean " << base.cosine_mean << " min " << base.cosine_min
                  << " top1 " << base.top1 << '\n';
        ok = ok && mine.cosine_mean >= base.cosine_mean - 2.0e-4 && mine.top1 >= base.top1 - 0.02;
    } else {
        ok = ok && mine.cosine_mean >= 0.999 && mine.top1 >= 0.98;
    }
    std::size_t allowed = generated.size();
    const fs::path bf16_greedy = dir / (name + ".greedy_bf16");
    if (fs::exists(bf16_greedy)) {
        const std::size_t base = common_prefix(read_raw<std::int32_t>(bf16_greedy), ref_greedy);
        std::cout << "  hf_bf16 greedy " << base << '/' << ref_greedy.size() << '\n';
        allowed = std::min(allowed, base);
    }
    allowed = std::min<std::size_t>(allowed, 32);
    ok = ok && (greedy_equal || diverge >= allowed);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    const char* fixtures = std::getenv("NINFER_GEMMA4_PARITY");
    const char* artifact = std::getenv("NINFER_GEMMA4_WEIGHTS");
    if (fixtures == nullptr || artifact == nullptr || !std::ifstream(artifact).good()) {
        std::cout << "SKIP: NINFER_GEMMA4_PARITY or NINFER_GEMMA4_WEIGHTS is unset\n";
        return 77;
    }
    try {
        const fs::path dir(fixtures);
        std::ifstream index_file(dir / "index.json");
        const auto index = nlohmann::json::parse(index_file);
        ninfer::DeviceContext device;
        auto loaded = g4::load_text(artifact, device);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { return 1; }
        int failures = 0;
        {
            Runner runner(loaded, stream);
            for (const auto& entry : index) {
                failures += run_prompt(runner, dir, entry.at("name").get<std::string>());
            }
        }
        cudaStreamDestroy(stream);
        std::cout << (failures ? "FAIL" : "OK") << " gemma4 parity\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
