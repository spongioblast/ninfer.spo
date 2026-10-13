#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "models/gemma4/load.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_WEIGHTS");
    const char* hf_dir = std::getenv("NINFER_GEMMA4_HF");
    const std::string tokenizer_path = hf_dir ? std::string(hf_dir) + "/tokenizer.json" : "";
    if (artifact == nullptr || !std::ifstream(artifact).good() ||
        !std::ifstream(tokenizer_path).good()) {
        std::cout << "SKIP: NINFER_GEMMA4_WEIGHTS or NINFER_GEMMA4_HF is unset\n";
        return 77;
    }
    try {
        std::ostringstream buffer;
        buffer << std::ifstream(tokenizer_path, std::ios::binary).rdbuf();
        const ninfer::models::gemma4::Tokenizer tokenizer(buffer.str());
        const auto ids_host = tokenizer.encode("The capital of France is");
        if (ids_host != std::vector<int>{818, 5279, 529, 7001, 563}) {
            std::cerr << "FAIL prompt ids\n";
            return 1;
        }
        ninfer::DeviceContext device;
        auto loaded = ninfer::models::gemma4::load_text(artifact, device);
        const int tokens = static_cast<int>(ids_host.size());
        const int hidden = loaded.hidden;
        const int vocab = loaded.vocab;
        std::vector<std::uint16_t> ones_bits(512, 0x3f80);
        ninfer::DeviceBuffer ones(ones_bits.size() * 2);
        ones.copy_from_host(ones_bits.data(), ones.bytes);
        ninfer::DeviceBuffer id_buf(static_cast<std::size_t>(tokens) * 4);
        id_buf.copy_from_host(ids_host.data(), id_buf.bytes);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (int i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = i; }
        ninfer::DeviceBuffer pos_buf(positions.size() * 4);
        pos_buf.copy_from_host(positions.data(), pos_buf.bytes);
        std::int32_t table = 0;
        ninfer::DeviceBuffer table_buf(4);
        table_buf.copy_from_host(&table, 4);

        std::vector<ninfer::DeviceBuffer> k_store;
        std::vector<ninfer::DeviceBuffer> v_store;
        k_store.reserve(loaded.layers.size());
        v_store.reserve(loaded.layers.size());
        for (const auto& layer : loaded.layers) {
            const std::size_t bytes = static_cast<std::size_t>(layer.head_dim) * 64 *
                                      static_cast<std::size_t>(layer.kv_heads) * 2;
            k_store.emplace_back(bytes);
            v_store.emplace_back(bytes);
            k_store.back().fill(0);
            v_store.back().fill(0);
        }
        std::vector<ninfer::Tensor> k_pages;
        std::vector<ninfer::Tensor> v_pages;
        std::vector<ninfer::models::gemma4::Gemma4CacheLayer> caches;
        k_pages.reserve(loaded.layers.size());
        v_pages.reserve(loaded.layers.size());
        caches.reserve(loaded.layers.size());
        ninfer::Tensor block(table_buf.p, ninfer::DType::I32, {1});
        for (std::size_t i = 0; i < loaded.layers.size(); ++i) {
            const auto& layer = loaded.layers[i];
            k_pages.emplace_back(k_store[i].p, ninfer::DType::BF16,
                                 std::initializer_list<std::int32_t>{layer.head_dim, 64,
                                                                     layer.kv_heads, 1});
            v_pages.emplace_back(v_store[i].p, ninfer::DType::BF16,
                                 std::initializer_list<std::int32_t>{layer.head_dim, 64,
                                                                     layer.kv_heads, 1});
            caches.push_back({&k_pages[i], &v_pages[i], &block});
        }

        ninfer::DeviceBuffer hidden_buf(static_cast<std::size_t>(hidden) * tokens * 2);
        ninfer::DeviceBuffer logit_buf(static_cast<std::size_t>(vocab) * tokens * 2);
        ninfer::Tensor ids(id_buf.p, ninfer::DType::I32, {tokens});
        ninfer::Tensor hidden_tensor(hidden_buf.p, ninfer::DType::BF16, {hidden, tokens});
        ninfer::Tensor logits(logit_buf.p, ninfer::DType::BF16, {vocab, tokens});
        ninfer::Tensor pos(pos_buf.p, ninfer::DType::I32, {tokens});
        ninfer::Tensor v_ones(ones.p, ninfer::DType::BF16, {512});
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { return 1; }
        ninfer::ops::Bf16GemmContext gemm(stream);
        ninfer::DeviceArena workspace(512u * 1024u * 1024u);
        ninfer::models::gemma4::gemma4_prefill(
            ids, hidden_tensor, logits, loaded.token_embedding, vocab, loaded.final_norm,
            loaded.output_head, loaded.softcap, loaded.rms_eps, loaded.layers, caches, pos, v_ones,
            gemm, workspace, stream);
        auto argmax_column = [&](ninfer::DeviceBuffer& buffer, int column) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                throw std::runtime_error("gemma4 france: sync failed");
            }
            std::vector<std::uint16_t> column_bits(static_cast<std::size_t>(vocab));
            buffer.copy_to_host(column_bits.data(), column_bits.size() * 2,
                                static_cast<std::size_t>(column) * vocab * 2);
            int chosen = 0;
            float chosen_value = -1.0e30f;
            float head[3] = {};
            for (int i = 0; i < vocab; ++i) {
                const float value = ninfer::test::bf16_to_f32(column_bits[static_cast<std::size_t>(i)]);
                if (i < 3) { head[i] = value; }
                if (value > chosen_value) {
                    chosen_value = value;
                    chosen = i;
                }
            }
            return std::tuple{chosen, chosen_value, head[0], head[1], head[2]};
        };
        const auto [best, best_value, l0, l1, l2] = argmax_column(logit_buf, tokens - 1);
        std::cout << "logits " << l0 << ' ' << l1 << ' ' << l2 << " argmax " << best << " value "
                  << best_value << '\n';
        const float hf[3] = {-6.375f, 2.546875f, -11.75f};
        int failures = 0;
        if (std::fabs(l0 - hf[0]) > 0.5f || std::fabs(l1 - hf[1]) > 0.5f ||
            std::fabs(l2 - hf[2]) > 0.5f || best != 7001) {
            ++failures;
        }

        ninfer::DeviceBuffer step_id(4);
        ninfer::DeviceBuffer step_pos(4);
        ninfer::DeviceBuffer step_hidden(static_cast<std::size_t>(hidden) * 2);
        ninfer::DeviceBuffer step_logits(static_cast<std::size_t>(vocab) * 2);
        ninfer::Tensor step_ids(step_id.p, ninfer::DType::I32, {1});
        ninfer::Tensor step_hidden_tensor(step_hidden.p, ninfer::DType::BF16, {hidden, 1});
        ninfer::Tensor step_logit_tensor(step_logits.p, ninfer::DType::BF16, {vocab, 1});
        ninfer::Tensor step_position(step_pos.p, ninfer::DType::I32, {1});
        std::vector<int> generated{best};
        int token = best;
        for (int step = 0; step < 8; ++step) {
            const std::int32_t position = tokens + step;
            step_id.copy_from_host(&token, 4);
            step_pos.copy_from_host(&position, 4);
            ninfer::models::gemma4::gemma4_prefill(
                step_ids, step_hidden_tensor, step_logit_tensor, loaded.token_embedding, vocab,
                loaded.final_norm, loaded.output_head, loaded.softcap, loaded.rms_eps, loaded.layers,
                caches, step_position, v_ones, gemm, workspace, stream);
            const auto [next, next_value, a, b, c] = argmax_column(step_logits, 0);
            (void)a;
            (void)b;
            (void)c;
            generated.push_back(next);
            std::cout << "step " << step << " id " << next << " logit " << next_value << '\n';
            token = next;
        }
        const std::vector<int> expect{7001, 563, 7001, 563, 7001, 563, 7001, 563, 7001};
        if (generated != expect) { ++failures; }
        cudaStreamDestroy(stream);
        std::cout << (failures ? "FAIL" : "OK") << " gemma4 france\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
