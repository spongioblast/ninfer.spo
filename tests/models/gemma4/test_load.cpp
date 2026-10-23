#include "core/device.h"
#include "models/gemma4/load.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "D:/01_PROJECTS/PICOT/tmp/ninfer/gemma4/tiny.ninfer";
    if (!std::ifstream(path).good()) {
        std::cout << "SKIP: tiny gemma artifact is absent\n";
        return 77;
    }
    try {
        ninfer::DeviceContext device;
        const auto loaded = ninfer::models::gemma4::load_text(path, device);
        if (loaded.layers.empty() || loaded.token_embedding.table.qdata == nullptr ||
            loaded.output_head.qdata == nullptr || loaded.final_norm == nullptr ||
            !loaded.layers.back().global_layer) {
            std::cerr << "FAIL pointers\n";
            return 1;
        }
        // A shared vocabulary table is one device object and the gather applies sqrt(hidden).
        const bool shared = loaded.token_embedding.table.qdata == loaded.output_head.qdata;
        const float normalizer = std::sqrt(static_cast<float>(loaded.hidden));
        if (shared ? std::fabs(loaded.token_embedding.scale - normalizer) > 0.01f * normalizer
                   : loaded.token_embedding.scale != 1.0f) {
            std::cerr << "FAIL embedding scale " << loaded.token_embedding.scale << " shared "
                      << shared << '\n';
            return 1;
        }
        // Sliding qkv carries query, key and value rows; global qkv has no value rows.
        const auto& first = loaded.layers.front();
        const auto& last  = loaded.layers.back();
        if (first.qkv.n != (first.q_heads + 2 * first.kv_heads) * first.head_dim ||
            last.qkv.n != (last.q_heads + last.kv_heads) * last.head_dim ||
            first.gate_up.n != 2 * first.intermediate || first.qkv.k != loaded.hidden) {
            std::cerr << "FAIL fused rows qkv " << first.qkv.n << ' ' << last.qkv.n << '\n';
            return 1;
        }
        if (loaded.layers.size() == 2) {
            if (loaded.hidden != 16 || loaded.vocab != 8 ||
                std::fabs(loaded.layers[0].layer_scalar - 0.25f) > 1.0e-6f) {
                std::cerr << "FAIL tiny scalar " << loaded.layers[0].layer_scalar << '\n';
                return 1;
            }
        } else if (loaded.layers.size() != 60 || loaded.hidden != 5376 || loaded.vocab != 262144) {
            std::cerr << "FAIL unexpected model " << loaded.layers.size() << ' ' << loaded.hidden
                      << '\n';
            return 1;
        }
        std::cout << "OK gemma4 load\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
