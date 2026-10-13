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
        if (loaded.layers.empty() || loaded.token_embedding == nullptr ||
            loaded.output_head == nullptr || loaded.final_norm == nullptr ||
            loaded.layers[0].v_proj == nullptr || !loaded.layers.back().global_layer ||
            loaded.layers.back().v_proj != nullptr) {
            std::cerr << "FAIL pointers\n";
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
