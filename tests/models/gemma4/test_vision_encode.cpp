#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/execution/vision.h"
#include "models/gemma4/load.h"
#include "ninfer/ops/linear.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

template <class T>
std::vector<T> read_pod(const std::string& path) {
    const auto bytes = read_bytes(path);
    if (bytes.size() % sizeof(T) != 0) { throw std::runtime_error(path + " is truncated"); }
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

int field(const std::string& json, const char* key) {
    const std::string needle = std::string("\"") + key + "\":";
    const auto at            = json.find(needle);
    if (at == std::string::npos) { throw std::runtime_error(std::string("missing ") + key); }
    return std::stoi(json.substr(at + needle.size()));
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_VISION_WEIGHTS");
    const char* dumps    = std::getenv("NINFER_GEMMA4_VISION_DUMPS");
    if (artifact == nullptr || dumps == nullptr || !std::ifstream(artifact).good()) {
        std::cout << "SKIP: NINFER_GEMMA4_VISION_WEIGHTS or NINFER_GEMMA4_VISION_DUMPS is unset\n";
        return 77;
    }
    try {
        ninfer::DeviceContext device;
        auto loaded = ninfer::models::gemma4::load_text(artifact, device);
        if (!loaded.vision) { throw std::runtime_error("artifact has no vision component"); }
        const auto& vision = *loaded.vision;
        ninfer::ops::Bf16GemmContext gemm(device.stream);
        int failures = 0;
        for (int index = 0; index < 3; ++index) {
            const std::string stem = std::string(dumps) + "/image" + std::to_string(index);
            const auto json_bytes  = read_bytes(stem + ".json");
            const std::string json(json_bytes.begin(), json_bytes.end());
            const int pixels = field(json, "pixels");
            const int tokens = field(json, "tokens");
            const int valid  = field(json, "valid");
            const int hidden = field(json, "hidden");
            auto patches     = read_pod<std::uint16_t>(stem + ".patches");
            auto positions   = read_pod<std::int32_t>(stem + ".pos");
            auto mask        = read_pod<std::int32_t>(stem + ".mask");
            auto hf          = read_pod<float>(stem + ".tokens");
            const int pooled = tokens / (vision.pooling_kernel_size * vision.pooling_kernel_size);
            ninfer::DeviceBuffer patch_buf(patches.size() * 2);
            ninfer::DeviceBuffer pos_buf(positions.size() * 4);
            ninfer::DeviceBuffer mask_buf(mask.size() * 4);
            ninfer::DeviceBuffer out_buf(static_cast<std::size_t>(hidden) * pooled * 2);
            patch_buf.copy_from_host(patches.data(), patch_buf.bytes);
            pos_buf.copy_from_host(positions.data(), pos_buf.bytes);
            mask_buf.copy_from_host(mask.data(), mask_buf.bytes);
            ninfer::Tensor patch_t(patch_buf.p, ninfer::DType::BF16, {pixels, tokens});
            ninfer::Tensor pos_t(pos_buf.p, ninfer::DType::I32, {tokens, 2});
            ninfer::Tensor mask_t(mask_buf.p, ninfer::DType::I32, {tokens});
            ninfer::Tensor out_t(out_buf.p, ninfer::DType::BF16, {hidden, pooled});
            const std::size_t work = ninfer::models::gemma4::gemma4_vision_workspace_bytes(
                vision, tokens, pooled);
            ninfer::DeviceArena workspace(work);
            // The first call warms kernels and the GEMM routes; the second is timed end to end.
            double encode_ms = 0.0;
            for (int pass = 0; pass < 2; ++pass) {
                workspace.reset();
                const auto start = std::chrono::steady_clock::now();
                ninfer::models::gemma4::gemma4_vision_encode(patch_t, pos_t, mask_t, out_t, vision,
                                                             gemm, workspace, device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                encode_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
            }
            std::vector<std::uint16_t> encoded(static_cast<std::size_t>(hidden) * pooled);
            out_buf.copy_to_host(encoded.data(), encoded.size() * 2);
            std::uint64_t hash = 1469598103934665603ULL;
            for (const std::uint16_t bits : encoded) { hash = (hash ^ bits) * 1099511628211ULL; }
            std::cout << "image" << index << " encode " << encode_ms << " ms, hash " << std::hex
                      << hash << std::dec << '\n';
            const auto kept = ninfer::models::gemma4::gemma4_vision_kept_bins(
                positions.data(), tokens, vision.pooling_kernel_size, pooled);
            if (static_cast<int>(kept.size()) != valid) {
                std::cerr << "FAIL image" << index << " packed " << kept.size() << " want " << valid
                          << '\n';
                ++failures;
                continue;
            }
            double worst = 1.0;
            double sum   = 0.0;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int column = 0; column < valid; ++column) {
                std::vector<float> ours(static_cast<std::size_t>(hidden));
                std::vector<float> theirs(static_cast<std::size_t>(hidden));
                const int src = kept[static_cast<std::size_t>(column)];
                for (int d = 0; d < hidden; ++d) {
                    const std::uint16_t bits =
                        encoded[static_cast<std::size_t>(d) +
                                static_cast<std::size_t>(hidden) * src];
                    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
                    std::memcpy(&ours[static_cast<std::size_t>(d)], &u, 4);
                    theirs[static_cast<std::size_t>(d)] =
                        hf[static_cast<std::size_t>(d) + static_cast<std::size_t>(hidden) * column];
                    dot += static_cast<double>(ours[static_cast<std::size_t>(d)]) *
                           theirs[static_cast<std::size_t>(d)];
                    na += static_cast<double>(ours[static_cast<std::size_t>(d)]) *
                          ours[static_cast<std::size_t>(d)];
                    nb += static_cast<double>(theirs[static_cast<std::size_t>(d)]) *
                          theirs[static_cast<std::size_t>(d)];
                }
                const double token = cosine(ours, theirs);
                sum += token;
                worst = std::min(worst, token);
                if (column == 0 && index == 0) {
                    std::cout << "kept0 " << src << " ours";
                    for (int d = 0; d < 8; ++d) { std::cout << ' ' << ours[static_cast<std::size_t>(d)]; }
                    std::cout << " theirs";
                    for (int d = 0; d < 8; ++d) {
                        std::cout << ' ' << theirs[static_cast<std::size_t>(d)];
                    }
                    std::cout << '\n';
                }
            }
            const double mean = sum / static_cast<double>(valid);
            const double flat = dot / (std::sqrt(na) * std::sqrt(nb));
            std::cout << "image" << index << " mean " << mean << " flat " << flat << " worst "
                      << worst << " valid " << valid << '\n';
            if (mean < 0.999) {
                std::cerr << "FAIL image" << index << " mean " << mean << " worst " << worst
                          << '\n';
                ++failures;
            }
        }
        std::cout << (failures ? "FAIL" : "OK") << " gemma4 vision encode\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
