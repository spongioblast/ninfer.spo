// NVFP4 Flash-Next MoE through per-expert pointer tables (--expert-vram-gib) must give the same
// bits as the contiguous bank, with some experts in VRAM and the rest in mapped pinned memory.
#include "ninfer/ops/flash_next_moe.h"
#include "ops/op_tester.h"
#include "ops/sparse_moe/flash_next/cpu_tier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden       = 2560;
constexpr int kExperts      = 512;
constexpr int kIntermediate = 640;

Weight bf16_weight(const DeviceBuffer& storage, int rows, int columns) {
    Weight out{};
    out.payload = out.qdata = storage.p;
    out.payload_bytes       = storage.bytes;
    out.qtype               = QType::BF16;
    out.layout              = QuantLayout::Contiguous;
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    out.k = out.shape[1] = out.padded_shape[1] = columns;
    out.ndim                                   = 2;
    return out;
}

DeviceBuffer random_bf16(std::mt19937& rng, std::size_t count, float scale) {
    std::normal_distribution<float> normal(0.0F, scale);
    std::vector<std::uint16_t> bits(count);
    for (auto& value : bits) { value = f32_to_bf16(normal(rng)); }
    return to_device(bits);
}

// One expert-major NVFP4 bank: codes, then scales, then one weight divisor per expert.
struct Bank {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
    std::size_t code_bytes  = 0;
    std::size_t scale_bytes = 0;
    DeviceBuffer device_codes;
    DeviceBuffer device_scales;
    DeviceBuffer weight_divisors;
    DeviceBuffer input_divisors;
};

Bank make_bank(std::mt19937& rng, int rows, int columns) {
    Bank bank;
    bank.code_bytes  = static_cast<std::size_t>(rows) * columns / 2;
    bank.scale_bytes = static_cast<std::size_t>(rows) * columns / 16;
    bank.codes.resize(bank.code_bytes * kExperts);
    bank.scales.resize(bank.scale_bytes * kExperts);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> scale(0x28, 0x38);  // E4M3 0.125 .. 1.0
    for (auto& value : bank.codes) { value = static_cast<std::uint8_t>(byte(rng)); }
    // Block scales are nonnegative E4M3, as the checkpoint stores them. The high bit set
    // would be a NaN or a sign the tensor-core scale does not use.
    for (auto& value : bank.scales) {
        value = static_cast<std::uint8_t>(scale(rng) & 0x7e);
    }
    bank.device_codes  = to_device(bank.codes);
    bank.device_scales = to_device(bank.scales);
    std::vector<float> weight(kExperts);
    std::vector<float> input(kExperts);
    std::uniform_real_distribution<float> divisor(0.5F, 2.0F);
    for (int expert = 0; expert < kExperts; ++expert) {
        weight[expert] = divisor(rng);
        input[expert]  = divisor(rng);
    }
    bank.weight_divisors = to_device(weight);
    bank.input_divisors  = to_device(input);
    return bank;
}

ops::FlashNextExpertBank contiguous(const Bank& bank, int rows, int columns) {
    return {.codes                 = bank.device_codes.p,
            .scales                = bank.device_scales.p,
            .weight_scale_divisors = static_cast<const float*>(bank.weight_divisors.p),
            .input_scale_divisors  = static_cast<const float*>(bank.input_divisors.p),
            .qtype                 = QType::NVFP4,
            .experts               = kExperts,
            .rows                  = rows,
            .columns               = columns};
}

// Odd experts in a separate VRAM buffer, even experts in mapped pinned host memory.
struct Scattered {
    DeviceBuffer hot;
    void* cold = nullptr;
    DeviceBuffer code_table;
    DeviceBuffer scale_table;

    Scattered(const Bank& bank)
        : hot(static_cast<std::size_t>(kExperts / 2) * (bank.code_bytes + bank.scale_bytes)),
          code_table(kExperts * sizeof(void*)), scale_table(kExperts * sizeof(void*)) {
        const std::size_t pair = bank.code_bytes + bank.scale_bytes;
        cuda_check(cudaMallocHost(&cold, static_cast<std::size_t>(kExperts / 2) * pair),
                   "cudaMallocHost");
        void* cold_device = nullptr;
        cuda_check(cudaHostGetDevicePointer(&cold_device, cold, 0), "cudaHostGetDevicePointer");
        std::vector<const std::uint8_t*> codes(kExperts);
        std::vector<const std::uint8_t*> scales(kExperts);
        for (int expert = 0; expert < kExperts; ++expert) {
            const std::size_t slot = static_cast<std::size_t>(expert / 2) * pair;
            const auto* src_codes  = bank.codes.data() + expert * bank.code_bytes;
            const auto* src_scales = bank.scales.data() + expert * bank.scale_bytes;
            if (expert % 2 == 1) {
                auto* base = static_cast<std::uint8_t*>(hot.p) + slot;
                cuda_check(cudaMemcpy(base, src_codes, bank.code_bytes, cudaMemcpyHostToDevice),
                           "copy codes");
                cuda_check(cudaMemcpy(base + bank.code_bytes, src_scales, bank.scale_bytes,
                                      cudaMemcpyHostToDevice),
                           "copy scales");
                codes[expert]  = base;
                scales[expert] = base + bank.code_bytes;
            } else {
                std::memcpy(static_cast<std::uint8_t*>(cold) + slot, src_codes, bank.code_bytes);
                std::memcpy(static_cast<std::uint8_t*>(cold) + slot + bank.code_bytes, src_scales,
                            bank.scale_bytes);
                codes[expert]  = static_cast<const std::uint8_t*>(cold_device) + slot;
                scales[expert] = static_cast<const std::uint8_t*>(cold_device) + slot + bank.code_bytes;
            }
        }
        code_table.copy_from_host(codes.data(), code_table.bytes);
        scale_table.copy_from_host(scales.data(), scale_table.bytes);
    }
    Scattered(const Scattered&)            = delete;
    Scattered& operator=(const Scattered&) = delete;
    ~Scattered() {
        if (cold != nullptr) { cudaFreeHost(cold); }
    }

    ops::FlashNextExpertBank apply(ops::FlashNextExpertBank bank) const {
        bank.codes              = nullptr;
        bank.scales             = nullptr;
        bank.expert_code_bases  = static_cast<const std::uint8_t* const*>(code_table.p);
        bank.expert_scale_bases = static_cast<const std::uint8_t* const*>(scale_table.p);
        return bank;
    }
};

std::vector<std::uint16_t> run(const ops::FlashNextMoeWeights& weights, const DeviceBuffer& input,
                               int tokens) {
    DeviceBuffer output(static_cast<std::size_t>(kHidden) * tokens * sizeof(std::uint16_t));
    Tensor in(input.p, DType::BF16, {kHidden, tokens});
    Tensor out(output.p, DType::BF16, {kHidden, tokens});
    WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(tokens));
    ops::flash_next_moe(in, weights, out, workspace, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kHidden) * tokens);
    output.copy_to_host(host.data(), host.size() * sizeof(std::uint16_t));
    return host;
}

int run_all() {
    std::mt19937 rng(7);
    DeviceBuffer router       = random_bf16(rng, static_cast<std::size_t>(kExperts) * kHidden, 0.05F);
    DeviceBuffer shared_gate  = random_bf16(rng, static_cast<std::size_t>(kIntermediate) * kHidden, 0.02F);
    DeviceBuffer shared_up    = random_bf16(rng, static_cast<std::size_t>(kIntermediate) * kHidden, 0.02F);
    DeviceBuffer shared_down  = random_bf16(rng, static_cast<std::size_t>(kHidden) * kIntermediate, 0.02F);
    DeviceBuffer shared_scale = random_bf16(rng, kHidden, 0.02F);
    const Bank gate_up = make_bank(rng, 2 * kIntermediate, kHidden);
    const Bank down    = make_bank(rng, kHidden, kIntermediate);
    const Scattered gate_up_tier(gate_up);
    const Scattered down_tier(down);

    ops::FlashNextMoeWeights plain{
        .router         = bf16_weight(router, kExperts, kHidden),
        .shared_gate    = bf16_weight(shared_gate, kIntermediate, kHidden),
        .shared_up      = bf16_weight(shared_up, kIntermediate, kHidden),
        .shared_down    = bf16_weight(shared_down, kHidden, kIntermediate),
        .shared_scale   = bf16_weight(shared_scale, 1, kHidden),
        .routed_gate_up = contiguous(gate_up, 2 * kIntermediate, kHidden),
        .routed_down    = contiguous(down, kHidden, kIntermediate),
    };
    ops::FlashNextMoeWeights tiered = plain;
    tiered.routed_gate_up = gate_up_tier.apply(plain.routed_gate_up);
    tiered.routed_down    = down_tier.apply(plain.routed_down);

    int failures = 0;
    // 1..16 take the decode route kernels, 17 and 64 the grouped ones, 4096 the large grouped.
    for (const int tokens : {1, 3, 5, 16, 17, 64, 4096}) {
        DeviceBuffer input = random_bf16(rng, static_cast<std::size_t>(kHidden) * tokens, 1.0F);
        const auto expected = run(plain, input, tokens);
        const auto actual   = run(tiered, input, tokens);
        std::size_t mismatches = 0;
        std::size_t nonzero    = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            mismatches += expected[i] != actual[i] ? 1U : 0U;
            nonzero += (expected[i] & 0x7FFFU) != 0 ? 1U : 0U;
        }
        if (mismatches != 0 || nonzero == 0) {
            std::cerr << "tokens " << tokens << ": " << mismatches << " differing outputs, "
                      << nonzero << " nonzero\n";
            ++failures;
        }
    }

    shared_gate.fill();
    shared_up.fill();
    shared_down.fill();
    shared_scale.fill();
    std::vector<float> gate_weight(kExperts), gate_input(kExperts), down_weight(kExperts),
        down_input(kExperts);
    gate_up.weight_divisors.copy_to_host(gate_weight.data(), gate_weight.size() * sizeof(float));
    gate_up.input_divisors.copy_to_host(gate_input.data(), gate_input.size() * sizeof(float));
    down.weight_divisors.copy_to_host(down_weight.data(), down_weight.size() * sizeof(float));
    down.input_divisors.copy_to_host(down_input.data(), down_input.size() * sizeof(float));
    std::vector<ops::CpuTierExpert> cpu_experts(kExperts);
    std::vector<std::uint32_t> mask(16, 0);
    for (int expert = 0; expert < kExperts; ++expert) {
        ops::CpuTierExpert& item = cpu_experts[expert];
        item.gate_codes = gate_up.codes.data() + static_cast<std::size_t>(expert) * gate_up.code_bytes;
        item.gate_scales = gate_up.scales.data() + static_cast<std::size_t>(expert) * gate_up.scale_bytes;
        item.down_codes = down.codes.data() + static_cast<std::size_t>(expert) * down.code_bytes;
        item.down_scales = down.scales.data() + static_cast<std::size_t>(expert) * down.scale_bytes;
        item.gate_weight_div = gate_weight[expert];
        item.gate_input_div  = gate_input[expert];
        item.down_weight_div = down_weight[expert];
        item.down_input_div  = down_input[expert];
        item.resident        = expert % 2 == 1;
        if (item.resident) { mask[expert >> 5] |= 1u << (expert & 31); }
    }
    DeviceBuffer device_mask(mask.size() * sizeof(std::uint32_t));
    device_mask.copy_from_host(mask.data(), device_mask.bytes);
    ops::CpuTierLayer layer{
        .experts         = cpu_experts.data(),
        .device_resident = static_cast<std::uint32_t*>(device_mask.p),
    };
    ops::CpuTierDeviceView view{};
    ops::cpu_tier_start({.layers = &layer, .layer_count = 1, .workers = 4, .adapt = false}, &view);
    ops::FlashNextMoeWeights mixed = tiered;
    mixed.routed_gate_up.resident_mask = static_cast<const std::uint32_t*>(device_mask.p);
    mixed.routed_down.resident_mask    = mixed.routed_gate_up.resident_mask;
    mixed.routed_gate_up.cpu_layer     = 0;
    mixed.routed_down.cpu_layer        = 0;
    for (const int tokens : {1, 64}) {
        DeviceBuffer input = random_bf16(rng, static_cast<std::size_t>(kHidden) * tokens, 1.0F);
        const auto expected = run(plain, input, tokens);
        const auto actual   = run(mixed, input, tokens);
        double max_abs = 0;
        double max_err = 0;
        std::size_t same = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const float want = bf16_to_f32(expected[i]);
            const float got  = bf16_to_f32(actual[i]);
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(want)));
            max_err = std::max(max_err, static_cast<double>(std::fabs(want - got)));
            same += expected[i] == actual[i] ? 1U : 0U;
        }
        const double rel = max_err / std::max(1e-6, max_abs);
        std::size_t over = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const float want = bf16_to_f32(expected[i]);
            const float got  = bf16_to_f32(actual[i]);
            if (std::fabs(want - got) > 1e-2f * std::max(1e-3f, std::fabs(want))) { ++over; }
        }
        std::cerr << "cpu tier tokens " << tokens << ": bitwise " << same << "/" << expected.size()
                  << " max rel " << rel << " rows over 1e-2 " << over << "\n";
        if ((tokens > 16 ? same != expected.size() : rel > 5e-2) || max_abs == 0) { ++failures; }
    }
    ops::cpu_tier_stop();
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run_all();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next MoE expert tier\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next MoE expert tier: " << error.what() << '\n';
        return 1;
    }
}
