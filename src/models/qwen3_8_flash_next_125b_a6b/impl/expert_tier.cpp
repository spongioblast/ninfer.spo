#include "models/qwen3_8_flash_next_125b_a6b/impl/expert_tier.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/runtime_options.h"
#include "core/weight_view.h"
#include "ops/sparse_moe/flash_next/cpu_tier.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {
namespace {

constexpr int kExperts = 512;

std::size_t code_bytes(const ops::FlashNextExpertBank& bank) {
    return static_cast<std::size_t>(bank.rows) * static_cast<std::size_t>(bank.columns) / 2;
}

std::size_t scale_bytes(const ops::FlashNextExpertBank& bank) {
    return static_cast<std::size_t>(bank.rows) * static_cast<std::size_t>(bank.columns) / 16;
}

std::size_t align256(std::size_t bytes) { return (bytes + 255U) & ~std::size_t{255}; }

std::vector<std::pair<int, int>> read_profile(const std::string& path) {
    std::ifstream in(path);
    if (!in) { return {}; }
    std::vector<int> numbers;
    int sign = 1;
    int value = 0;
    bool in_number = false;
    char ch = 0;
    while (in.get(ch)) {
        if (ch == '-') {
            sign = -1;
        } else if (ch >= '0' && ch <= '9') {
            in_number = true;
            value = value * 10 + (ch - '0');
        } else if (in_number) {
            numbers.push_back(sign * value);
            sign = 1;
            value = 0;
            in_number = false;
        }
    }
    if (in_number) { numbers.push_back(sign * value); }
    std::vector<std::pair<int, int>> pairs;
    for (std::size_t i = 0; i + 1 < numbers.size(); i += 2) {
        pairs.emplace_back(numbers[i], numbers[i + 1]);
    }
    return pairs;
}

ops::FlashNextMoeWeights* layer_moe(RuntimeModelView& runtime, const BindingPlan& plan,
                                    std::size_t layer, int& full_index, int& gdn_index) {
    if (plan.text_layers[layer].is_full_attention) {
        return &runtime.full_layers.at(static_cast<std::size_t>(full_index++)).post_mixer;
    }
    return &runtime.gdn_layers.at(static_cast<std::size_t>(gdn_index++)).post_mixer;
}

} // namespace

struct ExpertTierKeep::Impl {
    std::vector<ninfer::DeviceBuffer> device;
    std::optional<ninfer::PinnedHostBuffer> cold;
};

ExpertTierKeep::ExpertTierKeep() = default;
ExpertTierKeep::~ExpertTierKeep() {
    if (impl) { ops::cpu_tier_stop(); }
}
ExpertTierKeep::ExpertTierKeep(ExpertTierKeep&&) noexcept = default;
ExpertTierKeep& ExpertTierKeep::operator=(ExpertTierKeep&&) noexcept = default;

void apply_expert_tier(RuntimeModelView& runtime, const BindingPlan& plan, ExpertTierKeep& keep) {
    const int gib = runtime_switches().expert_vram_gib;
    if (gib <= 0) { return; }

    int full_index = 0;
    int gdn_index = 0;
    std::array<ops::FlashNextMoeWeights*, kTextLayers> layers{};
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        layers[layer] = layer_moe(runtime, plan, layer, full_index, gdn_index);
        if (layers[layer]->routed_gate_up.qtype != QType::NVFP4) {
            throw std::runtime_error(
                "expert tier needs the NVFP4 routed experts; this artifact stores another format");
        }
    }

    const auto& sample = layers[0]->routed_gate_up;
    const auto& down0 = layers[0]->routed_down;
    const std::size_t pair_bytes =
        code_bytes(sample) + scale_bytes(sample) + code_bytes(down0) + scale_bytes(down0);
    const std::size_t budget = static_cast<std::size_t>(gib) << 30U;
    const int slots = static_cast<int>(budget / pair_bytes);
    if (slots < kTextLayers || slots >= static_cast<int>(kTextLayers) * kExperts) {
        throw std::runtime_error(
            "--expert-vram-gib leaves no VRAM expert slots, or it already holds every expert");
    }

    std::vector<std::array<char, kExperts>> hot(kTextLayers);
    int placed = 0;
    const auto profile = read_profile(runtime_switches().expert_profile);
    const bool have_profile = !profile.empty();
    auto take = [&](int layer, int expert) {
        if (placed >= slots || layer < 0 || layer >= static_cast<int>(kTextLayers) ||
            expert < 0 || expert >= kExperts || hot[static_cast<std::size_t>(layer)][expert]) {
            return;
        }
        hot[static_cast<std::size_t>(layer)][expert] = 1;
        ++placed;
    };
    if (have_profile) {
        for (const auto& [layer, expert] : profile) { take(layer, expert); }
    } else {
        for (int expert = 0; expert < kExperts && placed < slots; ++expert) {
            for (int layer = 0; layer < static_cast<int>(kTextLayers) && placed < slots; ++layer) {
                take(layer, expert);
            }
        }
    }
    for (int layer = 0; layer < static_cast<int>(kTextLayers) && placed < slots; ++layer) {
        for (int expert = 0; expert < kExperts && placed < slots; ++expert) { take(layer, expert); }
    }

    std::size_t hot_bytes = 0;
    std::size_t cold_bytes = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        for (int expert = 0; expert < kExperts; ++expert) {
            std::size_t& used = hot[layer][expert] ? hot_bytes : cold_bytes;
            used = align256(used + code_bytes(layers[layer]->routed_gate_up));
            used = align256(used + scale_bytes(layers[layer]->routed_gate_up));
            used = align256(used + code_bytes(layers[layer]->routed_down));
            used = align256(used + scale_bytes(layers[layer]->routed_down));
        }
    }
    keep.impl = std::make_unique<ExpertTierKeep::Impl>();
    try {
        keep.impl->cold.emplace(cold_bytes);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("expert tier: pinning ") +
            std::to_string(cold_bytes >> 30U) +
            " GiB of cold experts failed (Windows caps pinned memory near RAM/2 - 8 GiB; a "
            "larger --expert-vram-gib pins less): " + error.what());
    }
    auto* cold_host = static_cast<std::uint8_t*>(keep.impl->cold->data());
    void* cold_device_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&cold_device_ptr, cold_host, 0));
    auto* cold_device = static_cast<std::uint8_t*>(cold_device_ptr);

    keep.impl->device.emplace_back(hot_bytes);
    auto* hot_base = static_cast<std::uint8_t*>(keep.impl->device.front().p);
    std::size_t hot_used = 0;
    std::size_t cold_used = 0;
    auto bump = [](std::size_t& used, std::size_t bytes) {
        const std::size_t at = align256(used);
        used = at + bytes;
        return at;
    };

    struct Bytes {
        const std::uint8_t* host = nullptr;
        std::uint8_t* device     = nullptr;
    };
    const std::size_t slots_n = kTextLayers * static_cast<std::size_t>(kExperts);
    std::vector<Bytes> gate_code(slots_n), gate_scale(slots_n), down_code(slots_n), down_scale(slots_n);
    std::vector<float> gate_div(slots_n, 1.f), down_div(slots_n, 1.f);
    std::vector<const std::uint8_t**> gate_code_table(kTextLayers), gate_scale_table(kTextLayers),
        down_code_table(kTextLayers), down_scale_table(kTextLayers);
    std::vector<std::byte> staging;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        ops::FlashNextMoeWeights& moe = *layers[layer];
        const MoePlan& moe_plan = plan.text_layers[layer].moe;
        if (!moe_plan.gate_file || !moe_plan.down_file) {
            throw std::logic_error("expert tier: expert banks were not mapped");
        }
        ops::FlashNextExpertBank* banks[2] = {&moe.routed_gate_up, &moe.routed_down};
        const artifact::MappedRange* files[2] = {&*moe_plan.gate_file, &*moe_plan.down_file};
        for (int which = 0; which < 2; ++which) {
            ops::FlashNextExpertBank* bank = banks[which];
            const artifact::MappedRange& file = *files[which];
            const std::uint64_t shape[] = {kExperts, static_cast<std::uint64_t>(bank->rows),
                                           static_cast<std::uint64_t>(bank->columns)};
            const auto geometry =
                weight_geometry(QType::NVFP4, QuantLayout::ExpertBlockScaleK16M128x4, shape);
            std::vector<const std::uint8_t*> code_ptrs(kExperts);
            std::vector<const std::uint8_t*> scale_ptrs(kExperts);
            const std::size_t codes_n = code_bytes(*bank);
            const std::size_t scales_n = scale_bytes(*bank);
            for (int expert = 0; expert < kExperts; ++expert) {
                const std::uint64_t code_from = static_cast<std::uint64_t>(expert) * codes_n;
                const std::uint64_t scale_from =
                    geometry.scale_offset + static_cast<std::uint64_t>(expert) * scales_n;
                if (hot[layer][static_cast<std::size_t>(expert)]) {
                    const std::size_t code_at = bump(hot_used, codes_n);
                    const std::size_t scale_at = bump(hot_used, scales_n);
                    staging.resize(codes_n + scales_n);
                    file.copy(code_from, std::span(staging).first(codes_n));
                    file.copy(scale_from, std::span(staging).subspan(codes_n));
                    CUDA_CHECK(cudaMemcpy(hot_base + code_at, staging.data(), codes_n,
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(hot_base + scale_at, staging.data() + codes_n, scales_n,
                                          cudaMemcpyHostToDevice));
                    code_ptrs[static_cast<std::size_t>(expert)] = hot_base + code_at;
                    scale_ptrs[static_cast<std::size_t>(expert)] = hot_base + scale_at;
                    const std::size_t at = layer * static_cast<std::size_t>(kExperts) + expert;
                    (which == 0 ? gate_code : down_code)[at] = {nullptr, hot_base + code_at};
                    (which == 0 ? gate_scale : down_scale)[at] = {nullptr, hot_base + scale_at};
                } else {
                    const std::size_t code_at = bump(cold_used, codes_n);
                    const std::size_t scale_at = bump(cold_used, scales_n);
                    file.copy(code_from,
                              std::span(reinterpret_cast<std::byte*>(cold_host + code_at), codes_n));
                    file.copy(scale_from, std::span(reinterpret_cast<std::byte*>(cold_host + scale_at),
                                                    scales_n));
                    code_ptrs[static_cast<std::size_t>(expert)] = cold_device + code_at;
                    scale_ptrs[static_cast<std::size_t>(expert)] = cold_device + scale_at;
                    const std::size_t at = layer * static_cast<std::size_t>(kExperts) + expert;
                    (which == 0 ? gate_code : down_code)[at] = {cold_host + code_at, cold_device + code_at};
                    (which == 0 ? gate_scale : down_scale)[at] = {cold_host + scale_at, cold_device + scale_at};
                }
            }
            std::vector<float> weight_divisors(kExperts);
            file.copy(geometry.divisor_offset,
                      std::as_writable_bytes(std::span(weight_divisors)));
            auto& div = which == 0 ? gate_div : down_div;
            for (int expert = 0; expert < kExperts; ++expert) {
                div[layer * static_cast<std::size_t>(kExperts) + expert] = weight_divisors[expert];
            }
            ninfer::DeviceBuffer code_table(kExperts * sizeof(const std::uint8_t*));
            ninfer::DeviceBuffer scale_table(kExperts * sizeof(const std::uint8_t*));
            code_table.copy_from_host(code_ptrs.data(), code_table.bytes);
            scale_table.copy_from_host(scale_ptrs.data(), scale_table.bytes);
            bank->expert_code_bases = static_cast<const std::uint8_t* const*>(code_table.p);
            bank->expert_scale_bases = static_cast<const std::uint8_t* const*>(scale_table.p);
            bank->codes = nullptr;
            bank->scales = nullptr;
            ninfer::DeviceBuffer divisors(static_cast<std::size_t>(kExperts) * sizeof(float));
            divisors.copy_from_host(weight_divisors.data(), divisors.bytes);
            bank->weight_scale_divisors = static_cast<const float*>(divisors.p);
            auto& code_table_out = which == 0 ? gate_code_table : down_code_table;
            auto& scale_table_out = which == 0 ? gate_scale_table : down_scale_table;
            code_table_out[layer] = const_cast<const std::uint8_t**>(
                static_cast<const std::uint8_t* const*>(code_table.p));
            scale_table_out[layer] = const_cast<const std::uint8_t**>(
                static_cast<const std::uint8_t* const*>(scale_table.p));
            keep.impl->device.push_back(std::move(code_table));
            keep.impl->device.push_back(std::move(scale_table));
            keep.impl->device.push_back(std::move(divisors));
        }
    }

    std::vector<ops::CpuTierExpert> cpu_experts(slots_n);
    std::vector<ops::CpuTierLayer> cpu_layers(kTextLayers);
    ninfer::DeviceBuffer resident_masks(kTextLayers * 16 * sizeof(std::uint32_t));
    std::vector<std::uint32_t> mask_host(kTextLayers * 16, 0);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        std::vector<float> gate_input(kExperts), down_input(kExperts);
        CUDA_CHECK(cudaMemcpy(gate_input.data(), layers[layer]->routed_gate_up.input_scale_divisors,
                              gate_input.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(down_input.data(), layers[layer]->routed_down.input_scale_divisors,
                              down_input.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int expert = 0; expert < kExperts; ++expert) {
            const std::size_t at = layer * static_cast<std::size_t>(kExperts) + expert;
            ops::CpuTierExpert& item = cpu_experts[at];
            item.gate_codes          = gate_code[at].host;
            item.gate_scales         = gate_scale[at].host;
            item.down_codes          = down_code[at].host;
            item.down_scales         = down_scale[at].host;
            item.gate_codes_device   = gate_code[at].device;
            item.gate_scales_device  = gate_scale[at].device;
            item.down_codes_device   = down_code[at].device;
            item.down_scales_device  = down_scale[at].device;
            item.gate_weight_div     = gate_div[at];
            item.down_weight_div     = down_div[at];
            item.gate_input_div      = gate_input[expert];
            item.down_input_div      = down_input[expert];
            item.resident            = hot[layer][static_cast<std::size_t>(expert)] != 0;
            if (item.resident) { mask_host[layer * 16 + static_cast<std::size_t>(expert >> 5)] |=
                                     1u << (expert & 31); }
        }
        auto* mask = static_cast<std::uint32_t*>(resident_masks.p) + layer * 16;
        layers[layer]->routed_gate_up.resident_mask = mask;
        layers[layer]->routed_down.resident_mask    = mask;
        layers[layer]->routed_gate_up.cpu_layer     = static_cast<std::int32_t>(layer);
        layers[layer]->routed_down.cpu_layer        = static_cast<std::int32_t>(layer);
        cpu_layers[layer] = {
            .experts                = cpu_experts.data() + layer * static_cast<std::size_t>(kExperts),
            .device_resident        = mask,
            .device_gate_codes      = gate_code_table[layer],
            .device_gate_scales     = gate_scale_table[layer],
            .device_down_codes      = down_code_table[layer],
            .device_down_scales     = down_scale_table[layer],
        };
    }
    CUDA_CHECK(cudaMemcpy(resident_masks.p, mask_host.data(), resident_masks.bytes,
                          cudaMemcpyHostToDevice));
    ops::CpuTierDeviceView unused{};
    ops::cpu_tier_start({.layers = cpu_layers.data(), .layer_count = static_cast<int>(kTextLayers)},
                        &unused);
    keep.impl->device.push_back(std::move(resident_masks));

    std::fprintf(stderr,
                 "expert tier: vram budget %d GiB, %d slots of %d experts, pinned cold %.2f GiB, "
                 "profile %s\n",
                 gib, placed, static_cast<int>(kTextLayers) * kExperts,
                 static_cast<double>(cold_bytes) / (1024.0 * 1024.0 * 1024.0),
                 have_profile ? runtime_switches().expert_profile.c_str() : "expert-id order");
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
