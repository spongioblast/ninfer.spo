#include "ops/sparse_moe/flash_next/cpu_tier.h"

#include "core/process_tuning.h"
#include "core/runtime_options.h"

#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr int kGateRows   = kCpuTierIntermediate * 2;
constexpr int kGroupsIn   = kCpuTierHidden / 16;
constexpr int kGroupsDown = kCpuTierIntermediate / 16;

int scale_at(int row, int groups, int group) {
    const int inner = row & 127;
    return ((row / 128) * (groups / 4) + (group >> 2)) * 512 + (inner & 31) * 16 + (inner >> 5) * 4 +
           (group & 3);
}

float e4m3_of(unsigned char code) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (code == 0 || code == 0x7F) { return 0.f; }
    if (exponent == 0) { return std::ldexp(static_cast<float>(mantissa), -9); }
    return std::ldexp(1.f + static_cast<float>(mantissa) / 8.f, exponent - 7);
}

float kE4m3[256];
const float kE2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};

unsigned char encode_e4m3(float value) {
    unsigned char best = 0;
    float err          = std::fabs(value);
    for (int code = 1; code < 256; ++code) {
        if (code == 0x7F) { continue; }
        const float delta = std::fabs(kE4m3[code] - value);
        if (delta < err) {
            err  = delta;
            best = static_cast<unsigned char>(code);
        }
    }
    return best;
}

int encode_e2m1(float value) {
    int best  = 0;
    float err = std::fabs(value);
    for (int code = 1; code < 16; ++code) {
        const float delta = std::fabs(value - kE2m1[code]);
        if (delta < err) {
            err  = delta;
            best = code;
        }
    }
    return best;
}

std::uint16_t f32_to_bf16(float value) {
    const auto bits         = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

struct ActInt {
    __m128i mag;
    __m128i sign;
};

__m128i nibble_index(const unsigned char* bytes) {
    const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bytes));
    const __m128i lo     = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
    const __m128i hi     = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f));
    return _mm_unpacklo_epi8(lo, hi);
}

const __m128i kMagLut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0);

ActInt decode_act(const unsigned char* bytes) {
    const __m128i index = nibble_index(bytes);
    return {_mm_shuffle_epi8(kMagLut, _mm_and_si128(index, _mm_set1_epi8(7))),
            _mm_and_si128(index, _mm_set1_epi8(8))};
}

__m128i signed_weight(const unsigned char* bytes, __m128i act_sign) {
    const __m128i index = nibble_index(bytes);
    const __m128i mag   = _mm_shuffle_epi8(kMagLut, _mm_and_si128(index, _mm_set1_epi8(7)));
    const __m128i neg =
        _mm_cmpeq_epi8(_mm_and_si128(_mm_xor_si128(index, act_sign), _mm_set1_epi8(8)), _mm_set1_epi8(8));
    return _mm_sub_epi8(_mm_xor_si128(mag, neg), neg);
}

float dot_scalar(const unsigned char* codes, const unsigned char* scales, int row, int k,
                 const unsigned char* act, const float* act_scale) {
    const int groups = k / 16;
    double acc       = 0;
    for (int group = 0; group < groups; ++group) {
        const float weight_scale = kE4m3[scales[scale_at(row, groups, group)]];
        const float act_s        = act_scale[group];
        for (int byte = 0; byte < 8; ++byte) {
            const unsigned char packed = codes[(static_cast<std::size_t>(row) * groups + group) * 8 + byte];
            const unsigned char activation = act[group * 8 + byte];
            acc += kE2m1[packed & 15] * kE2m1[activation & 15] * weight_scale * act_s;
            acc += kE2m1[packed >> 4] * kE2m1[activation >> 4] * weight_scale * act_s;
        }
    }
    return static_cast<float>(acc);
}

float dot_avx512(const unsigned char* codes, const unsigned char* scales, int row, int k, const ActInt* act,
                 const float* act_scale) {
    const int groups               = k / 16;
    const unsigned char* row_codes = codes + static_cast<std::size_t>(row) * groups * 8;
    __m512 acc                     = _mm512_setzero_ps();
    for (int group = 0; group < groups; ++group) {
        const __m512i prod = _mm512_dpbusd_epi32(
            _mm512_setzero_si512(), _mm512_zextsi128_si512(act[group].mag),
            _mm512_zextsi128_si512(signed_weight(row_codes + group * 8, act[group].sign)));
        const float scale = kE4m3[scales[scale_at(row, groups, group)]] * act_scale[group] * 0.25f;
        acc               = _mm512_fmadd_ps(_mm512_cvtepi32_ps(prod), _mm512_set1_ps(scale), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

int lanes_sum(__m256i prod) {
    alignas(32) int lanes[8] = {};
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), prod);
    int sum = 0;
    for (int lane : lanes) { sum += lane; }
    return sum;
}

float dot_avx2(const unsigned char* codes, const unsigned char* scales, int row, int k, const ActInt* act,
               const float* act_scale) {
    const int groups               = k / 16;
    const unsigned char* row_codes = codes + static_cast<std::size_t>(row) * groups * 8;
    float acc                      = 0.f;
    for (int group = 0; group < groups; ++group) {
        const __m256i prod = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(), _mm256_castsi128_si256(act[group].mag),
            _mm256_castsi128_si256(signed_weight(row_codes + group * 8, act[group].sign)));
        const float scale = kE4m3[scales[scale_at(row, groups, group)]] * act_scale[group] * 0.25f;
        acc += static_cast<float>(lanes_sum(prod)) * scale;
    }
    return acc;
}

enum class Isa { Scalar, Avx2, Avx512 };

Isa detect_isa() {
#if defined(_MSC_VER)
    int info[4] = {};
    __cpuidex(info, 7, 0);
    if ((info[2] & (1 << 11)) != 0) { return Isa::Avx512; }
    __cpuidex(info, 7, 1);
    if ((info[0] & (1 << 4)) != 0) { return Isa::Avx2; }
    return Isa::Scalar;
#else
    return Isa::Avx512;
#endif
}

struct Cold {
    int assignment = 0;
    int expert     = 0;
};

struct Pool {
    std::vector<std::vector<CpuTierExpert>> experts;
    std::vector<CpuTierLayer> layers;
    std::vector<std::vector<float>> counts;
    CpuTierDeviceView view{};
    unsigned char* host_base = nullptr;
    unsigned char* host_act  = nullptr;
    unsigned char* host_scales = nullptr;
    int* host_ids            = nullptr;
    int* host_meta           = nullptr;
    std::uint32_t* host_seq  = nullptr;
    std::uint32_t* host_done = nullptr;
    std::uint32_t* host_swap = nullptr;
    std::uint16_t* host_rows = nullptr;
    int workers              = 8;
    bool adapt               = true;
    Isa isa                  = Isa::Scalar;
    std::atomic<bool> running{false};
    std::vector<std::thread> threads;
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};
    std::atomic<int> cursor{0};
    std::vector<Cold> cold;
    std::vector<float> gate;
    std::vector<unsigned char> hidden_codes;
    std::vector<unsigned char> hidden_scales;
    std::vector<float> act_scale;
    std::vector<ActInt> act_int;
    cudaStream_t copy_stream = nullptr;
    std::vector<unsigned char> swap_temp;
    int device_id = 0;
    bool warned   = false;

    void barrier() {
        const int gen = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
            arrived.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
            return;
        }
        while (generation.load(std::memory_order_acquire) == gen) { _mm_pause(); }
    }

    float row_dot(const CpuTierExpert& expert, bool down, int row, int assignment) {
        const unsigned char* codes  = down ? expert.down_codes : expert.gate_codes;
        const unsigned char* scales = down ? expert.down_scales : expert.gate_scales;
        const int width             = down ? kCpuTierIntermediate : kCpuTierHidden;
        if (!down) {
            const auto* act = act_int.data() + static_cast<std::size_t>(assignment) * kGroupsIn;
            const auto* scale = act_scale.data() + static_cast<std::size_t>(assignment) * kGroupsIn;
            if (isa == Isa::Avx512) { return dot_avx512(codes, scales, row, width, act, scale); }
            if (isa == Isa::Avx2) { return dot_avx2(codes, scales, row, width, act, scale); }
            return dot_scalar(codes, scales, row, width,
                              host_act + static_cast<std::size_t>(assignment) * (kCpuTierHidden / 2), scale);
        }
        ActInt act[kGroupsDown];
        float scale[kGroupsDown];
        const auto* codes_h =
            hidden_codes.data() + static_cast<std::size_t>(assignment) * (kCpuTierIntermediate / 2);
        const auto* scales_h =
            hidden_scales.data() + static_cast<std::size_t>(assignment) * kGroupsDown;
        for (int group = 0; group < kGroupsDown; ++group) {
            act[group]   = decode_act(codes_h + group * 8);
            scale[group] = kE4m3[scales_h[group]];
        }
        if (isa == Isa::Avx512) { return dot_avx512(codes, scales, row, width, act, scale); }
        if (isa == Isa::Avx2) { return dot_avx2(codes, scales, row, width, act, scale); }
        return dot_scalar(codes, scales, row, width, codes_h, scale);
    }

    void quantize_at(int layer, const Cold& item) {
        const CpuTierExpert& expert = experts[layer][item.expert];
        auto* hidden = gate.data() + static_cast<std::size_t>(item.assignment) * kGateRows;
        auto* codes = hidden_codes.data() + static_cast<std::size_t>(item.assignment) * (kCpuTierIntermediate / 2);
        auto* scales = hidden_scales.data() + static_cast<std::size_t>(item.assignment) * kGroupsDown;
        const float alpha = 1.f / (expert.gate_weight_div * expert.gate_input_div);
        auto bf           = [](float value) { return bf16_to_f32(f32_to_bf16(value)); };
        for (int row = 0; row < kCpuTierIntermediate; ++row) {
            const float g = bf(hidden[row] * alpha);
            const float u = bf(hidden[kCpuTierIntermediate + row] * alpha);
            hidden[row]   = bf((g / (1.f + std::exp(-g))) * u);
        }
        for (int group = 0; group < kGroupsDown; ++group) {
            float max_abs = 0.f;
            for (int i = 0; i < 16; ++i) { max_abs = std::max(max_abs, std::fabs(hidden[group * 16 + i])); }
            const unsigned char scale = encode_e4m3(expert.down_input_div * max_abs / 6.f);
            scales[group]            = scale;
            const float decoded      = kE4m3[scale];
            for (int pair = 0; pair < 8; ++pair) {
                const auto enc = [&](float value) {
                    return encode_e2m1(decoded == 0.f ? 0.f : value * expert.down_input_div / decoded);
                };
                codes[group * 8 + pair] = static_cast<unsigned char>(
                    enc(hidden[group * 16 + pair * 2]) | (enc(hidden[group * 16 + pair * 2 + 1]) << 4));
            }
        }
    }

    void store_row(int assignment, int row, float value, float alpha) {
        host_rows[static_cast<std::size_t>(assignment) * kCpuTierHidden + row] = f32_to_bf16(value * alpha);
    }

    void prepare(int layer, int assignments) {
        cold.clear();
        for (int assignment = 0; assignment < assignments; ++assignment) {
            const int expert = host_ids[assignment];
            if (expert < 0 || expert >= kCpuTierExperts || layer < 0 ||
                layer >= static_cast<int>(experts.size())) {
                continue;
            }
            counts[layer][expert] += 1.f;
            if (!experts[layer][expert].resident) { cold.push_back({assignment, expert}); }
        }
        act_scale.resize(static_cast<std::size_t>(assignments) * kGroupsIn);
        act_int.resize(static_cast<std::size_t>(assignments) * kGroupsIn);
        for (int assignment = 0; assignment < assignments; ++assignment) {
            for (int group = 0; group < kGroupsIn; ++group) {
                const std::size_t at = static_cast<std::size_t>(assignment) * kGroupsIn + group;
                act_scale[at] = kE4m3[host_scales[at]];
                act_int[at] = decode_act(host_act + static_cast<std::size_t>(assignment) * (kCpuTierHidden / 2) +
                                         group * 8);
            }
        }
        cursor.store(0, std::memory_order_relaxed);
    }

    void exchange(int layer, int hot, int cold_expert) {
        CpuTierExpert& resident = experts[layer][hot];
        CpuTierExpert& missing  = experts[layer][cold_expert];
        auto move = [&](std::uint8_t* resident_device, const std::uint8_t* cold_host, std::size_t bytes) {
            if (cudaMemcpyAsync(swap_temp.data(), resident_device, bytes, cudaMemcpyDeviceToHost, copy_stream) !=
                    cudaSuccess ||
                cudaMemcpyAsync(resident_device, cold_host, bytes, cudaMemcpyHostToDevice, copy_stream) !=
                    cudaSuccess ||
                cudaStreamSynchronize(copy_stream) != cudaSuccess) {
                throw std::runtime_error("expert tier: adaptive swap copy failed");
            }
            std::memcpy(const_cast<std::uint8_t*>(cold_host), swap_temp.data(), bytes);
        };
        move(resident.gate_codes_device, missing.gate_codes, kGateRows * (kCpuTierHidden / 2));
        move(resident.gate_scales_device, missing.gate_scales,
             static_cast<std::size_t>((kGateRows / 128) * (kCpuTierHidden / 64) * 512));
        move(resident.down_codes_device, missing.down_codes, kCpuTierHidden * (kCpuTierIntermediate / 2));
        move(resident.down_scales_device, missing.down_scales,
             static_cast<std::size_t>((kCpuTierHidden / 128) * (kCpuTierIntermediate / 64) * 512));
        CpuTierExpert promoted = missing;
        promoted.resident            = true;
        promoted.gate_codes_device   = resident.gate_codes_device;
        promoted.gate_scales_device  = resident.gate_scales_device;
        promoted.down_codes_device   = resident.down_codes_device;
        promoted.down_scales_device  = resident.down_scales_device;
        CpuTierExpert evicted = resident;
        evicted.resident           = false;
        evicted.gate_codes         = missing.gate_codes;
        evicted.gate_scales        = missing.gate_scales;
        evicted.down_codes         = missing.down_codes;
        evicted.down_scales        = missing.down_scales;
        evicted.gate_codes_device  = missing.gate_codes_device;
        evicted.gate_scales_device = missing.gate_scales_device;
        evicted.down_codes_device  = missing.down_codes_device;
        evicted.down_scales_device = missing.down_scales_device;
        const CpuTierLayer& table = layers[layer];
        auto put = [&](const std::uint8_t** table_ptr, int expert, std::uint8_t* value) {
            if (cudaMemcpy(table_ptr + expert, &value, sizeof(value), cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("expert tier: pointer-table update failed");
            }
        };
        put(table.device_gate_codes, cold_expert, promoted.gate_codes_device);
        put(table.device_gate_scales, cold_expert, promoted.gate_scales_device);
        put(table.device_down_codes, cold_expert, promoted.down_codes_device);
        put(table.device_down_scales, cold_expert, promoted.down_scales_device);
        put(table.device_gate_codes, hot, evicted.gate_codes_device);
        put(table.device_gate_scales, hot, evicted.gate_scales_device);
        put(table.device_down_codes, hot, evicted.down_codes_device);
        put(table.device_down_scales, hot, evicted.down_scales_device);
        std::uint32_t mask[16];
        if (cudaMemcpy(mask, table.device_resident, sizeof(mask), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("expert tier: resident-mask read failed");
        }
        mask[cold_expert >> 5] |= 1u << (cold_expert & 31);
        mask[hot >> 5] &= ~(1u << (hot & 31));
        if (cudaMemcpy(table.device_resident, mask, sizeof(mask), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaStreamSynchronize(copy_stream) != cudaSuccess) {
            throw std::runtime_error("expert tier: resident-mask update failed");
        }
        experts[layer][cold_expert] = promoted;
        experts[layer][hot]         = evicted;
    }

    void maybe_swap() {
        struct Pick {
            int layer;
            int hot;
            int cold;
            float gain;
        };
        Pick best[4] = {};
        int found    = 0;
        for (int layer = 0; layer < static_cast<int>(experts.size()); ++layer) {
            int hot = -1;
            int cold_expert = -1;
            float hot_count = 1e30f;
            float cold_count = -1.f;
            for (int expert = 0; expert < kCpuTierExperts; ++expert) {
                const float count = counts[layer][expert];
                if (experts[layer][expert].resident && count < hot_count) {
                    hot_count = count;
                    hot       = expert;
                }
                if (!experts[layer][expert].resident && experts[layer][expert].gate_codes != nullptr &&
                    count > cold_count) {
                    cold_count = count;
                    cold_expert = expert;
                }
            }
            if (hot < 0 || cold_expert < 0 || cold_count <= hot_count * 1.5f) { continue; }
            const float gain = cold_count - hot_count;
            int slot         = found < 4 ? found : -1;
            if (found == 4) {
                float worst = best[0].gain;
                slot        = 0;
                for (int i = 1; i < 4; ++i) {
                    if (best[i].gain < worst) {
                        worst = best[i].gain;
                        slot  = i;
                    }
                }
                if (gain <= best[slot].gain) { slot = -1; }
            }
            if (slot >= 0) {
                best[slot] = {layer, hot, cold_expert, gain};
                if (found < 4) { ++found; }
            }
        }
        for (int i = 0; i < found; ++i) { exchange(best[i].layer, best[i].hot, best[i].cold); }
        for (auto& layer_counts : counts) {
            for (float& count : layer_counts) { count *= 0.9f; }
        }
    }

    void layer_work(int id, std::uint32_t seq) {
        const int assignments = host_meta[0];
        const int layer       = host_meta[1];
        if (id == 0) { prepare(layer, assignments); }
        barrier();
        const int gate_span = kGateRows / 32;
        const int gate_chunks = static_cast<int>(cold.size()) * gate_span;
        for (int chunk = cursor.fetch_add(1, std::memory_order_relaxed); chunk < gate_chunks;
             chunk     = cursor.fetch_add(1, std::memory_order_relaxed)) {
            const Cold item = cold[static_cast<std::size_t>(chunk / gate_span)];
            const int row0  = (chunk % gate_span) * 32;
            auto* out       = gate.data() + static_cast<std::size_t>(item.assignment) * kGateRows;
            for (int row = row0; row < row0 + 32; ++row) {
                out[row] = row_dot(experts[layer][item.expert], false, row, item.assignment);
            }
        }
        barrier();
        if (id == 0) { cursor.store(0, std::memory_order_relaxed); }
        barrier();
        for (std::size_t i = static_cast<std::size_t>(id); i < cold.size(); i += static_cast<std::size_t>(workers)) {
            quantize_at(layer, cold[i]);
        }
        barrier();
        if (id == 0) { cursor.store(0, std::memory_order_relaxed); }
        barrier();
        const int down_span   = kCpuTierHidden / 32;
        const int down_chunks = static_cast<int>(cold.size()) * down_span;
        for (int chunk = cursor.fetch_add(1, std::memory_order_relaxed); chunk < down_chunks;
             chunk     = cursor.fetch_add(1, std::memory_order_relaxed)) {
            const Cold item = cold[static_cast<std::size_t>(chunk / down_span)];
            const int row0  = (chunk % down_span) * 32;
            const CpuTierExpert& expert = experts[layer][item.expert];
            const float alpha           = 1.f / (expert.down_weight_div * expert.down_input_div);
            ActInt act[kGroupsDown];
            float scale[kGroupsDown];
            const auto* codes_h =
                hidden_codes.data() + static_cast<std::size_t>(item.assignment) * (kCpuTierIntermediate / 2);
            const auto* scales_h =
                hidden_scales.data() + static_cast<std::size_t>(item.assignment) * kGroupsDown;
            for (int group = 0; group < kGroupsDown; ++group) {
                act[group]   = decode_act(codes_h + group * 8);
                scale[group] = kE4m3[scales_h[group]];
            }
            for (int row = row0; row < row0 + 32; ++row) {
                float value = 0.f;
                if (isa == Isa::Avx512) {
                    value = dot_avx512(expert.down_codes, expert.down_scales, row, kCpuTierIntermediate, act, scale);
                } else if (isa == Isa::Avx2) {
                    value = dot_avx2(expert.down_codes, expert.down_scales, row, kCpuTierIntermediate, act, scale);
                } else {
                    value = dot_scalar(expert.down_codes, expert.down_scales, row, kCpuTierIntermediate, codes_h, scale);
                }
                store_row(item.assignment, row, value, alpha);
            }
        }
        barrier();
        if (id == 0) { publish_done(seq, layer); }
    }

    void publish_done(std::uint32_t seq, int layer) {
        const bool swap = adapt && layer + 1 == static_cast<int>(experts.size());
        if (swap) {
            *host_swap = 1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        *host_done = seq;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!swap) { return; }
        try {
            maybe_swap();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "expert tier: %s\n", error.what());
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *host_swap = 0;
    }

    void thread_main(int id) {
        ninfer::tune_current_thread(ninfer::ThreadRole::Worker);
        cudaSetDevice(device_id);
        std::uint32_t seen = 0;
        auto waited        = std::chrono::steady_clock::now();
        while (running.load(std::memory_order_acquire)) {
            const std::uint32_t now = *host_seq;
            if (now == seen) {
                if (id == 0 && *host_done != now && !warned &&
                    std::chrono::steady_clock::now() - waited > std::chrono::seconds(5)) {
                    std::fprintf(stderr, "expert tier: CPU workers stalled; releasing the GPU wait\n");
                    *host_done = now;
                    *host_swap = 0;
                    warned     = true;
                }
                _mm_pause();
                continue;
            }
            waited = std::chrono::steady_clock::now();
            std::atomic_thread_fence(std::memory_order_acquire);
            try {
                layer_work(id, now);
            } catch (const std::exception& error) {
                if (id == 0) { std::fprintf(stderr, "expert tier: %s\n", error.what()); }
                if (id == 0) { *host_done = now; }
            }
            seen = now;
        }
    }
};

Pool* g_pool = nullptr;

void check_isa(const Pool& pool) {
    const CpuTierExpert* sample = nullptr;
    for (const auto& layer : pool.experts) {
        for (const auto& expert : layer) {
            if (expert.gate_codes != nullptr) {
                sample = &expert;
                break;
            }
        }
        if (sample != nullptr) { break; }
    }
    if (sample == nullptr || pool.isa == Isa::Scalar) { return; }
    unsigned char act[kCpuTierHidden / 2];
    float scale[kGroupsIn];
    ActInt decoded[kGroupsIn];
    for (int i = 0; i < kCpuTierHidden / 2; ++i) { act[i] = static_cast<unsigned char>(i * 17); }
    for (int group = 0; group < kGroupsIn; ++group) {
        scale[group]   = kE4m3[0x38];
        decoded[group] = decode_act(act + group * 8);
    }
    const float scalar = dot_scalar(sample->gate_codes, sample->gate_scales, 3, kCpuTierHidden, act, scale);
    const float vector = pool.isa == Isa::Avx512
                             ? dot_avx512(sample->gate_codes, sample->gate_scales, 3, kCpuTierHidden, decoded, scale)
                             : dot_avx2(sample->gate_codes, sample->gate_scales, 3, kCpuTierHidden, decoded, scale);
    const float rel = std::fabs(scalar - vector) / std::max(1.f, std::fabs(scalar));
    if (rel > 1e-5f) {
        throw std::runtime_error("expert tier: CPU vector kernel does not match the scalar dot");
    }
}

}  // namespace

void cpu_tier_start(const CpuTierStart& start, CpuTierDeviceView* out) {
    if (g_pool != nullptr) { cpu_tier_stop(); }
    for (int i = 0; i < 256; ++i) { kE4m3[i] = e4m3_of(static_cast<unsigned char>(i)); }
    auto pool      = std::make_unique<Pool>();
    pool->isa      = detect_isa();
    const ninfer::RuntimeSwitches& switches = ninfer::runtime_switches();
    pool->adapt = start.adapt && switches.expert_adapt;
    pool->workers = switches.expert_cpu_threads > 0 ? switches.expert_cpu_threads : start.workers;
    if (pool->workers <= 0) { pool->workers = 8; }
    if (pool->workers > 32) { pool->workers = 32; }
    cudaGetDevice(&pool->device_id);
    pool->experts.resize(static_cast<std::size_t>(start.layer_count));
    pool->counts.resize(static_cast<std::size_t>(start.layer_count));
    pool->layers.resize(static_cast<std::size_t>(start.layer_count));
    for (int layer = 0; layer < start.layer_count; ++layer) {
        pool->layers[layer] = start.layers[layer];
        pool->experts[layer].assign(start.layers[layer].experts,
                                    start.layers[layer].experts + kCpuTierExperts);
        pool->counts[layer].assign(kCpuTierExperts, 0.f);
        for (int expert = 0; expert < kCpuTierExperts; ++expert) {
            if (pool->experts[layer][expert].resident) { pool->counts[layer][expert] = 1.f; }
        }
    }
    check_isa(*pool);
    const std::size_t bytes =
        64 + static_cast<std::size_t>(kCpuTierMaxAssign) * sizeof(int) +
        static_cast<std::size_t>(kCpuTierMaxAssign) * (kCpuTierHidden / 2) +
        static_cast<std::size_t>(kCpuTierMaxAssign) * kGroupsIn +
        static_cast<std::size_t>(kCpuTierMaxAssign) * kCpuTierHidden * sizeof(std::uint16_t);
    if (cudaHostAlloc(&pool->host_base, bytes, cudaHostAllocMapped) != cudaSuccess) {
        throw std::runtime_error("expert tier: pinning the CPU handshake failed");
    }
    void* device_base = nullptr;
    if (cudaHostGetDevicePointer(&device_base, pool->host_base, 0) != cudaSuccess) {
        cudaFreeHost(pool->host_base);
        throw std::runtime_error("expert tier: mapping the CPU handshake failed");
    }
    auto* host = pool->host_base;
    auto* dev  = static_cast<unsigned char*>(device_base);
    auto place = [&](std::size_t n, unsigned char*& cursor, unsigned char*& device_cursor) {
        unsigned char* at        = cursor;
        unsigned char* device_at = device_cursor;
        cursor += n;
        device_cursor += n;
        return std::pair{at, device_at};
    };
    unsigned char* cursor        = host;
    unsigned char* device_cursor = dev;
    auto align16                 = [](unsigned char*& a, unsigned char*& b) {
        const auto pad = (16 - (reinterpret_cast<std::uintptr_t>(a) & 15)) & 15;
        a += pad;
        b += pad;
    };
    auto seq = place(4, cursor, device_cursor);
    auto done                    = place(4, cursor, device_cursor);
    auto swapping                = place(4, cursor, device_cursor);
    auto meta                    = place(2 * sizeof(int), cursor, device_cursor);
    auto ids = place(kCpuTierMaxAssign * sizeof(int), cursor, device_cursor);
    align16(cursor, device_cursor);
    auto act = place(kCpuTierMaxAssign * (kCpuTierHidden / 2), cursor, device_cursor);
    auto scales                  = place(kCpuTierMaxAssign * kGroupsIn, cursor, device_cursor);
    auto rows                    = place(kCpuTierMaxAssign * kCpuTierHidden * sizeof(std::uint16_t), cursor,
                                         device_cursor);
    pool->host_seq    = reinterpret_cast<std::uint32_t*>(seq.first);
    pool->host_done   = reinterpret_cast<std::uint32_t*>(done.first);
    pool->host_swap   = reinterpret_cast<std::uint32_t*>(swapping.first);
    pool->host_meta   = reinterpret_cast<int*>(meta.first);
    pool->host_ids    = reinterpret_cast<int*>(ids.first);
    pool->host_act    = act.first;
    pool->host_scales = scales.first;
    pool->host_rows   = reinterpret_cast<std::uint16_t*>(rows.first);
    pool->view.seq       = reinterpret_cast<std::uint32_t*>(seq.second);
    pool->view.done      = reinterpret_cast<std::uint32_t*>(done.second);
    pool->view.swapping  = reinterpret_cast<std::uint32_t*>(swapping.second);
    pool->view.meta      = reinterpret_cast<int*>(meta.second);
    pool->view.ids       = reinterpret_cast<int*>(ids.second);
    pool->view.act_codes = act.second;
    pool->view.act_scales = scales.second;
    pool->view.rows      = rows.second;
    *pool->host_seq = *pool->host_done = *pool->host_swap = 0;
    for (int bytes_try : {512, 256, 0}) {
        if (bytes_try == 0) { break; }
        if (cudaMalloc(&pool->view.scratch, static_cast<std::size_t>(bytes_try) << 20) == cudaSuccess) {
            pool->view.scratch_slots =
                static_cast<int>((static_cast<std::size_t>(bytes_try) << 20) / kCpuTierSlotBytes);
            break;
        }
    }
    if (cudaMalloc(&pool->view.stage_flag, kCpuTierExperts * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&pool->view.stage_count, sizeof(int)) != cudaSuccess) {
        throw std::runtime_error("expert tier: staging flags failed");
    }
    if (cudaStreamCreateWithFlags(&pool->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("expert tier: copy stream failed");
    }
    pool->swap_temp.resize(4u << 20);
    pool->gate.assign(static_cast<std::size_t>(kCpuTierMaxAssign) * kGateRows, 0.f);
    pool->hidden_codes.assign(static_cast<std::size_t>(kCpuTierMaxAssign) * (kCpuTierIntermediate / 2), 0);
    pool->hidden_scales.assign(static_cast<std::size_t>(kCpuTierMaxAssign) * kGroupsDown, 0);
    pool->running.store(true);
    for (int id = 0; id < pool->workers; ++id) {
        pool->threads.emplace_back([pool_ptr = pool.get(), id] { pool_ptr->thread_main(id); });
    }
    *out   = pool->view;
    g_pool = pool.release();
    std::fprintf(stderr, "expert tier cpu: %d workers, %s, prefill scratch %d experts\n", g_pool->workers,
                 g_pool->isa == Isa::Avx512 ? "avx512" : (g_pool->isa == Isa::Avx2 ? "avx2" : "scalar"),
                 g_pool->view.scratch_slots);
}

void cpu_tier_stop() noexcept {
    if (g_pool == nullptr) { return; }
    g_pool->running.store(false, std::memory_order_release);
    for (auto& thread : g_pool->threads) {
        if (thread.joinable()) { thread.join(); }
    }
    if (g_pool->copy_stream != nullptr) { cudaStreamDestroy(g_pool->copy_stream); }
    if (g_pool->view.scratch != nullptr) { cudaFree(g_pool->view.scratch); }
    if (g_pool->view.stage_flag != nullptr) { cudaFree(g_pool->view.stage_flag); }
    if (g_pool->view.stage_count != nullptr) { cudaFree(g_pool->view.stage_count); }
    if (g_pool->host_base != nullptr) { cudaFreeHost(g_pool->host_base); }
    delete g_pool;
    g_pool = nullptr;
}

const CpuTierDeviceView* cpu_tier_device() noexcept {
    return g_pool == nullptr ? nullptr : &g_pool->view;
}

}  // namespace ninfer::ops
