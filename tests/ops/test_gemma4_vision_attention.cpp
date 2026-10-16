#include "ninfer/ops/gemma4_vision_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim   = 72;
constexpr int kHeads = 16;
constexpr PointwiseCriterion kCriterion{2.0e-5, 8.0e-3};

std::size_t index(int token, int head, int dim) {
    return static_cast<std::size_t>(dim) +
           static_cast<std::size_t>(kDim) *
               (static_cast<std::size_t>(head) + static_cast<std::size_t>(kHeads) * token);
}

int run_case(const char* label, int tokens, const std::vector<std::int32_t>* mask) {
    const std::size_t count = static_cast<std::size_t>(kDim) * kHeads * tokens;
    std::vector<float> q(count), k(count), v(count);
    fill_uniform(q, 11u, -1.0f, 1.0f);
    fill_uniform(k, 12u, -1.0f, 1.0f);
    fill_uniform(v, 13u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    auto valid = [&](int token) {
        return mask == nullptr || (*mask)[static_cast<std::size_t>(token)] != 0;
    };
    std::vector<double> expected(count, 0.0);
    for (int token = 0; token < tokens; ++token) {
        if (!valid(token)) { continue; }
        for (int head = 0; head < kHeads; ++head) {
            double maximum = -1.0e300;
            int seen       = 0;
            for (int key = 0; key < tokens; ++key) {
                if (!valid(key)) { continue; }
                double dot = 0.0;
                for (int dim = 0; dim < kDim; ++dim) {
                    dot += static_cast<double>(q[index(token, head, dim)]) *
                           static_cast<double>(k[index(key, head, dim)]);
                }
                maximum = std::max(maximum, dot);
                ++seen;
            }
            if (seen == 0) { continue; }
            double sum = 0.0;
            for (int key = 0; key < tokens; ++key) {
                if (!valid(key)) { continue; }
                double dot = 0.0;
                for (int dim = 0; dim < kDim; ++dim) {
                    dot += static_cast<double>(q[index(token, head, dim)]) *
                           static_cast<double>(k[index(key, head, dim)]);
                }
                sum += std::exp(dot - maximum);
            }
            for (int key = 0; key < tokens; ++key) {
                if (!valid(key)) { continue; }
                double dot = 0.0;
                for (int dim = 0; dim < kDim; ++dim) {
                    dot += static_cast<double>(q[index(token, head, dim)]) *
                           static_cast<double>(k[index(key, head, dim)]);
                }
                const double weight = std::exp(dot - maximum) / sum;
                for (int dim = 0; dim < kDim; ++dim) {
                    expected[index(token, head, dim)] +=
                        weight * static_cast<double>(v[index(key, head, dim)]);
                }
            }
        }
    }

    auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    };
    const auto q_bits = bits(q);
    const auto k_bits = bits(k);
    const auto v_bits = bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * 2), dk(k_bits.size() * 2), dv(v_bits.size() * 2),
        dout(q_bits.size() * 2);
    dq.copy_from_host(q_bits.data(), dq.bytes());
    dk.copy_from_host(k_bits.data(), dk.bytes());
    dv.copy_from_host(v_bits.data(), dv.bytes());
    Tensor q_tensor(dq.data(), DType::BF16, {kDim, kHeads, tokens});
    Tensor k_tensor(dk.data(), DType::BF16, {kDim, kHeads, tokens});
    Tensor v_tensor(dv.data(), DType::BF16, {kDim, kHeads, tokens});
    Tensor out_tensor(dout.data(), DType::BF16, {kDim, kHeads, tokens});
    GuardedDeviceBuffer dmask(mask == nullptr ? 4 : mask->size() * 4);
    const Tensor* mask_ptr = nullptr;
    Tensor mask_tensor;
    if (mask != nullptr) {
        dmask.copy_from_host(mask->data(), dmask.bytes());
        mask_tensor = Tensor(dmask.data(), DType::I32, {tokens});
        mask_ptr    = &mask_tensor;
    }
    ops::gemma4_vision_attention(q_tensor, k_tensor, v_tensor, mask_ptr, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(dout.data(), count), expected,
                                    kCriterion);
    failures += dq.verify_guards("q");
    failures += dk.verify_guards("k");
    failures += dv.verify_guards("v");
    failures += dout.verify_guards("out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case("vision attn T=1", 1, nullptr);
    failures += run_case("vision attn T=8", 8, nullptr);
    failures += run_case("vision attn T=17", 17, nullptr);
    std::vector<std::int32_t> mask(8, 1);
    mask[0] = 0;
    mask[3] = 0;
    mask[7] = 0;
    failures += run_case("vision attn T=8 masked", 8, &mask);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_vision_attention\n";
    return failures ? 1 : 0;
}
