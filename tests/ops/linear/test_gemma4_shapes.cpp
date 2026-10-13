// Gemma 4 31B NVFP4 (A16 and A4) and FP8 (A16) linear shapes against the FP64 oracle.
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr std::array kTrunk{std::pair{16384, 5376}, std::pair{18432, 5376},
                            std::pair{43008, 5376}, std::pair{5376, 8192},
                            std::pair{5376, 16384}, std::pair{5376, 21504}};

std::vector<Invocation> a16_calls() {
    std::vector<Invocation> calls;
    for (int t : {1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 25, 32, 33, 40, 48, 49, 64, 65, 96, 97, 128,
                  129, 512, 1024, 1600})
        calls.push_back({t});
    for (int t : {1, 7, 33, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    return calls;
}

std::vector<Invocation> a4_calls() {
    std::vector<Invocation> calls;
    for (int t : {1, 8, 17, 32, 33, 64, 96, 97, 128, 129, 192, 193, 384, 385, 512, 513, 1023, 1024,
                  1025, 2048})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (int t : {33, 513, 1024})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    return calls;
}

int run_gemma4_shapes() {
    int failures      = 0;
    std::uint32_t seed = 901U;
    const auto a16    = a16_calls();
    const auto a4     = a4_calls();
    for (auto [n, k] : kTrunk) {
        failures += run_shape("Gemma4 NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += run_shape("Gemma4 NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                              {n, k, seed++, Comparison::Sampled, true, a4});
        failures += run_shape("Gemma4 FP8_A16", ActivationCompute::A16, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
        failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, n, k);
    }
    failures += run_shape("Gemma4 FP8_A16 head", ActivationCompute::A16, make_fp8_weight,
                          {262144, 5376, seed++, Comparison::Sampled, true, a16});
    failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, 262144, 5376);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_gemma4_shapes();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 linear shapes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 linear shapes: " << error.what() << '\n';
        return 1;
    }
}
