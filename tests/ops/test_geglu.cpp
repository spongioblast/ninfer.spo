#include "ninfer/ops/geglu.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion kCriterion{2.0e-6, 4.05e-3};

double gelu_tanh(double z) {
    constexpr double c = 0.79788456080286535587989211986876;
    return 0.5 * z * (1.0 + std::tanh(c * (z + 0.044715 * z * z * z)));
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -6.0f, 6.0f);
    fill_uniform(up, seed + 1, -6.0f, 6.0f);
    round_to_bf16(gate);
    round_to_bf16(up);
    std::vector<double> expected(count);
    for (std::size_t i = 0; i < count; ++i) { expected[i] = gelu_tanh(gate[i]) * up[i]; }

    std::vector<std::uint16_t> gate_bits(count), up_bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        gate_bits[i] = f32_to_bf16(gate[i]);
        up_bits[i]   = f32_to_bf16(up[i]);
    }
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::geglu(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    kCriterion);
    failures += device_gate.verify_guards("geglu gate");
    failures += device_up.verify_guards("geglu up");
    failures += device_out.verify_guards("geglu out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case("geglu [21504,8]", 21504, 8, 41u);
    failures += run_case("geglu odd", 1, 7, 42u);
    std::cout << (failures ? "FAIL" : "OK") << " geglu\n";
    return failures ? 1 : 0;
}
