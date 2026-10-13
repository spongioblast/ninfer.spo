#include "core/arena.h"
#include "models/gemma4/execution/text_forward.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = 256;
constexpr int kDim = 256;
constexpr int kHeads = 2;
constexpr int kKv = 1;
constexpr int kInter = 256;
constexpr int kTokens = 2;

std::vector<float> rand_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const auto x0 = rand_bf16(static_cast<std::size_t>(kHidden) * kTokens, 3u, -0.4f, 0.4f);
    auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    };
    const auto x_bits = bits(x0);
    GuardedDeviceBuffer dx(x_bits.size() * 2);
    dx.copy_from_host(x_bits.data(), dx.bytes());
    std::vector<std::int32_t> positions{0, 1};
    std::int32_t table = 0;
    GuardedDeviceBuffer dpos(positions.size() * 4), dtable(4);
    dpos.copy_from_host(positions.data(), dpos.bytes());
    dtable.copy_from_host(&table, 4);
    const std::size_t page =
        static_cast<std::size_t>(kDim) * 64 * kKv;
    GuardedDeviceBuffer dk(page * 2), dv(page * 2);
    dk.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dk.bytes());
    dv.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dv.bytes());
    std::vector<float> ones(kDim, 1.0f);
    round_to_bf16(ones);
    const auto one_bits = bits(ones);
    GuardedDeviceBuffer dones(one_bits.size() * 2);
    dones.copy_from_host(one_bits.data(), dones.bytes());

    auto upload = [&](std::uint32_t seed, std::size_t count, float lo, float hi) {
        auto values = rand_bf16(count, seed, lo, hi);
        auto raw = bits(values);
        auto buffer = std::make_shared<GuardedDeviceBuffer>(raw.size() * 2);
        buffer->copy_from_host(raw.data(), buffer->bytes());
        return buffer;
    };
    // Keep buffers alive.
    std::vector<std::shared_ptr<GuardedDeviceBuffer>> owned;
    auto take = [&](std::uint32_t seed, int rows, int cols) {
        auto buffer = upload(seed, static_cast<std::size_t>(rows) * cols, -0.2f, 0.2f);
        owned.push_back(buffer);
        return buffer->data();
    };
    auto take_vec = [&](std::uint32_t seed, int n) {
        auto buffer = upload(seed, static_cast<std::size_t>(n), 0.5f, 1.5f);
        owned.push_back(buffer);
        return buffer->data();
    };

    using models::gemma4::bf16_matrix;
    constexpr int kQkv = (kHeads + 2 * kKv) * kDim;
    models::gemma4::Gemma4LayerWeights weights;
    weights.input_norm = take_vec(10, kHidden);
    weights.qkv = bf16_matrix(take(11, kQkv, kHidden), kQkv, kHidden);
    weights.q_norm = take_vec(14, kDim);
    weights.k_norm = take_vec(15, kDim);
    weights.o_proj = bf16_matrix(take(16, kHidden, kHeads * kDim), kHidden, kHeads * kDim);
    weights.post_attn_norm = take_vec(17, kHidden);
    weights.pre_ff_norm = take_vec(18, kHidden);
    weights.gate_up = bf16_matrix(take(19, 2 * kInter, kHidden), 2 * kInter, kHidden);
    weights.down = bf16_matrix(take(21, kHidden, kInter), kHidden, kInter);
    weights.post_ff_norm = take_vec(22, kHidden);
    weights.layer_scalar = 0.5f;
    weights.hidden = kHidden;
    weights.intermediate = kInter;
    weights.head_dim = kDim;
    weights.q_heads = kHeads;
    weights.kv_heads = kKv;
    weights.window = 1024;
    weights.rotary_dim = kDim;
    weights.rope_theta = 10000.0f;
    weights.global_layer = false;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL stream\n";
        return 1;
    }
    ops::Bf16GemmContext gemm(stream);
    DeviceArena workspace(64u * 1024u * 1024u);
    Tensor x(dx.data(), DType::BF16, {kHidden, kTokens});
    Tensor pos(dpos.data(), DType::I32, {kTokens});
    Tensor block(dtable.data(), DType::I32, {1});
    Tensor k_pages(dk.data(), DType::BF16, {kDim, 64, kKv, 1});
    Tensor v_pages(dv.data(), DType::BF16, {kDim, 64, kKv, 1});
    Tensor v_ones(dones.data(), DType::BF16, {kDim});
    models::gemma4::gemma4_layer(x, weights, pos, k_pages, v_pages, block, v_ones, gemm, workspace,
                                 stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "FAIL sync\n";
        return 1;
    }
    const auto got = from_device_bf16(dx.data(), x0.size());
    int non_finite = 0;
    double energy = 0.0;
    for (float value : got) {
        if (!std::isfinite(value)) { ++non_finite; }
        energy += static_cast<double>(value) * value;
    }
    if (non_finite != 0 || energy <= 0.0) {
        std::cerr << "FAIL gemma4 layer finite=" << non_finite << " energy=" << energy << '\n';
        cudaStreamDestroy(stream);
        return 1;
    }

    // One token per call reads q/k/v and gate/up as views of the fused output; two tokens per
    // call copy them out. Both must give the same layer output.
    dx.copy_from_host(x_bits.data(), dx.bytes());
    dk.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dk.bytes());
    dv.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dv.bytes());
    for (int token = 0; token < kTokens; ++token) {
        workspace.reset();
        Tensor column(static_cast<std::uint16_t*>(dx.data()) + token * kHidden, DType::BF16,
                      {kHidden, 1});
        Tensor one_pos(static_cast<std::int32_t*>(dpos.data()) + token, DType::I32, {1});
        models::gemma4::gemma4_layer(column, weights, one_pos, k_pages, v_pages, block, v_ones,
                                     gemm, workspace, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "FAIL sync stepwise\n";
        cudaStreamDestroy(stream);
        return 1;
    }
    const auto stepwise = from_device_bf16(dx.data(), x0.size());
    int split_mismatches = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(stepwise[i] - got[i]) > 1.0e-2f + 1.0e-2f * std::fabs(got[i])) {
            if (split_mismatches < 4) {
                std::cerr << "stepwise [" << i << "] " << stepwise[i] << " vs " << got[i] << '\n';
            }
            ++split_mismatches;
        }
    }
    if (split_mismatches != 0) {
        std::cerr << "FAIL gemma4 fused split T=1 vs T=2 " << split_mismatches << '\n';
        cudaStreamDestroy(stream);
        return 1;
    }

    auto zeros = [&](std::size_t elements) {
        auto buffer = std::make_shared<GuardedDeviceBuffer>(elements * 2);
        std::vector<std::uint16_t> raw(elements, 0);
        buffer->copy_from_host(raw.data(), buffer->bytes());
        owned.push_back(buffer);
        return buffer->data();
    };
    weights.qkv = bf16_matrix(zeros(static_cast<std::size_t>(kQkv) * kHidden), kQkv, kHidden);
    weights.o_proj = bf16_matrix(zeros(static_cast<std::size_t>(kHidden) * kHeads * kDim),
                                 kHidden, kHeads * kDim);
    weights.gate_up = bf16_matrix(zeros(static_cast<std::size_t>(2 * kInter) * kHidden),
                                  2 * kInter, kHidden);
    weights.down = bf16_matrix(zeros(static_cast<std::size_t>(kHidden) * kInter), kHidden, kInter);
    weights.layer_scalar = 0.5f;
    dx.copy_from_host(x_bits.data(), dx.bytes());
    dk.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dk.bytes());
    dv.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dv.bytes());
    workspace.reset();
    models::gemma4::gemma4_layer(x, weights, pos, k_pages, v_pages, block, v_ones, gemm, workspace,
                                 stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "FAIL sync zero\n";
        cudaStreamDestroy(stream);
        return 1;
    }
    const auto scaled = from_device_bf16(dx.data(), x0.size());
    int mismatches = 0;
    for (std::size_t i = 0; i < x0.size(); ++i) {
        const float expect = x0[i] * 0.5f;
        if (std::fabs(scaled[i] - expect) > 2.0e-3f &&
            std::fabs(scaled[i] - expect) > 2.0e-2f * std::fabs(expect)) {
            if (mismatches < 4) {
                std::cerr << "zero-proj [" << i << "] got " << scaled[i] << " expect " << expect
                          << '\n';
            }
            ++mismatches;
        }
    }
    cudaStreamDestroy(stream);
    if (mismatches != 0) {
        std::cerr << "FAIL gemma4 zero projections " << mismatches << '\n';
        return 1;
    }
    std::cout << "OK gemma4 layer energy=" << energy << '\n';
    return 0;
}
