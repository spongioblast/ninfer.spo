#include "core/arena.h"
#include "models/gemma4/execution/text_forward.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    constexpr int kHidden = 256;
    constexpr int kVocab  = 4;
    constexpr int kHeads  = 2;
    constexpr int kKv     = 1;
    constexpr int kDim    = 256;
    constexpr int kInter  = 256;
    // Large enough for the widest zero matrix, the fused qkv.
    const std::size_t zero_count = static_cast<std::size_t>(kHeads + 2 * kKv) * kDim * kHidden;
    std::vector<std::uint16_t> zeros(zero_count, 0);
    std::vector<float> ones_f(kHidden, 1.0f);
    round_to_bf16(ones_f);
    std::vector<std::uint16_t> ones(kHidden);
    for (int i = 0; i < kHidden; ++i) {
        ones[static_cast<std::size_t>(i)] = f32_to_bf16(ones_f[i]);
    }
    std::vector<std::uint16_t> embed(static_cast<std::size_t>(kVocab) * kHidden, 0);
    for (int i = 0; i < kHidden; ++i) { embed[static_cast<std::size_t>(i)] = ones[0]; }
    std::vector<std::uint16_t> head(static_cast<std::size_t>(kVocab) * kHidden, 0);
    for (int i = 0; i < kHidden; ++i) { head[static_cast<std::size_t>(i)] = ones[0]; }

    GuardedDeviceBuffer dzero(zeros.size() * 2), done(ones.size() * 2), dembed(embed.size() * 2),
        dhead(head.size() * 2), dhidden(static_cast<std::size_t>(kHidden) * 2),
        dlogits(static_cast<std::size_t>(kVocab) * 2);
    dzero.copy_from_host(zeros.data(), dzero.bytes());
    done.copy_from_host(ones.data(), done.bytes());
    dembed.copy_from_host(embed.data(), dembed.bytes());
    dhead.copy_from_host(head.data(), dhead.bytes());
    std::int32_t id       = 0;
    std::int32_t position = 0;
    std::int32_t table    = 0;
    GuardedDeviceBuffer did(4), dpos(4), dtable(4);
    did.copy_from_host(&id, 4);
    dpos.copy_from_host(&position, 4);
    dtable.copy_from_host(&table, 4);
    const std::size_t page = static_cast<std::size_t>(kDim) * 64 * kKv;
    GuardedDeviceBuffer dk(page * 2), dv(page * 2);
    dk.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dk.bytes());
    dv.copy_from_host(std::vector<std::uint16_t>(page, 0).data(), dv.bytes());

    using models::gemma4::bf16_matrix;
    models::gemma4::Gemma4LayerWeights layer;
    layer.input_norm     = done.data();
    layer.qkv            = bf16_matrix(dzero.data(), (kHeads + 2 * kKv) * kDim, kHidden);
    layer.q_norm         = done.data();
    layer.k_norm         = done.data();
    layer.o_proj         = bf16_matrix(dzero.data(), kHidden, kHeads * kDim);
    layer.post_attn_norm = done.data();
    layer.pre_ff_norm    = done.data();
    layer.gate_up        = bf16_matrix(dzero.data(), 2 * kInter, kHidden);
    layer.down           = bf16_matrix(dzero.data(), kHidden, kInter);
    layer.post_ff_norm   = done.data();
    layer.layer_scalar   = 0.5f;
    layer.hidden         = kHidden;
    layer.intermediate   = kInter;
    layer.head_dim       = kDim;
    layer.q_heads        = kHeads;
    layer.kv_heads       = kKv;
    layer.window         = 1024;
    layer.rotary_dim     = kDim;
    layer.rope_theta     = 10000.0f;
    layer.global_layer   = false;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) { return 1; }
    ops::Bf16GemmContext gemm(stream);
    DeviceArena workspace(64u * 1024u * 1024u);
    Tensor ids(did.data(), DType::I32, {1});
    Tensor hidden(dhidden.data(), DType::BF16, {kHidden, 1});
    Tensor logits(dlogits.data(), DType::BF16, {kVocab, 1});
    Tensor pos(dpos.data(), DType::I32, {1});
    Tensor block(dtable.data(), DType::I32, {1});
    Tensor k_pages(dk.data(), DType::BF16, {kDim, 64, kKv, 1});
    Tensor v_pages(dv.data(), DType::BF16, {kDim, 64, kKv, 1});
    models::gemma4::Gemma4CacheLayer cache{&k_pages, &v_pages, &block};
    const models::gemma4::Gemma4LayerWeights layers[] = {layer};
    const models::gemma4::Gemma4CacheLayer caches[]   = {cache};
    models::gemma4::gemma4_prefill(ids, hidden, logits, bf16_matrix(dembed.data(), kVocab, kHidden),
                                   done.data(), bf16_matrix(dhead.data(), kVocab, kHidden), 30.0f,
                                   1.0e-6f, layers, caches, pos, gemm, workspace, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaStreamDestroy(stream);
        std::cerr << "FAIL sync\n";
        return 1;
    }
    cudaStreamDestroy(stream);
    const auto got = from_device_bf16(dlogits.data(), kVocab);
    if (!(got[0] > 29.5f) || std::fabs(got[1]) > 0.1f || std::fabs(got[2]) > 0.1f ||
        std::fabs(got[3]) > 0.1f) {
        std::cerr << "FAIL prefill logits " << got[0] << ' ' << got[1] << ' ' << got[2] << ' '
                  << got[3] << '\n';
        return 1;
    }
    std::cout << "OK gemma4 prefill logit0=" << got[0] << '\n';
    return 0;
}
