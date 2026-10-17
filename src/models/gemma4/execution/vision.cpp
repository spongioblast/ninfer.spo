#include "models/gemma4/execution/vision.h"

#include "core/device.h"
#include "models/gemma4/execution/projection.h"
#include "models/gemma4/execution/text_forward.h"
#include "ninfer/ops/geglu.h"
#include "ninfer/ops/gemma4_vision_attention.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_scale.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

constexpr std::size_t kAlign = 256;
constexpr int kHeadDim       = 72;
constexpr int kHeads         = 16;

float bf16_to_f32(std::uint16_t h) {
    std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) { return static_cast<std::uint16_t>((u >> 16) | 0x0040u); }
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

void require_matrix(const Weight& weight, int rows, int cols, const char* name) {
    if (weight.n != rows || weight.k != cols) {
        throw std::invalid_argument(std::string("gemma4_vision: ") + name + " must be [" +
                                    std::to_string(rows) + "," + std::to_string(cols) + "]");
    }
}

void copy_rows(void* dst, const void* src, int row_offset, int rows, int src_rows, int tokens,
               cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst, static_cast<std::size_t>(rows) * 2,
                                 static_cast<const std::uint8_t*>(src) + row_offset * 2,
                                 static_cast<std::size_t>(src_rows) * 2,
                                 static_cast<std::size_t>(rows) * 2, static_cast<std::size_t>(tokens),
                                 cudaMemcpyDeviceToDevice, stream));
}

Tensor heads_view(Tensor& matrix, int tokens) {
    return matrix.view({kHeadDim, kHeads, tokens});
}

// Axial 2-D RoPE: 36-dim rotate-half on x, then on y. HF concatenates [xh,xh,yw,yw].
void gemma4_vision_axial_rope(const Tensor& positions, Tensor& q, Tensor& k, float theta,
                              cudaStream_t stream) {
    const int dim    = q.ne[0];
    const int heads  = q.ne[1];
    const int tokens = q.ne[2];
    if (dim != kHeadDim || heads != kHeads || k.ne[0] != dim || k.ne[1] != heads ||
        k.ne[2] != tokens) {
        throw std::invalid_argument("gemma4_vision: RoPE needs [72,16,T] q and k");
    }
    const std::size_t elems = static_cast<std::size_t>(dim) * heads * tokens;
    std::vector<std::uint16_t> q_host(elems);
    std::vector<std::uint16_t> k_host(elems);
    std::vector<std::int32_t> pos(static_cast<std::size_t>(tokens) * 2);
    CUDA_CHECK(cudaMemcpyAsync(q_host.data(), q.data, elems * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(k_host.data(), k.data, elems * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(pos.data(), positions.data, pos.size() * 4, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float freq[18];
    for (int i = 0; i < 18; ++i) {
        freq[i] = std::pow(theta, -2.0f * static_cast<float>(i) / 36.0f);
    }
    auto apply = [&](std::vector<std::uint16_t>& host) {
        for (int t = 0; t < tokens; ++t) {
            const float px = static_cast<float>(pos[static_cast<std::size_t>(t)]);
            const float py =
                static_cast<float>(pos[static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens)]);
            for (int head = 0; head < heads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>(t) * heads * dim + static_cast<std::size_t>(head) * dim;
                auto axis = [&](int offset, float coordinate) {
                    for (int i = 0; i < 18; ++i) {
                        const float angle = coordinate * freq[i];
                        const float c     = std::cos(angle);
                        const float s     = std::sin(angle);
                        const float a = bf16_to_f32(host[base + static_cast<std::size_t>(offset + i)]);
                        const float b =
                            bf16_to_f32(host[base + static_cast<std::size_t>(offset + 18 + i)]);
                        host[base + static_cast<std::size_t>(offset + i)]      = f32_to_bf16(a * c - b * s);
                        host[base + static_cast<std::size_t>(offset + 18 + i)] = f32_to_bf16(b * c + a * s);
                    }
                };
                axis(0, px);
                axis(36, py);
            }
        }
    };
    apply(q_host);
    apply(k_host);
    CUDA_CHECK(cudaMemcpyAsync(q.data, q_host.data(), elems * 2, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(k.data, k_host.data(), elems * 2, cudaMemcpyHostToDevice, stream));
}

void scale_patches(const Tensor& patches, Tensor& out, cudaStream_t stream) {
    const int pixels = patches.ne[0];
    const int tokens = patches.ne[1];
    std::vector<std::uint16_t> host(static_cast<std::size_t>(pixels) * tokens);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), patches.data, host.size() * 2, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::uint16_t& bits : host) {
        bits = f32_to_bf16(2.0f * (bf16_to_f32(bits) - 0.5f));
    }
    CUDA_CHECK(cudaMemcpyAsync(out.data, host.data(), host.size() * 2, cudaMemcpyHostToDevice,
                               stream));
}

void pool_and_standardize(const Tensor& hidden, const Tensor& positions, const Tensor& bias,
                          const Tensor& scale, int pool, Tensor& out, cudaStream_t stream) {
    const int width  = hidden.ne[0];
    const int tokens = hidden.ne[1];
    const int pooled = out.ne[1];
    const int k2     = pool * pool;
    if (tokens != pooled * k2) {
        throw std::invalid_argument("gemma4_vision: patch count must be pooled * k^2");
    }
    std::vector<std::uint16_t> src(static_cast<std::size_t>(width) * tokens);
    std::vector<std::int32_t> pos(static_cast<std::size_t>(tokens) * 2);
    std::vector<std::uint16_t> bias_bits(static_cast<std::size_t>(width));
    std::vector<std::uint16_t> scale_bits(static_cast<std::size_t>(width));
    CUDA_CHECK(cudaMemcpyAsync(src.data(), hidden.data, src.size() * 2, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(pos.data(), positions.data, pos.size() * 4, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(bias_bits.data(), bias.data, bias_bits.size() * 2,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_bits.data(), scale.data, scale_bits.size() * 2,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int max_x = 0;
    for (int t = 0; t < tokens; ++t) {
        const int x = pos[static_cast<std::size_t>(t)];
        if (x < 0) { continue; }
        max_x = std::max(max_x, x + 1);
    }
    if (max_x <= 0) { throw std::invalid_argument("gemma4_vision: no unpadded patches to pool"); }
    const int bins_x = max_x / pool;
    std::vector<double> acc(static_cast<std::size_t>(width) * pooled, 0.0);
    for (int t = 0; t < tokens; ++t) {
        const int x = pos[static_cast<std::size_t>(t)];
        const int y = pos[static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens)];
        if (x < 0 || y < 0) { continue; }
        const int bin = (x / pool) + bins_x * (y / pool);
        if (bin < 0 || bin >= pooled) {
            throw std::invalid_argument("gemma4_vision: pooled index is outside T_out");
        }
        for (int d = 0; d < width; ++d) {
            acc[static_cast<std::size_t>(d) + static_cast<std::size_t>(width) * bin] +=
                bf16_to_f32(src[static_cast<std::size_t>(d) + static_cast<std::size_t>(width) * t]);
        }
    }
    const double gain = std::sqrt(static_cast<double>(width)) / static_cast<double>(k2);
    std::vector<std::uint16_t> dst(static_cast<std::size_t>(width) * pooled);
    for (int t = 0; t < pooled; ++t) {
        for (int d = 0; d < width; ++d) {
            const std::size_t i = static_cast<std::size_t>(d) + static_cast<std::size_t>(width) * t;
            const double centered =
                acc[i] * gain - static_cast<double>(bf16_to_f32(bias_bits[static_cast<std::size_t>(d)]));
            dst[i] = f32_to_bf16(static_cast<float>(
                centered * static_cast<double>(bf16_to_f32(scale_bits[static_cast<std::size_t>(d)]))));
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(out.data, dst.data(), dst.size() * 2, cudaMemcpyHostToDevice, stream));
}

} // namespace

std::vector<int> gemma4_vision_kept_bins(const std::int32_t* positions, int tokens, int pool,
                                         int pooled) {
    if (positions == nullptr || tokens <= 0 || pool <= 0 || pooled <= 0) {
        throw std::invalid_argument("gemma4_vision: kept-bin geometry is empty");
    }
    int max_x = 0;
    for (int t = 0; t < tokens; ++t) {
        const int x = positions[t];
        if (x >= 0) { max_x = std::max(max_x, x + 1); }
    }
    if (max_x <= 0) { return {}; }
    const int bins_x = max_x / pool;
    std::vector<char> seen(static_cast<std::size_t>(pooled), 0);
    for (int t = 0; t < tokens; ++t) {
        const int x = positions[t];
        const int y = positions[t + tokens];
        if (x < 0 || y < 0) { continue; }
        const int bin = (x / pool) + bins_x * (y / pool);
        if (bin < 0 || bin >= pooled) {
            throw std::invalid_argument("gemma4_vision: pooled index is outside T_out");
        }
        seen[static_cast<std::size_t>(bin)] = 1;
    }
    std::vector<int> kept;
    kept.reserve(static_cast<std::size_t>(pooled));
    for (int bin = 0; bin < pooled; ++bin) {
        if (seen[static_cast<std::size_t>(bin)]) { kept.push_back(bin); }
    }
    return kept;
}

std::size_t gemma4_vision_workspace_bytes(const Gemma4VisionWeights& weights, int patches,
                                          int pooled) {
    const auto h     = static_cast<std::size_t>(weights.hidden);
    const auto inter = static_cast<std::size_t>(weights.intermediate);
    const auto pix   = static_cast<std::size_t>(weights.patch_pixels);
    const auto t     = static_cast<std::size_t>(patches);
    const auto p     = static_cast<std::size_t>(pooled);
    // scaled; x,h; qkv; q,k,v; qn,kn,vn; attn,proj,proj_n; gate,act; ones; pooled; attn scores.
    const std::size_t bf16 =
        pix * t + 14 * h * t + 3 * inter * t + 72 + h + 2 * h * p + 16 * t * t;
    const std::size_t score_fp32 = 16 * t * t * 4;
    const std::size_t extra = 8 * t * 4; // idx I32 [4,T] and wts FP32 [4,T]
    const std::size_t scratch = std::max({projection_bytes(weights.patch_projection,
                                                           ops::LinearPolicy::A16Only, patches),
                                          projection_bytes(weights.embedding_projection,
                                                           ops::LinearPolicy::A16Only, pooled),
                                          projection_bytes(weights.layers.empty() ? weights.patch_projection
                                                                                  : weights.layers[0].qkv,
                                                           ops::LinearPolicy::A16Only, patches)});
    return bf16 * 2 + score_fp32 + extra + scratch + 24 * kAlign;
}

void gemma4_vision_encode(const Tensor& patches, const Tensor& positions, const Tensor& mask,
                          Tensor& out, const Gemma4VisionWeights& weights,
                          ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    const int tokens = patches.ne[1];
    const int hidden = weights.hidden;
    const int dim    = weights.head_dim;
    const int heads  = weights.heads;
    const int inter  = weights.intermediate;
    const int pool   = weights.pooling_kernel_size;
    const int pooled = tokens / (pool * pool);
    if (patches.dtype != DType::BF16 || patches.ne[0] != weights.patch_pixels || tokens <= 0 ||
        !patches.is_contiguous()) {
        throw std::invalid_argument("gemma4_vision: patches must be contiguous BF16 [pixels,T]");
    }
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens || positions.ne[1] != 2 ||
        !positions.is_contiguous()) {
        throw std::invalid_argument("gemma4_vision: positions must be contiguous I32 [T,2]");
    }
    if (mask.dtype != DType::I32 || mask.ne[0] != tokens || !mask.is_contiguous()) {
        throw std::invalid_argument("gemma4_vision: mask must be contiguous I32 [T]");
    }
    if (out.dtype != DType::BF16 || out.ne[0] != weights.text_hidden || out.ne[1] != pooled ||
        !out.is_contiguous()) {
        throw std::invalid_argument("gemma4_vision: out must be contiguous BF16 [text_hidden,T_out]");
    }
    if (dim != kHeadDim || heads != kHeads || hidden != dim * heads) {
        throw std::invalid_argument("gemma4_vision: encoder is D72 with 16 heads");
    }
    require_matrix(weights.patch_projection, hidden, weights.patch_pixels, "patch_projection");
    require_matrix(weights.embedding_projection, weights.text_hidden, hidden,
                   "embedding_projection");

    auto scaled    = workspace.alloc(DType::BF16, {weights.patch_pixels, tokens});
    auto x         = workspace.alloc(DType::BF16, {hidden, tokens});
    auto h         = workspace.alloc(DType::BF16, {hidden, tokens});
    auto qkv       = workspace.alloc(DType::BF16, {3 * hidden, tokens});
    auto q         = workspace.alloc(DType::BF16, {hidden, tokens});
    auto k         = workspace.alloc(DType::BF16, {hidden, tokens});
    auto v         = workspace.alloc(DType::BF16, {hidden, tokens});
    auto qn        = workspace.alloc(DType::BF16, {hidden, tokens});
    auto kn        = workspace.alloc(DType::BF16, {hidden, tokens});
    auto vn        = workspace.alloc(DType::BF16, {hidden, tokens});
    auto attn      = workspace.alloc(DType::BF16, {hidden, tokens});
    auto proj      = workspace.alloc(DType::BF16, {hidden, tokens});
    auto proj_n    = workspace.alloc(DType::BF16, {hidden, tokens});
    auto gate      = workspace.alloc(DType::BF16, {2 * inter, tokens});
    auto act       = workspace.alloc(DType::BF16, {inter, tokens});
    auto ones_head = workspace.alloc(DType::BF16, {dim});
    auto ones_hid  = workspace.alloc(DType::BF16, {hidden});
    auto idx       = workspace.alloc(DType::I32, {4, tokens});
    auto wts       = workspace.alloc(DType::FP32, {4, tokens});
    auto pooled_x  = workspace.alloc(DType::BF16, {hidden, pooled});
    auto pooled_n  = workspace.alloc(DType::BF16, {hidden, pooled});
    auto scores       = workspace.alloc(DType::FP32, {tokens, tokens, kHeads});
    auto attn_weights = workspace.alloc(DType::BF16, {tokens, tokens, kHeads});
    std::vector<std::uint16_t> one_head(static_cast<std::size_t>(dim), f32_to_bf16(1.0f));
    std::vector<std::uint16_t> one_hid(static_cast<std::size_t>(hidden), f32_to_bf16(1.0f));
    CUDA_CHECK(cudaMemcpyAsync(ones_head.data, one_head.data(), one_head.size() * 2,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ones_hid.data, one_hid.data(), one_hid.size() * 2,
                               cudaMemcpyHostToDevice, stream));

    std::vector<std::int32_t> pos(static_cast<std::size_t>(tokens) * 2);
    CUDA_CHECK(cudaMemcpyAsync(pos.data(), positions.data, pos.size() * 4, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * tokens, 0);
    std::vector<float> weights_host(static_cast<std::size_t>(4) * tokens, 0.0f);
    const int table = weights.position_table_size;
    for (int t = 0; t < tokens; ++t) {
        const int x_pos = pos[static_cast<std::size_t>(t)];
        const int y_pos = pos[static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens)];
        if (x_pos < 0 || y_pos < 0) { continue; }
        if (x_pos >= table || y_pos >= table) {
            throw std::invalid_argument("gemma4_vision: position exceeds the table");
        }
        const std::size_t base                            = static_cast<std::size_t>(t) * 4;
        indices[base]                                     = x_pos;
        indices[base + 1]                                 = y_pos + table;
        weights_host[base]                                = 1.0f;
        weights_host[base + 1]                            = 1.0f;
    }
    CUDA_CHECK(cudaMemcpyAsync(idx.data, indices.data(), indices.size() * 4, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(wts.data, weights_host.data(), weights_host.size() * 4,
                               cudaMemcpyHostToDevice, stream));

    scale_patches(patches, scaled, stream);
    project(scaled, weights.patch_projection, ops::LinearPolicy::A16Only, x, gemm, workspace,
            stream);
    Tensor table_view(const_cast<void*>(weights.position_table), DType::BF16,
                      {hidden, 2 * table});
    ops::vision_pos_embed_add(table_view, idx, wts, x, stream);
    for (const auto& layer : weights.layers) {
        require_matrix(layer.qkv, 3 * hidden, hidden, "qkv");
        require_matrix(layer.o_proj, hidden, hidden, "o_proj");
        require_matrix(layer.gate_up, 2 * inter, hidden, "gate_up");
        require_matrix(layer.down, hidden, inter, "down");
        ops::rmsnorm(x, norm_weight(layer.input_norm, hidden), weights.rms_eps, false, h, stream);
        project(h, layer.qkv, layer.qkv_policy, qkv, gemm, workspace, stream);
        copy_rows(q.data, qkv.data, 0, hidden, 3 * hidden, tokens, stream);
        copy_rows(k.data, qkv.data, hidden, hidden, 3 * hidden, tokens, stream);
        copy_rows(v.data, qkv.data, 2 * hidden, hidden, 3 * hidden, tokens, stream);
        Tensor qh = heads_view(q, tokens);
        Tensor kh = heads_view(k, tokens);
        Tensor vh = heads_view(v, tokens);
        Tensor qn_h = heads_view(qn, tokens);
        Tensor kn_h = heads_view(kn, tokens);
        Tensor vn_h = heads_view(vn, tokens);
        ops::rmsnorm(qh, norm_weight(layer.q_norm, dim), weights.rms_eps, false, qn_h, stream);
        ops::rmsnorm(kh, norm_weight(layer.k_norm, dim), weights.rms_eps, false, kn_h, stream);
        ops::rmsnorm(vh, ones_head, weights.rms_eps, false, vn_h, stream);
        gemma4_vision_axial_rope(positions, qn_h, kn_h, weights.rope_theta, stream);
        Tensor ah = heads_view(attn, tokens);
        ops::gemma4_vision_attention(qn_h, kn_h, vn_h, &mask, scores, attn_weights, ah, stream);
        project(attn, layer.o_proj, layer.o_policy, proj, gemm, workspace, stream);
        ops::rmsnorm(proj, norm_weight(layer.post_attn_norm, hidden), weights.rms_eps, false, proj_n,
                     stream);
        ops::residual_add(proj_n, x, stream);
        ops::rmsnorm(x, norm_weight(layer.pre_ff_norm, hidden), weights.rms_eps, false, h, stream);
        project(h, layer.gate_up, layer.gate_up_policy, gate, gemm, workspace, stream);
        const Tensor gate_rows = gate.slice(0, 0, inter);
        const Tensor up_rows   = gate.slice(0, inter, inter);
        ops::geglu(gate_rows, up_rows, act, stream);
        project(act, layer.down, layer.down_policy, proj, gemm, workspace, stream);
        ops::rmsnorm(proj, norm_weight(layer.post_ff_norm, hidden), weights.rms_eps, false, proj_n,
                     stream);
        ops::residual_add(proj_n, x, stream);
        if (layer.layer_scalar != 1.0f) { ops::residual_scale(x, layer.layer_scalar, stream); }
    }
    Tensor bias(const_cast<void*>(weights.std_bias), DType::BF16, {hidden});
    Tensor scale(const_cast<void*>(weights.std_scale), DType::BF16, {hidden});
    pool_and_standardize(x, positions, bias, scale, pool, pooled_x, stream);
    ops::rmsnorm(pooled_x, ones_hid, weights.rms_eps, false, pooled_n, stream);
    project(pooled_n, weights.embedding_projection, ops::LinearPolicy::A16Only, out, gemm,
            workspace, stream);
}

} // namespace ninfer::models::gemma4
