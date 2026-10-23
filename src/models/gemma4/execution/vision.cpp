#include "models/gemma4/execution/vision.h"

#include "core/device.h"
#include "models/gemma4/execution/projection.h"
#include "models/gemma4/execution/text_forward.h"
#include "ninfer/ops/geglu.h"
#include "ninfer/ops/gemma4_vision_attention.h"
#include "ninfer/ops/gemma4_vision_prep.h"
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

// Axial 2-D RoPE angles, once per image: per token cos and sin of the 18 x angles, then of the
// 18 y angles (36-dim rotate-half on x, then on y; HF concatenates [xh,xh,yw,yw]). Pad tokens
// (-1, -1) are rotated like any other; their keys are masked.
std::vector<float> rope_table(const std::vector<std::int32_t>& pos, int tokens, float theta) {
    float freq[18];
    for (int i = 0; i < 18; ++i) {
        freq[i] = std::pow(theta, -2.0f * static_cast<float>(i) / 36.0f);
    }
    std::vector<float> table(static_cast<std::size_t>(tokens) * 72);
    for (int t = 0; t < tokens; ++t) {
        for (int axis = 0; axis < 2; ++axis) {
            const float coordinate = static_cast<float>(
                pos[static_cast<std::size_t>(t) + static_cast<std::size_t>(axis) * tokens]);
            float* row = table.data() + static_cast<std::size_t>(t) * 72 + axis * 36;
            for (int i = 0; i < 18; ++i) {
                const float angle = coordinate * freq[i];
                row[i]            = std::cos(angle);
                row[18 + i]       = std::sin(angle);
            }
        }
    }
    return table;
}

// Pooled bin of each unpadded patch, as offsets [pooled+1] and members in token order.
struct PoolPlan {
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> members;
};

PoolPlan pool_plan(const std::vector<std::int32_t>& pos, int tokens, int pool, int pooled) {
    int max_x = 0;
    for (int t = 0; t < tokens; ++t) {
        const int x = pos[static_cast<std::size_t>(t)];
        if (x >= 0) { max_x = std::max(max_x, x + 1); }
    }
    if (max_x <= 0) { throw std::invalid_argument("gemma4_vision: no unpadded patches to pool"); }
    const int bins_x = max_x / pool;
    std::vector<int> bins(static_cast<std::size_t>(tokens), -1);
    PoolPlan plan;
    plan.offsets.assign(static_cast<std::size_t>(pooled) + 1, 0);
    for (int t = 0; t < tokens; ++t) {
        const int x = pos[static_cast<std::size_t>(t)];
        const int y = pos[static_cast<std::size_t>(t) + static_cast<std::size_t>(tokens)];
        if (x < 0 || y < 0) { continue; }
        const int bin = (x / pool) + bins_x * (y / pool);
        if (bin < 0 || bin >= pooled) {
            throw std::invalid_argument("gemma4_vision: pooled index is outside T_out");
        }
        bins[static_cast<std::size_t>(t)] = bin;
        ++plan.offsets[static_cast<std::size_t>(bin) + 1];
    }
    for (int bin = 0; bin < pooled; ++bin) {
        plan.offsets[static_cast<std::size_t>(bin) + 1] += plan.offsets[static_cast<std::size_t>(bin)];
    }
    plan.members.resize(static_cast<std::size_t>(plan.offsets.back()));
    std::vector<std::int32_t> fill(plan.offsets.begin(), plan.offsets.end() - 1);
    for (int t = 0; t < tokens; ++t) {
        const int bin = bins[static_cast<std::size_t>(t)];
        if (bin >= 0) { plan.members[static_cast<std::size_t>(fill[static_cast<std::size_t>(bin)]++)] = t; }
    }
    return plan;
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

// The registered BF16 vision shapes. gate_up is two of the 4304x1152 rows, not one 8608-row GEMM.
bool encoder_bf16_shape(const Weight& weight) {
    if (weight.qtype != QType::BF16) { return false; }
    const int n = weight.n;
    const int k = weight.k;
    return (n == 1152 && (k == 768 || k == 1152 || k == 4304)) || (n == 3456 && k == 1152) ||
           (n == 4304 && k == 1152) || (n == 5376 && k == 1152);
}

Weight row_block(const Weight& weight, int row, int rows) {
    Weight view = weight;
    view.n = rows;
    if (view.ndim > 0) { view.shape[0] = rows; }
    view.padded_shape[0] = rows;
    const auto bytes = static_cast<std::size_t>(row) * static_cast<std::size_t>(weight.k) * 2U;
    view.qdata = static_cast<const std::uint8_t*>(weight.qdata) + bytes;
    return view;
}

// Registered BF16 shapes require a multiple of 4 columns. A short tail is zero-padded for the
// GEMM and dropped from the output.
void project_encoder(const Tensor& x, const Weight& weight, Tensor& out, ops::Bf16GemmContext& gemm,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    if (!encoder_bf16_shape(weight)) {
        project(x, weight, ops::LinearPolicy::A16Only, out, gemm, workspace, stream);
        return;
    }
    const int tokens = x.ne[1];
    const int padded = (tokens + 3) & ~3;
    if (padded == tokens) {
        ops::linear(x, weight, out, ops::LinearPolicy::A16Only, workspace, stream);
        return;
    }
    auto scope = workspace.scope();
    auto xpad  = workspace.alloc(DType::BF16, {x.ne[0], padded});
    auto opad  = workspace.alloc(DType::BF16, {out.ne[0], padded});
    CUDA_CHECK(cudaMemsetAsync(xpad.data, 0, xpad.bytes(), stream));
    CUDA_CHECK(cudaMemcpyAsync(xpad.data, x.data,
                               static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(tokens) * 2U,
                               cudaMemcpyDeviceToDevice, stream));
    Tensor padded_x(xpad.data, DType::BF16, {x.ne[0], padded});
    Tensor padded_out(opad.data, DType::BF16, {out.ne[0], padded});
    ops::linear(padded_x, weight, padded_out, ops::LinearPolicy::A16Only, workspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(out.data, opad.data,
                               static_cast<std::size_t>(out.ne[0]) * static_cast<std::size_t>(tokens) * 2U,
                               cudaMemcpyDeviceToDevice, stream));
}

std::size_t gemma4_vision_workspace_bytes(const Gemma4VisionWeights& weights, int patches,
                                          int pooled) {
    const auto h     = static_cast<std::size_t>(weights.hidden);
    const auto inter = static_cast<std::size_t>(weights.intermediate);
    const auto pix   = static_cast<std::size_t>(weights.patch_pixels);
    const auto t     = static_cast<std::size_t>(patches);
    const auto p     = static_cast<std::size_t>(pooled);
    // scaled; x,h; qkv; q,k,v; qn,kn,vn; attn,proj,proj_n; gate,act; ones; pooled.
    const std::size_t bf16 = pix * t + 14 * h * t + 3 * inter * t + 72 + h + 2 * h * p;
    // idx I32 [4,T], wts FP32 [4,T], RoPE table FP32 [72,T], pool offsets [P+1] and members [T].
    const std::size_t extra = 8 * t * 4 + 72 * t * 4 + (p + 1 + t) * 4;
    const std::size_t scratch = std::max({projection_bytes(weights.patch_projection,
                                                           ops::LinearPolicy::A16Only, patches),
                                          projection_bytes(weights.embedding_projection,
                                                           ops::LinearPolicy::A16Only, pooled),
                                          projection_bytes(weights.layers.empty() ? weights.patch_projection
                                                                                  : weights.layers[0].qkv,
                                                           ops::LinearPolicy::A16Only, patches)});
    // A token count that is not a multiple of 4 is padded for one GEMM at a time.
    const std::size_t pad = (static_cast<std::size_t>(std::max(weights.hidden * 3, weights.intermediate)) +
                             static_cast<std::size_t>(weights.text_hidden)) *
                            static_cast<std::size_t>(patches + 3) * 2U;
    return bf16 * 2 + extra + scratch + pad + 24 * kAlign;
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
    if (tokens != pooled * pool * pool) {
        throw std::invalid_argument("gemma4_vision: patch count must be pooled * k^2");
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
    auto gate_part = workspace.alloc(DType::BF16, {inter, tokens});
    auto up_part   = workspace.alloc(DType::BF16, {inter, tokens});
    auto act       = workspace.alloc(DType::BF16, {inter, tokens});
    auto ones_head = workspace.alloc(DType::BF16, {dim});
    auto ones_hid  = workspace.alloc(DType::BF16, {hidden});
    auto idx       = workspace.alloc(DType::I32, {4, tokens});
    auto wts       = workspace.alloc(DType::FP32, {4, tokens});
    auto pooled_x  = workspace.alloc(DType::BF16, {hidden, pooled});
    auto pooled_n  = workspace.alloc(DType::BF16, {hidden, pooled});
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
    const std::vector<float> angles = rope_table(pos, tokens, weights.rope_theta);
    const PoolPlan plan             = pool_plan(pos, tokens, pool, pooled);
    auto rope       = workspace.alloc(DType::FP32, {72, tokens});
    auto offsets    = workspace.alloc(DType::I32, {pooled + 1});
    auto members    = workspace.alloc(DType::I32, {static_cast<int>(plan.members.size())});
    CUDA_CHECK(cudaMemcpyAsync(rope.data, angles.data(), angles.size() * 4, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(offsets.data, plan.offsets.data(), plan.offsets.size() * 4,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(members.data, plan.members.data(), plan.members.size() * 4,
                               cudaMemcpyHostToDevice, stream));

    ops::gemma4_vision_scale_patches(patches, scaled, stream);
    project_encoder(scaled, weights.patch_projection, x, gemm, workspace, stream);
    Tensor table_view(const_cast<void*>(weights.position_table), DType::BF16,
                      {hidden, 2 * table});
    ops::vision_pos_embed_add(table_view, idx, wts, x, stream);
    for (const auto& layer : weights.layers) {
        require_matrix(layer.qkv, 3 * hidden, hidden, "qkv");
        require_matrix(layer.o_proj, hidden, hidden, "o_proj");
        require_matrix(layer.gate_up, 2 * inter, hidden, "gate_up");
        require_matrix(layer.down, hidden, inter, "down");
        ops::rmsnorm(x, norm_weight(layer.input_norm, hidden), weights.rms_eps, false, h, stream);
        project_encoder(h, layer.qkv, qkv, gemm, workspace, stream);
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
        ops::gemma4_vision_axial_rope(rope, qn_h, kn_h, stream);
        Tensor ah = heads_view(attn, tokens);
        ops::gemma4_vision_attention(qn_h, kn_h, vn_h, &mask, ah, stream);
        project_encoder(attn, layer.o_proj, proj, gemm, workspace, stream);
        ops::rmsnorm(proj, norm_weight(layer.post_attn_norm, hidden), weights.rms_eps, false, proj_n,
                     stream);
        ops::residual_add(proj_n, x, stream);
        ops::rmsnorm(x, norm_weight(layer.pre_ff_norm, hidden), weights.rms_eps, false, h, stream);
        const Weight gate_w = row_block(layer.gate_up, 0, inter);
        const Weight up_w   = row_block(layer.gate_up, inter, inter);
        if (encoder_bf16_shape(gate_w) && encoder_bf16_shape(up_w)) {
            project_encoder(h, gate_w, gate_part, gemm, workspace, stream);
            project_encoder(h, up_w, up_part, gemm, workspace, stream);
        } else {
            auto scope = workspace.scope();
            auto fused = workspace.alloc(DType::BF16, {2 * inter, tokens});
            project(h, layer.gate_up, layer.gate_up_policy, fused, gemm, workspace, stream);
            copy_rows(gate_part.data, fused.data, 0, inter, 2 * inter, tokens, stream);
            copy_rows(up_part.data, fused.data, inter, inter, 2 * inter, tokens, stream);
        }
        ops::geglu(gate_part, up_part, act, stream);
        project_encoder(act, layer.down, proj, gemm, workspace, stream);
        ops::rmsnorm(proj, norm_weight(layer.post_ff_norm, hidden), weights.rms_eps, false, proj_n,
                     stream);
        ops::residual_add(proj_n, x, stream);
        if (layer.layer_scalar != 1.0f) { ops::residual_scale(x, layer.layer_scalar, stream); }
    }
    Tensor bias(const_cast<void*>(weights.std_bias), DType::BF16, {hidden});
    Tensor scale(const_cast<void*>(weights.std_scale), DType::BF16, {hidden});
    const double gain = std::sqrt(static_cast<double>(hidden)) / static_cast<double>(pool * pool);
    ops::gemma4_vision_pool(x, offsets, members, bias, scale, gain, pooled_x, stream);
    ops::rmsnorm(pooled_x, ones_hid, weights.rms_eps, false, pooled_n, stream);
    project_encoder(pooled_n, weights.embedding_projection, out, gemm, workspace, stream);
}

} // namespace ninfer::models::gemma4
