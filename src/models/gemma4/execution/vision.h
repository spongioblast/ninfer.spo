#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::gemma4 {

struct Gemma4VisionLayerWeights {
    const void* input_norm     = nullptr;
    Weight qkv;
    const void* q_norm         = nullptr;
    const void* k_norm         = nullptr;
    Weight o_proj;
    const void* post_attn_norm = nullptr;
    const void* pre_ff_norm    = nullptr;
    Weight gate_up;
    Weight down;
    const void* post_ff_norm   = nullptr;
    ops::LinearPolicy qkv_policy     = ops::LinearPolicy::A16Only;
    ops::LinearPolicy o_policy       = ops::LinearPolicy::A16Only;
    ops::LinearPolicy gate_up_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy down_policy    = ops::LinearPolicy::A16Only;
    float layer_scalar               = 1.0f;
};

struct Gemma4VisionWeights {
    Weight patch_projection;
    const void* position_table = nullptr;
    const void* std_bias       = nullptr;
    const void* std_scale      = nullptr;
    Weight embedding_projection;
    std::vector<Gemma4VisionLayerWeights> layers;
    int hidden                 = 0;
    int text_hidden            = 0;
    int intermediate           = 0;
    int heads                  = 0;
    int head_dim               = 0;
    int patch_pixels           = 0;
    int pooling_kernel_size    = 0;
    int position_table_size    = 0;
    int default_output_length  = 0;
    int boi_token_id           = 0;
    int eoi_token_id           = 0;
    int image_token_id         = 0;
    float rope_theta           = 100.0f;
    float rms_eps              = 1.0e-6f;
};

// patches is contiguous BF16 [patch_pixels, T] in [0, 1]; the encoder applies 2*(x-0.5).
// positions is contiguous I32 [T,2] with -1 on both axes for pad.
// mask is contiguous I32 [T], nonzero = valid.
// out is contiguous BF16 [text_hidden, T_out] with T_out = T / (pool*pool).
// Reads positions back once, then queues the rest on stream: out is ready after a stream sync.
void gemma4_vision_encode(const Tensor& patches, const Tensor& positions, const Tensor& mask,
                          Tensor& out, const Gemma4VisionWeights& weights,
                          ops::Bf16GemmContext& gemm, WorkspaceArena& workspace,
                          cudaStream_t stream);

// Pooled bins that received at least one unpadded patch, in encoder column order.
[[nodiscard]] std::vector<int> gemma4_vision_kept_bins(const std::int32_t* positions, int tokens,
                                                       int pool, int pooled);

[[nodiscard]] std::size_t gemma4_vision_workspace_bytes(const Gemma4VisionWeights& weights,
                                                        int patches, int pooled);

} // namespace ninfer::models::gemma4
