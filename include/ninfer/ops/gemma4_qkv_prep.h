#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gemma 4 attention input preparation in one pass over the fused projection output.
 *
 * qkv is contiguous BF16 [Hq*D + Hkv*D (+ Hkv*D), T]: Q heads, then K heads, then V heads on
 * sliding layers. Global layers have no V rows (v_from_k): V is the raw K. For each token t at
 * position p = positions[t]:
 *
 *   q_out[:,h,t]  = rope(rmsnorm(Q_h, q_norm))
 *   K_h           = rope(rmsnorm(K_h, k_norm))
 *   V_h           = rmsnorm(V_h or raw K_h, 1)
 *
 * rmsnorm is x * (mean(x^2) + eps)^-1/2 * weight with each result rounded to BF16, and rope is the
 * split-half NeoX rotation of dimensions [0, rotary_dim) at angle p * theta^(-2i/rotary_dim),
 * as in ops::rmsnorm and ops::rope. K_h and V_h are stored at physical page block_table[p/64],
 * slot p%64 of k_pages and v_pages (BF16 [D,64,Hkv,pages]); tokens with a negative position or a
 * page outside the table write no K/V. D is 256 (rotary_dim 256) or 512 (rotary_dim 128 after the
 * proportional-RoPE permutation). q_out is contiguous BF16 [D,Hq,T]. The output is
 * bit-identical to rmsnorm, rope and gemma4_kv_append run one after another. No workspace.
 */
void gemma4_qkv_prep(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                     const Tensor& positions, int rotary_dim, float theta, float eps, bool v_from_k,
                     Tensor& q_out, Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                     cudaStream_t stream);

} // namespace ninfer::ops
