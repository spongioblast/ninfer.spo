#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Causal grouped-query attention over a BF16 paged KV cache.
 *
 *   score[j] = scale * dot(q[:,h,i], k[:,kv,j])
 *   out[:,h,i] = softmax(score) * v[:,kv,j]
 *
 * Key j is visible to query position p when j <= p and (window == 0 or p - j < window).
 * Query head h reads KV head floor(h / (Hq/Hkv)). Admitted profiles are D=256 with any
 * positive Hq divisible by Hkv, and D=512 with the same rule. Gemma 4 uses Hq=32, Hkv=16
 * (sliding, window 1024) and Hq=32, Hkv=4 (global, window 0), both with scale 1.
 *
 * q and out are contiguous BF16 [D,Hq,T]. positions is contiguous I32 [T]. k_pages and
 * v_pages are contiguous BF16 [D,64,Hkv,pages]. block_table is contiguous I32 [logical_pages]
 * and maps absolute position p to physical page block_table[p/64], slot p%64. The oracle
 * evaluates the visible set in FP64. The Op writes all of out and uses no workspace.
 */
void gemma4_attention(const Tensor& q, const Tensor& k_pages, const Tensor& v_pages,
                      const Tensor& block_table, const Tensor& positions, int window, float scale,
                      Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
