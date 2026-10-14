#pragma once

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One [Bc, Width] tile of keys [k0, k0 + Bc) from columns [column, column + Width) of the
// [D,64,Hkv,pages] cache into swizzled shared memory. Keys past `last` are zeroed so the
// unwritten cache tail never reaches the tensor cores.
template <int D, int Width, int Bc, int Threads>
__device__ __forceinline__ void gemma4_stage_keys(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                                  int kv_heads, int kv_head, int physical, int k0,
                                                  int last, int column, int tid) {
    constexpr int VecPerRow = Width / 8;
    const __nv_bfloat16* block =
        cache +
        ((static_cast<std::int64_t>(physical) * kv_heads + kv_head) * kPagedKVPageSize +
         (k0 & kPagedKVPageMask)) *
            D +
        column;
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key    = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = &dst[key * Width + causal_swizzle(key, d)];
        if (physical >= 0 && k0 + key <= last) {
            cp_async<16, Cache::cg>(p, &block[key * D + d]);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

} // namespace ninfer::ops::detail
