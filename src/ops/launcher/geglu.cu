// ninfer::ops — geglu launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/geglu.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/geglu.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

bool can_use_dim0_split_fast_path(const Tensor& gate, const Tensor& up, const Tensor& out) {
    if (gate.ne[2] != 1 || gate.ne[3] != 1) { return false; }
    if ((gate.ne[0] & 1) != 0) { return false; }
    if (gate.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        up.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16))) {
        return false;
    }
    if ((gate.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0 ||
        (up.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0) {
        return false;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    return ((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) == 0;
}

} // namespace

void geglu_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 256;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split_fast_path(gate, up, out)) {
            const std::int64_t row_pairs          = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kGegluPairsPerThread;
            const int grid_x =
                static_cast<int>(std::max<std::int64_t>(1, div_up(row_pairs, kPairsPerBlock)));
            for_each_token_slice(gate.ne[1], 1, [&](int token_offset, int token_count) {
                const Tensor gate_slice = gate.slice(1, token_offset, token_count);
                const Tensor up_slice   = up.slice(1, token_offset, token_count);
                Tensor out_slice        = out.slice(1, token_offset, token_count);
                const dim3 grid(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(token_count));
                geglu_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(gate_slice.data),
                    static_cast<const __nv_bfloat16*>(up_slice.data),
                    static_cast<__nv_bfloat16*>(out_slice.data), gate.ne[0],
                    gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                    up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
                CUDA_CHECK(cudaGetLastError());
            });
            return;
        }
        const int scalar_grid = static_cast<int>(std::min<std::int64_t>(
            div_up(n, static_cast<std::int64_t>(kBlock)), std::numeric_limits<int>::max()));
        geglu_strided_input_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n,
            gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3],
            up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(kBlock)), std::numeric_limits<int>::max()));
    geglu_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                              static_cast<const __nv_bfloat16*>(up.data),
                                              static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
