#pragma once

#include "core/performance.h"

#ifdef NINFER_PERFORMANCE_TRACE
#    include <algorithm>

namespace ninfer::ops::flash_next_work {
using performance::Work;
using U = std::uint64_t;

// These models count the dominant exact projection formulas and public input/output traffic.
// They intentionally exclude private scratch, launch cost, transcendentals, and padding in MMA
// tiles. T is the launch envelope (including padded columns), not committed output tokens.
constexpr Work dense(U parameters, U tokens, U io_bytes) {
    const U bytes = 2 * parameters + io_bytes;
    return {bytes, bytes, 2 * parameters * tokens, 0, 0};
}

constexpr Work moe(U t, bool nvfp4) {
    constexpr U expert        = 3 * 2560 * 640;
    constexpr U shared_router = expert + 2560 * (512 + 1);
    // Unique experts are data dependent: all tokens may share ten, or use disjoint top-10s.
    // NVFP4 has one code nibble and one E4M3 scale per 16 values, plus four FP32
    // gate-up/down weight/input divisors per expert. BF16 has no such scale planes.
    const U expert_bytes = nvfp4 ? expert / 2 + expert / 16 + 16 : 2 * expert;
    const U common       = 2 * shared_router + 2 * 2560 * t * 2;
    const U routed_flops = 2 * expert * 10 * t;
    return {common + 10 * expert_bytes, common + std::min<U>(512, 10 * t) * expert_bytes,
            2 * shared_router * t + (nvfp4 ? 0 : routed_flops), nvfp4 ? routed_flops : 0, 0};
}

constexpr Work gdn(U t, U batch, bool record, bool prefill = false) {
    constexpr U parameters = 2560 * (2 * 48 + 10240 + 6144) + 2560 * 6144;
    // One state read, and one write only for state-update paths. Record paths publish
    // convolution/key/value/gate records instead. Sequential recurrence has at least two
    // matrix-vector products and one outer-product update (6 FLOPs per state element).
    constexpr U state = 48 * 128 * 128;
    const U records   = record ? t * (2 * (10240 + 128 * 16 + 128 * 48) + 4 * 2 * 48) : 0;
    auto work = dense(parameters, t, 4 * 2560 * t + 4 * state * batch * (record ? 1 : 2) + records);
    // Chunked prefill uses a different arithmetic decomposition, including tensor cores;
    // do not charge the sequential formula to FP32 CUDA cores for that route.
    work.fp32_flops = prefill ? 0 : 6 * state * t;
    return work;
}

constexpr Work qsa(U t, bool reuse) {
    // Attention and selection depend on device-resident positions/valid extents and chosen
    // indices. Do not substitute a graph's maximum context for those values. This is the
    // projection floor only; the report identifies it as such and retains context metadata.
    const U parameters = 2560 * (12288 + 512 + 512 + 128 + (reuse ? 0 : 512) + 6144);
    return dense(parameters, t, 4 * 2560 * t);
}

constexpr Work hyper(U t, bool injection, bool combine) {
    const U parameters = 2 * 10240 * 320 + (injection ? 4 * 10240 : 0);
    return dense(parameters, t,
                 2 * t * (10240 + 2560 + (injection ? 4 : 0) + (combine ? 10240 + 2560 + 4 : 0)));
}

constexpr Work ple(U t) { return dense(2560 * (10240 + 2560), t, t * (2 * 10240 * 2 + 2560)); }
} // namespace ninfer::ops::flash_next_work
#endif
