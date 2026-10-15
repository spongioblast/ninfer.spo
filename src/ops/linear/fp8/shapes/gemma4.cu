// Gemma 4 31B projections and output heads in the row-scaled FP8 profile. A16 runs on the dense
// routes of fp8_dense_a16.cuh (not yet tuned). The trunk projections also admit A8 from 128
// tokens (gemma4_a8.h); both vocabulary heads stay A16.
#include "ops/linear/fp8/fp8_dense_a16.cuh"
#include "ops/linear/fp8/shapes/gemma4_a8.h"

namespace ninfer::ops::detail {
namespace {
template <int N, int K>
constexpr Fp8LinearShape trunk() {
    Fp8LinearShape shape         = fp8_dense::shape<N, K>();
    shape.a8                     = gemma4_fp8::launch_a8<N, K>;
    shape.uses_a8                = gemma4_fp8::uses_a8;
    shape.partial_capacity_bytes = gemma4_fp8::partial_capacity_bytes;
    return shape;
}
} // namespace

const Fp8LinearShape kFp8N16384K5376  = trunk<16384, 5376>();
const Fp8LinearShape kFp8N18432K5376  = trunk<18432, 5376>();
const Fp8LinearShape kFp8N43008K5376  = trunk<43008, 5376>();
const Fp8LinearShape kFp8N5376K8192   = trunk<5376, 8192>();
const Fp8LinearShape kFp8N5376K16384  = trunk<5376, 16384>();
const Fp8LinearShape kFp8N5376K21504  = trunk<5376, 21504>();
const Fp8LinearShape kFp8N262144K5376 = fp8_dense::shape<262144, 5376>();
const Fp8LinearShape kFp8N262144K1024 = fp8_dense::shape<262144, 1024>();
} // namespace ninfer::ops::detail
