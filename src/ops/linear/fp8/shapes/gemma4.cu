// Gemma 4 31B projections and output head in the row-scaled FP8 profile, weight-only, on the
// dense routes of fp8_dense_a16.cuh. Not yet tuned.
#include "ops/linear/fp8/fp8_dense_a16.cuh"

namespace ninfer::ops::detail {
const Fp8LinearShape kFp8N16384K5376  = fp8_dense::shape<16384, 5376>();
const Fp8LinearShape kFp8N18432K5376  = fp8_dense::shape<18432, 5376>();
const Fp8LinearShape kFp8N43008K5376  = fp8_dense::shape<43008, 5376>();
const Fp8LinearShape kFp8N5376K8192   = fp8_dense::shape<5376, 8192>();
const Fp8LinearShape kFp8N5376K16384  = fp8_dense::shape<5376, 16384>();
const Fp8LinearShape kFp8N5376K21504  = fp8_dense::shape<5376, 21504>();
const Fp8LinearShape kFp8N262144K5376 = fp8_dense::shape<262144, 5376>();
} // namespace ninfer::ops::detail
