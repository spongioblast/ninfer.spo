#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>

namespace ninfer::models::gemma4 {

// Appends the out.k most likely tokens of every column of `logits` (host BF16 bits, column-major
// [vocab, columns]) to `out`, most likely first, as natural-log probabilities over the column.
void append_top_tokens(std::span<const std::uint16_t> logits, std::int32_t vocab,
                       std::int32_t columns, TopTokenScores& out);

} // namespace ninfer::models::gemma4
