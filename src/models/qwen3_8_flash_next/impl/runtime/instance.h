#pragma once

#include "models/qwen3_8_flash_next_125b_a6b/impl/variant.h"
#include <ninfer/models/qwen3_8_flash_next/runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

using Variant                        = ::ninfer::models::qwen3_8_flash_next_125b_a6b::detail::Variant;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan    = qwen3_8_flash_next::SequencePlan<Variant>;
using SequencePlanner = qwen3_8_flash_next::SequencePlanner<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
// Adaptive MTP selects each round's draft count in [kAdaptiveMtpMinimumDraftTokens,
// kMaximumMtpDraftTokens]. Shorter windows lose more acceptance than they save in round time.
inline constexpr std::uint32_t kAdaptiveMtpMinimumDraftTokens = 3;
static_assert(kAdaptiveMtpMinimumDraftTokens <= kMaximumMtpDraftTokens);

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
