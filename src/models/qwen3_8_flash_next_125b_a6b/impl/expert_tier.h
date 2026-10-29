#pragma once

#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"

#include <memory>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {

struct RuntimeModelView;

// Owns the VRAM slots, the pinned cold-expert copy, and the device pointer tables.
// Empty unless --expert-vram-gib is set.
class ExpertTierKeep {
public:
    ExpertTierKeep();
    ~ExpertTierKeep();
    ExpertTierKeep(ExpertTierKeep&&) noexcept;
    ExpertTierKeep& operator=(ExpertTierKeep&&) noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Rewrites NVFP4 routed-expert pointers. Hottest experts (from the profile, else expert-id
// order) go to VRAM. The rest are copied into one mapped host allocation and read over PCIe.
void apply_expert_tier(RuntimeModelView& runtime, const BindingPlan& plan, ExpertTierKeep& keep);

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
