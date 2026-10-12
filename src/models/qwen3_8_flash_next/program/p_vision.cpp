// Vision prefill for one fresh prompt.

#include "models/qwen3_8_flash_next/program/p3.h"

namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime {

VisionPrefillPlan P3Program::plan_vision(const qwen3_8_flash_next::PreparedPromptData& prompt,
                                         std::optional<qwen3_8_flash_next::PreparedPromptData>& media) const {
    if (!workspace_plan.vision) {
        throw std::logic_error("Vision prompt has no startup workspace plan");
    }
    auto control = std::make_shared<qwen3_8_flash_next::VisionControlPlan>(
        qwen3_8_flash_next::plan_vision_control(prompt));
    VisionPrefillPlan vision;
    vision.control_plan        = control;
    const bool prepare_mtp     = backend == SpeculativeBackend::Mtp;
    std::uint32_t previous_end = 0;
    for (std::size_t index = 0; index < control->items.size(); ++index) {
        const qwen3_8_flash_next::VisionItemControlPlan& item = control->items[index];
        const std::uint32_t begin =
            prepare_mtp && item.token_begin != 0 ? item.token_begin - 1U : item.token_begin;
        if (begin < previous_end) {
            throw std::invalid_argument("vision item consumer spans overlap");
        }
        if (item.merged_count > workspace_plan.vision->max_merged_tokens ||
            schedule::VisionContext::workspace_bytes(prompt.vision_items[index].patch_count,
                                                     item.merged_count) >
                workspace_plan.vision->encode_peak_bytes) {
            throw std::invalid_argument("vision item exceeds the Program workspace envelope");
        }
        previous_end = item.token_end;
        vision.uses.push_back(VisionUseSpan{
            .begin               = begin,
            .end                 = item.token_end,
            .prepared_item_index = static_cast<std::uint32_t>(index),
        });
        vision.max_merged_count = std::max(vision.max_merged_count, item.merged_count);
    }
    if (vision.uses.empty()) { throw std::logic_error("Vision prompt produced no prefill span"); }
    media = prompt;
    return vision;
}

void P3Program::attach_vision() {
    if (!lane().request.vision || !lane().request.media || !workspace_plan.vision) {
        throw std::logic_error("Vision prefill has no startup workspace plan");
    }
    VisionPrefillPlan& vision                      = *lane().request.vision;
    qwen3_8_flash_next::PreparedPromptData& prompt = *lane().request.media;
    if (vision.uses.empty() || !vision.control_plan) {
        throw std::logic_error("Vision suffix plan has no prepared metadata");
    }
    const std::uint32_t first = vision.uses.front().prepared_item_index;
    auto control              = std::make_shared<qwen3_8_flash_next::VisionControl>(
        qwen3_8_flash_next::build_vision_control(prompt, *vision.control_plan, first));
    for (VisionUseSpan& use : vision.uses) {
        if (use.prepared_item_index < first) {
            throw std::logic_error("Vision suffix item order changed during admission");
        }
        use.control_index = use.prepared_item_index - first;
        if (use.control_index >= control->items.size()) {
            throw std::logic_error("Vision suffix control does not cover a planned item");
        }
    }
    vision.control = std::move(control);
    vision.control_plan.reset();
    lane().vision = std::make_unique<schedule::VisionPrefillSession>(
        device, model, DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
        *workspace_plan.vision, prompt, vision, lane().vision_handoff_peak);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::qwen3_8_flash_next_125b_a6b_runtime
