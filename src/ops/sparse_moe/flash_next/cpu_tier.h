#pragma once

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCpuTierExperts       = 512;
inline constexpr int kCpuTierMaxAssign     = 160;
inline constexpr int kCpuTierSlotBytes     = 2764800;
inline constexpr int kCpuTierHidden        = 2560;
inline constexpr int kCpuTierIntermediate  = 640;

// Host view of one expert. Cold experts have host pointers into the pinned copy.
// Device pointers are the bytes the GPU table names (VRAM for a resident expert,
// the mapped pinned address for a cold one) and are what an adaptive swap exchanges.
struct CpuTierExpert {
    const std::uint8_t* gate_codes        = nullptr;
    const std::uint8_t* gate_scales       = nullptr;
    const std::uint8_t* down_codes        = nullptr;
    const std::uint8_t* down_scales       = nullptr;
    std::uint8_t* gate_codes_device       = nullptr;
    std::uint8_t* gate_scales_device      = nullptr;
    std::uint8_t* down_codes_device       = nullptr;
    std::uint8_t* down_scales_device      = nullptr;
    float gate_weight_div = 1.f;
    float gate_input_div  = 1.f;
    float down_weight_div = 1.f;
    float down_input_div  = 1.f;
    bool resident = false;
};

struct CpuTierLayer {
    CpuTierExpert* experts                         = nullptr;  // 512
    std::uint32_t* device_resident                 = nullptr;  // 16 words
    const std::uint8_t** device_gate_codes         = nullptr;
    const std::uint8_t** device_gate_scales        = nullptr;
    const std::uint8_t** device_down_codes         = nullptr;
    const std::uint8_t** device_down_scales        = nullptr;
};

// Mapped device pointers. The MoE kernels publish into these and spin on `done`.
struct CpuTierDeviceView {
    std::uint32_t* seq       = nullptr;
    std::uint32_t* done      = nullptr;
    std::uint32_t* swapping  = nullptr;
    int* meta                = nullptr;  // [assignments, layer]
    int* ids                 = nullptr;
    std::uint8_t* act_codes  = nullptr;
    std::uint8_t* act_scales = nullptr;
    void* rows               = nullptr;  // bf16, [assignment, hidden]
    std::uint8_t* scratch    = nullptr;
    int scratch_slots        = 0;
    int* stage_flag          = nullptr;  // 512
    int* stage_count         = nullptr;
};

struct CpuTierStart {
    const CpuTierLayer* layers = nullptr;
    int layer_count            = 0;
    int workers                = 0;  // 0 -> NINFER_EXPERT_CPU_THREADS or 8
    bool adapt                 = true;
};

void cpu_tier_start(const CpuTierStart& start, CpuTierDeviceView* out);
void cpu_tier_stop() noexcept;
[[nodiscard]] const CpuTierDeviceView* cpu_tier_device() noexcept;

}  // namespace ninfer::ops
