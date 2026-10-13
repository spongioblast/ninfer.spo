#pragma once

#include "models/gemma4/execution/text_forward.h"

#include <memory>
#include <string>
#include <vector>

namespace ninfer {
class DeviceContext;
namespace artifact {
class MaterializedArtifact;
}
} // namespace ninfer

namespace ninfer::models::gemma4 {

struct LoadedText {
    LoadedText();
    ~LoadedText();
    LoadedText(LoadedText&&) noexcept;
    LoadedText& operator=(LoadedText&&) noexcept;
    LoadedText(const LoadedText&) = delete;
    LoadedText& operator=(const LoadedText&) = delete;

    std::unique_ptr<artifact::MaterializedArtifact> artifact;
    std::vector<Gemma4LayerWeights> layers;
    Weight token_embedding;
    const void* final_norm      = nullptr;
    Weight output_head;
    int vocab                   = 0;
    int hidden                  = 0;
    float softcap               = 30.0f;
    float rms_eps               = 1.0e-6f;
};

[[nodiscard]] LoadedText load_text(const std::string& artifact_path, DeviceContext& device);

} // namespace ninfer::models::gemma4
