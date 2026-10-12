#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/qwen3_8_flash_next/program/v3_program.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"
#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>

#include <cstdint>
#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct FlashNextInstance {
    using ModelContract = models::qwen3_8_flash_next::v3::RuntimeTypes;
    using LoadedModel   = models::qwen3_8_flash_next_125b_a6b::detail::LoadedModel;

    FlashNextInstance(std::unique_ptr<LoadedModel> model_in,
                      models::qwen3_8_flash_next::Frontend frontend_in,
                      ModelSamplingDefaults sampling_in, std::uint32_t capacity_in)
        : model(std::move(model_in)), frontend(std::move(frontend_in)), sampling(sampling_in),
          capacity(capacity_in) {}

    std::unique_ptr<LoadedModel> model;
    models::qwen3_8_flash_next::Frontend frontend;
    ModelSamplingDefaults sampling;
    KvCapacityResolution kv_capacity_resolution;
    std::uint32_t capacity = 0;
    std::unique_ptr<models::qwen3_8_flash_next::v3::Program> program;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    std::unique_ptr<FlashNextInstance> flash;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
