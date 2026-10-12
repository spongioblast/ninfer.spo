#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>
#include "models/qwen3_8_flash_next/program/v3_program.h"

#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/variant.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile profile, std::unique_ptr<ArtifactLoadPlan> value)
        : weights_profile(profile), plan(std::move(value)) {}

    WeightsProfile weights_profile;
    std::unique_ptr<ArtifactLoadPlan> plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

artifact::MaterializationPlan& LoadPlan::materialization() {
    if (!impl_) { throw std::logic_error("Flash-Next load plan is empty"); }
    return impl_->plan->materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail

namespace ninfer::models::qwen3_8_flash_next_125b_a6b {
namespace {
constexpr ModelSamplingDefaults kDefaults{
    .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
    .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.80F, .presence_penalty = 1.5F},
};
} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model != model_id) { throw std::runtime_error("unknown Flash-Next model identity"); }
    return kDefaults;
}

Package::WeightsProfile Package::weights_profile(const artifact::Reader& reader) {
    const auto& binding = reader.directory().bindings.at(kPleTableName);
    if (!binding.whole_object || binding.parts.size() != 1) {
        throw artifact::ArtifactError("PLE requires a complete raw table");
    }
    const auto& table = reader.directory().tensor(binding.parts.front().object);
    if (table.format == "fp8_e4m3fn") { return WeightsProfile::Nvfp4; }
    if (table.format == "bf16") { return WeightsProfile::Nvfp4Bf16Ple; }
    throw artifact::ArtifactError("unsupported PLE table format");
}

namespace {

// Measured 110 MiB per Flash-Next host state image; the budget uses an upper bound because the
// exact layout is only known once the program is built.
constexpr std::uint64_t kHostStateImageEstimateBytes = 128ULL << 20;
constexpr std::uint64_t kProcessOverheadBytes        = 8ULL << 30;

std::string gib_text(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", static_cast<double>(bytes) / (1ULL << 30U));
    return text;
}

void check_host_ram_budget(const EngineOptions& options, std::uint64_t resident_bytes) {
    const std::uint64_t host_kv    = options.context_cache.host_kv_capacity_bytes;
    const std::uint64_t host_state = static_cast<std::uint64_t>(
                                         options.context_cache.host_state_slots) *
                                     kHostStateImageEstimateBytes;
    const std::uint64_t planned = resident_bytes + host_kv + host_state + kProcessOverheadBytes;
    const std::string breakdown = gib_text(planned) + " GiB (resident PLE " +
                                  gib_text(resident_bytes) + ", host KV " + gib_text(host_kv) +
                                  ", host state <= " + gib_text(host_state) + ", overhead " +
                                  gib_text(kProcessOverheadBytes) + ")";
    const std::uint64_t budget = options.host_residency.host_ram_budget_bytes;
    if (budget != 0 && planned > budget) {
        throw std::invalid_argument("host RAM plan " + breakdown + " exceeds --host-ram-budget-gib " +
                                    gib_text(budget));
    }
    std::clog << "host RAM plan: " << breakdown << ", budget "
              << (budget == 0 ? std::string("unchecked") : gib_text(budget) + " GiB") << '\n';
}

} // namespace

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile profile) {
    if (profile != weights_profile(binder.reader())) {
        throw std::logic_error("invalid Flash-Next profile");
    }
    const auto& residency = options.host_residency;
    check_host_ram_budget(options,
                          ple_resident(residency) ? ple_table_bytes(binder.reader()) : 0);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        profile,
        plan_artifact(binder, qwen3_8_flash_next::startup_features(options), residency)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (!plan.impl_) { throw std::invalid_argument("Flash-Next load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan->bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (!model.impl_) { throw std::invalid_argument("Flash-Next loaded model is empty"); }
    return qwen3_8_flash_next::make_frontend(
        model.impl_->data.frontend, {.vision_enabled    = model.impl_->data.runtime.features.vision,
                                     .max_context       = options.max_context,
                                     .media_cache_bytes = options.media_cache_bytes,
                                     .media_live_bytes  = options.media_live_bytes,
                                     .media_preprocess_threads = options.media_preprocess_threads,
                                     .grammar_cache_bytes = options.grammar_cache_bytes});
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile profile) {
    return qwen3_8_flash_next::make_sequence_planner<detail::Variant>(device, options, profile);
}

Package::LoadFacts Package::load_facts(const LoadedModel& model) {
    if (!model.impl_) { throw std::invalid_argument("Flash-Next loaded model is empty"); }
    const auto& stats = model.impl_->data.backing.stats();
    return {.upload_seconds      = stats.upload_seconds,
            .read_bytes          = stats.read_bytes,
            .h2d_bytes           = stats.h2d_bytes,
            .peak_staging_bytes  = stats.peak_staging_bytes,
            .device_object_count = stats.device_object_count,
            .host_object_count   = stats.host_object_count};
}

std::unique_ptr<qwen3_8_flash_next::v3::Program>
Package::create_v3_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                           const StartupObserver& observer) {
    if (!model.impl_) { throw std::invalid_argument("Flash-Next loaded model is empty"); }
    return qwen3_8_flash_next::create_v3_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device, observer);
}

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b
