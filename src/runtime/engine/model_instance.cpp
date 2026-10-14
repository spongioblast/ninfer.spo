#include "runtime/engine/model_instance.h"
#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/nvml_memory.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include "models/qwen3_5/measurement.h"
#include "models/registry.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <iomanip>
#include <iostream>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

std::size_t current_free_device_bytes();
void log_startup_memory(const EngineOptions& options);

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

template <class Contract>
concept GenerationContract =
    requires(typename Contract::Program& program, typename Contract::PreparedPrompt prompt,
             const runtime::ResolvedExecutionOptions& options,
             const typename Contract::RequestBasePlan& base,
             typename Contract::SequenceHandle sequence, typename Contract::PendingBatch pending,
             std::span<const typename Contract::SequenceHandle> sequences,
             std::span<const runtime::RoundBudget> budgets,
             std::span<const runtime::CommitDecision> decisions) {
        program.plan_request(std::move(prompt), options);
        program.inspect_source(base, std::optional<typename Contract::CheckpointHandle>{});
        program.reserve_units(std::span<const typename Contract::ExecutionUnit>{});
        program.advance_prefill(sequence, nullptr, nullptr);
        program.decode(sequences, budgets, nullptr, nullptr);
        program.commit(std::move(pending), decisions);
        program.finish(sequence);
        program.abort(sequence);
        program.has_context_transaction();
        program.memory_summary();
        program.physical_usage();
    };

static_assert(GenerationContract<models::qwen3_5::RuntimeTypes>);
static_assert(GenerationContract<models::qwen3_8_flash_next::v3::RuntimeTypes>);

models::Architecture peek_architecture(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    const auto& names  = config.at("architectures");
    if (!names.is_array() || names.size() != 1 || !names[0].is_string() ||
        !config.at("model_type").is_string()) {
        throw std::invalid_argument("artifact text config architecture is invalid");
    }
    return models::resolve_architecture(names[0].get<std::string>(),
                                        config.at("model_type").get<std::string>());
}

ConstructedModel construct_flash_next(EngineOptions& options, DeviceContext& device,
                                      artifact::Reader& reader, Clock::time_point start) {
    namespace flash = models::qwen3_8_flash_next_125b_a6b;
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    artifact::Binder binder(reader);
    const auto profile = flash::Package::weights_profile(reader);
    auto plan          = flash::Package::plan_load(binder, options, profile);
    binding.complete();
    auto materialized = artifact::materialize(reader, std::move(plan.materialization()), device,
                                              &options.startup_observer);
    auto loaded = flash::Package::construct_loaded_model(std::move(plan), std::move(materialized));
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto model_frontend = flash::Package::make_frontend(*loaded, options);
    const auto sampling = flash::Package::sampling_defaults(flash::Package::model_id);
    auto instance       = std::make_unique<FlashNextInstance>(
        std::move(loaded), std::move(model_frontend), sampling, options.max_context);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::string signature = "qwen3_8_flash_next-prefill-1";
    auto context_cost           = resolve_context_machine_cost(
        {.hardware_class =
                       context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
                   .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = flash::Package::make_sequence_planner(device, options, profile);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes());
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = 0;
    instance->kv_capacity_resolution          = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = flash::Package::create_v3_program(*instance->model, std::move(sequence),
                                                          device, options.startup_observer);
    instance->program->configure_activation(options);
    if (!options.steering_pack.empty()) {
        const auto pack = models::qwen3_8_flash_next::read_steering_pack(options.steering_pack);
        instance->program->activate_steering(&pack);
    }
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto facts = flash::Package::load_facts(*instance->model);
    LoadSummary summary;
    summary.architecture =
        std::string(models::architecture_name(models::Architecture::Qwen3_8FlashNext));
    summary.model_name =
        reader.directory().metadata.value("name", std::string(flash::Package::model_id));
    summary.prefill_signature = signature;
    summary.weight_formats.push_back(
        profile == flash::Package::WeightsProfile::Nvfp4 ? "nvfp4" : "nvfp4-bf16-ple");
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = facts.upload_seconds;
    summary.artifact_bytes_read  = facts.read_bytes;
    summary.host_to_device_bytes = facts.h2d_bytes;
    summary.peak_staging_bytes   = facts.peak_staging_bytes;
    summary.device_object_count  = facts.device_object_count;
    summary.host_object_count    = facts.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    ConstructedModel constructed;
    constructed.flash        = std::move(instance);
    constructed.load         = std::move(summary);
    constructed.context_cost = std::move(context_cost.model);
    log_startup_memory(options);
    return constructed;
}

// Under WDDM cudaMemGetInfo does not see memory held by other processes or WSL guests, so KV
// sizing takes the smaller of it and NVML.
std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (const auto nvml = nvml_device_memory(device)) {
        free_bytes = std::min<std::size_t>(free_bytes, static_cast<std::size_t>(nvml->free_bytes));
    }
    return free_bytes;
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); }

void log_startup_memory(const EngineOptions& options) {
    std::size_t cuda_free  = 0;
    std::size_t cuda_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&cuda_free, &cuda_total));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const PinnedHostUsage pinned = pinned_host_usage();
    const std::uint64_t host_kv  = options.context_cache.host_kv_capacity_bytes;
    std::ostringstream line;
    line << std::fixed << std::setprecision(2)
         << "pinned host memory: " << gib(pinned.current_bytes) << " GiB (peak "
         << gib(pinned.peak_bytes) << ", host KV " << gib(host_kv) << ", host state + staging "
         << gib(pinned.current_bytes > host_kv ? pinned.current_bytes - host_kv : 0) << ")";
    if (pinned.ceiling_bytes != 0) { line << ", ceiling " << gib(pinned.ceiling_bytes); }
    line << "; device memory: cudaMemGetInfo free " << gib(cuda_free) << " of " << gib(cuda_total)
         << " GiB";
    if (const auto nvml = nvml_device_memory(device)) {
        line << ", NVML used " << gib(nvml->used_bytes) << " free " << gib(nvml->free_bytes)
             << " of " << gib(nvml->total_bytes) << " GiB";
    } else {
        line << ", NVML unavailable";
    }
    std::clog << line.str() << '\n';
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    if (!options.capture_path.empty()) {
        auto ancestor = std::filesystem::weakly_canonical(options.capture_path);
        const auto model_dir =
            std::filesystem::weakly_canonical(options.artifact_path).parent_path();
        while (!ancestor.empty()) {
            if (ancestor == model_dir || std::filesystem::exists(ancestor / "config.json") ||
                std::filesystem::exists(ancestor / "model.safetensors.index.json")) {
                throw std::invalid_argument("--capture-path must not be inside a model directory");
            }
            const auto parent = ancestor.parent_path();
            if (parent == ancestor) { break; }
            ancestor = parent;
        }
        if (options.capture_sites.empty()) {
            throw std::invalid_argument("capture-sites must be nonempty");
        }
        std::set<std::string> unique;
        for (const auto& site : options.capture_sites) {
            if ((site != "prompt_last" && site != "completion_last") ||
                !unique.insert(site).second) {
                throw std::invalid_argument("invalid or duplicate capture site: " + site);
            }
        }
        if (options.speculative.backend != SpeculativeBackend::None &&
            options.speculative.backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument("capture supports ordinary and MTP decoding");
        }
    }
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    cache.device_state_slots        = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .grammar_cache_bytes      = options.grammar_cache_bytes,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

std::string resource_text(artifact::Binder& binder, std::string_view role) {
    const auto bytes = binder.host_object(binder.resource("text", role));
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

ConstructedModel construct_gemma4(EngineOptions& options, DeviceContext& device,
                                  artifact::Reader& reader,
                                  std::chrono::steady_clock::time_point start) {
    // One lane until the program batches. Global layers keep full-length KV; sliding layers use a
    // fixed ring, counted in the fixed bytes.
    options.max_concurrency = 1;
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument("gemma4 KV cache is bf16 only");
    }
    artifact::Binder binder(reader);
    const std::string tokenizer = resource_text(binder, "tokenizer.json");
    const std::string chat      = resource_text(binder, "chat_template.jinja");
    auto loaded                 = std::make_unique<models::gemma4::LoadedText>(
        models::gemma4::load_text(options.artifact_path.string(), device));
    device.synchronize();
    std::set<std::string> formats;
    for (const Weight* weight : {&loaded->token_embedding, &loaded->output_head}) {
        formats.emplace(artifact::format_name(weight->qtype));
    }
    for (const auto& layer : loaded->layers) {
        for (const Weight* weight : {&layer.qkv, &layer.o_proj, &layer.gate_up, &layer.down}) {
            formats.emplace(artifact::format_name(weight->qtype));
        }
    }
    const std::size_t kHeadroom  = options.kv_capacity.mode == KvCapacityMode::Automatic &&
                                          options.kv_capacity.automatic_headroom_bytes != 0
                                       ? options.kv_capacity.automatic_headroom_bytes
                                       : std::size_t{3} << 30U;
    const std::size_t free_bytes = current_free_device_bytes();
    const std::size_t fixed = models::gemma4::Program::fixed_bytes(*loaded, options.prefill_chunk);
    const std::size_t per_token = models::gemma4::Program::kv_bytes_per_token(*loaded);
    const std::size_t fit =
        free_bytes > fixed + kHeadroom ? (free_bytes - fixed - kHeadroom) / per_token : 0;
    std::uint64_t tokens = std::min<std::uint64_t>(options.max_context, fit / 64U * 64U);
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        if (options.kv_capacity.explicit_tokens > fit) {
            throw std::invalid_argument(
                "gemma4 explicit KV capacity does not fit in device memory");
        }
        tokens = std::min<std::uint64_t>(tokens, options.kv_capacity.explicit_tokens);
    }
    if (tokens < 64) {
        throw std::invalid_argument("gemma4 KV capacity does not fit in device memory");
    }
    options.max_context = static_cast<std::uint32_t>(tokens);
    auto program        = std::make_unique<models::gemma4::Program>(*loaded, options.max_context,
                                                                    options.prefill_chunk, device.stream);
    device.synchronize();
    auto instance                         = std::make_unique<Gemma4Instance>(std::move(loaded),
                                                                             models::gemma4::Frontend(tokenizer, chat),
                                                                             std::move(program), options.max_context);
    instance->kv_capacity_resolution.mode = options.kv_capacity.mode;
    instance->kv_capacity_resolution.resolved_tokens  = options.max_context;
    instance->kv_capacity_resolution.main_page_groups = (options.max_context + 63U) / 64U;
    instance->kv_capacity_resolution.maximum_main_page_groups =
        instance->kv_capacity_resolution.main_page_groups;
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const std::string signature                                    = "gemma4-prefill-1";
    auto context_cost                                              = resolve_context_machine_cost(
        {.hardware_class =
                                                          context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
                                                      .prefill_signature = signature},
        options.context_cost.preset_path);
    options.context_cache.host_capacity_bytes = 0;
    LoadSummary summary;
    summary.architecture = std::string(models::architecture_name(models::Architecture::Gemma4));
    summary.model_name   = reader.directory().metadata.value("name", std::string("gemma4"));
    summary.prefill_signature = signature;
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.context_cost = std::move(context_cost.summary);
    ConstructedModel constructed;
    constructed.gemma        = std::move(instance);
    constructed.load         = std::move(summary);
    constructed.context_cost = std::move(context_cost.model);
    log_startup_memory(options);
    return constructed;
}

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    const auto architecture = peek_architecture(reader);
    if (architecture == models::Architecture::Qwen3_8FlashNext) {
        return construct_flash_next(options, device, reader, start);
    }
    if (architecture == models::Architecture::Gemma4) {
        return construct_gemma4(options, device, reader, start);
    }
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes());
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    instance->kv_capacity_resolution          = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    ConstructedModel constructed;
    constructed.instance     = std::move(instance);
    constructed.load         = std::move(summary);
    constructed.context_cost = std::move(context_cost.model);
    log_startup_memory(options);
    return constructed;
}

} // namespace ninfer::runtime
