#include "models/gemma4/program/program.h"

#include "core/arena.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/program/top_tokens.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

constexpr std::int32_t kScoreTile = 128;

runtime::ExecutionTiming timed(std::chrono::steady_clock::time_point start) {
    runtime::ExecutionTiming timing;
    timing.device_wait_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - start)
                                       .count());
    return timing;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 program: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

// Workspace bytes of the widest layer for a `chunk`-token prefill.
std::size_t layer_bytes(const LoadedText& model, std::uint32_t chunk) {
    std::size_t widest = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        widest = std::max(widest, gemma4_layer_workspace_bytes(layer, static_cast<int>(chunk)));
    }
    return widest;
}

} // namespace

struct Program::Impl {
    LoadedText* model   = nullptr;
    cudaStream_t stream = nullptr;
    ops::Bf16GemmContext gemm;
    int pages              = 0;
    int vocab              = 0;
    int hidden             = 0;
    std::uint32_t capacity = 0;
    std::uint32_t chunk    = 0;
    DeviceArena workspace;
    DeviceBuffer id_buf;
    DeviceBuffer pos_buf;
    DeviceBuffer hidden_buf;
    DeviceBuffer table_buf;
    DeviceBuffer config_buf;
    DeviceBuffer counts_buf;
    std::vector<DeviceBuffer> k_store;
    std::vector<DeviceBuffer> v_store;
    std::vector<Tensor> k_pages;
    std::vector<Tensor> v_pages;
    Tensor block;
    std::vector<Gemma4CacheLayer> caches;

    bool binding = false;
    std::shared_ptr<detail::RequestPlan> plan;
    runtime::BeginSummary begin;
    std::uint32_t bind_reused = 0;
    // Tokens whose K/V sit at positions [0, size) of the cache; kept across requests for reuse.
    std::vector<TokenId> resident;
    std::vector<std::uint16_t> top_logits;

    bool occupied         = false;
    std::uint64_t epoch   = 1;
    std::uint32_t fed     = 0;
    bool has_next         = false;
    TokenId next_id       = 0;
    bool prefilled        = false;
    TokenId pending_token = 0;
    std::vector<TokenId> pending_tokens;
    std::uint64_t transaction = 1;
    runtime::ExecutionTiming pending_timing;
    SequenceHandle handle;

    static std::size_t workspace_bytes(const LoadedText& loaded, std::uint32_t chunk) {
        const std::size_t layers = layer_bytes(loaded, chunk);
        const std::size_t logits = static_cast<std::size_t>(loaded.vocab) * kScoreTile * 2U +
                                   gemma4_logits_workspace_bytes(loaded.output_head, kScoreTile);
        const std::size_t sampling = ops::sampling_workspace_capacity_bytes(loaded.vocab, 1, 1);
        return std::max(layers, logits + sampling) + (64U << 20U);
    }

    Impl(LoadedText& loaded, std::uint32_t token_capacity, std::uint32_t prefill_chunk,
         cudaStream_t device_stream)
        : model(&loaded), stream(device_stream), gemm(device_stream), vocab(loaded.vocab),
          hidden(loaded.hidden), capacity(token_capacity),
          chunk(std::clamp<std::uint32_t>(prefill_chunk, 16U, 2048U)),
          workspace(workspace_bytes(loaded, chunk)) {
        if (vocab <= 0 || hidden <= 0 || loaded.layers.empty() || capacity == 0) {
            throw std::invalid_argument("gemma4 program: model is empty");
        }
        pages      = static_cast<int>((capacity + 63U) / 64U);
        id_buf     = DeviceBuffer(static_cast<std::size_t>(chunk) * 4);
        pos_buf    = DeviceBuffer(static_cast<std::size_t>(chunk) * 4);
        hidden_buf = DeviceBuffer(static_cast<std::size_t>(hidden) * chunk * 2);
        config_buf = DeviceBuffer(sizeof(ops::SamplingConfig));
        counts_buf = DeviceBuffer(static_cast<std::size_t>(vocab) * 4);
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
        for (int page = 0; page < pages; ++page) { table[static_cast<std::size_t>(page)] = page; }
        table_buf = DeviceBuffer(table.size() * 4);
        table_buf.copy_from_host(table.data(), table_buf.bytes);
        block = Tensor(table_buf.p, DType::I32, {pages});
        k_store.reserve(loaded.layers.size());
        v_store.reserve(loaded.layers.size());
        k_pages.reserve(loaded.layers.size());
        v_pages.reserve(loaded.layers.size());
        caches.reserve(loaded.layers.size());
        for (const Gemma4LayerWeights& layer : loaded.layers) {
            const std::size_t bytes = static_cast<std::size_t>(layer.head_dim) * 64U *
                                      static_cast<std::size_t>(layer.kv_heads) *
                                      static_cast<std::size_t>(pages) * 2U;
            k_store.emplace_back(bytes);
            v_store.emplace_back(bytes);
            k_store.back().fill(0);
            v_store.back().fill(0);
        }
        for (std::size_t index = 0; index < loaded.layers.size(); ++index) {
            const Gemma4LayerWeights& layer = loaded.layers[index];
            k_pages.emplace_back(
                k_store[index].p, DType::BF16,
                std::initializer_list<std::int32_t>{layer.head_dim, 64, layer.kv_heads, pages});
            v_pages.emplace_back(
                v_store[index].p, DType::BF16,
                std::initializer_list<std::int32_t>{layer.head_dim, 64, layer.kv_heads, pages});
        }
        for (std::size_t index = 0; index < loaded.layers.size(); ++index) {
            caches.push_back({&k_pages[index], &v_pages[index], &block});
        }
    }

    [[nodiscard]] bool owns(SequenceHandle sequence) const {
        return program::HandleAccess::owner(sequence) == this && occupied &&
               program::HandleAccess::epoch(sequence) == epoch &&
               program::HandleAccess::lane(sequence).value == 0;
    }

    void clear_lane() {
        occupied  = false;
        fed       = 0;
        has_next  = false;
        prefilled = false;
        pending_tokens.clear();
    }

    // Runs `count` tokens at positions [first, first + count) through every layer. The hidden
    // states land in hidden_buf columns [0, count). Earlier positions must already be in the KV.
    void run_layers(const TokenId* ids, std::uint32_t first, std::uint32_t count) {
        if (count == 0 || count > chunk || first + count > capacity || first > resident.size()) {
            throw std::logic_error("gemma4 forward is outside the chunk or KV capacity");
        }
        resident.resize(first);
        std::vector<std::int32_t> positions(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            positions[index] = static_cast<std::int32_t>(first + index);
        }
        check(cudaStreamSynchronize(stream), "stream sync");
        id_buf.copy_from_host(ids, static_cast<std::size_t>(count) * 4);
        pos_buf.copy_from_host(positions.data(), static_cast<std::size_t>(count) * 4);
        const auto width = static_cast<std::int32_t>(count);
        Tensor id_tensor(id_buf.p, DType::I32, {width});
        Tensor position_tensor(pos_buf.p, DType::I32, {width});
        Tensor hidden_tensor(hidden_buf.p, DType::BF16, {hidden, width});
        workspace.reset();
        gemma4_layers(id_tensor, hidden_tensor, model->token_embedding, model->layers, caches,
                      position_tensor, gemm, workspace, stream);
        resident.insert(resident.end(), ids, ids + count);
    }

    // Prompt tokens whose K/V can be kept from the resident cache. At least the last prompt
    // token is always recomputed, because its logits pick the first output token.
    [[nodiscard]] std::uint32_t reusable_tokens(const detail::RequestPlan& request) const {
        if (!request.allow_prefix_reuse || request.tokens.size() < 2) { return 0; }
        const std::size_t limit = std::min(resident.size(), request.tokens.size() - 1);
        std::size_t length      = 0;
        while (length < limit && resident[length] == request.tokens[length]) { ++length; }
        return static_cast<std::uint32_t>(length);
    }

    [[nodiscard]] Tensor hidden_columns(std::int32_t start, std::int32_t count) const {
        Tensor all(hidden_buf.p, DType::BF16, {hidden, static_cast<std::int32_t>(chunk)});
        return all.slice(1, start, count);
    }

    void upload_sampling(const ResolvedSamplingParameters& sampling) {
        ops::SamplingConfig config;
        config.temperature       = sampling.temperature;
        config.top_k             = sampling.top_k;
        config.top_p             = sampling.top_p;
        config.min_p             = sampling.min_p;
        config.presence_penalty  = sampling.presence_penalty;
        config.frequency_penalty = sampling.frequency_penalty;
        config.seed              = sampling.seed;
        if (sampling.presence_penalty != 0.0F || sampling.frequency_penalty != 0.0F) {
            counts_buf.fill(0);
            config.token_counts = static_cast<std::int32_t*>(counts_buf.p);
        }
        config_buf.copy_from_host(&config, sizeof(config));
    }

    // Samples the next token from hidden column `column`, which predicts logical position
    // `position`.
    [[nodiscard]] TokenId sample_column(std::int32_t column, std::uint32_t position,
                                        std::int32_t purpose) {
        workspace.reset();
        Tensor logits  = workspace.alloc(DType::BF16, {vocab, 1});
        Tensor out     = workspace.alloc(DType::I32, {1});
        Tensor logical = workspace.alloc(DType::I32, {1});
        gemma4_logits(hidden_columns(column, 1), logits, model->final_norm, model->output_head,
                      model->softcap, model->rms_eps, gemm, workspace, stream);
        const auto logical_position = static_cast<std::int32_t>(position);
        check(cudaMemcpyAsync(logical.data, &logical_position, 4, cudaMemcpyHostToDevice, stream),
              "position upload");
        ops::sample(logits, out, vocab, static_cast<const ops::SamplingConfig*>(config_buf.p),
                    logical, purpose, workspace, stream);
        std::int32_t token = 0;
        check(cudaMemcpyAsync(&token, out.data, 4, cudaMemcpyDeviceToHost, stream), "token copy");
        check(cudaStreamSynchronize(stream), "sample sync");
        return token;
    }

    // Log-probabilities of targets[0..count) under hidden columns [start, start + count).
    void score_columns(std::int32_t start, std::int32_t count, const TokenId* targets,
                       std::vector<float>& out, TopTokenScores* top) {
        for (std::int32_t done = 0; done < count; done += kScoreTile) {
            const std::int32_t columns = std::min(kScoreTile, count - done);
            workspace.reset();
            Tensor logits     = workspace.alloc(DType::BF16, {vocab, columns});
            Tensor target_ids = workspace.alloc(DType::I32, {columns});
            Tensor logprobs   = workspace.alloc(DType::FP32, {columns});
            gemma4_logits(hidden_columns(start + done, columns), logits, model->final_norm,
                          model->output_head, model->softcap, model->rms_eps, gemm, workspace,
                          stream);
            check(cudaMemcpyAsync(target_ids.data, targets + done,
                                  static_cast<std::size_t>(columns) * 4, cudaMemcpyHostToDevice,
                                  stream),
                  "target upload");
            ops::target_logprobs(logits, target_ids, vocab, logprobs, stream);
            const std::size_t at = out.size();
            out.resize(at + static_cast<std::size_t>(columns));
            check(cudaMemcpyAsync(out.data() + at, logprobs.data,
                                  static_cast<std::size_t>(columns) * 4, cudaMemcpyDeviceToHost,
                                  stream),
                  "logprob copy");
            if (top != nullptr) {
                top_logits.resize(static_cast<std::size_t>(vocab) * columns);
                check(cudaMemcpyAsync(top_logits.data(), logits.data, top_logits.size() * 2,
                                      cudaMemcpyDeviceToHost, stream),
                      "top logits copy");
            }
            check(cudaStreamSynchronize(stream), "score sync");
            if (top != nullptr) { append_top_tokens(top_logits, vocab, columns, *top); }
        }
    }

    PendingBatch stage(TokenId token, runtime::ExecutionTiming timing) {
        pending_token  = token;
        pending_tokens = {token};
        pending_timing = timing;
        ++transaction;
        return program::HandleAccess::make_pending(this, transaction,
                                                   std::span<const SequenceHandle>(&handle, 1), 1,
                                                   pending_tokens, {}, 1, timing);
    }
};

Program::Program(LoadedText& model, std::uint32_t capacity, std::uint32_t prefill_chunk,
                 cudaStream_t stream)
    : impl_(std::make_unique<Impl>(model, capacity, prefill_chunk, stream)) {}

Program::~Program() = default;

std::size_t Program::kv_bytes_per_token(const LoadedText& model) {
    std::size_t bytes = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        bytes += 2U * static_cast<std::size_t>(layer.head_dim) * layer.kv_heads * 2U;
    }
    return bytes;
}

std::size_t Program::fixed_bytes(const LoadedText& model, std::uint32_t prefill_chunk) {
    const std::uint32_t chunk = std::clamp<std::uint32_t>(prefill_chunk, 16U, 2048U);
    return Impl::workspace_bytes(model, chunk) +
           static_cast<std::size_t>(model.hidden) * chunk * 2U +
           static_cast<std::size_t>(model.vocab) * 4U;
}

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    if (prompt.tokens().empty()) { throw std::invalid_argument("gemma4 prompt is empty"); }
    if (prompt.tokens().size() > impl_->capacity) {
        throw std::invalid_argument("gemma4 prompt does not fit the KV capacity");
    }
    auto plan                             = std::make_shared<detail::RequestPlan>();
    plan->tokens                          = prompt.tokens();
    plan->sampling                        = options.sampling;
    plan->allow_prefix_reuse              = options.allow_prefix_reuse;
    plan->summary.prompt_tokens           = static_cast<std::uint32_t>(plan->tokens.size());
    plan->summary.requested_output_tokens = options.requested_output_tokens;
    plan->summary.effective_output_tokens = options.requested_output_tokens;
    plan->summary.effective_limit_reason  = FinishReason::OutputLimit;
    const std::uint32_t room              = impl_->capacity - plan->summary.prompt_tokens + 1U;
    if (plan->summary.effective_output_tokens > room) {
        plan->summary.effective_output_tokens = room;
        plan->summary.effective_limit_reason  = FinishReason::ContextCapacity;
    }
    plan->summary.publish_continuation = false;
    return RequestBasePlan(std::move(plan));
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                         TopTokenScores* top) {
    if (impl_->occupied || impl_->binding) {
        throw std::logic_error("gemma4 causal scoring needs an idle program");
    }
    const auto& tokens      = prompt.tokens();
    const std::size_t count = tokens.size();
    if (count < 2 || count > impl_->capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= count) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    const auto predictors = static_cast<std::uint32_t>(count - 1);
    std::vector<float> output;
    output.reserve(count - first_target);
    for (std::uint32_t cursor = 0; cursor < predictors;) {
        const std::uint32_t width = std::min(impl_->chunk, predictors - cursor);
        impl_->run_layers(tokens.data() + cursor, cursor, width);
        const std::uint32_t first = std::max(cursor, first_target - 1U);
        if (first < cursor + width) {
            impl_->score_columns(static_cast<std::int32_t>(first - cursor),
                                 static_cast<std::int32_t>(cursor + width - first),
                                 tokens.data() + first + 1, output, top);
        }
        cursor += width;
    }
    if (output.size() != count - first_target) {
        throw std::logic_error("gemma4 causal score produced the wrong number of logprobs");
    }
    return output;
}

std::optional<SourceCandidate> Program::inspect_source(const RequestBasePlan& base,
                                                       std::optional<CheckpointHandle> checkpoint,
                                                       bool, std::span<const CheckpointHandle>,
                                                       std::span<const CheckpointHandle>) const {
    if (checkpoint) { return std::nullopt; }
    SourceCandidate source;
    source.reused_tokens =
        impl_->occupied || impl_->binding ? 0 : impl_->reusable_tokens(*base.impl_);
    source.remaining_work = runtime::make_prefill_work(
        source.reused_tokens, base.summary().prompt_tokens - source.reused_tokens, 0, 0,
        impl_->chunk);
    return source;
}

PrefixShortlistKey Program::checkpoint_key(CheckpointHandle, std::uint32_t) const { return {}; }

runtime::ContextResourceUsage
Program::checkpoint_footprint(std::span<const CheckpointHandle>) const {
    return {};
}

CheckpointSummary Program::checkpoint_summary(CheckpointHandle) const { return {}; }

CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle) const { return {}; }

bool Program::checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const { return false; }

std::uint32_t Program::checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&,
                                                    std::uint32_t target) const {
    return target;
}

std::uint64_t Program::checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                std::span<const CheckpointHandle>) const {
    return 0;
}

bool Program::valid_checkpoint(CheckpointHandle) const noexcept { return false; }

bool Program::release_checkpoint(CheckpointHandle) noexcept { return false; }

bool Program::revoke_snapshot(ResumeState&) noexcept { return false; }

runtime::ContextResourceUsage Program::snapshot_resources(const ResumeState&) const { return {}; }

std::size_t Program::host_bytes_released(std::span<const CheckpointHandle>) const { return 0; }

std::optional<std::size_t> Program::pause_host_bytes(SequenceHandle) const { return std::nullopt; }

std::size_t Program::release_redundant_host(std::span<const CheckpointHandle>,
                                            std::optional<SequenceHandle>) {
    return 0;
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    if (units.size() != 1 || !impl_->owns(units[0].sequence)) {
        throw std::logic_error("gemma4 reservation does not match the active sequence");
    }
    return {.reserved = true};
}

bool Program::reclaim_capture_reservation(runtime::ContextResourceUsage) { return false; }

void Program::release_units(std::span<const SequenceHandle>) noexcept {}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState*,
                                          ExecutionUnitKind, std::uint32_t) {
    if (source.checkpoint || source.reused_tokens > impl_->reusable_tokens(*base.impl_)) {
        return {.reserved = false, .source_valid = false, .capacity_possible = true};
    }
    if (lane.value != 0 || impl_->binding || impl_->occupied) {
        return {.reserved = false, .source_valid = true, .capacity_possible = true};
    }
    impl_->plan        = base.impl_;
    impl_->binding     = true;
    impl_->bind_reused = source.reused_tokens;
    impl_->begin       = {.prompt_tokens        = base.summary().prompt_tokens,
                          .reused_prompt_tokens = source.reused_tokens,
                          .prefix_reuse_path =
                        source.reused_tokens ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
    return {.reserved = true};
}

bool Program::start_capture(SequenceHandle) { return false; }

bool Program::capture_is_input(SequenceHandle) const { return false; }

std::optional<CapturePreparation> Program::prepare_capture(SequenceHandle) { return std::nullopt; }

void Program::skip_capture(SequenceHandle) {}

ContextReclaimPlan Program::plan_reclaim(std::span<const CheckpointHandle>,
                                         std::span<const CheckpointHandle>,
                                         runtime::ContextResourceUsage) const {
    return {};
}

std::vector<ContextRelease> Program::plan_releases(std::span<const CheckpointHandle>,
                                                   std::span<const CheckpointHandle>,
                                                   runtime::ContextResourceUsage) const {
    return {};
}

bool Program::start_demote(const ContextDemotion&) { return false; }

bool Program::start_pause(SequenceHandle, bool, runtime::ExecutionTiming*) { return false; }

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    ContextProgress progress;
    progress.kind = ContextOperationKind::Bind;
    if (!impl_->binding) { return progress; }
    progress.advanced = true;
    progress.complete = true;
    if (cancellation.requested()) {
        impl_->binding = false;
        impl_->plan.reset();
        return progress;
    }
    impl_->upload_sampling(impl_->plan->sampling);
    impl_->occupied  = true;
    impl_->fed       = impl_->bind_reused;
    impl_->has_next  = false;
    impl_->prefilled = false;
    impl_->pending_tokens.clear();
    impl_->handle =
        program::HandleAccess::make_sequence(impl_.get(), runtime::LaneId{0}, impl_->epoch);
    impl_->binding     = false;
    progress.published = true;
    progress.sequence  = impl_->handle;
    return progress;
}

bool Program::has_context_transaction() const noexcept { return impl_->binding; }

bool Program::context_blocks(SequenceHandle) const noexcept { return false; }

bool Program::recovery_pending(SequenceHandle) const noexcept { return false; }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider*) {
    if (!impl_->owns(sequence) || impl_->prefilled) {
        throw std::logic_error("gemma4 prefill sequence is not waiting");
    }
    const auto start          = std::chrono::steady_clock::now();
    const auto& tokens        = impl_->plan->tokens;
    const auto total          = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t width = std::min(impl_->chunk, total - impl_->fed);
    impl_->run_layers(tokens.data() + impl_->fed, impl_->fed, width);
    impl_->fed += width;
    PrefillProgress progress;
    progress.summary                 = impl_->begin;
    progress.processed_prompt_tokens = width;
    if (impl_->fed < total) {
        check(cudaStreamSynchronize(impl_->stream), "prefill sync");
        progress.timing = timed(start);
        return progress;
    }
    const TokenId next = impl_->sample_column(static_cast<std::int32_t>(width) - 1, impl_->fed,
                                              ops::kSamplePurposePrefill);
    impl_->prefilled   = true;
    progress.complete  = true;
    progress.timing    = timed(start);
    progress.pending.emplace(impl_->stage(next, progress.timing));
    return progress;
}

ReplayProgress Program::advance_replay(SequenceHandle, runtime::ExecutionTiming*) {
    throw std::logic_error("gemma4 replay is not wired");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider*) {
    if (sequences.size() != 1 || budgets.size() != 1 || !impl_->owns(sequences[0]) ||
        !impl_->has_next) {
        throw std::logic_error("gemma4 decode has no accepted token to feed");
    }
    if (impl_->fed >= impl_->capacity) {
        throw std::logic_error("gemma4 decode ran past the KV capacity");
    }
    const auto start = std::chrono::steady_clock::now();
    const TokenId id = impl_->next_id;
    impl_->run_layers(&id, impl_->fed, 1);
    ++impl_->fed;
    impl_->has_next    = false;
    const TokenId next = impl_->sample_column(0, impl_->fed, ops::kSamplePurposeDecode);
    return impl_->stage(next, timed(start));
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle>, std::span<const TokenId>,
                              std::uint32_t, std::span<const std::optional<std::uint32_t>>,
                              runtime::ExecutionTiming*) {
    throw std::logic_error("gemma4 has no forced target-control tokens");
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    if (program::HandleAccess::owner(pending) != impl_.get() ||
        program::HandleAccess::transaction(pending) != impl_->transaction ||
        decisions.size() != 1) {
        throw std::logic_error("gemma4 commit does not match the pending batch");
    }
    const runtime::CommitDecision& decision = decisions[0];
    if (!decision.failed && !decision.cancelled && decision.accepted_tokens == 1) {
        impl_->next_id  = impl_->pending_token;
        impl_->has_next = true;
    } else if (decision.accepted_tokens != 0) {
        throw std::logic_error("gemma4 commit accepted an unexpected token count");
    }
    impl_->pending_tokens.clear();
    program::HandleAccess::consume(pending);
    CommitResult result;
    result.row_count           = 1;
    result.timing              = impl_->pending_timing;
    result.rows[0].disposition = decision.failed ? runtime::CommitDisposition::FailedReleased
                                 : decision.cancelled
                                     ? runtime::CommitDisposition::CancelledReleased
                                 : decision.terminal ? runtime::CommitDisposition::Finishable
                                                     : runtime::CommitDisposition::Active;
    return result;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    if (program::HandleAccess::owner(pending) != impl_.get() ||
        program::HandleAccess::transaction(pending) != impl_->transaction) {
        return {};
    }
    impl_->pending_tokens.clear();
    program::HandleAccess::consume(pending);
    return {.status = runtime::ConsumeStatus::Consumed, .row_count = 1};
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    if (!impl_->owns(sequence)) { return {}; }
    ++impl_->epoch;
    impl_->clear_lane();
    impl_->plan.reset();
    return {.status = runtime::ConsumeStatus::Consumed};
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (!impl_->owns(sequence)) { return {}; }
    ++impl_->epoch;
    impl_->clear_lane();
    impl_->plan.reset();
    return {.status = runtime::ConsumeStatus::Consumed};
}

void Program::fail_all_cleanup() noexcept {
    impl_->resident.clear();
    impl_->binding = false;
    impl_->pending_tokens.clear();
    impl_->clear_lane();
    impl_->plan.reset();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    PhysicalUsageSnapshot usage;
    usage.capacity.main_kv_pages = static_cast<std::uint32_t>(impl_->pages);
    return usage;
}

MemorySummary Program::memory_summary() const noexcept {
    MemorySummary summary;
    summary.max_context              = impl_->capacity;
    summary.kv_capacity              = impl_->capacity;
    summary.kv_cache                 = KvCacheStorage::BFloat16;
    summary.workspace.capacity_bytes = impl_->workspace.capacity();
    return summary;
}

void Program::reset_memory_peaks() noexcept { impl_->workspace.reset_peak(); }

} // namespace ninfer::models::gemma4
