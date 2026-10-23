#include "models/gemma4/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "models/gemma4/execution/draft_forward.h"
#include "models/gemma4/execution/projection.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/execution/vision.h"
#include "models/gemma4/program/prefill_graphs.h"
#include "models/gemma4/program/sliding_ring.h"
#include "models/gemma4/program/top_tokens.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
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

constexpr std::uint32_t kMinChunk = 16;
constexpr std::uint32_t kMaxChunk = 16384;
// A round verifies the anchor and its drafts in one forward of at most kMinChunk tokens.
constexpr std::uint32_t kMaxDraftTokens = kMinChunk - 1;
// Round slots: the drafter's position, the anchor then the drafts (the verify ids), the verify
// positions, and the verify argmax.
constexpr std::size_t kRoundPosition = 0;
constexpr std::size_t kRoundIds      = 1;
constexpr std::size_t kRoundVerify   = kRoundIds + kMinChunk;
constexpr std::size_t kRoundOut      = kRoundVerify + kMinChunk;
constexpr std::size_t kRoundSlots    = kRoundOut + kMinChunk;

// Workspace bytes of the widest layer for a `chunk`-token prefill.
std::size_t layer_bytes(const LoadedText& model, std::uint32_t chunk) {
    std::size_t widest = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        widest = std::max(widest, gemma4_layer_workspace_bytes(layer, static_cast<int>(chunk)));
    }
    return widest;
}

// The widest sliding window, 0 when every layer is global.
int sliding_window(const LoadedText& model) {
    int window = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        if (!layer.global_layer) { window = std::max(window, layer.window); }
    }
    return window;
}

std::size_t layer_kv_bytes(const Gemma4LayerWeights& layer, int pages) {
    return 2U * static_cast<std::size_t>(layer.head_dim) * kKvPageTokens *
           static_cast<std::size_t>(layer.kv_heads) * static_cast<std::size_t>(pages) * 2U;
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
    DeviceBuffer ring_table_buf;
    DeviceBuffer config_buf;
    DeviceBuffer counts_buf;
    // One-token decode step as a CUDA graph. step_buf holds the token id, its position and the
    // logical position it predicts; step_host stages those three and receives the sampled token.
    DeviceBuffer step_buf;
    PinnedHostBuffer step_host{4 * sizeof(std::int32_t)};
    DecodeGraphDefinition decode_definition;
    DecodeGraphExecutable decode_graph;
    PrefillGraphs prefill_graphs;
    std::vector<DeviceBuffer> k_store;
    std::vector<DeviceBuffer> v_store;
    std::vector<Tensor> k_pages;
    std::vector<Tensor> v_pages;
    Tensor block;
    // Sliding layers: `ring_pages` physical pages behind `ring_block`. ring_high is the highest
    // position written since the cache last restarted at 0; a slot may hold it instead of the
    // position ring_pages * 64 below.
    int window     = 0;
    int ring_pages = 0;
    Tensor ring_block;
    std::int64_t ring_high = -1;
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
    std::vector<std::int32_t> pending_counts;
    DeviceBuffer vision_embed;
    bool vision_ready     = false;
    int vision_columns    = 0;
    std::uint64_t transaction = 1;
    runtime::ExecutionTiming pending_timing;
    SequenceHandle handle;

    // MTP. A round drafts up to draft_tokens tokens from carry_buf (the target's final-normed
    // hidden state at fed - 1), verifies the anchor next_id and the drafts in one forward at
    // positions [fed, fed + drafts], and offers the argmax of every column up to the first
    // mismatch. Rejected K/V past the committed length is overwritten by the next round.
    std::uint32_t draft_tokens = 0;
    bool speculating           = false;
    bool round_pending         = false;
    std::uint32_t round_drafts = 0;
    std::uint32_t round_agreed = 0;
    DeviceBuffer carry_buf;
    // I32 round slots (kRound* offsets), mirrored in round_host. A round with draft_tokens
    // drafts runs as one CUDA graph; a shorter one near the budget or capacity runs eagerly.
    DeviceBuffer round_buf;
    PinnedHostBuffer round_host{kRoundSlots * sizeof(std::int32_t)};
    DecodeGraphDefinition round_definition;
    DecodeGraphExecutable round_graph;
    std::vector<TokenId> round_ids;
    SpeculativeStats spec_stats;

    static std::size_t workspace_bytes(const LoadedText& loaded, std::uint32_t chunk) {
        const std::size_t layers = layer_bytes(loaded, chunk);
        const std::size_t logits = static_cast<std::size_t>(loaded.vocab) * kScoreTile * 2U +
                                   gemma4_logits_workspace_bytes(loaded.output_head, kScoreTile);
        const std::size_t sampling = ops::sampling_workspace_capacity_bytes(loaded.vocab, 1, 1);
        const std::size_t draft =
            loaded.draft != nullptr ? gemma4_draft_workspace_bytes(*loaded.draft) : 0;
        std::size_t vision = 0;
        if (loaded.vision) {
            const int pooled  = loaded.vision->default_output_length;
            const int patches = pooled * loaded.vision->pooling_kernel_size *
                                loaded.vision->pooling_kernel_size;
            vision = gemma4_vision_workspace_bytes(*loaded.vision, patches, pooled);
        }
        return std::max({layers, logits + sampling, draft, vision}) + (64U << 20U);
    }

    Impl(LoadedText& loaded, std::uint32_t token_capacity, std::uint32_t prefill_chunk,
         std::uint32_t drafts, cudaStream_t device_stream)
        : model(&loaded), stream(device_stream), gemm(device_stream), vocab(loaded.vocab),
          hidden(loaded.hidden), capacity(token_capacity),
          chunk(std::clamp<std::uint32_t>(prefill_chunk, kMinChunk, kMaxChunk)),
          workspace(workspace_bytes(loaded, chunk)), draft_tokens(drafts) {
        if (vocab <= 0 || hidden <= 0 || loaded.layers.empty() || capacity == 0) {
            throw std::invalid_argument("gemma4 program: model is empty");
        }
        if (draft_tokens > kMaxDraftTokens || (draft_tokens > 0 && loaded.draft == nullptr)) {
            throw std::invalid_argument("gemma4 program: MTP needs the assistant and 1..15 drafts");
        }
        if (draft_tokens > 0) {
            carry_buf = DeviceBuffer(static_cast<std::size_t>(hidden) * 2);
            round_buf = DeviceBuffer(kRoundSlots * sizeof(std::int32_t));
        }
        pages      = static_cast<int>((capacity + 63U) / 64U);
        id_buf     = DeviceBuffer(static_cast<std::size_t>(chunk) * 4);
        pos_buf    = DeviceBuffer(static_cast<std::size_t>(chunk) * 4);
        hidden_buf = DeviceBuffer(static_cast<std::size_t>(hidden) * chunk * 2);
        config_buf = DeviceBuffer(sizeof(ops::SamplingConfig));
        counts_buf = DeviceBuffer(static_cast<std::size_t>(vocab) * 4);
        step_buf   = DeviceBuffer(3 * sizeof(std::int32_t));
        window     = sliding_window(loaded);
        ring_pages = window > 0 ? std::min(pages, sliding_ring_pages(chunk, window)) : pages;
        const std::vector<std::int32_t> table = ring_block_table(pages, pages);
        table_buf                             = DeviceBuffer(table.size() * 4);
        table_buf.copy_from_host(table.data(), table_buf.bytes);
        block                                = Tensor(table_buf.p, DType::I32, {pages});
        const std::vector<std::int32_t> ring = ring_block_table(pages, ring_pages);
        ring_table_buf                       = DeviceBuffer(ring.size() * 4);
        ring_table_buf.copy_from_host(ring.data(), ring_table_buf.bytes);
        ring_block = Tensor(ring_table_buf.p, DType::I32, {pages});
        k_store.reserve(loaded.layers.size());
        v_store.reserve(loaded.layers.size());
        k_pages.reserve(loaded.layers.size());
        v_pages.reserve(loaded.layers.size());
        caches.reserve(loaded.layers.size());
        for (const Gemma4LayerWeights& layer : loaded.layers) {
            const int layer_pages   = layer.global_layer ? pages : ring_pages;
            const std::size_t bytes = layer_kv_bytes(layer, layer_pages) / 2U;
            k_store.emplace_back(bytes);
            v_store.emplace_back(bytes);
            k_store.back().fill(0);
            v_store.back().fill(0);
            const std::initializer_list<std::int32_t> shape{layer.head_dim, kKvPageTokens,
                                                            layer.kv_heads, layer_pages};
            k_pages.emplace_back(k_store.back().p, DType::BF16, shape);
            v_pages.emplace_back(v_store.back().p, DType::BF16, shape);
        }
        for (std::size_t index = 0; index < loaded.layers.size(); ++index) {
            const Tensor* table_of = loaded.layers[index].global_layer ? &block : &ring_block;
            caches.push_back({&k_pages[index], &v_pages[index], table_of});
        }
    }

    [[nodiscard]] bool owns(SequenceHandle sequence) const {
        return program::HandleAccess::owner(sequence) == this && occupied &&
               program::HandleAccess::epoch(sequence) == epoch &&
               program::HandleAccess::lane(sequence).value == 0;
    }

    void clear_lane() {
        occupied       = false;
        fed            = 0;
        has_next       = false;
        prefilled      = false;
        round_pending  = false;
        vision_ready   = false;
        vision_columns = 0;
        pending_tokens.clear();
    }

    void encode_vision(const detail::RequestPlan& request) {
        vision_ready = true;
        if (request.images.empty()) { return; }
        if (!model->vision) {
            throw std::invalid_argument("gemma4 vision weights are missing");
        }
        const Gemma4VisionWeights& vision = *model->vision;
        int total = 0;
        for (const Gemma4PreparedImage& image : request.images) { total += image.valid; }
        if (static_cast<int>(request.image_scatter.size()) != total) {
            throw std::invalid_argument("gemma4 image scatter does not match the encoder output");
        }
        vision_embed   = DeviceBuffer(static_cast<std::size_t>(hidden) * total * 2);
        vision_columns = total;
        int written    = 0;
        for (const Gemma4PreparedImage& image : request.images) {
            const int pooled = image.tokens / (vision.pooling_kernel_size * vision.pooling_kernel_size);
            DeviceBuffer patch_buf(image.patches.size() * 2);
            DeviceBuffer pos_buf(image.positions.size() * 4);
            DeviceBuffer mask_buf(image.mask.size() * 4);
            DeviceBuffer encoded_buf(static_cast<std::size_t>(vision.text_hidden) * pooled * 2);
            patch_buf.copy_from_host(image.patches.data(), patch_buf.bytes);
            pos_buf.copy_from_host(image.positions.data(), pos_buf.bytes);
            mask_buf.copy_from_host(image.mask.data(), mask_buf.bytes);
            Tensor patches(patch_buf.p, DType::BF16, {vision.patch_pixels, image.tokens});
            Tensor positions(pos_buf.p, DType::I32, {image.tokens, 2});
            Tensor mask(mask_buf.p, DType::I32, {image.tokens});
            Tensor encoded(encoded_buf.p, DType::BF16, {vision.text_hidden, pooled});
            workspace.reset();
            gemma4_vision_encode(patches, positions, mask, encoded, vision, gemm, workspace, stream);
            check(cudaStreamSynchronize(stream), "vision sync");
            const std::vector<int> kept =
                gemma4_vision_kept_bins(image.positions.data(), image.tokens,
                                        vision.pooling_kernel_size, pooled);
            if (static_cast<int>(kept.size()) != image.valid) {
                throw std::invalid_argument("gemma4 vision packed fewer tokens than the prompt");
            }
            std::vector<std::uint16_t> host(
                static_cast<std::size_t>(vision.text_hidden) * pooled);
            encoded_buf.copy_to_host(host.data(), host.size() * 2);
            std::vector<std::uint16_t> packed(
                static_cast<std::size_t>(vision.text_hidden) * image.valid);
            for (int column = 0; column < image.valid; ++column) {
                const int src = kept[static_cast<std::size_t>(column)];
                std::memcpy(packed.data() + static_cast<std::size_t>(vision.text_hidden) * column,
                            host.data() + static_cast<std::size_t>(vision.text_hidden) * src,
                            static_cast<std::size_t>(vision.text_hidden) * 2);
            }
            vision_embed.copy_from_host(
                packed.data(), packed.size() * 2,
                static_cast<std::size_t>(written) * vision.text_hidden * 2);
            written += image.valid;
        }
    }

    // Layers only, reading id_buf, pos_buf and the caches. Captured for the bucket widths.
    void prefill_layers_body(std::int32_t width) {
        workspace.reset();
        Tensor id_tensor(id_buf.p, DType::I32, {width});
        Tensor position_tensor(pos_buf.p, DType::I32, {width});
        Tensor hidden_tensor(hidden_buf.p, DType::BF16, {hidden, width});
        gemma4_layers(id_tensor, hidden_tensor, model->token_embedding, model->layers, caches,
                      position_tensor, gemm, workspace, stream);
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
        const Tensor* visual_embeddings = nullptr;
        const Tensor* visual_indices    = nullptr;
        Tensor visual_src;
        Tensor visual_idx;
        std::vector<std::int32_t> local_idx;
        if (plan && vision_ready && vision_columns > 0) {
            for (std::size_t i = 0; i < plan->image_scatter.size(); ++i) {
                const auto pos = static_cast<std::uint32_t>(plan->image_scatter[i]);
                if (pos >= first && pos < first + count) {
                    local_idx.push_back(static_cast<std::int32_t>(pos - first));
                }
            }
            if (!local_idx.empty()) {
                visual_src = workspace.alloc(DType::BF16,
                                             {hidden, static_cast<std::int32_t>(local_idx.size())});
                visual_idx = workspace.alloc(DType::I32, {static_cast<std::int32_t>(local_idx.size())});
                std::size_t packed = 0;
                for (std::size_t i = 0; i < plan->image_scatter.size(); ++i) {
                    const auto pos = static_cast<std::uint32_t>(plan->image_scatter[i]);
                    if (pos < first || pos >= first + count) { continue; }
                    check(cudaMemcpyAsync(
                              static_cast<std::uint8_t*>(visual_src.data) + packed * hidden * 2U,
                              static_cast<std::uint8_t*>(vision_embed.p) + i * hidden * 2U,
                              static_cast<std::size_t>(hidden) * 2U, cudaMemcpyDeviceToDevice, stream),
                          "visual column copy");
                    ++packed;
                }
                check(cudaMemcpyAsync(visual_idx.data, local_idx.data(), local_idx.size() * 4,
                                      cudaMemcpyHostToDevice, stream),
                      "visual index upload");
                visual_embeddings = &visual_src;
                visual_indices    = &visual_idx;
            }
        }
        const bool graphed =
            visual_embeddings == nullptr &&
            prefill_graphs.run(count, stream, [this, width] { prefill_layers_body(width); });
        if (!graphed) {
            gemma4_layers(id_tensor, hidden_tensor, model->token_embedding, model->layers, caches,
                          position_tensor, gemm, workspace, stream, visual_embeddings,
                          visual_indices);
        }
        resident.insert(resident.end(), ids, ids + count);
        note_written(first, count);
    }

    void note_written(std::uint32_t first, std::uint32_t count) {
        const std::int64_t last = static_cast<std::int64_t>(first) + count - 1;
        ring_high               = first == 0 ? last : std::max(ring_high, last);
    }

    // Prompt tokens whose K/V can be kept from the resident cache. At least the last prompt
    // token is always recomputed, because its logits pick the first output token.
    [[nodiscard]] std::uint32_t reusable_tokens(const detail::RequestPlan& request) const {
        if (!request.images.empty() || !request.allow_prefix_reuse || request.tokens.size() < 2) {
            return 0;
        }
        const std::size_t limit = std::min(resident.size(), request.tokens.size() - 1);
        std::size_t length      = 0;
        while (length < limit && resident[length] == request.tokens[length]) { ++length; }
        // The sliding window before the reuse point must still be in the ring.
        const std::int64_t oldest =
            std::max<std::int64_t>(0, static_cast<std::int64_t>(length) - (window - 1));
        if (length > 0 && oldest + std::int64_t{ring_pages} * kKvPageTokens <= ring_high) {
            return 0;
        }
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

    // Layers, logits and sampling for one token, reading only device buffers whose addresses
    // stay fixed, so it can be captured once and replayed for every decode step.
    void decode_body() {
        workspace.reset();
        auto* step = static_cast<std::int32_t*>(step_buf.p);
        Tensor id_tensor(step, DType::I32, {1});
        Tensor position(step + 1, DType::I32, {1});
        Tensor logical(step + 2, DType::I32, {1});
        Tensor column(hidden_buf.p, DType::BF16, {hidden, 1});
        gemma4_layers(id_tensor, column, model->token_embedding, model->layers, caches, position,
                      gemm, workspace, stream);
        Tensor logits = workspace.alloc(DType::BF16, {vocab, 1});
        Tensor out    = workspace.alloc(DType::I32, {1});
        gemma4_logits(column, logits, model->final_norm, model->output_head, model->softcap,
                      model->rms_eps, gemm, workspace, stream);
        ops::sample(logits, out, vocab, static_cast<const ops::SamplingConfig*>(config_buf.p),
                    logical, ops::kSamplePurposeDecode, workspace, stream);
        auto* host = static_cast<std::int32_t*>(step_host.data());
        check(cudaMemcpyAsync(host + 3, out.data, 4, cudaMemcpyDeviceToHost, stream), "token copy");
    }

    // Feeds `id` at position `fed` and returns the token sampled for position fed + 1.
    [[nodiscard]] TokenId decode_step(TokenId id) {
        if (fed >= capacity || fed > resident.size()) {
            throw std::logic_error("gemma4 decode is outside the KV capacity");
        }
        resident.resize(fed);
        auto* host = static_cast<std::int32_t*>(step_host.data());
        host[0]    = id;
        host[1]    = static_cast<std::int32_t>(fed);
        host[2]    = static_cast<std::int32_t>(fed + 1);
        check(cudaMemcpyAsync(step_buf.p, host, step_buf.bytes, cudaMemcpyHostToDevice, stream),
              "decode step upload");
        if (!decode_graph.ready()) {
            decode_definition.capture(stream, [this] { decode_body(); });
            decode_graph.instantiate(decode_definition);
        }
        decode_graph.launch(stream);
        check(cudaStreamSynchronize(stream), "decode sync");
        resident.push_back(id);
        note_written(fed, 1);
        return host[3];
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

    // The final-normed hidden state of hidden column `column` into carry_buf.
    void carry_column(std::int32_t column) {
        Tensor carry(carry_buf.p, DType::BF16, {hidden, 1});
        ops::rmsnorm(hidden_columns(column, 1), norm_weight(model->final_norm, hidden),
                     model->rms_eps, false, carry, stream);
    }

    // Drafts `drafts` tokens from carry_buf, then verifies the anchor and the drafts in one
    // forward and writes the argmax of every column. Reads and writes only round_buf, carry_buf,
    // hidden_buf, the caches and the workspace, so it can be captured once and replayed.
    void round_body(std::uint32_t drafts) {
        workspace.reset();
        auto* slots = static_cast<std::int32_t*>(round_buf.p);
        const Tensor position(slots + kRoundPosition, DType::I32, {1});
        Tensor carry(carry_buf.p, DType::BF16, {hidden, 1});
        for (std::uint32_t step = 0; step < drafts; ++step) {
            const Tensor token(slots + kRoundIds + step, DType::I32, {1});
            Tensor next(slots + kRoundIds + step + 1, DType::I32, {1});
            gemma4_draft_step(token, position, carry, next, model->token_embedding, *model->draft,
                              caches, gemm, workspace, stream);
        }
        const auto width = static_cast<std::int32_t>(drafts + 1U);
        const Tensor ids(slots + kRoundIds, DType::I32, {width});
        const Tensor positions(slots + kRoundVerify, DType::I32, {width});
        Tensor columns(hidden_buf.p, DType::BF16, {hidden, width});
        gemma4_layers(ids, columns, model->token_embedding, model->layers, caches, positions, gemm,
                      workspace, stream);
        Tensor logits = workspace.alloc(DType::BF16, {vocab, width});
        Tensor out(slots + kRoundOut, DType::I32, {width});
        gemma4_logits(columns, logits, model->final_norm, model->output_head, model->softcap,
                      model->rms_eps, gemm, workspace, stream);
        ops::argmax(logits, out, vocab, stream);
    }

    PendingBatch speculative_round(std::uint32_t remaining,
                                   std::chrono::steady_clock::time_point start) {
        const std::uint32_t room = std::min(capacity - fed, chunk);
        const std::uint32_t drafts =
            std::min({draft_tokens, remaining > 0 ? remaining - 1U : 0U, room - 1U});
        const std::uint32_t width = drafts + 1U;
        if (fed == 0 || fed > resident.size()) {
            throw std::logic_error("gemma4 round is outside the resident cache");
        }
        auto* host                  = static_cast<std::int32_t*>(round_host.data());
        host[kRoundPosition]        = static_cast<std::int32_t>(fed) - 1;
        host[kRoundIds]             = next_id;
        for (std::uint32_t index = 0; index < width; ++index) {
            host[kRoundVerify + index] = static_cast<std::int32_t>(fed + index);
        }
        check(cudaMemcpyAsync(round_buf.p, host, kRoundOut * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, stream),
              "round upload");
        if (drafts == draft_tokens) {
            if (!round_graph.ready()) {
                round_definition.capture(stream, [this, drafts] { round_body(drafts); });
                round_graph.instantiate(round_definition);
            }
            round_graph.launch(stream);
        } else {
            round_body(drafts);
        }
        check(cudaMemcpyAsync(host, round_buf.p, kRoundSlots * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, stream),
              "round copy");
        check(cudaStreamSynchronize(stream), "round sync");
        round_ids.assign(host + kRoundIds, host + kRoundIds + width);
        pending_tokens.assign(host + kRoundOut, host + kRoundOut + width);
        resident.resize(fed);
        resident.insert(resident.end(), round_ids.begin(), round_ids.end());
        note_written(fed, width);
        std::uint32_t agreed = 0;
        while (agreed < drafts && round_ids[agreed + 1U] == pending_tokens[agreed]) { ++agreed; }
        round_drafts   = drafts;
        round_agreed   = agreed;
        round_pending  = true;
        has_next       = false;
        pending_counts = {static_cast<std::int32_t>(agreed + 1U)};
        pending_timing = timed(start);
        ++transaction;
        return program::HandleAccess::make_pending(this, transaction,
                                                   std::span<const SequenceHandle>(&handle, 1), 1,
                                                   pending_tokens, pending_counts, width,
                                                   pending_timing);
    }

    // Keeps the first `accepted` verified tokens of the pending round.
    void commit_round(std::uint32_t accepted, bool terminal) {
        if (accepted > round_agreed + 1U) {
            throw std::logic_error("gemma4 commit accepted more tokens than the round verified");
        }
        round_pending = false;
        ++spec_stats.rounds;
        spec_stats.drafted_tokens += round_drafts;
        const std::uint32_t drafted = std::min(round_agreed, accepted);
        spec_stats.accepted_tokens += drafted;
        for (std::uint32_t index = 0; index < drafted; ++index) {
            ++spec_stats.accepted_per_position[index];
        }
        resident.resize(fed + accepted);
        if (accepted == 0) { return; }
        fed += accepted;
        next_id  = pending_tokens[accepted - 1U];
        has_next = true;
        if (!terminal) { carry_column(static_cast<std::int32_t>(accepted) - 1); }
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
                 std::uint32_t draft_tokens, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(model, capacity, prefill_chunk, draft_tokens, stream)) {}

Program::~Program() = default;

std::size_t Program::kv_bytes_per_token(const LoadedText& model) {
    std::size_t bytes = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        if (layer.global_layer) {
            bytes += 2U * static_cast<std::size_t>(layer.head_dim) * layer.kv_heads * 2U;
        }
    }
    return bytes;
}

std::size_t Program::fixed_bytes(const LoadedText& model, std::uint32_t prefill_chunk) {
    const std::uint32_t chunk = std::clamp<std::uint32_t>(prefill_chunk, kMinChunk, kMaxChunk);
    const int window          = sliding_window(model);
    std::size_t rings         = 0;
    for (const Gemma4LayerWeights& layer : model.layers) {
        if (!layer.global_layer) {
            rings += layer_kv_bytes(layer, sliding_ring_pages(chunk, window));
        }
    }
    return Impl::workspace_bytes(model, chunk) + rings +
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
    plan->images                          = std::move(prompt.images_);
    plan->image_scatter                   = std::move(prompt.image_scatter_);
    plan->sampling                        = options.sampling;
    plan->allow_prefix_reuse              = options.allow_prefix_reuse && plan->images.empty();
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
    const ResolvedSamplingParameters& sampling = impl_->plan->sampling;
    impl_->upload_sampling(sampling);
    impl_->speculating = impl_->draft_tokens > 0 && sampling.temperature <= 0.0F &&
                         sampling.presence_penalty == 0.0F && sampling.frequency_penalty == 0.0F;
    impl_->spec_stats  = {};
    if (impl_->speculating) {
        impl_->spec_stats.backend      = SpeculativeBackend::Mtp;
        impl_->spec_stats.enabled      = true;
        impl_->spec_stats.draft_window = impl_->draft_tokens;
        impl_->spec_stats.accepted_per_position.assign(impl_->draft_tokens, 0);
    }
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
    if (!impl_->vision_ready) { impl_->encode_vision(*impl_->plan); }
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
    if (impl_->speculating) { impl_->carry_column(static_cast<std::int32_t>(width) - 1); }
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
    if (impl_->speculating) {
        return impl_->speculative_round(budgets[0].generated_tokens_remaining, start);
    }
    const TokenId next = impl_->decode_step(impl_->next_id);
    ++impl_->fed;
    impl_->has_next = false;
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
    if (impl_->round_pending) {
        const bool dropped = decision.failed || decision.cancelled;
        impl_->commit_round(dropped ? 0U : decision.accepted_tokens, dropped || decision.terminal);
    } else if (!decision.failed && !decision.cancelled && decision.accepted_tokens == 1) {
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
    const SpeculativeStats& stats             = impl_->spec_stats;
    result.rows[0].speculative_counters       = {stats.rounds, stats.drafted_tokens,
                                                 stats.accepted_tokens, stats.fallback_steps};
    if (decision.failed || decision.cancelled) { result.rows[0].speculative = stats; }
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
    if (impl_->round_pending) {
        impl_->round_pending = false;
        impl_->resident.resize(impl_->fed);
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
    return {.status = runtime::ConsumeStatus::Consumed, .speculative = std::move(impl_->spec_stats)};
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (!impl_->owns(sequence)) { return {}; }
    ++impl_->epoch;
    impl_->clear_lane();
    impl_->plan.reset();
    return {.status = runtime::ConsumeStatus::Consumed, .speculative = std::move(impl_->spec_stats)};
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
