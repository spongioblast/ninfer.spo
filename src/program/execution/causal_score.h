#pragma once

// Causal scoring staging shared by the model programs. The model prefills the prompt in chunks
// and projects each tile of hidden columns to logits; this file checks the request, stages the
// scored columns into fixed tiles, and turns a tile of logits into target-token logprobs.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/types.h"
#include "program/storage/kv_address_space.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::program {

inline void validate_causal_score_request(std::size_t token_count, std::uint32_t capacity,
                                          std::uint32_t first_target, bool has_media) {
    if (token_count < 2 || token_count > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (has_media) { throw std::invalid_argument("causal scoring accepts text tokens only"); }
}

// Predictor column p (the hidden state after token p) scores token p + 1. Columns before
// first_target - 1 are prefilled but not scored.
class CausalScoreTiles {
public:
    CausalScoreTiles(std::span<const TokenId> tokens, std::uint32_t first_target, Tensor tile,
                     std::uint32_t tile_columns, PinnedHostBuffer& host, DeviceContext& device)
        : tokens_(tokens), first_target_(first_target), tile_(tile), tile_columns_(tile_columns),
          host_(host), device_(device) {
        output_.reserve(tokens.size() - first_target);
        targets_.reserve(tile_columns);
    }

    [[nodiscard]] std::uint32_t staged_columns() const noexcept { return staged_; }
    [[nodiscard]] Tensor staged_hidden() const {
        return tile_.slice(1, 0, static_cast<std::int32_t>(staged_));
    }

    // Copies the scored columns of the chunk [chunk_begin, chunk_end) out of chunk_hidden, calling
    // flush() each time the tile fills. flush() must end with score().
    template <class Flush>
    void stage(const Tensor& chunk_hidden, std::uint32_t chunk_begin, std::uint32_t chunk_end,
               Flush&& flush) {
        std::uint32_t selected = std::max(chunk_begin, first_target_ - 1U);
        while (selected < chunk_end) {
            const std::uint32_t count = std::min(chunk_end - selected, tile_columns_ - staged_);
            Tensor source = chunk_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                               static_cast<std::int32_t>(count));
            Tensor destination = tile_.slice(1, static_cast<std::int32_t>(staged_),
                                             static_cast<std::int32_t>(count));
            CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                       cudaMemcpyDeviceToDevice, device_.stream));
            for (std::uint32_t column = 0; column < count; ++column) {
                targets_.push_back(tokens_[selected + column + 1U]);
            }
            selected += count;
            staged_ += count;
            if (staged_ == tile_columns_) { flush(); }
        }
    }

    // logits holds one row block per staged column. Writes their target logprobs to the output
    // and empties the tile. target_ids and logprobs hold one element per staged column.
    void score(const Tensor& logits, const Tensor& target_ids, Tensor logprobs,
               std::int32_t token_domain) {
        CUDA_CHECK(cudaMemcpyAsync(target_ids.data, targets_.data(), target_ids.bytes(),
                                   cudaMemcpyHostToDevice, device_.stream));
        ops::target_logprobs(logits, target_ids, token_domain, logprobs, device_.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_.data(), logprobs.data, logprobs.bytes(),
                                   cudaMemcpyDeviceToHost, device_.stream));
        device_.synchronize();
        const auto* host = static_cast<const float*>(host_.data());
        output_.insert(output_.end(), host, host + staged_);
        targets_.clear();
        staged_ = 0;
    }

    [[nodiscard]] std::vector<float> finish() && {
        if (output_.size() != tokens_.size() - first_target_) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        return std::move(output_);
    }

private:
    std::span<const TokenId> tokens_;
    std::uint32_t first_target_;
    Tensor tile_;
    std::uint32_t tile_columns_;
    PinnedHostBuffer& host_;
    DeviceContext& device_;
    std::vector<TokenId> targets_;
    std::vector<float> output_;
    std::uint32_t staged_ = 0;
};

// The idle lane a causal score runs on: a reset state slot and an active KV address mapped for
// the predictor columns. Declare it outside the try block that synchronizes the device on
// failure, so its pages are released only after the GPU stops using them.
template <class StateStore>
class CausalScoreLane {
public:
    using StateHandle = typename decltype(std::declval<StateStore&>().reserve_reset(
        cudaStream_t{}))::value_type;

    CausalScoreLane(StateStore& states, KVAddressSpaceStore& addresses, std::uint32_t pages,
                    std::uint32_t predictor_count, cudaStream_t stream)
        : states_(states), addresses_(addresses) {
        try {
            state_ = states.reserve_reset(stream);
            if (!state_) { throw std::bad_alloc(); }
            address_ = addresses.create_active(pages, 0, stream);
            if (!address_) { throw std::bad_alloc(); }
            addresses.ensure_mapped_to_tokens(*address_, predictor_count, stream);
        } catch (...) {
            (void)release_all();
            throw;
        }
    }
    CausalScoreLane(const CausalScoreLane&)            = delete;
    CausalScoreLane& operator=(const CausalScoreLane&) = delete;
    ~CausalScoreLane() { (void)release_all(); }

    [[nodiscard]] KVAddressSpaceHandle address() const { return *address_; }
    [[nodiscard]] std::int32_t state_slot() const { return states_.physical_slot(*state_); }

    void release() {
        if (!release_all()) {
            throw std::logic_error("causal score resources could not be released");
        }
    }

private:
    bool release_all() noexcept {
        bool released = true;
        if (address_) {
            try {
                if (addresses_.active(*address_)) { addresses_.deactivate(*address_); }
            } catch (...) {
                released = false;
            }
            released = addresses_.release(*address_) && released;
            address_.reset();
        }
        if (state_) {
            released = states_.release(*state_) && released;
            state_.reset();
        }
        return released;
    }

    StateStore& states_;
    KVAddressSpaceStore& addresses_;
    std::optional<StateHandle> state_;
    std::optional<KVAddressSpaceHandle> address_;
};

// Prefills the predictor columns in chunks of at most `chunk` and stages every scored column.
// prefill(cursor, nominal) runs one chunk into chunk_hidden and returns the model's chunk result
// (finalized, processed_tokens). flush projects the staged tile and calls tiles.score().
template <class StateStore, class Prefill, class Flush>
void prefill_scored_columns(CausalScoreTiles& tiles, CausalScoreLane<StateStore>& lane,
                            KVAddressSpaceStore& addresses, std::uint32_t predictor_count,
                            std::uint32_t chunk, const Tensor& chunk_hidden, Prefill&& prefill,
                            Flush&& flush) {
    std::uint32_t cursor = 0;
    while (cursor < predictor_count) {
        const std::uint32_t nominal = std::min(chunk, predictor_count - cursor);
        const auto result           = prefill(cursor, nominal);
        if (result.finalized || result.processed_tokens == 0 ||
            result.processed_tokens > nominal) {
            throw std::logic_error("causal score Prefill made invalid progress");
        }
        const std::uint32_t chunk_begin = cursor;
        cursor += result.processed_tokens;
        addresses.commit_frontier(lane.address(), cursor);
        tiles.stage(chunk_hidden, chunk_begin, cursor, flush);
    }
    flush();
}

} // namespace ninfer::program
