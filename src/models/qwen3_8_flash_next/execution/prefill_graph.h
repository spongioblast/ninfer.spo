#pragma once

#include <cstdint>
#include <functional>

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr int kPrefillGraphTokenCap = 512;

// Why a small text pass stayed eager. 0 means the pass was eligible.
enum class PrefillGraphSkip : std::uint32_t {
    None = 0,
    GraphsOff = 1,
    TooLong = 2,
    Multimodal = 3,
    StateAction = 5,
    StateDir = 6,
};

// Host values that select the kernel schedule. The position base is not here: it lives in the
// ingress scalar so one graph replays across positions in the same envelope bucket.
struct PrefillGraphKey {
    std::int32_t tokens                 = 0;
    std::uint32_t envelope              = 0;
    std::uint32_t autoregressive_steps  = 0;
    std::int32_t source_slot            = 0;
    std::int32_t destination_slot       = 0;
    bool final_chunk                    = false;
    bool ends_at_split                  = false;
    bool mtp                            = false;
    bool rope_adjusted                  = false;
    bool sampled                        = false;
    bool activation                     = false;
    // The last MTP column reads the sampled token on the device, which adds a copy node.
    bool mtp_generated_column           = false;
    const void* checkpoint_hidden       = nullptr;
    const void* checkpoint_mtp_hidden   = nullptr;
    const void* sampling                = nullptr;
    // Each lane has its own block-table row, and the kernels take the row pointer by value.
    const void* text_kv_table           = nullptr;
    const void* mtp_kv_table            = nullptr;

    friend bool operator==(const PrefillGraphKey& lhs, const PrefillGraphKey& rhs) noexcept {
        return lhs.tokens == rhs.tokens && lhs.envelope == rhs.envelope &&
               lhs.autoregressive_steps == rhs.autoregressive_steps &&
               lhs.source_slot == rhs.source_slot && lhs.destination_slot == rhs.destination_slot &&
               lhs.final_chunk == rhs.final_chunk && lhs.ends_at_split == rhs.ends_at_split &&
               lhs.mtp == rhs.mtp && lhs.rope_adjusted == rhs.rope_adjusted &&
               lhs.sampled == rhs.sampled && lhs.activation == rhs.activation &&
               lhs.mtp_generated_column == rhs.mtp_generated_column &&
               lhs.checkpoint_hidden == rhs.checkpoint_hidden &&
               lhs.checkpoint_mtp_hidden == rhs.checkpoint_mtp_hidden &&
               lhs.sampling == rhs.sampling && lhs.text_kv_table == rhs.text_kv_table &&
               lhs.mtp_kv_table == rhs.mtp_kv_table;
    }
};

enum class PrefillGraphAction : std::uint8_t {
    Skip,
    EagerFirst,
    Capture,
    Launch,
};

struct PrefillGraphSettings {
    bool graphs            = false;
    int max_tokens         = 0;
    int cache_entries      = 64;
    bool audit             = false;
    bool state_dir         = false;
};

// Called once from the program constructor. max_tokens stays in force when graphs are off so
// eager and graph runs share one attention envelope.
void configure_prefill_graphs(bool use_cuda_graph);
// Drops captured prefill graphs. Called when an engine is created or destroyed, because those
// graphs hold pointers into that engine's workspace.
void reset_prefill_graphs() noexcept;
[[nodiscard]] const PrefillGraphSettings& prefill_graph_settings();

// Round `visible` up to the decode-profile bucket, and never past `capacity_tokens` when that
// capacity is known. The result is at least `visible`.
[[nodiscard]] std::uint32_t prefill_graph_envelope(std::uint32_t visible,
                                                   std::uint32_t capacity_tokens);

struct PrefillIngress {
    std::int32_t* position_base    = nullptr;
    std::int32_t* sample_position  = nullptr;
    std::int32_t* ids              = nullptr;
    std::int32_t* mtp_ids          = nullptr;
    void* ple_host                 = nullptr;
    void* ingress_device           = nullptr;
    void* ids_device               = nullptr;
    void* mtp_ids_device           = nullptr;
    void* position_base_device     = nullptr;
    void* sample_position_device   = nullptr;
    void* ple_device               = nullptr;
    void* positions_device         = nullptr;
};

// Host scratch for one small pass. The device pointers are stable for the life of the process.
[[nodiscard]] PrefillIngress prefill_ingress(int tokens, std::size_t ple_bytes);

// Copies the ingress struct and the gathered PLE rows onto the device. Call this inside the
// captured body so replay sees the bytes written since the last fill.
void prefill_ingress_upload(std::size_t ple_bytes, cudaStream_t stream);

[[nodiscard]] PrefillGraphAction select_prefill_graph(const PrefillGraphKey& key);
void capture_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream,
                           const std::function<void()>& body);
// When the audit env is set, the third use of a key captures again and compares kernel
// parameters with the graph captured on the second use. On a difference it prints the node and
// kernel to stderr, drops the key, and returns false; the caller then runs the pass eagerly.
// Both outcomes are counted in the request's prefill_graph_audited / _audit_mismatches.
[[nodiscard]] bool audit_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream,
                                       const std::function<void()>& body);
void launch_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream);
void forget_prefill_graph(const PrefillGraphKey& key);

// 4 MiB reserved per instantiated prefill graph: the median device-memory drop measured
// around capture, instantiate and upload at Tmax 32-512 (graphs up to 8 MiB; driver
// allocations are 2 MiB granular). CUDA does not report executable bytes. Added to the
// memory summary after capture.
[[nodiscard]] std::size_t prefill_graph_reserved_bytes();

} // namespace ninfer::models::qwen3_8_flash_next::detail
