#include "models/qwen3_8_flash_next/execution/prefill_graph.h"

#include "core/arena.h"
#include "core/runtime_options.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "runtime/engine/prefill_graph_stats.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

struct Slot {
    PrefillGraphKey key;
    bool ready = false;
    bool audited = false;
    std::uint64_t stamp = 0;
    ninfer::DecodeGraphDefinition definition;
    ninfer::DecodeGraphExecutable executable;
};

struct Cache {
    std::vector<Slot> slots;
    std::uint64_t clock = 0;
    int limit = 64;
};

struct Buffers {
    ninfer::PinnedHostBuffer host_ingress{1};
    ninfer::DeviceBuffer device_ingress{1};
    ninfer::PinnedHostBuffer ple_host{1};
    ninfer::DeviceBuffer ple_device{1};
    ninfer::DeviceBuffer positions{1};
    bool sized = false;
};

Cache& cache() {
    static Cache store;
    return store;
}

Buffers& buffers() {
    static Buffers store;
    return store;
}

std::size_t ingress_bytes() {
    return sizeof(std::int32_t) * static_cast<std::size_t>(4 + 2 * kPrefillGraphTokenCap);
}

PrefillGraphSettings& settings_mut() {
    static PrefillGraphSettings settings;
    return settings;
}

Slot* find_slot(const PrefillGraphKey& key) {
    for (Slot& slot : cache().slots) {
        if (slot.key == key) { return &slot; }
    }
    return nullptr;
}

void touch(Slot& slot) { slot.stamp = ++cache().clock; }

void evict_if_needed() {
    auto& store = cache();
    while (static_cast<int>(store.slots.size()) >= store.limit && !store.slots.empty()) {
        const auto oldest = std::min_element(
            store.slots.begin(), store.slots.end(),
            [](const Slot& lhs, const Slot& rhs) { return lhs.stamp < rhs.stamp; });
        store.slots.erase(oldest);
    }
}

std::string node_text(std::size_t index, CUfunction function) {
    const char* name = nullptr;
    if (function == nullptr || cuFuncGetName(&name, function) != CUDA_SUCCESS || name == nullptr) {
        name = "?";
    }
    return "node " + std::to_string(index) + " (" + name + ")";
}

CUfunction kernel_function(const CUDA_KERNEL_NODE_PARAMS& params) {
    if (params.func != nullptr) { return params.func; }
    CUfunction function = nullptr;
    if (params.kern != nullptr && cuKernelGetFunction(&function, params.kern) == CUDA_SUCCESS) {
        return function;
    }
    return nullptr;
}

// Empty when the two captures launch the same work with the same arguments.
std::string compare_graphs(cudaGraph_t left, cudaGraph_t right) {
    std::size_t left_count = 0;
    std::size_t right_count = 0;
    CUDA_CHECK(cudaGraphGetNodes(left, nullptr, &left_count));
    CUDA_CHECK(cudaGraphGetNodes(right, nullptr, &right_count));
    if (left_count != right_count) {
        return "node count " + std::to_string(left_count) + " vs " + std::to_string(right_count);
    }
    std::vector<cudaGraphNode_t> left_nodes(left_count);
    std::vector<cudaGraphNode_t> right_nodes(right_count);
    CUDA_CHECK(cudaGraphGetNodes(left, left_nodes.data(), &left_count));
    CUDA_CHECK(cudaGraphGetNodes(right, right_nodes.data(), &right_count));
    for (std::size_t index = 0; index < left_count; ++index) {
        cudaGraphNodeType left_type{};
        cudaGraphNodeType right_type{};
        CUDA_CHECK(cudaGraphNodeGetType(left_nodes[index], &left_type));
        CUDA_CHECK(cudaGraphNodeGetType(right_nodes[index], &right_type));
        if (left_type != right_type) { return node_text(index, nullptr) + ": node type"; }
        if (left_type == cudaGraphNodeTypeKernel) {
            CUDA_KERNEL_NODE_PARAMS lhs{};
            CUDA_KERNEL_NODE_PARAMS rhs{};
            if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(left_nodes[index]),
                                           &lhs) != CUDA_SUCCESS ||
                cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(right_nodes[index]),
                                           &rhs) != CUDA_SUCCESS) {
                return node_text(index, nullptr) + ": kernel node query failed";
            }
            const CUfunction function = kernel_function(lhs);
            if (function == nullptr || function != kernel_function(rhs)) {
                return node_text(index, function) + ": kernel function";
            }
            if (lhs.gridDimX != rhs.gridDimX || lhs.gridDimY != rhs.gridDimY ||
                lhs.gridDimZ != rhs.gridDimZ || lhs.blockDimX != rhs.blockDimX ||
                lhs.blockDimY != rhs.blockDimY || lhs.blockDimZ != rhs.blockDimZ ||
                lhs.sharedMemBytes != rhs.sharedMemBytes) {
                return node_text(index, function) + ": launch shape";
            }
            if (lhs.kernelParams == nullptr || rhs.kernelParams == nullptr) {
                return node_text(index, function) + ": no kernel parameter array";
            }
            for (std::size_t param = 0;; ++param) {
                std::size_t offset = 0;
                std::size_t size = 0;
                const CUresult info = cuFuncGetParamInfo(function, param, &offset, &size);
                if (info == CUDA_ERROR_INVALID_VALUE) { break; }
                if (info != CUDA_SUCCESS) {
                    return node_text(index, function) + ": parameter query failed";
                }
                if (std::memcmp(lhs.kernelParams[param], rhs.kernelParams[param], size) != 0) {
                    return node_text(index, function) + ": parameter " + std::to_string(param);
                }
            }
        } else if (left_type == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms lhs{};
            cudaMemcpy3DParms rhs{};
            CUDA_CHECK(cudaGraphMemcpyNodeGetParams(left_nodes[index], &lhs));
            CUDA_CHECK(cudaGraphMemcpyNodeGetParams(right_nodes[index], &rhs));
            if (lhs.srcPtr.ptr != rhs.srcPtr.ptr || lhs.dstPtr.ptr != rhs.dstPtr.ptr ||
                lhs.extent.width != rhs.extent.width || lhs.kind != rhs.kind) {
                return node_text(index, nullptr) + ": copy";
            }
        } else if (left_type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams lhs{};
            cudaMemsetParams rhs{};
            CUDA_CHECK(cudaGraphMemsetNodeGetParams(left_nodes[index], &lhs));
            CUDA_CHECK(cudaGraphMemsetNodeGetParams(right_nodes[index], &rhs));
            if (lhs.dst != rhs.dst || lhs.value != rhs.value || lhs.width != rhs.width ||
                lhs.height != rhs.height || lhs.elementSize != rhs.elementSize) {
                return node_text(index, nullptr) + ": memset";
            }
        }
    }
    return {};
}

} // namespace

void reset_prefill_graphs() noexcept { cache().slots.clear(); }

void configure_prefill_graphs(bool use_cuda_graph) {
    reset_prefill_graphs();
    PrefillGraphSettings& settings = settings_mut();
    const RuntimeSwitches& switches = runtime_switches();
    settings.max_tokens = std::clamp(switches.prefill_graph_max_tokens, 0, kPrefillGraphTokenCap);
    settings.cache_entries = std::max(1, switches.prefill_graph_cache);
    settings.audit         = switches.prefill_graph_audit;
    settings.state_dir     = !switches.state_dir.empty();
    const bool forced_off  = switches.prefill_graphs_forced_off;
    settings.graphs = use_cuda_graph && settings.max_tokens > 0 && !forced_off && !settings.state_dir;
    cache().limit = settings.cache_entries;
}

const PrefillGraphSettings& prefill_graph_settings() { return settings_mut(); }

std::uint32_t prefill_graph_envelope(std::uint32_t visible, std::uint32_t capacity_tokens) {
    if (visible == 0) { return 0; }
    // Decode profiles end at frontiers 127, 511, ... and attend frontier + 1 keys, so these are
    // the visible-key counts decode uses for the same length.
    static constexpr std::uint32_t kBuckets[] = {128U,   512U,   2048U,   8192U,   16384U,
                                                 32768U, 65536U, 131072U, 196608U, 262144U};
    std::uint32_t rounded = kBuckets[std::size(kBuckets) - 1];
    for (const std::uint32_t bucket : kBuckets) {
        if (visible <= bucket) {
            rounded = bucket;
            break;
        }
    }
    if (capacity_tokens != 0 && rounded > capacity_tokens) { rounded = capacity_tokens; }
    return std::max(rounded, visible);
}

PrefillIngress prefill_ingress(int tokens, std::size_t ple_bytes) {
    if (tokens < 1 || tokens > kPrefillGraphTokenCap) {
        throw std::invalid_argument("prefill ingress token count is outside the capture cap");
    }
    Buffers& store = buffers();
    if (!store.sized) {
        const std::size_t ple_cap =
            static_cast<std::size_t>(kPrefillGraphTokenCap) * 2560U * 2U;
        store.host_ingress = ninfer::PinnedHostBuffer(ingress_bytes());
        store.device_ingress = ninfer::DeviceBuffer(ingress_bytes());
        store.ple_host = ninfer::PinnedHostBuffer(ple_cap);
        store.ple_device = ninfer::DeviceBuffer(ple_cap);
        store.positions =
            ninfer::DeviceBuffer(static_cast<std::size_t>(kPrefillGraphTokenCap) * sizeof(std::int32_t));
        store.sized = true;
    }
    if (ple_bytes > store.ple_host.size()) {
        throw std::invalid_argument("prefill PLE staging exceeds the capture buffer");
    }
    auto* host = static_cast<std::int32_t*>(store.host_ingress.data());
    auto* device = static_cast<std::byte*>(store.device_ingress.p);
    PrefillIngress ingress;
    ingress.position_base = host;
    ingress.sample_position = host + 1;
    ingress.ids = host + 4;
    ingress.mtp_ids = host + 4 + kPrefillGraphTokenCap;
    ingress.ple_host = store.ple_host.data();
    ingress.ingress_device = store.device_ingress.p;
    ingress.position_base_device = device;
    ingress.sample_position_device = device + sizeof(std::int32_t);
    ingress.ids_device = device + sizeof(std::int32_t) * 4;
    ingress.mtp_ids_device = device + sizeof(std::int32_t) * static_cast<std::size_t>(4 + kPrefillGraphTokenCap);
    ingress.ple_device = store.ple_device.p;
    ingress.positions_device = store.positions.p;
    return ingress;
}

void prefill_ingress_upload(std::size_t ple_bytes, cudaStream_t stream) {
    Buffers& store = buffers();
    CUDA_CHECK(cudaMemcpyAsync(store.device_ingress.p, store.host_ingress.data(), ingress_bytes(),
                               cudaMemcpyHostToDevice, stream));
    if (ple_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(store.ple_device.p, store.ple_host.data(), ple_bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

PrefillGraphAction select_prefill_graph(const PrefillGraphKey& key) {
    if (!settings_mut().graphs) { return PrefillGraphAction::Skip; }
    if (Slot* existing = find_slot(key)) {
        touch(*existing);
        return existing->ready ? PrefillGraphAction::Launch : PrefillGraphAction::Capture;
    }
    evict_if_needed();
    Slot slot;
    slot.key = key;
    touch(slot);
    cache().slots.push_back(std::move(slot));
    return PrefillGraphAction::EagerFirst;
}

void capture_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream,
                           const std::function<void()>& body) {
    Slot* slot = find_slot(key);
    if (slot == nullptr || slot->ready) {
        throw std::logic_error("prefill graph capture has no pending key");
    }
    // Executables are allocated by the driver outside the startup plan, so they come out of the
    // KV headroom. This log measures that cost per graph.
    static const bool memlog = runtime_switches().prefill_graph_memlog;
    std::size_t free_before = 0;
    std::size_t total       = 0;
    if (memlog) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    }
    slot->definition.capture(stream, body);
    slot->executable.instantiate(slot->definition);
    slot->executable.upload(stream);
    slot->ready = true;
    touch(*slot);
    if (memlog) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::size_t free_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
        std::size_t ready = 0;
        for (const Slot& entry : cache().slots) { ready += entry.ready ? 1U : 0U; }
        std::fprintf(stderr,
                     "prefill graph memory: tokens=%d envelope=%u final=%d ready=%zu bytes=%lld "
                     "free=%zu\n",
                     key.tokens, key.envelope, key.final_chunk ? 1 : 0, ready,
                     static_cast<long long>(free_before) - static_cast<long long>(free_after),
                     free_after);
    }
}

bool audit_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream,
                         const std::function<void()>& body) {
    Slot* slot = find_slot(key);
    if (slot == nullptr || !slot->ready || slot->audited || !settings_mut().audit) { return true; }
    std::string difference;
    try {
        ninfer::DecodeGraphDefinition second;
        second.capture(stream, body);
        difference = compare_graphs(slot->definition.native_graph(), second.native_graph());
    } catch (const std::exception& error) {
        difference = std::string("second capture failed: ") + error.what();
    }
    if (!difference.empty()) {
        ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::AuditMismatch);
        std::fprintf(stderr,
                     "prefill graph audit: tokens=%d envelope=%u final=%d mtp=%d: %s\n",
                     key.tokens, key.envelope, key.final_chunk ? 1 : 0, key.mtp ? 1 : 0,
                     difference.c_str());
        forget_prefill_graph(key);
        return false;
    }
    ninfer::runtime::note_prefill_graph(ninfer::runtime::PrefillGraphNote::AuditMatched);
    slot = find_slot(key);
    if (slot != nullptr) { slot->audited = true; }
    return true;
}

void launch_prefill_graph(const PrefillGraphKey& key, cudaStream_t stream) {
    Slot* slot = find_slot(key);
    if (slot == nullptr || !slot->ready) {
        throw std::logic_error("prefill graph launch has no executable");
    }
    touch(*slot);
    slot->executable.launch(stream);
}

void forget_prefill_graph(const PrefillGraphKey& key) {
    auto& slots = cache().slots;
    slots.erase(std::remove_if(slots.begin(), slots.end(),
                               [&](const Slot& slot) { return slot.key == key; }),
                slots.end());
}

std::size_t prefill_graph_reserved_bytes() {
    constexpr std::size_t kMiB = 1024U * 1024U;
    std::size_t ready = 0;
    for (const Slot& slot : cache().slots) {
        if (slot.ready) { ++ready; }
    }
    return ready * 4U * kMiB;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
