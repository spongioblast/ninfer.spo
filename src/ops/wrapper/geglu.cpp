#include "ninfer/ops/geglu.h"

#include "ops/launcher/geglu.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t contiguous_count(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(label) + " must be BF16");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] < 0) {
            throw std::invalid_argument(std::string(label) + " has a negative dimension");
        }
        if (tensor.ne[i] == 0) { return 0; }
        if (n > std::numeric_limits<std::int64_t>::max() / tensor.ne[i]) {
            throw std::overflow_error(std::string(label) + " size overflows int64");
        }
        n *= tensor.ne[i];
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(label) + " data must be non-null");
    }
    return n;
}

bool same_shape(const Tensor& a, const Tensor& b) {
    for (int i = 0; i < 4; ++i) {
        if (a.ne[i] != b.ne[i]) { return false; }
    }
    return true;
}

} // namespace

void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = contiguous_count(gate, "geglu: gate");
    if (contiguous_count(up, "geglu: up") != n || contiguous_count(out, "geglu: out") != n ||
        !same_shape(gate, up) || !same_shape(gate, out)) {
        throw std::invalid_argument("geglu: gate, up, and out must share a shape");
    }
    if (n == 0) { return; }
    const auto* gate_bytes = static_cast<const char*>(gate.data);
    const auto* up_bytes   = static_cast<const char*>(up.data);
    const auto* out_bytes  = static_cast<const char*>(out.data);
    const auto bytes       = n * static_cast<std::int64_t>(sizeof(std::uint16_t));
    if ((out_bytes >= gate_bytes && out_bytes < gate_bytes + bytes) ||
        (gate_bytes >= out_bytes && gate_bytes < out_bytes + bytes) ||
        (out_bytes >= up_bytes && out_bytes < up_bytes + bytes) ||
        (up_bytes >= out_bytes && up_bytes < out_bytes + bytes)) {
        throw std::invalid_argument("geglu: out must not overlap gate or up");
    }
    detail::geglu_launch(gate, up, out, stream);
}

} // namespace ninfer::ops
