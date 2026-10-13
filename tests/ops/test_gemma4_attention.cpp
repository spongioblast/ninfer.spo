#include "ninfer/ops/gemma4_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kPage = 64;
constexpr PointwiseCriterion kCriterion{2.0e-5, 8.0e-3};

// Logical page p lives in physical page pages - 1 - p, so the block table is exercised.
std::size_t kv_index(int head_dim, int kv_heads, int pages, int head, int position, int dim) {
    const int slot     = position % kPage;
    const int physical = pages - 1 - position / kPage;
    return (static_cast<std::size_t>(head_dim) * kPage *
            (static_cast<std::size_t>(head) + static_cast<std::size_t>(kv_heads) * physical)) +
           static_cast<std::size_t>(head_dim) * slot + static_cast<std::size_t>(dim);
}

// Queries sit at the last `tokens` of `kv_len` cached positions.
int run_case(const char* label, int head_dim, int q_heads, int kv_heads, int tokens, int kv_len,
             int window) {
    const int pages            = (kv_len + kPage - 1) / kPage;
    const int first_query      = kv_len - tokens;
    const std::size_t q_count  = static_cast<std::size_t>(head_dim) * q_heads * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(head_dim) * kPage * kv_heads * pages;
    std::vector<float> q(q_count), k(kv_count, 0.0f), v(kv_count, 0.0f);
    fill_uniform(q, 70u, -1.0f, 1.0f);
    round_to_bf16(q);
    std::vector<float> k_logical(static_cast<std::size_t>(head_dim) * kv_heads * kv_len);
    std::vector<float> v_logical = k_logical;
    fill_uniform(k_logical, 71u, -1.0f, 1.0f);
    fill_uniform(v_logical, 72u, -1.0f, 1.0f);
    round_to_bf16(k_logical);
    round_to_bf16(v_logical);
    for (int position = 0; position < kv_len; ++position) {
        for (int head = 0; head < kv_heads; ++head) {
            for (int dim = 0; dim < head_dim; ++dim) {
                const std::size_t logical =
                    (static_cast<std::size_t>(position) * kv_heads + head) * head_dim + dim;
                k[kv_index(head_dim, kv_heads, pages, head, position, dim)] = k_logical[logical];
                v[kv_index(head_dim, kv_heads, pages, head, position, dim)] = v_logical[logical];
            }
        }
    }

    const int group = q_heads / kv_heads;
    std::vector<double> expected(q_count, 0.0);
    const auto score = [&](int token, int qh, int key) {
        const int kvh = qh / group;
        double dot    = 0.0;
        for (int dim = 0; dim < head_dim; ++dim) {
            dot += static_cast<double>(
                       q[(static_cast<std::size_t>(token) * q_heads + qh) * head_dim + dim]) *
                   k_logical[(static_cast<std::size_t>(key) * kv_heads + kvh) * head_dim + dim];
        }
        return dot;
    };
    for (int token = 0; token < tokens; ++token) {
        const int position = first_query + token;
        for (int qh = 0; qh < q_heads; ++qh) {
            const int kvh    = qh / group;
            const int first  = window == 0 ? 0 : std::max(0, position - window + 1);
            double max_score = -1.0e300;
            for (int key = first; key <= position; ++key) {
                max_score = std::max(max_score, score(token, qh, key));
            }
            double sum = 0.0;
            for (int key = first; key <= position; ++key) {
                sum += std::exp(score(token, qh, key) - max_score);
            }
            for (int key = first; key <= position; ++key) {
                const double weight = std::exp(score(token, qh, key) - max_score) / sum;
                for (int dim = 0; dim < head_dim; ++dim) {
                    expected[(static_cast<std::size_t>(token) * q_heads + qh) * head_dim + dim] +=
                        weight *
                        v_logical[(static_cast<std::size_t>(key) * kv_heads + kvh) * head_dim +
                                  dim];
                }
            }
        }
    }

    auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    };
    const auto q_bits = bits(q);
    const auto k_bits = bits(k);
    const auto v_bits = bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * 2), dk(k_bits.size() * 2), dv(v_bits.size() * 2),
        dout(q_bits.size() * 2);
    std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
    for (int page = 0; page < pages; ++page) {
        table[static_cast<std::size_t>(page)] = pages - 1 - page;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = first_query + i; }
    GuardedDeviceBuffer dtable(table.size() * sizeof(std::int32_t)),
        dpos(positions.size() * sizeof(std::int32_t));
    dq.copy_from_host(q_bits.data(), dq.bytes());
    dk.copy_from_host(k_bits.data(), dk.bytes());
    dv.copy_from_host(v_bits.data(), dv.bytes());
    dtable.copy_from_host(table.data(), dtable.bytes());
    dpos.copy_from_host(positions.data(), dpos.bytes());
    Tensor q_tensor(dq.data(), DType::BF16, {head_dim, q_heads, tokens});
    Tensor k_tensor(dk.data(), DType::BF16, {head_dim, kPage, kv_heads, pages});
    Tensor v_tensor(dv.data(), DType::BF16, {head_dim, kPage, kv_heads, pages});
    Tensor table_tensor(dtable.data(), DType::I32, {pages});
    Tensor pos_tensor(dpos.data(), DType::I32, {tokens});
    Tensor out_tensor(dout.data(), DType::BF16, {head_dim, q_heads, tokens});
    const std::size_t capacity =
        ops::gemma4_attention_workspace_capacity_bytes(head_dim, q_heads, kv_heads, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gemma4_attention(q_tensor, k_tensor, v_tensor, table_tensor, pos_tensor, window, 1.0f,
                          out_tensor, workspace, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> first_bits(q_count), second_bits(q_count);
    dout.copy_to_host(first_bits.data(), dout.bytes());
    int failures =
        verify_pointwise(label, from_device_bf16(dout.data(), q_count), expected, kCriterion);
    ops::gemma4_attention(q_tensor, k_tensor, v_tensor, table_tensor, pos_tensor, window, 1.0f,
                          out_tensor, workspace, nullptr);
    cuda_synchronize();
    dout.copy_to_host(second_bits.data(), dout.bytes());
    if (second_bits != first_bits) {
        std::cerr << label << ": a second run changed the output\n";
        ++failures;
    }
    failures += dq.verify_guards("q");
    failures += dout.verify_guards("out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case("sliding window", 256, 32, 16, 6, 6, 1024);
    failures += run_case("sliding narrow", 256, 32, 16, 5, 5, 2);
    failures += run_case("global", 512, 32, 4, 4, 4, 0);
    failures += run_case("sliding pages", 256, 32, 16, 150, 150, 40);
    failures += run_case("global pages", 512, 32, 4, 130, 130, 0);
    failures += run_case("sliding decode", 256, 32, 16, 1, 300, 100);
    failures += run_case("global decode", 512, 32, 4, 1, 333, 0);
    failures += run_case("global chunk", 512, 32, 4, 20, 200, 0);
    failures += run_case("sliding decode past window", 256, 32, 16, 1, 1500, 1024);
    failures += run_case("global decode long", 512, 32, 4, 1, 4096, 0);
    failures += run_case("global decode short", 512, 32, 4, 1, 3, 0);
    failures += run_case("global verify tokens", 512, 32, 4, 5, 2000, 0);
    failures += run_case("sliding verify tokens", 256, 32, 16, 6, 1100, 1024);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_attention\n";
    return failures ? 1 : 0;
}
