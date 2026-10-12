#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
    std::size_t max_tokens  = std::numeric_limits<std::size_t>::max();
};

struct ByteSpan {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct DecodedTokenView {
    std::string_view bytes;
    bool special = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

struct BpeMergeRule {
    int rank   = 0;
    int result = -1;
};

struct TokenBoundaryResult {
    // exact_frontier is present when the byte marker is also a boundary in the complete token
    // stream. stable_frontier retains only a normalization- and pre-tokenization-complete prefix.
    std::optional<std::size_t> exact_frontier;
    std::size_t stable_frontier = 0;
};

struct BoundaryEncodedText {
    std::vector<int> input_ids;
    std::vector<TokenBoundaryResult> boundaries;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    // A request whose text repeats or extends an earlier one is answered from that encode plus an
    // encode of the tail alone. The seam is a boundary of the cached encode whose preceding window
    // re-encodes to the cached ids and across which the new text encodes as its two halves, which
    // proves the splice does not change the token sequence; anything unproven is a full encode
    // (5090 port ADR-0010, plus the cross-seam check and the literal-span check below the window).
    BoundaryEncodedText encode_with_boundaries(std::string_view text,
                                               std::span<const std::size_t> byte_boundaries,
                                               EncodeOptions options                   = {},
                                               std::span<const ByteSpan> literal_spans = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    [[nodiscard]] DecodedTokenView decoded_token(int id) const;
    [[nodiscard]] std::string_view decode_token_bytes(int id,
                                                      bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;
    [[nodiscard]] const std::vector<std::string>& token_bytes() const noexcept {
        return decoded_token_bytes_;
    }

    // Encodes answered from the cache instead of the whole text.
    [[nodiscard]] std::size_t encode_cache_splices() const noexcept;
    // Cache paths that threw and fell back to a full encode. Expected to stay zero.
    [[nodiscard]] std::size_t encode_cache_fallbacks() const noexcept;

private:
    BoundaryEncodedText encode_with_boundaries_uncached(
        std::string_view text, std::span<const std::size_t> byte_boundaries, EncodeOptions options,
        std::span<const ByteSpan> literal_spans) const;
    [[nodiscard]] std::optional<BoundaryEncodedText>
    splice_from_cache(std::string_view text, std::span<const std::size_t> byte_boundaries,
                      EncodeOptions options, std::span<const ByteSpan> literal_spans) const;
    void remember_encode(std::string_view text, std::span<const std::size_t> byte_boundaries,
                         EncodeOptions options, std::span<const ByteSpan> literal_spans,
                         const BoundaryEncodedText& encoded) const;

    struct EncodeCacheEntry {
        std::string text;
        EncodeOptions options;
        std::vector<std::size_t> boundaries;
        std::vector<ByteSpan> literal_spans;
        std::vector<int> input_ids;
        std::vector<TokenBoundaryResult> results;
    };

    // Most recent first. A few entries survive interleaved sessions at --max-concurrency 2.
    static constexpr std::size_t kEncodeCacheEntries = 4;
    // prepare() runs on HTTP worker threads.
    mutable std::mutex encode_cache_mutex_;
    mutable std::vector<EncodeCacheEntry> encode_cache_;
    mutable std::atomic<std::size_t> encode_cache_splices_{0};
    mutable std::atomic<std::size_t> encode_cache_fallbacks_{0};

    std::vector<std::string> decoded_token_bytes_;
    std::vector<bool> valid_token_ids_;
    std::vector<bool> special_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::uint64_t, BpeMergeRule> bpe_merge_rules_;
    std::array<int, 256> byte_token_ids_{};
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
