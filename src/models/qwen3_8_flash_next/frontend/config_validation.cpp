#include "models/qwen3_8_flash_next/frontend/config_validation.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::config_validation {
namespace {

using Json   = nlohmann::json;
namespace fi = frontend_internal;

constexpr double kRescaleFactor                       = 1.0 / 255.0;
constexpr double kVideoFps                            = 2.0;
constexpr int kVideoMinFrames                         = 4;
constexpr int kVideoMaxFrames                         = 768;
constexpr std::uint64_t kRegisteredImageMinimumPixels = 65'536;
constexpr std::uint64_t kRegisteredImageMaximumPixels = 16'777'216;
constexpr std::uint64_t kRegisteredVideoMinimumPixels = 4'096;
constexpr std::uint64_t kRegisteredVideoMaximumPixels = 25'165'824;

constexpr std::array<std::pair<std::string_view, TokenId>, 4> kVisionSpecialTokens = {{
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|image_pad|>", 248056},
    {"<|video_pad|>", 248057},
}};

constexpr std::array<std::pair<std::string_view, TokenId>, 7> kConfigOnlyTokens = {{
    {"<|audio_start|>", 248070},
    {"<|audio_end|>", 248071},
    {"<tts_pad>", 248072},
    {"<tts_text_bos>", 248073},
    {"<tts_text_eod>", 248074},
    {"<tts_text_bos_single>", 248075},
    {"<|audio_pad|>", 248076},
}};

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (number_or_default(config, "rescale_factor", resource, kRescaleFactor) != kRescaleFactor) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (tokenizer_config.value("add_bos_token", false) ||
        tokenizer_config.value("add_prefix_space", false)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Flash-Next tokenizer prefix semantics");
    }
    if (!tokenizer_config.contains("pad_token") || !tokenizer_config.at("pad_token").is_string() ||
        tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
    if (tokenizer_config.contains("chat_template") &&
        (!tokenizer_config.at("chat_template").is_string() ||
         tokenizer_config.at("chat_template").get_ref<const std::string&>() !=
             resources.chat_template_jinja)) {
        throw std::invalid_argument(
            "tokenizer_config.json.chat_template does not match frontend/chat_template.jinja");
    }
}

} // namespace

fi::ProcessorOptions processor_options(const FrontendResources& resources) {
    const Json image =
        parse_resource_json(resources.preprocessor_config_json, "preprocessor_config.json");
    const Json video = parse_resource_json(resources.video_preprocessor_config_json,
                                           "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps =
        number_or_default(video, "fps", "video_preprocessor_config.json", kVideoFps);
    options.video_min_frames = static_cast<int>(
        integer_or_default(video, "min_frames", "video_preprocessor_config.json", kVideoMinFrames));
    options.video_max_frames = static_cast<int>(
        integer_or_default(video, "max_frames", "video_preprocessor_config.json", kVideoMaxFrames));
    if (options.video_fps != kVideoFps || options.video_min_frames != kVideoMinFrames ||
        options.video_max_frames != kVideoMaxFrames) {
        throw std::invalid_argument(
            "video_preprocessor_config.json does not match registered sampling defaults");
    }

    return options;
}

void validate_registered_processor(const fi::ProcessorOptions& options) {
    if (options.image_min_pixels != kRegisteredImageMinimumPixels ||
        options.image_max_pixels != kRegisteredImageMaximumPixels ||
        options.video_min_pixels != kRegisteredVideoMinimumPixels ||
        options.video_max_pixels != kRegisteredVideoMaximumPixels) {
        throw std::invalid_argument(
            "registered processor pixel bounds do not match the compiled Vision item capacity");
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources) {
    validate_tokenizer_config(resources);
    return fi::CompiledChatTemplate::resolve(resources.chat_template_jinja);
}

void validate_registered_tokenizer(const fi::Tokenizer& tokenizer) {
    if (!tokenizer.has_exact_token_domain(kTokenDomain)) {
        throw std::invalid_argument(
            "artifact tokenizer does not expose the registered 248077-token domain");
    }
    for (const auto& [text, expected] : kVisionSpecialTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected) {
            throw std::invalid_argument("artifact tokenizer does not match registered Vision token "
                                        "IDs");
        }
    }
    for (const auto& [text, expected] : kConfigOnlyTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected ||
            !tokenizer.is_special_token(expected)) {
            throw std::invalid_argument(
                "artifact tokenizer does not expose registered special tokens");
        }
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::config_validation
