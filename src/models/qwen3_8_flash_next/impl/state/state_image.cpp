#include <ninfer/models/qwen3_8_flash_next/state_image.h>

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

constexpr std::size_t kStateImageAlignment = 256;

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype;
}

bool same_ple_spec(const std::optional<PleStateSpec>& left,
                   const std::optional<PleStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    return !left || (left->channels == right->channels && left->width == right->width);
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_optional_region(const std::optional<LayoutRegion>& left,
                          const std::optional<LayoutRegion>& right) noexcept {
    return left.has_value() == right.has_value() && (!left || same_region(*left, *right));
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           left.spec.hidden == right.spec.hidden && left.spec.mtp_hidden == right.spec.mtp_hidden &&
           same_ple_spec(left.spec.ple, right.spec.ple) &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_optional_region(left.mtp_hidden, right.mtp_hidden) &&
           same_optional_region(left.ple, right.ple) &&
           left.image_bytes == right.image_bytes;
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    if (spec.linear.layers == 0 || spec.linear.conv_channels <= 0 || spec.linear.conv_width <= 0 ||
        spec.linear.value_heads <= 0 || spec.linear.value_head_dim <= 0 ||
        spec.linear.key_head_dim <= 0 || spec.linear.slot_count <= 0 ||
        (spec.linear.conv_dtype != DType::BF16 && spec.linear.conv_dtype != DType::FP32) ||
        spec.hidden <= 0) {
        throw std::invalid_argument("StateImage host geometry is invalid");
    }
    if (spec.ple && (spec.ple->channels <= 0 || spec.ple->width <= 0)) {
        throw std::invalid_argument("StateImage host PLE geometry is invalid");
    }
    if (spec.mtp_hidden && *spec.mtp_hidden <= 0) {
        throw std::invalid_argument("StateImage host MTP hidden geometry is invalid");
    }

    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                           {spec.linear.conv_channels, spec.linear.conv_width});
    const Tensor recurrent_slot(
        nullptr, DType::FP32,
        {spec.linear.key_head_dim, spec.linear.value_head_dim, spec.linear.value_heads});
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});
    host.linear_conv_layer_bytes = conv_slot.bytes();
    host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                               "StateImage host convolution bytes overflow"),
                                   kStateImageAlignment, "StateImage host convolution");
    host.linear_recurrent_layer_bytes = recurrent_slot.bytes();
    host.linear_recurrent =
        builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                "StateImage host recurrent bytes overflow"),
                    kStateImageAlignment, "StateImage host recurrent");
    host.continuation_hidden = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                           "StateImage host continuation hidden");
    if (spec.mtp_hidden) {
        const Tensor mtp_hidden_slot(nullptr, DType::BF16, {*spec.mtp_hidden});
        host.mtp_hidden = builder.add(mtp_hidden_slot.bytes(), kStateImageAlignment,
                                      "StateImage host MTP hidden");
    }
    if (spec.ple) {
        const Tensor ple_slot(nullptr, DType::BF16, {spec.ple->channels, spec.ple->width});
        host.ple = builder.add(ple_slot.bytes(), kStateImageAlignment, "StateImage host PLE");
    }
    host.image_bytes = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    if (spec.hidden <= 0) {
        throw std::invalid_argument("StateImage hidden width must be positive");
    }

    StateImageDeviceLayout out;
    out.linear = plan_linear_attention_state_pool(builder, spec.linear);
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    if (spec.mtp_hidden) {
        out.mtp_hidden =
            builder.add_tensor(DType::BF16, {*spec.mtp_hidden, spec.linear.slot_count},
                               kStateImageAlignment, "StateImage MTP continuation hidden");
    }
    if (spec.ple) {
        out.ple = builder.add_tensor(DType::BF16,
                                     {spec.ple->channels, spec.ple->width, spec.linear.slot_count},
                                     kStateImageAlignment, "StateImage PLE");
    }

    out.host = plan_host_state_image(spec);
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload             = checked_add(payload, layout.continuation_hidden.bytes,
                                      "StateImage transfer payload overflow");
    if (layout.mtp_hidden) {
        payload =
            checked_add(payload, layout.mtp_hidden->bytes, "StateImage transfer payload overflow");
    }
    if (layout.ple) {
        payload = checked_add(payload, layout.ple->bytes, "StateImage transfer payload overflow");
    }
    const std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 1ULL + (layout.spec.mtp_hidden ? 1ULL : 0ULL) +
        (layout.spec.ple ? 1ULL : 0ULL);
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(StateImageHostLayout layout, std::uint32_t capacity)
    : layout_(std::move(layout)), slots_(capacity), free_slots_(capacity), free_count_(capacity) {
    if (!same_host_layout(layout_, plan_host_state_image(layout_.spec))) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    const std::size_t bytes =
        checked_mul(layout_.image_bytes, capacity, "HostStatePool backing size overflow");
    if (bytes != 0) { backing_.emplace(bytes); }
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.occupied             = true;
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation};
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot    = slots_[handle.index];
    slot.occupied = false;
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.index < slots_.size() && slots_[handle.index].occupied &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return static_cast<std::byte*>(backing_->data()) +
           static_cast<std::size_t>(index) * layout_.image_bytes;
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : linear_(backing, layout.linear),
      continuation_hidden_(layout.continuation_hidden.bind(backing)), host_layout_(layout.host) {
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != host_layout_.spec.hidden ||
        continuation_hidden_.ne[1] != linear_.slot_count()) {
        throw std::invalid_argument("StateImage continuation hidden layout is inconsistent");
    }
    if (layout.ple.has_value() != host_layout_.spec.ple.has_value()) {
        throw std::invalid_argument("StateImage PLE layout is inconsistent");
    }
    if (layout.mtp_hidden.has_value() != host_layout_.spec.mtp_hidden.has_value()) {
        throw std::invalid_argument("StateImage MTP hidden layout is inconsistent");
    }
    StateImageSpec device_spec{.linear = layout.linear.spec, .hidden = continuation_hidden_.ne[0]};
    if (layout.mtp_hidden) {
        mtp_hidden_.emplace(layout.mtp_hidden->bind(backing));
        if (mtp_hidden_->dtype != DType::BF16 || !mtp_hidden_->is_contiguous() ||
            mtp_hidden_->ne[1] != linear_.slot_count()) {
            throw std::invalid_argument("StateImage MTP hidden tensor is inconsistent");
        }
        device_spec.mtp_hidden = mtp_hidden_->ne[0];
    }
    if (layout.ple) {
        ple_.emplace(layout.ple->bind(backing));
        if (ple_->dtype != DType::BF16 || !ple_->is_contiguous() ||
            ple_->ne[2] != linear_.slot_count()) {
            throw std::invalid_argument("StateImage PLE tensor is inconsistent");
        }
        device_spec.ple = PleStateSpec{.channels = ple_->ne[0], .width = ple_->ne[1]};
    }
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
}

StateImageDeviceSlotView StateImageDevicePool::slot_view(std::int32_t slot) const {
    StateImageDeviceSlotView view{
        .linear              = linear_.slot_view(slot),
        .continuation_hidden = continuation_hidden_slot(slot),
    };
    if (ple_) { view.ple = ple_slot(slot); }
    if (mtp_hidden_) { view.mtp_hidden = mtp_hidden_slot(slot); }
    return view;
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

Tensor StateImageDevicePool::ple_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage PLE slot is out of range");
    if (!ple_) { throw std::logic_error("StateImage has no PLE component"); }
    return ple_->slice(2, slot, 1).view({ple_->ne[0], ple_->ne[1]});
}

Tensor StateImageDevicePool::mtp_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage MTP hidden slot is out of range");
    if (!mtp_hidden_) { throw std::logic_error("StateImage has no MTP hidden component"); }
    return mtp_hidden_->slice(1, slot, 1).view({mtp_hidden_->ne[0]});
}

void StateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_slot(slot, slot_count(), "StateImage zero slot is out of range");
    linear_.zero_slot(slot, stream);
    const Tensor hidden = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, hidden.bytes(), stream));
    if (mtp_hidden_) {
        const Tensor state = mtp_hidden_slot(slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
    if (ple_) {
        const Tensor state = ple_slot(slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
}

void StateImageDevicePool::zero_all(cudaStream_t stream) {
    linear_.zero_all(stream);
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    if (mtp_hidden_) {
        CUDA_CHECK(cudaMemsetAsync(mtp_hidden_->data, 0, mtp_hidden_->bytes(), stream));
    }
    if (ple_) { CUDA_CHECK(cudaMemsetAsync(ple_->data, 0, ple_->bytes(), stream)); }
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    linear_.copy_slot(source, destination, stream);
    const Tensor source_hidden      = continuation_hidden_slot(source);
    const Tensor destination_hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(destination_hidden.data, source_hidden.data,
                               destination_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
    if (mtp_hidden_) {
        const Tensor source_mtp      = mtp_hidden_slot(source);
        const Tensor destination_mtp = mtp_hidden_slot(destination);
        CUDA_CHECK(cudaMemcpyAsync(destination_mtp.data, source_mtp.data, source_mtp.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (ple_) {
        const Tensor source_ple      = ple_slot(source);
        const Tensor destination_ple = ple_slot(destination);
        CUDA_CHECK(cudaMemcpyAsync(destination_ple.data, source_ple.data, source_ple.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        cudaStream_t stream) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination.data, host_layout_.linear_conv.offset +
                                              layer * host_layout_.linear_conv_layer_bytes),
            conv.data, conv.bytes(), cudaMemcpyDeviceToHost, stream));
        const Tensor recurrent = linear_.recurrent_slot(layer, source);
        CUDA_CHECK(cudaMemcpyAsync(
            byte_offset(destination.data, host_layout_.linear_recurrent.offset +
                                              layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.data, recurrent.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    const Tensor hidden = continuation_hidden_slot(source);
    CUDA_CHECK(
        cudaMemcpyAsync(byte_offset(destination.data, host_layout_.continuation_hidden.offset),
                        hidden.data, hidden.bytes(), cudaMemcpyDeviceToHost, stream));
    if (mtp_hidden_) {
        const Tensor state = mtp_hidden_slot(source);
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination.data, host_layout_.mtp_hidden->offset),
                                   state.data, state.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    if (ple_) {
        const Tensor state = ple_slot(source);
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination.data, host_layout_.ple->offset),
                                   state.data, state.bytes(), cudaMemcpyDeviceToHost, stream));
    }
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          cudaStream_t stream) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        const Tensor conv = linear_.conv_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            conv.data,
            byte_offset(source.data, host_layout_.linear_conv.offset +
                                         layer * host_layout_.linear_conv_layer_bytes),
            conv.bytes(), cudaMemcpyHostToDevice, stream));
        const Tensor recurrent = linear_.recurrent_slot(layer, destination);
        CUDA_CHECK(cudaMemcpyAsync(
            recurrent.data,
            byte_offset(source.data, host_layout_.linear_recurrent.offset +
                                         layer * host_layout_.linear_recurrent_layer_bytes),
            recurrent.bytes(), cudaMemcpyHostToDevice, stream));
    }
    const Tensor hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data,
                               byte_offset(source.data, host_layout_.continuation_hidden.offset),
                               hidden.bytes(), cudaMemcpyHostToDevice, stream));
    if (mtp_hidden_) {
        const Tensor state = mtp_hidden_slot(destination);
        CUDA_CHECK(cudaMemcpyAsync(state.data,
                                   byte_offset(source.data, host_layout_.mtp_hidden->offset),
                                   state.bytes(), cudaMemcpyHostToDevice, stream));
    }
    if (ple_) {
        const Tensor state = ple_slot(destination);
        CUDA_CHECK(cudaMemcpyAsync(state.data, byte_offset(source.data, host_layout_.ple->offset),
                                   state.bytes(), cudaMemcpyHostToDevice, stream));
    }
}

} // namespace ninfer::models::qwen3_8_flash_next
