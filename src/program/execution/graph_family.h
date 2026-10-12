#pragma once

// Decode CUDA Graph families: one executable per topology class, updated in place to the profile
// that covers the current batch size and frontier.

#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::program {

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

namespace detail {

inline void update_graph_profile(DecodeGraphTopology& topology, const DecodeGraphProfile& profile,
                                 std::size_t profile_index, const char* label) {
    if (topology.installed_profile == profile_index) { return; }
    try {
        topology.executable.update(profile.definition);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(label) + " CUDA Graph profile " +
                                 std::to_string(profile_index) + " (topology class " +
                                 std::to_string(profile.topology_class) + "): " + error.what());
    }
    topology.installed_profile = profile_index;
}

} // namespace detail

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;

    [[nodiscard]] DecodeGraphProfile& select(std::uint32_t batch_size, std::uint32_t frontier,
                                             const char* label = "decode") {
        const auto it =
            std::find_if(profiles.begin(), profiles.end(), [&](const DecodeGraphProfile& profile) {
                return profile.batch_size == batch_size &&
                       profile.min_execution_frontier <= frontier &&
                       frontier <= profile.max_execution_frontier;
            });
        if (it == profiles.end()) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
        }
        return *it;
    }

    [[nodiscard]] DecodeGraphExecutable& install(DecodeGraphProfile& profile,
                                                 const char* label = "decode") {
        const auto it = std::find_if(topologies.begin(), topologies.end(),
                                     [&](const DecodeGraphTopology& topology) {
                                         return topology.topology_class == profile.topology_class;
                                     });
        if (it == topologies.end()) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
        }
        detail::update_graph_profile(*it, profile,
                                     static_cast<std::size_t>(&profile - profiles.data()), label);
        return it->executable;
    }
};

// Profiles must cover [0, max_frontier] without gaps or overlaps, in order.
template <class Profile>
void validate_graph_profiles(const std::vector<Profile>& profiles, std::uint32_t max_frontier,
                             const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

// Instantiates one executable per topology class and uploads every profile once, leaving each
// topology on its first profile. With `warm`, the first profile of each topology is also prepared
// through warm(min_execution_frontier, batch_size) and launched once.
template <class Warm>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Warm&& warm) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }
    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }
        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        detail::update_graph_profile(topology, family.profiles[profile_index], profile_index, label);
        topology.executable.upload(device.stream);
        device.synchronize();
    };
    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class != topology.topology_class) { continue; }
            install_and_upload(topology, i);
            if (first_profile) { continue; }
            first_profile = i;
            if constexpr (!std::is_same_v<std::decay_t<Warm>, std::nullptr_t>) {
                const DecodeGraphProfile& profile = family.profiles[i];
                warm(profile.min_execution_frontier, profile.batch_size);
                device.synchronize();
                topology.executable.launch(device.stream);
                device.synchronize();
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

inline void instantiate_graph_family(DecodeGraphFamily& family, const char* label,
                                     DeviceContext& device) {
    instantiate_graph_family(family, label, device, nullptr);
}

} // namespace ninfer::program
