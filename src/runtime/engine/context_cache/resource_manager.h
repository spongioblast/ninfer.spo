#pragma once

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/prefix_index.h"
#include "runtime/engine/context_cache/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Policy owns bounded logical continuations. Native owns checkpoint contents and every byte.
template <class Model>
class ResourceManager {
public:
    using Program    = typename Model::Program;
    using Base       = typename Model::RequestBasePlan;
    using Handle     = typename Model::CheckpointHandle;
    using Source     = typename Model::SourceCandidate;
    using SessionKey = typename Model::CacheSessionKey;
    using OwnerToken = ContinuationOwnerToken;

    using DemandKey =
        typename decltype(std::declval<const Base&>().prefix_shortlist_key(0))::value_type;

    // Optional writes carry only their own demand evidence. Necessary execution has no
    // admission restriction. The request owns base for the lifetime of the reclaim decision.
    struct Admission {
        CacheRetentionPriority priority;
        CacheRetentionPriority demand;
        const Base* base       = nullptr;
        std::uint32_t frontier = 0;
        std::optional<Handle> checkpoint;
        std::vector<Handle> sources;
    };

    struct ReclaimCursor {
        struct Victim {
            Handle handle;
            CacheRetentionPriority priority;
            CacheRetentionPriority demand;
        };

        struct OwnerProtection {
            OwnerToken owner;
            CacheRetentionPriority priority;
            std::uint32_t proven_frontier;
            std::vector<Handle> points;
        };

        // Fixed membership, order and writeback rights for one decision. Native residency
        // supplies progress; a history may still have other pages needed by a later shortage.
        std::vector<Victim> victims;
        std::vector<OwnerProtection> owners;
        std::optional<Admission> admission;
        std::optional<std::uint64_t> gain_limit;
        std::uint64_t sacrificed = 0;
        ReclaimRights rights;
    };

    struct SourceChoice {
        Source source;
        OwnerToken owner           = 0;
        std::uint64_t shared_entry = 0;
        bool take_over             = false;
        std::optional<OwnerToken> resume_owner;
        bool session_hint               = false;
        std::uint64_t publication_order = 0;
        std::uint64_t ordinal           = 0;
    };

    explicit ResourceManager(bool enabled, ContextMachineCostModel costs)
        : enabled_(enabled), costs_(std::move(costs)) {}

    [[nodiscard]] std::vector<SourceChoice>
    candidates(Program& program, const Base& base, std::uint32_t maximum_frontier = UINT32_MAX,
               std::optional<OwnerToken> own_resume = std::nullopt, std::uint64_t request = 0,
               std::uint64_t publication_order = UINT64_MAX);


    // Evaluation grants consumption rights but does not retire anything. Native checks the
    // complete binding before applying these grants in its ownership handoff.
    [[nodiscard]] bool prepare_source(Program& program, const Base& base, SourceChoice& choice,
                                      std::uint64_t request                   = 0,
                                      std::optional<ReclaimRights> revocation = std::nullopt);


    void retain_source(Program& program, std::uint64_t request, const SourceChoice& choice);


    [[nodiscard]] std::optional<SourceChoice> retained_source(std::uint64_t request) const;


    [[nodiscard]] bool has_source_record(std::uint64_t request) const;


    [[nodiscard]] bool can_transfer_private(OwnerToken owner, std::uint64_t request) const;


    [[nodiscard]] std::uint32_t source_revocations(std::uint64_t request) const;


    void binding_started(std::uint64_t request, std::span<const Handle> retired,
                         std::optional<Handle> consumed);


    void release_source(Program& program, std::uint64_t request);


    // Called exactly once after Native's irreversible bind commit. The choice preserves logical
    // ownership even when Native consumed its original checkpoint handle during that commit.
    [[nodiscard]] OwnerToken adopt(Program& program, const SourceChoice& choice, const Base& base,
                                   std::uint64_t publication_order,
                                   std::span<const Handle> carried = {},
                                   std::span<const Handle> retired = {});


    void publish(Program& program, OwnerToken token, Handle handle);


    void finish(Program& program, OwnerToken token, const Base& base,
                std::uint64_t publication_order);


    void abandon(Program& program, OwnerToken token);


    [[nodiscard]] std::vector<Handle> points(OwnerToken token) const;


    [[nodiscard]] CacheRetentionPriority retention(OwnerToken token) const;


    bool recycle_input(Program& program, OwnerToken token);


    // The caller supplies each request's strictly advancing, non-overlapping committed
    // prefill ranges. Replay and preparation never call this; one candidate is crossed at
    // most once by a request, including when other requests commit between its chunks.
    void observe_committed(const Base& base, std::uint64_t request_id, std::uint32_t begin,
                           std::uint32_t end);


    [[nodiscard]] Admission capture_admission(Program&, OwnerToken owner, const Base& base,
                                              std::uint32_t frontier,
                                              bool inherit_private = true) const;


    [[nodiscard]] ReclaimCursor begin_reclaim(Program& program,
                                              std::optional<Admission> admission = std::nullopt,
                                              ReclaimRights rights               = {});


    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims(Program& program, std::size_t bytes, std::optional<Handle> incoming,
                 std::span<const Handle> later_snapshots = {},
                 std::span<const Handle> excluded        = {},
                 std::optional<Admission> admission      = std::nullopt,
                 const ReclaimCursor* decision = nullptr, ReclaimRights rights = {});


    void commit_host_victims(Program& program, std::span<const Handle> handles,
                             ReclaimCursor& cursor);


    bool erase(Program& program, Handle handle, ReclaimRights rights = {});


    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> excluded = {});


    [[nodiscard]] ReclaimProgress reclaim(Program& program, ContextResourceUsage shortage,
                                          std::span<const Handle> excluded, ReclaimCursor& cursor);


    void release_all(Program& program) noexcept;


private:
    struct WaitingSource {
        std::uint64_t request;
        SourceChoice choice;
        std::vector<Handle> held;
        std::uint32_t revocations = 0;
        bool binding              = false;
    };

    WaitingSource* find_waiting(std::uint64_t request) {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    const WaitingSource* find_waiting(std::uint64_t request) const {
        const auto found = std::find_if(waiting_.begin(), waiting_.end(),
                                        [=](const auto& item) { return item.request == request; });
        return found == waiting_.end() ? nullptr : &*found;
    }

    bool consumable(Handle handle, OwnerToken owner, std::uint64_t request,
                    std::optional<ReclaimRights> revocation = std::nullopt) const;


    [[nodiscard]] std::uint64_t waiting_ticket(std::span<const Handle> handles) const;


    bool may_revoke(std::span<const Handle> handles, ReclaimRights rights) const;


    struct Demand {
        DemandKey key;
        std::uint64_t request_id;
        CacheRetentionPriority priority;
        std::uint64_t observed_at;
    };

    void remember_demand(const DemandKey& key, std::uint64_t request_id,
                         std::optional<CacheRetentionPriority> adopted = std::nullopt);


    [[nodiscard]] CacheRetentionPriority demand_priority(std::optional<DemandKey> key) const;


    struct Point {
        Handle handle;
        CheckpointRole role;
    };

    struct Owner {
        OwnerToken token;
        CacheRetentionPriority priority;
        std::uint32_t proven_frontier   = 0;
        std::uint64_t retained_at       = 0;
        std::uint64_t publication_order = 0;
        bool active                     = false;
        std::optional<SessionKey> session;
        std::vector<Point> points;
    };

    struct Shared {
        Handle handle;
        std::uint64_t ordinal;
        CacheRetentionPriority priority;
        std::uint64_t retained_at;
    };

    static bool contains(std::span<const Handle> handles, Handle handle) {
        return std::find(handles.begin(), handles.end(), handle) != handles.end();
    }

    Owner* find_owner(OwnerToken token) {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    const Owner* find_owner(OwnerToken token) const {
        const auto it = std::find_if(owners_.begin(), owners_.end(),
                                     [&](const auto& owner) { return owner.token == token; });
        return it == owners_.end() ? nullptr : &*it;
    }

    Shared* find_shared(std::uint64_t ordinal) {
        const auto it = std::find_if(shared_.begin(), shared_.end(),
                                     [&](const auto& entry) { return entry.ordinal == ordinal; });
        return it == shared_.end() ? nullptr : &*it;
    }

    std::size_t catalog_reference_count(Handle handle) const;


    std::size_t reference_count(Handle handle) const;


    bool references(Handle handle) const;


    bool remove_point(Program& program, Owner& owner, Handle handle);


    void forget(Handle handle);


    void release_unreferenced(Program& program, Handle handle);


    void replace_points(Program& program, Owner& owner, std::span<const Handle> carried);


    void prune(Program& program);


    CacheRetentionPriority priority_for(Handle handle) const;


    std::vector<Handle> victims() const;


    [[nodiscard]] ReclaimCursor reclaim_order(Program& program) const;


    static std::uint64_t useful(ContextResourceUsage released, ContextResourceUsage shortage);


    [[nodiscard]] std::vector<Handle> surviving(Program& program,
                                                std::span<const Handle> removed) const;


    using NativeReclaimPlan = decltype(std::declval<Program&>().plan_reclaim(
        std::span<const Handle>{}, std::span<const Handle>{}, ContextResourceUsage{}));

    // These facts live only until this synchronous evaluation mutates resources or yields.
    // The fixed victim cursor may outlive them; newly published points still count as fallback.
    struct Evaluation {
        const NativeReclaimPlan* native = nullptr;
        std::optional<std::vector<Handle>> retained;
    };

    struct ActionFacts {
        CacheRetentionPriority priority;
        std::size_t order = 0;
        std::optional<std::vector<Handle>> kept;
        std::optional<std::uint64_t> loss;
    };

    [[nodiscard]] std::span<const Handle> retained(Program& program, Evaluation& evaluation) const;


    [[nodiscard]] std::uint64_t recovery_loss(Program& program, std::span<const Handle> removed,
                                              std::span<const Handle> kept,
                                              const Evaluation& evaluation) const;


    [[nodiscard]] std::span<const Handle> action_survivors(Program& program,
                                                           std::span<const Handle> sources,
                                                           ActionFacts& facts,
                                                           Evaluation& evaluation) const;


    [[nodiscard]] std::uint64_t action_loss(Program& program, std::span<const Handle> sources,
                                            ActionFacts& facts, Evaluation& evaluation) const;


    [[nodiscard]] CacheRetentionPriority action_priority(Program& program,
                                                         std::span<const Handle> sources,
                                                         const ReclaimCursor& cursor,
                                                         const Evaluation& evaluation) const;


    [[nodiscard]] ActionFacts action_facts(Program& program, std::span<const Handle> sources,
                                           const ReclaimCursor& cursor,
                                           const Evaluation& evaluation) const;


    struct ActionRank {
        CacheRetentionPriority priority;
        std::uint64_t loss    = 0;
        std::uint64_t units   = 0;
        std::size_t order     = 0;
        std::uint64_t waiting = 0;
    };

    [[nodiscard]] ActionRank rank_action(Program& program, std::span<const Handle> sources,
                                         ActionFacts& facts, std::uint64_t units,
                                         Evaluation& evaluation, bool preserving = false) const;


    static bool less_rank(const ActionRank& left, const ActionRank& right);


    [[nodiscard]] Admission preservation_admission(Program& program,
                                                   std::span<const Handle> sources,
                                                   const ReclaimCursor& cursor,
                                                   CacheRetentionPriority priority) const;


    [[nodiscard]] std::uint64_t recovery_gain(Program& program, const Admission& admission,
                                              std::span<const Handle> kept,
                                              const Evaluation& evaluation = {}) const;


    [[nodiscard]] bool admits(Program& program, std::span<const Handle> removed, ActionFacts& facts,
                              const std::optional<Admission>& admission,
                              const ReclaimCursor& cursor, Evaluation& evaluation,
                              bool preserving = false) const;


    void commit_release(Program& program, std::span<const Handle> handles, ReclaimCursor& cursor,
                        std::optional<std::uint64_t> quoted_loss = std::nullopt);


    [[nodiscard]] std::optional<std::vector<Handle>>
    host_victims_in_order(Program& program, std::size_t bytes, std::optional<Handle> incoming,
                          std::span<const Handle> later_snapshots, std::span<const Handle> excluded,
                          std::optional<Admission> admission, const ReclaimCursor& cursor,
                          Evaluation& evaluation, bool bytes_are_shortage = false) const;


    void balance_reused(Program& program);


    bool enabled_;
    ContextMachineCostModel costs_;
    PrefixIndex<Model> index_;
    std::vector<Owner> owners_;
    std::vector<Shared> shared_;
    std::vector<WaitingSource> waiting_;
    static constexpr std::size_t kDemandCapacity = 256;
    std::vector<Demand> demand_;
    std::uint64_t clock_   = 0;
    std::uint64_t ordinal_ = 0;
};
} // namespace ninfer::runtime

#include "runtime/engine/context_cache/resource_sources.inl"
#include "runtime/engine/context_cache/resource_reclaim.inl"
