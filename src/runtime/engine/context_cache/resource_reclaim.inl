#pragma once

namespace ninfer::runtime {

template <class Model>
auto
ResourceManager<Model>::remember_demand(const DemandKey& key, std::uint64_t request_id,
                     std::optional<CacheRetentionPriority> adopted ) -> void
{
    auto found = std::find_if(demand_.begin(), demand_.end(),
                              [&](const auto& entry) { return entry.key == key; });
    if (found != demand_.end()) {
        if (!adopted && found->request_id == request_id) { return; }
        found->request_id = request_id;
        found->priority =
            adopted.value_or(CacheRetentionPriority{.reused = true, .last_demand = clock_ + 1});
        if (!adopted) { ++clock_; }
        found->observed_at = clock_;
        return;
    }
    if (demand_.size() == kDemandCapacity) {
        const auto oldest =
            std::min_element(demand_.begin(), demand_.end(), [](const auto& a, const auto& b) {
                return a.observed_at < b.observed_at;
            });
        demand_.erase(oldest);
    }
    demand_.push_back({key, request_id, adopted.value_or(CacheRetentionPriority{}), ++clock_});
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::demand_priority(std::optional<DemandKey> key) const -> CacheRetentionPriority
{
    if (!key) { return {}; }
    const auto found = std::find_if(demand_.begin(), demand_.end(),
                                    [&](const auto& entry) { return entry.key == *key; });
    return found == demand_.end() ? CacheRetentionPriority{} : found->priority;
}

template <class Model>
auto
ResourceManager<Model>::catalog_reference_count(Handle handle) const -> std::size_t
{
    std::size_t count = 0;
    for (const auto& owner : owners_) {
        count += std::count_if(owner.points.begin(), owner.points.end(),
                               [&](const auto& point) { return point.handle == handle; });
    }
    count += std::count_if(shared_.begin(), shared_.end(),
                           [&](const auto& entry) { return entry.handle == handle; });
    return count;
}

template <class Model>
auto
ResourceManager<Model>::reference_count(Handle handle) const -> std::size_t
{
    auto count = catalog_reference_count(handle);
    for (const auto& waiting : waiting_) { count += contains(waiting.held, handle); }
    return count;
}

template <class Model>
auto
ResourceManager<Model>::references(Handle handle) const -> bool
{ return reference_count(handle) != 0; }

template <class Model>
auto
ResourceManager<Model>::remove_point(Program& program, Owner& owner, Handle handle) -> bool
{
    if (reference_count(handle) == 1 && program.valid_checkpoint(handle) &&
        !program.release_checkpoint(handle)) {
        return false;
    }
    std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
    if (!catalog_reference_count(handle)) { index_.erase(handle); }
    return true;
}

template <class Model>
auto
ResourceManager<Model>::forget(Handle handle) -> void
{
    index_.erase(handle);
    for (auto& owner : owners_) {
        std::erase_if(owner.points, [&](const auto& point) { return point.handle == handle; });
    }
    std::erase_if(shared_, [&](const auto& entry) { return entry.handle == handle; });
    for (auto& waiting : waiting_) {
        if (std::erase(waiting.held, handle) && !waiting.binding) { ++waiting.revocations; }
    }
}

template <class Model>
auto
ResourceManager<Model>::release_unreferenced(Program& program, Handle handle) -> void
{
    if (!catalog_reference_count(handle)) { index_.erase(handle); }
    if (references(handle)) { return; }
    if (program.valid_checkpoint(handle) && !program.release_checkpoint(handle)) {
        throw std::logic_error("Native could not release an unreferenced checkpoint owner");
    }
}

template <class Model>
auto
ResourceManager<Model>::replace_points(Program& program, Owner& owner, std::span<const Handle> carried) -> void
{
    std::vector<Handle> old;
    for (const auto& point : owner.points) { old.push_back(point.handle); }
    owner.points.clear();
    for (const auto handle : carried) {
        if (!program.valid_checkpoint(handle)) { continue; }
        const auto summary = program.checkpoint_metadata(handle);
        if (summary.role == CheckpointRole::SharedPrefix) {
            throw std::logic_error("Native carried a public point as a private continuation");
        }
        owner.points.push_back({handle, summary.role});
        index_.insert(program, handle, summary.frontier);
    }
    for (const auto handle : old) { release_unreferenced(program, handle); }
}

template <class Model>
auto
ResourceManager<Model>::prune(Program& program) -> void
{
    std::vector<Handle> stale;
    const auto inspect = [&](Handle handle) {
        if (!program.valid_checkpoint(handle) && !contains(stale, handle)) {
            stale.push_back(handle);
        }
    };
    for (const auto& owner : owners_) {
        for (const auto& point : owner.points) { inspect(point.handle); }
    }
    for (const auto& shared : shared_) { inspect(shared.handle); }
    for (const auto& waiting : waiting_) {
        for (const auto handle : waiting.held) { inspect(handle); }
    }
    for (const auto handle : stale) { forget(handle); }
    std::erase_if(owners_,
                  [](const auto& owner) { return !owner.active && owner.points.empty(); });
}

template <class Model>
auto
ResourceManager<Model>::priority_for(Handle handle) const -> CacheRetentionPriority
{
    CacheRetentionPriority result;
    const auto include = [&](CacheRetentionPriority priority) {
        if (priority.reused && (!result.reused || priority.last_demand > result.last_demand)) {
            result = priority;
        }
    };
    for (const auto& owner : owners_) {
        if (std::any_of(owner.points.begin(), owner.points.end(),
                        [&](const auto& point) { return point.handle == handle; })) {
            include(owner.priority);
        }
    }
    for (const auto& shared : shared_) {
        if (shared.handle == handle) { include(shared.priority); }
    }
    return result;
}

template <class Model>
auto
ResourceManager<Model>::victims() const -> std::vector<Handle>
{
    struct Ranked {
        Handle handle;
        CacheRetentionPriority priority;
        std::uint64_t age;
    };

    std::vector<Ranked> ranked;
    const auto add = [&](Handle handle, CacheRetentionPriority priority, std::uint64_t age) {
        const auto it = std::find_if(ranked.begin(), ranked.end(),
                                     [&](const auto& entry) { return entry.handle == handle; });
        if (it == ranked.end()) {
            ranked.push_back({handle, priority, age});
        } else if ((priority.reused && !it->priority.reused) ||
                   (priority.reused == it->priority.reused &&
                    (priority.reused ? priority.last_demand > it->priority.last_demand
                                     : age > it->age))) {
            *it = {handle, priority, age};
        }
    };
    for (const auto& owner : owners_) {
        for (const auto& point : owner.points) {
            add(point.handle, owner.priority, owner.retained_at);
        }
    }
    for (const auto& shared : shared_) {
        add(shared.handle, shared.priority, shared.retained_at);
    }
    for (const auto& waiting : waiting_) {
        for (const auto handle : waiting.held) { add(handle, {}, waiting.request); }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.priority.reused != b.priority.reused) { return !a.priority.reused; }
        return (a.priority.reused ? a.priority.last_demand : a.age) <
               (b.priority.reused ? b.priority.last_demand : b.age);
    });
    std::vector<Handle> out;
    for (const auto& entry : ranked) { out.push_back(entry.handle); }
    return out;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::reclaim_order(Program& program) const -> ReclaimCursor
{
    ReclaimCursor cursor;
    for (const auto handle : victims()) {
        const auto frontier = program.checkpoint_metadata(handle).frontier;
        cursor.victims.push_back({handle, priority_for(handle),
                                  demand_priority(program.checkpoint_key(handle, frontier))});
    }
    for (const auto& owner : owners_) {
        cursor.owners.push_back(
            {owner.token, owner.priority, owner.proven_frontier, points(owner.token)});
    }
    return cursor;
}

template <class Model>
auto
ResourceManager<Model>::useful(ContextResourceUsage released, ContextResourceUsage shortage) -> std::uint64_t
{
    if (shortage.state_slots) { return std::min(released.state_slots, shortage.state_slots); }
    if (shortage.main_kv_pages) {
        return std::min(released.main_kv_pages, shortage.main_kv_pages);
    }
    if (shortage.backend_kv_pages) {
        return std::min(released.backend_kv_pages, shortage.backend_kv_pages);
    }
    return std::min(released.host_bytes, shortage.host_bytes);
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::surviving(Program& program,
                                            std::span<const Handle> removed) const -> std::vector<Handle>
{
    // Only advertised entries can replace a future cache lookup. Waiting-only
    // checkpoints remain physical holders, but are not a caller's lookup fallback.
    std::vector<Handle> kept;
    const auto include = [&](Handle handle) {
        if (!contains(removed, handle) && program.valid_checkpoint(handle) &&
            !contains(kept, handle)) {
            kept.push_back(handle);
        }
    };
    for (const auto& owner : owners_) {
        for (const auto& point : owner.points) { include(point.handle); }
    }
    for (const auto& shared : shared_) { include(shared.handle); }
    return kept;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::retained(Program& program, Evaluation& evaluation) const -> std::span<const Handle>
{
    if (!evaluation.retained) { evaluation.retained = surviving(program, {}); }
    return *evaluation.retained;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::recovery_loss(Program& program, std::span<const Handle> removed,
                                          std::span<const Handle> kept,
                                          const Evaluation& evaluation) const -> std::uint64_t
{
    if (removed.empty()) { return 0; }
    return evaluation.native ? evaluation.native->recovery_loss(removed, kept)
                             : program.checkpoint_recovery_loss(removed, kept);
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::action_survivors(Program& program,
                                                       std::span<const Handle> sources,
                                                       ActionFacts& facts,
                                                       Evaluation& evaluation) const -> std::span<const Handle>
{
    if (!facts.kept) {
        facts.kept.emplace();
        const auto all = retained(program, evaluation);
        facts.kept->reserve(all.size());
        for (const auto handle : all) {
            if (!contains(sources, handle)) { facts.kept->push_back(handle); }
        }
    }
    return *facts.kept;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::action_loss(Program& program, std::span<const Handle> sources,
                                        ActionFacts& facts, Evaluation& evaluation) const -> std::uint64_t
{
    if (!facts.loss) {
        facts.loss = sources.empty()
                         ? 0
                         : recovery_loss(program, sources,
                                         action_survivors(program, sources, facts, evaluation),
                                         evaluation);
    }
    return *facts.loss;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::action_priority(Program& program,
                                                     std::span<const Handle> sources,
                                                     const ReclaimCursor& cursor,
                                                     const Evaluation& evaluation) const -> CacheRetentionPriority
{
    CacheRetentionPriority priority;
    if (sources.empty()) { return priority; }
    const auto include = [&](CacheRetentionPriority candidate) {
        if (candidate.reused &&
            (!priority.reused || candidate.last_demand > priority.last_demand)) {
            priority = candidate;
        }
    };
    for (const auto& victim : cursor.victims) {
        if (!contains(sources, victim.handle)) { continue; }
        const auto private_owner =
            std::any_of(cursor.owners.begin(), cursor.owners.end(), [&](const auto& owner) {
                return contains(owner.points, victim.handle);
            });
        if (!private_owner) { include(victim.priority); }
    }
    for (const auto& proof : cursor.owners) {
        if (!proof.priority.reused) { continue; }
        const auto* owner = find_owner(proof.owner);
        bool affected     = false;
        bool covered      = owner != nullptr;
        for (const auto handle : proof.points) {
            if (!contains(sources, handle)) { continue; }
            affected      = true;
            bool fallback = false;
            if (owner) {
                const auto frontier = program.checkpoint_metadata(handle).frontier;
                for (const auto& point : owner->points) {
                    if (contains(sources, point.handle) ||
                        !program.valid_checkpoint(point.handle)) {
                        continue;
                    }
                    const auto candidate = program.checkpoint_metadata(point.handle).frontier;
                    if (candidate < proof.proven_frontier || candidate > frontier) { continue; }
                    const auto loss =
                        recovery_loss(program, {&handle, 1}, {&point.handle, 1}, evaluation);
                    if (loss == frontier - candidate) {
                        fallback = true;
                        break;
                    }
                }
            }
            covered = covered && fallback;
        }
        // The hot grant protects already adopted progress. A later exact same-owner
        // checkpoint can cover that proof while an unproven suffix is reclaimed.
        if (affected && !covered) { include(proof.priority); }
    }
    return priority;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::action_facts(Program& program, std::span<const Handle> sources,
                                       const ReclaimCursor& cursor,
                                       const Evaluation& evaluation) const -> ActionFacts
{
    ActionFacts facts{.priority = action_priority(program, sources, cursor, evaluation)};
    if (!sources.empty()) {
        for (std::size_t i = 0; i < cursor.victims.size(); ++i) {
            if (contains(sources, cursor.victims[i].handle)) { facts.order = i; }
        }
    }
    return facts;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::rank_action(Program& program, std::span<const Handle> sources,
                                     ActionFacts& facts, std::uint64_t units,
                                     Evaluation& evaluation, bool preserving ) const -> ActionRank
{
    return {.priority = facts.priority,
            .loss     = preserving ? 0 : action_loss(program, sources, facts, evaluation),
            .units    = units,
            .order    = facts.order,
            .waiting  = preserving ? 0 : waiting_ticket(sources)};
}

template <class Model>
auto
ResourceManager<Model>::less_rank(const ActionRank& left, const ActionRank& right) -> bool
{
    if (bool(left.waiting) != bool(right.waiting)) { return !left.waiting; }
    if (left.waiting != right.waiting) { return left.waiting > right.waiting; }
    if (left.priority.reused != right.priority.reused) { return !left.priority.reused; }
    if (left.priority.reused && left.priority.last_demand != right.priority.last_demand) {
        return left.priority.last_demand < right.priority.last_demand;
    }
    if (!left.priority.reused) {
        const auto a = ninfer::Uint128{left.loss} * right.units;
        const auto b = ninfer::Uint128{right.loss} * left.units;
        if (a != b) { return a < b; }
    }
    return left.order < right.order;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::preservation_admission(Program& program,
                                               std::span<const Handle> sources,
                                               const ReclaimCursor& cursor,
                                               CacheRetentionPriority priority) const -> Admission
{
    Admission result{.priority = priority,
                     .sources  = std::vector<Handle>(sources.begin(), sources.end())};
    std::uint32_t deepest = 0;
    for (const auto handle : sources) {
        const auto frontier = program.checkpoint_metadata(handle).frontier;
        if (!result.checkpoint || frontier > deepest) {
            result.checkpoint = handle;
            deepest           = frontier;
        }
        const auto found =
            std::find_if(cursor.victims.begin(), cursor.victims.end(),
                         [&](const auto& victim) { return victim.handle == handle; });
        const auto demand =
            found == cursor.victims.end() ? CacheRetentionPriority{} : found->demand;
        if (demand.reused &&
            (!result.demand.reused || demand.last_demand > result.demand.last_demand)) {
            result.demand = demand;
        }
    }
    if (result.demand.reused) { result.priority = result.demand; }
    return result;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::recovery_gain(Program& program, const Admission& admission,
                                          std::span<const Handle> kept,
                                          const Evaluation& evaluation ) const -> std::uint64_t
{
    if (admission.base) {
        std::uint32_t fallback = 0;
        for (const auto handle : kept) {
            fallback = std::max(fallback, program.checkpoint_recovery_frontier(
                                              handle, *admission.base, admission.frontier));
        }
        return admission.frontier - fallback;
    }
    if (admission.checkpoint && program.valid_checkpoint(*admission.checkpoint)) {
        auto sources = admission.sources;
        if (sources.empty()) { sources.push_back(*admission.checkpoint); }
        auto other = std::vector<Handle>(kept.begin(), kept.end());
        std::erase_if(other, [&](auto handle) { return contains(sources, handle); });
        return recovery_loss(program, sources, other, evaluation);
    }
    return 0;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::admits(Program& program, std::span<const Handle> removed, ActionFacts& facts,
                          const std::optional<Admission>& admission,
                          const ReclaimCursor& cursor, Evaluation& evaluation,
                          bool preserving ) const -> bool
{
    if (!preserving && !may_revoke(removed, cursor.rights)) { return false; }
    if (!admission) { return true; }
    const auto victim = facts.priority;
    if (victim.reused && (!admission->priority.reused ||
                          victim.last_demand >= admission->priority.last_demand)) {
        return false;
    }
    const auto loss = preserving ? 0 : action_loss(program, removed, facts, evaluation);
    // This outer budget also covers Host victims of a preservation transfer. A hot
    // migration source cannot lend its permissions to the capture that requested space.
    if (cursor.gain_limit && (cursor.sacrificed >= *cursor.gain_limit ||
                              loss >= *cursor.gain_limit - cursor.sacrificed)) {
        return false;
    }
    if (admission->demand.reused) { return true; }
    const auto kept = preserving ? retained(program, evaluation)
                                 : action_survivors(program, removed, facts, evaluation);
    return recovery_gain(program, *admission, kept, evaluation) > loss;
}

template <class Model>
auto
ResourceManager<Model>::commit_release(Program& program, std::span<const Handle> handles, ReclaimCursor& cursor,
                    std::optional<std::uint64_t> quoted_loss ) -> void
{
    if (handles.empty()) { return; }
    if (!may_revoke(handles, cursor.rights)) {
        throw std::logic_error("cache release lacks waiting-source revocation rights");
    }
    const auto loss =
        cursor.gain_limit
            ? (quoted_loss
                   ? *quoted_loss
                   : program.checkpoint_recovery_loss(handles, surviving(program, handles)))
            : 0;
    for (const auto handle : handles) {
        if (!erase(program, handle, cursor.rights)) {
            throw std::logic_error("quoted cache release changed before commit");
        }
    }
    cursor.sacrificed =
        loss > UINT64_MAX - cursor.sacrificed ? UINT64_MAX : cursor.sacrificed + loss;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::host_victims_in_order(Program& program, std::size_t bytes, std::optional<Handle> incoming,
                      std::span<const Handle> later_snapshots, std::span<const Handle> excluded,
                      std::optional<Admission> admission, const ReclaimCursor& cursor,
                      Evaluation& evaluation, bool bytes_are_shortage ) const -> std::optional<std::vector<Handle>>
{
    const auto usage    = program.physical_usage();
    const auto free     = usage.capacity.host_bytes - usage.occupied.host_bytes;
    const auto required = bytes_are_shortage ? bytes : (bytes > free ? bytes - free : 0);
    if (!required) { return std::vector<Handle>{}; }
    if (!bytes_are_shortage && bytes > usage.capacity.host_bytes) { return std::nullopt; }
    if (!admission && incoming) {
        admission = preservation_admission(
            program, {&*incoming, 1}, cursor,
            action_priority(program, {&*incoming, 1}, cursor, evaluation));
    }
    std::vector<Handle> allowed;
    for (const auto& victim : cursor.victims) {
        const auto handle = victim.handle;
        if (incoming != handle && !contains(excluded, handle) && references(handle) &&
            program.valid_checkpoint(handle)) {
            allowed.push_back(handle);
        }
    }
    // Paused snapshots are offered only by required recovery and keep their explicit
    // Engine order; ordinary optional writes cannot acquire these rights.
    if (!admission) {
        for (const auto handle : later_snapshots) {
            if (!contains(excluded, handle) && !contains(allowed, handle) &&
                program.valid_checkpoint(handle)) {
                allowed.push_back(handle);
            }
        }
    }
    auto actions = program.plan_releases(allowed, excluded, {.host_bytes = required});
    std::vector<Handle> selected;
    std::size_t released = 0;
    while (released < required) {
        std::optional<std::size_t> best;
        std::vector<Handle> best_union;
        std::size_t best_bytes = 0;
        ActionRank best_rank;
        std::size_t best_snapshot_order = 0;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            auto combined = selected;
            for (const auto handle : actions[i].sources) {
                if (!contains(combined, handle)) { combined.push_back(handle); }
            }
            const auto amount = program.host_bytes_released(combined);
            if (amount <= released) { continue; }
            auto facts = action_facts(program, combined, cursor, evaluation);
            // A migration must satisfy both its own preservation rights and the
            // optional capture that requested space. Source heat cannot grant the
            // outer capture permission to erase unrelated hot Host history.
            if (!admits(program, combined, facts, admission, cursor, evaluation) ||
                !admits(program, combined, facts, cursor.admission, cursor, evaluation)) {
                continue;
            }
            const auto rank            = rank_action(program, combined, facts,
                                                     std::min<std::size_t>(amount, required), evaluation);
            std::size_t snapshot_order = 0;
            for (std::size_t snapshot = 0; snapshot < later_snapshots.size(); ++snapshot) {
                if (contains(combined, later_snapshots[snapshot])) {
                    snapshot_order = snapshot + 1;
                }
            }
            // Ordinary history is exhausted before Engine-owned paused snapshots.
            // Snapshot candidates retain the Engine's explicit youngest-first order.
            if (!best || snapshot_order < best_snapshot_order ||
                (snapshot_order == best_snapshot_order && less_rank(rank, best_rank))) {
                best                = i;
                best_union          = std::move(combined);
                best_bytes          = amount;
                best_rank           = rank;
                best_snapshot_order = snapshot_order;
            }
        }
        if (!best) { return std::nullopt; }
        selected = std::move(best_union);
        released = best_bytes;
        actions.erase(actions.begin() + *best);
    }
    return selected;
}

template <class Model>
auto
ResourceManager<Model>::balance_reused(Program& program) -> void
{
    const auto amounts = [](ContextResourceUsage r) {
        return std::array<std::uint64_t, 4>{r.state_slots, r.main_kv_pages, r.backend_kv_pages,
                                            r.host_bytes};
    };

    struct ProtectedEntry {
        CacheRetentionPriority* priority;
        std::uint64_t* retained_at;
        std::uint64_t ordinal;
        std::vector<Handle> handles;
        std::optional<std::array<std::uint64_t, 4>> footprint;
    };

    std::vector<ProtectedEntry> entries;
    for (auto& owner : owners_) {
        if (!owner.priority.reused) { continue; }
        std::vector<Handle> handles;
        handles.reserve(owner.points.size());
        for (const auto& point : owner.points) { handles.push_back(point.handle); }
        entries.push_back(
            {&owner.priority, &owner.retained_at, owner.token, std::move(handles)});
    }
    for (auto& shared : shared_) {
        if (shared.priority.reused) {
            entries.push_back(
                {&shared.priority, &shared.retained_at, shared.ordinal, {shared.handle}});
        }
    }
    if (entries.empty()) { return; }
    const auto capacities = amounts(program.physical_usage().capacity);
    std::vector<Handle> reused;
    for (;;) {
        reused.clear();
        for (const auto& entry : entries) {
            if (!entry.priority->reused) { continue; }
            for (const auto handle : entry.handles) {
                if (!contains(reused, handle)) { reused.push_back(handle); }
            }
        }
        if (reused.empty()) { return; }
        // Shared physical objects still count once in the whole protected set.
        const auto usage       = amounts(program.checkpoint_footprint(reused));
        ProtectedEntry* demote = nullptr;
        for (std::size_t resource = 0; resource < usage.size() && !demote; ++resource) {
            if (usage[resource] <= capacities[resource] * 3ULL / 4ULL) { continue; }
            ProtectedEntry* newest = nullptr;
            ProtectedEntry* oldest = nullptr;
            for (auto& entry : entries) {
                if (!entry.priority->reused) { continue; }
                // This loop only changes protection priority, never Native contents.
                if (!entry.footprint) {
                    entry.footprint = amounts(program.checkpoint_footprint(entry.handles));
                }
                if (!(*entry.footprint)[resource]) { continue; }
                if (!newest || entry.priority->last_demand > newest->priority->last_demand ||
                    (entry.priority->last_demand == newest->priority->last_demand &&
                     entry.ordinal > newest->ordinal)) {
                    newest = &entry;
                }
                if (!oldest || entry.priority->last_demand < oldest->priority->last_demand ||
                    (entry.priority->last_demand == oldest->priority->last_demand &&
                     entry.ordinal < oldest->ordinal)) {
                    oldest = &entry;
                }
            }
            if (oldest != newest) { demote = oldest; }
        }
        if (!demote) { return; }
        demote->priority->reused = false;
        *demote->retained_at     = ++clock_;
    }
}

} // namespace ninfer::runtime
