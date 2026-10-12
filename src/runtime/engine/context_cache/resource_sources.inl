#pragma once

namespace ninfer::runtime {

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::candidates(Program& program, const Base& base, std::uint32_t maximum_frontier ,
           std::optional<OwnerToken> own_resume , std::uint64_t request ,
           std::uint64_t publication_order ) -> std::vector<SourceChoice>
{
    prune(program);
    std::vector<SourceChoice> result;
    if (enabled_) {
        for (const auto handle : index_.matches(program, base)) {
            if (program.checkpoint_metadata(handle).frontier > maximum_frontier) { continue; }
            for (const auto& owner : owners_) {
                const auto point =
                    std::find_if(owner.points.begin(), owner.points.end(),
                                 [&](const auto& p) { return p.handle == handle; });
                if (point == owner.points.end()) { continue; }
                const auto& advice = base.context_cache();
                const bool take =
                    own_resume
                        ? *own_resume == owner.token
                        : !owner.active && owner.publication_order <= publication_order &&
                              (!owner.session || (advice.update_session_index &&
                                                  advice.session_key == owner.session));
                result.push_back(
                    {.source            = Source{.checkpoint = handle},
                     .owner             = owner.token,
                     .take_over         = take,
                     .resume_owner      = own_resume,
                     .session_hint      = owner.session && owner.session == advice.session_key,
                     .publication_order = owner.publication_order,
                     .ordinal           = owner.token});
            }
            for (const auto& shared : shared_) {
                if (shared.handle != handle) { continue; }
                result.push_back({.source       = Source{.checkpoint = handle},
                                  .shared_entry = shared.ordinal,
                                  .resume_owner = own_resume,
                                  .ordinal      = shared.ordinal});
            }
        }
    }
    result.push_back({.source = Source{}, .resume_owner = own_resume});
    if (const auto* waiting = find_waiting(request);
        waiting && waiting->choice.source.checkpoint &&
        contains(waiting->held, *waiting->choice.source.checkpoint)) {
        result.push_back(waiting->choice);
    }
    std::erase_if(
        result, [&](auto& choice) { return !prepare_source(program, base, choice, request); });
    const auto cost = [&](const Source& source) {
        const auto transfer = price_context_transfer_requirements(costs_, source.transfers);
        const auto prefill  = costs_.prefill_ns(source.remaining_work);
        return transfer > UINT64_MAX - prefill ? UINT64_MAX : transfer + prefill;
    };
    const auto bytes = [](const Source& source) {
        std::uint64_t total = 0;
        for (const auto& transfer : source.transfers) {
            total = transfer.work.payload_bytes > UINT64_MAX - total
                        ? UINT64_MAX
                        : total + transfer.work.payload_bytes;
        }
        return total;
    };
    std::stable_sort(result.begin(), result.end(), [&](const auto& a, const auto& b) {
        const auto left = cost(a.source), right = cost(b.source);
        if (left != right) { return left < right; }
        if (bytes(a.source) != bytes(b.source)) { return bytes(a.source) < bytes(b.source); }
        if (a.source.reused_tokens != b.source.reused_tokens) {
            return a.source.reused_tokens > b.source.reused_tokens;
        }
        if (a.session_hint != b.session_hint) { return a.session_hint; }
        if (bool(a.owner) != bool(b.owner)) { return bool(a.owner); }
        if (a.publication_order != b.publication_order) {
            return a.publication_order > b.publication_order;
        }
        return a.ordinal < b.ordinal;
    });
    // Equal handles are the same Native contents. Independently computed states with equal
    // semantic keys remain separate candidates; Native must not splice their KV and state.
    std::vector<SourceChoice> unique;
    for (auto& choice : result) {
        const auto duplicate =
            std::any_of(unique.begin(), unique.end(), [&](const auto& prior) {
                return prior.source.checkpoint == choice.source.checkpoint &&
                       prior.source.consume_source == choice.source.consume_source &&
                       prior.source.private_points == choice.source.private_points;
            });
        if (!duplicate) { unique.push_back(std::move(choice)); }
    }
    return unique;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::prepare_source(Program& program, const Base& base, SourceChoice& choice,
                                  std::uint64_t request                   ,
                                  std::optional<ReclaimRights> revocation ) -> bool
{
    const auto selected = choice.source.checkpoint;
    if (selected && !program.valid_checkpoint(*selected)) { return false; }
    auto* owner = find_owner(choice.owner);
    if (choice.take_over && (!owner || owner->publication_order != choice.publication_order ||
                             (owner->active && choice.resume_owner != choice.owner))) {
        choice.take_over = false;
    }
    std::vector<Handle> retired;
    if (choice.take_over && selected && owner) {
        const auto frontier = program.checkpoint_metadata(*selected).frontier;
        for (const auto& point : owner->points) {
            if (point.handle == *selected || !program.valid_checkpoint(point.handle)) {
                continue;
            }
            const auto summary = program.checkpoint_metadata(point.handle);
            if (summary.frontier > frontier && !summary.leased &&
                consumable(point.handle, owner->token, request, revocation)) {
                retired.push_back(point.handle);
            }
        }
    }
    auto carried = points(choice.resume_owner.value_or(choice.owner));
    if (const auto* waiting = find_waiting(request);
        waiting && waiting->choice.source.checkpoint == selected) {
        carried = waiting->choice.source.private_points;
        std::erase_if(carried, [&](auto point) { return !contains(waiting->held, point); });
    }
    const auto private_owner =
        choice.resume_owner.value_or(choice.take_over ? choice.owner : 0);
    const bool consume = selected && consumable(*selected, private_owner, request, revocation);
    auto inspected     = program.inspect_source(base, selected, consume, carried, retired);
    if (!inspected) { return false; }
    inspected->take_private =
        std::all_of(inspected->private_points.begin(), inspected->private_points.end(),
                    [&](auto point) { return consumable(point, private_owner, request); });
    choice.source = std::move(*inspected);
    return true;
}

template <class Model>
auto
ResourceManager<Model>::retain_source(Program& program, std::uint64_t request, const SourceChoice& choice) -> void
{
    std::vector<Handle> held;
    if (choice.source.checkpoint) { held.push_back(*choice.source.checkpoint); }
    for (const auto point : choice.source.private_points) {
        if (!contains(held, point)) { held.push_back(point); }
    }
    auto* waiting = find_waiting(request);
    if (!waiting) {
        if (held.empty()) { return; }
        waiting_.push_back({.request = request, .choice = choice, .held = std::move(held)});
        return;
    }
    auto old        = std::move(waiting->held);
    waiting->choice = choice;
    waiting->held   = std::move(held);
    for (const auto point : old) { release_unreferenced(program, point); }
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::retained_source(std::uint64_t request) const -> std::optional<SourceChoice>
{
    const auto* waiting = find_waiting(request);
    if (!waiting || !waiting->choice.source.checkpoint ||
        !contains(waiting->held, *waiting->choice.source.checkpoint)) {
        return std::nullopt;
    }
    return waiting->choice;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::has_source_record(std::uint64_t request) const -> bool
{
    return find_waiting(request) != nullptr;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::can_transfer_private(OwnerToken owner, std::uint64_t request) const -> bool
{
    const auto carried = points(owner);
    return std::all_of(carried.begin(), carried.end(),
                       [&](auto point) { return consumable(point, owner, request); });
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::source_revocations(std::uint64_t request) const -> std::uint32_t
{
    const auto* waiting = find_waiting(request);
    return waiting ? waiting->revocations : 0;
}

template <class Model>
auto
ResourceManager<Model>::binding_started(std::uint64_t request, std::span<const Handle> retired,
                     std::optional<Handle> consumed) -> void
{
    if (auto* waiting = find_waiting(request)) { waiting->binding = true; }
    for (const auto handle : retired) { forget(handle); }
    if (consumed) {
        for (auto& waiting : waiting_) {
            if (waiting.request != request && std::erase(waiting.held, *consumed)) {
                ++waiting.revocations;
            }
        }
    }
}

template <class Model>
auto
ResourceManager<Model>::release_source(Program& program, std::uint64_t request) -> void
{
    const auto found = std::find_if(waiting_.begin(), waiting_.end(), [=](const auto& entry) {
        return entry.request == request;
    });
    if (found == waiting_.end()) { return; }
    auto held = std::move(found->held);
    waiting_.erase(found);
    for (const auto handle : held) { release_unreferenced(program, handle); }
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::adopt(Program& program, const SourceChoice& choice, const Base& base,
                               std::uint64_t publication_order,
                               std::span<const Handle> carried ,
                               std::span<const Handle> retired ) -> OwnerToken
{
    for (const auto handle : retired) { forget(handle); }
    if (!enabled_ || !base.summary().publish_continuation) {
        for (const auto handle : carried) { release_unreferenced(program, handle); }
        return 0;
    }
    OwnerToken token = 0;
    CacheRetentionPriority priority;
    std::uint32_t proven_frontier = 0;
    if (choice.resume_owner) {
        token = *choice.resume_owner;
        if (!find_owner(token)) { throw std::logic_error("resume lost continuation owner"); }
    } else {
        if (choice.source.reused_tokens) {
            const auto used_at = ++clock_;
            if (const auto key = base.prefix_shortlist_key(choice.source.reused_tokens)) {
                remember_demand(*key, 0,
                                CacheRetentionPriority{.reused = true, .last_demand = used_at});
            }
            if (auto* owner = find_owner(choice.owner)) {
                priority        = {.reused = true, .last_demand = used_at};
                proven_frontier = choice.source.reused_tokens;
                // Consuming a branch replaces its proof. A shallower independent
                // Fork has its own new evidence and leaves its donor's pair intact.
                if (choice.take_over || proven_frontier >= owner->proven_frontier) {
                    owner->priority        = priority;
                    owner->proven_frontier = proven_frontier;
                }
                if (choice.take_over) { token = owner->token; }
            } else if (auto* shared = find_shared(choice.shared_entry)) {
                shared->priority = {.reused = true, .last_demand = used_at};
            }
        }
        if (!token) {
            token = ++ordinal_;
            owners_.push_back({.token           = token,
                               .priority        = priority,
                               .proven_frontier = proven_frontier,
                               .retained_at     = ++clock_});
        }
    }
    auto* owner              = find_owner(token);
    owner->active            = true;
    owner->publication_order = publication_order;
    replace_points(program, *owner, carried);
    prune(program);
    balance_reused(program);
    return token;
}

template <class Model>
auto
ResourceManager<Model>::publish(Program& program, OwnerToken token, Handle handle) -> void
{
    if (!program.valid_checkpoint(handle)) { return; }
    const auto summary = program.checkpoint_metadata(handle);
    if (!enabled_ || !token) {
        release_unreferenced(program, handle);
        return;
    }
    if (summary.role == CheckpointRole::SharedPrefix) {
        if (std::none_of(shared_.begin(), shared_.end(),
                         [&](const auto& entry) { return entry.handle == handle; })) {
            shared_.push_back(
                {.handle   = handle,
                 .ordinal  = ++ordinal_,
                 .priority = demand_priority(program.checkpoint_key(handle, summary.frontier)),
                 .retained_at = ++clock_});
            index_.insert(program, handle, summary.frontier);
        }
    } else {
        auto* owner = find_owner(token);
        if (!owner) {
            throw std::logic_error("checkpoint publication lost continuation owner");
        }
        const auto observed = demand_priority(program.checkpoint_key(handle, summary.frontier));
        if (summary.role == CheckpointRole::InputReplay && observed.reused &&
            (!owner->priority.reused || observed.last_demand > owner->priority.last_demand)) {
            owner->priority        = observed;
            owner->proven_frontier = summary.frontier;
        }
        if (std::none_of(owner->points.begin(), owner->points.end(),
                         [&](const auto& p) { return p.handle == handle; })) {
            std::optional<Handle> replace;
            std::size_t anchors = 0;
            std::optional<Handle> oldest_anchor;
            for (const auto& point : owner->points) {
                if (point.role != summary.role) { continue; }
                if (summary.role != CheckpointRole::LongAnchor) {
                    replace = point.handle;
                    break;
                }
                ++anchors;
                if (!oldest_anchor) { oldest_anchor = point.handle; }
                if (program.checkpoint_metadata(point.handle).frontier == summary.frontier) {
                    replace = point.handle;
                }
            }
            if (!replace && anchors >= 4) { replace = oldest_anchor; }
            if (replace && !remove_point(program, *owner, *replace)) {
                // Optional replacement never turns a legitimate request into a failure.
                release_unreferenced(program, handle);
                return;
            }
            owner->points.push_back({handle, summary.role});
            index_.insert(program, handle, summary.frontier);
        }
    }
    balance_reused(program);
}

template <class Model>
auto
ResourceManager<Model>::finish(Program& program, OwnerToken token, const Base& base,
            std::uint64_t publication_order) -> void
{
    auto* owner = find_owner(token);
    if (!owner) { return; }
    owner->active            = false;
    owner->publication_order = publication_order;
    const auto& advice       = base.context_cache();
    if (advice.update_session_index && advice.session_key) {
        const auto newer = std::any_of(owners_.begin(), owners_.end(), [&](const auto& other) {
            return other.token != token && other.session == advice.session_key &&
                   other.publication_order > publication_order;
        });
        if (!newer) {
            for (auto& other : owners_) {
                if (other.session == advice.session_key) { other.session.reset(); }
            }
            owner->session = advice.session_key;
        }
    }
    prune(program);
    balance_reused(program);
}

template <class Model>
auto
ResourceManager<Model>::abandon(Program& program, OwnerToken token) -> void
{
    auto* owner = find_owner(token);
    if (!owner) { return; }
    const auto old = points(token);
    std::erase_if(owners_, [&](const auto& entry) { return entry.token == token; });
    for (const auto handle : old) { release_unreferenced(program, handle); }
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::points(OwnerToken token) const -> std::vector<Handle>
{
    std::vector<Handle> out;
    if (const auto* owner = find_owner(token)) {
        for (const auto& point : owner->points) { out.push_back(point.handle); }
    }
    return out;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::retention(OwnerToken token) const -> CacheRetentionPriority
{
    const auto* owner = find_owner(token);
    return owner ? owner->priority : CacheRetentionPriority{};
}

template <class Model>
auto
ResourceManager<Model>::recycle_input(Program& program, OwnerToken token) -> bool
{
    auto* owner = find_owner(token);
    if (!owner) { return false; }
    const auto point =
        std::find_if(owner->points.begin(), owner->points.end(),
                     [](const auto& p) { return p.role == CheckpointRole::InputReplay; });
    if (point == owner->points.end()) { return false; }
    const auto handle = point->handle;
    if (program.valid_checkpoint(handle) && program.checkpoint_metadata(handle).leased) {
        return false;
    }
    return remove_point(program, *owner, handle);
}

template <class Model>
auto
ResourceManager<Model>::observe_committed(const Base& base, std::uint64_t request_id, std::uint32_t begin,
                       std::uint32_t end) -> void
{
    if (!enabled_ || begin >= end) { return; }
    for (const auto frontier : base.capture_frontiers()) {
        if (frontier <= begin || frontier > end) { continue; }
        const auto key = base.prefix_shortlist_key(frontier);
        if (!key) { continue; }
        remember_demand(*key, request_id);
    }
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::capture_admission(Program&, OwnerToken owner, const Base& base,
                                          std::uint32_t frontier,
                                          bool inherit_private ) const -> Admission
{
    const auto demand = demand_priority(base.prefix_shortlist_key(frontier));
    // A recurrent continuation's heat can rank its owner, but a new optional point
    // has not itself demonstrated repeat demand. Its added coverage remains bounded.
    auto priority = demand;
    if (!demand.reused && inherit_private) { priority = retention(owner); }
    return {.priority = priority, .demand = demand, .base = &base, .frontier = frontier};
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::begin_reclaim(Program& program,
                                          std::optional<Admission> admission ,
                                          ReclaimRights rights               ) -> ReclaimCursor
{
    prune(program);
    balance_reused(program);
    auto cursor      = reclaim_order(program);
    cursor.admission = admission;
    cursor.rights    = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
    if (admission && !admission->demand.reused) {
        cursor.gain_limit = recovery_gain(program, *admission, surviving(program, {}));
    }
    return cursor;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::host_victims(Program& program, std::size_t bytes, std::optional<Handle> incoming,
             std::span<const Handle> later_snapshots ,
             std::span<const Handle> excluded        ,
             std::optional<Admission> admission      ,
             const ReclaimCursor* decision , ReclaimRights rights ) -> std::optional<std::vector<Handle>>
{
    prune(program);
    Evaluation evaluation;
    if (decision) {
        return host_victims_in_order(program, bytes, incoming, later_snapshots, excluded,
                                     decision->admission, *decision, evaluation);
    }
    auto cursor   = reclaim_order(program);
    cursor.rights = admission ? ReclaimRights{ReclaimPurpose::OptionalWrite} : rights;
    return host_victims_in_order(program, bytes, incoming, later_snapshots, excluded, admission,
                                 cursor, evaluation);
}

template <class Model>
auto
ResourceManager<Model>::commit_host_victims(Program& program, std::span<const Handle> handles,
                         ReclaimCursor& cursor) -> void
{
    commit_release(program, handles, cursor);
}

template <class Model>
auto
ResourceManager<Model>::erase(Program& program, Handle handle, ReclaimRights rights ) -> bool
{
    if (!references(handle)) { return false; }
    if (!may_revoke({&handle, 1}, rights)) { return false; }
    if (program.valid_checkpoint(handle) && !program.release_checkpoint(handle)) {
        return false;
    }
    forget(handle);
    prune(program);
    return true;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::reclaim(Program& program, ContextResourceUsage shortage,
                                      std::span<const Handle> excluded ) -> ReclaimProgress
{
    if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
    auto cursor = begin_reclaim(program);
    return reclaim(program, shortage, excluded, cursor);
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::reclaim(Program& program, ContextResourceUsage shortage,
                                      std::span<const Handle> excluded, ReclaimCursor& cursor) -> ReclaimProgress
{
    if (program.has_context_transaction()) { return ReclaimProgress::Transferring; }
    prune(program);
    if (shortage.host_bytes && program.release_redundant_host(excluded)) {
        return ReclaimProgress::Changed;
    }
    std::vector<Handle> allowed;
    for (const auto& victim : cursor.victims) {
        if (references(victim.handle) && program.valid_checkpoint(victim.handle) &&
            !contains(excluded, victim.handle)) {
            allowed.push_back(victim.handle);
        }
    }
    if (shortage.host_bytes) {
        Evaluation evaluation;
        const auto victims =
            host_victims_in_order(program, shortage.host_bytes, std::nullopt, {}, excluded,
                                  cursor.admission, cursor, evaluation, true);
        if (!victims) { return ReclaimProgress::Blocked; }
        commit_release(program, *victims, cursor);
        return victims->empty() ? ReclaimProgress::Blocked : ReclaimProgress::Changed;
    }
    auto plans = program.plan_reclaim(allowed, excluded, shortage);
    Evaluation evaluation{.native = &plans};
    using Demotion = typename decltype(plans.demotions)::value_type;
    using Release  = typename decltype(plans.releases)::value_type;

    struct Action {
        std::optional<Demotion> demotion;
        std::optional<Release> release;
        std::vector<Handle> host_victims;
        std::optional<Admission> preservation;
        ActionRank rank;
    };

    const auto usage     = program.physical_usage();
    const auto free_host = usage.capacity.host_bytes - usage.occupied.host_bytes;
    const bool needs_host_cleanup =
        std::any_of(plans.demotions.begin(), plans.demotions.end(),
                    [&](const auto& plan) { return plan.host_bytes > free_host; });
    const bool existing_host_destination =
        std::any_of(plans.demotions.begin(), plans.demotions.end(),
                    [](const auto& plan) { return plan.host_bytes == 0; });
    // Redundant-replica cleanup is a distinct capacity change. Quotes below are read
    // only and never observe side effects from evaluating another candidate.
    if (needs_host_cleanup && !existing_host_destination &&
        program.release_redundant_host(excluded)) {
        return ReclaimProgress::Changed;
    }

    std::vector<Action> actions;
    actions.reserve(plans.demotions.size() + plans.releases.size());
    for (auto& plan : plans.demotions) {
        const auto units = useful(plan.released, shortage);
        if (!units) { continue; }
        auto facts = action_facts(program, plan.sources, cursor, evaluation);
        if (!admits(program, plan.sources, facts, cursor.admission, cursor, evaluation, true)) {
            continue;
        }
        auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
        protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                 plan.sources.end());
        auto preservation =
            preservation_admission(program, plan.sources, cursor, facts.priority);
        auto victims =
            host_victims_in_order(program, plan.host_bytes, preservation.checkpoint, {},
                                  protected_sources, preservation, cursor, evaluation);
        if (!victims) { continue; }
        auto rank       = rank_action(program, plan.sources, facts, units, evaluation, true);
        auto host_facts = action_facts(program, *victims, cursor, evaluation);
        const auto host_rank = rank_action(program, *victims, host_facts, units, evaluation);
        rank.loss            = host_rank.loss;
        rank.waiting         = host_rank.waiting;
        rank.order           = std::max(rank.order, host_rank.order);
        if (host_rank.priority.reused &&
            (!rank.priority.reused ||
             host_rank.priority.last_demand > rank.priority.last_demand)) {
            rank.priority = host_rank.priority;
        }
        actions.push_back({.demotion     = std::move(plan),
                           .host_victims = std::move(*victims),
                           .preservation = std::move(preservation),
                           .rank         = rank});
    }
    for (auto& plan : plans.releases) {
        const auto units = useful(plan.released, shortage);
        if (!units) { continue; }
        auto facts = action_facts(program, plan.sources, cursor, evaluation);
        if (!admits(program, plan.sources, facts, cursor.admission, cursor, evaluation)) {
            continue;
        }
        auto rank = rank_action(program, plan.sources, facts, units, evaluation);
        actions.push_back({.release = std::move(plan), .rank = rank});
    }
    std::stable_sort(actions.begin(), actions.end(),
                     [](const auto& a, const auto& b) { return less_rank(a.rank, b.rank); });
    for (std::size_t index = 0; index < actions.size(); ++index) {
        const auto& action = actions[index];
        if (action.demotion) {
            if (!shortage.state_slots && action.host_victims.empty()) {
                // Preserve the ranked action order. Independent KV groups can share
                // one transfer transaction while every holder keeps its own rights.
                auto builder           = plans.begin_kv_batch(shortage);
                std::size_t components = 0;
                std::uint64_t released = 0;
                const auto needed =
                    shortage.main_kv_pages ? shortage.main_kv_pages : shortage.backend_kv_pages;
                for (std::size_t next = index; next < actions.size() && released < needed;
                     ++next) {
                    const auto& candidate = actions[next];
                    if (!candidate.demotion) {
                        // These are alternative ways to reclaim physical resources.
                        // A delete covered by earlier migrations adds no pages to this
                        // deficit; keep it in actions for failure fallback, not the batch.
                        if (builder.covers(*candidate.release)) { continue; }
                        break;
                    }
                    if (!candidate.host_victims.empty()) { break; }
                    if (!builder.append(*candidate.demotion)) { break; }
                    ++components;
                    released += useful(candidate.demotion->released, shortage);
                }
                if (components > 1) {
                    if (const auto batch = builder.finish()) {
                        auto facts = action_facts(program, batch->sources, cursor, evaluation);
                        if (admits(program, batch->sources, facts, cursor.admission, cursor,
                                   evaluation, true) &&
                            program.start_demote(*batch)) {
                            return program.has_context_transaction()
                                       ? ReclaimProgress::Transferring
                                       : ReclaimProgress::Changed;
                        }
                    }
                    // A combined reservation may fail without changing any replica.
                    // The original first action remains the next legal choice.
                }
            }
            const auto& plan       = *action.demotion;
            auto protected_sources = std::vector<Handle>(excluded.begin(), excluded.end());
            protected_sources.insert(protected_sources.end(), plan.sources.begin(),
                                     plan.sources.end());
            const auto current = host_victims_in_order(
                program, plan.host_bytes, action.preservation->checkpoint, {},
                protected_sources, action.preservation, cursor, evaluation);
            if (!current || *current != action.host_victims) { continue; }
            commit_release(program, action.host_victims, cursor, action.rank.loss);
            if (program.start_demote(plan)) {
                return program.has_context_transaction() ? ReclaimProgress::Transferring
                                                         : ReclaimProgress::Changed;
            }
            if (!action.host_victims.empty()) { return ReclaimProgress::Changed; }
            continue;
        }
        // Native quoted every holder required for this actual physical release.
        commit_release(program, action.release->sources, cursor, action.rank.loss);
        return ReclaimProgress::Changed;
    }
    return ReclaimProgress::Blocked;
}

template <class Model>
auto
ResourceManager<Model>::release_all(Program& program) noexcept -> void
{
    const auto handles = victims();
    index_.clear();
    owners_.clear();
    shared_.clear();
    waiting_.clear();
    demand_.clear();
    for (const auto handle : handles) {
        if (program.valid_checkpoint(handle)) { (void)program.release_checkpoint(handle); }
    }
}

template <class Model>
auto
ResourceManager<Model>::consumable(Handle handle, OwnerToken owner, std::uint64_t request,
                std::optional<ReclaimRights> revocation ) const -> bool
{
    std::size_t allowed = 0;
    if (const auto* entry = find_owner(owner)) {
        allowed = std::count_if(entry->points.begin(), entry->points.end(),
                                [&](const auto& point) { return point.handle == handle; });
    }
    if (catalog_reference_count(handle) != allowed) { return false; }
    for (const auto& waiting : waiting_) {
        if (!contains(waiting.held, handle)) { continue; }
        if (waiting.binding) { return false; }
        if (waiting.request != request &&
            (!revocation || revocation->purpose == ReclaimPurpose::OptionalWrite ||
             (revocation->purpose == ReclaimPurpose::FreshAdmission &&
              waiting.request <= revocation->request))) {
            return false;
        }
        ++allowed;
    }
    return allowed != 0;
}

template <class Model>
[[nodiscard]] auto
ResourceManager<Model>::waiting_ticket(std::span<const Handle> handles) const -> std::uint64_t
{
    std::uint64_t oldest = 0;
    for (const auto& waiting : waiting_) {
        if (std::any_of(handles.begin(), handles.end(),
                        [&](auto handle) { return contains(waiting.held, handle); })) {
            oldest = oldest ? std::min(oldest, waiting.request) : waiting.request;
        }
    }
    return oldest;
}

template <class Model>
auto
ResourceManager<Model>::may_revoke(std::span<const Handle> handles, ReclaimRights rights) const -> bool
{
    for (const auto& waiting : waiting_) {
        if (!std::any_of(handles.begin(), handles.end(),
                         [&](auto handle) { return contains(waiting.held, handle); })) {
            continue;
        }
        if (waiting.binding || rights.purpose == ReclaimPurpose::OptionalWrite ||
            (rights.purpose == ReclaimPurpose::FreshAdmission &&
             waiting.request <= rights.request)) {
            return false;
        }
    }
    return true;
}

} // namespace ninfer::runtime
