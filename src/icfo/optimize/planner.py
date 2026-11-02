"""End-to-end planning: pick the carry, prune the pool, search.

With ~160 champions and up to 10 slots the full space is far too large to
search for every candidate carry, so planning is staged:

1. **Carry shortlist.** Every allowed ``dps`` champion is scored alone; the
   best few go forward (or just the carry the user named).
2. **Pool pruning.** For each shortlisted carry, every other champion's
   marginal value next to it is probed in a few representative slots
   (adjacent, same column, far away). The top ``keep`` survive, plus anything
   the variant requires.
3. **Search.** Annealing on the pruned pool from a constructive start; the
   ``thorough`` profile adds CP-SAT (when OR-Tools is installed) and longer
   annealing on the best carries.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from ..model.definitions import Champion
from ..model.variants import Constraints
from .search import Infeasible, SearchResult, _seed, anneal

PROFILES = {
    #             shortlist, finalists, keep, anneal iters, final iters, cpsat
    "fast": dict(shortlist=4, finalists=1, keep=22, iters=700, final_iters=0, cpsat=False),
    "thorough": dict(shortlist=8, finalists=2, keep=35, iters=2500, final_iters=6000, cpsat=True),
}


@dataclass
class Plan:
    carry: Champion
    placement: dict[int, Champion]
    result: object  # ScoreResult
    evaluations: int
    carries_tried: list[str] = field(default_factory=list)
    specs: dict[int, dict[int, int]] = field(default_factory=dict)  # hero -> {group: upgrade id}
    spec_changes: list[tuple[str, str, str]] = field(default_factory=list)  # (champion, from, to)


def optimize_specs(scorer, placement: dict[int, Champion], carry: Champion, rounds: int = 2):
    """Coordinate ascent over specialization choices of the placed champions.

    Specializations are saved per formation in game, so the best choice for
    this formation and objective is a legitimate recommendation. Leaves the
    scorer set to the chosen specs; returns (result, specs, changes).
    """
    base = scorer.scorer if hasattr(scorer, "scorer") else scorer
    data = base.data
    start = {h: dict(base.specs.get(h, {})) for h in (c.id for c in placement.values())}
    best = scorer.score(placement, carry.id)
    for _ in range(rounds):
        improved = False
        for champ in placement.values():
            for gi, group in enumerate(champ.spec_groups):
                if len(group) < 2:
                    continue
                current = base.specs.get(champ.id, {}).get(gi, group[0].id)
                for option in group:
                    if option.id == current:
                        continue
                    base.specs.setdefault(champ.id, {})[gi] = option.id
                    base.invalidate(champ.id)
                    res = scorer.score(placement, carry.id)
                    if res.log10 > best.log10 + 1e-9:
                        best, current, improved = res, option.id, True
                base.specs.setdefault(champ.id, {})[gi] = current
                base.invalidate(champ.id)
        if not improved:
            break
    changes = []
    for champ in placement.values():
        for gi, group in enumerate(champ.spec_groups):
            before = start.get(champ.id, {}).get(gi, group[0].id)
            after = base.specs.get(champ.id, {}).get(gi, group[0].id)
            if before != after:
                name = lambda uid: data.upgrades[uid].specialization_name or data.upgrades[uid].name
                changes.append((champ.name, name(before), name(after)))
    chosen = {c.id: dict(base.specs.get(c.id, {})) for c in placement.values() if c.spec_groups}
    return scorer.score(placement, carry.id), chosen, changes


def _anchor_slot(constraints: Constraints, carry: Champion) -> int | None:
    """An open slot for the carry with the most neighbours (where buffs land)."""
    layout = constraints.layout
    options = [
        s for s in constraints.open_slots
        if s not in constraints.forced_slot_rules or constraints.forced_slot_rules[s](carry)
    ]
    if not options:
        return None
    return max(options, key=lambda s: (len(layout.slots[s].adj - constraints.blocked_slots), -s))


def _priority_tag(scorer) -> str | None:
    """Role the objective depends on beyond damage: speed (gems) or gold (favor)."""
    name = getattr(getattr(scorer, "objective", None), "name", None)
    return {"gems": "speed", "favor": "gold"}.get(name)


def _channel_keys(scorer) -> frozenset:
    from ..engine.catalog import GOLD_KEYS, SPEED_KEYS
    name = getattr(getattr(scorer, "objective", None), "name", None)
    return {"gems": SPEED_KEYS, "favor": GOLD_KEYS}.get(name, frozenset())


def _touches_channel(champ: Champion, keys, feats=()) -> bool:
    """Has an effect (from upgrades or the given feats) in the objective's channel."""
    if not keys:
        return False
    for u in champ.upgrades:
        if any(e.key in keys for e in u.effects):
            return True
    return any(e.key in keys for f in feats for e in f.effects)


def prune_pool(scorer, constraints: Constraints, pool: list[Champion], carry: Champion, keep: int) -> list[Champion]:
    k = _anchor_slot(constraints, carry)
    if k is None:
        return pool
    layout = constraints.layout
    others = [s for s in constraints.open_slots if s != k]
    probes: list[int] = []
    adj = [s for s in others if s in layout.slots[k].adj]
    same_col = [s for s in others if layout.slots[s].col == layout.slots[k].col]
    if adj:
        probes.append(adj[0])
    if same_col:
        probes.append(same_col[0])
    if others:
        far = max(others, key=lambda s: layout.distances(k).get(s, 99))
        probes.append(far)
    probes = list(dict.fromkeys(probes))
    base = scorer.score({k: carry}, carry.id).log10
    gains = []
    for c in pool:
        if c.id == carry.id or c.seat == carry.seat:
            continue
        best = max(scorer.score({k: carry, s: c}, carry.id).log10 for s in probes) if probes else 0.0
        gains.append((best - base, c))
    gains.sort(key=lambda t: -t[0])
    kept = {c.id: c for _, c in gains[:keep]}
    role = _priority_tag(scorer)
    keys = _channel_keys(scorer)
    for c in pool:
        # variant requirements must stay available; so must champions serving the
        # objective's channel (speed/gems, gold), whose value damage probes can't see
        if (c.id in constraints.required_ids or any(r(c) for r in constraints.forced_slot_rules.values())
                or (c.seat != carry.seat and ((role and role in c.tags) or _touches_channel(c, keys, c.feats)))):
            kept.setdefault(c.id, c)
    kept[carry.id] = carry
    return list(kept.values())


def _constructive_start(scorer, constraints, pool, carry) -> dict[int, Champion]:
    """Seed + fill free slots in order of marginal gain (cheap, never infeasible)."""
    k = _anchor_slot(constraints, carry)
    if k is None:
        raise Infeasible("no slot can hold the carry")
    placement = _seed(constraints, pool, carry, k)
    used = {c.seat for c in placement.values()}
    for c in pool:
        if c.seat in used:
            continue
        free = [s for s in constraints.open_slots if s not in placement]
        if not free:
            break
        best = None
        for s in free:
            trial = {**placement, s: c}
            if constraints.placement_ok(trial):
                v = scorer.score(trial, carry.id).log10
                if best is None or v > best[0]:
                    best = (v, s)
        if best is not None and best[0] >= scorer.score(placement, carry.id).log10:
            placement[best[1]] = c
            used.add(c.seat)
    return placement


def plan(scorer, constraints: Constraints, pool: list[Champion], carry: Champion | None = None,
         profile: str = "fast", seed: int = 0) -> Plan:
    cfg = PROFILES[profile]
    if carry is not None:
        shortlist = [carry]
    else:
        dps = [c for c in pool if "dps" in c.tags] or list(pool)
        alone = []
        for c in dps:
            k = _anchor_slot(constraints, c)
            if k is None:
                continue
            try:
                _seed(constraints, pool, c, k)
            except Infeasible:
                continue
            alone.append((scorer.score({k: c}, c.id).log10, c))
        alone.sort(key=lambda t: -t[0])
        shortlist = [c for _, c in alone[: cfg["shortlist"]]]
    if not shortlist:
        raise Infeasible("no carry can be placed under these restrictions")

    results: list[tuple[SearchResult, Champion, list[Champion]]] = []
    evals = 0
    for c in shortlist:
        sub = prune_pool(scorer, constraints, pool, c, cfg["keep"])
        try:
            start = _constructive_start(scorer, constraints, sub, c)
            res = anneal(scorer, constraints, sub, c, start=start, iterations=cfg["iters"], seed=seed)
        except Infeasible:
            continue
        evals += res.evaluations
        results.append((res, c, sub))
    if not results:
        raise Infeasible("no feasible formation found")
    results.sort(key=lambda t: -t[0].score.log10)

    finals = results[: cfg["finalists"]]
    best_res, best_carry, _ = finals[0]
    if cfg["final_iters"] or cfg["cpsat"]:
        for res, c, sub in finals:
            cand = res
            if cfg["cpsat"] and importlib.util.find_spec("ortools") is not None:
                from .cpsat import solve
                try:
                    cp = solve(scorer, constraints, sub, c, time_limit_s=5, polish_iterations=0)
                    if cp.score.log10 > cand.score.log10:
                        cand = cp
                except Infeasible:
                    pass
            if cfg["final_iters"]:
                pol = anneal(scorer, constraints, sub, c, start=cand.placement,
                             iterations=cfg["final_iters"], t_start=0.2, seed=seed + 1)
                evals += pol.evaluations
                if pol.score.log10 > cand.score.log10:
                    cand = pol
            if cand.score.log10 > best_res.score.log10:
                best_res, best_carry = cand, c
    final, specs, changes = optimize_specs(scorer, best_res.placement, best_carry)
    return Plan(best_carry, best_res.placement, final, evals, [c.name for c in shortlist], specs, changes)


# -- feats and acquisition tiers -------------------------------------------------

TIERS = ("base", "gems", "full")
TIER_LABELS = {
    "base": "current account (owned feats, free choice of which to equip)",
    "gems": "almost full (+ feats with a gem price)",
    "full": "full (+ platinum-only shop, event, chest, patron and campaign feats)",
}


def feat_choices(champ: Champion, user, tier: str) -> list:
    """Feats a champion may equip in ``tier``."""
    owned = user.unlocked_feats.get(champ.id, set()) if user.from_account else set()
    out = []
    for f in champ.feats:
        if f.id in owned or f.tier == "free" and f.available:
            out.append(f)
        elif tier in ("gems", "full") and f.tier == "gems" and f.available:
            out.append(f)
        elif tier == "full" and f.available:
            out.append(f)
    return out


def _relevant(feat) -> bool:
    """Skip feats whose effects can't move any score (pure bookkeeping/cosmetic)."""
    from ..engine.catalog import NON_DAMAGE_KEYS, PARAM_KEYS
    return any(e.key not in PARAM_KEYS and e.key not in NON_DAMAGE_KEYS for e in feat.effects)


PURCHASE_MARGIN = 0.002  # log10 (~0.5%): a feat you'd have to buy must clearly beat what you own


def optimize_feats(scorer, placement: dict[int, Champion], carry: Champion, user, tier: str, rounds: int = 1):
    """Coordinate ascent over each placed champion's feat slots.

    Starts from what's equipped (kept where nothing modelled beats it), then
    tries every allowed feat in every slot. Leaves the scorer set to the
    chosen feats; returns (result, {hero: feat ids}).
    """
    base = scorer.scorer if hasattr(scorer, "scorer") else scorer
    best = scorer.score(placement, carry.id)
    for _ in range(rounds):
        improved = False
        for champ in placement.values():
            options = [f for f in feat_choices(champ, user, tier) if _relevant(f)]
            if not options:
                continue
            allowed = {f.id for f in feat_choices(champ, user, tier)}
            n_slots = user.feat_slots.get(champ.id, 4) if user.from_account else 4
            current = [f for f in base.feats.get(champ.id, ()) if f in allowed][:n_slots]
            current += [None] * (n_slots - len(current))
            owned = user.unlocked_feats.get(champ.id, set()) if user.from_account else set()
            for slot in range(n_slots):
                for f in options:
                    if f.id in current:
                        continue
                    trial = list(current)
                    trial[slot] = f.id
                    base.feats[champ.id] = tuple(x for x in trial if x is not None)
                    base.invalidate(champ.id)
                    res = scorer.score(placement, carry.id)
                    margin = 1e-6 if (f.id in owned or f.tier == "free") else PURCHASE_MARGIN
                    if res.log10 > best.log10 + margin:
                        best, current, improved = res, trial, True
                base.feats[champ.id] = tuple(x for x in current if x is not None)
                base.invalidate(champ.id)
        if not improved:
            break
    chosen = {c.id: tuple(base.feats.get(c.id, ())) for c in placement.values()}
    return scorer.score(placement, carry.id), chosen


def _prefit_feats(scorer, constraints, candidates, placement, carry, user, tier) -> None:
    """Give each champion outside the formation its best feats next to the carry
    in isolation, so the re-search can see what new feats make them worth."""
    k = next((sl for sl, ch in placement.items() if ch.id == carry.id), None)
    if k is None:
        return
    layout = constraints.layout
    near = [sl for sl in constraints.open_slots if sl != k and sl in layout.slots[k].adj] or \
           [sl for sl in constraints.open_slots if sl != k]
    if not near:
        return
    placed = {ch.id for ch in placement.values()}
    keys = _channel_keys(scorer)
    for c in candidates:
        if c.id in placed or c.seat == carry.seat:
            continue
        options = [f for f in feat_choices(c, user, tier) if _relevant(f)]
        if not options:
            continue
        if keys and not _touches_channel(c, keys, options) and tier == "base":
            continue
        optimize_feats(scorer, {k: carry, near[0]: c}, carry, user, tier)


@dataclass
class TierPlan:
    tier: str
    plan: Plan
    feats: dict[int, tuple[int, ...]]  # hero -> equipped feat ids in this optimum
    purchases: list  # [(champion name, Feat)] not yet owned
    gem_cost: int

    @property
    def value(self) -> float:
        return self.plan.result.log10


def plan_tiers(scorer, constraints: Constraints, pool: list[Champion], user, carry: Champion | None = None,
               profile: str = "fast", tiers=TIERS, seed: int = 0) -> dict[str, TierPlan]:
    """Base / almost-full / full optimum. Each tier starts from the previous
    tier's formation and feats and only accepts improvements, so the values
    are monotone (full >= gems >= base)."""
    cfg = PROFILES[profile]
    base = scorer.scorer if hasattr(scorer, "scorer") else scorer
    data = base.data
    out: dict[str, TierPlan] = {}
    prev: Plan | None = None
    for tier in tiers:
        if prev is None:
            p = plan(scorer, constraints, pool, carry, profile, seed=seed)
        else:
            p = prev
        placement, c = dict(p.placement), p.carry
        res, _ = optimize_feats(scorer, placement, c, user, tier)
        # the formation may change once feats change: pre-fit feats for candidates
        # outside the formation, re-search briefly, then re-fit feats/specs
        sub = prune_pool(scorer, constraints, pool, c, cfg["keep"])
        _prefit_feats(scorer, constraints, sub, placement, c, user, tier)
        again = anneal(scorer, constraints, sub, c, start=placement, iterations=max(300, cfg["iters"] // 2),
                       t_start=0.1, seed=seed + 7)
        if again.score.log10 > res.log10 + 1e-9:
            placement = again.placement
        res, specs, changes = optimize_specs(scorer, placement, c)
        res, feats = optimize_feats(scorer, placement, c, user, tier)
        if prev is not None and res.log10 < prev.result.log10:
            # never report a worse optimum for a richer tier
            res, placement, specs, changes = prev.result, prev.placement, prev.specs, prev.spec_changes
            feats = out[tiers[tiers.index(tier) - 1]].feats
            for h, f in feats.items():
                base.feats[h] = f
                base.invalidate(h)
        tp = Plan(c, placement, res, p.evaluations, p.carries_tried, specs, list(p.spec_changes) + changes)
        owned = user.unlocked_feats
        purchases = []
        for champ in placement.values():
            for fid in feats.get(champ.id, ()):
                feat = data.feats.get(fid)
                if feat is not None and feat.tier != "free" and fid not in owned.get(champ.id, set()):
                    purchases.append((champ.name, feat))
        cost = sum(f.gem_cost or 0 for _, f in purchases if f.tier == "gems")
        out[tier] = TierPlan(tier, tp, feats, purchases, cost)
        prev = tp
    return out
