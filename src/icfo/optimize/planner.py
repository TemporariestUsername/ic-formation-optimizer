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
    for c in pool:
        # variant requirements must stay available; so must the objective's role
        # champions, whose value marginal damage probes can't see
        if (c.id in constraints.required_ids or any(r(c) for r in constraints.forced_slot_rules.values())
                or (role and role in c.tags and c.seat != carry.seat)):
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
    return Plan(best_carry, best_res.placement, best_res.score, evals, [c.name for c in shortlist])
