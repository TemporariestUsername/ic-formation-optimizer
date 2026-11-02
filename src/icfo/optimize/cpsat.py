"""CP-SAT formulation (requires ``pip install icfo[solver]``).

With the carry pinned to slot ``k``, most buffs reaching the carry depend only
on *which champion sits in which slot* — so the objective is (close to) linear
in assignment booleans ``x[c, s]``. We build that linear surrogate from the
exact scorer:

    u[c, s] = score({carry@k, c@s}) - score({carry@k})

which is exact for independent multipliers and approximate when effects
interact (amplifiers, per-tag counts). CP-SAT solves the surrogate to proven
optimality for every ``k``; the best few are re-scored with the exact scorer
and polished by low-temperature annealing.

Roadmap: add pairwise terms ``p[(c,s),(c',s')]`` for known interactions
(buff_upgrade on a positional buff) via product booleans.
"""

from __future__ import annotations

from ..engine.score import Scorer
from ..model.definitions import Champion
from ..model.effects import match_tags
from ..model.variants import Constraints
from .search import Infeasible, SearchResult, anneal

SCALE = 10_000  # log10 units -> integer objective


def solve(
    scorer: Scorer,
    constraints: Constraints,
    pool: list[Champion],
    carry: Champion,
    *,
    time_limit_s: float = 10.0,
    polish_iterations: int = 5000,
) -> SearchResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CP-SAT needs OR-Tools: pip install 'icfo[solver]'") from exc

    others = [c for c in pool if c.id != carry.id and c.seat != carry.seat]
    best: SearchResult | None = None

    for k in constraints.open_slots:
        if k in constraints.forced_slot_rules and not constraints.forced_slot_rules[k](carry):
            continue
        base = scorer.score({k: carry}, carry.id).log10
        slots = [s for s in constraints.open_slots if s != k]

        model = cp_model.CpModel()
        x = {(c.id, s): model.NewBoolVar(f"x_{c.id}_{s}") for c in others for s in slots}
        for s in slots:
            model.Add(sum(x[c.id, s] for c in others) <= 1)
        seats: dict[int, list[Champion]] = {}
        for c in others:
            seats.setdefault(c.seat, []).append(c)
        for members in seats.values():
            model.Add(sum(x[c.id, s] for c in members for s in slots) <= 1)
        for rid in constraints.required_ids - {carry.id}:
            if not any(c.id == rid for c in others):
                raise Infeasible(f"required champion {rid} not available")
            model.Add(sum(x[rid, s] for s in slots) == 1)
        for tag, limit in constraints.max_by_tag.items():
            carry_has = 1 if match_tags(tag, carry.tags) else 0
            model.Add(sum(x[c.id, s] for c in others if match_tags(tag, c.tags) for s in slots) <= limit - carry_has)
        for slot, rule in constraints.forced_slot_rules.items():
            if slot == k:
                continue
            model.Add(sum(x[c.id, slot] for c in others if rule(c)) == 1)

        terms = []
        for c in others:
            for s in slots:
                gain = scorer.score({k: carry, s: c}, carry.id).log10 - base
                terms.append(int(round(gain * SCALE)) * x[c.id, s])
        model.Maximize(sum(terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        placement = {k: carry}
        by_id = {c.id: c for c in others}
        for (cid, s), var in x.items():
            if solver.Value(var):
                placement[s] = by_id[cid]
        if not constraints.placement_ok(placement):
            continue
        res = scorer.score(placement, carry.id)
        if best is None or res.log10 > best.score.log10:
            best = SearchResult(placement, res, 0)

    if best is None:
        raise Infeasible("CP-SAT found no feasible placement")
    if polish_iterations:
        polished = anneal(
            scorer, constraints, pool, carry, start=best.placement,
            iterations=polish_iterations, t_start=0.05, t_end=0.001,
        )
        if polished.score.log10 > best.score.log10:
            best = polished
    return best
