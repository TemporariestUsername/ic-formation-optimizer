"""Heuristic and exact search over placements (pure Python, no dependencies).

A *placement* is ``dict[slot_index, Champion]``. All search functions keep the
carry in the formation and only ever return placements that satisfy
``Constraints.placement_ok``.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from ..engine.score import Scorer, ScoreResult
from ..model.definitions import Champion
from ..model.variants import Constraints

Placement = dict[int, Champion]


class Infeasible(RuntimeError):
    pass


@dataclass
class SearchResult:
    placement: Placement
    score: ScoreResult
    evaluations: int


def candidate_pool(champions: list[Champion], constraints: Constraints, owned: set[int]) -> list[Champion]:
    """Owned, allowed champions, plus champions the variant forces in (the game
    provides those even if not owned) and owned key champions for forced slots
    (exempt from the allow rules)."""
    out = []
    for c in champions:
        if c.id in constraints.required_ids:
            out.append(c)
        elif c.id in owned and (constraints.allows(c) or any(r(c) for r in constraints.forced_slot_rules.values())):
            out.append(c)
    return out


def _seed(constraints: Constraints, pool: list[Champion], carry: Champion, carry_slot: int) -> Placement:
    """Carry + forced-slot + required champions, or raise Infeasible."""
    placement: Placement = {carry_slot: carry}
    used_seats = {carry.seat}
    by_id = {c.id: c for c in pool}
    for slot, rule in sorted(constraints.forced_slot_rules.items()):
        if slot in placement:
            if not rule(placement[slot]):
                raise Infeasible(f"carry cannot sit in forced slot {slot}")
            continue
        pick = next((c for c in pool if c.seat not in used_seats and rule(c)), None)
        if pick is None:
            raise Infeasible(f"no champion available for forced slot {slot} ({rule.description})")
        placement[slot] = pick
        used_seats.add(pick.seat)
    for rid in sorted(constraints.required_ids):
        if any(c.id == rid for c in placement.values()):
            continue
        champ = by_id.get(rid)
        if champ is None or champ.seat in used_seats:
            raise Infeasible(f"required champion {rid} unavailable or seat clash")
        free = [s for s in constraints.open_slots if s not in placement]
        if not free:
            raise Infeasible("no room for required champions")
        placement[free[0]] = champ
        used_seats.add(champ.seat)
    if not constraints.placement_ok(placement):
        raise Infeasible("seed placement violates constraints")
    return placement


def greedy(scorer: Scorer, constraints: Constraints, pool: list[Champion], carry: Champion) -> SearchResult:
    """Try the carry in every open slot; fill the rest by best marginal gain."""
    best: SearchResult | None = None
    evals = 0
    for carry_slot in constraints.open_slots:
        try:
            placement = _seed(constraints, pool, carry, carry_slot)
        except Infeasible:
            continue
        current = scorer.score(placement, carry.id)
        evals += 1
        while True:
            used_seats = {c.seat for c in placement.values()}
            free = [s for s in constraints.open_slots if s not in placement]
            step_best: tuple[float, int, Champion] | None = None
            for champ in pool:
                if champ.seat in used_seats:
                    continue
                for s in free:
                    trial = {**placement, s: champ}
                    if not constraints.placement_ok(trial):
                        continue
                    val = scorer.score(trial, carry.id).log10
                    evals += 1
                    if step_best is None or val > step_best[0]:
                        step_best = (val, s, champ)
            if step_best is None or step_best[0] < current.log10 - 1e-12:
                break
            placement[step_best[1]] = step_best[2]
            current = scorer.score(placement, carry.id)
        if best is None or current.log10 > best.score.log10:
            best = SearchResult(dict(placement), current, 0)
    if best is None:
        raise Infeasible("no feasible placement for this carry")
    best.evaluations = evals
    return best


def anneal(
    scorer: Scorer,
    constraints: Constraints,
    pool: list[Champion],
    carry: Champion,
    *,
    start: Placement | None = None,
    iterations: int = 20000,
    t_start: float = 0.5,
    t_end: float = 0.002,
    seed: int = 0,
) -> SearchResult:
    """Simulated annealing over swap / replace / remove moves (log10 units)."""
    rng = random.Random(seed)
    placement = dict(start) if start else greedy(scorer, constraints, pool, carry).placement
    locked = {carry.id} | constraints.required_ids
    current = scorer.score(placement, carry.id).log10
    best_p, best_v = dict(placement), current
    slots = constraints.open_slots
    evals = 0

    for i in range(iterations):
        t = t_start * (t_end / t_start) ** (i / max(1, iterations - 1))
        trial = dict(placement)
        move = rng.random()
        if move < 0.45:  # swap two slots (either may be empty)
            a, b = rng.sample(slots, 2) if len(slots) > 1 else (slots[0], slots[0])
            ca, cb = trial.pop(a, None), trial.pop(b, None)
            if ca:
                trial[b] = ca
            if cb:
                trial[a] = cb
        elif move < 0.9:  # put an unused champion in a slot (replacing whoever is there)
            s = rng.choice(slots)
            if s in trial and trial[s].id in locked:
                continue
            trial.pop(s, None)
            used = {c.seat for c in trial.values()}
            options = [c for c in pool if c.seat not in used]
            if not options:
                continue
            trial[s] = rng.choice(options)
        else:  # remove someone
            removable = [s for s, c in trial.items() if c.id not in locked]
            if not removable:
                continue
            trial.pop(rng.choice(removable))
        if not constraints.placement_ok(trial):
            continue
        val = scorer.score(trial, carry.id).log10
        evals += 1
        if val >= current or rng.random() < math.exp((val - current) / t):
            placement, current = trial, val
            if val > best_v:
                best_p, best_v = dict(trial), val
    return SearchResult(best_p, scorer.score(best_p, carry.id), evals)


def exhaustive(
    scorer: Scorer, constraints: Constraints, pool: list[Champion], carry: Champion, max_evals: int = 2_000_000
) -> SearchResult:
    """Provably optimal by brute force. Only for small pools/layouts (tests, sanity checks)."""
    slots = constraints.open_slots
    others = [c for c in pool if c.id != carry.id]
    best: SearchResult | None = None
    evals = 0
    for k in range(0, len(slots)):
        for group in itertools.combinations(others, k):
            if len({c.seat for c in group} | {carry.seat}) != k + 1:
                continue
            members = (carry, *group)
            for chosen_slots in itertools.permutations(slots, k + 1):
                placement = dict(zip(chosen_slots, members))
                if not constraints.placement_ok(placement):
                    continue
                res = scorer.score(placement, carry.id)
                evals += 1
                if evals > max_evals:
                    raise RuntimeError("search space too large for exhaustive()")
                if best is None or res.log10 > best.score.log10:
                    best = SearchResult(placement, res, 0)
    if best is None:
        raise Infeasible("no feasible placement")
    best.evaluations = evals
    return best
