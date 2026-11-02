"""Optimise every adventure and variant, for each requested mode.

Adventures that end up with identical rules for this account (same layout,
same eligible champions, same required/blocked/forced/limit rules) are solved
once and the answer is shared: ~1,985 adventures collapse to ~800 groups.
Groups are solved in parallel worker processes (fork: the game data is
loaded once and shared).
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from ..engine.score import Scorer
from ..model.definitions import GameData, LayoutNotFound
from ..model.user import UserState
from ..model.variants import Constraints, constraints_for
from ..objectives import Calibration, FavorObjective, GemObjective, GemParams, ObjectiveScorer, PushObjective
from .planner import plan
from .search import Infeasible, candidate_pool

MODES = ("push", "favor", "gems")


def signature(constraints: Constraints, allowed: frozenset[int]) -> tuple:
    layout = tuple((s.col, s.y, tuple(sorted(s.adj))) for s in constraints.layout.slots)
    return (
        layout,
        allowed,
        frozenset(constraints.required_ids),
        frozenset(constraints.blocked_slots),
        tuple(sorted(constraints.max_by_tag.items())),
        tuple(sorted((k, r.description) for k, r in constraints.forced_slot_rules.items())),
    )


def make_objective(mode: str, data: GameData, user: UserState, calibration: Calibration | None,
                   gem_params: GemParams | None = None):
    if mode == "push":
        return PushObjective(data, calibration)
    if mode == "favor":
        return FavorObjective(data, user, calibration)
    if mode == "gems":
        return GemObjective(data, user, gem_params or GemParams(), calibration)
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class _Job:
    group: int
    adventure_id: int
    modes: tuple[str, ...]


# worker globals (set before fork)
_STATE: dict = {}


def _solve(job: _Job) -> list[dict]:
    """Solve one group in every mode, then cross-check: each mode's formation is
    also scored under the other objectives and the best one wins. (A favor
    search often finds a stronger damage formation than a short push search.)"""
    data: GameData = _STATE["data"]
    user: UserState = _STATE["user"]
    constraints = constraints_for(data, job.adventure_id)
    pool = candidate_pool(list(data.champions.values()), constraints, user.owned)
    scorer = Scorer(data, constraints.layout)
    if _STATE["use_account"]:
        scorer.apply_account(user)
    scorer.stack_fraction = _STATE.get("stack_fraction", 1.0)
    scorer.stack_overrides = dict(_STATE.get("stack_overrides") or {})
    objectives = {m: make_objective(m, data, user, _STATE["calibration"], _STATE["gem_params"]) for m in job.modes}
    oscorers = {m: ObjectiveScorer(scorer, o) for m, o in objectives.items()}

    account_specs = {h: dict(v) for h, v in scorer.specs.items()}

    def use_specs(specs):
        scorer.specs = {h: dict(v) for h, v in account_specs.items()}
        for h, v in specs.items():
            scorer.specs[h] = dict(v)
        scorer.invalidate()

    found = {}
    errors = {}
    for m in job.modes:
        use_specs({})
        try:
            p = plan(oscorers[m], constraints, pool, None, _STATE["profile"])
            found[m] = (p.carry, p.placement, p.specs, p.spec_changes)
        except Infeasible as exc:
            errors[m] = str(exc)

    rows = []
    for m in job.modes:
        row = {"group": job.group, "mode": m}
        if not found:
            rows.append({**row, "error": errors.get(m, "no feasible formation")})
            continue
        best = None
        for src_mode, (carry, placement, specs, changes) in found.items():
            use_specs(specs)
            res = oscorers[m].score(placement, carry.id)
            if best is None or res.log10 > best[0].log10:
                best = (res, carry, placement, src_mode, changes)
        res, carry, placement, src_mode, changes = best
        rows.append({
            **row,
            "carry": carry.name,
            "formation": "; ".join(f"{s}:{c.name}" for s, c in sorted(placement.items())),
            "value": round(res.log10, 3),
            "found_by": src_mode,
            "spec_changes": "; ".join(f"{c}: {a} -> {b}" for c, a, b in changes),
            "coverage": round(res.coverage, 3),
            "approx_share": round(res.approx_share, 3),
            "warnings": " | ".join(constraints.warnings),
            **objectives[m].summary(res, carry, placement),
        })
    return rows


def run_sweep(
    data: GameData,
    user: UserState,
    modes: Iterable[str],
    *,
    calibration: Calibration | None,
    out_csv: str,
    out_json: str | None = None,
    profile: str = "fast",
    workers: int = 4,
    adventure_ids: Iterable[int] | None = None,
    use_account: bool = True,
    gem_params: GemParams | None = None,
    stack_fraction: float = 1.0,
    stack_overrides: dict[int, float] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    champs = list(data.champions.values())
    groups: dict[tuple, list[int]] = {}
    skipped = []
    for aid in sorted(adventure_ids or data.adventures):
        try:
            c = constraints_for(data, aid)
        except LayoutNotFound:
            skipped.append(aid)
            continue
        allowed = frozenset(ch.id for ch in champs if ch.id in user.owned and c.allows(ch))
        groups.setdefault(signature(c, allowed), []).append(aid)
    group_list = list(groups.values())
    modes = tuple(modes)
    jobs = [_Job(gi, members[0], modes) for gi, members in enumerate(group_list)]

    _STATE.update(data=data, user=user, calibration=calibration, profile=profile,
                  use_account=use_account, gem_params=gem_params,
                  stack_fraction=stack_fraction, stack_overrides=stack_overrides)
    started = time.time()
    results: list[dict] = []
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_solve, jobs, chunksize=1)):
                results.extend(r)
                if progress:
                    progress(i + 1, len(jobs))
    else:
        for i, job in enumerate(jobs):
            results.extend(_solve(job))
            if progress:
                progress(i + 1, len(jobs))

    by_key = {(r["group"], r["mode"]): r for r in results}
    rows = []
    for gi, members in enumerate(group_list):
        for aid in members:
            adv = data.adventures[aid]
            for m in modes:
                r = by_key[(gi, m)]
                rows.append({
                    "adventure_id": aid,
                    "adventure": adv.name,
                    "campaign_id": adv.campaign_id,
                    "variant_of": adv.variant_of or "",
                    "restrictions": " ".join(adv.restrictions_text.split())[:300],
                    "group_size": len(members),
                    **{k: v for k, v in r.items() if k != "group"},
                })
    rows.sort(key=lambda r: (r["adventure_id"], MODES.index(r["mode"])))
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"calibration": None if calibration is None else vars(calibration),
                       "profile": profile, "rows": rows}, f)
    return {
        "adventures": sum(len(m) for m in group_list),
        "groups": len(group_list),
        "jobs": len(jobs),
        "skipped_no_layout": skipped,
        "seconds": round(time.time() - started, 1),
        "errors": sum(1 for r in results if "error" in r),
    }
