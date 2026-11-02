"""Track real progress and check the model against it.

``icfo track`` reads the account (read-only) and records, for every running
instance, the adventure, patron, formation and highest area reached. Each
record is an *observation*:

* By default an observation is a **lower bound**: the run may not have hit
  its wall yet.
* With ``--stalled`` the current runs are marked as **at their wall**, which
  makes them exact calibration points.

For each observation the model's damage for that exact formation (best carry
in it) gives an offset ``log10 HP(observed area) − model damage``. If the
model ranked formations correctly, every wall observation would need the same
offset; the spread between them measures how wrong it is, in areas. Refitting
with different shares of assumed max stacks shows which assumption reality
agrees with.

Observations are stored in the local cache (account data: never committed).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

from .fetch.cache import Cache
from .model.definitions import GameData
from .model.user import UserState
from .objectives import Progression, carry_base_log10

STORE = "observations"


@dataclass
class Observation:
    recorded_at: float
    instance_id: int
    adventure_id: int
    patron_id: int | None
    formation: tuple[int, ...]
    highest_area: int
    stalled: bool = False

    @property
    def key(self) -> tuple:
        return (self.instance_id, self.adventure_id, self.patron_id, self.formation)


def load(cache: Cache) -> list[Observation]:
    raw = cache.load(STORE) or []
    return [Observation(**{**o, "formation": tuple(o["formation"])}) for o in raw]


def save(cache: Cache, observations: list[Observation]) -> None:
    cache.save(STORE, [asdict(o) for o in observations])


def record(cache: Cache, user: UserState, stalled: bool = False, min_area: int = 10) -> tuple[list[Observation], list[Observation]]:
    """Add the account's current instances; returns (all, added_or_updated)."""
    obs = load(cache)
    by_key = {o.key: o for o in obs}
    changed = []
    now = time.time()
    for inst in user.instances:
        if inst.adventure_id <= 0 or inst.highest_area < min_area:
            continue
        o = Observation(now, inst.instance_id, inst.adventure_id, inst.patron_id,
                        tuple(inst.formation), inst.highest_area, stalled)
        old = by_key.get(o.key)
        if old is None:
            obs.append(o)
            by_key[o.key] = o
            changed.append(o)
        elif o.highest_area > old.highest_area or (stalled and not old.stalled):
            old.highest_area = max(old.highest_area, o.highest_area)
            old.stalled = old.stalled or stalled
            old.recorded_at = now
            changed.append(old)
    save(cache, obs)
    return obs, changed


@dataclass
class Evaluated:
    obs: Observation
    carry: str
    model: float  # model log10 damage (base attack + formation)
    offset: float  # log10 HP(observed) - model


def evaluate(data: GameData, observations: list[Observation], scorer_for, stack_fraction: float | None = None) -> list[Evaluated]:
    prog = Progression(data)
    out = []
    for o in observations:
        if o.adventure_id not in data.adventures:
            continue
        try:
            layout = data.layout_for(o.adventure_id)
        except KeyError:
            continue
        placement = {sl: data.champions[h] for sl, h in enumerate(o.formation)
                     if h > 0 and h in data.champions and sl < len(layout)}
        dps = [c for c in placement.values() if "dps" in c.tags] or list(placement.values())
        if not dps:
            continue
        scorer = scorer_for(layout, o.adventure_id, o.patron_id)
        if stack_fraction is not None:
            scorer.stack_fraction = stack_fraction
        model, carry = max(((carry_base_log10(data, c) + scorer.score(placement, c.id).log10, c) for c in dps),
                           key=lambda t: t[0])
        out.append(Evaluated(o, carry.name, model, prog.log10_hp(o.highest_area) - model))
    return out


def offset_from(evaluated: list[Evaluated]) -> tuple[float | None, str]:
    """Calibration offset: median of wall observations if any; otherwise the
    largest lower bound (the tightest one)."""
    walls = [e.offset for e in evaluated if e.obs.stalled]
    if walls:
        return statistics.median(walls), f"median of {len(walls)} wall observation(s)"
    if evaluated:
        return max(e.offset for e in evaluated), f"tightest of {len(evaluated)} lower-bound observation(s)"
    return None, "no observations"


def consistency(evaluated: list[Evaluated], slope: float) -> float | None:
    """Spread (stdev) of wall offsets, in areas. Small = the model ranks formations well."""
    walls = [e.offset for e in evaluated if e.obs.stalled]
    if len(walls) < 2:
        return None
    return statistics.pstdev(walls) / slope


def fit_stack_fraction(data: GameData, observations: list[Observation], scorer_for,
                       fractions=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[tuple[float, float | None]]:
    slope = Progression(data).slope(1000)
    return [(f, consistency(evaluate(data, observations, scorer_for, f), slope)) for f in fractions]
