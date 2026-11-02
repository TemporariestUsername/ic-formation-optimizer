"""What "best" means: push, favor and gem objectives.

All three build on the engine's log10 channels (damage reaching the carry,
gold find, speed effects) plus constants from the game's own rules:

* Monster health grows x2.031 per area (x3.031 from 2001, x4.531 from 2251),
  ``monster_base_stats.health_growth_rate_curve``. So +1 log10 of damage is
  worth ~3.25 areas below area 2001.
* Monster gold is a fixed share of monster health (``health_gold_ratio`` ~0.33
  past area 200), so gold grows at the same rate as health.
* Favor grows x``currency_growth_rate`` (1.25) for every x``gold_growth_rate``
  (2.031) of gold: favor ∝ gold^(ln 1.25 / ln 2.031) = gold^0.315.

**Push**   maximise the carry's damage: log10(base attack) + formation multiplier.
**Favor**  favor per run ∝ (gold find × gold at the area you stall at)^0.315,
           and the stall area is where damage meets monster health, so
           log10(favor) = 0.315 · (log10 damage + log10 gold find) + const.
**Gems**   maximise boss areas per hour × gems per boss, up to a reset area:
           Briv jumps, Thellora's start skip, spawn/quest/time-scale effects;
           +% boss gems (Ellywick's gem cards from her deck: hand size × gem
           odds or the "Gem" feat, each card's amount buffed by her gear/feats);
           and a damage floor: the formation must reach the reset area.

Absolute area estimates need one calibration point: an instance on the
account with a known formation and highest area. The model's damage for that
formation is anchored to that area's monster health. Because the instance may
not be at its wall, estimates are conservative (lower bounds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .engine.score import Scorer, ScoreResult
from .model.definitions import Champion, GameData
from .model.user import UserState

BRIV_ID = 58
THELLORA_ID = 139
ELLYWICK_ID = 83


# -- progression ---------------------------------------------------------------

class Progression:
    """log10 monster health by area, from the game's growth-rate curve."""

    def __init__(self, data: GameData):
        rules = {r.get("rule_name"): r.get("rule") for r in data.raw.get("game_rule_defines") or []}
        stats = rules.get("monster_base_stats") or {}
        curve = stats.get("health_growth_rate_curve") or {"1": stats.get("health_growth_rate", 2.031)}
        self.base = math.log10(float(stats.get("base_health", 10)))
        self.steps = sorted((int(a), math.log10(float(r))) for a, r in curve.items())
        self.max_area = int((rules.get("max_area") or {}).get("area", 2501))

    def log10_hp(self, area: float) -> float:
        total, prev_area, prev_slope = self.base, 1, self.steps[0][1]
        for start, slope in self.steps[1:]:
            if area <= start:
                break
            total += (start - prev_area) * prev_slope
            prev_area, prev_slope = start, slope
        return total + (area - prev_area) * prev_slope

    def area_for(self, log10_hp: float) -> float:
        lo, hi = 1.0, float(self.max_area)
        if log10_hp >= self.log10_hp(hi):
            return hi
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.log10_hp(mid) < log10_hp:
                lo = mid
            else:
                hi = mid
        return lo

    def slope(self, area: float) -> float:
        s = self.steps[0][1]
        for start, slope in self.steps:
            if area >= start:
                s = slope
        return s


def carry_base_log10(data: GameData, champ: Champion) -> float:
    """log10 of the carry's base attack per second: base_damage × modifier / cooldown.

    Levels and the champion's own upgrades are in the formation score; gear,
    legendaries and blessings are not (yet).
    """
    attack = data.attacks.get(champ.base_attack_id) or {}
    modifier = float(attack.get("damage_modifier") or 1.0)
    cooldown = float(attack.get("cooldown") or 1.0) or 1.0
    return math.log10(max(champ.base_damage, 1e-9) * max(modifier, 1e-9) / cooldown)


@dataclass
class Calibration:
    offset: float  # true log10 DPS ≈ model log10 DPS + offset
    adventure_id: int
    area: int
    carry: str
    note: str = "lower bound: anchored to the instance's highest area, which may be below its wall"


def calibrate(data: GameData, user: UserState, scorer_for) -> Calibration | None:
    """Anchor model damage to the account's furthest-progressed instance."""
    prog = Progression(data)
    best: Calibration | None = None
    for inst in user.instances:
        if inst.adventure_id <= 0 or inst.highest_area < 10 or inst.adventure_id not in data.adventures:
            continue
        try:
            layout = data.layout_for(inst.adventure_id)
        except KeyError:
            continue
        placement = {
            slot: data.champions[hid]
            for slot, hid in enumerate(inst.formation)
            if hid > 0 and hid in data.champions and slot < len(layout)
        }
        dps = [c for c in placement.values() if "dps" in c.tags] or list(placement.values())
        if not dps:
            continue
        scorer = scorer_for(layout, inst.adventure_id)  # with that campaign's blessings
        model, carry = max(
            ((carry_base_log10(data, c) + scorer.score(placement, c.id).log10, c) for c in dps),
            key=lambda t: t[0],
        )
        cal = Calibration(prog.log10_hp(inst.highest_area) - model, inst.adventure_id, inst.highest_area, carry.name)
        if best is None or cal.area > best.area:
            best = cal
    return best


# -- objectives ----------------------------------------------------------------

class Objective(Protocol):
    name: str

    def value(self, res: ScoreResult, carry: Champion, placement: dict[int, Champion]) -> float: ...

    def summary(self, res: ScoreResult, carry: Champion, placement: dict[int, Champion]) -> dict: ...


@dataclass
class PushObjective:
    data: GameData
    calibration: Calibration | None = None
    name: str = "push"

    def damage(self, res: ScoreResult, carry: Champion) -> float:
        dmg = res.damage_log10 if res.damage_log10 is not None else res.log10
        return carry_base_log10(self.data, carry) + dmg

    def est_area(self, res: ScoreResult, carry: Champion) -> float | None:
        if self.calibration is None:
            return None
        return Progression(self.data).area_for(self.damage(res, carry) + self.calibration.offset)

    def value(self, res, carry, placement):
        return self.damage(res, carry)

    def summary(self, res, carry, placement):
        area = self.est_area(res, carry)
        return {
            "log10_damage": round(self.damage(res, carry), 2),
            "est_area": None if area is None else int(area),
        }


@dataclass
class FavorObjective:
    data: GameData
    user: UserState
    calibration: Calibration | None = None
    name: str = "favor"

    @property
    def exponent(self) -> float:
        return math.log(self.user.currency_growth_rate) / math.log(self.user.gold_growth_rate)

    def value(self, res, carry, placement):
        push = PushObjective(self.data, self.calibration)
        return self.exponent * (push.damage(res, carry) + res.gold_log10)

    def summary(self, res, carry, placement):
        push = PushObjective(self.data, self.calibration)
        return {
            "log10_favor_rel": round(self.value(res, carry, placement), 2),
            "log10_gold_find": round(res.gold_log10, 2),
            "favor_exponent": round(self.exponent, 3),
            **push.summary(res, carry, placement),
        }


@dataclass
class GemParams:
    reset_area: int = 1000
    kills_per_area: float = 25.0
    spawn_interval_s: float = 1.0
    transition_s: float = 4.0
    reset_overhead_s: float = 30.0
    briv_stacks: float | None = None  # None: account's briv_sprint_stacks
    thellora_areas: float | None = None  # None: account stat
    time_scale_cap: float = 10.0
    transition_scale_cap: float = 5.0  # game rule area_transition_time_scale_cap
    skipped_bosses_pay: bool = True  # count boss areas Briv/Thellora skip (community "BPH")
    ellywick_hand: int = 5  # cards Ellywick holds


def briv_jump(item_multiplier: float) -> tuple[int, float]:
    """(areas advanced per jump, chance of the full jump) from Briv's slot-4 item.

    effect = 0.25 + 0.25·(100·m)/100, halved until <= 1: the halvings give the
    jump size, the remainder (0.5..1 mapped to 1%..100%) the chance of jumping
    the full amount instead of one area less. Same as ic.byteglow.com.
    """
    effect = 0.25 + 0.25 * item_multiplier
    effect = min(effect, 256.0)
    skips, n = 1, effect
    while n > 1:
        n /= 2
        skips += 1
    chance = (n - 0.5) * (0.99 / 0.5) + 0.01 if effect > 1 else effect
    return skips + 1, max(0.0, min(1.0, chance))


def briv_jumps_from_stacks(stacks: float, consume_pct: float) -> int:
    jumps = 0
    while stacks >= 50 and jumps < 100000:
        stacks = round(stacks - stacks * consume_pct / 100.0)
        jumps += 1
    return jumps


@dataclass
class GemObjective:
    data: GameData
    user: UserState
    params: GemParams = field(default_factory=GemParams)
    calibration: Calibration | None = None
    name: str = "gems"

    def model(self, res: ScoreResult, carry: Champion, placement: dict[int, Champion]) -> dict:
        p = self.params
        ids = {c.id for c in placement.values()}
        spawn_mult, quest_mult, credit_mult, time_scale, transition_scale = 1.0, 1.0, 1.0, 1.0, 1.0
        consume_pct = 4.0
        for s in res.speed:
            a = (s.amount or 0.0) * (s.stacks if s.stacks else 1.0)
            k = s.key
            if k == "increase_monster_spawn_time_mult":
                spawn_mult *= 1 + a / 100
            elif k in ("decrease_monster_spawn_time_mult",):
                spawn_mult *= max(0.05, 1 - a / 100)
            elif k == "decrease_monster_spawn_delay_mult":
                spawn_mult *= 1 + a / 100
            elif k in ("spawn_additional_monsters", "simultaneous_monster_spawn_chance_mult", "simultaneous_monster_spawn_chance_add"):
                spawn_mult *= 1 + a / 100
            elif k == "chance_reduce_quest_requirement":
                chance = min(1.0, _num(s.args, 1, 100.0) / 100)
                quest_mult *= max(0.05, 1 - chance * min(a, 100.0) / 100)
            elif k == "reduce_quest_requirements_free_play":
                quest_mult *= max(0.05, 1 - a / 100)
            elif k in ("chance_multiply_monster_quest_rewards", "chance_multiply_monster_quest_rewards_new"):
                chance = min(1.0, a / 100)  # amount is the chance; stacks raise it, capped at 100%
                mult = _num(s.args, 1, 2.0)
                credit_mult *= 1 + chance * (mult - 1)
            elif k == "time_scale":
                time_scale *= max(1.0, s.amount or 1.0)
            elif k == "time_scale_when_not_attacked":
                time_scale *= 1 + a / 100
            elif k == "area_transition_time_scale":
                transition_scale *= max(1.0, s.amount or 1.0)
            elif k == "briv_unnatural_haste_stack_consume_percent_override":
                consume_pct = s.amount or consume_pct
        time_scale = min(time_scale, p.time_scale_cap)
        t_trans = p.transition_s / min(time_scale * transition_scale, p.transition_scale_cap)
        kills = p.kills_per_area * quest_mult / credit_mult
        t_area = kills * p.spawn_interval_s / spawn_mult / time_scale + t_trans

        start = 1.0
        if THELLORA_ID in ids:
            start += p.thellora_areas if p.thellora_areas is not None else self.user.stat("thellora_plateaus_of_unicorn_run_areas")
        remaining = max(0.0, p.reset_area - start)
        jump, chance, jumps = 1, 0.0, 0
        if BRIV_ID in ids:
            item = next((g for g in self.user.gear.get(BRIV_ID, []) if g.slot_id == 4), None)
            if item is not None:
                jump, chance = briv_jump(item.multiplier * {1: 10, 2: 30, 3: 50, 4: 100}.get(item.rarity, 0) / 100)
                stacks = p.briv_stacks if p.briv_stacks is not None else self.user.stat("briv_sprint_stacks")
                jumps = briv_jumps_from_stacks(stacks, consume_pct)
        adv = chance * jump + (1 - chance) * (jump - 1) if jump > 1 else 1.0
        if jumps and jumps * adv >= remaining:
            completed = remaining / adv
        else:
            completed = jumps + max(0.0, remaining - jumps * adv)
        run_s = completed * t_area + p.reset_overhead_s
        areas_per_hour = (p.reset_area - 1) / run_s * 3600 if run_s > 0 else 0.0
        bosses = (p.reset_area - 1) / 5 if p.skipped_bosses_pay else completed / 5
        boss_per_hour = bosses / run_s * 3600 if run_s > 0 else 0.0

        # gems per boss: +% sources add; Ellywick's gem cards come from her deck
        bonus, gem_cards, per_card = 0.0, 0.0, 0.0
        deck = next((s for s in res.speed if s.key == "ellywick_deck_of_many_things"), None)
        for s in res.speed:
            if s.key != "increase_boss_gems_percent" or s.amount is None:
                continue
            if s.hero_id == ELLYWICK_ID and deck is not None:
                per_card = s.amount
                draws = completed * kills / max(1.0, float(deck.raw.get("monster_kills_for_card_draw", 25)))
                draws = min(draws, run_s / max(1.0, float(deck.raw.get("card_draw_cooldown", 60))))
                prioritized = any(x.key == "ellywick_gem_prioritized" for x in res.speed)
                share = 1.0 if prioritized else float(deck.raw.get("gem_odds", 20)) / 100
                gem_cards = min(p.ellywick_hand, draws) * share
                bonus += gem_cards * per_card
            elif not s.raw.get("manual_stacking"):
                bonus += s.amount * (s.stacks or 1.0)
        gem_mult = 1 + bonus / 100

        push = PushObjective(self.data, self.calibration)
        reach = push.est_area(res, carry)
        return {
            "rel_gems_per_hour": round(boss_per_hour * gem_mult, 1),
            "gem_multiplier": round(gem_mult, 3),
            "ellywick_gem_cards": round(gem_cards, 2) if deck is not None else "-",
            "areas_per_hour": round(areas_per_hour, 1),
            "boss_areas_per_hour": round(boss_per_hour, 1),
            "run_minutes": round(run_s / 60, 2),
            "start_area": round(start, 1),
            "briv_jump": f"{jump} ({chance:.0%})" if jump > 1 else "-",
            "briv_jumps": jumps,
            "sec_per_area": round(t_area, 2),
            "est_reach_area": None if reach is None else int(reach),
            "reaches_reset": None if reach is None else reach >= p.reset_area,
        }

    def value(self, res, carry, placement):
        m = self.model(res, carry, placement)
        v = math.log10(max(m["rel_gems_per_hour"], 1e-6))
        reach = m["est_reach_area"]
        if reach is not None and reach < self.params.reset_area:
            v -= 0.05 * (self.params.reset_area - reach)  # can't reach the reset area: steep penalty
        return v

    def summary(self, res, carry, placement):
        return self.model(res, carry, placement)


def _num(args, i, default):
    try:
        return float(args[i])
    except (IndexError, ValueError):
        return default


class ObjectiveScorer:
    """Scorer facade: ``score().log10`` is the objective value; the raw damage
    multiplier stays in ``damage_log10``. Search code works unchanged."""

    def __init__(self, scorer: Scorer, objective: Objective):
        self.scorer = scorer
        self.objective = objective
        self.data = scorer.data
        self.layout = scorer.layout

    def score(self, placement: dict[int, Champion], carry_id: int) -> ScoreResult:
        res = self.scorer.score(placement, carry_id)
        if res.log10 == float("-inf"):
            return res
        carry = next(c for c in placement.values() if c.id == carry_id)
        res.damage_log10 = res.log10
        res.log10 = self.objective.value(res, carry, placement)
        return res

    def __getattr__(self, name):
        return getattr(self.scorer, name)
