"""Score a placement by the damage multiplier reaching one champion (the "carry").

Idle Champions damage is (mostly) a product of independent multipliers, so we
work in log10 space: the score is ``sum(log10(multiplier))`` over every buff
that reaches the carry. A score of 12.3 means "x10^12.3 from the formation".

The carry's base damage is a constant for a fixed carry and is left out, so
scores compare formations for the *same* carry. Comparing different carries
needs a base-damage model (roadmap).

Every effect on a placed champion is classified as:

* modelled      -> contributes to the score
* non-damage    -> known not to affect the carry's damage (healing, gold, ...)
* unmodelled    -> could matter but isn't implemented; reported in coverage
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..model.definitions import Champion, GameData
from ..model.effects import Effect, match_tags
from ..model.expr import UnsupportedExpr, evaluate
from ..model.formation import Layout

AMPLIFIER_KEYS = {"buff_upgrade", "buff_upgrades", "buff_upgrade_per_any_tagged_crusader_mult"}

NON_DAMAGE_KEYS = {
    "health_add",
    "health_mult",
    "heal",
    "increase_health_by_source_percent",
    "gold_multiplier_mult",
    "do_nothing",
    "set_ultimate_attack",
    "change_base_attack",
    "add_monster_hit_effects",
    "damage_reduction",
    "add_attack_targets",
    "favored_foe",
    "add_hero_tags",
}


@dataclass(frozen=True)
class Contribution:
    source: str
    key: str
    upgrade_id: int | None
    log10: float


@dataclass
class ScoreResult:
    log10: float
    contributions: list[Contribution] = field(default_factory=list)
    unmodelled: Counter = field(default_factory=Counter)
    unmodelled_by_hero: dict[str, Counter] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of potentially-relevant effects on placed champions we model."""
        modelled = len(self.contributions)
        missing = sum(self.unmodelled.values())
        return 1.0 if modelled + missing == 0 else modelled / (modelled + missing)


def _log_mult(pct: float) -> float:
    """log10 of a "+pct%" multiplier, guarded against nonsense values."""
    factor = 1.0 + pct / 100.0
    return math.log10(factor) if factor > 0 else 0.0


class Scorer:
    def __init__(self, data: GameData, layout: Layout):
        self.data = data
        self.layout = layout
        self._effects: dict[int, list[Effect]] = {}
        self.levels: dict[int, int | None] = {}
        self.specs: dict[int, dict[int, int]] = {}
        self.feats: dict[int, tuple[int, ...]] = {}

    def effects_for(self, champ: Champion) -> list[Effect]:
        if champ.id not in self._effects:
            self._effects[champ.id] = champ.effects(
                level=self.levels.get(champ.id),
                specs=self.specs.get(champ.id),
                feat_ids=self.feats.get(champ.id, ()),
            )
        return self._effects[champ.id]

    def invalidate(self, hero_id: int | None = None) -> None:
        if hero_id is None:
            self._effects.clear()
        else:
            self._effects.pop(hero_id, None)

    # ------------------------------------------------------------------
    def score(self, placement: dict[int, Champion], carry_id: int) -> ScoreResult:
        slot_of = {c.id: s for s, c in placement.items()}
        if carry_id not in slot_of:
            return ScoreResult(log10=float("-inf"))
        carry_slot = slot_of[carry_id]
        carry = placement[carry_slot]
        present = list(placement.values())
        occupied = frozenset(placement)

        amp = self._amplifiers(present)
        result = ScoreResult(log10=0.0)

        for src_slot, src in placement.items():
            for e in self.effects_for(src):
                if e.key in AMPLIFIER_KEYS or e.key in NON_DAMAGE_KEYS:
                    continue
                contrib = self._effect_on_carry(e, src, src_slot, carry, carry_slot, present, occupied)
                if contrib is None:
                    result.unmodelled[e.key] += 1
                    result.unmodelled_by_hero.setdefault(src.name, Counter())[e.key] += 1
                    continue
                if contrib == 0.0:
                    continue
                pct_scale = amp.get(e.upgrade_id, 1.0) if e.upgrade_id is not None else 1.0
                value = contrib(pct_scale)
                if value:
                    result.log10 += value
                    result.contributions.append(Contribution(src.name, e.key, e.upgrade_id, value))
        return result

    def _amplifiers(self, present: list[Champion]) -> dict[int, float]:
        """Per-upgrade amplification factor from buff_upgrade-style effects."""
        add: dict[int, float] = {}
        mult: dict[int, float] = {}
        for src in present:
            for e in self.effects_for(src):
                if e.key == "buff_upgrade":
                    uid = int(e.num(1))
                    add[uid] = add.get(uid, 0.0) + e.num(0) / 100.0
                elif e.key == "buff_upgrades":
                    for raw in e.args[1:]:
                        if raw.strip().isdigit():
                            uid = int(raw)
                            add[uid] = add.get(uid, 0.0) + e.num(0) / 100.0
                elif e.key == "buff_upgrade_per_any_tagged_crusader_mult":
                    uid, expr = int(e.num(1)), (e.args[2] if len(e.args) > 2 else "")
                    n = sum(
                        1
                        for c in present
                        if match_tags(expr, c.tags) and not (e.exclude_self and c.id == src.id)
                    )
                    mult[uid] = mult.get(uid, 1.0) * (1.0 + e.num(0) / 100.0) ** n
        return {uid: (1.0 + add.get(uid, 0.0)) * mult.get(uid, 1.0) for uid in set(add) | set(mult)}

    def _targets_carry(self, e: Effect, src: Champion, src_slot: int, carry: Champion, carry_slot: int) -> bool | None:
        hit = False
        for rel in e.targets:
            slots = self.layout.targets(rel, src_slot)
            if slots is None:
                return None
            if isinstance(rel, dict):
                if rel.get("type") == "by_tags" and not match_tags(rel.get("tags", ""), carry.tags):
                    continue
                if rel.get("type") == "attack_type":
                    return None
            if carry_slot in slots:
                hit = True
        if e.exclude_self and carry.id == src.id:
            return False
        for flt in e.filters:
            kind = flt.get("type") if isinstance(flt, dict) else None
            if kind == "by_tags":
                if not match_tags(flt.get("tags", ""), carry.tags):
                    return False
            elif kind == "hero_expr":
                try:
                    if not evaluate(str(flt.get("hero_expr", "")), carry):
                        return False
                except UnsupportedExpr:
                    return None
            else:
                return None  # stat_score and other filters: not modelled yet
        return hit

    def _effect_on_carry(self, e, src, src_slot, carry, carry_slot, present, occupied):
        """Return a callable(pct_scale) -> log10 contribution, 0.0 if no effect,
        or None if unmodelled."""
        key = e.key
        if key == "global_dps_multiplier_mult":
            pct = e.num(0)
            return lambda s: _log_mult(pct * s)

        if key not in (
            "hero_dps_multiplier_mult",
            "hero_dps_mult_per_tagged_crusader_mult",
            "hero_dps_mult_per_target_crusader",
        ):
            return None

        targeted = self._targets_carry(e, src, src_slot, carry, carry_slot)
        if targeted is None:
            return None
        if not targeted:
            return 0.0

        pct = e.num(0)
        if key == "hero_dps_multiplier_mult":
            return lambda s: _log_mult(pct * s)

        if key == "hero_dps_mult_per_tagged_crusader_mult":
            expr = e.args[1] if len(e.args) > 1 else ""
            n = sum(1 for c in present if match_tags(expr, c.tags) and not (e.exclude_self and c.id == src.id))
        else:  # per_target_crusader: count occupied slots in a relation from the source
            rel = e.args[1] if len(e.args) > 1 else "adj"
            slots = self.layout.targets(rel, src_slot)
            if slots is None:
                return None
            n = len((slots - {src_slot}) & occupied)
        if n == 0:
            return 0.0
        if e.stacks_multiply:
            return lambda s: n * _log_mult(pct * s)
        return lambda s: _log_mult(n * pct * s)
