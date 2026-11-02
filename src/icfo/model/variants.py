"""Turn adventure/variant ``game_changes`` into formation constraints.

Champion eligibility is a list of *rules*, one per ``only_allow_crusaders`` /
``disallow_crusaders`` entry. Criteria *inside* one entry are alternatives
(OR): e.g. "Odd Ones Out" allows ``by_seat`` even seats OR ``by_expr`` Farideh /
Mehen. Separate entries stack (AND). Champions forced in by
``force_use_heroes`` bypass the allow rules, as in game.

Criteria handled: ``by_ids``, ``by_tags``, ``by_seat``, ``by_stat``,
``by_expr``. Also: ``formation`` (layout override), ``force_use_heroes``,
``slot_escort`` / ``slot_escort_by_area`` (blocked slots), ``max_active_by_tags``,
``key_champion``. Everything else that could affect formation choice goes in
``warnings`` so the user knows the result ignores it.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable

from .definitions import Champion, GameData
from .effects import match_tags
from .expr import UnsupportedExpr, evaluate
from .formation import Layout

# game_change types that never matter for choosing a formation
COSMETIC_OR_IRRELEVANT = {
    "formation_saves_campaign_id",
    "initial_formation",
    "override_quest_graphic",
    "has_save_data",
    "no_abandon",
    "force_reset",
    "skip_cinematics",
    "buff_display_by_area",
    "weather",
    "weather_by_weather",
    "darken_by_area",
    "force_allow_hero",
}

# keys inside only_allow/disallow entries that are presentation, not criteria
NON_CRITERIA = {"type", "blocked_portrait", "dragging_graphic_id", "can_swap_blocked", "greyscale_blocked", "can_field_crusaders"}

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


@dataclass(frozen=True)
class Rule:
    description: str
    test: Callable[[Champion], bool]
    slots: frozenset[int] | None = None  # only applies to these slots ("target_slots")

    def __call__(self, champ: Champion) -> bool:
        return self.test(champ)


@dataclass
class Constraints:
    layout: Layout
    allow_rules: list[Rule] = field(default_factory=list)
    ban_rules: list[Rule] = field(default_factory=list)
    required_ids: set[int] = field(default_factory=set)
    blocked_slots: set[int] = field(default_factory=set)
    max_by_tag: dict[str, int] = field(default_factory=dict)
    forced_slot_rules: dict[int, Rule] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def open_slots(self) -> list[int]:
        return [i for i in range(len(self.layout)) if i not in self.blocked_slots]

    def allows(self, champ: Champion) -> bool:
        if champ.id in self.required_ids:
            return True
        return (all(r(champ) for r in self.allow_rules if r.slots is None)
                and not any(r(champ) for r in self.ban_rules if r.slots is None))

    def allows_in(self, champ: Champion, slot: int) -> bool:
        """Slot-scoped rules ("champions in the top of each column must be...")."""
        if champ.id in self.required_ids:
            return True
        for r in self.allow_rules:
            if r.slots is not None and slot in r.slots and not r(champ):
                return False
        for r in self.ban_rules:
            if r.slots is not None and slot in r.slots and r(champ):
                return False
        return True

    def placement_ok(self, placement: dict[int, Champion]) -> bool:
        """Check a full ``slot -> champion`` placement against every constraint."""
        seats: set[int] = set()
        for slot, champ in placement.items():
            if slot in self.blocked_slots:
                return False
            # a key champion in its forced slot is exempt from the allow rules (as in game)
            forced = slot in self.forced_slot_rules and self.forced_slot_rules[slot](champ)
            if not forced and not (self.allows(champ) and self.allows_in(champ, slot)):
                return False
            if champ.seat in seats:
                return False
            seats.add(champ.seat)
        ids = {c.id for c in placement.values()}
        if not self.required_ids <= ids:
            return False
        for tag, limit in self.max_by_tag.items():
            if sum(1 for c in placement.values() if match_tags(tag, c.tags)) > limit:
                return False
        for slot, rule in self.forced_slot_rules.items():
            champ = placement.get(slot)
            if champ is None or not rule(champ):
                return False
        return True


def _ids(value: Any) -> set[int]:
    raw = value.get("ids", []) if isinstance(value, dict) else value
    return {int(i) for i in raw or []}


def _criterion(kind: str, value: Any) -> Rule | None:
    """Build a rule for one criterion, or None if it isn't modelled."""
    if kind == "by_ids":
        ids = _ids(value)
        return Rule(f"id in {sorted(ids)}", lambda c: c.id in ids)
    if kind == "by_tags" and isinstance(value, dict) and value.get("tags"):
        expr = str(value["tags"])
        return Rule(f"tags {expr}", lambda c: match_tags(expr, c.tags))
    if kind == "by_seat" and isinstance(value, dict):
        seats = {int(s) for s in value.get("seats", [])}
        return Rule(f"seat in {sorted(seats)}", lambda c: c.seat in seats)
    if kind == "by_stat" and isinstance(value, dict):
        checks = []
        for s in value.get("stats", []):
            op = _OPS.get(str(s.get("comp")))
            if op is None:
                return None
            checks.append((str(s.get("stat")).lower(), op, float(s.get("value", 0))))
        return Rule(
            f"stats {checks}",
            lambda c: all(op(c.stats.get(stat, 0), v) for stat, op, v in checks),
        )
    if kind == "by_expr":
        expr = value.get("expr") if isinstance(value, dict) else value
        if not isinstance(expr, str):
            return None
        return Rule(f"expr {expr}", lambda c, e=expr: evaluate(e, c))
    return None


def _entry_rule(gc: dict, kind: str, c: Constraints, champs: list[Champion]) -> Rule | None:
    """One OR-combined rule for all modelled criteria of an entry."""
    rules = _probe(_rules_for(gc, kind, c), champs, c)
    if not rules:
        return None
    slots: set[int] = set()
    for key, value in gc.items():
        ts = value.get("target_slots") if isinstance(value, dict) else None
        if isinstance(ts, dict) and ts.get("type") == "slots":
            slots |= {int(x) for x in ts.get("data") or []}
    scope = frozenset(slots) if slots else None
    if len(rules) == 1:
        r = rules[0]
        return Rule(r.description + (f" (slots {sorted(scope)})" if scope else ""), r.test, scope)
    return Rule(" OR ".join(r.description for r in rules) + (f" (slots {sorted(scope)})" if scope else ""),
                lambda ch: any(r(ch) for r in rules), scope)


def _rules_for(gc: dict, kind: str, c: Constraints) -> list[Rule]:
    rules = []
    for key, value in gc.items():
        if key in NON_CRITERIA or key == "update_upgrades_ids":
            continue
        rule = _criterion(key, value)
        if rule is None:
            c.warnings.append(f"{kind}.{key} not modelled")
        else:
            rules.append(rule)
    return rules


def _probe(rules: list[Rule], champs: list[Champion], c: Constraints) -> list[Rule]:
    """Drop expression rules that use syntax we can't evaluate (with a warning)."""
    ok = []
    for r in rules:
        try:
            for champ in champs:
                r(champ)
            ok.append(r)
        except UnsupportedExpr as exc:
            c.warnings.append(f"{r.description}: {exc}; restriction ignored")
    return ok


def constraints_for(data: GameData, adventure_id: int) -> Constraints:
    adv = data.adventures[adventure_id]
    c = Constraints(layout=data.layout_for(adventure_id))
    champs = list(data.champions.values())
    for gc in adv.game_changes:
        kind = gc.get("type")
        if kind == "formation" or kind in COSMETIC_OR_IRRELEVANT:
            continue
        if kind in ("only_allow_crusaders", "disallow_crusaders"):
            rule = _entry_rule(gc, kind, c, champs)
            if rule is not None:
                (c.allow_rules if kind == "only_allow_crusaders" else c.ban_rules).append(rule)
        elif kind == "force_use_heroes":
            for hid in (int(i) for i in gc.get("hero_ids", [])):
                if hid in data.champions:
                    c.required_ids.add(hid)
                else:
                    c.warnings.append(f"forced hero {hid} is not a playable champion; ignored")
            if gc.get("forced_slots"):
                c.warnings.append("force_use_heroes.forced_slots (moving required champion) not modelled")
        elif kind == "slot_escort":
            slots = gc.get("slot_ids")
            if slots is None and "slot_id" in gc:
                slots = [gc["slot_id"]]
            c.blocked_slots |= {int(s) for s in slots or []}
        elif kind == "slot_escort_by_area":
            escorts = sorted(gc.get("escorts", []), key=lambda e: e.get("min_area", 0))
            target = adv.objective_area or 250
            active = [e for e in escorts if e.get("min_area", 0) <= target] or escorts[:1]
            if active:
                c.blocked_slots |= {int(s) for s in active[-1].get("slot_ids", [])}
                c.warnings.append(f"slot_escort_by_area: using the stage active at area {target} (the objective)")
        elif kind == "max_active_by_tags":
            c.max_by_tag.update({str(k): int(v) for k, v in (gc.get("tag_maxes") or {}).items()})
            if gc.get("excluded_heroes"):
                c.warnings.append("max_active_by_tags.excluded_heroes not modelled")
        elif kind == "key_champion":
            rule = _criterion("by_tags", gc["by_tags"]) if "by_tags" in gc else (
                _criterion("by_seat", gc["by_seat"]) if "by_seat" in gc else None
            )
            if gc.get("force_slot") and rule is not None:
                c.forced_slot_rules[int(gc.get("slot_id", 0))] = rule
            else:
                c.warnings.append("key_champion without a forced slot not modelled")
        else:
            c.warnings.append(f"{kind} not modelled")
    c.blocked_slots &= set(range(len(c.layout)))
    return c
