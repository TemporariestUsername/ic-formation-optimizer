"""Effect parsing.

Game effects are strings like ``hero_dps_multiplier_mult,100`` wrapped in
loosely-typed objects. Shapes observed in live data:

* a bare string:            ``"hero_dps_multiplier_mult,100"``
* a JSON-encoded string:    ``'{"effect_string":"buff_upgrade,50,12"}'``
* an indirection:           ``"effect_def,310"`` -> ``effect_defines[310].effect_keys``
* several indirections:     ``"effect_def,234,235,236,237"``
* ``effect_keys`` as a list of objects, a single object, or plain strings
* the string under ``effect_string`` or (rarely) ``effect``
* ``targets`` as a list, a string, an int, or missing (= the source itself)

Amounts are not always in the string: ``pre_stack_amount,60`` next to
``hero_dps_multiplier_mult,0`` + ``"amount_expr": "upgrade_amount(12982,0)"``
means "the real amount is effect #0 of upgrade 12982". Those formulas are
resolved by :class:`~icfo.model.definitions.GameData` after loading, and the
result is stored on the effect (``resolved_amount``).

``normalize`` flattens all of that into a list of :class:`Effect`. Each effect
keeps its full raw key, so upgrade modifiers (``change_upgrade_data``) can
patch any field and the derived properties follow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

MAX_EFFECT_DEF_DEPTH = 5


@dataclass(frozen=True)
class Effect:
    key: str
    args: tuple[str, ...]
    raw: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    hero_id: int | None = None
    upgrade_id: int | None = None
    feat_id: int | None = None
    index: int = 0  # position among the upgrade's/feat's effect keys
    resolved_amount: float | None = None  # from amount_expr, when present
    # (upgrade id, effect index) parameters a pure-product amount formula reads;
    # buffs aimed at those parameters scale this effect (pre-stack)
    amount_refs: tuple[tuple[int, int], ...] = ()
    amount_cap: float | None = None  # effect_cap along the formula chain

    # -- arguments ---------------------------------------------------------
    def num(self, i: int, default: float = 0.0) -> float:
        try:
            return float(self.args[i])
        except (IndexError, ValueError):
            return default

    @property
    def base_amount(self) -> float | None:
        """The effect's numeric amount, or None if it can't be determined."""
        if "amount_expr" in self.raw:
            return self.resolved_amount
        if not self.args:
            return 0.0
        try:
            return float(self.args[0])
        except ValueError:
            return None  # e.g. "$replace": supplied by a system we don't model

    # -- targeting ---------------------------------------------------------
    @property
    def targets(self) -> tuple[Any, ...]:
        return tuple(_as_list(self.raw.get("targets"))) or ("self",)

    @property
    def filters(self) -> tuple[Any, ...]:
        return tuple(_as_list(self.raw.get("filter_targets")))

    @property
    def exclude_self(self) -> bool:
        return bool(self.raw.get("exclude_self"))

    # -- stacking ----------------------------------------------------------
    @property
    def stacks_multiply(self) -> bool:
        return bool(self.raw.get("stacks_multiply"))

    @property
    def stack_func(self) -> str | None:
        f = self.raw.get("stack_func")
        return str(f) if f else None

    @property
    def amount_func(self) -> str | None:
        f = self.raw.get("amount_func")
        return str(f) if f else None

    @property
    def max_stacks(self) -> float | None:
        v = self.raw.get("max_stacks")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def triggered(self) -> bool:
        """Stacks or activates at runtime (kills, attacks, timers, manual logic)."""
        return "stacks_on_trigger" in self.raw or bool(self.raw.get("manual_stacking"))

    @property
    def manually_stacked(self) -> bool:
        """Stacks driven by champion-specific code, not a generic trigger."""
        trig = self.raw.get("stacks_on_trigger")
        return bool(self.raw.get("manual_stacking")) or trig in ("will_stack_manually", "will_manually_stack")

    @property
    def timed(self) -> bool:
        return "for_time" in self.raw

    # -- modification ------------------------------------------------------
    def patched(self, data: dict) -> "Effect":
        """A copy with raw fields overridden (``change_upgrade_data``)."""
        raw = {**self.raw, **data}
        args = self.args
        if "amount" in data and isinstance(data["amount"], (int, float)):
            args = (str(data["amount"]), *self.args[1:])
            raw.pop("amount_expr", None)
        return replace(self, raw=raw, args=args)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "" or value is False:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_entry(entry: Any) -> dict | None:
    """Turn one raw effect entry into a dict with an ``effect_string``."""
    if isinstance(entry, str):
        s = entry.strip()
        if s.startswith("{"):
            try:
                entry = json.loads(s)
            except json.JSONDecodeError:
                return None
        else:
            return {"effect_string": s} if s else None
    if not isinstance(entry, dict):
        return None
    if "effect_string" not in entry and isinstance(entry.get("effect"), str):
        entry = {**entry, "effect_string": entry["effect"]}
    if not isinstance(entry.get("effect_string"), str):
        return None
    return entry


def normalize(
    value: Any,
    effect_defines: dict[int, dict],
    *,
    hero_id: int | None = None,
    upgrade_id: int | None = None,
    feat_id: int | None = None,
) -> list[Effect]:
    out: list[Effect] = []
    _normalize_into(out, value, effect_defines, hero_id, upgrade_id, feat_id, 0, None)
    return [replace(e, index=i) for i, e in enumerate(out)]


def _normalize_into(out, value, effect_defines, hero_id, upgrade_id, feat_id, depth, wrapper_targets):
    for raw in _as_list(value):
        entry = _coerce_entry(raw)
        if entry is None:
            continue
        key, *args = [p.strip() for p in entry["effect_string"].split(",")]
        if key == "effect_def":
            if depth >= MAX_EFFECT_DEF_DEPTH:
                continue
            targets = entry.get("targets", wrapper_targets)
            for ref in args:  # "effect_def,234,235,236" -> all four
                try:
                    target_def = effect_defines.get(int(ref))
                except ValueError:
                    target_def = None
                if target_def is not None:
                    _normalize_into(
                        out, target_def.get("effect_keys"), effect_defines,
                        hero_id, upgrade_id, feat_id, depth + 1, targets,
                    )
            continue
        if "targets" not in entry and wrapper_targets is not None:
            entry = {**entry, "targets": wrapper_targets}
        out.append(Effect(key=key, args=tuple(args), raw=entry, hero_id=hero_id, upgrade_id=upgrade_id, feat_id=feat_id))


def match_tags(expr: str, tags: Iterable[str]) -> bool:
    """Evaluate a tag expression such as ``dwarf``, ``!human`` or ``druid|ranger``.

    ``|`` is OR; ``&`` and ``^`` are AND (bind tighter); a leading ``!`` negates
    a term. Compound terms that aren't tags themselves (``lawful_good``) mean all
    of their parts (``lawful`` AND ``good``).
    """
    tagset = set(tags)

    def has(name: str) -> bool:
        if name in tagset:
            return True
        parts = name.split("_")
        return len(parts) > 1 and all(p in tagset for p in parts)

    for alt in str(expr).split("|"):
        ok = True
        for term in re.split(r"[&^]", alt):
            term = term.strip()
            if not term:
                continue
            neg = term.startswith("!")
            if has(term.lstrip("!")) == neg:
                ok = False
                break
        if ok:
            return True
    return False
