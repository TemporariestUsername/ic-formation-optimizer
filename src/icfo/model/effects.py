"""Effect parsing.

Game effects are strings like ``hero_dps_multiplier_mult,100`` wrapped in
loosely-typed objects. The same concept shows up in several shapes in the
definitions (observed in live data):

* a bare string:            ``"hero_dps_multiplier_mult,100"``
* a JSON-encoded string:    ``'{"effect_string":"buff_upgrade,50,12"}'``
* an indirection:           ``"effect_def,310"`` -> ``effect_defines[310].effect_keys``
* ``effect_keys`` as a list of objects, a single object, or plain strings
* the string under ``effect_string`` or (rarely) ``effect``
* ``targets`` as a list, a string, an int, or missing (= the source itself)

``normalize`` flattens all of that into a list of :class:`Effect`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

MAX_EFFECT_DEF_DEPTH = 5


@dataclass(frozen=True)
class Effect:
    key: str
    args: tuple[str, ...]
    targets: tuple[Any, ...] = ("self",)
    filters: tuple[Any, ...] = ()
    stacks_multiply: bool = False
    exclude_self: bool = False
    hero_id: int | None = None
    upgrade_id: int | None = None
    raw: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def num(self, i: int, default: float = 0.0) -> float:
        try:
            return float(self.args[i])
        except (IndexError, ValueError):
            return default


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
    _depth: int = 0,
) -> list[Effect]:
    out: list[Effect] = []
    for raw in _as_list(value):
        entry = _coerce_entry(raw)
        if entry is None:
            continue
        key, *args = [p.strip() for p in entry["effect_string"].split(",")]
        if key == "effect_def" and args and _depth < MAX_EFFECT_DEF_DEPTH:
            try:
                target_def = effect_defines.get(int(args[0]))
            except ValueError:
                target_def = None
            if target_def is not None:
                inner = normalize(
                    target_def.get("effect_keys"),
                    effect_defines,
                    hero_id=hero_id,
                    upgrade_id=upgrade_id,
                    _depth=_depth + 1,
                )
                # Targets on the wrapper override defaults on inner keys that had none.
                if "targets" in entry:
                    tgt = tuple(_as_list(entry["targets"]))
                    inner = [
                        e if "targets" in e.raw else _replace_targets(e, tgt) for e in inner
                    ]
                out.extend(inner)
            continue
        targets = tuple(_as_list(entry.get("targets"))) or ("self",)
        out.append(
            Effect(
                key=key,
                args=tuple(args),
                targets=targets,
                filters=tuple(_as_list(entry.get("filter_targets"))),
                stacks_multiply=bool(entry.get("stacks_multiply")),
                exclude_self=bool(entry.get("exclude_self")),
                hero_id=hero_id,
                upgrade_id=upgrade_id,
                raw=entry,
            )
        )
    return out


def _replace_targets(e: Effect, targets: tuple[Any, ...]) -> Effect:
    return Effect(
        key=e.key,
        args=e.args,
        targets=targets,
        filters=e.filters,
        stacks_multiply=e.stacks_multiply,
        exclude_self=e.exclude_self,
        hero_id=e.hero_id,
        upgrade_id=e.upgrade_id,
        raw=e.raw,
    )


def match_tags(expr: str, tags: Iterable[str]) -> bool:
    """Evaluate a tag expression such as ``dwarf``, ``!human`` or ``druid|ranger``.

    ``|`` is OR, ``&`` is AND (binds tighter), leading ``!`` negates a term.
    """
    tagset = set(tags)
    for alt in str(expr).split("|"):
        ok = True
        for term in alt.split("&"):
            term = term.strip()
            if not term:
                continue
            neg = term.startswith("!")
            name = term.lstrip("!")
            if (name in tagset) == neg:
                ok = False
                break
        if ok:
            return True
    return False
