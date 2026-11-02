"""Blessings and patron perks: account-wide bonuses outside the formation.

* **Blessings** (``reset_upgrade_defines``, from ``getcampaigndetails``) are
  bought with a campaign's favor. ``type 1`` blessings are **local**: they
  apply only in adventures of that campaign. ``type 2`` blessings are
  **global**: they apply everywhere (e.g. "The Gift of Mechanus").
* **Patron perks** (``patron_perk_defines``) are bought with patron currency.
  ``type 1`` perks are **local**: only while playing under that patron.
  ``type 2`` perks are **global**: always on.

Amounts are ``per_level × your level``, substituted for ``$replace`` in the
effect strings (including inside ``effect_def`` expansions). Effects without
explicit targets apply to all champions.

Several of these change which formation is best, not just how strong it is:
slot-targeted blessings (Hold the Line), tag-filtered ones (Common Races,
Hearts of Neutrality), and per-unique species/class/alignment multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .definitions import GameData
from .effects import Effect, normalize
from .user import UserState

PATRONS = {1: "Mirt", 2: "Vajra", 3: "Strahd", 4: "Zariel", 5: "Elminster"}


@dataclass(frozen=True)
class Bonus:
    name: str
    kind: str  # "local blessing", "global blessing", "local perk", "global perk"
    level: int
    effects: tuple[Effect, ...]
    currency_id: int | None = None
    patron_id: int | None = None


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _materialise(entries: list, level: int, data: GameData, name: str) -> tuple[Effect, ...]:
    out: list[Effect] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        per_level = entry.get("per_level")
        amount = None if per_level is None else float(per_level) * level
        for e in normalize([entry], data.effect_defines):
            args = e.args
            if amount is not None and "$replace" in args:
                args = tuple(f"{amount:.10g}" if a == "$replace" else a for a in args)
            elif "$replace" in args:
                continue  # amount supplied by a system we don't model
            raw = dict(e.raw)
            raw.setdefault("targets", ["all"])  # account-wide: applies to every champion
            raw["_bonus"] = name
            out.append(replace(e, args=args, raw=raw))
    return tuple(out)


class Bonuses:
    def __init__(self, data: GameData, user: UserState, campaign_details: dict | None):
        self.data = data
        self.user = user
        defines = (campaign_details or {}).get("defines") or {}
        self.blessings: list[Bonus] = []
        for b in defines.get("reset_upgrade_defines") or []:
            level = user.blessing_levels.get(_int(b.get("id")), 0)
            if level <= 0:
                continue
            kind = "global blessing" if _int(b.get("type")) == 2 else "local blessing"
            name = str(b.get("name") or b.get("id"))
            self.blessings.append(Bonus(name, kind, level, _materialise(b.get("effects"), level, data, name),
                                        currency_id=_int(b.get("reset_currency_id"))))
        self.perks: list[Bonus] = []
        for p in data.raw.get("patron_perk_defines") or []:
            level = user.perk_levels.get(_int(p.get("id")), 0)
            if level <= 0:
                continue
            kind = "global perk" if _int(p.get("type")) == 2 else "local perk"
            name = str(p.get("name") or p.get("id"))
            self.perks.append(Bonus(name, kind, level, _materialise(p.get("effects"), level, data, name),
                                    patron_id=_int(p.get("patron_id"))))
        self.campaign_currency = {
            _int(c.get("id")): _int(c.get("reset_currency_id")) for c in data.raw.get("campaign_defines") or []
        }
        # Divine favor itself: "gold_multiplier_add,$(reset_currency_count)" = +count% gold find
        self.favor_gold: dict[int, Bonus] = {}
        for rc in defines.get("reset_currency_defines") or []:
            cid = _int(rc.get("id"))
            count = user.favor.get(cid, 0.0)
            if count <= 0:
                continue
            for eff in rc.get("effects") or []:
                s = str(eff.get("effect_string", "")) if isinstance(eff, dict) else ""
                if s.startswith("gold_multiplier_add,"):
                    name = f"{rc.get('name', 'Favor')} ({count:.2e})"
                    e = Effect(key="gold_multiplier_mult", args=(f"{count:.10g}",),
                               raw={"effect_string": f"gold_multiplier_mult,{count:.10g}", "targets": ["all"], "_bonus": name})
                    self.favor_gold[cid] = Bonus(name, "favor", 1, (e,), currency_id=cid)

    @property
    def available(self) -> bool:
        return bool(self.blessings or self.perks)

    def for_adventure(self, adventure_id: int | None, patron_id: int | None = None) -> list[Bonus]:
        currency = None
        if adventure_id is not None and adventure_id in self.data.adventures:
            currency = self.campaign_currency.get(self.data.adventures[adventure_id].campaign_id)
        out = [b for b in self.blessings if b.kind == "global blessing" or b.currency_id == currency]
        if currency in self.favor_gold:
            out.append(self.favor_gold[currency])
        out += [p for p in self.perks if p.kind == "global perk" or (patron_id and p.patron_id == patron_id)]
        return out

    def effects_for(self, adventure_id: int | None, patron_id: int | None = None) -> list[Effect]:
        return [e for b in self.for_adventure(adventure_id, patron_id) for e in b.effects]


def patron_game_changes(data: GameData, patron_id: int | None) -> list[dict]:
    """Restrictions that apply when playing under a patron (e.g. Mirt: Good or Evil only)."""
    if not patron_id:
        return []
    for p in data.raw.get("patron_defines") or []:
        if _int(p.get("id")) == patron_id:
            changes = p.get("game_changes") or {}
            if isinstance(changes, dict):
                return [g for lvl in sorted(changes) for g in changes[lvl] if isinstance(g, dict)]
            return [g for g in changes if isinstance(g, dict)]
    return []


def patron_id(name_or_id: str | int | None) -> int | None:
    if name_or_id in (None, "", "none", 0, "0"):
        return None
    s = str(name_or_id).strip().lower()
    if s.isdigit():
        return int(s)
    for pid, short in PATRONS.items():
        if s in short.lower() or short.lower() in s:
            return pid
    raise KeyError(f"unknown patron {name_or_id!r} (Mirt, Vajra, Strahd, Zariel, Elminster)")
