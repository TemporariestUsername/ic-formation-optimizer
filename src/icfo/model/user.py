"""Typed view over the ``getuserdetails`` payload.

Parsed (shape confirmed against a live account)::

    details.heroes[]          hero_id, owned "1", level, upgrades[],
                              specialization_choices[], active_feats[]
    details.loot[]            hero_id, slot_id, rarity 1-4, gild 0-2, enchant
    details.stats{}           briv_sprint_stacks, thellora_plateaus_of_unicorn_run_areas, ...
    details.game_instances[]  current_adventure_id, current_area, highest_area,
                              formation[] (hero id per slot, -1 = empty)
    details.modron_saves{}    instance_id, area_goal (auto-reset area)
    details.adventure_data    gold_growth_rate, currency_growth_rate (favor)

The payload also contains personal data (email, linked accounts); nothing
here reads or stores those, and the raw cache must never be committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GearItem:
    hero_id: int
    slot_id: int
    rarity: int  # 1 common .. 4 epic
    gild: int  # 0 normal, 1 shiny, 2 golden
    enchant: int

    @property
    def multiplier(self) -> float:
        """Scaling of the item's listed effect: (1 + 0.004·enchant)(1 + 0.5·gild).

        Same formula ic.byteglow.com uses for item effects.
        """
        return (1.0 + 0.004 * self.enchant) * (1.0 + 0.5 * self.gild)


@dataclass(frozen=True)
class GameInstance:
    instance_id: int
    adventure_id: int
    current_area: int
    highest_area: int
    formation: tuple[int, ...]  # hero id per slot, -1 empty
    reset_area: int | None = None  # Modron auto-reset area for this instance


@dataclass
class UserState:
    owned: set[int]
    levels: dict[int, int] = field(default_factory=dict)
    spec_choices: dict[int, list[int]] = field(default_factory=dict)  # hero -> chosen spec upgrade ids
    feats: dict[int, tuple[int, ...]] = field(default_factory=dict)  # hero -> active feat ids
    unlocked_feats: dict[int, set[int]] = field(default_factory=dict)  # hero -> owned feat ids
    feat_slots: dict[int, int] = field(default_factory=dict)
    gear: dict[int, list[GearItem]] = field(default_factory=dict)
    stats: dict[str, str] = field(default_factory=dict)
    instances: list[GameInstance] = field(default_factory=list)
    blessing_levels: dict[int, int] = field(default_factory=dict)  # reset upgrade id -> level
    perk_levels: dict[int, int] = field(default_factory=dict)  # patron perk id -> level
    favor: dict[int, float] = field(default_factory=dict)  # reset currency id -> current favor
    gold_growth_rate: float = 2.031
    currency_growth_rate: float = 1.25
    hero_raw: dict[int, dict] = field(default_factory=dict)
    from_account: bool = False

    @classmethod
    def from_details(cls, payload: dict[str, Any]) -> "UserState":
        details = payload.get("details", payload)
        owned: set[int] = set()
        levels: dict[int, int] = {}
        specs: dict[int, list[int]] = {}
        feats: dict[int, tuple[int, ...]] = {}
        unlocked: dict[int, set[int]] = {}
        slots: dict[int, int] = {}
        hero_raw: dict[int, dict] = {}
        for h in details.get("heroes") or []:
            hid = _int(h.get("hero_id"), -1)
            if hid < 0:
                continue
            hero_raw[hid] = h
            if str(h.get("owned")) == "1":
                owned.add(hid)
            levels[hid] = _int(h.get("level"))
            specs[hid] = [_int(x) for x in h.get("specialization_choices") or []]
            feats[hid] = tuple(_int(x) for x in h.get("active_feats") or [])
            unlocked[hid] = {_int(x) for x in h.get("unlocked_feats") or []}
            slots[hid] = _int(h.get("feat_slots"), 4)
        gear: dict[int, list[GearItem]] = {}
        for item in details.get("loot") or []:
            g = GearItem(
                hero_id=_int(item.get("hero_id")),
                slot_id=_int(item.get("slot_id")),
                rarity=_int(item.get("rarity")),
                gild=_int(item.get("gild")),
                enchant=_int(item.get("enchant")),
            )
            gear.setdefault(g.hero_id, []).append(g)
        reset_by_instance: dict[int, int] = {}
        for save in (details.get("modron_saves") or {}).values():
            if isinstance(save, dict):
                reset_by_instance[_int(save.get("instance_id"))] = _int(save.get("area_goal")) or None
        instances = []
        for gi in details.get("game_instances") or []:
            iid = _int(gi.get("game_instance_id"))
            instances.append(GameInstance(
                instance_id=iid,
                adventure_id=_int(gi.get("current_adventure_id"), -1),
                current_area=_int(gi.get("current_area")),
                highest_area=_int(gi.get("highest_area")),
                formation=tuple(_int(x, -1) for x in gi.get("formation") or []),
                reset_area=reset_by_instance.get(iid),
            ))
        adv = details.get("adventure_data") or {}
        return cls(
            owned=owned,
            levels=levels,
            spec_choices=specs,
            feats=feats,
            unlocked_feats=unlocked,
            feat_slots=slots,
            gear=gear,
            stats={str(k): str(v) for k, v in (details.get("stats") or {}).items()},
            instances=instances,
            blessing_levels={_int(k): _int(v) for k, v in (details.get("reset_upgrade_levels") or {}).items()},
            perk_levels={_int(p.get("patron_perk_id")): _int(p.get("level"))
                         for p in details.get("patron_perks") or [] if isinstance(p, dict)},
            favor={_int(r.get("currency_id")): _float(r.get("current_amount"))
                   for r in details.get("reset_currencies") or [] if isinstance(r, dict)},
            gold_growth_rate=_float(adv.get("gold_growth_rate"), 2.031) or 2.031,
            currency_growth_rate=_float(adv.get("currency_growth_rate"), 1.25) or 1.25,
            hero_raw=hero_raw,
            from_account=True,
        )

    @classmethod
    def everything(cls, hero_ids: set[int]) -> "UserState":
        """A hypothetical account that owns every champion (theory-crafting)."""
        return cls(owned=set(hero_ids))

    def stat(self, name: str, default: float = 0.0) -> float:
        return _float(self.stats.get(name), default)
