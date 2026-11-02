"""Typed view over the ``getDefinitions`` payload."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .effects import Effect, normalize
from .formation import Layout


class LayoutNotFound(KeyError):
    """Adventure has no formation layout (e.g. special modes whose campaign isn't defined)."""


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@dataclass
class Upgrade:
    id: int
    hero_id: int
    required_level: int
    required_upgrade_id: int
    raw_effect: Any
    name: str = ""
    specialization_name: str | None = None
    effects: list[Effect] = field(default_factory=list)

    @property
    def is_specialization(self) -> bool:
        return bool(self.specialization_name)


@dataclass
class Feat:
    id: int
    hero_id: int
    name: str
    effects: list[Effect]


@dataclass
class Champion:
    id: int
    name: str
    seat: int
    tags: frozenset[str]
    stats: dict[str, int] = field(default_factory=dict)  # ability scores: str, dex, ...
    upgrades: list[Upgrade] = field(default_factory=list)
    feats: list[Feat] = field(default_factory=list)

    @cached_property
    def spec_groups(self) -> list[list[Upgrade]]:
        """Mutually exclusive specialization choices, grouped by unlock point."""
        groups: dict[tuple[int, int], list[Upgrade]] = {}
        for u in self.upgrades:
            if u.is_specialization:
                groups.setdefault((u.required_level, u.required_upgrade_id), []).append(u)
        return [sorted(g, key=lambda u: u.id) for _, g in sorted(groups.items())]

    def active_upgrades(
        self, level: int | None = None, specs: dict[int, int] | None = None
    ) -> list[Upgrade]:
        """Upgrades active at ``level`` (None = all) with the given spec picks.

        ``specs`` maps spec-group index -> chosen upgrade id; unspecified groups
        default to the first option (placeholder until spec optimisation lands).
        """
        specs = specs or {}
        chosen: set[int] = set()
        for gi, group in enumerate(self.spec_groups):
            pick = specs.get(gi, group[0].id)
            chosen.add(pick)
        candidates = [
            u
            for u in self.upgrades
            if (level is None or u.required_level <= level)
            and (not u.is_specialization or u.id in chosen)
        ]
        present = {u.id for u in candidates}
        # Upgrades gated behind an unchosen spec (or a locked upgrade) are inactive.
        return [u for u in candidates if not u.required_upgrade_id or u.required_upgrade_id in present]

    def effects(
        self,
        level: int | None = None,
        specs: dict[int, int] | None = None,
        feat_ids: tuple[int, ...] = (),
    ) -> list[Effect]:
        effs: list[Effect] = []
        for u in self.active_upgrades(level, specs):
            effs.extend(u.effects)
        feats = {f.id: f for f in self.feats}
        for fid in feat_ids:
            if fid in feats:
                effs.extend(feats[fid].effects)
        return effs


@dataclass
class Adventure:
    id: int
    name: str
    campaign_id: int
    variant_of: int | None
    restrictions_text: str
    game_changes: list[dict]


class GameData:
    def __init__(self, defs: dict[str, Any]):
        self.raw = defs
        self.effect_defines: dict[int, dict] = {
            _int(e.get("id")): e for e in defs.get("effect_defines") or []
        }
        self.champions: dict[int, Champion] = {}
        for h in defs.get("hero_defines") or []:
            seat = _int(h.get("seat_id"))
            if seat <= 0 or not h.get("name"):
                continue
            sheet = h.get("character_sheet_details") or {}
            scores = sheet.get("ability_scores") if isinstance(sheet, dict) else None
            self.champions[_int(h["id"])] = Champion(
                id=_int(h["id"]),
                name=str(h["name"]),
                seat=seat,
                tags=frozenset(str(t) for t in h.get("tags") or []),
                stats={str(k).lower(): _int(v) for k, v in (scores or {}).items()},
            )
        self.upgrades: dict[int, Upgrade] = {}
        for u in defs.get("upgrade_defines") or []:
            hid = _int(u.get("hero_id"))
            champ = self.champions.get(hid)
            if champ is None:
                continue
            up = Upgrade(
                id=_int(u.get("id")),
                hero_id=hid,
                required_level=_int(u.get("required_level")),
                required_upgrade_id=_int(u.get("required_upgrade_id")),
                raw_effect=u.get("effect"),
                name=str(u.get("name") or ""),
                specialization_name=u.get("specialization_name") or None,
            )
            up.effects = normalize(up.raw_effect, self.effect_defines, hero_id=hid, upgrade_id=up.id)
            self.upgrades[up.id] = up
            champ.upgrades.append(up)
        for f in defs.get("hero_feat_defines") or []:
            champ = self.champions.get(_int(f.get("hero_id")))
            if champ is None:
                continue
            champ.feats.append(
                Feat(
                    id=_int(f.get("id")),
                    hero_id=champ.id,
                    name=str(f.get("name") or ""),
                    effects=normalize(f.get("effects"), self.effect_defines, hero_id=champ.id),
                )
            )
        self.campaigns: dict[int, dict] = {
            _int(c.get("id")): c for c in defs.get("campaign_defines") or []
        }
        self.adventures: dict[int, Adventure] = {}
        for a in defs.get("adventure_defines") or []:
            self.adventures[_int(a.get("id"))] = Adventure(
                id=_int(a.get("id")),
                name=str(a.get("name") or ""),
                campaign_id=_int(a.get("campaign_id")),
                variant_of=_int(a.get("variant_adventure_id")) or None,
                restrictions_text=str(a.get("restrictions_text") or ""),
                game_changes=[g for g in a.get("game_changes") or [] if isinstance(g, dict)],
            )

    # -- lookups -------------------------------------------------------------
    def champion(self, name_or_id: str | int) -> Champion:
        if isinstance(name_or_id, int) or str(name_or_id).isdigit():
            return self.champions[int(name_or_id)]
        needle = str(name_or_id).casefold()
        matches = [c for c in self.champions.values() if c.name.casefold() == needle]
        if not matches:
            matches = [c for c in self.champions.values() if needle in c.name.casefold()]
        if len(matches) != 1:
            names = ", ".join(sorted(c.name for c in matches)) or "none"
            raise KeyError(f"Champion {name_or_id!r} matched: {names}")
        return matches[0]

    def layout_for(self, adventure_id: int) -> Layout:
        adv = self.adventures[adventure_id]
        for gc in adv.game_changes:
            if gc.get("type") == "formation":
                return Layout.from_game_change(gc)
        campaign = self.campaigns.get(adv.campaign_id) or {}
        for gc in campaign.get("game_changes") or []:
            if isinstance(gc, dict) and gc.get("type") == "formation":
                return Layout.from_game_change(gc)
        raise LayoutNotFound(f"No formation layout found for adventure {adventure_id}")
