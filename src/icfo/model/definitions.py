"""Typed view over the ``getDefinitions`` payload."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from .effects import Effect, normalize
from .expr import BASE_FUNCS, UnsupportedExpr, evaluate_with
from .formation import Layout


def _species(race: Any, tags: list) -> str | None:
    """Map the character-sheet race ("Dwarf (Shield)", "Half-Orc") to its tag."""
    if not isinstance(race, str):
        return None
    slug = re.sub(r"\s*\(.*\)", "", race).strip().lower().replace(" ", "-")
    tagset = {str(t) for t in tags}
    if slug in tagset:
        return slug
    last = slug.split("-")[-1]  # "wood-elf" -> "elf"
    return last if last in tagset else slug or None


class LayoutNotFound(KeyError):
    """Adventure has no formation layout (e.g. special modes whose campaign isn't defined)."""


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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
    species: str | None = None  # race tag, e.g. "dwarf", "half-orc" (for per-unique-species effects)
    base_damage: float = 1.0
    base_attack_id: int = 0
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
    objective_area: int | None = None  # "complete_area" goal, if any


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
                stats={**{str(k).lower(): _int(v) for k, v in (scores or {}).items()},
                       **({"age": _int(sheet.get("age"))} if isinstance(sheet, dict) and "age" in sheet else {})},
                species=_species(sheet.get("race") if isinstance(sheet, dict) else None, h.get("tags") or []),
                base_damage=_float(h.get("base_damage"), 1.0),
                base_attack_id=_int(h.get("base_attack_id")),
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
                    effects=normalize(f.get("effects"), self.effect_defines, hero_id=champ.id, feat_id=_int(f.get("id"))),
                )
            )
        self.feats: dict[int, Feat] = {f.id: f for c in self.champions.values() for f in c.feats}
        self._resolve_amounts()
        # Upgrades holding a pre_stack parameter: buffs to them scale the
        # per-stack amount ("Buffs apply to the pre-stack value"). Everywhere
        # else buff_upgrade is "post stack" (game key notes).
        self.pre_stack_uids: frozenset[int] = frozenset(
            u.id for u in self.upgrades.values()
            if any(e.key in ("pre_stack", "pre_stack_amount") for e in u.effects)
        )
        # gear: (hero, slot, rarity) -> effects listed for that item
        self.loot: dict[tuple[int, int, int], list[Effect]] = {}
        for item in defs.get("loot_defines") or []:
            hid = _int(item.get("hero_id"))
            if hid in self.champions:
                key = (hid, _int(item.get("slot_id")), _int(item.get("rarity")))
                self.loot[key] = normalize(item.get("effects"), self.effect_defines, hero_id=hid)
        self.attacks: dict[int, dict] = {_int(a.get("id")): a for a in defs.get("attack_defines") or []}
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
                objective_area=next((_int(o.get("area")) for o in a.get("objectives") or []
                                     if isinstance(o, dict) and o.get("condition") == "complete_area"), None),
                game_changes=[g for g in a.get("game_changes") or [] if isinstance(g, dict)],
            )

    # -- amount formulas -------------------------------------------------------
    def _effect_amount(self, effects: list[Effect], index: Any, depth: int) -> float:
        if not isinstance(index, int) or not 0 <= index < len(effects):
            raise UnsupportedExpr(f"effect index {index!r}")
        e = effects[index]
        if "amount_expr" in e.raw:
            if depth > 4:
                raise UnsupportedExpr("amount formula nests too deep")
            return self._eval_amount(e.raw["amount_expr"], depth + 1)
        amount = e.base_amount
        if amount is None:
            raise UnsupportedExpr(f"non-numeric amount in {e.key}")
        return amount

    def _eval_amount(self, expr: str, depth: int = 0) -> float:
        def upgrade_amount(uid, idx):
            up = self.upgrades.get(int(uid))
            if up is None:
                raise UnsupportedExpr(f"unknown upgrade {uid}")
            return self._effect_amount(up.effects, idx, depth)

        def feat_amount(fid, idx):
            feat = self.feats.get(int(fid))
            if feat is None:
                raise UnsupportedExpr(f"unknown feat {fid}")
            return self._effect_amount(feat.effects, idx, depth)

        funcs = {**BASE_FUNCS, "upgrade_amount": upgrade_amount,
                 "max_upgrade_amount": upgrade_amount, "feat_amount": feat_amount}

        def names(name: str):
            raise KeyError(name)

        value = evaluate_with(str(expr), names, funcs)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedExpr(f"amount formula {expr!r} is not numeric")
        return float(value)

    def _amount_refs(self, expr: str, depth: int = 0) -> tuple[tuple[tuple[int, int], ...], float | None]:
        """Parameters a pure-product formula reads (recursively), and the tightest cap.

        Only products of ``upgrade_amount(U,i)`` and constants qualify: a buff
        to a referenced parameter then scales the result proportionally.
        """
        s = str(expr)
        if "+" in s or "-" in s or "/" in s or depth > 4:
            return (), None
        refs: list[tuple[int, int]] = []
        cap: float | None = None
        for m in re.finditer(r"upgrade_amount\(\s*(\d+)\s*,\s*(\d+)\s*\)", s):
            uid, idx = int(m.group(1)), int(m.group(2))
            refs.append((uid, idx))
            up = self.upgrades.get(uid)
            if up is not None and idx < len(up.effects):
                ref = up.effects[idx]
                c = ref.raw.get("effect_cap")
                if isinstance(c, (int, float)):
                    cap = c if cap is None else min(cap, c)
                if "amount_expr" in ref.raw:
                    sub_refs, sub_cap = self._amount_refs(ref.raw["amount_expr"], depth + 1)
                    refs += sub_refs
                    if sub_cap is not None:
                        cap = sub_cap if cap is None else min(cap, sub_cap)
        stripped = re.sub(r"upgrade_amount\([^)]*\)", "", s)
        if re.search(r"[A-Za-z_]", stripped):
            return (), cap  # other functions involved: not a pure product
        return tuple(dict.fromkeys(refs)), cap

    def _resolve_amounts(self) -> None:
        """Fill ``resolved_amount`` for effects whose amount lives in a formula."""
        def resolve(effects: list[Effect]) -> list[Effect]:
            out = []
            for e in effects:
                if "amount_expr" in e.raw:
                    try:
                        e = replace(e, resolved_amount=self._eval_amount(e.raw["amount_expr"]))
                    except (UnsupportedExpr, ValueError, TypeError):
                        pass  # stays None -> reported as unmodelled
                    refs, cap = self._amount_refs(e.raw["amount_expr"])
                    own_cap = e.raw.get("effect_cap")
                    if isinstance(own_cap, (int, float)):
                        cap = own_cap if cap is None else min(cap, own_cap)
                    e = replace(e, amount_refs=refs, amount_cap=cap)
                out.append(e)
            return out

        resolved_up = {uid: resolve(u.effects) for uid, u in self.upgrades.items()}
        resolved_feat = {fid: resolve(f.effects) for fid, f in self.feats.items()}
        for uid, effs in resolved_up.items():
            self.upgrades[uid].effects = effs
        for fid, effs in resolved_feat.items():
            self.feats[fid].effects = effs

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
