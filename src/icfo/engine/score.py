"""Score a placement by the damage multiplier reaching one champion (the "carry").

Idle Champions damage is (mostly) a product of independent multipliers, so we
work in log10 space: the score is ``sum(log10(multiplier))`` over every buff
that reaches the carry. A score of 12.3 means "x10^12.3 from the formation".

The carry's base damage is a constant for a fixed carry and is left out, so
scores compare formations for the *same* carry.

Pipeline for one placement:

1. **Modifiers** from placed champions patch other upgrades' data and targets
   (``change_upgrade_data``, ``add_target_to_upgrade``, ...).
2. **Amplifiers** (``buff_upgrade`` family) are resolved to a factor per
   upgrade (or per effect index of an upgrade)::

       factor = (1 + sum(additive buffs)) * prod(multiplicative buffs)

   Amplifiers can amplify other amplifiers, so this iterates to a fixed point.
3. **Damage effects** are checked against the carry (targets + filters) and
   stacked. The amplifier factor scales the *stacked total* ("post stack",
   the game's default for buff_upgrade) unless the upgrade holds a
   ``pre_stack`` parameter, in which case it scales the per-stack amount.

Stack counts come from the formation where possible (champions with a tag,
adjacent champions, unique species, per-champion expressions, ...). Stacks
earned at runtime (kills, attacks, timers) are assumed to be at ``max_stacks``
and the contribution is marked *approximate*.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..model.definitions import Champion, GameData
from ..model.effects import Effect, match_tags
from ..model.expr import UnsupportedExpr, evaluate
from ..model.formation import Layout
from ..model.user import GearItem, UserState
from .catalog import (
    AMPLIFIER_KEYS,
    DAMAGE_KEYS,
    GOLD_KEYS,
    SPEED_KEYS,
    INTRINSIC_KEYS,
    MODIFIER_KEYS,
    NON_DAMAGE_KEYS,
    PARAM_KEYS,
)

AMPLIFIER_PASSES = 4
# Runtime stacks are assumed to be maxed only for generic triggers with a
# reachable cap; bigger caps (10k .. 4e9 exist) are never reached in practice.
MAX_ASSUMED_STACKS = 500
_OPS = {"<": float.__lt__, "<=": float.__le__, ">": float.__gt__, ">=": float.__ge__, "==": float.__eq__, "!=": float.__ne__}


@dataclass(frozen=True)
class Contribution:
    source: str
    key: str
    upgrade_id: int | None
    log10: float
    exact: bool = True
    channel: str = "damage"


@dataclass(frozen=True)
class SpeedEffect:
    source: str
    hero_id: int
    key: str
    args: tuple[str, ...]
    amount: float | None  # amplified amount (None when the key carries no amount)
    stacks: float


@dataclass
class ScoreResult:
    log10: float
    contributions: list[Contribution] = field(default_factory=list)
    unmodelled: Counter = field(default_factory=Counter)
    unmodelled_by_hero: dict[str, Counter] = field(default_factory=dict)
    modelled_exact: int = 0
    modelled_approx: int = 0
    gold_log10: float = 0.0
    speed: list[SpeedEffect] = field(default_factory=list)
    damage_log10: float | None = None  # set when log10 is replaced by an objective value

    @property
    def coverage(self) -> float:
        """Share of potentially relevant effects on placed champions that are modelled."""
        modelled = self.modelled_exact + self.modelled_approx
        missing = sum(self.unmodelled.values())
        return 1.0 if modelled + missing == 0 else modelled / (modelled + missing)

    @property
    def approx_share(self) -> float:
        """Share of the score (log10) that rests on max-stack / uptime assumptions."""
        total = sum(abs(c.log10) for c in self.contributions)
        approx = sum(abs(c.log10) for c in self.contributions if not c.exact)
        return 0.0 if total == 0 else approx / total


def _log_mult(pct: float) -> float:
    """log10 of a "+pct%" multiplier, guarded against nonsense values."""
    factor = 1.0 + pct / 100.0
    return math.log10(factor) if factor > 0 else 0.0


_FACTOR_CAP = 1e250  # keep amplifier factors finite; anything near this is flagged as assumed anyway


def _safe_pow(base: float, n: float) -> float:
    if base <= 0:
        return 0.0
    exp = n * math.log10(base)
    return 10.0 ** min(exp, 250.0)


def _replace_amount(e: Effect, amount: float) -> Effect:
    from dataclasses import replace

    return replace(e, args=(f"{amount:.10g}", *e.args[1:]))


def _self_only(e: Effect) -> bool:
    t = e.raw.get("targets")
    return t is None or t == ["self"] or t == "self"


_FORMATION_STACK_FUNCS = frozenset({
    "per_hero_attribute", "per_crusader", "per_hero", "per_tagged_crusader_mult",
    "adjacent_champions", "per_unique_race", "per_upgrade_targets",
})


# amplifiers whose stack count can't depend on the formation when they have no
# stack_func/trigger: compiled once per champion
_STATIC_AMP_KEYS = frozenset({
    "buff_upgrade", "buff_upgrades", "buff_upgrade_multiplicative",
    "buff_upgrades_multiplicative", "buff_upgrade_add_then_mult", "buff_upgrade_add_flat_amount",
})


@dataclass
class _Compiled:
    """A champion's effects sorted by how the engine handles them."""
    modifiers: list = field(default_factory=list)
    static_amps: list = field(default_factory=list)  # (effect, keys, mode, n, exact)
    bad_amps: list = field(default_factory=list)
    dynamic_amps: list = field(default_factory=list)
    self_only: list = field(default_factory=list)  # hero_dps buffs targeting only the owner
    others: list = field(default_factory=list)


class _Unmodelled(Exception):
    pass


@dataclass
class _Ctx:
    placement: dict[int, Champion]
    carry: Champion
    carry_slot: int
    present: list[Champion]
    occupied: frozenset[int]
    effects: dict[int, list[Effect]]  # slot -> patched effects


class Scorer:
    def __init__(self, data: GameData, layout: Layout):
        self.data = data
        self.layout = layout
        self._effects: dict[int, list[Effect]] = {}
        self.levels: dict[int, int | None] = {}
        self.specs: dict[int, dict[int, int]] = {}
        self.feats: dict[int, tuple[int, ...]] = {}
        # user-asserted stack counts for runtime-stacked effects, by upgrade id
        self.stack_overrides: dict[int, float] = {}
        self.gear: dict[int, list[GearItem]] = {}
        # share of max_stacks assumed for runtime-stacked effects (1.0 = fully stacked)
        self.stack_fraction: float = 1.0
        self._uid_cache: dict[int, frozenset[int]] = {}
        self._compiled_cache: dict[int, _Compiled] = {}
        self._patched_cache: dict[tuple, tuple[list[Effect], _Compiled]] = {}

    def effects_for(self, champ: Champion) -> list[Effect]:
        if champ.id not in self._effects:
            effs = champ.effects(
                level=self.levels.get(champ.id),
                specs=self.specs.get(champ.id),
                feat_ids=self.feats.get(champ.id, ()),
            )
            self._effects[champ.id] = effs + self._gear_effects(champ)
        return self._effects[champ.id]

    def _gear_effects(self, champ: Champion) -> list[Effect]:
        out = []
        for item in self.gear.get(champ.id, ()):
            for e in self.data.loot.get((champ.id, item.slot_id, item.rarity), ()):
                base = e.base_amount
                if base is None or not e.args:
                    out.append(e)
                    continue
                out.append(_replace_amount(e, base * item.multiplier))
        return out

    def apply_account(self, user: UserState) -> None:
        """Use the account's specializations, feats and gear."""
        self.feats = {h: f for h, f in user.feats.items() if f}
        self.gear = {h: g for h, g in user.gear.items()}
        self.specs = {}
        for hid, chosen in user.spec_choices.items():
            champ = self.data.champions.get(hid)
            if champ is None or not chosen:
                continue
            picks = {}
            for gi, group in enumerate(champ.spec_groups):
                for u in group:
                    if u.id in chosen:
                        picks[gi] = u.id
            self.specs[hid] = picks
        self.invalidate()

    def invalidate(self, hero_id: int | None = None) -> None:
        if hero_id is None:
            self._effects.clear()
            self._compiled_cache.clear()
            self._patched_cache.clear()
        else:
            self._effects.pop(hero_id, None)
            self._compiled_cache.pop(hero_id, None)
            self._patched_cache = {k: v for k, v in self._patched_cache.items() if k[0] != hero_id}

    # ------------------------------------------------------------------
    def score(self, placement: dict[int, Champion], carry_id: int) -> ScoreResult:
        slot_of = {c.id: s for s, c in placement.items()}
        if carry_id not in slot_of:
            return ScoreResult(log10=float("-inf"))
        carry_slot = slot_of[carry_id]
        ctx = _Ctx(
            placement=placement,
            carry=placement[carry_slot],
            carry_slot=carry_slot,
            present=list(placement.values()),
            occupied=frozenset(placement),
            effects={},
        )
        result = ScoreResult(log10=0.0)
        compiled = self._apply_modifiers(ctx, result)
        factor, flat, approx_keys = self._amplifiers(ctx, compiled, result)

        for src_slot, src in placement.items():
            comp = compiled[src_slot]
            if src_slot == carry_slot:
                todo = comp.others + comp.self_only
            else:
                todo = comp.others
                result.modelled_exact += len(comp.self_only)  # buff only their owner
            for e in todo:
                try:
                    if e.key in GOLD_KEYS:
                        value, exact = self._gold(e, src, src_slot, ctx, factor, flat)
                        result.modelled_exact += exact
                        result.modelled_approx += not exact
                        if value:
                            result.gold_log10 += value
                            result.contributions.append(Contribution(src.name, e.key, e.upgrade_id, value, exact, "gold"))
                        continue
                    if e.key in SPEED_KEYS:
                        result.speed.append(self._speed(e, src, src_slot, ctx, factor, flat))
                        result.modelled_exact += 1
                        continue
                    if e.key in INTRINSIC_KEYS:
                        self._check_intrinsic(e, src, src_slot, ctx)
                        continue
                    if e.key not in DAMAGE_KEYS:
                        raise _Unmodelled
                    value, exact = self._damage(e, src, src_slot, ctx, factor, flat)
                    if exact and e.upgrade_id is not None and (
                        (e.upgrade_id, None) in approx_keys or (e.upgrade_id, e.index) in approx_keys
                    ):
                        exact = False  # amplified by an assumed-max-stacks buff
                except (_Unmodelled, UnsupportedExpr):
                    self._note_unmodelled(result, src, e)
                    continue
                if exact:
                    result.modelled_exact += 1
                else:
                    result.modelled_approx += 1
                if value:
                    result.log10 += value
                    result.contributions.append(Contribution(src.name, e.key, e.upgrade_id, value, exact))
        return result

    @staticmethod
    def _note_unmodelled(result: ScoreResult, src: Champion, e: Effect) -> None:
        result.unmodelled[e.key] += 1
        result.unmodelled_by_hero.setdefault(src.name, Counter())[e.key] += 1

    # -- per-champion compilation -----------------------------------------
    def _compile(self, effs: list[Effect]) -> "_Compiled":
        comp = _Compiled()
        for e in effs:
            key = e.key
            if key in MODIFIER_KEYS:
                comp.modifiers.append(e)
            elif key in AMPLIFIER_KEYS:
                if key in _STATIC_AMP_KEYS and e.stack_func is None and not e.triggered:
                    try:
                        if e.base_amount is None:
                            raise _Unmodelled
                        keys, mode, n, exact = self._amplifier_terms(e, None, 0, None)
                        comp.static_amps.append((e, keys, mode, n, exact))
                    except (_Unmodelled, ValueError, IndexError):
                        comp.bad_amps.append(e)
                else:
                    comp.dynamic_amps.append(e)
            elif key in PARAM_KEYS or key in NON_DAMAGE_KEYS:
                continue
            elif key == "hero_dps_multiplier_mult" and _self_only(e):
                comp.self_only.append(e)
            else:
                comp.others.append(e)
        return comp

    def _compiled(self, champ: Champion) -> "_Compiled":
        comp = self._compiled_cache.get(champ.id)
        if comp is None:
            comp = self._compiled_cache[champ.id] = self._compile(self.effects_for(champ))
        return comp

    # -- 1. modifiers ----------------------------------------------------
    def _apply_modifiers(self, ctx: _Ctx, result: ScoreResult) -> dict[int, "_Compiled"]:
        patches: dict[int, list[tuple[int | None, dict]]] = {}
        add_targets: dict[int, list] = {}
        for src_slot, src in ctx.placement.items():
            for e in self._compiled(src).modifiers:
                try:
                    if e.key == "change_upgrade_data":
                        idx = int(e.args[1]) if len(e.args) > 1 and e.args[1].strip().lstrip("-").isdigit() else None
                        data = e.raw.get("data")
                        if not isinstance(data, dict):
                            raise _Unmodelled
                        patches.setdefault(int(e.args[0]), []).append((idx, data))
                    elif e.key == "change_upgrade_targets":
                        new = e.raw.get("new_targets")
                        patches.setdefault(int(e.args[0]), []).append((None, {"targets": new if isinstance(new, list) else [new]}))
                    else:  # add_target_to_upgrade / force_add_target_to_upgrade: rel, upgrade id
                        if "active_condition" in e.raw:
                            raise _Unmodelled
                        rels = [e.args[0]] if e.args and e.args[0] else []
                        rels += list(e.raw.get("add_targets") or [])
                        add_targets.setdefault(int(e.args[1]), []).extend(rels)
                    result.modelled_exact += 1
                except (_Unmodelled, ValueError, IndexError):
                    self._note_unmodelled(result, src, e)

        touched = set(patches) | set(add_targets)
        compiled: dict[int, _Compiled] = {}
        for slot, champ in ctx.placement.items():
            effs = self.effects_for(champ)
            hit = touched & self._upgrade_ids(champ) if touched else None
            if hit:
                # patch data dicts come from the definitions, so their ids are stable
                key = (champ.id, tuple(sorted(
                    (uid, tuple((idx, id(d)) for idx, d in patches.get(uid, ())),
                     tuple(map(repr, add_targets.get(uid, ()))))
                    for uid in hit
                )))
                cached = self._patched_cache.get(key)
                if cached is None:
                    patched = []
                    for e in effs:
                        if e.upgrade_id in patches:
                            for idx, data in patches[e.upgrade_id]:
                                if idx is None or idx == e.index:
                                    e = e.patched(data)
                        if e.upgrade_id in add_targets:
                            e = e.patched({"targets": list(e.targets) + add_targets[e.upgrade_id]})
                        patched.append(e)
                    cached = self._patched_cache[key] = (patched, self._compile(patched))
                ctx.effects[slot], compiled[slot] = cached
            else:
                ctx.effects[slot] = effs
                compiled[slot] = self._compiled(champ)
        return compiled

    # -- 2. amplifiers ---------------------------------------------------
    def _amplifiers(self, ctx: _Ctx, compiled: dict[int, "_Compiled"], result: ScoreResult):
        """Return (factor, flat, approx_keys), keyed by (upgrade_id, index or None).

        Stack counts are computed once; the fixed-point passes (amplifiers of
        amplifiers) only redo the amount scaling.
        """
        present_uids: set[int] = set()
        for c in ctx.present:
            present_uids |= self._upgrade_ids(c)
        terms = []  # (effect, keys, mode, n)
        approx_keys: set[tuple[int, int | None]] = set()

        def add_term(e, keys, mode, n, exact):
            keys = [k for k in keys if k[0] in present_uids]
            if keys and n:
                terms.append((e, keys, mode, n))
                if not exact:
                    approx_keys.update(keys)

        for slot, comp in compiled.items():
            src = ctx.placement[slot]
            for e in comp.bad_amps:
                self._note_unmodelled(result, src, e)
            result.modelled_exact += len(comp.static_amps)
            for e, keys, mode, n, exact in comp.static_amps:
                add_term(e, keys, mode, n, exact)
            for e in comp.dynamic_amps:
                try:
                    if e.base_amount is None:
                        raise _Unmodelled
                    keys, mode, n, exact = self._amplifier_terms(e, src, slot, ctx)
                except (_Unmodelled, UnsupportedExpr, ValueError, IndexError):
                    self._note_unmodelled(result, src, e)
                    continue
                result.modelled_exact += 1
                add_term(e, keys, mode, n, exact)

        factor: dict[tuple[int, int | None], float] = {}
        flat: dict[tuple[int, int | None], float] = {}
        amplified_uids: set[int] = set()
        for _ in range(AMPLIFIER_PASSES):
            add: dict[tuple[int, int | None], float] = {}
            mult: dict[tuple[int, int | None], float] = {}
            new_flat: dict[tuple[int, int | None], float] = {}
            for e, keys, mode, n in terms:
                amplified = e.upgrade_id in amplified_uids
                a = self._scaled_amount(e, factor, flat) if amplified else e.base_amount
                if mode == "mult":
                    # stacked factor of this amplifier; buffs to the amplifier itself
                    # scale its stacked bonus ("post stack") unless it's a pre_stack upgrade
                    if amplified and n != 1 and e.upgrade_id not in self.data.pre_stack_uids:
                        k_all, k_idx = (e.upgrade_id, None), (e.upgrade_id, e.index)
                        base = e.base_amount + flat.get(k_all, 0.0) + flat.get(k_idx, 0.0)
                        f_own = factor.get(k_all, 1.0) * factor.get(k_idx, 1.0)
                        stacked = _safe_pow(1.0 + base / 100.0, n)
                        term = 1.0 + (stacked - 1.0) * f_own
                    else:
                        term = _safe_pow(1.0 + a / 100.0, n)
                for k in keys:
                    if mode == "flat":
                        new_flat[k] = new_flat.get(k, 0.0) + a * n
                    elif mode == "mult":
                        mult[k] = min(mult.get(k, 1.0) * term, _FACTOR_CAP)
                    elif mode == "if":
                        add[k] = add.get(k, 0.0) + (a if n > 0 else 0.0)
                    elif mode == "if_not":
                        add[k] = add.get(k, 0.0) + (a if n == 0 else 0.0)
                    else:  # additive: joins the shared additive pool of that upgrade
                        add[k] = add.get(k, 0.0) + n * a
            new_factor = {k: (1.0 + add.get(k, 0.0) / 100.0) * mult.get(k, 1.0) for k in set(add) | set(mult)}
            converged = new_factor == factor and new_flat == flat
            factor, flat = new_factor, new_flat
            amplified_uids = {k[0] for k in factor} | {k[0] for k in flat}
            # another pass only matters if some amplifier is itself amplified
            if converged or not any(e.upgrade_id in amplified_uids for e, *_ in terms):
                break
        return factor, flat, approx_keys

    def _upgrade_ids(self, champ: Champion) -> frozenset[int]:
        cached = self._uid_cache.get(champ.id)
        if cached is None:
            cached = self._uid_cache[champ.id] = frozenset(u.id for u in champ.upgrades)
        return cached

    def _scaled_amount(self, e: Effect, factor, flat) -> float:
        base = e.base_amount
        if base is None:
            raise _Unmodelled
        if e.upgrade_id is None:
            return base
        k_all, k_idx = (e.upgrade_id, None), (e.upgrade_id, e.index)
        f = min(factor.get(k_all, 1.0) * factor.get(k_idx, 1.0), _FACTOR_CAP)
        return (base + flat.get(k_all, 0.0) + flat.get(k_idx, 0.0)) * f

    def _amplifier_terms(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx):
        """(target keys, combine mode, stack count, exact?) for one amplifier."""
        key, args = e.key, e.args

        def idx_at(i):
            return int(args[i]) if len(args) > i and args[i].strip().isdigit() else None

        if key == "buff_upgrade":
            n, exact = self._stacks(e, src, src_slot, ctx)
            return [(int(args[1]), idx_at(2))], self._mode(e), n, exact
        if key == "buff_upgrades":
            n, exact = self._stacks(e, src, src_slot, ctx)
            return [(int(u), None) for u in args[1:] if u.strip().isdigit()], self._mode(e), n, exact
        if key in ("buff_upgrade_multiplicative", "buff_upgrade_add_then_mult"):
            n, exact = self._stacks(e, src, src_slot, ctx)
            mode = "mult" if key == "buff_upgrade_multiplicative" else "add"
            return [(int(args[1]), idx_at(2))], mode, n, exact
        if key == "buff_upgrades_multiplicative":
            n, exact = self._stacks(e, src, src_slot, ctx)
            return [(int(u), None) for u in args[1:] if u.strip().isdigit()], "mult", n, exact
        if key == "buff_upgrade_by_tag_mult":  # amount, tag, id
            n = self._count(ctx, src, lambda c: match_tags(args[1], c.tags), e.exclude_self)
            return [(int(args[2]), None)], "mult", n, True
        if key == "buff_upgrade_by_tag_additive":  # amount, id, targets, tag (other champions)
            slots = self.layout.targets(args[2], src_slot)
            if slots is None:
                raise _Unmodelled
            n = sum(1 for s in slots - {src_slot} if s in ctx.placement and match_tags(args[3], ctx.placement[s].tags))
            return [(int(args[1]), None)], "add", n, True
        if key == "buff_upgrade_per_crusader":
            return [(int(args[1]), None)], "mult", len(ctx.present), True
        if key == "buff_upgrade_per_unique_race":
            n = len({c.species for c in ctx.present if c.species})
            return [(int(args[1]), None)], self._mode(e), n, True
        if key in ("buff_upgrade_per_target_crusader", "buff_upgrade_per_target_crusader_mult"):
            slots = self.layout.targets(args[2], src_slot)
            if slots is None:
                raise _Unmodelled
            n = len((slots - {src_slot}) & ctx.occupied)
            return [(int(args[1]), None)], "mult" if key.endswith("_mult") else "add", n, True
        if key == "buff_upgrade_add_flat_amount":
            return [(int(args[1]), idx_at(2))], "flat", 1, True
        if key in ("buff_upgrade_per_any_tagged_crusader_mult", "buff_upgrade_per_any_tagged_crusader"):
            expr = args[2] if len(args) > 2 else ""
            n = self._count(ctx, src, lambda c: match_tags(expr, c.tags), e.exclude_self)
            return [(int(args[1]), None)], "mult" if key.endswith("_mult") else "add", n, True
        if key == "buff_upgrade_per_any_crusader_where_mult":
            stat, op, val = args[2].lower(), _OPS.get(args[3]), float(args[4])
            if op is None:
                raise _Unmodelled
            n = self._count(ctx, src, lambda c: op(float(c.stats.get(stat, 0)), val), e.exclude_self)
            return [(int(args[1]), None)], "mult", n, True
        if key == "buff_upgrade_by_target_tag_mult":
            rel, expr, uid = args[1], args[2], int(args[3])
            slots = self.layout.targets(rel, src_slot)
            if slots is None:
                raise _Unmodelled
            n = sum(1 for s in slots - {src_slot} if s in ctx.placement and match_tags(expr, ctx.placement[s].tags))
            return [(uid, None)], "mult" if e.stacks_multiply else "add", n, True
        raise _Unmodelled

    # -- 3. damage -------------------------------------------------------
    def _damage(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx, factor, flat) -> tuple[float, bool]:
        key = e.key
        if key == "global_dps_multiplier_mult":
            if not self._passes_filters(e, ctx.carry):  # targets don't restrict "global"
                return 0.0, True
        elif not self._targets_carry(e, src, src_slot, ctx):
            return 0.0, True

        if key in ("hero_dps_multiplier_mult", "global_dps_multiplier_mult"):
            n, exact = self._stacks(e, src, src_slot, ctx)
            mode = self._mode(e)
        else:
            n, exact, mode = self._key_count(e, src, src_slot, ctx), True, ("mult" if e.stacks_multiply else "add")
        return self._amplified(e, mode, n, factor, flat), exact

    def _amplified(self, e: Effect, mode: str, n: float, factor, flat) -> float:
        """log10 multiplier of a stacked effect with buffs applied pre- or post-stack."""
        if e.upgrade_id is None or e.upgrade_id in self.data.pre_stack_uids:
            return self._combine(e, mode, n, self._scaled_amount(e, factor, flat))
        base = e.base_amount
        if base is None:
            raise _Unmodelled
        k_all, k_idx = (e.upgrade_id, None), (e.upgrade_id, e.index)
        base += flat.get(k_all, 0.0) + flat.get(k_idx, 0.0)
        f = factor.get(k_all, 1.0) * factor.get(k_idx, 1.0)
        stacked = self._combine(e, mode, n, base)  # log10(1 + bonus)
        if f == 1.0 or stacked == 0.0:
            return stacked
        if stacked > 12:  # 1 + bonus*f ~= bonus*f, avoids float overflow
            return stacked + math.log10(f) if f > 0 else 0.0
        return _log_mult((10.0 ** stacked - 1.0) * f * 100.0)

    def _key_count(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx) -> float:
        """Stack count for keys that carry their own counting rule."""
        key = e.key
        if key == "hero_dps_mult_per_tagged_crusader_mult":
            expr = e.args[1] if len(e.args) > 1 else ""
            return self._count(ctx, src, lambda c: match_tags(expr, c.tags), e.exclude_self)
        # count occupied slots in a relation from the source (optionally by tag)
        rel = e.args[1] if len(e.args) > 1 else "adj"
        slots = self.layout.targets(rel, src_slot)
        if slots is None:
            raise _Unmodelled
        expr = e.args[2] if key.startswith("hero_dps_mult_per_target_tag") and len(e.args) > 2 else None
        return sum(
            1 for s in (slots - {src_slot}) & ctx.occupied
            if expr is None or match_tags(expr, ctx.placement[s].tags)
        )

    def _gold(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx, factor, flat) -> tuple[float, bool]:
        if e.key == "gold_multiplier_mult":
            n, exact = self._stacks(e, src, src_slot, ctx)
            return self._amplified(e, self._mode(e), n, factor, flat), exact
        expr = e.args[1] if len(e.args) > 1 else ""
        n = self._count(ctx, src, lambda c: match_tags(expr, c.tags), e.exclude_self)
        mode = "mult" if e.key.endswith("_mult") else "add"
        return self._amplified(e, mode, n, factor, flat), True

    def _speed(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx, factor, flat) -> "SpeedEffect":
        try:
            amount = self._scaled_amount(e, factor, flat)
        except _Unmodelled:
            amount = None
        try:
            n, _ = self._stacks(e, src, src_slot, ctx)
        except (_Unmodelled, UnsupportedExpr):
            n = 1.0
        return SpeedEffect(src.name, src.id, e.key, e.args, amount, n)

    @staticmethod
    def _mode(e: Effect) -> str:
        if e.amount_func in ("mult", "mult_with_self"):
            return "mult"
        if e.amount_func in ("add", "if", "if_not"):
            return e.amount_func
        if e.amount_func is not None:
            raise _Unmodelled  # set, source_percent_health_add, bespoke...
        return "mult" if e.stacks_multiply else "add"

    @staticmethod
    def _combine(e: Effect, mode: str, n: float, pct: float) -> float:
        if mode == "mult":
            return n * _log_mult(pct)
        if mode == "if":
            return _log_mult(pct) if n > 0 else 0.0
        if mode == "if_not":
            return _log_mult(pct) if n == 0 else 0.0
        return _log_mult(n * pct)

    # -- stack counts ----------------------------------------------------
    @staticmethod
    def _count(ctx: _Ctx, src: Champion, pred, exclude_self: bool) -> int:
        return sum(1 for c in ctx.present if pred(c) and not (exclude_self and c.id == src.id))

    def _stacks(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx) -> tuple[float, bool]:
        """(stack count, exact?) for an effect, or raise _Unmodelled."""
        sf = e.stack_func
        if e.upgrade_id in self.stack_overrides and (e.triggered or (sf and sf not in _FORMATION_STACK_FUNCS)):
            return self.stack_overrides[e.upgrade_id], False  # user assumption
        if sf is None or sf in ("mult", "add"):
            if e.triggered:
                if e.manually_stacked or e.max_stacks is None or e.max_stacks > MAX_ASSUMED_STACKS:
                    raise _Unmodelled
                return e.max_stacks * self.stack_fraction, False  # assumed share of max stacks
            if sf is not None:
                raise _Unmodelled
            return 1.0, not e.timed
        exact = not e.triggered
        if sf == "per_hero_attribute":
            expr = e.raw.get("per_hero_expr")
            if not isinstance(expr, str):
                raise _Unmodelled
            total = 0.0
            for c in ctx.present:
                if e.exclude_self and c.id == src.id:
                    continue
                v = evaluate(expr, c, e.raw)
                total += float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
            return total, exact
        if sf in ("per_crusader", "per_hero", "per_tagged_crusader_mult"):
            tag = e.raw.get("tag")
            return float(self._count(ctx, src, lambda c: not tag or match_tags(str(tag), c.tags), e.exclude_self)), exact
        if sf == "adjacent_champions":
            return float(len(self.layout.slots[src_slot].adj & ctx.occupied)), exact
        if sf == "per_unique_race":
            return float(len({c.species for c in ctx.present if c.species})), exact
        if sf == "per_upgrade_targets":
            data = e.raw.get("stack_func_data") or {}
            uids = set(data.get("upgrade_ids") or ([data["upgrade_id"]] if "upgrade_id" in data else []))
            if not uids:
                raise _Unmodelled
            slots: set[int] = set()
            for other in ctx.effects[src_slot]:
                if other.upgrade_id in uids:
                    for rel in other.targets:
                        t = self.layout.targets(rel, src_slot)
                        if t is None:
                            raise _Unmodelled
                        slots |= t
            return float(len(slots & ctx.occupied)), exact
        raise _Unmodelled

    # -- targeting -------------------------------------------------------
    def _targets_carry(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx) -> bool:
        carry, carry_slot = ctx.carry, ctx.carry_slot
        hit = False
        for rel in e.targets:
            slots = self.layout.targets(rel, src_slot)
            if slots is None:
                raise _Unmodelled
            if isinstance(rel, dict):
                kind = rel.get("type")
                if kind in ("by_tags", "tags") and not match_tags(rel.get("tags", ""), carry.tags):
                    continue
                if kind == "attack_type":
                    raise _Unmodelled
            if carry_slot in slots:
                hit = True
        if e.exclude_self and carry.id == src.id:
            return False
        return hit and self._passes_filters(e, carry)

    @staticmethod
    def _passes_filters(e: Effect, carry: Champion) -> bool:
        for flt in e.filters:
            kind = flt.get("type") if isinstance(flt, dict) else None
            if kind == "by_tags":
                if not match_tags(flt.get("tags", ""), carry.tags):
                    return False
            elif kind == "hero_expr":
                if not evaluate(str(flt.get("hero_expr", "")), carry):
                    return False
            elif kind == "stat_score":
                if "expr" in flt:
                    if not evaluate(str(flt["expr"]), carry):
                        return False
                else:
                    op = _OPS.get(str(flt.get("check")))
                    if op is None:
                        raise _Unmodelled
                    if not op(float(carry.stats.get(str(flt.get("stat", "")).lower(), 0)), float(flt.get("score", 0))):
                        return False
            elif kind == "exclude_heroes":
                if carry.id in {int(i) for i in flt.get("hero_ids", flt.get("ids", []))}:
                    return False
            elif kind == "hero_ids":
                if carry.id not in {int(i) for i in flt.get("hero_ids", [])}:
                    return False
            else:
                raise _Unmodelled
        return True

    def _check_intrinsic(self, e: Effect, src: Champion, src_slot: int, ctx: _Ctx) -> None:
        """Self-only, formation-independent effects can't change the best formation."""
        for rel in e.targets:
            if self.layout.targets(rel, src_slot) != frozenset({src_slot}):
                raise _Unmodelled  # reaches other champions: real buff we don't score
        if src.id == ctx.carry.id and e.stack_func is not None:
            raise _Unmodelled  # carry's own bonus depends on the formation


def static_class(e: Effect, layout: Layout) -> str:
    """Placement-independent classification, for coverage reports.

    Returns one of: damage, amplifier, modifier, param, non_damage, intrinsic,
    approx (modelled, but on a max-stacks/uptime assumption), unmodelled.
    """
    from .catalog import COUNTED_STACK_FUNCS, SUPPORTED_AMOUNT_FUNCS

    key = e.key
    if key in PARAM_KEYS:
        return "param"
    if key in GOLD_KEYS:
        return "gold"
    if key in SPEED_KEYS:
        return "speed"
    if key in NON_DAMAGE_KEYS:
        return "non_damage"
    if key in INTRINSIC_KEYS:
        ok = all(layout.targets(rel, 0) == frozenset({0}) for rel in e.targets)
        return "intrinsic" if ok else "unmodelled"
    if key in MODIFIER_KEYS:
        return "unmodelled" if "active_condition" in e.raw else "modifier"
    if key not in DAMAGE_KEYS and key not in AMPLIFIER_KEYS:
        return "unmodelled"
    if e.base_amount is None and key != "buff_upgrade_add_flat_amount":
        return "unmodelled"
    if e.amount_func not in SUPPORTED_AMOUNT_FUNCS:
        return "unmodelled"
    sf = e.stack_func
    approx = False
    if sf is None or sf in ("mult", "add"):
        if e.triggered:
            if e.manually_stacked or e.max_stacks is None or e.max_stacks > MAX_ASSUMED_STACKS:
                return "unmodelled"
            approx = True
        elif sf is not None:
            return "unmodelled"
        approx = approx or e.timed
    elif sf not in COUNTED_STACK_FUNCS:
        return "unmodelled"
    elif sf == "per_hero_attribute" and not isinstance(e.raw.get("per_hero_expr"), str):
        return "unmodelled"
    if key in DAMAGE_KEYS and key != "global_dps_multiplier_mult":
        for rel in e.targets:
            if layout.targets(rel, 0) is None or (isinstance(rel, dict) and rel.get("type") == "attack_type"):
                return "unmodelled"
    for flt in e.filters:
        kind = flt.get("type") if isinstance(flt, dict) else None
        if kind not in ("by_tags", "hero_expr", "stat_score", "exclude_heroes", "hero_ids"):
            return "unmodelled"
    if approx:
        return "approx"
    return "amplifier" if key in AMPLIFIER_KEYS else "damage"
