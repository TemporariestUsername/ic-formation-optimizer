"""Command-line interface.

    icfo fetch-defs                       # download + cache game definitions
    icfo fetch-user                       # needs IC_USER_ID / IC_HASH env vars
    icfo adventures --search "chicken"    # find adventure / variant ids
    icfo layout 4                         # show layout + parsed variant rules
    icfo optimize 4                       # best push formation, carry chosen automatically
    icfo optimize 4 --mode favor          # ... for favor farming
    icfo optimize 4 --mode gems --reset-area 900
    icfo sweep --modes push,favor,gems    # every adventure and variant -> sweep.csv
    icfo calibrate                        # how damage is anchored to your progress
    icfo coverage                         # which effect types are still unmodelled
    icfo stacks Walnut Krux               # runtime-stacked effects you can set with --stacks
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .engine.score import Scorer, static_class
from .fetch import Cache, Credentials, IdleChampionsClient
from .model import GameData, UserState, constraints_for
from .model.bonuses import PATRONS, Bonuses, patron_id
from .objectives import GemParams, ObjectiveScorer, calibrate
from .optimize import Infeasible, anneal, candidate_pool, greedy
from .optimize.planner import TIER_LABELS, plan_tiers
from .optimize.sweep import make_objective, run_sweep

def _load_defs(args) -> GameData:
    if args.defs:
        with open(args.defs, encoding="utf-8") as f:
            return GameData(json.load(f))
    return GameData(IdleChampionsClient(Cache(args.cache)).definitions())


def _load_user(args, data: GameData) -> UserState:
    if args.all_champions:
        return UserState.everything(set(data.champions))
    if args.user:
        with open(args.user, encoding="utf-8") as f:
            return UserState.from_details(json.load(f))
    cached = Cache(args.cache).load("user_details")
    if cached is None:
        sys.exit("No user data: run `icfo fetch-user`, pass --user FILE, or use --all-champions.")
    return UserState.from_details(cached)


def _load_stacks(path: str, data: GameData) -> dict[int, float]:
    """Stack assumptions file: {"<upgrade id>": stacks} or {"Champion:Upgrade name": stacks}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, float] = {}
    for key, value in raw.items():
        if str(key).isdigit():
            out[int(key)] = float(value)
            continue
        champ_name, _, upgrade_name = str(key).partition(":")
        champ = data.champion(champ_name.strip())
        matches = [u for u in champ.upgrades if (u.name or u.specialization_name or "").casefold() == upgrade_name.strip().casefold()]
        if not matches:
            raise KeyError(f"{champ.name} has no upgrade named {upgrade_name.strip()!r}")
        for u in matches:
            out[u.id] = float(value)
    return out


def cmd_stacks(args) -> None:
    """List runtime-stacked effects for champions, to fill a --stacks file."""
    data = _load_defs(args)
    for name in args.champions:
        champ = data.champion(name)
        print(f"{champ.name}:")
        for u in champ.upgrades:
            for e in u.effects:
                if e.triggered or (e.stack_func and e.stack_func not in ("per_hero_attribute", "per_crusader", "per_hero",
                                                                          "per_tagged_crusader_mult", "adjacent_champions",
                                                                          "per_unique_race", "per_upgrade_targets")):
                    how = e.raw.get("stacks_on_trigger") or e.stack_func or "manual"
                    cap = f"max {e.max_stacks:g}" if e.max_stacks else "no cap"
                    print(f"  {u.id:>6}  {(u.name or u.specialization_name or '?')[:28]:<28} {e.key:<28} {str(how)[:28]:<28} {cap}")
                    break


def cmd_fetch_defs(args) -> None:
    data = IdleChampionsClient(Cache(args.cache)).definitions(refresh=args.refresh)
    print(f"definitions cached: {len(data.get('hero_defines', []))} champions, "
          f"{len(data.get('adventure_defines', []))} adventures")


def cmd_fetch_user(args) -> None:
    client = IdleChampionsClient(Cache(args.cache))
    creds = Credentials.from_env()
    user = UserState.from_details(client.user_details(creds, max_age_s=0))
    client.campaign_details(creds, max_age_s=0)
    print(f"user data cached: {len(user.owned)} champions owned; campaign details (blessings) cached")


def cmd_adventures(args) -> None:
    data = _load_defs(args)
    needle = (args.search or "").casefold()
    for adv in sorted(data.adventures.values(), key=lambda a: a.id):
        if needle and needle not in adv.name.casefold() and needle not in adv.restrictions_text.casefold():
            continue
        tag = f"variant of {adv.variant_of}" if adv.variant_of else "base"
        print(f"{adv.id:>6}  {adv.name[:40]:<40}  {tag:<16} {' '.join(adv.restrictions_text.split())[:70]}")


def cmd_layout(args) -> None:
    data = _load_defs(args)
    c = constraints_for(data, args.adventure)
    adv = data.adventures[args.adventure]
    print(f"{adv.name}: {c.layout.name} ({len(c.layout)} slots)\n")
    labels = {i: ("XX" if i in c.blocked_slots else str(i)) for i in range(len(c.layout))}
    print(c.layout.render(labels))
    print(f"\nrestrictions: {adv.restrictions_text}")
    allowed = [ch.name for ch in data.champions.values() if c.allows(ch)]
    print(f"allowed champions: {len(allowed)}/{len(data.champions)}")
    for r in c.allow_rules:
        print("  only allow:", r.description)
    for r in c.ban_rules:
        print("  disallow:  ", r.description)
    for slot, r in c.forced_slot_rules.items():
        print(f"  slot {slot} must hold:", r.description)
    if c.required_ids:
        print("required:", ", ".join(data.champions[i].name for i in c.required_ids if i in data.champions))
    for w in c.warnings:
        print("warning:", w)


def _bonuses(args, data: GameData, user: UserState) -> Bonuses | None:
    if not user.from_account or getattr(args, "no_bonuses", False):
        return None
    details = Cache(args.cache).load("campaign_details")
    if details is None:
        print("note: no campaign details cached (run fetch-user); blessings not applied", file=sys.stderr)
    b = Bonuses(data, user, details)
    return b if b.available else None


def _scorer_factory(data: GameData, user: UserState, args):
    stacks = _load_stacks(args.stacks, data) if getattr(args, "stacks", None) else {}
    bonuses = _bonuses(args, data, user)
    patron = patron_id(getattr(args, "patron", None))

    def make(layout, adventure_id: int | None = None, patron_override: int | None = None):
        scorer = Scorer(data, layout)
        if user.from_account:
            scorer.apply_account(user)
        scorer.stack_overrides = dict(stacks)
        scorer.stack_fraction = args.stack_assumption
        if getattr(args, "run_area", None):
            scorer.run_area = float(args.run_area)
        if bonuses is not None:
            p = patron if patron_override is None else patron_override
            scorer.set_bonuses(bonuses.effects_for(adventure_id, p), p)
        return scorer

    make.bonuses = bonuses
    make.patron = patron
    return make


def _gem_params(args, user: UserState, adventure_id: int | None = None) -> GemParams:
    reset = args.reset_area
    if reset is None and adventure_id is not None:
        reset = next((i.reset_area for i in user.instances if i.adventure_id == adventure_id and i.reset_area), None)
    return GemParams(
        reset_area=reset or 1000,
        kills_per_area=args.kills_per_area,
        spawn_interval_s=args.spawn_interval,
        transition_s=args.transition_time,
        briv_stacks=args.briv_stacks,
        skipped_bosses_pay=not args.skipped_bosses_dont_pay,
    )


def _calibration(args, data, user, make):
    if getattr(args, "no_calibration", False) or not user.from_account:
        return None
    return calibrate(data, user, make)


def cmd_calibrate(args) -> None:
    data = _load_defs(args)
    user = _load_user(args, data)
    cal = _calibration(args, data, user, _scorer_factory(data, user, args))
    if cal is None:
        sys.exit("No calibration: needs account data with an instance past area 10.")
    adv = data.adventures[cal.adventure_id]
    print(f"anchored to {adv.name} (id {cal.adventure_id}) at area {cal.area}, carry {cal.carry}")
    print(f"offset: true log10 DPS ≈ model + {cal.offset:.2f}   ({cal.note})")


def cmd_optimize(args) -> None:
    data = _load_defs(args)
    user = _load_user(args, data)
    make = _scorer_factory(data, user, args)
    constraints = constraints_for(data, args.adventure, make.patron)
    carry = None
    if args.carry and args.carry != "auto":
        carry = data.champion(args.carry)
        if carry.id not in user.owned:
            sys.exit(f"{carry.name} is not owned on this account.")
        if not constraints.allows(carry):
            sys.exit(f"{carry.name} is not allowed in this adventure.")
    pool = candidate_pool(list(data.champions.values()), constraints, user.owned)
    cal = _calibration(args, data, user, make)
    objective = make_objective(args.mode, data, user, cal, _gem_params(args, user, args.adventure))
    oscorer = ObjectiveScorer(make(constraints.layout, args.adventure), objective)

    try:
        if args.solver == "planner":
            tier_names = [t.strip() for t in args.tiers.split(",") if t.strip()]
            tiers = plan_tiers(oscorer, constraints, pool, user, carry, args.profile, tier_names, seed=args.seed)
            base_tier = tiers[tier_names[0]]
            p = base_tier.plan
            for h, f in base_tier.feats.items():  # report the first tier with its own feats
                oscorer.scorer.feats[h] = f
                oscorer.scorer.invalidate(h)
            carry, placement, evals, tried = p.carry, p.placement, p.evaluations, p.carries_tried
            sc = oscorer.score(placement, carry.id)
            spec_changes = p.spec_changes
        else:
            if carry is None:
                sys.exit("--solver greedy/anneal/cpsat need --carry; use the default planner for auto carry")
            if args.solver == "greedy":
                r = greedy(oscorer, constraints, pool, carry)
            elif args.solver == "cpsat":
                from .optimize.cpsat import solve
                r = solve(oscorer, constraints, pool, carry)
            else:
                r = anneal(oscorer, constraints, pool, carry, iterations=args.iterations, seed=args.seed)
            placement, sc, evals, tried = r.placement, r.score, r.evaluations, [carry.name]
            spec_changes = []
            tiers, tier_names = {}, []
    except Infeasible as exc:
        sys.exit(f"infeasible: {exc}")

    adv = data.adventures[args.adventure]
    labels = {i: "XX" for i in constraints.blocked_slots}
    labels.update({s: c.name for s, c in placement.items()})
    if tiers:
        print(f"== {TIER_LABELS[tier_names[0]]} ==")
    print(f"{adv.name} (id {adv.id}) - mode: {args.mode}, carry: {carry.name}"
          + (f"  (auto, compared: {', '.join(tried)})" if len(tried) > 1 else ""))
    print(f"account: {'yours (specs, feats, gear applied)' if user.from_account else 'theory (owns everything, default specs)'}")
    if make.bonuses is not None:
        applied = make.bonuses.for_adventure(args.adventure, make.patron)
        kinds = Counter(b.kind for b in applied)
        print("bonuses: " + ", ".join(f"{n} {k}s" for k, n in sorted(kinds.items()))
              + (f" (patron: {PATRONS[make.patron]})" if make.patron else " (no patron)"))
    print()
    print(constraints.layout.render(labels, width=12))
    if spec_changes:
        print("\nspecialization changes for this formation:")
        for name, before, after in spec_changes:
            print(f"  {name:<18} {before} -> {after}")
    print()
    for k, v in objective.summary(sc, carry, placement).items():
        print(f"  {k:<18} {v}")
    dmg = sc.damage_log10 if sc.damage_log10 is not None else sc.log10
    print(f"\nformation damage multiplier on {carry.name}: x10^{dmg:.2f}; gold find x10^{sc.gold_log10:.2f}   ({evals} evaluations)")
    print(f"effect coverage {sc.coverage:.0%} of {sc.modelled_exact + sc.modelled_approx + sum(sc.unmodelled.values())} "
          f"relevant effects; {sc.approx_share:.0%} of the score assumes max stacks / full uptime")
    if cal is not None:
        print(f"area estimates calibrated on {data.adventures[cal.adventure_id].name} @ {cal.area} ({cal.note})")
    by_source: Counter = Counter()
    approx_by_source: Counter = Counter()
    gold_by_source: Counter = Counter()
    for contrib in sc.contributions:
        if contrib.channel == "gold":
            gold_by_source[contrib.source] += contrib.log10
            continue
        by_source[contrib.source] += contrib.log10
        if not contrib.exact:
            approx_by_source[contrib.source] += contrib.log10
    print("\ndamage contributions (log10):")
    for name, val in by_source.most_common():
        note = f"   ({approx_by_source[name]:+.2f} assumed)" if approx_by_source[name] else ""
        print(f"  {name:<20} {val:+8.2f}{note}")
    if gold_by_source and args.mode != "push":
        print("gold find contributions (log10):")
        for name, val in gold_by_source.most_common():
            print(f"  {name:<20} {val:+8.2f}")
    if sc.speed and args.mode == "gems":
        print("speed effects used:")
        for e in sc.speed:
            print(f"  {e.source:<20} {e.key} {'' if e.amount is None else f'{e.amount:g}'}")
    if sc.bonus_unmodelled:
        print("\nblessings/perks not scored (monster- or area-specific, or not damage):")
        print("  " + ", ".join(sorted(sc.bonus_unmodelled)))
    if sc.unmodelled_by_hero:
        print("\nunmodelled effects (result may under/over-rate these champions):")
        for name, keys in sorted(sc.unmodelled_by_hero.items()):
            print(f"  {name:<20} " + ", ".join(f"{k}×{n}" for k, n in keys.most_common(4)))
    for w in constraints.warnings:
        print("warning:", w)
    _print_tiers(data, constraints, tiers, tier_names, objective, user)


def _print_tiers(data, constraints, tiers, tier_names, objective, user) -> None:
    if not tiers:
        return
    first = tiers[tier_names[0]]
    for name in tier_names:
        t = tiers[name]
        current_feats = {h: tuple(user.feats.get(h, ())) for h in t.feats}
        print(f"\n== {TIER_LABELS[name]} ==")
        print(f"  objective {t.value:.2f}" + (f"  ({t.value - first.value:+.2f} vs base)" if name != tier_names[0] else ""))
        if name != tier_names[0] and t.plan.placement != first.plan.placement:
            labels = {i: "XX" for i in constraints.blocked_slots}
            labels.update({sl: c.name for sl, c in t.plan.placement.items()})
            print(constraints.layout.render(labels, width=12))
        for h, feats in sorted(t.feats.items()):
            if feats != current_feats.get(h, ()):
                names = ", ".join(data.feats[f].name for f in feats if f in data.feats) or "(none)"
                print(f"  feats {data.champions[h].name:<16} {names}")
        if t.purchases:
            print("  to acquire:")
            for champ, feat in t.purchases:
                print(f"    {champ:<16} {feat.name:<28} {feat.describe_source()}")
            if t.gem_cost:
                print(f"    gem total: {t.gem_cost:,}")
        summary = objective.summary(t.plan.result, t.plan.carry, t.plan.placement)
        keys = [k for k in ("est_area", "log10_damage", "log10_favor_rel", "rel_gems_per_hour", "gem_multiplier") if k in summary]
        print("  " + ", ".join(f"{k} {summary[k]}" for k in keys))


def cmd_sweep(args) -> None:
    data = _load_defs(args)
    user = _load_user(args, data)
    make = _scorer_factory(data, user, args)
    cal = _calibration(args, data, user, make)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ids = [int(x) for x in args.adventures.split(",")] if args.adventures else None
    if args.limit and ids is None:
        ids = sorted(data.adventures)[: args.limit]

    def progress(done, total):
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total} solved", flush=True)

    summary = run_sweep(
        data, user, modes, calibration=cal, out_csv=args.out, out_json=args.json,
        profile=args.profile, workers=args.workers, adventure_ids=ids,
        use_account=user.from_account, gem_params=_gem_params(args, user), progress=progress,
        stack_fraction=args.stack_assumption,
        stack_overrides=_load_stacks(args.stacks, data) if args.stacks else None,
        tiers=[t.strip() for t in args.tiers.split(",") if t.strip()],
        bonuses=make.bonuses, patron=make.patron, run_area=args.run_area,
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}" + (f" and {args.json}" if args.json else ""))


def cmd_coverage(args) -> None:
    data = _load_defs(args)
    layout = data.layout_for(1)
    classes: Counter = Counter()
    unmodelled: Counter = Counter()
    champs_by_key: dict[str, set[str]] = {}
    for champ in data.champions.values():
        for e in champ.effects():
            cls = static_class(e, layout)
            classes[cls] += 1
            if cls == "unmodelled":
                unmodelled[e.key] += 1
                champs_by_key.setdefault(e.key, set()).add(champ.name)
    total = sum(classes.values())
    print(f"effects on all champions (default specs, max level): {total}")
    groups = [
        ("scored exactly", ("damage", "amplifier", "modifier")),
        ("scored, assumes max stacks/uptime", ("approx",)),
        ("gold find / speed (used by favor and gem modes)", ("gold", "speed")),
        ("safely ignored (bookkeeping, non-damage, self-only)", ("param", "non_damage", "intrinsic")),
        ("unmodelled", ("unmodelled",)),
    ]
    for label, members in groups:
        n = sum(classes[m] for m in members)
        print(f"  {n:>6} ({n / total:5.1%})  {label}")
    print(f"\ntop {args.top} unmodelled effect keys (by champions affected):")
    ranked = sorted(champs_by_key, key=lambda k: (-len(champs_by_key[k]), -unmodelled[k]))
    for k in ranked[: args.top]:
        print(f"  {len(champs_by_key[k]):>3} champions  {unmodelled[k]:>4} effects  {k}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="icfo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=Path(".icfo-cache"), help="cache directory")
    p.add_argument("--defs", help="load definitions from a JSON file instead of the API/cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch-defs")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_fetch_defs)

    sub.add_parser("fetch-user").set_defaults(func=cmd_fetch_user)

    s = sub.add_parser("adventures")
    s.add_argument("--search")
    s.set_defaults(func=cmd_adventures)

    s = sub.add_parser("layout")
    s.add_argument("adventure", type=int)
    s.set_defaults(func=cmd_layout)

    def account_args(sp):
        sp.add_argument("--user", help="getuserdetails JSON file (default: cached fetch-user data)")
        sp.add_argument("--all-champions", action="store_true",
                        help="theory-craft: own every champion, default specs, no gear")
        sp.add_argument("--stacks", help='JSON stack assumptions, e.g. {"Walnut:Pen-Paws": 20, "11654": 60}')
        sp.add_argument("--no-calibration", action="store_true", help="don't anchor damage to your progress")
        sp.add_argument("--no-bonuses", action="store_true", help="ignore blessings, patron perks and favor gold")
        sp.add_argument("--run-area", type=float,
                        help="area a run reaches, for caps like Rust's Riches (default: your deepest instance)")
        sp.add_argument("--patron", help="play under a patron: Mirt, Vajra, Strahd, Zariel, Elminster")
        sp.add_argument("--stack-assumption", type=float, default=1.0,
                        help="share of max stacks assumed for kill/attack/timer stacks (1 = full, 0 = ignore)")

    def mode_args(sp):
        sp.add_argument("--profile", choices=["fast", "thorough"], default="thorough" if sp.prog.endswith("optimize") else "fast")
        sp.add_argument("--reset-area", type=int, help="gems: Modron reset area (default: your Modron setting or 1000)")
        sp.add_argument("--kills-per-area", type=float, default=25.0)
        sp.add_argument("--spawn-interval", type=float, default=1.0, help="gems: seconds between spawns at base speed")
        sp.add_argument("--transition-time", type=float, default=4.0, help="gems: seconds per area transition")
        sp.add_argument("--briv-stacks", type=float, help="gems: Sprint stacks per run (default: your current stacks)")
        sp.add_argument("--skipped-bosses-dont-pay", action="store_true",
                        help="gems: only completed boss areas give gems (default counts skipped ones too)")

    s = sub.add_parser("optimize", help="best formation for one adventure")
    s.add_argument("adventure", type=int)
    s.add_argument("--mode", choices=["push", "favor", "gems"], default="push")
    s.add_argument("--carry", default="auto", help="champion name/id, or auto (default)")
    s.add_argument("--solver", choices=["planner", "anneal", "greedy", "cpsat"], default="planner")
    s.add_argument("--tiers", default="base,gems,full",
                   help="optimums to compute: base (owned feats), gems (+gem-priced feats), full (+everything)")
    s.add_argument("--iterations", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    account_args(s)
    mode_args(s)
    s.set_defaults(func=cmd_optimize)

    s = sub.add_parser("sweep", help="optimise every adventure/variant")
    s.add_argument("--modes", default="push,favor,gems")
    s.add_argument("--out", default="sweep.csv")
    s.add_argument("--json")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--adventures", help="comma-separated ids (default: all)")
    s.add_argument("--limit", type=int, help="only the first N adventures (for a quick look)")
    s.add_argument("--tiers", default="base,gems,full", help="optimums per row: base, gems, full")
    account_args(s)
    mode_args(s)
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("calibrate", help="show how model damage is anchored to your progress")
    account_args(s)
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("stacks", help="list runtime-stacked effects (for --stacks)")
    s.add_argument("champions", nargs="+")
    s.set_defaults(func=cmd_stacks)

    s = sub.add_parser("coverage")
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(func=cmd_coverage)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except KeyError as exc:  # unknown champion / adventure / layout
        sys.exit(f"error: {exc.args[0] if exc.args else exc}")
    except BrokenPipeError:  # e.g. piped into `head`
        pass


if __name__ == "__main__":
    main()
