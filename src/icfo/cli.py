"""Command-line interface.

    icfo fetch-defs                       # download + cache game definitions
    icfo fetch-user                       # needs IC_USER_ID / IC_HASH env vars
    icfo adventures --search "chicken"    # find adventure / variant ids
    icfo layout 4                         # show layout + parsed variant rules
    icfo optimize 4 --carry Nrakk         # best formation for a carry
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
from .optimize import Infeasible, anneal, candidate_pool, greedy

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
    user = UserState.from_details(client.user_details(Credentials.from_env(), max_age_s=0))
    print(f"user data cached: {len(user.owned)} champions owned")


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


def cmd_optimize(args) -> None:
    data = _load_defs(args)
    user = _load_user(args, data)
    constraints = constraints_for(data, args.adventure)
    carry = data.champion(args.carry)
    if carry.id not in user.owned:
        sys.exit(f"{carry.name} is not owned on this account.")
    if not constraints.allows(carry):
        sys.exit(f"{carry.name} is not allowed in this adventure.")
    pool = candidate_pool(list(data.champions.values()), constraints, user.owned)
    scorer = Scorer(data, constraints.layout)
    if args.stacks:
        scorer.stack_overrides = _load_stacks(args.stacks, data)

    try:
        if args.solver == "greedy":
            result = greedy(scorer, constraints, pool, carry)
        elif args.solver == "cpsat":
            from .optimize.cpsat import solve
            result = solve(scorer, constraints, pool, carry)
        else:
            result = anneal(scorer, constraints, pool, carry, iterations=args.iterations, seed=args.seed)
    except Infeasible as exc:
        sys.exit(f"infeasible: {exc}")

    labels = {i: "XX" for i in constraints.blocked_slots}
    labels.update({s: c.name for s, c in result.placement.items()})
    print(constraints.layout.render(labels, width=12))
    sc = result.score
    print(f"\nformation multiplier on {carry.name}: x10^{sc.log10:.2f}   ({result.evaluations} evaluations)")
    print(f"effect coverage {sc.coverage:.0%} of {sc.modelled_exact + sc.modelled_approx + sum(sc.unmodelled.values())} "
          f"relevant effects; {sc.approx_share:.0%} of the score assumes max stacks / full uptime")
    by_source: Counter = Counter()
    approx_by_source: Counter = Counter()
    for contrib in result.score.contributions:
        by_source[contrib.source] += contrib.log10
        if not contrib.exact:
            approx_by_source[contrib.source] += contrib.log10
    print("\ncontributions (log10):")
    for name, val in by_source.most_common():
        note = f"   ({approx_by_source[name]:+.2f} assumed)" if approx_by_source[name] else ""
        print(f"  {name:<20} {val:+8.2f}{note}")
    if result.score.unmodelled_by_hero:
        print("\nunmodelled effects (result may under/over-rate these champions):")
        for name, keys in sorted(result.score.unmodelled_by_hero.items()):
            print(f"  {name:<20} " + ", ".join(f"{k}×{n}" for k, n in keys.most_common(4)))
    for w in constraints.warnings:
        print("warning:", w)


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

    s = sub.add_parser("optimize")
    s.add_argument("adventure", type=int)
    s.add_argument("--carry", required=True, help="champion name or id")
    s.add_argument("--solver", choices=["anneal", "greedy", "cpsat"], default="anneal")
    s.add_argument("--iterations", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--user", help="getuserdetails JSON file")
    s.add_argument("--all-champions", action="store_true", help="pretend every champion is owned")
    s.add_argument("--stacks", help='JSON stack assumptions, e.g. {"Walnut:Pen-Paws": 20, "11654": 60}')
    s.set_defaults(func=cmd_optimize)

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
