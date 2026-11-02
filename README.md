# ic-formation-optimizer

Finds the best formation for a chosen champion (the "carry") in any Idle Champions
adventure or variant. It uses your real account data and the game's own definitions.

## How it works

```
fetch/      game API client (same endpoints ic.byteglow.com uses) + on-disk cache
model/      definitions → champions, upgrades, effects, layouts, variant rules
engine/     scores a placement: log10 of the damage multiplier reaching the carry
optimize/   greedy, simulated annealing, CP-SAT (exact on a linear surrogate), brute force
cli.py      `icfo` command
```

**Scoring.** IC damage is mostly a product of independent multipliers, so the engine
works in log10 space. `x10^55.1` means the formation's buffs multiply the carry's damage
by 10^55.1. The carry's base damage is a constant and is left out, so scores compare
formations **for the same carry**.

**Effect rules** (taken from the game's own key definitions where possible):

- *Amounts held elsewhere.* `pre_stack_amount,60` next to `hero_dps_multiplier_mult,0`
  plus `amount_expr: upgrade_amount(U,0)` means the real amount is 60. These formulas
  are resolved when definitions load.
- *Amplifiers.* Plain `buff_upgrade`s on the same upgrade **add**. `_mult` and
  `_multiplicative` variants **multiply**:
  `factor = (1 + Σ additive) × Π multiplicative`. Amplifiers of amplifiers are resolved
  to a fixed point, and `buff_upgrade,A,U,i` targets only effect *i* of upgrade *U*.
- *Pre- vs post-stack.* A buff scales an effect's total after stacking (the game marks
  `buff_upgrade` "post stack"). The exception is an upgrade that holds a `pre_stack`
  parameter, where it scales the per-stack amount ("buffs apply to the pre-stack value").
- *Stack counts from the formation:* champions matching a tag or hero expression,
  adjacent champions, unique species, champions hit by another upgrade.
- *Runtime stacks* (kills, attacks, timers) are assumed to be at `max_stacks` and marked
  **approximate**. Manual or champion-specific stacking, and caps above 500, are not
  guessed: they're reported as unmodelled unless you supply a stack count with `--stacks`.
- *Upgrade modifiers* (`change_upgrade_data`, `change_upgrade_targets`,
  `add_target_to_upgrade`) patch the targeted upgrade's data before scoring.

**Coverage.** Each effect on a placed champion is either scored exactly, scored on an
assumption, safely ignored (bookkeeping, non-damage, or self-only and
formation-independent), or **unmodelled**. Results print coverage, the share of the
score that rests on assumptions (per champion too), and the unmodelled effects.
Currently about 85% of all champion effects are scored exactly, 8% are safely ignored,
and 7% are unmodelled, mostly champion-specific mechanics. `icfo coverage` ranks what's left.

**Search.** With the carry fixed to slot *k*, most buffs depend only on which
champion sits in which slot, so the objective is close to linear in booleans
`x[champion, slot]`. `--solver cpsat` builds that linear surrogate from the exact
scorer and solves it to optimality for each *k* with OR-Tools. It then re-scores
with the exact engine and polishes the result with annealing. `anneal` (the default)
needs no extra dependency, and `greedy` is the fastest. Tests check all of them
against brute force.

**Variants.** These restrictions become constraints:
- allowed or banned champions by id, tag, seat, ability score, or hero expression
  (e.g. ``(!HasTag(`dps`)&&hero_seat_id!=10)||...``, parsed safely without `eval`)
- required champions, blocked (escort) slots, per-tag limits, forced key-champion slots
- layout overrides

Any restriction that isn't modelled is printed as a warning, never silently ignored.

## Usage

```bash
pip install -e '.[solver,dev]'      # 'solver' adds OR-Tools for --solver cpsat

icfo fetch-defs                     # ~20 MB, cached in .icfo-cache/
export IC_USER_ID=... IC_HASH=...   # from the game's support/settings screen; keep private
icfo fetch-user

icfo adventures --search chicken    # find adventure / variant ids
icfo layout 469                     # layout picture + parsed restrictions
icfo optimize 469 --carry Havilar --solver cpsat
icfo optimize 4 --carry Torogar --all-champions   # theory-craft: pretend you own everyone
icfo coverage
icfo stacks Walnut "Commodore Krux"  # runtime-stacked effects and their upgrade ids
icfo optimize 469 --carry Havilar --stacks my-stacks.json
```

A stacks file sets your own assumption for runtime-stacked effects, keyed by upgrade
id or `"Champion:Upgrade name"`:

```json
{"Walnut:Pen-Paws": 10, "Commodore Krux:Battle Stations!": 30}
```

Example output (`icfo optimize 469 --carry Havilar --all-champions --solver cpsat`):

```
[ Farideh  ]              [  Regis   ]
                                       [   Fen    ]
[    ·     ]              [ Penelope ]              [Commodore ]
             [  Walnut  ]              [ Havilar  ]
                          [    ·     ]
             [  Mehen   ]
back                                                        front

formation multiplier on Havilar: x10^150.31   (4372 evaluations)
effect coverage 92% of 636 relevant effects; 54% of the score assumes max stacks / full uptime

contributions (log10):
  Commodore Krux         +36.15   (+31.31 assumed)
  Walnut                 +30.77   (+25.93 assumed)
  Regis                  +29.52   (+24.55 assumed)
  Havilar                +24.64
  ...
```

## Tests

```bash
pytest                                        # synthetic fixtures, no network
ICFO_DEFS=.icfo-cache/definitions.json pytest # + smoke test over every live adventure
```

## Assumptions to verify in game

- Column 0 is the front. `next_col`/`ahead` point toward the front and
  `prev_col`/`behind` toward the back. `ahead`/`behind` mean adjacent slots in that column.
- Plain `buff_upgrade`s add and `_mult`/`_multiplicative` ones multiply. This is
  inferred from the game defining explicit multiplicative and "additively increases" keys.
- Buffs apply after stacking unless the upgrade has a `pre_stack` parameter.
- Runtime stacks reach `max_stacks`. That is often true late in an area, but not at
  its start.
- Within a single amplifier, `stacks_multiply` / `amount_func: mult` means
  `(1+a)^n`, otherwise `n·a`.
- Specializations default to the first option in each choice group, and all upgrades
  are treated as unlocked (max level) unless levels are supplied.

## Roadmap

1. **Champion-specific handlers** for the mechanics generic rules can't express
   (`expression_on_trigger`, `broadcast_on_trigger`, `*_stacks` keys, Aerois Synergy,
   Mithral Hall), starting with the supports that show up most in optimized formations.
   Also: effects conditional on specific feats, and `affected_by_upgrade` filters.
2. **Account fidelity.** Read champion levels, equipment, feats, legendaries and
   familiars from `getuserdetails`, and let the optimizer choose specializations and feats
   (alternating with placement).
3. **Comparing carries.** Model base damage (level, gear) so different carries can be
   ranked, not just formations for one carry.
4. **Pairwise CP-SAT terms.** Add product variables for interactions the surrogate
   misses, such as an amplifier boosting a positional buff.
5. **More objectives.** Survivability constraints, gold find, and gem-farm speed
   (areas per hour).
6. **Calibration.** Golden tests comparing predicted multipliers with in-game damage readouts.
