# ic-formation-optimizer

Finds the best formation in any Idle Champions adventure or variant, or in all of
them at once, for three goals:

| Mode | Maximises | How |
|---|---|---|
| `push` | the highest reachable area | damage of the best carry (base attack × formation multiplier) |
| `favor` | favor per run | `0.315 × (log10 damage + log10 gold find)`, derived from the game's growth constants |
| `gems` | boss areas per hour × gems per boss, to a reset area | Briv jumps, Thellora's start skip, spawn/quest/time-scale effects, Ellywick's gem cards, plus a damage floor |

It uses your real account data (owned champions, specializations, feats, gear) and
the game's own definitions. It picks the carry automatically.

## How it works

```
fetch/      game API client (same endpoints ic.byteglow.com uses) + on-disk cache
model/      definitions → champions, upgrades, effects, layouts, variant rules
engine/     scores a placement: log10 of the damage multiplier reaching the carry
optimize/   greedy, simulated annealing, CP-SAT (exact on a linear surrogate), brute force
objectives.py  push / favor / gem objectives, progression model, calibration
optimize/planner.py  carry shortlist → pool pruning → search (fast / thorough)
optimize/sweep.py    every adventure & variant, grouped by identical rules, in parallel
cli.py      `icfo` command
```

**Progression model** (all from `monster_base_stats` in the game rules). Monster health
grows ×2.031 per area (×3.031 from 2001, ×4.531 from 2251), so +1 log10 of damage is
worth about 3.25 areas. Monster gold is a fixed share of health. Favor grows ×1.25 per
×2.031 of gold (`currency_growth_rate` / `gold_growth_rate` from your account), so
favor ∝ gold^0.315. Since the area you stall at is where damage meets health,
log10(favor) ≈ 0.315·(log10 damage + log10 gold find) + const.

**Calibration.** The model leaves out champion levels, legendaries, blessings, patron
perks and potions. Those mostly don't depend on the formation, so they don't change
rankings, but they do affect absolute areas. `icfo calibrate` anchors model damage to
your furthest instance (its formation and highest area) to estimate reachable areas.
That instance may not be at its wall, so the estimates are lower bounds.

**Gems model.** seconds per area = kills × spawn interval ÷ (spawn speed × kill credit)
÷ game speed + transition. Briv: jump size and chance from his slot-4 item
(`0.25 + 0.25·m`, halved until ≤1, the method ic.byteglow.com uses), and jumps per run
from Sprint stacks (−4% per jump while ≥50). Thellora: start skip from your stats.
Gems per boss: +% boss-gem effects add. Ellywick holds 5 cards; expected gem cards =
5 × gem odds (20% base, 40% with All That Sparkles, all 5 with the "Gem" feat), each worth
her gem amount buffed by her gear and feats (capped at +100% per card). Base gems per
boss aren't in the game data, so gem output is relative. `--skipped-bosses-dont-pay`
counts only boss areas actually completed. The
formation must reach the reset area, using the calibrated damage. Base timings are
flags (`--kills-per-area`, `--spawn-interval`, `--transition-time`, `--briv-stacks`);
only relative values matter for ranking.

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
icfo calibrate                      # anchor model damage to your progress

icfo optimize 469                   # push; carry chosen automatically
icfo optimize 469 --mode favor
icfo optimize 469 --mode gems --reset-area 900
icfo optimize 469 --carry Havilar --profile fast
icfo optimize 4 --all-champions     # theory-craft: own everyone, default specs, no gear

icfo sweep --out sweep.csv          # all ~1,985 adventures × 3 modes (~1 h on 4 cores)
icfo sweep --modes gems --adventures 1,4,469 --profile thorough
icfo coverage
icfo stacks Walnut "Commodore Krux"  # runtime-stacked effects and their upgrade ids
icfo optimize 469 --carry Havilar --stacks my-stacks.json
```

A stacks file sets your own assumption for runtime-stacked effects, keyed by upgrade
id or `"Champion:Upgrade name"`:

```json
{"Walnut:Pen-Paws": 10, "Commodore Krux:Battle Stations!": 30}
```

`sweep.csv` has one row per adventure × mode: carry, formation (slot:champion),
objective value, estimated area / favor / areas-per-hour, coverage, the share of the
score that rests on assumptions, and any variant rules that aren't modelled. Adventures
with identical rules for your roster are solved once. Within a group, each mode's
formation is also scored under the other objectives and the best one is kept.

Stack assumptions: `--stack-assumption 0.5` scores runtime stacks at half of max (0
ignores them), and `--stacks file.json` sets them per upgrade.

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

## Account data

`icfo fetch-user` reads `IC_USER_ID` / `IC_HASH` and calls only read endpoints. The hash
gives full control of the account, so keep it private. The cached `user_details.json` also
contains personal data (email, linked accounts). Only the fields listed in
`model/user.py` are parsed, and `.icfo-cache/` and sweep outputs are git-ignored.

## Roadmap

1. **Champion-specific handlers** for the mechanics generic rules can't express
   (`expression_on_trigger`, `broadcast_on_trigger`, `*_stacks` keys, Aerois Synergy,
   Mithral Hall), starting with the supports that show up most in optimized formations.
   Also: effects conditional on specific feats, and `affected_by_upgrade` filters.
2. **Account fidelity.** Specs, feats and gear are applied, and the planner picks the
   best specialization per formation and objective. Still missing: feat selection,
   legendaries, familiars, blessings and patron perks.
3. **Better carry comparison.** Add levels and ultimates to the base-attack model.
4. **Pairwise CP-SAT terms.** Add product variables for interactions the surrogate
   misses, such as an amplifier boosting a positional buff.
5. **Objective fidelity.** Survivability, Briv stack regeneration per run, and
   per-adventure quest sizes and timings for the gem model.
6. **Calibration.** Golden tests comparing predicted multipliers with in-game damage readouts.
