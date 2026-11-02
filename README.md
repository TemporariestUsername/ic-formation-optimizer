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

**Coverage.** Every effect on a placed champion is classified as *modelled*,
*non-damage* (healing, gold, ...) or *unmodelled*. Results print a coverage
percentage and list the unmodelled effects per champion, so you know where the
number may be wrong. `icfo coverage` ranks the effect types still to implement.

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
```

Example output (`icfo optimize 469 --carry Havilar --all-champions --solver cpsat`):

```
[    ·     ]              [ Havilar  ]
                                       [   Tess   ]
[Sgt. Knox ]              [   Melf   ]              [    ·     ]
             [ Farideh  ]              [  Krydle  ]
                          [  Mehen   ]
             [  Skylla  ]
back                                                        front

formation multiplier on Havilar: x10^64.60   (effect coverage 66%, 4334 evaluations)
```

## Tests

```bash
pytest                                        # synthetic fixtures, no network
ICFO_DEFS=.icfo-cache/definitions.json pytest # + smoke test over every live adventure
```

## Assumptions to verify in game

- Column 0 is the front. `next_col`/`ahead` point toward the front and
  `prev_col`/`behind` toward the back. `ahead`/`behind` mean adjacent slots in that column.
- `buff_upgrade` amplifiers add together for the same upgrade; per-tag `..._mult`
  amplifiers multiply.
- Specializations default to the first option in each choice group, and all upgrades
  are treated as unlocked (max level) unless levels are supplied.

## Roadmap

1. **Effect coverage.** Implement the top unmodelled keys from `icfo coverage`
   (`pre_stack*`, `change_upgrade_data`, `buff_upgrade_add_flat_amount`, ...) and
   bespoke handlers for widely used supports.
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
