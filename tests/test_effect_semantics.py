"""Engine semantics, each checked against a hand-computed multiplier."""

import math

import pytest

from icfo.engine import Scorer
from icfo.engine.score import static_class
from icfo.model import GameData

from conftest import LAYOUT

L = math.log10

# slots: 0 front (adj 1,2) | 1, 2 middle (adj each other + 0, 3) | 3 back (adj 1,2)


def build(heroes, upgrades, effect_defines=()):
    return GameData({
        "hero_defines": [
            {"id": hid, "name": name, "seat_id": hid, "tags": tags,
             "character_sheet_details": {"race": race, "ability_scores": stats}}
            for hid, name, tags, race, stats in heroes
        ],
        "upgrade_defines": [
            {"id": uid, "hero_id": hid, "required_level": 1, "effect": eff, "name": f"U{uid}"}
            for uid, hid, eff in upgrades
        ],
        "effect_defines": [{"id": i, "effect_keys": keys} for i, keys in effect_defines],
        "campaign_defines": [{"id": 1, "game_changes": [LAYOUT]}],
        "adventure_defines": [{"id": 1, "campaign_id": 1, "game_changes": []}],
    })


HEROES = [
    (1, "Carry", ["dps", "human"], "Human", {"str": 16}),
    (2, "Weak", ["support", "elf"], "Wood Elf", {"str": 10}),
    (3, "Middling", ["support", "dwarf"], "Dwarf (Shield)", {"str": 12}),
    (4, "Helper", ["support", "human"], "Human", {"str": 14}),
]


def score(data, placement_ids, carry=1, **overrides):
    scorer = Scorer(data, data.layout_for(1))
    scorer.stack_overrides.update(overrides.get("stacks", {}))
    return scorer.score({slot: data.champions[h] for slot, h in placement_ids.items()}, carry)


def test_multi_effect_def_expands_every_id_with_indexes():
    data = build(HEROES, [(10, 2, "effect_def,1,2")], [
        (1, [{"effect_string": "pre_stack,5"}]),
        (2, [{"effect_string": "global_dps_multiplier_mult,50"}, {"effect_string": "heal,1"}]),
    ])
    effs = data.upgrades[10].effects
    assert [(e.key, e.index) for e in effs] == [("pre_stack", 0), ("global_dps_multiplier_mult", 1), ("heal", 2)]


def test_species_from_character_sheet():
    data = build(HEROES, [])
    assert [data.champions[i].species for i in (1, 2, 3)] == ["human", "elf", "dwarf"]


def test_pre_stack_amount_and_per_hero_expression():
    """Thellora-style: +60% per champion with STR <= 12, amount held by pre_stack_amount."""
    keys = [
        {"effect_string": "pre_stack_amount,60"},
        {"effect_string": "hero_dps_multiplier_mult,0", "amount_expr": "upgrade_amount(10,0)",
         "targets": ["all_slots"], "amount_func": "mult", "stacks_multiply": True,
         "stack_func": "per_hero_attribute", "min_stat_amount": 12,
         "per_hero_expr": "clamp(min_stat_amount+1-str,0,1)"},
    ]
    data = build(HEROES, [(10, 2, "effect_def,1"), (20, 4, "buff_upgrade,100,10")], [(1, keys)])
    assert data.upgrades[10].effects[1].base_amount == 60
    # Weak (str 10) and Middling (str 12) count; Carry (16) doesn't
    res = score(data, {0: 1, 1: 2, 2: 3})
    assert math.isclose(res.log10, 2 * L(1.6))
    # Helper's +100% buff applies to the per-stack amount (pre-stack): 60% -> 120%
    res = score(data, {0: 1, 1: 2, 2: 3, 3: 4})
    assert math.isclose(res.log10, 2 * L(2.2))
    assert res.coverage == 1.0 and res.approx_share == 0.0


def test_buffs_to_same_upgrade_add_but_multiplicative_variants_multiply():
    ups = [(10, 2, '{"effect_string":"hero_dps_multiplier_mult,100","targets":["all"]}'),
           (20, 3, "buff_upgrade,100,10"), (30, 4, "buff_upgrade,100,10")]
    data = build(HEROES, ups)
    assert math.isclose(score(data, {0: 1, 1: 2, 2: 3, 3: 4}).log10, L(1 + 1.0 * 3))  # (1+1+1) x 100%
    ups[2] = (30, 4, "buff_upgrade_multiplicative,100,10")
    data = build(HEROES, ups)
    assert math.isclose(score(data, {0: 1, 1: 2, 2: 3, 3: 4}).log10, L(1 + 1.0 * 2 * 2))


def test_buffs_apply_post_stack_by_default_and_runtime_stacks_are_approximate():
    stacked = ('{"effect_string":"hero_dps_multiplier_mult,10","targets":["all"],'
               '"stacks_on_trigger":"monster_killed","max_stacks":10,"stacks_multiply":true}')
    data = build(HEROES, [(10, 2, stacked), (20, 3, "buff_upgrade,100,10")])
    alone = score(data, {0: 1, 1: 2})
    assert math.isclose(alone.log10, 10 * L(1.1))
    assert alone.modelled_approx == 1 and alone.approx_share == 1.0
    buffed = score(data, {0: 1, 1: 2, 2: 3})
    bonus = 1.1 ** 10 - 1  # stacked first, then doubled
    assert math.isclose(buffed.log10, L(1 + 2 * bonus))


def test_stack_policies():
    manual = '{"effect_string":"hero_dps_multiplier_mult,10","targets":["all"],"manual_stacking":true}'
    huge = ('{"effect_string":"hero_dps_multiplier_mult,10","targets":["all"],'
            '"stacks_on_trigger":"owner_attack","max_stacks":4000000000}')
    for eff in (manual, huge):
        data = build(HEROES, [(10, 2, eff)])
        res = score(data, {0: 1, 1: 2})
        assert res.log10 == 0 and res.unmodelled["hero_dps_multiplier_mult"] == 1
    # a user assumption makes the manual one count (and marks it approximate)
    data = build(HEROES, [(10, 2, manual)])
    res = score(data, {0: 1, 1: 2}, stacks={10: 5})
    assert math.isclose(res.log10, L(1.5)) and res.modelled_approx == 1
    assert static_class(data.upgrades[10].effects[0], data.layout_for(1)) == "unmodelled"


def test_change_upgrade_data_retargets_another_upgrade():
    adj_buff = '{"effect_string":"hero_dps_multiplier_mult,50","targets":["adj"]}'
    retarget = '{"effect_string":"change_upgrade_data,10","data":{"targets":["all"]}}'
    data = build(HEROES, [(10, 2, adj_buff), (20, 3, retarget)])
    assert score(data, {0: 1, 3: 2}).log10 == 0  # slot 3 isn't adjacent to slot 0
    assert math.isclose(score(data, {0: 1, 3: 2, 1: 3}).log10, L(1.5))


def test_add_target_and_filters():
    ups = [
        (10, 2, '{"effect_string":"hero_dps_multiplier_mult,50","targets":["behind"]}'),
        (20, 3, "add_target_to_upgrade,next_col,10"),
        (30, 4, '{"effect_string":"global_dps_multiplier_mult,100","filter_targets":[{"type":"hero_ids","hero_ids":[1]}]}'),
    ]
    data = build(HEROES, ups)
    # Weak in slot 1 (col 1): "behind" points away from the carry in slot 0 (col 0)
    assert score(data, {0: 1, 1: 2}).log10 == 0
    assert math.isclose(score(data, {0: 1, 1: 2, 2: 3}).log10, L(1.5))
    assert math.isclose(score(data, {0: 1, 1: 4}).log10, L(2))  # filter matches the carry
    assert score(data, {0: 2, 1: 4}, carry=2).log10 == 0  # ...and excludes others


def test_per_unique_species_and_amplifier_of_amplifier():
    ups = [
        (10, 2, '{"effect_string":"global_dps_multiplier_mult,20","stack_func":"per_unique_race","amount_func":"add"}'),
        (20, 3, "buff_upgrade,100,10"),
        (30, 4, "buff_upgrade,50,20"),  # amplifies the amplifier: 100% -> 150%
    ]
    data = build(HEROES, ups)
    # species: human (Carry, Helper), elf, dwarf -> 3 unique; 20% x 3 = 60%, x2.5 post-stack
    res = score(data, {0: 1, 1: 2, 2: 3, 3: 4})
    assert math.isclose(res.log10, L(1 + 0.6 * 2.5))


def test_per_hero_targets_limits_the_count_to_neighbours():
    """Dob-style "for each Champion next to him": only adjacent champions count."""
    eff = ('{"effect_string":"global_dps_multiplier_mult,100","amount_func":"mult","stack_func":"per_hero_attribute",'
           '"per_hero_expr":"true","per_hero_targets":["adj"]}')
    data = build(HEROES, [(10, 2, eff)])
    # Weak in slot 3 (adjacent to 1 and 2); Carry in slot 0 is not adjacent
    assert score(data, {0: 1, 3: 2}).log10 == pytest.approx(0.0)
    assert score(data, {0: 1, 3: 2, 1: 3}).log10 == pytest.approx(L(2))  # Middling is next to Weak
    assert score(data, {0: 1, 3: 2, 1: 3, 2: 4}).log10 == pytest.approx(2 * L(2))


def test_egbert_atonement_and_banked_stacks():
    egbert = ('{"effect_string":"global_dps_multiplier_mult,10","amount_func":"add","stack_func":"egbert_atonement"}')
    heroes = [(1, "Carry", ["dps", "lawful"], "Human", {}), (2, "Egbert", ["support"], "Human", {}),
              (3, "Law", ["lawful"], "Human", {}), (4, "Chaos", ["chaotic"], "Human", {})]
    data = build(heroes, [(10, 2, egbert)])
    assert score(data, {0: 1, 1: 2, 2: 3}).log10 == pytest.approx(L(1 + 4 * 0.10))  # (2 lawful)^2 = 4
    assert score(data, {0: 1, 1: 2, 2: 3, 3: 4}).log10 == pytest.approx(L(1.1))  # (2-1)^2 = 1
    # Rust-style cap: stacks = ceil(run_area / 5) for a manually stacked buff
    rust = [{"effect_string": "buff_upgrade,5,10", "stacks_multiply": True, "manual_stacking": True},
            {"effect_string": "stacks_max_stack_expr,0,highest_available_area/5", "rounding_mode": "ceil"}]
    data = build(HEROES, [(10, 2, '{"effect_string":"global_dps_multiplier_mult,100"}'), (20, 2, "effect_def,1")], [(1, rust)])
    from icfo.engine import Scorer
    s = Scorer(data, data.layout_for(1))
    s.run_area = 48  # -> 10 stacks
    res = s.score({0: data.champions[1], 1: data.champions[2]}, 1)
    assert res.log10 == pytest.approx(L(1 + 1.05 ** 10))
