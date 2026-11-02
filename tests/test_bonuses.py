"""Blessings (local/global) and patron perks (local/global)."""

import math

import pytest

from icfo.engine import Scorer
from icfo.model import GameData, UserState, constraints_for
from icfo.model.bonuses import Bonuses, patron_id

from conftest import DEFS

L = math.log10

CAMPAIGN = {"defines": {"reset_upgrade_defines": [
    # local to campaign 1's favor (currency 1): +50%/level to slot 0
    {"id": 1, "reset_currency_id": 1, "name": "Hold the Front", "type": 1, "levels": 5,
     "effects": [{"effect_string": "effect_def,50", "per_level": 50}]},
    # local to another campaign's favor: must not apply
    {"id": 2, "reset_currency_id": 9, "name": "Elsewhere", "type": 1, "levels": 5,
     "effects": [{"effect_string": "global_dps_multiplier_mult,$replace", "per_level": 1000}]},
    # global blessing from another campaign: applies everywhere
    {"id": 3, "reset_currency_id": 9, "name": "Everywhere", "type": 2, "levels": 1,
     "effects": [{"effect_string": "global_dps_mult_per_unique_race_character_mult,$replace", "per_level": 10}]},
    # attack speed for everyone
    {"id": 4, "reset_currency_id": 1, "name": "Speedy", "type": 1, "levels": 5,
     "effects": [{"effect_string": "reduce_attack_cooldown,$replace", "per_level": 0.5}]},
]}}


def data_with_bonuses():
    defs = {
        **DEFS,
        "campaign_defines": [{**DEFS["campaign_defines"][0], "reset_currency_id": 1}],
        "effect_defines": DEFS["effect_defines"] + [
            {"id": 50, "effect_keys": [{"effect_string": "hero_dps_multiplier_mult,$replace",
                                         "targets": [{"type": "slots", "slot_ids": [0]}]}]},
        ],
        "attack_defines": [{"id": 1, "cooldown": 4, "damage_modifier": 1}],
        "hero_defines": [{**h, "base_attack_id": 1} for h in DEFS["hero_defines"]],
        "patron_perk_defines": [
            {"id": 1, "patron_id": 1, "type": 1, "name": "Mirt Local", "levels": 10,
             "effects": [{"effect_string": "global_dps_multiplier_mult,$replace", "per_level": 100}]},
            {"id": 2, "patron_id": 3, "type": 2, "name": "Weakened Foes", "levels": 20,
             "effects": [{"effect_string": "monster_health_reduce,$replace", "per_level": 4}]},
        ],
        "patron_defines": [{"id": 1, "name": "Mirt", "game_changes": {"1": [
            {"type": "disallow_crusaders", "by_tags": {"tags": "!(good|evil)"}}]}}],
    }
    return GameData(defs)


def user():
    u = UserState.everything({1, 2, 3, 4, 5, 6})
    u.from_account = True
    u.blessing_levels = {1: 2, 2: 5, 3: 1, 4: 2}
    u.perk_levels = {1: 10, 2: 10}
    return u


def test_local_and_global_scoping():
    data = data_with_bonuses()
    b = Bonuses(data, user(), CAMPAIGN)
    names = {x.name for x in b.for_adventure(1)}
    assert names == {"Hold the Front", "Everywhere", "Speedy", "Weakened Foes"}  # no "Elsewhere", no local perk
    assert {x.name for x in b.for_adventure(1, patron_id("Mirt"))} == names | {"Mirt Local"}
    hold = next(x for x in b.blessings if x.name == "Hold the Front")
    assert hold.effects[0].args == ("100",)  # 50 per level x level 2


def test_bonuses_change_scores_and_placement_value():
    data = data_with_bonuses()
    b = Bonuses(data, user(), CAMPAIGN)
    s = Scorer(data, data.layout_for(1))
    ch = data.champions
    ch[1].species, ch[2].species = "human", "elf"
    plain = s.score({1: ch[1]}, 1).log10
    s.set_bonuses(b.effects_for(1))
    off_slot = s.score({1: ch[1]}, 1).log10
    on_slot = s.score({0: ch[1]}, 1).log10
    # species: human -> 1 unique: x1.1 ; weakened foes 40% -> 1/0.6 ; cooldown 4 -> 3 s
    common = L(1.1) - L(0.6) + L(4 / 3)
    assert off_slot - plain == pytest.approx(common)
    assert on_slot - off_slot == pytest.approx(L(2))  # Hold the Front +100% only in slot 0
    two_species = s.score({0: ch[1], 1: ch[2]}, 1).log10 - s.score({0: ch[1]}, 1).log10
    assert two_species == pytest.approx(L(3) + L(1.1))  # Adjy's adjacency buff + one more species


def test_patron_restrictions_apply():
    data = data_with_bonuses()
    data.champions[1].tags = frozenset({"dps", "human", "good"})
    c = constraints_for(data, 1, patron_id("Mirt"))
    assert c.allows(data.champions[1]) and not c.allows(data.champions[2])
    assert constraints_for(data, 1).allows(data.champions[2])
