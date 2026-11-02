"""A tiny synthetic definitions payload that mirrors the real schema shapes."""

import pytest

from icfo.model import GameData

# 4-slot layout:   col2   col1   col0 (front)
#                         [1]
#                  [3]           [0]
#                         [2]
LAYOUT = {
    "type": "formation",
    "name": "Test Kite",
    "formation": [
        {"x": 40, "y": 20, "col": 0, "adj": [1, 2]},
        {"x": 20, "y": 10, "col": 1, "adj": [0, 2, 3]},
        {"x": 20, "y": 30, "col": 1, "adj": [0, 1, 3]},
        {"x": 0, "y": 20, "col": 2, "adj": [1, 2]},
    ],
}


def hero(id_, name, seat, tags):
    return {"id": id_, "name": name, "seat_id": seat, "tags": tags}


def upgrade(id_, hero_id, effect, level=10, **extra):
    return {"id": id_, "hero_id": hero_id, "required_level": level, "required_upgrade_id": 0, "effect": effect, **extra}


DEFS = {
    "hero_defines": [
        hero(1, "Carry", 1, ["dps", "human"]),
        hero(2, "Adjy", 2, ["support", "elf"]),
        hero(3, "Globo", 3, ["support", "dwarf"]),
        hero(4, "Ampy", 4, ["support", "human"]),
        hero(5, "Rival", 2, ["support", "gnome"]),
        hero(6, "Counter", 5, ["support", "tiefling"]),
        hero(7, "Placeholder", 0, []),  # seat 0: not a real champion
    ],
    "upgrade_defines": [
        upgrade(10, 1, "hero_dps_multiplier_mult,100"),
        upgrade(20, 2, "effect_def,1"),
        upgrade(30, 3, '{"effect_string":"global_dps_multiplier_mult,50"}'),  # JSON-string form
        upgrade(40, 4, "buff_upgrade,100,20"),
        upgrade(50, 5, "effect_def,2"),
        upgrade(60, 6, "effect_def,3"),
        # specialization choice for Counter at level 100, plus an upgrade gated on spec 62
        upgrade(61, 6, "global_dps_multiplier_mult,10", level=100, specialization_name="Spec A"),
        upgrade(62, 6, "global_dps_multiplier_mult,20", level=100, specialization_name="Spec B"),
        upgrade(63, 6, "global_dps_multiplier_mult,30", level=120, required_upgrade_id=62),
    ],
    "effect_defines": [
        {"id": 1, "effect_keys": [{"effect_string": "hero_dps_multiplier_mult,200", "targets": ["adj"]}]},
        # dict-shaped effect_keys, as seen in live data
        {"id": 2, "effect_keys": {"effect_string": "hero_dps_multiplier_mult,50", "targets": ["all"]}},
        {
            "id": 3,
            "effect_keys": [
                {
                    "effect_string": "hero_dps_mult_per_tagged_crusader_mult,100,human",
                    "targets": ["all"],
                    "stacks_multiply": True,
                }
            ],
        },
    ],
    "hero_feat_defines": [],
    "campaign_defines": [{"id": 1, "name": "Test", "game_changes": [LAYOUT]}],
    "adventure_defines": [
        {"id": 1, "campaign_id": 1, "name": "Base", "game_changes": []},
        {
            "id": 2,
            "campaign_id": 1,
            "name": "Variant",
            "variant_adventure_id": 1,
            "restrictions_text": "slot 3 blocked; no dwarves; Ampy required",
            "game_changes": [
                {"type": "slot_escort", "slot_ids": [3]},
                {"type": "disallow_crusaders", "by_tags": {"tags": "dwarf"}},
                {"type": "force_use_heroes", "hero_ids": [4]},
                {"type": "random_monster", "chance": 1},
            ],
        },
    ],
}


@pytest.fixture()
def data() -> GameData:
    return GameData(DEFS)
