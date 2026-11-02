import math
import pytest

from icfo.model import Layout, constraints_for, match_tags, normalize

DIAMOND = {
    "type": "formation",
    "name": "Diamond Formation",
    "formation": [
        {"x": 80, "y": 30, "col": 0, "adj": [1, 2]},
        {"x": 60, "y": 20, "col": 1, "adj": [0, 2, 3, 4]},
        {"x": 60, "y": 40, "col": 1, "adj": [0, 1, 4, 5]},
        {"x": 40, "y": 10, "col": 2, "adj": [1, 4, 6]},
        {"x": 40, "y": 30, "col": 2, "adj": [1, 2, 3, 5, 6, 7]},
        {"x": 40, "y": 50, "col": 2, "adj": [2, 4, 7]},
        {"x": 20, "y": 20, "col": 3, "adj": [3, 4, 7, 8]},
        {"x": 20, "y": 40, "col": 3, "adj": [4, 5, 6, 8]},
        {"x": 0, "y": 30, "col": 4, "adj": [6, 7]},
    ],
}


def test_real_layout_adjacency_is_symmetric():
    layout = Layout.from_game_change(DIAMOND)
    for s in layout.slots:
        for a in s.adj:
            assert s.index in layout.slots[a].adj


def test_relations():
    layout = Layout.from_game_change(DIAMOND)
    assert layout.targets("adj", 4) == {1, 2, 3, 5, 6, 7}
    assert layout.targets("col", 4) == {3, 4, 5}
    assert layout.targets("next_col", 4) == {1, 2}  # column in front
    assert layout.targets("prev_col", 4) == {6, 7}  # column behind
    assert layout.targets("ahead", 3) == {1}
    assert layout.targets("behind", 0) == {1, 2}
    assert layout.targets("non_adj", 0) == {3, 4, 5, 6, 7, 8}
    assert layout.targets({"type": "distance", "distance": 2}, 0) == {1, 2, 3, 4, 5}
    assert layout.targets("some_future_relation", 0) is None


def test_render_has_front_on_right():
    layout = Layout.from_game_change(DIAMOND)
    out = layout.render({0: "F", 8: "B"})
    middle = [line for line in out.splitlines() if "F" in line][0]
    assert middle.index("B") < middle.index("F")


def test_match_tags():
    tags = {"human", "fighter", "good"}
    assert match_tags("human", tags)
    assert not match_tags("!human", tags)
    assert match_tags("elf|fighter", tags)
    assert match_tags("good&fighter", tags)
    assert not match_tags("good&elf", tags)


def test_normalize_shapes():
    defs = {7: {"effect_keys": {"effect_string": "hero_dps_multiplier_mult,10", "targets": "adj"}}}
    effs = normalize(["effect_def,7", '{"effect":"global_dps_multiplier_mult,5"}', None, 3], defs)
    assert [(e.key, e.targets) for e in effs] == [
        ("hero_dps_multiplier_mult", ("adj",)),
        ("global_dps_multiplier_mult", ("self",)),
    ]
    assert math.isclose(effs[0].num(0), 10)


def test_champions_skip_placeholders(data):
    assert 7 not in data.champions
    assert data.champion("carry").id == 1


def test_specialization_choice_and_gating(data):
    counter = data.champions[6]
    assert [[u.id for u in g] for g in counter.spec_groups] == [[61, 62]]
    default = {u.id for u in counter.active_upgrades()}
    assert default == {60, 61}  # first spec by default; 63 is gated on spec 62
    picked = {u.id for u in counter.active_upgrades(specs={0: 62})}
    assert picked == {60, 62, 63}
    assert {u.id for u in counter.active_upgrades(level=50)} == {60}


def test_variant_constraints(data):
    c = constraints_for(data, 2)
    assert c.blocked_slots == {3}
    assert c.required_ids == {4}
    assert not c.allows(data.champions[3])  # dwarf banned
    assert any("random_monster" in w for w in c.warnings)
    ch = data.champions
    assert c.placement_ok({0: ch[1], 1: ch[4]})
    assert not c.placement_ok({0: ch[1]})  # missing required Ampy
    assert not c.placement_ok({0: ch[1], 1: ch[4], 3: ch[2]})  # blocked slot
    assert not c.placement_ok({0: ch[1], 1: ch[4], 2: ch[2], 3: ch[5]})  # seat clash too


def test_hero_expressions(data):
    from icfo.model.expr import UnsupportedExpr, evaluate

    carry = data.champions[1]  # seat 1, tags dps/human
    carry.stats.update({"str": 16, "int": 10, "wis": 12})
    assert evaluate("HasTag(`dps`)", carry)
    assert evaluate("hero_id == 33 || hero_id == 1", carry)
    assert evaluate("(!HasTag(`dps`)&&hero_seat_id!=1)||(HasTag(`dps`)&&hero_seat_id==1)", carry)
    assert evaluate("GetStat(`str`) > GetStat(`int`) && STR>=16", carry)
    assert evaluate("max(INT,WIS)>=12", carry)
    assert not evaluate("total_ability_score > 100", carry)
    with pytest.raises(UnsupportedExpr):
        evaluate("avg_i_levels < 1000", carry)
    with pytest.raises(UnsupportedExpr):
        evaluate("__import__('os')", carry)


def test_expr_and_seat_criteria(data):
    data.adventures[2].game_changes = [
        {"type": "only_allow_crusaders", "by_expr": {"expr": "HasTag(`support`) || hero_id == 1"}},
        {"type": "disallow_crusaders", "by_seat": {"seats": [5]}, "blocked_portrait": 1},
        {"type": "only_allow_crusaders", "by_attack_types": {"attack_types": ["magic"]}},
    ]
    c = constraints_for(data, 2)
    allowed = {ch.name for ch in data.champions.values() if c.allows(ch)}
    assert allowed == {"Carry", "Adjy", "Globo", "Ampy", "Rival"}
    assert c.warnings == ["only_allow_crusaders.by_attack_types not modelled"]


def test_criteria_within_one_entry_are_alternatives(data):
    # mirrors "Odd Ones Out": even seats OR specific ids, with a forced champion
    data.adventures[2].game_changes = [
        {"type": "only_allow_crusaders", "by_seat": {"seats": [2, 4]}, "by_expr": {"expr": "hero_id == 6"}},
        {"type": "force_use_heroes", "hero_ids": [3]},
    ]
    c = constraints_for(data, 2)
    allowed = {ch.name for ch in data.champions.values() if c.allows(ch)}
    assert allowed == {"Adjy", "Rival", "Ampy", "Counter", "Globo"}  # Globo (seat 3) is forced in


def test_extended_relations():
    layout = Layout.from_game_change(DIAMOND)
    assert layout.targets("next_two_col", 6) == {1, 2, 3, 4, 5}
    assert layout.targets("prev_and_next_col", 4) == {1, 2, 6, 7}
    assert layout.targets("self_and_behind_and_ahead", 3) == {1, 3, 6}
    assert layout.targets("col_and_behind", 0) == {0, 1, 2}
    assert layout.targets("front_2_columns", 8) == {0, 1, 2}
    assert layout.targets("top_row_of_each_column", 0) == {0, 1, 3, 6, 8}
    assert layout.targets({"type": "col_and_back_x", "num_back_cols": 1}, 1) == {1, 2, 3, 4, 5}
