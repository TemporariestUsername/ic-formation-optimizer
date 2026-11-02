"""Account data, objectives (push / favor / gems), planner and sweep."""

import csv
import math

import pytest

from icfo.engine import Scorer
from icfo.model import GameData, UserState
from icfo.model.user import GearItem
from icfo.objectives import (
    FavorObjective, GemObjective, GemParams, ObjectiveScorer, Progression, PushObjective,
    briv_jump, briv_jumps_from_stacks,
)
from icfo.optimize import candidate_pool
from icfo.optimize.planner import plan
from icfo.optimize.sweep import run_sweep
from icfo.model import constraints_for

from conftest import DEFS

L = math.log10

USER = {
    "details": {
        "heroes": [
            {"hero_id": "1", "owned": "1", "level": "500", "specialization_choices": [], "active_feats": [7]},
            {"hero_id": "2", "owned": "1", "level": "300"},
            {"hero_id": "6", "owned": "1", "level": "100", "specialization_choices": ["62"]},
            {"hero_id": "3", "owned": "0", "level": "0"},
        ],
        "loot": [{"hero_id": 2, "slot_id": 1, "rarity": 4, "gild": 1, "enchant": 250}],
        "stats": {"briv_sprint_stacks": "1153", "thellora_plateaus_of_unicorn_run_areas": "19.8"},
        "game_instances": [{"game_instance_id": 3, "current_adventure_id": 1, "current_area": 120,
                            "highest_area": 120, "formation": [1, 2, -1, -1]}],
        "modron_saves": {"1": {"instance_id": "3", "area_goal": "100"}},
        "adventure_data": {"gold_growth_rate": 2.031, "currency_growth_rate": 1.25},
        "newsletter_email": "someone@example.com",
    }
}


def with_gear_defs():
    defs = {**DEFS, "loot_defines": [
        {"hero_id": 2, "slot_id": 1, "rarity": 4, "effects": [{"effect_string": "global_dps_multiplier_mult,100"}]},
    ]}
    return GameData(defs)


def test_user_details_parsing():
    u = UserState.from_details(USER)
    assert u.owned == {1, 2, 6} and u.from_account
    assert u.levels[1] == 500 and u.feats[1] == (7,) and u.spec_choices[6] == [62]
    assert u.gear[2] == [GearItem(2, 1, 4, 1, 250)]
    assert u.instances[0].reset_area == 100 and u.instances[0].formation == (1, 2, -1, -1)
    assert u.stat("briv_sprint_stacks") == 1153
    assert not any("example.com" in str(v) for v in vars(u).values())  # personal data isn't kept


def test_gear_scaling_and_account_specs():
    data = with_gear_defs()
    u = UserState.from_details(USER)
    assert u.gear[2][0].multiplier == pytest.approx((1 + 0.004 * 250) * 1.5)  # = 3.0
    s = Scorer(data, data.layout_for(1))
    s.apply_account(u)
    ch = data.champions
    # Adjy's epic shiny gear: +100% x 3.0 = +300% global, on top of its +200% adjacency buff
    res = s.score({0: ch[1], 1: ch[2]}, 1)
    assert res.log10 == pytest.approx(L(2) + L(3) + L(4))
    # Counter's account spec is 62 (Spec B) -> +20%, and unlocks 63 (+30%)
    keys = {e.upgrade_id for e in s.effects_for(ch[6])}
    assert {62, 63} <= keys and 61 not in keys


def test_progression_matches_game_rule_constants():
    data = GameData({**DEFS, "game_rule_defines": [{"rule_name": "monster_base_stats", "rule": {
        "base_health": 10, "health_growth_rate_curve": {"1": 2.031, "2001": 3.031}}}]})
    p = Progression(data)
    assert p.log10_hp(1) == pytest.approx(1.0)
    assert p.log10_hp(101) == pytest.approx(1 + 100 * L(2.031))
    assert p.log10_hp(2101) == pytest.approx(1 + 2000 * L(2.031) + 100 * L(3.031))
    assert p.area_for(p.log10_hp(777)) == pytest.approx(777, abs=1e-6)


def test_favor_is_damage_plus_gold_times_exponent():
    data = GameData(DEFS)
    u = UserState.from_details(USER)
    fav = FavorObjective(data, u)
    assert fav.exponent == pytest.approx(math.log(1.25) / math.log(2.031))
    s = Scorer(data, data.layout_for(1))
    res = s.score({0: data.champions[1]}, 1)
    res.gold_log10 = 2.0
    assert fav.value(res, data.champions[1], {}) == pytest.approx(fav.exponent * (res.log10 + 2.0))


def test_briv_jump_matches_byteglow():
    # epic shiny item at enchant 490: multiplier 100*(1+1.96)*1.5 = 444%
    jump, chance = briv_jump(4.44)
    assert jump == 3 and chance == pytest.approx(0.3664, abs=1e-4)
    assert briv_jump(0.5) == (2, pytest.approx(0.375))  # effect 0.375 <= 1: that's the chance
    assert briv_jumps_from_stacks(1153, 4) == 78
    assert briv_jumps_from_stacks(49, 4) == 0


def test_gem_model_counts_skips_and_speed():
    data = GameData(DEFS)
    u = UserState.from_details(USER)
    from icfo.engine.score import ScoreResult, SpeedEffect
    res = ScoreResult(log10=0.0)
    params = GemParams(reset_area=101, kills_per_area=20, spawn_interval_s=1, transition_s=0, reset_overhead_s=0)
    base = GemObjective(data, u, params).model(res, data.champions[1], {0: data.champions[1]})
    assert base["sec_per_area"] == pytest.approx(20)
    assert base["areas_per_hour"] == pytest.approx(100 / (100 * 20) * 3600)
    res.speed = [SpeedEffect("X", 2, "increase_monster_spawn_time_mult", ("100",), 100.0, 1.0)]
    fast = GemObjective(data, u, params).model(res, data.champions[1], {0: data.champions[1]})
    assert fast["sec_per_area"] == pytest.approx(10)


def test_planner_picks_carry_and_objectives_rank_formations():
    data = GameData(DEFS)
    u = UserState.everything(set(data.champions))
    c = constraints_for(data, 1)
    pool = candidate_pool(list(data.champions.values()), c, u.owned)
    s = Scorer(data, c.layout)
    p = plan(ObjectiveScorer(s, PushObjective(data)), c, pool, None, "fast")
    assert p.carry.name == "Carry"  # the only dps champion
    assert c.placement_ok(p.placement)
    # push optimum = the brute-force optimum of the raw damage score
    from icfo.optimize import exhaustive
    best = exhaustive(s, c, pool, p.carry)
    assert p.result.damage_log10 == pytest.approx(best.score.log10)


def test_sweep_groups_and_writes_csv(tmp_path):
    data = GameData(DEFS)
    u = UserState.everything(set(data.champions))
    out = tmp_path / "sweep.csv"
    summary = run_sweep(data, u, ["push", "favor"], calibration=None, out_csv=str(out),
                        workers=1, use_account=False)
    assert summary["adventures"] == 2 and summary["groups"] == 2 and summary["errors"] == 0
    rows = list(csv.DictReader(out.open()))
    assert {(r["adventure_id"], r["mode"]) for r in rows} == {("1", "push"), ("1", "favor"), ("2", "push"), ("2", "favor")}
    assert all(r["carry"] == "Carry" for r in rows)
    assert all("3:" not in r["formation"] for r in rows if r["adventure_id"] == "2")  # slot 3 is blocked there
