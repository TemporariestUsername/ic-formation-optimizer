import importlib.util
import math
import os

import pytest

from icfo.engine import Scorer
from icfo.model import GameData, UserState, constraints_for
from icfo.optimize import anneal, candidate_pool, exhaustive, greedy

L = math.log10


def setup(data, adventure=1):
    c = constraints_for(data, adventure)
    pool = candidate_pool(list(data.champions.values()), c, set(data.champions))
    return c, pool, Scorer(data, c.layout)


def test_scores(data):
    c, _, scorer = setup(data)
    ch = data.champions
    alone = scorer.score({0: ch[1]}, 1)
    assert math.isclose(alone.log10, L(2))  # own +100%

    adjacent = scorer.score({0: ch[1], 1: ch[2]}, 1).log10
    far = scorer.score({0: ch[1], 3: ch[2]}, 1).log10
    assert math.isclose(adjacent, L(2) + L(3))
    assert math.isclose(far, L(2))

    # Ampy doubles Adjy's +200% -> +400%, and is human for Counter's per-human stacks
    amped = scorer.score({0: ch[1], 1: ch[2], 3: ch[4]}, 1).log10
    assert math.isclose(amped, L(2) + L(5))
    counted = scorer.score({0: ch[1], 1: ch[2], 3: ch[4], 2: ch[6]}, 1)
    # Counter: x2 per human (Carry, Ampy) = 2 * log10(2), plus default spec A global +10%
    assert math.isclose(counted.log10, L(2) + L(5) + 2 * L(2) + L(1.1))
    assert counted.coverage == 1.0


def test_unmodelled_effects_are_reported():
    defs = {
        "hero_defines": [{"id": 1, "name": "A", "seat_id": 1, "tags": []}],
        "upgrade_defines": [{"id": 1, "hero_id": 1, "required_level": 1, "effect": "mystery_effect,5"}],
        "campaign_defines": [{"id": 1, "game_changes": [{"type": "formation", "formation": [{"x": 0, "y": 0, "col": 0, "adj": []}]}]}],
        "adventure_defines": [{"id": 1, "campaign_id": 1, "game_changes": []}],
    }
    data = GameData(defs)
    res = Scorer(data, data.layout_for(1)).score({0: data.champions[1]}, 1)
    assert res.unmodelled["mystery_effect"] == 1
    assert res.coverage == 0.0


@pytest.mark.parametrize("adventure", [1, 2])
def test_heuristics_match_exhaustive(data, adventure):
    c, pool, scorer = setup(data, adventure)
    carry = data.champions[1]
    best = exhaustive(scorer, c, pool, carry)
    assert c.placement_ok(best.placement)
    for result in (greedy(scorer, c, pool, carry), anneal(scorer, c, pool, carry, iterations=3000)):
        assert c.placement_ok(result.placement)
        assert math.isclose(result.score.log10, best.score.log10)


@pytest.mark.skipif(importlib.util.find_spec("ortools") is None, reason="ortools not installed")
@pytest.mark.parametrize("adventure", [1, 2])
def test_cpsat_matches_exhaustive(data, adventure):
    from icfo.optimize.cpsat import solve

    c, pool, scorer = setup(data, adventure)
    carry = data.champions[1]
    best = exhaustive(scorer, c, pool, carry)
    result = solve(scorer, c, pool, carry, time_limit_s=5)
    assert c.placement_ok(result.placement)
    assert math.isclose(result.score.log10, best.score.log10)


@pytest.mark.skipif(not os.environ.get("ICFO_DEFS"), reason="set ICFO_DEFS=path/to/definitions.json")
def test_live_definitions_smoke():
    import json

    with open(os.environ["ICFO_DEFS"], encoding="utf-8") as f:
        data = GameData(json.load(f))
    assert len(data.champions) > 100
    from icfo.model.definitions import LayoutNotFound

    missing = 0
    for adv_id in data.adventures:
        try:
            constraints_for(data, adv_id)  # every layout + restriction set parses
        except LayoutNotFound:
            missing += 1
    assert missing < 20  # a handful of special-mode adventures have no layout
    c = constraints_for(data, 1)
    user = UserState.everything(set(data.champions))
    pool = candidate_pool(list(data.champions.values()), c, user.owned)
    carry = data.champion("Bruenor")
    result = anneal(Scorer(data, c.layout), c, pool, carry, iterations=500)
    assert c.placement_ok(result.placement)
