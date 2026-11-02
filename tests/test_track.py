import pytest

from icfo.engine import Scorer
from icfo.fetch.cache import Cache
from icfo.model import GameData, UserState
from icfo.model.user import GameInstance
from icfo.objectives import Progression, calibrate
from icfo import track

from conftest import DEFS


def make_user(instances):
    u = UserState.everything({1, 2, 3, 4, 5, 6})
    u.from_account = True
    u.instances = instances
    return u


def scorer_for(data):
    def make(layout, adventure_id=None, patron=None):
        return Scorer(data, layout)
    return make


def test_record_dedupes_and_marks_walls(tmp_path):
    cache = Cache(tmp_path)
    u = make_user([GameInstance(1, 1, 50, 50, (1, 2, -1, -1)), GameInstance(2, 2, 5, 5, (1,))])
    obs, changed = track.record(cache, u)
    assert len(obs) == 1 and len(changed) == 1  # area-5 instance below the minimum
    obs, changed = track.record(cache, u)
    assert changed == []  # nothing new
    u.instances[0] = GameInstance(1, 1, 80, 80, (1, 2, -1, -1))
    obs, changed = track.record(cache, u, stalled=True)
    assert len(obs) == 1 and obs[0].highest_area == 80 and obs[0].stalled
    assert track.load(cache)[0].formation == (1, 2, -1, -1)


def test_offsets_consistency_and_calibration(tmp_path):
    data = GameData(DEFS)
    prog = Progression(data)
    make = scorer_for(data)
    # same carry, two formations: alone (x2) and with Adjy adjacent (x2 x3)
    alone = track.Observation(0, 1, 1, None, (1, -1, -1, -1), 100, True)
    paired = track.Observation(0, 2, 1, None, (1, 2, -1, -1), 100, True)
    ev = track.evaluate(data, [alone, paired], make)
    assert ev[1].model - ev[0].model == pytest.approx(0.47712, abs=1e-4)  # log10(3)
    # both stalled at the same area although the model says the pair is 3x stronger:
    spread = track.consistency(ev, prog.slope(100))
    assert spread == pytest.approx(0.47712 / 2 / prog.slope(100), rel=1e-3)
    offset, how = track.offset_from(ev)
    assert "median of 2 wall" in how
    cal = calibrate(data, make_user([]), make, [alone, paired])
    assert cal.offset == pytest.approx(offset) and "icfo track" in cal.note
    # lower bounds only: the tightest (largest) offset wins
    lb = [track.Observation(0, 1, 1, None, (1, -1, -1, -1), 100), track.Observation(0, 2, 1, None, (1, -1, -1, -1), 60)]
    off, how = track.offset_from(track.evaluate(data, lb, make))
    assert off == pytest.approx(prog.log10_hp(100) - ev[0].model) and "lower-bound" in how
