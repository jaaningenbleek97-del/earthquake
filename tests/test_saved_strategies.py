import numpy as np

from eqpred.eval.saved_strategies import _cards, _predict_region, drop_card, load_store, save_card


def _tiny_tape(tmp_path, n=400):
    flags = np.zeros(n, dtype=bool)
    arrays = {
        "matrix": np.zeros((n, 2), dtype=np.float32),
        "30_fit": flags.copy(),
        "30_val": flags.copy(),
        "30_exam": flags.copy(),
        "30_kermadec_model": np.full(n, 0.3, dtype=np.float32),
        "30_kermadec_live": np.full(n, 0.6, dtype=np.float32),
        "30_kermadec_base": np.full(n, 0.3, dtype=np.float32),
        "30_kermadec_y": np.zeros(n, dtype=np.float32),
        "30_kermadec_c": np.full(n, 0.3, dtype=np.float32),
    }
    np.savez(tmp_path / "tape.npz", **arrays)
    meta = {
        "start": "2000-01-01",
        "days": n,
        "horizons": {"30": {"regions": ["kermadec"], "mags": {"kermadec": 6.0}, "groups": {"kermadec": []}}},
    }
    return np.load(tmp_path / "tape.npz"), meta


def test_saved_card_with_reading_votes_scores_the_vote_series(tmp_path):
    packed, meta = _tiny_tape(tmp_path)
    built, matrix, _days = _cards(30, packed, meta)
    spec = {"strategy": "current", "region": "kermadec"}
    plain = _predict_region(spec, built, matrix, {"use_readings": False})
    votes = _predict_region(spec, built, matrix, {"use_readings": True})
    assert np.allclose(plain, 0.3)
    assert np.allclose(votes, 0.6)


def test_a_card_written_late_does_not_count_as_prospective():
    from eqpred.eval.honest import counts_as_prospective

    on_time = {"prospective": True, "issue_day": "2026-09-21", "written_at": "2026-09-21T08:00:00+00:00"}
    late = dict(on_time, written_at="2026-09-25T03:35:13+00:00")
    assert counts_as_prospective(on_time)
    assert not counts_as_prospective(late)
    assert not counts_as_prospective(dict(on_time, prospective=False))


def test_saving_the_same_card_twice_keeps_one_and_remove_drops_it(tmp_path):
    spec = {"strategy": "alarm", "horizon": 30, "region": "alpine", "knobs": {"alarm_share": 0.2}}
    first = save_card(spec, root=tmp_path)
    again = save_card(dict(spec), root=tmp_path)
    assert first["id"] == again["id"]
    ids = [card["id"] for card in load_store(tmp_path)["cards"]]
    assert ids.count(first["id"]) == 1
    drop_card(first["id"], root=tmp_path)
    assert first["id"] not in [card["id"] for card in load_store(tmp_path)["cards"]]
