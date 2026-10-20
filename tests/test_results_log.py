import json

from eqpred.eval.results_log import record_results, results_log_path, results_text_path


def test_results_log_records_cards_without_changing_them(tmp_path):
    folder = tmp_path / "data" / "eval"
    folder.mkdir(parents=True)
    (folder / "mainshock.json").write_text(
        json.dumps(
            {
                "week": [
                    {
                        "region": "global",
                        "label": "Earth-wide",
                        "horizon_days": 30,
                        "chance_plain": "57 in 100",
                        "climatology_chance_plain": "67 in 100",
                        "issue_day": "2026-09-21",
                        "p": 0.57,
                    }
                ],
                "short_horizons": [
                    {
                        "region": "hikurangi",
                        "label": "Hikurangi",
                        "horizon_days": 3,
                        "chance_plain": "8 in 100",
                        "climatology_chance_plain": "3 in 100",
                        "issue_day": "2026-09-21",
                        "p": 0.08,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (folder / "hunt_fresh_quiet.json").write_text(
        json.dumps(
            {
                "fresh": True,
                "quiet_slice": True,
                "n_knobs": 3,
                "n_tests": 10,
                "n_discovered": 0,
                "n_confirmed": 0,
                "n_replayed": 0,
                "writes_issued_p": False,
                "confirmed": [],
            }
        ),
        encoding="utf-8",
    )
    entry = record_results(tmp_path)
    assert entry["writes_issued_p"] is False
    assert entry["cards"]["day_3"][0]["chance_plain"] == "8 in 100"
    saved = json.loads(results_log_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert saved["cards"]["day_30"][0]["chance_plain"] == "57 in 100"
    text = results_text_path(tmp_path).read_text(encoding="utf-8")
    assert "8 in 100" in text
    assert "confirmed 0" in text
