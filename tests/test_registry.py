import json
from pathlib import Path

from eqpred.models.registry import (
    MODEL_FILES,
    ensure_registered,
    pin_unpinned_batches,
    preserve_current,
    register_current,
    stamp_cards,
)
from eqpred.paths import model_dir, prediction_batches, prediction_dir
from eqpred.records import prediction
from datetime import datetime, timezone


def _write_brain(root: Path, mu: float = 0.1) -> None:
    folder = model_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "etas.json").write_text(json.dumps({"mu": mu, "k": 0.2}), encoding="utf-8")
    (folder / "gr.json").write_text(json.dumps({"b": 1.2, "m0": 4.5}), encoding="utf-8")
    (folder / "calibration.json").write_text(
        json.dumps({"scale": 1.5, "train_end": "2018-12-31"}), encoding="utf-8"
    )


def test_register_same_brain_is_idempotent(tmp_path):
    _write_brain(tmp_path)
    first = register_current(tmp_path)
    second = register_current(tmp_path)
    assert first["id"] == second["id"]
    assert first["brain_hash"] == second["brain_hash"]
    assert first["code_hash"]
    assert first["model_ids"] == ["fresh.mainshock.v1", "alarm.spf.v2"]
    assert (model_dir(tmp_path) / "registry" / first["id"] / "etas.json").exists()


def test_changed_brain_keeps_the_old_snapshot(tmp_path):
    _write_brain(tmp_path, mu=0.1)
    old = preserve_current(tmp_path)
    assert old is not None
    _write_brain(tmp_path, mu=0.9)
    new = register_current(tmp_path)
    assert new["id"] != old["id"]
    frozen = json.loads(
        (model_dir(tmp_path) / "registry" / old["id"] / "etas.json").read_text(encoding="utf-8")
    )
    assert frozen["mu"] == 0.1
    live = json.loads((model_dir(tmp_path) / "etas.json").read_text(encoding="utf-8"))
    assert live["mu"] == 0.9


def test_stamp_and_batch_sidecar(tmp_path):
    _write_brain(tmp_path)
    entry = ensure_registered(tmp_path)
    dest = prediction_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    batch = dest / "batch-holdout.json"
    cards = [
        prediction(
            prediction_id="x",
            issued_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2020, 1, 8, tzinfo=timezone.utc),
            horizon_days=7,
            region={"kind": "named", "name": "global"},
            magnitude_min=7.0,
            kind="probability",
            value=0.1,
            model_id="fusion.etas_x_mod.v1",
            drivers=["etas_calibrated"],
        )
    ]
    batch.write_text(json.dumps(cards), encoding="utf-8")
    written = pin_unpinned_batches(tmp_path, entry)
    assert written
    meta = json.loads((dest / "batch-holdout.meta.json").read_text(encoding="utf-8"))
    assert meta["registry_id"] == entry["id"]
    stamped = stamp_cards(cards, entry)
    assert stamped[0]["registry_id"] == entry["id"]
    assert stamped[0]["code_hash"] == entry["code_hash"]


def test_prediction_batches_skip_meta_sidecars(tmp_path):
    dest = prediction_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "batch-holdout.json").write_text("[]", encoding="utf-8")
    (dest / "batch-holdout.meta.json").write_text(
        '{"registry_id": "x", "n_cards": 0}', encoding="utf-8"
    )
    names = [path.name for path in prediction_batches(tmp_path)]
    assert names == ["batch-holdout.json"]


def test_schema_allows_registry_fields():
    from eqpred.paths import repo_root

    schema = json.loads((repo_root() / "schemas" / "prediction.schema.json").read_text(encoding="utf-8"))
    assert "registry_id" in schema["properties"]
    assert "code_hash" in schema["properties"]
    assert "competitor_climatology_p" in schema["properties"]
    assert all(name.endswith(".json") for name in MODEL_FILES)
