import json
import math
from pathlib import Path

import pytest

from eqpred.paths import repo_root
from eqpred.models.attribution import (
    alarm_attrs,
    anti_attrs,
    ensure_attrs,
    flatten_driver_field,
    fusion_attrs,
    lag_hours,
    log_contrib,
)
from eqpred.records import prediction
from datetime import datetime, timezone


def test_lag_hours_are_locked_feature_windows():
    assert lag_hours("spf_significant_window") == 40 * 24
    assert lag_hours("catalog_nowcast") == 7 * 24
    assert lag_hours("gnss_east_7d") == 7 * 24
    assert lag_hours("mig_along_hikurangi") == 14 * 24
    assert lag_hours("coulomb_hikurangi") == 0
    assert lag_hours("etas_calibrated") == 0


def test_log_contrib_is_ln_gain():
    assert log_contrib(1.0) == 0.0
    assert log_contrib(math.e) == pytest.approx(1.0, abs=1e-6)
    assert log_contrib(0.0) == 0.0


def test_fusion_attrs_omit_empty_and_sort_by_abs():
    rows = fusion_attrs({"etas_calibrated": 1.2, "tec_tecu": 0.9, "catalog_nowcast": 1.0})
    names = [r["driver"] for r in rows]
    assert names[0] in {"etas_calibrated", "tec_tecu"}
    assert abs(rows[0]["contribution"]) >= abs(rows[1]["contribution"])
    tec = next(r for r in rows if r["driver"] == "tec_tecu")
    assert tec["contribution"] == pytest.approx(math.log(0.9), abs=1e-6)
    assert tec["lag_hours"] == 0
    assert "spf_significant_window" not in names


def test_alarm_and_anti_attrs():
    spf = alarm_attrs("spf_significant_window")
    assert spf == [
        {"driver": "spf_significant_window", "lag_hours": 960, "contribution": 1.0}
    ]
    anti = anti_attrs(0.05, 0.10)
    assert anti[0]["driver"] == "anti_quiet"
    assert anti[1]["driver"] == "issued_p_low"
    assert anti[1]["contribution"] == pytest.approx(0.5)


def test_ensure_attrs_coerces_strings():
    out = ensure_attrs(["proton_spike"])
    assert out[0]["driver"] == "proton_spike"
    assert out[0]["lag_hours"] == 0
    with pytest.raises(ValueError):
        ensure_attrs([])


def test_prediction_record_is_structured():
    rec = prediction(
        prediction_id="x",
        issued_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2020, 1, 8, tzinfo=timezone.utc),
        horizon_days=7,
        region={"kind": "named", "name": "global"},
        magnitude_min=6.0,
        kind="probability",
        value=0.12,
        model_id="fusion.etas_x_mod.v1",
        drivers=["spf_significant_window"],
    )
    assert rec["drivers"][0]["driver"] == "spf_significant_window"
    assert rec["drivers"][0]["lag_hours"] == 960


def test_schema_allows_region_label():
    schema = json.loads((Path(repo_root()) / "schemas" / "prediction.schema.json").read_text(encoding="utf-8"))
    assert "label" in schema["properties"]["region"]["properties"]
    assert "registry_id" in schema["properties"]
    assert "code_hash" in schema["properties"]


def test_flatten_accepts_objects_or_strings():
    assert flatten_driver_field(["etas_quiet", {"driver": "proton_spike"}]) == [
        "etas_quiet",
        "proton_spike",
    ]
