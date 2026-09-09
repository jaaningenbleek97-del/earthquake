import pytest

from eqpred.models.human import HUMAN_MODEL, write_human_card


def test_human_card_writes_model_id_human(tmp_path):
    card = write_human_card(
        region_id="global",
        magnitude_min=7.0,
        horizon_days=7,
        value=0.22,
        note="lab discretionary",
        issued_at="2026-09-07",
        root=tmp_path,
    )
    assert card["model_id"] == HUMAN_MODEL
    assert card["value"] == 0.22
    assert card["region"]["name"] == "global"
    path = tmp_path / "data" / "predictions" / "batch-human-2026-09-07.json"
    assert path.exists()


def test_human_card_refuses_transfer_only_region(tmp_path):
    with pytest.raises(ValueError, match="transfer-only"):
        write_human_card(region_id="japan", value=0.4, root=tmp_path)
