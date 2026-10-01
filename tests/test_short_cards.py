from eqpred.eval.horizon import HORIZON_DAYS
from eqpred.eval.mainshock import SHORT_CARD_HORIZONS


def test_short_cards_sit_beside_the_30_day_horizon():
    assert HORIZON_DAYS == 30
    assert SHORT_CARD_HORIZONS == (7, 3)
