import struct

import numpy as np

from eqpred.eval.mainshock import _logit
from eqpred.ingest.unwin import scan_fitacf


def _fitacf_bytes() -> bytes:
    def cstr(name: bytes) -> bytes:
        return name + b"\x00"

    body = b""
    for name, kind, packed in (
        (b"time.yr", 2, struct.pack("<h", 2018)),
        (b"time.mo", 2, struct.pack("<h", 1)),
        (b"time.dy", 2, struct.pack("<h", 2)),
    ):
        body += cstr(name) + bytes([kind]) + packed
    for name, kind, count, packed in (
        (b"qflg", 1, 2, struct.pack("2B", 1, 1)),
        (b"gflg", 1, 2, struct.pack("2B", 0, 1)),
        (b"v", 4, 2, struct.pack("<2f", 100.0, -20.0)),
    ):
        body += cstr(name) + bytes([kind]) + struct.pack("<ii", 1, count) + packed
    size = 16 + len(body)
    header = struct.pack("<4i", 65537, size, 3, 3)
    return header + body


def test_ionospheric_share_ignores_ground_scatter():
    days = scan_fitacf(_fitacf_bytes())
    qual, iono, vsum, nv = next(iter(days.values()))
    assert qual == 2
    assert iono == 1
    assert nv == 1
    assert vsum == 100.0


def test_blank_morning_is_not_scored_as_an_average_reading():
    x = np.array(
        [
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 10.0],
            [1.0, 10.0],
            [1.0, 10.0],
            [1.0, 10.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = _logit(C=1.0)
    model.fit(x, y)
    blank = float(model.predict_proba(np.array([[0.0, np.nan]]))[0, 1])
    average = float(model.predict_proba(np.array([[0.0, 0.0]]))[0, 1])
    assert abs(blank - average) > 1e-4
