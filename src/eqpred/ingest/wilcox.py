from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso

WILCOX_URL = "http://wso.stanford.edu/Polar.html"
LICENSE = "http://wso.stanford.edu/Polar.html"
LATENCY = timedelta(days=40)

LINE_RE = re.compile(
    r"(?P<y>\d{4}):(?P<m>\d{2}):(?P<d>\d{2})_\S+\s+"
    r"(?P<n>[+-]?\d+)N\s+(?P<s>[+-]?\d+)S\s+(?P<avg>[+-]?\d+)Avg"
)


def parse_wilcox(text: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        observed = datetime(
            int(match.group("y")),
            int(match.group("m")),
            int(match.group("d")),
            tzinfo=timezone.utc,
        )
        north = float(match.group("n"))
        south = float(match.group("s"))
        avg = float(match.group("avg"))
        rows.append(
            {
                "source": "wilcox.spf",
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "spf_north_ut": north,
                "spf_south_ut": south,
                "spf_average_ut": avg,
            }
        )
    return pd.DataFrame(rows)


def pull_wilcox(ingest_run_id: str) -> pd.DataFrame:
    frame = parse_wilcox(fetch_text(WILCOX_URL, timeout=60))
    frame["ingest_run_id"] = ingest_run_id
    return frame
