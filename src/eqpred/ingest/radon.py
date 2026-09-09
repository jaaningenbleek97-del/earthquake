from __future__ import annotations

from datetime import datetime

import pandas as pd

SOURCE = "radon.public"
LICENSE = "https://www.usgs.gov/faqs/can-you-predict-earthquakes"


def pull_radon(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """No lightweight public daily radon series we can timestamp as-of.

    Claimed radon precursors are not a public daily product. Skip honestly
    rather than scrape a portal.
    """
    _ = (start, end, ingest_run_id)
    print(
        "  radon.public skipped: no lightweight public daily radon product",
        flush=True,
    )
    return pd.DataFrame()
