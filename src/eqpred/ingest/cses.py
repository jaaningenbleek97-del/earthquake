from __future__ import annotations

from datetime import datetime

import pandas as pd

SOURCE = "esa.cses"
LICENSE = "https://www.leos.ac.cn/"


def pull_cses(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """CSES needs a CSES / ESA account and a heavy L2 product.

    No lightweight public daily file. Skip honestly rather than scrape a portal.
    """
    _ = (start, end, ingest_run_id)
    print(
        "  esa.cses skipped: no lightweight public daily product "
        "(needs CSES / ESA account)",
        flush=True,
    )
    return pd.DataFrame()
