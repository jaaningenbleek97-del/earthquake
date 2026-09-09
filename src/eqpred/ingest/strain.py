from __future__ import annotations

from datetime import datetime

import pandas as pd

SOURCE = "earthscope.strain"
LICENSE = "https://www.unavco.org/data/strain-seismo/bsm/bsm.html"


def pull_strain(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """EarthScope / UNAVCO borehole strain is not a lightweight public daily.

    Level-2 PBO files need an account or a heavy archive pull. Skip honestly
    rather than scrape a portal.
    """
    _ = (start, end, ingest_run_id)
    print(
        "  earthscope.strain skipped: no lightweight public daily borehole-strain file "
        "(needs EarthScope / UNAVCO archive access)",
        flush=True,
    )
    return pd.DataFrame()
