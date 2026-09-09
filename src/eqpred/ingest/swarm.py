from __future__ import annotations

from datetime import datetime

import pandas as pd

SOURCE = "esa.swarm"
LICENSE = "https://earth.esa.int/eogateway/missions/swarm"


def pull_swarm(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Swarm / CSES need an ESA or CSES account and a heavy L2 product.

    No lightweight public daily file. Skip honestly rather than scrape a portal.
    """
    print(
        "  esa.swarm skipped: no lightweight public daily product "
        "(needs ESA VirES / CSES account)",
        flush=True,
    )
    return pd.DataFrame()
