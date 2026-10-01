from __future__ import annotations

"""Southern Cross laser / Haast DAS — partnership, not a wget.

GNS / Southern Cross fibre work is real. There is no 20-year public
archive. This stub records that fact so the lab does not pretend a
download is running.
"""

from datetime import datetime, timedelta

import pandas as pd

from eqpred.stamp import utc_ts

SOURCE = "das.haast"
LICENSE = "https://www.gns.cri.nz/"
LATENCY = timedelta(days=30)
NOTE = (
    "Southern Cross / Haast DAS is a partnership archive, not a public wget. "
    "No files were downloaded."
)


def pull_das(start: datetime, end: datetime, run_id: str) -> pd.DataFrame:
    _ = (run_id, utc_ts(start), utc_ts(end))
    print(f"  das.haast skipped: {NOTE}", flush=True)
    return pd.DataFrame()
