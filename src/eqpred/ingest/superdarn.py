from __future__ import annotations

"""SuperDARN Unwin (Invercargill) FITACF inventory.

Unwin (`unw`) is the South Island HF radar, running since 2004. CDAWeb's
`darn/` dump has older radars only — gbay/hank/icew/kapu/pace/pykk/sask —
so this probe walks La Trobe / VT / NIPR / JHUAPL listings. A 20-year
FITACF wget is not started unless a public HTTP listing actually answers.
"""

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import pandas as pd

from eqpred.ingest.http import fetch_bytes, fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "superdarn.unwin"
LICENSE = "https://www.superdarn.ac.uk/"
LATENCY = timedelta(days=3)
RADAR = "unw"
MIRRORS = (
    "http://sd-data.ece.vt.edu/pub/fitacf/",
    "http://sd-data.ece.vt.edu/common/dtdata/fitacf/",
    "https://data.superdarn.latrobe.edu.au/",
    "https://data.superdarn.latrobe.edu.au/fitacf/",
    "http://polaris.nipr.ac.jp/~sdarn/sd-data/fitacf/",
    "https://data-out.superdarn.jhuapl.edu/fitacf/",
    "https://superdarn.jhuapl.edu/data/fitacf/",
)
FILE_RE = re.compile(r'href="([^"]*(?:unw|Unwin)[^"]*\.(?:fitacf|bz2|gz|gz\.fitacf)[^"]*)"', re.I)
HREF_RE = re.compile(r'href="([^"]+)"', re.I)
CDAWEB_DARN = "https://cdaweb.gsfc.nasa.gov/pub/data/aaa_radioscience/darn/"


def _list(url: str, *, fetch_page=fetch_text) -> tuple[str, list[str]]:
    try:
        html = fetch_page(url, timeout=25)
    except Exception:
        return url, []
    hrefs = []
    for href in HREF_RE.findall(html or ""):
        if href.startswith("?") or href.startswith("#"):
            continue
        hrefs.append(urljoin(url, href))
    return url, hrefs


def discover_unwin(*, fetch_page=fetch_text) -> dict[str, object]:
    """Return the first mirror that lists Unwin files, plus CDAWeb note."""
    cda_url, cda_hrefs = _list(CDAWEB_DARN, fetch_page=fetch_page)
    cda_names = [h.rstrip("/").split("/")[-1].lower() for h in cda_hrefs]
    note = "CDAWeb darn/ has no Unwin" if "unw" not in cda_names else "CDAWeb lists Unwin"
    for base in MIRRORS:
        for candidate in (
            base,
            urljoin(base, "unw/"),
            urljoin(base, "Unwin/"),
            urljoin(base, f"{datetime.now(timezone.utc).year}/unw/"),
        ):
            url, hrefs = _list(candidate, fetch_page=fetch_page)
            files = [h for h in hrefs if FILE_RE.search(h) or h.lower().endswith((".fitacf", ".bz2"))]
            if files:
                return {
                    "mirror": url,
                    "n_files": len(files),
                    "sample": files[:8],
                    "cdaweb": cda_url,
                    "cdaweb_note": note,
                    "blocked": False,
                }
    return {
        "mirror": None,
        "n_files": 0,
        "sample": [],
        "cdaweb": cda_url,
        "cdaweb_note": note,
        "blocked": True,
        "note": (
            "SuperDARN Unwin is not on CDAWeb and no public HTTP FITACF listing answered. "
            "La Trobe / VT Globus login is the usual path — not a 20-year wget."
        ),
    }


def pull_superdarn(start: datetime, end: datetime, run_id: str, *, fetch_page=fetch_text, fetch=fetch_bytes) -> pd.DataFrame:
    _ = (run_id, fetch, fetch_page, utc_ts(start), utc_ts(end))
    print(
        "  superdarn.unwin skipped: Unwin FITACF needs a La Trobe / VT Globus account "
        "(not on CDAWeb). No mirror walk.",
        flush=True,
    )
    return pd.DataFrame()


def _stamp_from_name(name: str) -> datetime | None:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None
