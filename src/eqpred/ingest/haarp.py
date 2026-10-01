from __future__ import annotations

"""HAARP research-campaign on/off flag.

Official UAF campaign pages plus transmission-notice text. This is a
negative-control overlay: HF ionospheric heating does not couple into the
crust. Days are 1 only when a published campaign window covers them.
"""

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

SOURCE = "gi.haarp"
LICENSE = "https://haarp.gi.alaska.edu/"
LATENCY = timedelta(days=1)
PAGES = (
    "https://haarp.gi.alaska.edu/campaigns",
    "https://haarp.gi.alaska.edu/",
    "https://haarp.gi.alaska.edu/faq",
    "https://haarp.gi.alaska.edu/transmissions",
)
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December)"


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# Official notices that /campaigns does not list as date ranges.
EXTRA_WINDOWS: tuple[tuple[datetime, datetime, str], ...] = (
    (_utc(2024, 10, 15), _utc(2024, 10, 17), "October 15-17, 2024"),
    (_utc(2025, 1, 27), _utc(2025, 1, 31), "January 27-31, 2025"),
    (_utc(2025, 7, 23), _utc(2025, 7, 24), "July 23-24, 2025"),
    (_utc(2025, 8, 5), _utc(2025, 8, 14), "PARS 2025 August 5-14"),
    (_utc(2025, 11, 17), _utc(2025, 11, 22), "November 17-22, 2025"),
)


def _month_end(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (_utc(year, month + 1, 1) - timedelta(days=1)).day


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def parse_campaign_windows(text: str) -> list[tuple[datetime, datetime, str]]:
    """Published HAARP campaign date ranges from page text."""
    if not text:
        return []
    found: dict[tuple[datetime, datetime], str] = {}

    def add(start: datetime, end: datetime, label: str) -> None:
        if end < start:
            start, end = end, start
        found[(start, end)] = label

    cross = re.compile(
        rf"{_MONTH}\s+(\d{{1,2}})\s*[-–]\s*{_MONTH}\s+(\d{{1,2}}),?\s+(\d{{4}})",
        re.I,
    )
    for match in cross.finditer(text):
        y = int(match.group(5))
        add(
            _utc(y, MONTHS[match.group(1).lower()], int(match.group(2))),
            _utc(y, MONTHS[match.group(3).lower()], int(match.group(4))),
            match.group(0),
        )
    same = re.compile(
        rf"{_MONTH}\s+(\d{{1,2}})\s*[-–]\s*(\d{{1,2}}),?\s+(\d{{4}})",
        re.I,
    )
    for match in same.finditer(text):
        y = int(match.group(4))
        month = MONTHS[match.group(1).lower()]
        add(
            _utc(y, month, int(match.group(2))),
            _utc(y, month, int(match.group(3))),
            match.group(0),
        )
    single = re.compile(rf"{_MONTH}\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I)
    spanned = {match.group(0) for match in list(cross.finditer(text)) + list(same.finditer(text))}
    for match in single.finditer(text):
        if any(match.group(0) in chunk for chunk in spanned):
            continue
        y = int(match.group(3))
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        add(_utc(y, month, day), _utc(y, month, day), match.group(0))
    month_only = re.compile(rf"{_MONTH}\s+(\d{{4}})\s+Campaign", re.I)
    for match in month_only.finditer(text):
        y = int(match.group(2))
        month = MONTHS[match.group(1).lower()]
        add(_utc(y, month, 1), _utc(y, month, _month_end(y, month)), match.group(0))
    return sorted((a, b, label) for (a, b), label in found.items())


def windows_to_daily(
    windows: list[tuple[datetime, datetime, str]],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    days = pd.date_range(utc_ts(start), utc_ts(end), freq="D", inclusive="left")
    if days.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day in days:
        when = day.to_pydatetime()
        hits = [label for a, b, label in windows if a <= when <= b]
        rows.append(
            {
                "source": SOURCE,
                "observed_at": iso(when),
                "available_at": iso(when),
                "license": LICENSE,
                "haarp_on": 1.0 if hits else 0.0,
                "haarp_campaigns": "; ".join(dict.fromkeys(hits)),
            }
        )
    return pd.DataFrame(rows)


def pull_haarp(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """Daily HAARP on/off from official campaign pages."""
    texts: list[str] = []
    for url in PAGES:
        try:
            html = fetch_text(url, timeout=30)
        except Exception as exc:  # noqa: BLE001 — optional negative control
            print(f"  gi.haarp skip {url}: {exc}", flush=True)
            continue
        texts.append(_strip_html(html))
        print(f"  gi.haarp {url}", flush=True)
    found = {(a, b): label for a, b, label in parse_campaign_windows(" ".join(texts))}
    for start_w, end_w, label in EXTRA_WINDOWS:
        found.setdefault((start_w, end_w), label)
    windows = sorted((a, b, label) for (a, b), label in found.items())
    print(f"  gi.haarp {len(windows)} campaign windows", flush=True)
    daily = windows_to_daily(windows, start, end)
    if daily.empty:
        return daily
    daily["ingest_run_id"] = ingest_run_id
    return daily
