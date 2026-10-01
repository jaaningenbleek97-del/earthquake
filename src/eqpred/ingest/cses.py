from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

from eqpred.ingest.http import USER_AGENT
from eqpred.records import iso

SOURCE = "esa.cses"
LICENSE = "https://limadou.ssdc.asi.it/"
CAS_LOGIN = "https://tools.ssdc.asi.it/cas/login"
QUERY = "https://limadou.ssdc.asi.it/query.php"
# HEPD L2 is usually days late once it clears the Italian ground segment.
LATENCY = timedelta(days=7)
MISSION_START = datetime(2018, 2, 2)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_local_env() -> None:
    path = _repo_root() / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    allowed = {"LIMADOU_USERNAME", "LIMADOU_PASSWORD", "CSES_USERNAME", "CSES_PASSWORD"}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or os.environ.get(key):
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def limadou_credentials() -> tuple[str, str] | None:
    _load_local_env()
    user = (os.environ.get("LIMADOU_USERNAME") or os.environ.get("CSES_USERNAME") or "").strip()
    password = (os.environ.get("LIMADOU_PASSWORD") or os.environ.get("CSES_PASSWORD") or "").strip()
    if user and password:
        return user, password
    return None


class _InputParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "input":
            return
        got = {k.lower(): (v or "") for k, v in attrs}
        name = got.get("name")
        if name:
            self.fields[name] = got.get("value", "")


def _opener() -> object:
    return build_opener(HTTPCookieProcessor(CookieJar()))


def _read(opener, url: str, data: bytes | None = None, timeout: int = 15) -> tuple[str, str]:
    req = Request(url, data=data, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return str(resp.geturl()), body
    except HTTPError as exc:
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def cas_session(user: str, password: str):
    opener = _opener()
    login_url = f"{CAS_LOGIN}?{urlencode({'service': QUERY})}"
    _final, html = _read(opener, login_url)
    parser = _InputParser()
    parser.feed(html)
    lt = parser.fields.get("lt")
    if not lt:
        raise RuntimeError("Limadou CAS login form had no lt ticket")
    payload = urlencode(
        {
            "username": user,
            "password": password,
            "lt": lt,
            "warn": parser.fields.get("warn", "true"),
        }
    ).encode()
    _final, html = _read(opener, login_url, data=payload)
    if "name=\"password\"" in html and "login_form" in html:
        raise RuntimeError("Limadou CAS login failed (bad password or account not enabled)")
    return opener, html


def _hrefs(html: str, base: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"""href=["']([^"']+)["']""", html, flags=re.I):
        found.append(urljoin(base, match.group(1)))
    return found


def daily_cses_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Daily mean HEPD electron / proton flux from a logged-in table."""
    if samples.empty:
        return pd.DataFrame()
    work = samples.copy()
    if "observed_at" not in work.columns:
        return pd.DataFrame()
    work["observed_at"] = pd.to_datetime(work["observed_at"], utc=True, errors="coerce")
    flux_e = next((c for c in work.columns if c in {"cses_hepd_e", "electron", "e_flux"}), None)
    flux_p = next((c for c in work.columns if c in {"cses_hepd_p", "proton", "p_flux"}), None)
    if flux_e is None and flux_p is None:
        return pd.DataFrame()
    work = work.dropna(subset=["observed_at"])
    if work.empty:
        return pd.DataFrame()
    work["day"] = work["observed_at"].dt.floor("D")
    grouped = work.groupby("day", sort=True)
    out = pd.DataFrame({"observed_at": grouped.size().index})
    if flux_e is not None:
        out["cses_hepd_e"] = grouped[flux_e].mean().to_numpy()
    if flux_p is not None:
        out["cses_hepd_p"] = grouped[flux_p].mean().to_numpy()
    out["source"] = SOURCE
    out["license"] = LICENSE
    out["available_at"] = [iso(ts.to_pydatetime() + LATENCY) for ts in out["observed_at"]]
    out["observed_at"] = [iso(ts.to_pydatetime()) for ts in out["observed_at"]]
    return out.reset_index(drop=True)


def pull_cses(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    """CSES HEPD counts via Italian Limadou CAS. LEOS is not used."""
    _ = (start, end, ingest_run_id)
    creds = limadou_credentials()
    if creds is None:
        print(
            "  esa.cses skipped: set LIMADOU_USERNAME and LIMADOU_PASSWORD in .env "
            "(Italian SSDC / Limadou, not the Chinese LEOS portal). Do not paste them in chat.",
            flush=True,
        )
        return pd.DataFrame()
    user, password = creds
    try:
        opener, _html = cas_session(user, password)
        final, html = _read(opener, QUERY)
    except Exception as exc:  # noqa: BLE001 — optional overlay
        print(f"  esa.cses skipped: {exc}", flush=True)
        return pd.DataFrame()

    links = [u for u in _hrefs(html, final) if re.search(r"hepd|csv|cdf|download|l2", u, re.I)]
    print(
        f"  esa.cses logged in to Limadou ({final}); "
        f"{len(links)} product links on query.php — HEPD daily parse waits on a stable file URL",
        flush=True,
    )
    if "login_form" in html:
        print(
            "  esa.cses: CAS bounced back to login. Account may still need CSES-Data access "
            "(email alessandro.sotgiu@roma2.infn.it).",
            flush=True,
        )
    return pd.DataFrame()
