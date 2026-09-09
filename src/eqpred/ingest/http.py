from __future__ import annotations

import json
import netrc
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

URS_HOST = "urs.earthdata.nasa.gov"
USER_AGENT = "eqpred/0.1 (earthquake prediction lab)"


def _looks_like_html(data: bytes) -> bool:
    head = data.lstrip()[:400].lower()
    return (
        head.startswith(b"<!doctype")
        or head.startswith(b"<html")
        or b"<title>" in head
        or b"earthdata login" in head
    )


def earthdata_credentials() -> tuple[str, str] | None:
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if user and password:
        return user, password
    home = Path.home()
    for path in (
        Path(os.environ["NETRC"]) if os.environ.get("NETRC") else None,
        home / "_netrc",
        home / ".netrc",
    ):
        if path is None or not path.is_file():
            continue
        try:
            auth = netrc.netrc(str(path)).authenticators(URS_HOST)
        except (netrc.NetrcParseError, OSError):
            continue
        if auth and auth[0] and auth[2]:
            return auth[0], auth[2]
    return None


def netrc_path() -> Path | None:
    home = Path.home()
    for path in (
        Path(os.environ["NETRC"]) if os.environ.get("NETRC") else None,
        home / "_netrc",
        home / ".netrc",
    ):
        if path is not None and path.is_file():
            return path
    return None


def _earthdata_netrc() -> Path:
    existing = netrc_path()
    if existing is not None:
        return existing
    creds = earthdata_credentials()
    if creds is None:
        raise RuntimeError(
            "Earthdata login missing: set EARTHDATA_USERNAME / EARTHDATA_PASSWORD "
            "or a home _netrc for urs.earthdata.nasa.gov"
        )
    user, password = creds
    handle = tempfile.NamedTemporaryFile(
        prefix="eqpred_netrc_",
        delete=False,
        mode="w",
        encoding="ascii",
    )
    handle.write(f"machine {URS_HOST}\n    login {user}\n    password {password}\n")
    handle.close()
    path = Path(handle.name)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _fetch_earthdata(url: str, timeout: int) -> bytes:
    """CDDIS uses an OAuth form redirect. curl --netrc follows it; urllib does not."""
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise RuntimeError("curl is required to download CDDIS IONEX with Earthdata login")
    netrc_file = _earthdata_netrc()
    cookie = Path(tempfile.gettempdir()) / "eqpred_urs_cookies"
    dest = Path(tempfile.gettempdir()) / f"eqpred_cddis_{uuid4().hex}"
    cmd = [
        curl,
        "-sS",
        "-L",
        "-f",
        "--max-time",
        str(timeout),
        "-A",
        USER_AGENT,
        "-n",
        "--netrc-file",
        str(netrc_file),
        "-c",
        str(cookie),
        "-b",
        str(cookie),
        "-o",
        str(dest),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"curl exit {proc.returncode}").strip()
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"GET {url} failed: {err}")
    data = dest.read_bytes()
    dest.unlink(missing_ok=True)
    if _looks_like_html(data):
        raise RuntimeError(
            f"GET {url} failed: Earthdata returned login HTML — "
            "open https://cddis.nasa.gov/archive/ once, authorize CDDIS, then retry"
        )
    return data


def fetch_bytes(
    url: str,
    timeout: int = 120,
    *,
    earthdata: bool = False,
    data: bytes | dict | None = None,
) -> bytes:
    if earthdata:
        return _fetch_earthdata(url, timeout)
    body: bytes | None
    if data is None:
        body = None
    elif isinstance(data, (bytes, bytearray)):
        body = bytes(data)
    else:
        body = urlencode(data, doseq=True).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def fetch_text(url: str, timeout: int = 120, *, earthdata: bool = False) -> str:
    return fetch_bytes(url, timeout=timeout, earthdata=earthdata).decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int = 120) -> object:
    return json.loads(fetch_text(url, timeout=timeout))


def write_cache(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
