from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd

from eqpred.ingest.http import fetch_text
from eqpred.records import iso
from eqpred.stamp import utc_ts

# Daily averages — same column order as hourly OMNI2.
OMNI_DAILY_URLS = (
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_01_av.dat",
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_daily.dat",
)
LICENSE = "https://omniweb.gsfc.nasa.gov/html/ow_data.html"
LATENCY = timedelta(days=3)

# Fill sentinels from the OMNI2 description.
FILLS = {
    "proton_density_cm3": 999.9,
    "solar_wind_speed_kms": 9999.0,
    "imf_bz_nt": 999.9,
    "imf_bx_nt": 999.9,
    "imf_by_nt": 999.9,
    "imf_b_nt": 999.9,
    "proton_temp_k": 9999999.0,
    "flow_pressure_npa": 99.99,
    "electric_field_mvm": 999.99,
    "plasma_beta": 999.99,
    "kp": 99,
    "dst_nt": 99999,
    "ae_nt": 9999,
    "al_nt": 99999,
    "au_nt": 99999,
    "pc_index": 999.9,
    "ap": 999,
    "f107": 999.9,
    "sunspot_number": 999,
}


def _as_float(value: str, fill: float) -> float | None:
    try:
        num = float(value)
    except ValueError:
        return None
    if abs(num - fill) < 1e-6 or num >= fill:
        return None
    return num


def parse_omni_daily(text: str) -> pd.DataFrame:
    """Parse OMNI daily or hourly-style whitespace rows to one row per day."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 51:
            continue
        try:
            year = int(parts[0])
            doy = int(parts[1])
            hour = int(float(parts[2]))
        except ValueError:
            continue
        observed = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        proton = _as_float(parts[23], FILLS["proton_density_cm3"])
        speed = _as_float(parts[24], FILLS["solar_wind_speed_kms"])
        bz = _as_float(parts[16], FILLS["imf_bz_nt"])
        kp_raw = _as_float(parts[38], FILLS["kp"])
        dst = _as_float(parts[40], FILLS["dst_nt"])
        ap = _as_float(parts[49], FILLS["ap"])
        f107 = _as_float(parts[50], FILLS["f107"])
        extras: dict[str, float | None] = {}
        if len(parts) >= 42:
            extras["imf_b_nt"] = _as_float(parts[8], FILLS["imf_b_nt"])
            extras["imf_bx_nt"] = _as_float(parts[12], FILLS["imf_bx_nt"])
            extras["imf_by_nt"] = _as_float(parts[15], FILLS["imf_by_nt"])
            extras["proton_temp_k"] = _as_float(parts[22], FILLS["proton_temp_k"])
            extras["flow_pressure_npa"] = _as_float(parts[28], FILLS["flow_pressure_npa"])
            extras["electric_field_mvm"] = _as_float(parts[35], FILLS["electric_field_mvm"])
            extras["plasma_beta"] = _as_float(parts[36], FILLS["plasma_beta"])
            extras["ae_nt"] = _as_float(parts[41], FILLS["ae_nt"])
        if len(parts) >= 54:
            extras["pc_index"] = _as_float(parts[51], FILLS["pc_index"])
            extras["al_nt"] = _as_float(parts[52], FILLS["al_nt"])
            extras["au_nt"] = _as_float(parts[53], FILLS["au_nt"])
        rows.append(
            {
                "source": "nasa.omni",
                "observed_at": iso(observed),
                "available_at": iso(observed + LATENCY),
                "license": LICENSE,
                "hour": hour,
                "proton_density_cm3": proton,
                "solar_wind_speed_kms": speed,
                "imf_bz_nt": bz,
                "kp": None if kp_raw is None else kp_raw / 10.0,
                "dst_nt": dst,
                "ap": ap,
                "f107": f107,
                **extras,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["observed_at", "hour"])
    # Prefer hour-0 (daily file / first hourly sample); one row per UTC day.
    return frame.drop_duplicates("observed_at", keep="first").drop(columns=["hour"]).reset_index(drop=True)


def pull_omni(start: datetime, end: datetime, ingest_run_id: str) -> pd.DataFrame:
    last_error: Exception | None = None
    frame = pd.DataFrame()
    for url in OMNI_DAILY_URLS:
        try:
            text = fetch_text(url, timeout=180)
            frame = parse_omni_daily(text)
            if frame.empty:
                frame = _parse_fallback(text)
            if not frame.empty:
                break
        except Exception as exc:  # noqa: BLE001 — try the next archive URL
            last_error = exc
    if frame.empty:
        raise RuntimeError(f"OMNI daily download failed: {last_error}")
    frame["ingest_run_id"] = ingest_run_id
    stamp = pd.to_datetime(frame["observed_at"], utc=True)
    mask = (stamp >= utc_ts(start)) & (stamp < utc_ts(end))
    return frame.loc[mask].reset_index(drop=True)


def _parse_fallback(text: str) -> pd.DataFrame:
    raw = pd.read_csv(StringIO(text), sep=r"\s+", header=None, engine="python")
    if raw.shape[1] < 51:
        return pd.DataFrame()
    return parse_omni_daily(text)
