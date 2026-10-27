"""Research ideas, chosen on old years and checked once on 2009 onward.

Each idea turns data known by the morning into one daily number. Odds come from
windows that closed before 1990. 1990–2008 picks the variation. 2009 onward is
scored once and picks nothing. A chance only moves the usual rate by the odds
the idea earned on old years. Nothing here changes the live exam.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from eqpred.eval.attempts import attempts_dir, forecast_frame, save_forecasts
from eqpred.eval.decluster import decluster_causal, haversine_km, uhrhammer_km
from eqpred.eval.honest import week_mask
from eqpred.eval.separate_scores import LABELS, split_windows
from eqpred.eval.strategy_desk import STRATEGIES, TAPE_FOR_LABEL, load_tape
from eqpred.paths import data_dir, model_dir
from eqpred.regions import REGIONS, in_bbox
from eqpred.safe_io import write_json
from eqpred.stamp import parse_utc

PLACES = ("global", "hikurangi", "alpine", "kermadec")
HORIZONS = (1, 3, 7, 30)
TARGET = {"global": 7.0, "hikurangi": 5.0, "alpine": 5.0, "kermadec": 6.0}
FLOOR = {"global": 5.0, "hikurangi": 3.0, "alpine": 3.0, "kermadec": 4.5}
CLUSTER_FLOOR = {"global": 5.5, "hikurangi": 3.5, "alpine": 3.5, "kermadec": 4.5}
RATE_FLOOR = {"global": 5.0, "hikurangi": 4.0, "alpine": 4.0, "kermadec": 4.5}
CELL_DEG = {"global": 5.0, "hikurangi": 0.25, "alpine": 0.25, "kermadec": 0.5}
SHARES = (0.1, 0.2, 0.3)
PRIOR_WINDOWS = 20.0
USEFUL_GAIN = 0.01
EARLY_END = pd.Timestamp("1990-01-01", tz="UTC")
LATER_START = pd.Timestamp("2009-01-01", tz="UTC")
YEAR = 365.25

IDEAS = (
    ("bpositive", "Stress meter: the b-value drops", "Gulia and Wiemer 2019; van der Elst 2021"),
    ("tidal", "Tides: big quakes favored when tide stress is high", "Ide, Yabe and Tanaka 2016; Yin LURR"),
    ("natural_time", "Natural time near the critical 0.070", "Varotsos, Sarlis and others 2011"),
    ("etas_quiet", "Quieter than the aftershock model expects", "Ogata 1992"),
    ("quiet_aftershocks", "Aftershocks that died out too fast", "Ogata 2001"),
    ("localization", "Small quakes crowd into less space", "Ben-Zion and Zaliapin 2020"),
    ("coalescence", "Small quakes clump into clusters", "Ben-Zion and Zaliapin 2020"),
    ("earth_rotation", "Earth's spin slowed four to six years earlier", "Bendick and Bilham 2017"),
    ("hidden_state", "Hidden active state of the catalog", "Poisson hidden Markov model"),
    ("crowd", "The crowd: every rule already tried, weighted on old years", "Stacked ensemble"),
    ("nowcast_eps", "Nowcasting: small-quake count since the last big shock", "Rundle, Turcotte and Holliday 2016"),
    ("burstiness", "Interevent burstiness and memory between quakes", "Goh and Barabasi 2008; Molchan 2005"),
    ("rtl_score", "Region-Time-Length quiescence and activation", "Sobolev and Tyupkin 1999; Huang 2004"),
    ("m8_tip", "M8 seismic flow and Benioff strain acceleration", "Keilis-Borok and Kossobokov 1990"),
    ("depth_shift", "Hypocentral depth shallowing into the locked zone", "Scholz 2019; Kato and Ben-Zion 2020"),
    ("rake_regime", "Moment-tensor slip alignment and thrust balance", "Global CMT 1976-2026; Wang and Diao 2005"),
    ("polar_wobble", "Chandler polar-motion excitation and wobble torque", "IERS EOP 1964-2026; Varga 2012"),
    ("spatial_migration", "Epicentral centroid drift and diffusivity", "Kato 2012; Brodsky 2011"),
    ("volcano_magmatic", "Volcanic eruption pulse and tectonic coupling", "Smithsonian GVP 1964-2026; Linde and Sacks 1998"),
    ("hedge_expert", "Online expert mixture of all 69 rules, weighted by closed-window skill", "Vovk 1990; Cesa-Bianchi 2006"),
    ("pattern_informatics", "Pattern informatics: spatial cells changing rate together", "Rundle and Tiampo 2002; Holliday 2005"),
    ("amr_curvature", "Accelerating moment release curvature", "Bowman 1998; Hardebeck 2008"),
    ("fractal_entropy", "Fractal dimension and spatial entropy of epicenters", "Kiyashchenko 2003; Murase 2003"),
    ("etas_residual", "ETAS transformed-time residual shape", "Ogata 1988"),
    ("network_graph", "Successive-event network clustering and hubs", "Abe and Suzuki 2004; earthquake-network studies"),
    ("bath_gap", "Bath's-law aftershock magnitude gap", "Bath 1965; Helmstetter and Sornette 2003"),
    ("stochastic_background", "Stochastic-declustering background probability", "Zhuang, Ogata and Vere-Jones 2002"),
    ("moment_balance", "Seismic moment budget deficit and release concentration", "Ward 1998; moment-deficit studies"),
    ("inverse_omori", "Inverse-Omori foreshock acceleration", "Papazachos 1973; Helmstetter and Sornette 2003"),
    ("reversal_entropy", "Natural-time entropy change under time reversal", "Sarlis, Skordas and Varotsos 2018"),
)
IDEA_LABELS = {name: label for name, label, _source in IDEAS}


def bench_path(root: Path | None = None) -> Path:
    return data_dir(root) / "eval" / "idea_bench.json"


# --- small helpers ---------------------------------------------------------


def _logit(p: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _at_mornings(known: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Value after the latest event that each morning could see."""
    out = np.full(n, np.nan)
    if len(known) == 0:
        return out
    idx = np.searchsorted(known, np.arange(n), side="right") - 1
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _last_n_mean(x: np.ndarray, size: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < size or size <= 0:
        return out
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out[size - 1 :] = (cs[size:] - cs[:-size]) / size
    return out


def _daily_sum(known: np.ndarray, weights: np.ndarray | None, n: int) -> np.ndarray:
    keep = known < n
    w = None if weights is None else np.asarray(weights, dtype=float)[keep]
    return np.bincount(known[keep], weights=w, minlength=n)[:n].astype(float)


def _trailing(daily: np.ndarray, width: int) -> np.ndarray:
    """Sum over the mornings (i - width, i]. Shorter histories stay blank."""
    n = len(daily)
    out = np.full(n, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(daily)])
    idx = np.arange(n)
    ok = idx + 1 >= width
    out[ok] = cs[idx[ok] + 1] - cs[idx[ok] + 1 - width]
    return out


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if lag <= 0:
        return values.copy()
    if lag < len(values):
        out[lag:] = values[:-lag]
    return out


# --- data -------------------------------------------------------------------


def _events(frame: pd.DataFrame, start: pd.Timestamp, n: int, floor: float) -> dict[str, np.ndarray]:
    """Events above the floor, in time order, with the first morning that could use each one."""
    work = frame.loc[pd.to_numeric(frame["magnitude"], errors="coerce") >= floor].copy()
    if work.empty:
        empty = np.zeros(0)
        return {"t": empty, "known": empty.astype(np.int64), "mag": empty, "lat": empty, "lon": empty, "depth": empty}
    work["observed_at"] = parse_utc(work["observed_at"])
    avail = parse_utc(work["available_at"]) if "available_at" in work.columns else work["observed_at"]
    work = work.assign(_avail=avail).dropna(subset=["observed_at", "_avail", "magnitude", "lat", "lon"])
    work = work.sort_values("observed_at")
    t = ((work["observed_at"] - start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    seen = ((work["_avail"] - start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    known = np.floor(seen).astype(np.int64) + 1
    known = np.clip(np.maximum.accumulate(known), 0, n)
    depth = (
        pd.to_numeric(work["depth_km"], errors="coerce").fillna(33.0).to_numpy(dtype=float)
        if "depth_km" in work.columns
        else np.full(len(work), 33.0)
    )
    return {
        "t": t,
        "known": known,
        "mag": pd.to_numeric(work["magnitude"], errors="coerce").to_numpy(dtype=float),
        "lat": work["lat"].to_numpy(dtype=float),
        "lon": work["lon"].to_numpy(dtype=float),
        "depth": depth,
    }


def _box(frame: pd.DataFrame, place: str) -> pd.DataFrame:
    region = REGIONS[place]
    if region.get("kind") != "bbox":
        return frame
    return frame.loc[in_bbox(frame, region)]


def _catalogs(root: Path | None) -> dict[str, pd.DataFrame]:
    from eqpred.catalog import load_comcat
    from eqpred.ingest.run import latest_raw

    usgs = load_comcat(root)
    geonet = latest_raw("geonet.fdsn", root)
    out = {}
    for name, frame in (("usgs", usgs), ("geonet", geonet)):
        work = frame.copy()
        if "lat" not in work.columns and "latitude" in work.columns:
            work["lat"] = work["latitude"]
            work["lon"] = work["longitude"]
        out[name] = work
    return out


def _place_catalog(catalogs: dict[str, pd.DataFrame], place: str) -> pd.DataFrame:
    """New Zealand boxes use GeoNet for the small quakes. The rest use the USGS list."""
    source = "geonet" if place in {"hikurangi", "alpine"} else "usgs"
    return _box(catalogs[source], place)


# --- 1. b-value drop --------------------------------------------------------

LOG10E = math.log10(math.e)


def b_positive(magnitudes: np.ndarray, delta: float = 0.2) -> float | None:
    """b-positive from the positive magnitude steps between consecutive quakes (van der Elst 2021)."""
    m = np.asarray(magnitudes, dtype=float)
    steps = np.diff(m)
    excess = steps[steps >= delta] - delta
    if excess.size < 10 or float(excess.mean()) <= 0:
        return None
    return LOG10E / float(excess.mean())


def signals_bpositive(ev: dict[str, np.ndarray], n: int, delta: float = 0.2) -> dict[str, np.ndarray]:
    m, t, known = ev["mag"], ev["t"], ev["known"]
    out: dict[str, np.ndarray] = {}
    if len(m) < 60:
        return out
    steps = m[1:] - m[:-1]
    keep = steps >= delta
    excess = steps[keep] - delta
    when = t[1:][keep]
    seen = known[1:][keep]
    if excess.size < 60:
        return out
    cs = np.concatenate([[0.0], np.cumsum(excess)])
    lo = np.searchsorted(when, when - 5 * YEAR, side="left")
    hi = np.arange(1, excess.size + 1)
    count = hi - lo
    ref_mean = np.where(count >= 50, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)
    for size in (50, 100, 200):
        recent = _last_n_mean(excess, size)
        ratio = np.log(np.where(recent > 0, ref_mean / recent, np.nan))
        out[f"b drop, last {size} steps vs 5 years"] = _at_mornings(seen, ratio, n)
    raw = _last_n_mean(excess, 100)
    out["b-positive, last 100 steps"] = _at_mornings(seen, np.where(raw > 0, LOG10E / raw, np.nan), n)
    return out


# --- 2. tides ---------------------------------------------------------------


def tide_potential(unix_days: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Relative tide-raising potential of the Moon and Sun at a place and time (low-precision ephemeris)."""
    rad = np.pi / 180.0
    jd = np.asarray(unix_days, dtype=float) + 2440587.5
    cent = (jd - 2451545.0) / 36525.0
    lp = 218.3164477 + 481267.88123421 * cent
    elong = (297.8501921 + 445267.1114034 * cent) * rad
    sun_anom = (357.5291092 + 35999.0502909 * cent) * rad
    moon_anom = (134.9633964 + 477198.8675055 * cent) * rad
    node = (93.2720950 + 483202.0175233 * cent) * rad
    moon_lon = (
        lp
        + 6.289 * np.sin(moon_anom)
        + 1.274 * np.sin(2 * elong - moon_anom)
        + 0.658 * np.sin(2 * elong)
        + 0.214 * np.sin(2 * moon_anom)
        - 0.186 * np.sin(sun_anom)
        - 0.114 * np.sin(2 * node)
    ) * rad
    moon_lat = (
        5.128 * np.sin(node)
        + 0.2806 * np.sin(moon_anom + node)
        + 0.2777 * np.sin(moon_anom - node)
        + 0.1732 * np.sin(2 * elong - node)
    ) * rad
    moon_km = (
        385000.56
        - 20905.355 * np.cos(moon_anom)
        - 3699.111 * np.cos(2 * elong - moon_anom)
        - 2955.968 * np.cos(2 * elong)
        + 569.925 * np.cos(2 * moon_anom)
    )
    sun_mean = (280.46646 + 36000.76983 * cent) * rad
    sun_lon = sun_mean + (1.914602 * np.sin(sun_anom) + 0.019993 * np.sin(2 * sun_anom)) * rad
    sun_au = 1.00014 - 0.01671 * np.cos(sun_anom) - 0.00014 * np.cos(2 * sun_anom)
    tilt = (23.439291 - 0.0130042 * cent) * rad
    sidereal = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) * rad

    def zenith_cos(ecl_lon: np.ndarray, ecl_lat: np.ndarray) -> np.ndarray:
        ra = np.arctan2(np.sin(ecl_lon) * np.cos(tilt) - np.tan(ecl_lat) * np.sin(tilt), np.cos(ecl_lon))
        dec = np.arcsin(np.sin(ecl_lat) * np.cos(tilt) + np.cos(ecl_lat) * np.sin(tilt) * np.sin(ecl_lon))
        hour = sidereal + np.asarray(lon, dtype=float) * rad - ra
        phi = np.asarray(lat, dtype=float) * rad
        return np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour)

    def legendre2(x: np.ndarray) -> np.ndarray:
        return 0.5 * (3.0 * x * x - 1.0)

    moon = (384400.0 / moon_km) ** 3 * legendre2(zenith_cos(moon_lon, moon_lat))
    sun = 0.4602 * (1.0 / sun_au) ** 3 * legendre2(zenith_cos(sun_lon, np.zeros_like(sun_lon)))
    return moon + sun


def _tide_marks(ev: dict[str, np.ndarray], start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """For each quake: was the tide rising, and how big was the tide that day."""
    unix0 = (start - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)
    when = ev["t"] + unix0
    lat, lon = ev["lat"], ev["lon"]
    step = 0.25 / 24.0
    rising = tide_potential(when + step, lat, lon) > tide_potential(when - step, lat, lon)
    hours = np.arange(-12, 13) / 24.0
    top = np.full(len(when), -np.inf)
    low = np.full(len(when), np.inf)
    for offset in hours:
        value = tide_potential(when + offset, lat, lon)
        top = np.maximum(top, value)
        low = np.minimum(low, value)
    return rising, top - low


def signals_tidal(ev: dict[str, np.ndarray], n: int, start: pd.Timestamp, floor: float, early_end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    m, known, t = ev["mag"], ev["known"], ev["t"]
    if len(m) < 100:
        return out
    rising, amplitude = _tide_marks(ev, start)
    early = t < early_end
    cut = float(np.median(amplitude[early])) if int(early.sum()) >= 50 else float(np.median(amplitude))
    high = amplitude >= cut
    offset = floor - 0.05
    for size in (100, 200):
        n_high = _last_n_mean(high.astype(float), size) * size
        n_low = size - n_high
        sum_high = _last_n_mean(np.where(high, m, 0.0), size) * size
        sum_low = _last_n_mean(np.where(high, 0.0, m), size) * size
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_high = sum_high / n_high - offset
            mean_low = sum_low / n_low - offset
            b_high = np.where((n_high >= 15) & (mean_high > 0), LOG10E / mean_high, np.nan)
            b_low = np.where((n_low >= 15) & (mean_low > 0), LOG10E / mean_low, np.nan)
        out[f"b gap at high tide, last {size} quakes"] = _at_mornings(known, b_low - b_high, n)
    energy = 10.0 ** (1.5 * (m - floor))
    for power, name in ((0.0, "count"), (0.5, "Benioff")):
        weight = energy**power
        load = _daily_sum(known, np.where(rising, weight, 0.0), n)
        unload = _daily_sum(known, np.where(rising, 0.0, weight), n)
        for width in (90, 365):
            ratio = np.log((_trailing(load, width) + 1.0) / (_trailing(unload, width) + 1.0))
            out[f"rising-tide ratio, {name}, {width} days"] = ratio
    return out


# --- 3. natural time --------------------------------------------------------


def kappa1(magnitudes: np.ndarray) -> float:
    """Variance of natural time, weighted by quake energy (Varotsos)."""
    m = np.asarray(magnitudes, dtype=float)
    q = 10.0 ** (1.5 * (m - m.max()))
    p = q / q.sum()
    chi = np.arange(1, len(m) + 1) / len(m)
    return float(np.sum(p * chi**2) - np.sum(p * chi) ** 2)


def signals_natural_time(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view

    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    for size in (40, 100, 200):
        if len(m) < size + 10:
            continue
        values = np.full(len(m), np.nan)
        chi = np.arange(1, size + 1) / size
        windows = sliding_window_view(m, size)
        for lo in range(0, windows.shape[0], 20000):
            block = windows[lo : lo + 20000]
            q = 10.0 ** (1.5 * (block - block.max(axis=1, keepdims=True)))
            p = q / q.sum(axis=1, keepdims=True)
            first = p @ chi
            values[size - 1 + lo : size - 1 + lo + block.shape[0]] = p @ (chi**2) - first**2
        out[f"natural time, last {size} quakes"] = _at_mornings(known, values, n)
        out[f"closeness to 0.070, last {size} quakes"] = _at_mornings(known, -np.abs(values - 0.070), n)
    return out


# --- 4. quieter than the aftershock model ------------------------------------


def _etas_params(root: Path | None):
    from eqpred.models.etas import EtasParams, params_from_dict

    path = model_dir(root) / "etas.json"
    if path.exists():
        return params_from_dict(json.loads(path.read_text(encoding="utf-8"))).clip()
    return EtasParams()


def etas_expected(t: np.ndarray, mag: np.ndarray, floor: float, n: int, params, mu: float, max_age: int = 1825) -> np.ndarray:
    """Expected quakes per day: a steady rate plus Omori decay after each earlier quake."""
    rate = np.full(n, float(mu))
    one = 1.0 - params.p
    productivity = params.k * np.exp(params.alpha * (mag - floor))
    for when, k in zip(t, productivity):
        first = max(int(np.floor(when)) + 1, 0)
        if first >= n:
            continue
        last = min(n, first + max_age)
        day = np.arange(first, last, dtype=float)
        rate[first:last] += k * ((day + 1.0 - when + params.c) ** one - (day - when + params.c) ** one) / one
    return rate


def signals_etas_quiet(ev: dict[str, np.ndarray], n: int, floor: float, early_end: int, params) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t, m, known = ev["t"], ev["mag"], ev["known"]
    if len(t) < 50:
        return out
    early = (t >= 0) & (t < early_end)
    mu = 0.5 * float(early.sum()) / max(early_end, 1)
    if mu <= 0:
        mu = 0.5 * len(t) / max(n, 1)
    expected = etas_expected(t, m, floor, n, params, mu)
    observed = _daily_sum(known, None, n)
    # A quake can take two days to be reported, so the expected count lags by two days.
    long_ratio = np.log((_trailing(observed, 1826) + 1.0) / (_shift(_trailing(expected, 1826), 2) + 1.0))
    for width in (30, 90, 365):
        short_ratio = np.log((_trailing(observed, width) + 1.0) / (_shift(_trailing(expected, width), 2) + 1.0))
        out[f"observed over expected, {width} days vs 5 years"] = short_ratio - long_ratio
    return out


# --- 5. aftershocks that died out too fast ---------------------------------


def signals_quiet_aftershocks(ev: dict[str, np.ndarray], n: int, floor: float, target: float, early_end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t, m, lat, lon, known = ev["t"], ev["mag"], ev["lat"], ev["lon"], ev["known"]
    main_floor = max(floor + 1.0, target - 1.0)
    mains = np.flatnonzero(m >= main_floor)
    if mains.size < 12:
        return out
    for span in (10, 30):
        counts = np.zeros(mains.size)
        for pos, j in enumerate(mains):
            hi = np.searchsorted(t, t[j] + span, side="right")
            if hi <= j + 1:
                continue
            cand = slice(j + 1, hi)
            big = m[cand] >= max(floor, m[j] - 2.0)
            if not big.any():
                continue
            dist = haversine_km(lon[j], lat[j], lon[cand][big], lat[cand][big])
            counts[pos] = float((dist <= max(float(uhrhammer_km(m[j])), 20.0)).sum())
        size = np.log10(counts + 1.0)
        early = t[mains] + span < early_end
        if int(early.sum()) < 8:
            continue
        slope, intercept = np.polyfit(m[mains][early], size[early], 1)
        residual = size - (intercept + slope * m[mains])
        ready = known[mains] + span + 2
        order = np.argsort(ready, kind="mergesort")
        ready, residual, main_t = ready[order], residual[order], t[mains][order]
        cs = np.concatenate([[0.0], np.cumsum(residual)])
        mornings = np.arange(n)
        hi = np.searchsorted(ready, mornings, side="right")
        for years in (2, 6):
            signal = np.full(n, np.nan)
            for i in range(n):
                upto = hi[i]
                if upto <= 0:
                    continue
                lo = np.searchsorted(main_t[:upto], i - years * YEAR, side="left")
                count = upto - lo
                if count >= 3:
                    signal[i] = (cs[upto] - cs[lo]) / count
            out[f"aftershock surplus, {span} days, last {years} years"] = signal
    return out


# --- 6. localization --------------------------------------------------------


def _weekly_expand(marks: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    daily = np.full(n, np.nan)
    daily[marks] = values
    return pd.Series(daily).ffill(limit=int(marks[1] - marks[0] - 1) if len(marks) > 1 else 0).to_numpy()


def _cell_ids_size(lat: np.ndarray, lon: np.ndarray, place: str, size: float) -> tuple[np.ndarray, int]:
    region = REGIONS[place]
    if region.get("kind") != "bbox":
        lat0, lon0, n_lat, n_lon = -90.0, -180.0, int(180 / size), int(360 / size)
        x = lon
    else:
        lat0 = float(region["min_lat"])
        lon0 = float(region["min_lon"])
        span_lat = float(region["max_lat"]) - lat0
        width = (float(region["max_lon"]) - lon0) % 360.0 or 360.0
        n_lat = int(np.ceil(span_lat / size))
        n_lon = int(np.ceil(width / size))
        x = lon0 + (np.asarray(lon, dtype=float) - lon0) % 360.0
    row = np.clip(((np.asarray(lat, dtype=float) - lat0) // size).astype(int), 0, n_lat - 1)
    col = np.clip(((x - lon0) // size).astype(int), 0, n_lon - 1)
    return row * n_lon + col, n_lat * n_lon


def _cell_ids(lat: np.ndarray, lon: np.ndarray, place: str) -> tuple[np.ndarray, int]:
    return _cell_ids_size(lat, lon, place, CELL_DEG[place])


def _spread(counts: np.ndarray) -> tuple[float, float]:
    total = counts.sum()
    if total <= 0 or counts.size == 0:
        return float("nan"), float("nan")
    ranked = np.sort(counts)[::-1]
    share = np.cumsum(ranked) / total
    area = (int(np.searchsorted(share, 0.9)) + 1) / counts.size
    ordered = np.sort(counts)
    k = np.arange(1, counts.size + 1)
    gini = float((2.0 * np.sum(k * ordered)) / (counts.size * total) - (counts.size + 1.0) / counts.size)
    return area, gini


def signals_localization(frame: pd.DataFrame, place: str, start: pd.Timestamp, n: int, floor: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    work = frame.loc[pd.to_numeric(frame["magnitude"], errors="coerce") >= floor]
    if len(work) < 200:
        return out
    background = decluster_causal(work, space_km=200.0, time_days=10.0)
    ev = _events(background, start, n, floor)
    cells, n_cells = _cell_ids(ev["lat"], ev["lon"], place)
    known = ev["known"]
    marks = np.arange(0, n, 7)
    results = {width: np.full((marks.size, 2), np.nan) for width in (365, 1095)}
    for row, morning in enumerate(marks):
        ref_lo = np.searchsorted(known, morning - 3652, side="right")
        hi = np.searchsorted(known, morning, side="right")
        if hi - ref_lo < 50:
            continue
        active = np.bincount(cells[ref_lo:hi], minlength=n_cells) > 0
        for width in (365, 1095):
            lo = np.searchsorted(known, morning - width, side="right")
            if hi - lo < 20:
                continue
            counts = np.bincount(cells[lo:hi], minlength=n_cells)[active]
            results[width][row] = _spread(counts.astype(float))
    for width, table in results.items():
        for col, name in ((0, "area holding 90%"), (1, "Gini")):
            weekly = table[:, col]
            daily = np.full(n, np.nan)
            daily[marks] = weekly
            daily = pd.Series(daily).ffill(limit=6).to_numpy()
            out[f"{name}, {width} days"] = daily
            trail = pd.Series(weekly).rolling(522, min_periods=100).mean().to_numpy()
            rel = np.full(n, np.nan)
            rel[marks] = weekly - trail
            out[f"{name}, {width} days vs 10 years"] = pd.Series(rel).ffill(limit=6).to_numpy()
    return out


# --- 7. coalescence ---------------------------------------------------------


def nearest_eta(t: np.ndarray, m: np.ndarray, lat: np.ndarray, lon: np.ndarray, floor: float, span: float = 1095.0) -> np.ndarray:
    """Nearest earlier neighbor in time, distance and size (Zaliapin and Ben-Zion), as a log10."""
    eta = np.full(len(t), np.nan)
    for j in range(len(t)):
        lo = np.searchsorted(t, t[j] - span, side="left")
        if lo >= j:
            continue
        dt = (t[j] - t[lo:j]) / YEAR
        ok = dt > 0
        if not ok.any():
            continue
        dist = np.maximum(haversine_km(lon[j], lat[j], lon[lo:j][ok], lat[lo:j][ok]), 1.0)
        eta[j] = float(np.min(np.log10(dt[ok]) + 1.6 * np.log10(dist) - (m[lo:j][ok] - floor)))
    return eta


def signals_coalescence(ev: dict[str, np.ndarray], n: int, floor: float, early_end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t, m, lat, lon, known = ev["t"], ev["mag"], ev["lat"], ev["lon"], ev["known"]
    if len(t) < 200:
        return out
    eta = nearest_eta(t, m, lat, lon, floor)
    early = (t < early_end) & np.isfinite(eta)
    if int(early.sum()) < 50:
        return out
    every = _daily_sum(known, np.isfinite(eta).astype(float), n)
    long_all = _trailing(every, 3652)
    for q in (0.3, 0.5):
        cut = float(np.quantile(eta[early], q))
        clustered = _daily_sum(known, (np.isfinite(eta) & (eta < cut)).astype(float), n)
        long_share = _trailing(clustered, 3652) / np.where(long_all > 0, long_all, np.nan)
        for width in (365, 730):
            total = _trailing(every, width)
            share = _trailing(clustered, width) / np.where(total >= 10, total, np.nan)
            out[f"clustered share, {width} days, cut {int(q * 100)}%"] = share - long_share
    return out


# --- 8. Earth's spin --------------------------------------------------------


def signals_earth_rotation(root: Path | None, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    from eqpred.ingest.run import latest_raw

    out: dict[str, np.ndarray] = {}
    try:
        eop = latest_raw("iers.eop", root)
    except FileNotFoundError:
        return out
    if eop is None or eop.empty or "eop_lod_s" not in eop.columns:
        return out
    work = eop.copy()
    work["observed_at"] = parse_utc(work["observed_at"]).dt.normalize()
    lod = (
        work.dropna(subset=["eop_lod_s"])
        .drop_duplicates("observed_at", keep="last")
        .set_index("observed_at")["eop_lod_s"]
        .reindex(days)
        .interpolate(limit=30, limit_area="inside")
    )
    known = _shift(lod.to_numpy(dtype=float) * 1000.0, 6)
    for smooth in (365, 1095):
        level = pd.Series(known).rolling(smooth, min_periods=int(smooth * 0.9)).mean().to_numpy()
        slope = level - _shift(level, 365)
        for lag_years in (4, 5, 6):
            out[f"spin slowing, {smooth}-day mean, {lag_years} years earlier"] = _shift(slope, int(round(lag_years * YEAR)))
        if smooth == 365:
            out["day length, 365-day mean, 5 years earlier"] = _shift(level, int(round(5 * YEAR)))
    return out


# --- 9. hidden active state ---------------------------------------------------


def _poisson_log(counts: np.ndarray, rate: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    lam = np.maximum(rate, 1e-9)
    return counts[:, None] * np.log(lam) - lam - gammaln(counts + 1.0)[:, None]


def _forward(log_emit: np.ndarray, trans: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps, k = log_emit.shape
    alpha = np.zeros((steps, k))
    scale = np.zeros(steps)
    prior = start
    for i in range(steps):
        row = log_emit[i]
        top = row.max()
        a = prior * np.exp(row - top)
        total = a.sum()
        if total <= 0 or not np.isfinite(total):
            a = np.full(k, 1.0 / k)
            total = 1.0
        alpha[i] = a / total
        scale[i] = math.log(total) + top
        prior = alpha[i] @ trans
    return alpha, scale


def fit_hidden_state(counts: np.ndarray, base: np.ndarray, states: int = 2, rounds: int = 25) -> dict[str, np.ndarray]:
    """Poisson hidden Markov model whose rates are multiples of the recent usual rate."""
    factors = np.array([0.5, 3.0]) if states == 2 else np.array([0.3, 1.0, 4.0])
    trans = np.full((states, states), 0.05 / (states - 1))
    np.fill_diagonal(trans, 0.95)
    start = np.full(states, 1.0 / states)
    for _ in range(rounds):
        log_emit = _poisson_log(counts, base[:, None] * factors[None, :])
        alpha, _scale = _forward(log_emit, trans, start)
        steps = len(counts)
        beta = np.ones((steps, states))
        for i in range(steps - 2, -1, -1):
            row = log_emit[i + 1]
            b = trans @ (np.exp(row - row.max()) * beta[i + 1])
            beta[i] = b / max(b.sum(), 1e-300)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi = np.zeros((states, states))
        for i in range(steps - 1):
            row = log_emit[i + 1]
            e = np.exp(row - row.max()) * beta[i + 1]
            block = alpha[i][:, None] * trans * e[None, :]
            xi += block / max(block.sum(), 1e-300)
        trans = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        factors = np.maximum((gamma * counts[:, None]).sum(axis=0) / np.maximum((gamma * base[:, None]).sum(axis=0), 1e-9), 1e-3)
        start = gamma[0]
    order = np.argsort(factors)
    return {"factors": factors[order], "trans": trans[np.ix_(order, order)], "start": start[order]}


def signals_hidden_state(frame: pd.DataFrame, place: str, start: pd.Timestamp, n: int, early_end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    floors = (FLOOR[place], FLOOR[place] + 0.5)
    for floor in floors:
        ev = _events(frame, start, n, floor)
        if len(ev["t"]) < 200:
            continue
        counts = _daily_sum(ev["known"], None, n)
        trail = _shift(_trailing(counts, 365) / 365.0, 1)
        floor_rate = max(float(np.nanmean(counts[:early_end])) / 10.0, 1e-3)
        base = np.where(np.isfinite(trail), np.maximum(trail, floor_rate), np.nan)
        fit_days = np.arange(366, early_end)
        fit_days = fit_days[np.isfinite(base[fit_days])]
        if fit_days.size < 1000:
            continue
        for states in (2, 3):
            model = fit_hidden_state(counts[fit_days], base[fit_days], states=states)
            ready = np.isfinite(base)
            log_emit = _poisson_log(counts[ready], base[ready][:, None] * model["factors"][None, :])
            alpha, _scale = _forward(log_emit, model["trans"], model["start"])
            busy = np.full(n, np.nan)
            busy[ready] = alpha[:, -1]
            out[f"active-state chance, {states} states, M{floor:g}+"] = busy
    return out


# --- 10. the crowd ------------------------------------------------------------


def _ridge_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray, penalty: float) -> np.ndarray:
    from scipy.optimize import minimize

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = offset + x @ w
        p = _expit(z)
        nll = float(np.sum(np.logaddexp(0.0, z) - y * z)) + penalty * float(w @ w)
        grad = x.T @ (p - y) + 2.0 * penalty * w
        return nll, grad

    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return result.x


def _crowd_frames(root: Path | None, tape_name: str, horizon: int) -> pd.DataFrame | None:
    index = attempts_dir(root) / "index.jsonl"
    if not index.exists():
        return None
    names = {name for name, _label in STRATEGIES}
    latest: dict[str, dict[str, Any]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("source") != "strategy" or row.get("tape") != tape_name:
            continue
        if row.get("name") not in names or horizon not in (row.get("horizons") or []):
            continue
        latest[str(row["name"])] = row
    parts = []
    for name, row in latest.items():
        path = attempts_dir(root) / str(row["file"])
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["issue_day", "region", "horizon_days", "p", "climatology_p", "happened", "closed"])
        frame = frame[frame["horizon_days"] == horizon]
        frame = frame.assign(rule=name)
        parts.append(frame)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


# --- 11. nowcasting interevent count -----------------------------------------


def signals_nowcast_eps(ev: dict[str, np.ndarray], n: int, floor: float, target: float, early_end: int) -> dict[str, np.ndarray]:
    """Count of small quakes and Benioff strain since the last big shock (Rundle and Turcotte 2016)."""
    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    if len(m) < 80:
        return out
    small_daily = _daily_sum(known, None, n)
    benioff_daily = _daily_sum(known, 10.0 ** (0.75 * (m - floor)), n)
    cs_count = np.concatenate([[0.0], np.cumsum(small_daily)])
    cs_benioff = np.concatenate([[0.0], np.cumsum(benioff_daily)])
    for gap, label in ((1.0, "M-1"), (0.5, "M-0.5")):
        anchor_mag = max(floor + 0.5, target - gap)
        anchors = np.flatnonzero(m >= anchor_mag)
        if anchors.size < 8:
            continue
        anchor_known = np.unique(np.clip(known[anchors], 0, n - 1))
        if anchor_known.size < 6:
            continue
        last_idx = np.searchsorted(anchor_known, np.arange(n), side="right") - 1
        since_day = np.where(last_idx >= 0, anchor_known[np.maximum(last_idx, 0)], 0)
        count_since = cs_count[np.arange(1, n + 1)] - cs_count[since_day]
        strain_since = cs_benioff[np.arange(1, n + 1)] - cs_benioff[since_day]
        out[f"small-quake count since last {label} shock"] = count_since
        out[f"Benioff strain since last {label} shock"] = np.log1p(np.maximum(strain_since, 0.0))
        early_anchors = anchor_known[anchor_known < early_end]
        if early_anchors.size >= 6:
            cycles = cs_count[early_anchors[1:]] - cs_count[early_anchors[:-1]]
            cycles = np.sort(cycles[cycles > 0])
            if cycles.size >= 5:
                eps = np.searchsorted(cycles, count_since, side="right") / float(cycles.size)
                out[f"nowcast potential score, {label} cycle"] = eps
    return out


# --- 12. interevent burstiness and memory ------------------------------------


def signals_burstiness(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Burstiness, memory, and Molchan background fraction of waiting times."""
    from numpy.lib.stride_tricks import sliding_window_view

    out: dict[str, np.ndarray] = {}
    t, known = ev["t"], ev["known"]
    if len(t) < 220:
        return out
    dt = np.maximum(np.diff(t), 1e-5)
    seen = known[1:]
    for size in (50, 100, 200):
        if dt.size < size + 5:
            continue
        win = sliding_window_view(dt, size)
        mu = win.mean(axis=1)
        sd = win.std(axis=1)
        burst = np.full(dt.size, np.nan)
        burst[size - 1 :] = (sd - mu) / np.maximum(sd + mu, 1e-9)
        out[f"burstiness, last {size} intervals"] = _at_mornings(seen, burst, n)
        bg = np.full(dt.size, np.nan)
        bg[size - 1 :] = mu / np.maximum(sd * sd, 1e-6)
        out[f"background fraction, last {size} intervals"] = _at_mornings(seen, np.log1p(np.maximum(bg, 0.0)), n)
        if size in (50, 100):
            a = win[:, :-1]
            b = win[:, 1:]
            da = a - a.mean(axis=1, keepdims=True)
            db = b - b.mean(axis=1, keepdims=True)
            denom = np.sqrt(np.maximum((da * da).sum(axis=1) * (db * db).sum(axis=1), 1e-12))
            mem = np.full(dt.size, np.nan)
            mem[size - 1 :] = (da * db).sum(axis=1) / denom
            out[f"interval memory, last {size} intervals"] = _at_mornings(seen, mem, n)
    return out


# --- 13. Region-Time-Length (RTL) --------------------------------------------


def signals_rtl(ev: dict[str, np.ndarray], n: int, floor: float) -> dict[str, np.ndarray]:
    """Sobolev and Tyupkin RTL quiescence and activation score."""
    out: dict[str, np.ndarray] = {}
    m, lat, lon, known = ev["mag"], ev["lat"], ev["lon"], ev["known"]
    if len(m) < 120:
        return out
    med_lat = float(np.median(lat[: max(50, len(lat) // 3)]))
    med_lon = float(np.median(lon[: max(50, len(lon) // 3)]))
    dist = haversine_km(med_lon, med_lat, lon, lat)
    r0 = max(float(np.median(dist)) if dist.size else 100.0, 25.0)
    r_weight = np.exp(-dist / r0)
    l_weight = 10.0 ** (0.5 * np.clip(m - floor, 0.0, 4.0))
    daily_r = _daily_sum(known, r_weight, n)
    daily_t = _daily_sum(known, None, n)
    daily_l = _daily_sum(known, l_weight, n)
    for span in (60, 180, 365):
        r_s = _trailing(daily_r, span)
        t_s = _trailing(daily_t, span)
        l_s = _trailing(daily_l, span)
        parts = []
        for series in (r_s, t_s, l_s):
            s = pd.Series(series)
            mu = s.rolling(1826, min_periods=365).mean().to_numpy()
            sd = s.rolling(1826, min_periods=365).std().to_numpy()
            parts.append((series - mu) / np.where(sd > 1e-6, sd, np.nan))
        rtl_sum = parts[0] + parts[1] + parts[2]
        out[f"RTL activation score, {span} days"] = rtl_sum
        out[f"RTL rupture-length deviation, {span} days"] = parts[2]
    return out


# --- 14. Keilis-Borok M8 seismic flow ----------------------------------------


def signals_m8_tip(ev: dict[str, np.ndarray], n: int, floor: float) -> dict[str, np.ndarray]:
    """M8 seismic flow concentration and Benioff strain acceleration."""
    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    if len(m) < 120:
        return out
    counts = _daily_sum(known, None, n)
    rupture = _daily_sum(known, 10.0 ** (0.5 * np.clip(m - floor, 0.0, 4.5)), n)
    benioff = _daily_sum(known, 10.0 ** (0.75 * np.clip(m - floor, 0.0, 4.5)), n)
    base_n = _trailing(counts, 1826) / 1826.0
    base_b = _trailing(benioff, 1826) / 1826.0
    for span in (180, 365, 1095):
        n_s = _trailing(counts, span)
        l_s = _trailing(rupture, span)
        b_s = _trailing(benioff, span)
        mean_len = l_s / np.maximum(n_s, 1.0)
        spacing = (np.maximum(n_s, 1.0)) ** (-1.0 / 3.0)
        conc = np.where(n_s >= 5, mean_len / spacing, np.nan)
        out[f"M8 source concentration, {span} days"] = conc
        out[f"M8 Benioff acceleration, {span} days vs 5 years"] = np.log(
            ((b_s / span) + 1e-4) / (np.maximum(base_b, 0.0) + 1e-4)
        )
        flow = np.log(((n_s / span) + 1e-4) / (np.maximum(base_n, 0.0) + 1e-4))
        out[f"M8 composite flow, {span} days"] = flow + np.log1p(np.maximum(conc, 0.0))
    return out


# --- 15. hypocentral depth shift ---------------------------------------------


def signals_depth_shift(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Shallowing of focal depths into the upper locked crust."""
    out: dict[str, np.ndarray] = {}
    depth, known = ev["depth"], ev["known"]
    if len(depth) < 120:
        return out
    for size in (50, 150):
        recent = _last_n_mean(depth, size)
        long_d = _last_n_mean(depth, min(max(size * 6, 300), max(len(depth) // 2, size + 1)))
        out[f"depth shallowing, last {size} quakes"] = _at_mornings(known, long_d - recent, n)
    shallow = _daily_sum(known, (depth <= 25.0).astype(float), n)
    deep = _daily_sum(known, (depth > 40.0).astype(float), n)
    total = _daily_sum(known, None, n)
    long_sh = _trailing(shallow, 1826) / np.maximum(_trailing(total, 1826), 1.0)
    for span in (90, 365):
        tot = _trailing(total, span)
        sh = np.where(tot >= 10, _trailing(shallow, span) / np.maximum(tot, 1.0), np.nan)
        out[f"shallow-crust share, {span} days vs 5 years"] = sh - long_sh
        dp = _trailing(deep, span)
        out[f"shallow-to-deep ratio, {span} days"] = np.log((_trailing(shallow, span) + 1.0) / (dp + 1.0))
    return out


# --- 16. moment-tensor rake organization -------------------------------------


def _cmt_for_place(cmt: pd.DataFrame, place: str) -> pd.DataFrame:
    if cmt is None or cmt.empty:
        return pd.DataFrame()
    if place == "global":
        return cmt
    if place == "kermadec":
        return _box(cmt, "kermadec")
    lat = cmt["lat"]
    lon = cmt["lon"]
    nz_arc = (lat >= -48.0) & (lat <= -25.0) & ((lon >= 165.0) | (lon <= -174.0))
    return cmt.loc[nz_arc]


def signals_rake_regime(cmt: pd.DataFrame, place: str, start: pd.Timestamp, n: int) -> dict[str, np.ndarray]:
    """Slip-angle alignment and thrust dominance from Global CMT focal mechanisms."""
    out: dict[str, np.ndarray] = {}
    work = _cmt_for_place(cmt, place)
    if work.empty or "rake" not in work.columns:
        return out
    work = work.copy()
    work["observed_at"] = parse_utc(work["observed_at"])
    avail = parse_utc(work["available_at"]) if "available_at" in work.columns else work["observed_at"]
    work = work.assign(_avail=avail).dropna(subset=["observed_at", "_avail", "rake"]).sort_values("observed_at")
    if len(work) < 60:
        return out
    seen = ((work["_avail"] - start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    known = np.clip(np.maximum.accumulate(np.floor(seen).astype(np.int64) + 1), 0, n)
    rad = np.deg2rad(pd.to_numeric(work["rake"], errors="coerce").to_numpy(dtype=float))
    c_rake = np.cos(rad)
    s_rake = np.sin(rad)
    for size in (25, 60):
        if len(rad) < size + 5:
            continue
        mc = _last_n_mean(c_rake, size)
        ms = _last_n_mean(s_rake, size)
        align = np.sqrt(mc * mc + ms * ms)
        out[f"mechanism alignment, last {size} CMTs"] = _at_mornings(known, align, n)
        out[f"thrust balance, last {size} CMTs"] = _at_mornings(known, ms, n)
    return out


# --- 17. Chandler polar wobble -----------------------------------------------


def signals_polar_wobble(root: Path | None, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Polar motion radius, drift velocity, and wobble excitation from IERS EOP."""
    from eqpred.ingest.run import latest_raw

    out: dict[str, np.ndarray] = {}
    try:
        eop = latest_raw("iers.eop", root)
    except FileNotFoundError:
        return out
    if eop is None or eop.empty or "eop_pmx_arcsec" not in eop.columns:
        return out
    work = eop.copy()
    work["observed_at"] = parse_utc(work["observed_at"]).dt.normalize()
    work = work.dropna(subset=["eop_pmx_arcsec", "eop_pmy_arcsec"]).drop_duplicates("observed_at", keep="last").set_index("observed_at")
    px = _shift(work["eop_pmx_arcsec"].reindex(days).interpolate(limit=30, limit_area="inside").to_numpy(dtype=float), 6)
    py = _shift(work["eop_pmy_arcsec"].reindex(days).interpolate(limit=30, limit_area="inside").to_numpy(dtype=float), 6)
    cx = pd.Series(px).rolling(2337, min_periods=730).mean().to_numpy()
    cy = pd.Series(py).rolling(2337, min_periods=730).mean().to_numpy()
    radius = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    out["Chandler wobble radius"] = radius
    for step in (30, 90, 180):
        vx = px - _shift(px, step)
        vy = py - _shift(py, step)
        speed = np.sqrt(vx * vx + vy * vy)
        out[f"polar drift speed, {step} days"] = speed
        ax = vx - _shift(vx, step)
        ay = vy - _shift(vy, step)
        out[f"polar wobble excitation, {step} days"] = np.sqrt(ax * ax + ay * ay)
    return out


# --- 18. spatial centroid drift and diffusivity ------------------------------


def signals_spatial_migration(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Drift of the seismicity center of mass and consecutive step diffusivity."""
    out: dict[str, np.ndarray] = {}
    t, lat, lon, known = ev["t"], ev["lat"], ev["lon"], ev["known"]
    if len(t) < 220:
        return out
    rad = np.pi / 180.0
    clat = np.cos(lat * rad)
    ux = clat * np.cos(lon * rad)
    uy = clat * np.sin(lon * rad)
    uz = np.sin(lat * rad)
    for short, long_n in ((30, 150), (60, 300)):
        if len(t) < long_n + 5:
            continue
        sx, sy, sz = _last_n_mean(ux, short), _last_n_mean(uy, short), _last_n_mean(uz, short)
        lx, ly, lz = _last_n_mean(ux, long_n), _last_n_mean(uy, long_n), _last_n_mean(uz, long_n)
        chord = np.sqrt((sx - lx) ** 2 + (sy - ly) ** 2 + (sz - lz) ** 2)
        out[f"centroid shift, last {short} vs {long_n} quakes"] = _at_mornings(known, 6371.0 * chord, n)
    step_km = np.minimum(haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]), 500.0)
    dt = np.maximum(np.diff(t), 0.05)
    diffusivity = np.log1p((step_km * step_km) / dt)
    seen = known[1:]
    for size in (40, 100):
        if diffusivity.size >= size + 5:
            out[f"spatial diffusivity, last {size} steps"] = _at_mornings(seen, _last_n_mean(diffusivity, size), n)
            out[f"step distance, last {size} steps"] = _at_mornings(seen, _last_n_mean(step_km, size), n)
    return out


# --- 19. volcanic eruption pulse ---------------------------------------------


def signals_volcano_magmatic(root: Path | None, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Global and New Zealand volcanic eruption pulse from Smithsonian GVP."""
    from eqpred.ingest.run import latest_raw

    out: dict[str, np.ndarray] = {}
    try:
        gvp = latest_raw("si.gvp", root)
    except FileNotFoundError:
        return out
    if gvp is None or gvp.empty:
        return out
    work = gvp.copy()
    work["observed_at"] = parse_utc(work["observed_at"]).dt.normalize()
    work = work.drop_duplicates("observed_at", keep="last").set_index("observed_at").reindex(days)
    erupt = _shift(pd.to_numeric(work.get("volcano_eruption"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 3)
    vei = _shift(pd.to_numeric(work.get("volcano_vei_max"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 3)
    nz = _shift(pd.to_numeric(work.get("volcano_nz"), errors="coerce").fillna(0.0).to_numpy(dtype=float), 3)
    energy = np.where(erupt > 0, 10.0 ** (0.5 * np.clip(vei, 0.0, 6.0)), 0.0)
    base_e = _trailing(energy, 1826) / 1826.0
    base_nz = _trailing(nz, 1826) / 1826.0
    for span in (30, 90, 365):
        out[f"volcanic energy pulse, {span} days vs 5 years"] = (_trailing(energy, span) / span) - base_e
        out[f"NZ volcanic activity, {span} days vs 5 years"] = (_trailing(nz, span) / span) - base_nz
    return out


# --- 20. causal online expert mixture ----------------------------------------


def _hedge_series(x: np.ndarray, y: np.ndarray, usual: np.ndarray, closed: np.ndarray, eta: float, decay: float) -> np.ndarray:
    """Causal exponential-weights forecaster: updates only on windows that have already closed."""
    m, k = x.shape
    scores = np.zeros(k)
    out = np.zeros(m)
    base_ll = _logit(usual)
    p_mat = _expit(base_ll[:, None] + x)
    eps = 1e-6
    p_mat = np.clip(p_mat, eps, 1.0 - eps)
    b_vec = np.clip(usual, eps, 1.0 - eps)
    gain_mat = (y[:, None] * np.log(p_mat) + (1.0 - y[:, None]) * np.log(1.0 - p_mat)) - (
        y * np.log(b_vec) + (1.0 - y) * np.log(1.0 - b_vec)
    )[:, None]
    for i in range(m):
        top = scores.max()
        w = np.exp(np.clip(eta * (scores - top), -40.0, 0.0))
        w /= max(w.sum(), 1e-12)
        out[i] = float(x[i] @ w)
        if closed[i] and np.isfinite(y[i]):
            scores = decay * scores + np.where(np.isfinite(gain_mat[i]), gain_mat[i], 0.0)
    return out


def _hedge_cards(root: Path | None, tape_name: str, version: int, mags: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    cards: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for horizon in (3, 7, 30):
        table = _crowd_frames(root, tape_name, horizon)
        if table is None or table.empty:
            continue
        for place in PLACES:
            part = table[table["region"] == place]
            if part.empty:
                continue
            z = _logit(part["p"].to_numpy()) - _logit(part["climatology_p"].to_numpy())
            part = part.assign(z=z)
            wide = part.pivot_table(index="issue_day", columns="rule", values="z", aggfunc="last").fillna(0.0)
            base = part.drop_duplicates("issue_day").set_index("issue_day").loc[wide.index]
            days = pd.DatetimeIndex(pd.to_datetime(wide.index, utc=True))
            ends = days + pd.Timedelta(days=horizon)
            closed = base["closed"].to_numpy(dtype=bool)
            hits = base["happened"].to_numpy(dtype=float)
            usual = base["climatology_p"].to_numpy(dtype=float)
            early = closed & np.asarray(ends <= EARLY_END)
            develop = closed & np.asarray(days >= EARLY_END) & np.asarray(ends <= LATER_START)
            later = closed & np.asarray(days >= LATER_START)
            x = wide.to_numpy(dtype=float)
            if int(early.sum()) < 40 or x.shape[1] == 0:
                continue
            signals: dict[str, np.ndarray] = {}
            for eta in (0.5, 2.0, 8.0):
                for decay, tag in ((1.0, "full history"), (0.98, "recent memory")):
                    signals[f"online experts, eta {eta:g}, {tag}"] = _hedge_series(x, hits, usual, closed, eta, decay)
            best, tried = pick_variation(signals, hits, usual, early, develop)
            adopted = bool(best is not None and _choice_key(best["develop"])[0])
            row: dict[str, Any] = {
                "idea": "hedge_expert",
                "variations_tried": tried,
                "adopted": adopted,
                "variation": None if best is None else best["variation"],
                "mapping": None if best is None else best["mapping"],
                "develop": None if best is None else best["develop"],
                "later": None,
                "pointed_right": False,
                "passed": False,
                "place": place,
                "place_label": LABELS.get(place, place),
                "horizon_days": horizon,
                "magnitude_min": mags.get(str(horizon), {}).get(place),
                "answer_key": f"fresh.v{version}",
                "tape": tape_name,
            }
            if adopted:
                chance = best["chance"]
                scored = split_windows(chance[later], hits[later], usual[later])
                gain = scored.get("information_gain") or 0.0
                row["later"] = scored
                row["pointed_right"] = bool(scored.get("verdict") == "up" and gain > 0)
                row["passed"] = bool(scored.get("verdict") == "up" and gain >= USEFUL_GAIN)
                frames.append(
                    forecast_frame(
                        days=days,
                        region=place,
                        label=LABELS.get(place, place),
                        horizon=horizon,
                        magnitude=float(mags.get(str(horizon), {}).get(place) or 0.0),
                        p=chance,
                        climatology=usual,
                        happened=hits,
                        mask=closed | (np.arange(len(days)) == len(days) - 1),
                        catalog_end=days[-1],
                    )
                )
            cards.append(row)
    return cards, frames


# --- 21. Pattern Informatics ---------------------------------------------------


def signals_pattern_informatics(frame: pd.DataFrame, place: str, start: pd.Timestamp, n: int, floor: float) -> dict[str, np.ndarray]:
    """Pattern Informatics: standardized rate changes summed over spatial cells."""
    out: dict[str, np.ndarray] = {}
    ev = _events(frame, start, n, floor)
    if len(ev["t"]) < 200:
        return out
    cells, n_cells = _cell_ids(ev["lat"], ev["lon"], place)
    known = ev["known"]
    marks = np.arange(0, n, 7)
    tables = {span: np.full((marks.size, 3), np.nan) for span in (730, 1825)}
    for row, morning in enumerate(marks):
        hi = np.searchsorted(known, morning, side="right")
        for span, table in tables.items():
            recent_lo = np.searchsorted(known, morning - span, side="right")
            base_lo = np.searchsorted(known, morning - 2 * span, side="right")
            if hi - base_lo < 40:
                continue
            recent = np.bincount(cells[recent_lo:hi], minlength=n_cells).astype(float)
            base = np.bincount(cells[base_lo:recent_lo], minlength=n_cells).astype(float)
            active = (recent + base) > 0
            if int(active.sum()) < 4:
                continue
            z = (recent[active] - base[active]) / np.sqrt(np.maximum(recent[active] + base[active], 1.0))
            table[row] = [float(np.mean(z * z)), float(np.mean(np.clip(z, 0.0, None))), float(np.mean(z > 1.0))]
    for span, table in tables.items():
        for col, name in ((0, "mean squared change"), (1, "activation"), (2, "strong-cell share")):
            out[f"PI {name}, {span} days"] = _weekly_expand(marks, table[:, col], n)
    return out


# --- 22. accelerating moment release -------------------------------------------


def _segment_increments(prefix: np.ndarray, width: int) -> np.ndarray:
    n = len(prefix) - 1
    out = np.full((n, 4), np.nan)
    quarter = width // 4
    idx = np.arange(n)
    start = idx + 1 - width
    ok = start >= 0
    if not ok.any():
        return out
    bounds = start[ok, None] + quarter * np.arange(5)[None, :]
    out[ok] = np.diff(prefix[bounds], axis=1)
    return out


def signals_amr_curvature(ev: dict[str, np.ndarray], n: int, floor: float) -> dict[str, np.ndarray]:
    """Curvature of cumulative Benioff strain over fixed lookback windows."""
    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    if len(m) < 120:
        return out
    benioff = _daily_sum(known, 10.0 ** (0.75 * np.clip(m - floor, 0.0, 4.5)), n)
    prefix = np.concatenate([[0.0], np.cumsum(benioff)])
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    for width in (365, 1095, 1825):
        inc = _segment_increments(prefix, width)
        total = np.nansum(inc, axis=1)
        logged = np.log(inc + 1e-9)
        slope = (logged @ x) / float(x @ x)
        accel = np.log((inc[:, 3] + 1e-9) / (np.median(inc[:, :3], axis=1) + 1e-9))
        convexity = (inc[:, 3] - inc[:, 0]) / np.maximum(total, 1e-9)
        out[f"AMR log-increment slope, {width} days"] = slope
        out[f"AMR last-quarter acceleration, {width} days"] = accel
        out[f"AMR convexity, {width} days"] = convexity
    return out


# --- 23. fractal dimension and spatial entropy ---------------------------------


def signals_fractal_entropy(frame: pd.DataFrame, place: str, start: pd.Timestamp, n: int, floor: float) -> dict[str, np.ndarray]:
    """Correlation-dimension slope and Shannon entropy across spatial scales."""
    out: dict[str, np.ndarray] = {}
    ev = _events(frame, start, n, floor)
    if len(ev["t"]) < 200:
        return out
    factors = np.array([0.5, 1.0, 2.0])
    cell_sets = [_cell_ids_size(ev["lat"], ev["lon"], place, CELL_DEG[place] * float(f)) for f in factors]
    known = ev["known"]
    marks = np.arange(0, n, 7)
    tables = {width: np.full((marks.size, 3), np.nan) for width in (365, 1095)}
    for row, morning in enumerate(marks):
        hi = np.searchsorted(known, morning, side="right")
        for width, table in tables.items():
            lo = np.searchsorted(known, morning - width, side="right")
            if hi - lo < 30:
                continue
            entropies: list[float] = []
            correlations: list[float] = []
            for cells, n_cells in cell_sets:
                counts = np.bincount(cells[lo:hi], minlength=n_cells).astype(float)
                counts = counts[counts > 0]
                p = counts / counts.sum()
                entropies.append(float(-np.sum(p * np.log(p))))
                correlations.append(float(np.sum(p * p)))
            d2 = float(np.polyfit(-np.log(factors), np.log(correlations), 1)[0])
            table[row] = [d2, entropies[1], correlations[1]]
    for width, table in tables.items():
        for col, name in ((0, "correlation dimension"), (1, "spatial entropy"), (2, "correlation sum")):
            out[f"{name}, {width} days"] = _weekly_expand(marks, table[:, col], n)
    return out


# --- 24. ETAS transformed-time residuals ----------------------------------------


def signals_etas_residual(ev: dict[str, np.ndarray], n: int, floor: float, early_end: int, params) -> dict[str, np.ndarray]:
    """Shape of Ogata transformed-time intervals, not just the observed/expected rate."""
    out: dict[str, np.ndarray] = {}
    t, m, known = ev["t"], ev["mag"], ev["known"]
    if len(t) < 50:
        return out
    early = (t >= 0) & (t < early_end)
    mu = 0.5 * float(early.sum()) / max(early_end, 1)
    if mu <= 0:
        mu = 0.5 * len(t) / max(n, 1)
    expected = etas_expected(t, m, floor, n, params, mu)
    cumulative = np.concatenate([[0.0], np.cumsum(expected)])
    day = np.clip(np.floor(t).astype(int), 0, n - 1)
    transformed = cumulative[day] + expected[day] * np.clip(t - day, 0.0, 1.0)
    increments = np.maximum(np.diff(transformed), 1e-9)
    seen = known[1:]
    for size in (50, 100):
        if increments.size < size + 5:
            continue
        mean = _last_n_mean(increments, size)
        second = _last_n_mean(increments * increments, size)
        cv = np.sqrt(np.maximum(second - mean * mean, 0.0)) / np.maximum(mean, 1e-9)
        out[f"ETAS residual interval CV, last {size}"] = _at_mornings(seen, cv, n)
        out[f"ETAS residual log interval, last {size}"] = _at_mornings(seen, _last_n_mean(np.log(increments), size), n)
        out[f"ETAS residual short-interval share, last {size}"] = _at_mornings(
            seen, _last_n_mean((increments < 0.2).astype(float), size), n
        )
    return out


# --- 25. successive-event network ------------------------------------------------


def _graph_metrics(seq: np.ndarray) -> tuple[float, float, float, float]:
    if len(seq) < 10:
        return (np.nan, np.nan, np.nan, np.nan)
    nodes, _counts = np.unique(seq, return_counts=True)
    adjacency: dict[int, set[int]] = {int(node): set() for node in nodes}
    for a, b in zip(seq[:-1], seq[1:]):
        if a == b:
            continue
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    clustering: list[float] = []
    for node, neighbours in adjacency.items():
        degree = len(neighbours)
        if degree < 2:
            continue
        ordered = sorted(neighbours)
        links = 0
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if right in adjacency[left]:
                    links += 1
        clustering.append(2.0 * links / (degree * (degree - 1)))
    seen_nodes: set[int] = set()
    largest = 0
    for node in adjacency:
        if node in seen_nodes:
            continue
        stack = [node]
        seen_nodes.add(node)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in adjacency[current]:
                if nxt not in seen_nodes:
                    seen_nodes.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    hub = max((len(v) for v in adjacency.values()), default=0) / max(len(adjacency) - 1, 1)
    return (
        float(np.mean(clustering)) if clustering else np.nan,
        float(np.mean(seq[1:] == seq[:-1])),
        float(hub),
        float(largest / max(len(adjacency), 1)),
    )


def signals_network_graph(frame: pd.DataFrame, place: str, start: pd.Timestamp, n: int, floor: float) -> dict[str, np.ndarray]:
    """Clustering, repeats, hubs, and connectedness of the successive-event cell network."""
    out: dict[str, np.ndarray] = {}
    ev = _events(frame, start, n, floor)
    if len(ev["t"]) < 220:
        return out
    cells, _n_cells = _cell_ids(ev["lat"], ev["lon"], place)
    known = ev["known"]
    marks = np.arange(0, n, 7)
    tables = {size: np.full((marks.size, 4), np.nan) for size in (100, 200)}
    for row, morning in enumerate(marks):
        hi = np.searchsorted(known, morning, side="right")
        for size, table in tables.items():
            if hi >= size:
                table[row] = _graph_metrics(cells[hi - size : hi])
    for size, table in tables.items():
        for col, name in ((0, "clustering"), (1, "same-cell repeat share"), (2, "hub strength"), (3, "largest component")):
            out[f"network {name}, last {size} quakes"] = _weekly_expand(marks, table[:, col], n)
    return out


# --- 26. Bath's-law magnitude gap --------------------------------------------------


def signals_bath_gap(ev: dict[str, np.ndarray], n: int, floor: float, target: float) -> dict[str, np.ndarray]:
    """Gap between a large quake and its largest nearby follower in the next 30 days."""
    out: dict[str, np.ndarray] = {}
    t, m, lat, lon, known = ev["t"], ev["mag"], ev["lat"], ev["lon"], ev["known"]
    anchor_floor = max(floor + 1.0, target - 1.0)
    anchors = np.flatnonzero(m >= anchor_floor)
    gaps: list[float] = []
    ready: list[int] = []
    for j in anchors:
        hi = np.searchsorted(t, t[j] + 30.0, side="right")
        if hi <= j + 1:
            continue
        cand = slice(j + 1, hi)
        radius = max(50.0, float(uhrhammer_km(m[j])))
        dist = haversine_km(lon[j], lat[j], lon[cand], lat[cand])
        within = dist <= radius
        if not within.any():
            continue
        gaps.append(float(m[j] - np.max(m[cand][within])))
        ready.append(int(min(known[j] + 32, n)))
    if len(gaps) < 15:
        return out
    gap_arr = np.asarray(gaps, dtype=float)
    ready_arr = np.asarray(ready, dtype=np.int64)
    for size in (5, 10):
        if len(gap_arr) < size:
            continue
        out[f"Bath gap mean, last {size} sequences"] = _at_mornings(ready_arr, _last_n_mean(gap_arr, size), n)
        low = pd.Series(gap_arr).rolling(size, min_periods=size).min().to_numpy()
        out[f"Bath gap minimum, last {size} sequences"] = _at_mornings(ready_arr, low, n)
        out[f"Bath close-follower share, last {size} sequences"] = _at_mornings(
            ready_arr, _last_n_mean((gap_arr < 0.5).astype(float), size), n
        )
    return out


# --- 27. stochastic-declustering background probability -----------------------------


def signals_stochastic_background(ev: dict[str, np.ndarray], n: int, floor: float, early_end: int, params) -> dict[str, np.ndarray]:
    """Share of seismicity assigned to the background by temporal ETAS probabilities."""
    out: dict[str, np.ndarray] = {}
    t, m, known = ev["t"], ev["mag"], ev["known"]
    if len(t) < 50:
        return out
    early = (t >= 0) & (t < early_end)
    mu = 0.5 * float(early.sum()) / max(early_end, 1)
    if mu <= 0:
        mu = 0.5 * len(t) / max(n, 1)
    expected = etas_expected(t, m, floor, n, params, mu)
    day = np.clip(np.floor(t).astype(int), 0, n - 1)
    background_probability = np.clip(mu / np.maximum(expected[day], mu), 0.0, 1.0)
    bg_daily = _daily_sum(known, background_probability, n)
    total_daily = _daily_sum(known, None, n)
    long_bg = _trailing(bg_daily, 1826)
    long_total = _trailing(total_daily, 1826)
    long_share = long_bg / np.maximum(long_total, 1.0)
    for width in (90, 365):
        bg = _trailing(bg_daily, width)
        total = _trailing(total_daily, width)
        share = np.where(total >= 10, bg / np.maximum(total, 1.0), np.nan)
        out[f"background share, {width} days vs 5 years"] = share - long_share
        out[f"background rate ratio, {width} days"] = np.log(((bg / width) + 1e-4) / ((long_bg / 1826.0) + 1e-4))
    return out


# --- 28. seismic moment budget -------------------------------------------------------


def signals_moment_balance(ev: dict[str, np.ndarray], n: int, floor: float) -> dict[str, np.ndarray]:
    """Moment release against the long-run budget, plus release concentration."""
    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    if len(m) < 120:
        return out
    relative_moment = 10.0 ** (1.5 * np.clip(m - floor, 0.0, 6.5))
    daily = _daily_sum(known, relative_moment, n)
    long_total = _trailing(daily, 3652)
    long_rate = long_total / 3652.0
    daily_series = pd.Series(daily)
    for width in (365, 1826):
        recent = _trailing(daily, width)
        expected = long_rate * width
        out[f"moment release ratio, {width} days vs 10 years"] = np.log(
            ((recent / width) + 1e-12) / (long_rate + 1e-12)
        )
        out[f"moment deficit share, {width} days"] = np.clip(1.0 - recent / np.maximum(expected, 1e-12), 0.0, 1.0)
        largest = daily_series.rolling(width, min_periods=width).max().to_numpy()
        out[f"largest-day moment share, {width} days"] = largest / np.maximum(recent, 1e-12)
    return out


# --- 29. inverse-Omori acceleration --------------------------------------------------- 


def signals_inverse_omori(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Acceleration of the quake rate toward the present, without assuming a failure date."""
    out: dict[str, np.ndarray] = {}
    known = ev["known"]
    if len(known) < 120:
        return out
    counts = _daily_sum(known, None, n)
    prefix = np.concatenate([[0.0], np.cumsum(counts)])
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    for width in (90, 365, 730):
        inc = _segment_increments(prefix, width)
        slope = (np.log(inc + 1.0) @ x) / float(x @ x)
        out[f"inverse-Omori segment slope, {width} days"] = slope
        out[f"inverse-Omori last/first ratio, {width} days"] = np.log((inc[:, 3] + 1.0) / (inc[:, 0] + 1.0))
        out[f"inverse-Omori last-half ratio, {width} days"] = np.log(
            (inc[:, 2] + inc[:, 3] + 1.0) / (inc[:, 0] + inc[:, 1] + 1.0)
        )
    return out


# --- 30. natural-time entropy under time reversal --------------------------------------


def natural_time_entropy(magnitudes: np.ndarray) -> float:
    """Entropy S in natural time for one ordered magnitude window."""
    m = np.asarray(magnitudes, dtype=float)
    q = 10.0 ** (1.5 * (m - np.max(m)))
    p = q / q.sum()
    chi = np.arange(1, len(m) + 1) / len(m)
    mean = float(np.sum(p * chi))
    return float(np.sum(p * chi * np.log(chi)) - mean * math.log(mean))


def _natural_time_stats(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chi = np.arange(1, block.shape[1] + 1) / block.shape[1]
    chi_log = chi * np.log(chi)
    q = 10.0 ** (1.5 * (block - block.max(axis=1, keepdims=True)))
    p = q / q.sum(axis=1, keepdims=True)
    mean = p @ chi
    entropy = p @ chi_log - mean * np.log(mean)
    reversed_p = p[:, ::-1]
    reversed_mean = reversed_p @ chi
    reversed_entropy = reversed_p @ chi_log - reversed_mean * np.log(reversed_mean)
    return entropy, entropy - reversed_entropy


def signals_reversal_entropy(ev: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Entropy change when the magnitude order is reversed, and its fluctuation."""
    from numpy.lib.stride_tricks import sliding_window_view

    out: dict[str, np.ndarray] = {}
    m, known = ev["mag"], ev["known"]
    for size in (40, 100, 200):
        if len(m) < size + 10:
            continue
        entropy = np.full(len(m), np.nan)
        delta = np.full(len(m), np.nan)
        windows = sliding_window_view(m, size)
        for lo in range(0, windows.shape[0], 20000):
            block = windows[lo : lo + 20000]
            s, d = _natural_time_stats(block)
            entropy[size - 1 + lo : size - 1 + lo + block.shape[0]] = s
            delta[size - 1 + lo : size - 1 + lo + block.shape[0]] = d
        out[f"natural-time entropy, last {size} quakes"] = _at_mornings(known, entropy, n)
        out[f"time-reversal entropy change, last {size} quakes"] = _at_mornings(known, delta, n)
        mean = _last_n_mean(delta, 20)
        second = _last_n_mean(delta * delta, 20)
        fluctuation = np.sqrt(np.maximum(second - mean * mean, 0.0))
        out[f"time-reversal entropy fluctuation, last {size} quakes"] = _at_mornings(known, fluctuation, n)
    return out


# --- shared test ---------------------------------------------------------------


def _masks(days: pd.DatetimeIndex, horizon: int, catalog_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closed = np.asarray(days + pd.Timedelta(days=int(horizon)) <= catalog_end)
    early = week_mask(closed & np.asarray(days + pd.Timedelta(days=int(horizon)) <= EARLY_END), stride=int(horizon), fallback=False)
    develop = week_mask(
        closed & np.asarray(days >= EARLY_END) & np.asarray(days + pd.Timedelta(days=int(horizon)) <= LATER_START),
        stride=int(horizon),
        fallback=False,
    )
    later = week_mask(closed & np.asarray(days >= LATER_START), stride=int(horizon), fallback=False)
    full = week_mask(closed, stride=int(horizon), fallback=False)
    return early, develop, later, full


def _odds_shift(hits: np.ndarray, chosen: np.ndarray, base: float) -> float:
    count = float(chosen.sum())
    share = (float(hits[chosen].sum()) + PRIOR_WINDOWS * base) / (count + PRIOR_WINDOWS)
    return float(_logit(share) - _logit(base))


def mapped_chances(signal: np.ndarray, hits: np.ndarray, usual: np.ndarray, early: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    """Ways to turn a signal into a chance. Every cut and every odds ratio uses old windows only."""
    finite = np.isfinite(signal)
    pool = early & finite & np.isfinite(hits) & np.isfinite(usual)
    if int(pool.sum()) < 40:
        return
    base = float(hits[pool].mean())
    if base <= 0.0 or base >= 1.0:
        return
    values = signal[pool]
    center = _logit(usual)
    for side in ("high", "low"):
        for share in SHARES:
            cut = float(np.quantile(values, 1.0 - share if side == "high" else share))
            on = finite & ((signal >= cut) if side == "high" else (signal <= cut))
            off = finite & ~on
            if int((on & pool).sum()) < 8 or int((off & pool).sum()) < 8:
                continue
            up = _odds_shift(hits[pool], on[pool], base)
            down = _odds_shift(hits[pool], off[pool], base)
            shift = np.where(on, up, np.where(off, down, 0.0))
            yield f"{side} {int(share * 100)}%", _expit(center + shift)
    edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
    if edges.size == 3:
        band = np.digitize(signal, edges)
        shifts = []
        for level in range(4):
            chosen = (band == level) & finite
            if int((chosen & pool).sum()) < 8:
                break
            shifts.append(_odds_shift(hits[pool], chosen[pool], base))
        if len(shifts) == 4:
            shift = np.where(finite, np.asarray(shifts)[np.clip(band, 0, 3)], 0.0)
            yield "four bins", _expit(center + shift)


def _choice_key(scored: dict[str, Any] | None) -> tuple:
    if not scored:
        return (False, -9.0, -9.0)
    gain = scored.get("information_gain")
    gap = scored.get("gap")
    gain_n = -9.0 if gain is None else float(gain)
    gap_n = -9.0 if gap is None else float(gap)
    return (scored.get("verdict") == "up" and gain_n > 0.0, gain_n, gap_n)


def pick_variation(
    signals: dict[str, np.ndarray],
    hits: np.ndarray,
    usual: np.ndarray,
    early: np.ndarray,
    develop: np.ndarray,
) -> tuple[dict[str, Any] | None, int]:
    """Best variation on 1990–2008. 2009 onward is never read here."""
    best: dict[str, Any] | None = None
    tried = 0
    for name, signal in signals.items():
        signal = np.asarray(signal, dtype=float)
        if signal.shape != hits.shape:
            continue
        for mapping, chance in mapped_chances(signal, hits, usual, early):
            tried += 1
            scored = split_windows(chance[develop], hits[develop], usual[develop])
            if best is None or _choice_key(scored) > _choice_key(best["develop"]):
                best = {"variation": name, "mapping": mapping, "develop": scored, "chance": chance}
    return best, tried


def _card(
    idea: str,
    signals: dict[str, np.ndarray],
    hits: np.ndarray,
    usual: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray | None]:
    early, develop, later, _full = masks
    best, tried = pick_variation(signals, hits, usual, early, develop)
    adopted = bool(best is not None and _choice_key(best["develop"])[0])
    row: dict[str, Any] = {
        "idea": idea,
        "variations_tried": tried,
        "adopted": adopted,
        "variation": None if best is None else best["variation"],
        "mapping": None if best is None else best["mapping"],
        "develop": None if best is None else best["develop"],
        "later": None,
        "pointed_right": False,
        "passed": False,
    }
    if not adopted:
        return row, None
    chance = best["chance"]
    scored = split_windows(chance[later], hits[later], usual[later])
    gain = scored.get("information_gain") or 0.0
    row["later"] = scored
    row["pointed_right"] = bool(scored.get("verdict") == "up" and gain > 0)
    row["passed"] = bool(scored.get("verdict") == "up" and gain >= USEFUL_GAIN)
    return row, chance


def _signals_for(idea: str, place: str, context: dict[str, Any]) -> dict[str, np.ndarray]:
    start, n, early_end = context["start"], context["n"], context["early_end"]
    catalog = context["place_catalogs"][place]
    floor = FLOOR[place]
    if idea == "bpositive":
        return signals_bpositive(_events(catalog, start, n, floor), n)
    if idea == "tidal":
        return signals_tidal(_events(catalog, start, n, floor), n, start, floor, early_end)
    if idea == "natural_time":
        return signals_natural_time(_events(catalog, start, n, floor), n)
    if idea == "etas_quiet":
        rate_floor = RATE_FLOOR[place]
        return signals_etas_quiet(_events(catalog, start, n, rate_floor), n, rate_floor, early_end, context["etas"])
    if idea == "quiet_aftershocks":
        return signals_quiet_aftershocks(_events(catalog, start, n, floor), n, floor, TARGET[place], early_end)
    if idea == "localization":
        return signals_localization(catalog, place, start, n, RATE_FLOOR[place])
    if idea == "coalescence":
        cfloor = CLUSTER_FLOOR[place]
        return signals_coalescence(_events(catalog, start, n, cfloor), n, cfloor, early_end)
    if idea == "earth_rotation":
        return context["rotation"]
    if idea == "hidden_state":
        return signals_hidden_state(catalog, place, start, n, early_end)
    if idea == "nowcast_eps":
        return signals_nowcast_eps(_events(catalog, start, n, floor), n, floor, TARGET[place], early_end)
    if idea == "burstiness":
        return signals_burstiness(_events(catalog, start, n, floor), n)
    if idea == "rtl_score":
        return signals_rtl(_events(catalog, start, n, floor), n, floor)
    if idea == "m8_tip":
        return signals_m8_tip(_events(catalog, start, n, floor), n, floor)
    if idea == "depth_shift":
        return signals_depth_shift(_events(catalog, start, n, floor), n)
    if idea == "rake_regime":
        return signals_rake_regime(context["cmt"], place, start, n)
    if idea == "polar_wobble":
        return context["wobble"]
    if idea == "spatial_migration":
        return signals_spatial_migration(_events(catalog, start, n, floor), n)
    if idea == "volcano_magmatic":
        return context["volcano"]
    if idea == "pattern_informatics":
        return signals_pattern_informatics(catalog, place, start, n, floor)
    if idea == "amr_curvature":
        return signals_amr_curvature(_events(catalog, start, n, floor), n, floor)
    if idea == "fractal_entropy":
        return signals_fractal_entropy(catalog, place, start, n, floor)
    if idea == "etas_residual":
        rate_floor = RATE_FLOOR[place]
        return signals_etas_residual(_events(catalog, start, n, rate_floor), n, rate_floor, early_end, context["etas"])
    if idea == "network_graph":
        return signals_network_graph(catalog, place, start, n, floor)
    if idea == "bath_gap":
        return signals_bath_gap(_events(catalog, start, n, floor), n, floor, TARGET[place])
    if idea == "stochastic_background":
        rate_floor = RATE_FLOOR[place]
        return signals_stochastic_background(
            _events(catalog, start, n, rate_floor), n, rate_floor, early_end, context["etas"]
        )
    if idea == "moment_balance":
        return signals_moment_balance(_events(catalog, start, n, floor), n, floor)
    if idea == "inverse_omori":
        return signals_inverse_omori(_events(catalog, start, n, floor), n)
    if idea == "reversal_entropy":
        return signals_reversal_entropy(_events(catalog, start, n, floor), n)
    return {}


def _save_idea(
    idea: str,
    frames: list[pd.DataFrame],
    tape_name: str,
    version: int,
    chosen: list[dict[str, Any]],
    root: Path | None,
) -> dict[str, Any] | None:
    kept = [frame for frame in frames if frame is not None and len(frame)]
    if not kept:
        return None
    return save_forecasts(
        pd.concat(kept, ignore_index=True),
        source="idea",
        name=f"idea_{idea}",
        label=IDEA_LABELS.get(idea, idea),
        root=root,
        label_version=version,
        tape=tape_name,
        answer_key=f"fresh.v{version}",
        knobs={"cards": chosen},
        note=(
            "New idea, chosen on years before 2009. 2009 onward was not used to pick it. "
            "This does not change the live exam."
        ),
    )


def _crowd_cards(root: Path | None, tape_name: str, version: int, mags: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    cards: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for horizon in (3, 7, 30):
        table = _crowd_frames(root, tape_name, horizon)
        if table is None or table.empty:
            continue
        for place in PLACES:
            part = table[table["region"] == place]
            if part.empty:
                continue
            z = _logit(part["p"].to_numpy()) - _logit(part["climatology_p"].to_numpy())
            part = part.assign(z=z)
            wide = part.pivot_table(index="issue_day", columns="rule", values="z", aggfunc="last").fillna(0.0)
            base = part.drop_duplicates("issue_day").set_index("issue_day").loc[wide.index]
            days = pd.DatetimeIndex(pd.to_datetime(wide.index, utc=True))
            ends = days + pd.Timedelta(days=horizon)
            closed = base["closed"].to_numpy(dtype=bool)
            hits = base["happened"].to_numpy(dtype=float)
            usual = base["climatology_p"].to_numpy(dtype=float)
            early = closed & np.asarray(ends <= EARLY_END)
            develop = closed & np.asarray(days >= EARLY_END) & np.asarray(ends <= LATER_START)
            later = closed & np.asarray(days >= LATER_START)
            x = wide.to_numpy(dtype=float)
            if int(early.sum()) < 40 or x.shape[1] == 0:
                continue
            signals: dict[str, np.ndarray] = {}
            votes = np.sign(x).sum(axis=1)
            signals["rules marking up minus rules marking down"] = votes
            best: dict[str, Any] | None = None
            tried = 0
            offset = _logit(usual)
            for penalty in (1.0, 3.0, 10.0, 30.0, 100.0):
                weights = _ridge_logistic(x[early], hits[early], offset[early], penalty)
                chance = _expit(offset + x @ weights)
                tried += 1
                scored = split_windows(chance[develop], hits[develop], usual[develop])
                if best is None or _choice_key(scored) > _choice_key(best["develop"]):
                    best = {"variation": f"weighted rules, penalty {penalty:g}", "mapping": "logistic", "develop": scored, "chance": chance}
            vote_best, vote_tried = pick_variation(signals, hits, usual, early, develop)
            tried += vote_tried
            if vote_best is not None and (best is None or _choice_key(vote_best["develop"]) > _choice_key(best["develop"])):
                best = vote_best
            adopted = bool(best is not None and _choice_key(best["develop"])[0])
            row: dict[str, Any] = {
                "idea": "crowd",
                "variations_tried": tried,
                "adopted": adopted,
                "variation": None if best is None else best["variation"],
                "mapping": None if best is None else best["mapping"],
                "develop": None if best is None else best["develop"],
                "later": None,
                "pointed_right": False,
                "passed": False,
                "place": place,
                "place_label": LABELS.get(place, place),
                "horizon_days": horizon,
                "magnitude_min": mags.get(str(horizon), {}).get(place),
                "answer_key": f"fresh.v{version}",
                "tape": tape_name,
            }
            if adopted:
                chance = best["chance"]
                scored = split_windows(chance[later], hits[later], usual[later])
                gain = scored.get("information_gain") or 0.0
                row["later"] = scored
                row["pointed_right"] = bool(scored.get("verdict") == "up" and gain > 0)
                row["passed"] = bool(scored.get("verdict") == "up" and gain >= USEFUL_GAIN)
                frames.append(
                    forecast_frame(
                        days=days,
                        region=place,
                        label=LABELS.get(place, place),
                        horizon=horizon,
                        magnitude=float(mags.get(str(horizon), {}).get(place) or 0.0),
                        p=chance,
                        climatology=usual,
                        happened=hits,
                        mask=closed | (np.arange(len(days)) == len(days) - 1),
                        catalog_end=days[-1],
                    )
                )
            cards.append(row)
    return cards, frames


def run_idea_bench(root: Path | None = None, ideas: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Score the ten ideas for each place, window length, and answer key."""
    wanted = [name for name, _label, _source in IDEAS if ideas is None or name in ideas]
    tapes = []
    for version in (1, 2):
        tape = load_tape(root, name=TAPE_FOR_LABEL[version])
        if tape is None:
            print(f"skip answer key {version}: tape is not built", flush=True)
            continue
        tapes.append((version, TAPE_FOR_LABEL[version], tape))
    if not tapes:
        return {"ready": False, "note": "No strategy tape is built yet.", "writes_issued_p": False}
    meta = tapes[0][2]["meta"]
    days = pd.date_range(meta["start"], periods=int(meta["days"]), freq="D", tz="UTC")
    n = len(days)
    start = days[0]
    early_end = int((EARLY_END - start) / pd.Timedelta(days=1))
    catalogs = _catalogs(root)
    from eqpred.ingest.run import latest_raw

    try:
        cmt = latest_raw("gcmt.cmt", root).copy()
    except FileNotFoundError:
        cmt = pd.DataFrame()
    context = {
        "start": start,
        "n": n,
        "early_end": early_end,
        "place_catalogs": {place: _place_catalog(catalogs, place) for place in PLACES},
        "etas": _etas_params(root),
        "rotation": signals_earth_rotation(root, days) if "earth_rotation" in wanted else {},
        "wobble": signals_polar_wobble(root, days) if "polar_wobble" in wanted else {},
        "volcano": signals_volcano_magmatic(root, days) if "volcano_magmatic" in wanted else {},
        "cmt": cmt if "rake_regime" in wanted else pd.DataFrame(),
    }
    catalog_end = days[-1]
    mask_cache = {h: _masks(days, h, catalog_end) for h in HORIZONS}
    cards: list[dict[str, Any]] = []
    for idea in wanted:
        if idea in {"crowd", "hedge_expert"}:
            builder = _crowd_cards if idea == "crowd" else _hedge_cards
            for version, tape_name, tape in tapes:
                print(f"idea {idea}, answer key {version}", flush=True)
                mags = {h: tape["meta"]["horizons"][h]["mags"] for h in tape["meta"]["horizons"]}
                rows, frames = builder(root, tape_name, version, mags)
                chosen = [{k: r[k] for k in ("place", "horizon_days", "variation", "mapping")} for r in rows if r["adopted"]]
                _save_idea(idea, frames, tape_name, version, chosen, root)
                cards.extend(rows)
            continue
        signal_cache: dict[str, dict[str, np.ndarray]] = {}
        for place in PLACES:
            print(f"idea {idea}, {LABELS.get(place, place)}", flush=True)
            signal_cache[place] = _signals_for(idea, place, context)
        for version, tape_name, tape in tapes:
            packed = tape["npz"]
            frames: list[pd.DataFrame] = []
            chosen: list[dict[str, Any]] = []
            for horizon in HORIZONS:
                info = tape["meta"]["horizons"].get(str(horizon))
                if not info:
                    continue
                masks = mask_cache[horizon]
                for place in PLACES:
                    if place not in info["regions"]:
                        continue
                    hits = np.asarray(packed[f"{horizon}_{place}_y"], dtype=float)
                    usual = np.asarray(packed[f"{horizon}_{place}_c"], dtype=float)
                    row, chance = _card(idea, signal_cache[place], hits, usual, masks)
                    row.update(
                        {
                            "place": place,
                            "place_label": LABELS.get(place, place),
                            "horizon_days": horizon,
                            "magnitude_min": info["mags"][place],
                            "answer_key": f"fresh.v{version}",
                            "tape": tape_name,
                        }
                    )
                    cards.append(row)
                    if chance is not None:
                        keep = np.array(masks[3], copy=True)
                        keep[-1] = True
                        frames.append(
                            forecast_frame(
                                days=days,
                                region=place,
                                label=LABELS.get(place, place),
                                horizon=horizon,
                                magnitude=float(info["mags"][place]),
                                p=chance,
                                climatology=usual,
                                happened=hits,
                                mask=keep,
                                catalog_end=catalog_end,
                            )
                        )
                        chosen.append({"place": place, "horizon_days": horizon, "variation": row["variation"], "mapping": row["mapping"]})
            _save_idea(idea, frames, tape_name, version, chosen, root)
    if ideas is not None and bench_path(root).exists():
        prior = load_idea_bench(root) or {}
        kept_old = [c for c in (prior.get("cards") or []) if c.get("idea") not in set(wanted)]
        cards = kept_old + cards
    summary = _summaries(cards)
    report = {
        "ready": True,
        "writes_issued_p": False,
        "useful_gain": USEFUL_GAIN,
        "ideas": summary,
        "cards": cards,
        "n_cards": len(cards),
        "n_adopted": sum(1 for c in cards if c["adopted"]),
        "n_pointed_right": sum(1 for c in cards if c["pointed_right"]),
        "n_passed": sum(1 for c in cards if c["passed"]),
        "note": (
            "Each idea sets its odds on windows that closed before 1990, picks its settings on 1990–2008, "
            "and is scored once on 2009 onward, separately for each place and window length. "
            "A chance only moves the usual rate by the odds it earned on old years. "
            "A pass needs the marked-up windows to happen more often and a gain of at least 0.01. "
            "This does not change the live exam."
        ),
    }
    write_json(bench_path(root), report)
    return report


def _summaries(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, label, source in IDEAS:
        mine = [c for c in cards if c["idea"] == name]
        if not mine:
            continue
        row: dict[str, Any] = {"idea": name, "label": label, "source": source}
        for key in ("fresh.v1", "fresh.v2"):
            group = [c for c in mine if c["answer_key"] == key]
            if not group:
                continue
            gains = [
                float((c.get("later") or {}).get("information_gain") or 0.0) if c["adopted"] else 0.0
                for c in group
            ]
            tag = "v1" if key == "fresh.v1" else "v2"
            row[f"{tag}_cards"] = len(group)
            row[f"{tag}_adopted"] = sum(1 for c in group if c["adopted"])
            row[f"{tag}_pointed_right"] = sum(1 for c in group if c["pointed_right"])
            row[f"{tag}_passed"] = sum(1 for c in group if c["passed"])
            row[f"{tag}_mean_later_gain"] = round(float(np.mean(gains)), 4)
            steady = []
            for place in PLACES:
                rows = {c["horizon_days"]: c for c in group if c["place"] == place}
                if all(rows.get(h) and rows[h]["pointed_right"] for h in (3, 7, 30)):
                    steady.append(place)
            row[f"{tag}_steady_places"] = steady
        out.append(row)
    return out


def record_in_rank(report: dict[str, Any]) -> int:
    """Add the ideas to the list of rules already tried. Old scores are not changed."""
    path = Path(__file__).resolve().parent / "strategy_rank.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {f"idea_{row['idea']}" for row in report.get("ideas") or []}
    ranked = [row for row in data.get("ranked") or [] if row.get("strategy") not in names]
    for row in report.get("ideas") or []:
        if "v1_mean_later_gain" not in row:
            continue
        ranked.append(
            {
                "strategy": f"idea_{row['idea']}",
                "label": row["label"],
                "mean_later_information_gain": row["v1_mean_later_gain"],
                "cards": row["v1_cards"],
                "kind": "idea",
                "source": row["source"],
            }
        )
    ranked.sort(key=lambda r: (-(r.get("mean_later_information_gain") if r.get("mean_later_information_gain") is not None else -9.0)))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    data["ranked"] = ranked
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return len(ranked)


def load_idea_bench(root: Path | None = None) -> dict[str, Any] | None:
    path = bench_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
