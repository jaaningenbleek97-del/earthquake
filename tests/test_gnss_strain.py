import numpy as np
import pandas as pd

from eqpred.features.gnss_strain import gnss_strain_columns


def test_gnss_strain_dilatation_positive_when_sites_move_apart():
    n = 80
    days = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    t = np.arange(n, dtype=float)
    feat = pd.DataFrame(
        {
            "issue_time": days,
            # AUCK north of WGTN: push AUCK farther north and WGTN farther south.
            "gnss_auck_east_mm": np.zeros(n),
            "gnss_auck_north_mm": 2.0 * t,
            "gnss_wgtn_east_mm": np.zeros(n),
            "gnss_wgtn_north_mm": -2.0 * t,
            "gnss_lytt_east_mm": np.zeros(n),
            "gnss_lytt_north_mm": np.zeros(n),
            "gnss_dund_east_mm": np.zeros(n),
            "gnss_dund_north_mm": np.zeros(n),
        }
    )
    out = gnss_strain_columns(feat)
    dil = out["gnss_strain_nz_dilatation_30d"].to_numpy()
    assert np.isfinite(dil[40])
    assert float(dil[40]) > 0
    ext = out["gnss_strain_hikurangi_extension_30d"].to_numpy()
    assert np.isfinite(ext[40])
    assert float(ext[40]) > 0


def test_gnss_strain_empty_without_sites():
    feat = pd.DataFrame({"issue_time": pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")})
    out = gnss_strain_columns(feat)
    assert "gnss_strain_nz_dilatation_30d" in out.columns
    assert out["gnss_strain_nz_dilatation_30d"].isna().all()
