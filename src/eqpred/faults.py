from __future__ import annotations

# Locked receiver planes. Not an NSHM. Strike/dip/rake in degrees, Aki convention.
# Alpine: NE-striking dextral. Hikurangi / Kermadec: W-dipping megathrust.
FAULTS = {
    "hikurangi": {"strike": 45.0, "dip": 12.0, "rake": 90.0, "mu": 0.4},
    "alpine": {"strike": 55.0, "dip": 50.0, "rake": 180.0, "mu": 0.4},
    "kermadec": {"strike": 20.0, "dip": 20.0, "rake": 90.0, "mu": 0.4},
}
