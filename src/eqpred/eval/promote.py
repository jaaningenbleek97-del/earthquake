from __future__ import annotations

"""Promote chip. Does not lock or rewrite issued p.

Issued cards are the fresh-shock model. This score asks whether that
model beat climatology / ETAS on later years.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eqpred.eval.brain import load_brain
from eqpred.paths import data_dir, model_dir
from eqpred.records import iso


def load_promote(root: Path | None = None) -> dict[str, Any] | None:
    path = data_dir(root) / "eval" / "promote.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_promote(brain: dict[str, Any] | None) -> dict[str, Any]:
    if not brain:
        return {
            "pass": False,
            "writes_issued_p": False,
            "reason": "no brain report",
            "walk_forward_lift": None,
            "locked_brier": None,
            "locked_etas_brier": None,
        }
    wf = (brain.get("walk_forward") or {}).get("mean_lift")
    brier = brain.get("locked_brier")
    etas = brain.get("locked_etas_brier")
    lift_ok = wf is not None and float(wf) > 0
    brier_ok = brier is not None and etas is not None and float(brier) < float(etas)
    passed = bool(lift_ok and brier_ok)
    if passed:
        reason = (
            f"walk-forward lift {float(wf) * 100:+.1f} pt and fresh-shock error "
            f"{float(brier):.4f} < ETAS {float(etas):.4f}"
        )
    else:
        bits = []
        if not lift_ok:
            bits.append("walk-forward lift is not above always-on")
        if not brier_ok:
            bits.append("fresh-shock error is not better than ETAS")
        reason = "; ".join(bits) or "chip failed"
    return {
        "pass": passed,
        "writes_issued_p": False,
        "reason": reason,
        "walk_forward_lift": None if wf is None else float(wf),
        "locked_brier": None if brier is None else float(brier),
        "locked_etas_brier": None if etas is None else float(etas),
        "locked_lift": brain.get("locked_lift"),
    }


def run_promote(root: Path | None = None, brain: dict[str, Any] | None = None) -> dict[str, Any]:
    report = evaluate_promote(brain if brain is not None else load_brain(root))
    report["scanned_at"] = iso(datetime.now(timezone.utc))
    report["note"] = (
        f"Promote {'passed' if report['pass'] else 'held'} as a score chip: {report['reason']}. "
        "Issued cards stay the fresh-shock model. This chip does not lock or rewrite p."
    )
    dest = data_dir(root) / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "promote.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # Never write issued model files from this path.
    _ = model_dir(root)
    return report
