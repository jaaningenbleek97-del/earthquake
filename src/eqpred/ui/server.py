from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from eqpred.eval.backtest import (
    available_lags,
    available_targets,
    list_drivers,
    load_backtest,
    paper_trade,
)
from eqpred.eval.brain import load_brain, run_brain
from eqpred.eval.learn import run_learn
from eqpred.eval.promote import load_promote
from eqpred.eval.transfer import load_transfer, run_transfer
from eqpred.eval.hunt import load_hunt, load_hunt_fresh
from eqpred.ui.data import _trim_hunt, day_view, feature_series, globe_view, map_quakes, query_ledger, snapshot

STATIC = Path(__file__).resolve().parent / "static"


class LabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"ui {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if path == "/api/ping":
            from eqpred.ui import data as ui_data
            return self._json(
                {
                    "ok": True,
                    "data_file": str(ui_data.__file__),
                    "has_hunt_fresh": "hunt_fresh" in snapshot.__code__.co_consts,
                }
            )
        if path == "/api/snapshot":
            return self._json(snapshot())
        if path == "/api/hunt":
            fresh = (query.get("fresh") or "1") not in {"0", "false", "no"}
            pack = load_hunt_fresh() if fresh else load_hunt()
            return self._json(_trim_hunt(pack))
        if path == "/api/ledger":
            return self._json(
                query_ledger(
                    model=query.get("model") or None,
                    region=query.get("region") or None,
                    kind=query.get("kind") or None,
                    hit=query.get("hit") or None,
                    limit=int(query.get("limit") or 150),
                    offset=int(query.get("offset") or 0),
                )
            )
        if path == "/api/day":
            day = query.get("date") or query.get("day")
            if not day:
                return self._error(400, "date required")
            return self._json(day_view(day))
        if path == "/api/series":
            return self._json(feature_series(start=query.get("start"), end=query.get("end")))
        if path == "/api/quakes":
            return self._json(
                map_quakes(
                    start=query.get("start"),
                    end=query.get("end"),
                    minmag=float(query.get("minmag") or 6.0),
                )
            )
        if path == "/api/globe":
            return self._json(globe_view())
        if path == "/api/backtest":
            driver = query.get("driver")
            if not driver:
                return self._json(
                    {
                        "drivers": list_drivers(),
                        "lags": available_lags(),
                        "targets": available_targets(),
                        "pack": load_backtest(),
                    }
                )
            try:
                return self._json(
                    paper_trade(
                        driver,
                        lag_days=int(query.get("lag") or 0),
                        region_id=query.get("region") or "global",
                        share=float(query.get("share") or 0.2),
                        magnitude_min=float(query["mag"]) if query.get("mag") else None,
                    )
                )
            except (ValueError, FileNotFoundError) as exc:
                return self._error(400, str(exc))
        if path == "/api/brain":
            if query.get("train") == "1":
                return self._json(run_brain())
            return self._json(load_brain() or {"writes_issued_p": False, "note": "Run eqpred brain or press Learn."})
        if path == "/api/transfer":
            if query.get("train") == "1":
                return self._json(run_transfer())
            return self._json(load_transfer() or {"writes_issued_p": False, "note": "Run eqpred transfer or Learn."})
        if path == "/api/promote":
            return self._json(load_promote() or {"pass": False, "writes_issued_p": False, "note": "Run eqpred learn."})
        if path == "/api/learn":
            if query.get("train") == "1":
                return self._json(run_learn())
            brain = load_brain() or {}
            return self._json(
                {
                    "writes_issued_p": False,
                    "brain": brain,
                    "transfer": load_transfer(),
                    "promote": load_promote(),
                    "note": brain.get("note") or "Run eqpred learn.",
                }
            )
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), LabHandler)
    print(f"lab UI http://{host}:{port}", flush=True)
    httpd.serve_forever()
