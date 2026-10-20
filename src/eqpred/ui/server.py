from __future__ import annotations

import gzip
import json
import traceback
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
from eqpred.eval.hunt import load_hunt, load_hunt_fresh, load_hunt_sequence
from eqpred.eval.earth_state import load_earth_state, run_earth_state
from eqpred.eval.circus import load_circus, run_circus
from eqpred.eval.leftover import load_leftover, run_leftover
from eqpred.ui.data import _trim_hunt, day_view, feature_series, globe_view, map_quakes, query_ledger, snapshot

STATIC = Path(__file__).resolve().parent / "static"
_TRAINERS = {
    "/api/learn": run_learn,
    "/api/brain": run_brain,
    "/api/transfer": run_transfer,
    "/api/leftover": run_leftover,
    "/api/circus": run_circus,
    "/api/earth-state": run_earth_state,
}


class LabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"ui {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            if parsed.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        if not self._same_origin(post=False):
            return self._error(403, "Open the lab from its own address.")
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self._guarded(self._api_get, parsed.path, query)

    def _guarded(self, handler, *args) -> None:
        """A bad request or a failed load answers with JSON instead of dropping the connection."""
        self._responded = False
        try:
            handler(*args)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return
        except (ValueError, KeyError) as exc:
            self._fail(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._fail(500, f"{type(exc).__name__}: {exc}")

    def _fail(self, status: int, message: str) -> None:
        if not getattr(self, "_responded", False):
            self._error(status, message)

    def _api_get(self, path: str, query: dict[str, str]) -> None:
        if path == "/api/ping":
            from eqpred.ui import data as ui_data
            return self._json(
                {
                    "ok": True,
                    "data_file": str(ui_data.__file__),
                    "has_hunt_fresh": "hunt_fresh" in snapshot.__code__.co_consts,
                    "has_hunt_sequence": "hunt_sequence" in snapshot.__code__.co_consts,
                }
            )
        if path == "/api/snapshot":
            return self._json(snapshot())
        if path == "/api/hunt":
            if (query.get("sequence") or "") in {"1", "true", "yes"}:
                return self._json(_trim_hunt(load_hunt_sequence(), keep_catalog=True))
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
        if query.get("train") == "1" and path in _TRAINERS:
            return self._error(405, "Start training with a POST request from the lab page.")
        if path == "/api/brain":
            return self._json(load_brain() or {"writes_issued_p": False, "note": "Run eqpred brain or press Learn."})
        if path == "/api/transfer":
            return self._json(load_transfer() or {"writes_issued_p": False, "note": "Run eqpred transfer or Learn."})
        if path == "/api/promote":
            return self._json(load_promote() or {"pass": False, "writes_issued_p": False, "note": "Run eqpred learn."})
        if path == "/api/learn":
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
        if path == "/api/leftover":
            return self._json(load_leftover() or {"writes_issued_p": False, "note": "Run eqpred leftover."})
        if path == "/api/circus":
            return self._json(load_circus() or {"writes_issued_p": False, "note": "Run eqpred circus."})
        if path == "/api/earth-state":
            return self._json(
                load_earth_state()
                or {
                    "writes_issued_p": False,
                    "note": "Run eqpred analog to fit the Earth-state tools.",
                }
            )
        if path == "/api/strategy":
            from eqpred.eval.strategy_desk import score_strategy

            def _num(key: str) -> float | None:
                raw = query.get(key)
                if raw is None or raw == "":
                    return None
                return float(raw)

            try:
                return self._json(
                    score_strategy(
                        horizon=int(query.get("horizon") or 3),
                        region=query.get("region") or "all",
                        strategy=query.get("strategy") or "alarm_rarity",
                        shrink=_num("shrink"),
                        odds=_num("odds"),
                        alarm_share=float(query.get("alarm_share") or 0.2),
                        rarity_n0=float(query.get("rarity_n0") or 50),
                        bins=int(query.get("bins") or 5),
                        agree_min=int(query.get("agree_min") or 3),
                        matched_k=None if _num("matched_k") is None else int(float(query["matched_k"])),
                        use_readings=query.get("use_readings") == "1",
                        lag_days=int(float(query.get("lag_days") or 0)),
                        tape_name=query.get("tape") or None,
                    )
                )
            except (ValueError, KeyError, OSError) as exc:
                return self._error(400, str(exc))
        if path == "/api/tapes":
            from eqpred.eval.strategy_desk import DEFAULT_TAPE, TAPE_FOR_LABEL, list_tapes

            return self._json({"default": DEFAULT_TAPE, "for_label": TAPE_FOR_LABEL, "tapes": list_tapes()})
        if path == "/api/saved-strategies":
            from eqpred.eval.saved_strategies import score_saved

            if query.get("drop"):
                return self._error(405, "Remove a saved card with a POST request from the lab page.")
            return self._json(score_saved())
        return self._error(404, "Unknown path.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._same_origin(post=True):
            self._error(403, "Changes must come from this lab's own page as JSON.")
            return
        if parsed.path == "/api/saved-strategies":
            self._guarded(self._saved_strategy_action)
            return
        trainer = _TRAINERS.get(parsed.path)
        if trainer is not None:
            self._guarded(lambda: self._json(trainer()))
            return
        self._error(404, "Unknown path.")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object.")
        return body

    def _saved_strategy_action(self) -> None:
        from eqpred.eval.saved_strategies import drop_card, freeze_from_request, score_saved

        body = self._read_json_body()
        if body.get("action") == "drop":
            drop_card(str(body.get("id") or ""))
        else:
            freeze_from_request(body)
        self._json(score_saved())

    def _host_allowed(self, value: str | None) -> bool:
        host, port = self.server.server_address[:2]
        if host in ("0.0.0.0", "::", ""):
            return True
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"{host}:{port}".lower()}
        return (value or "").strip().lower() in allowed

    def _same_origin(self, post: bool) -> bool:
        """Only this lab's own page may read the API or change anything (blocks other sites and DNS rebinding)."""
        if not self._host_allowed(self.headers.get("Host")):
            return False
        if not post:
            return True
        origin = self.headers.get("Origin")
        if origin and not self._host_allowed(urlparse(origin).netloc):
            return False
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        return content_type == "application/json"

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        packed = len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "").lower()
        if packed:
            body = gzip.compress(body, compresslevel=5)
        self._responded = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if packed:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
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
