from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone


def _ts(text: str) -> datetime:
    raw = datetime.fromisoformat(text)
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="eqpred", description="Unconventional earthquake prediction lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Wave 1–4 connectors")
    ing.add_argument("--start", required=True)
    ing.add_argument("--end", required=True)
    ing.add_argument("--minmag", type=float, default=4.5)
    ing.add_argument("--wave", default="1", choices=["1", "2", "3", "4", "all"])
    ing.add_argument(
        "--only",
        default=None,
        help="single source id, e.g. usgs.comcat or usgs.comcat.versions (merges into the latest parquet)",
    )
    ing.add_argument(
        "--issued-boxes",
        action="store_true",
        help="with usgs.comcat.versions, only Hikurangi / Alpine / Kermadec (M>=5.5 floor)",
    )

    feats = sub.add_parser("features", help="Build daily point-in-time features")
    feats.add_argument("--train-end", default="2018-12-31")

    fit = sub.add_parser("fit", help="Fit ETAS × modulator on the training era")
    fit.add_argument("--train-end", default="2018-12-31")

    pred = sub.add_parser("predict", help="Issue dated prediction cards")
    pred.add_argument("--start", required=True)
    pred.add_argument("--end", required=True)
    pred.add_argument("--horizon", type=int, default=30)

    bat = sub.add_parser("battery", help="Paper tests + 180-day solar time-shift null")
    bat.add_argument("--holdout-start", default="2019-01-01")

    sub.add_parser("score", help="Score issued cards against ComCat")

    human = sub.add_parser("human", help="Log a discretionary card (model_id=human); not used in fit")
    human.add_argument("--region", default="global")
    human.add_argument("--mag", type=float, default=7.0)
    human.add_argument("--horizon", type=int, default=30)
    human.add_argument("--kind", default="probability", choices=["probability", "alarm"])
    human.add_argument("--value", type=float, required=True)
    human.add_argument("--note", default="")
    human.add_argument("--issued-at", default=None, help="UTC issue morning; default is today")

    eepas = sub.add_parser("eepas", help="Paper-score the 90-day EEPAS foil vs ETAS; no issued p")
    eepas.add_argument("--train-end", default="2018-12-31")
    eepas.add_argument("--holdout-start", default="2019-01-01")
    widenet = sub.add_parser("widenet", help="Japan+Chile 7-day M5+ foil; no issued p")
    widenet.add_argument("--train-end", default="2018-12-31")
    widenet.add_argument("--holdout-start", default="2019-01-01")
    sub.add_parser("das-foil", help="Record local SISSLE Haast clips; no issued p")

    stack = sub.add_parser("issued-stack", help="Paper-score issued fusion vs ETAS; no cards")
    stack.add_argument("--holdout-start", default="2019-01-01")
    stack.add_argument("--horizon", type=int, nargs="*", default=[1, 7, 30, 90])

    watch = sub.add_parser("watchdog", help="Source freshness as-of a morning")
    watch.add_argument("--as-of", default=None, help="UTC timestamp; default is now")

    sources = sub.add_parser("sources", help="Coverage and freshness of every ingested source")
    sources.add_argument("--as-of", default=None, help="UTC timestamp; default is now")

    sub.add_parser("registry", help="Pin or list the issued model versions")

    hunt = sub.add_parser("hunt", help="Scan every feature knob for a next-week connection")
    hunt.add_argument("--train-end", default="2018-12-31")
    hunt.add_argument("--holdout-start", default="2019-01-01")
    hunt.add_argument(
        "--fresh",
        action="store_true",
        default=True,
        help="Hunt vs a first large quake after aftershock swarms are stripped (default)",
    )
    hunt.add_argument(
        "--etas-residual",
        action="store_true",
        help="Old hunt: non-catalog knobs vs what the aftershock model leaves behind",
    )
    hunt.add_argument(
        "--sequence",
        action="store_true",
        help="Hunt sequence/foreshock catalog columns vs a first large quake; never issued p",
    )

    back = sub.add_parser("backtest", help="Replay a knob on weeks that already closed")
    back.add_argument("--driver", default=None, help="feature column, e.g. f107")
    back.add_argument("--lag", type=int, default=0)
    back.add_argument("--region", default="global")
    back.add_argument("--share", type=float, default=0.2, help="fraction of train days that are 'on'")
    back.add_argument("--from-hunt", action="store_true", help="paper-trade hunt survivors")
    back.add_argument("--train-end", default="2018-12-31")
    back.add_argument("--holdout-start", default="2019-01-01")

    brain = sub.add_parser("brain", help="Residual learner on every unconventional source at once")
    brain.add_argument("--train-end", default="2018-12-31")
    brain.add_argument("--holdout-start", default="2019-01-01")

    mainshock = sub.add_parser(
        "mainshock",
        help="Fresh-shock model call for a new big quake (not aftershocks); backtest later years",
    )
    mainshock.add_argument("--train-end", default="2018-12-31")
    mainshock.add_argument("--holdout-start", default="2019-01-01")
    mainshock_mode = mainshock.add_mutually_exclusive_group()
    mainshock_mode.add_argument(
        "--last-week",
        action="store_true",
        help="Only score the last 7 days of non-catalog sources (does not rewrite issued p)",
    )
    mainshock_mode.add_argument(
        "--locked-thin",
        action="store_true",
        help="Paper-score the frozen pre-registered source list (does not rewrite issued p)",
    )

    after = sub.add_parser(
        "aftershock",
        help="Score the aftershock (ETAS) track on its own question; never issued p",
    )
    after.add_argument("--train-end", default="2018-12-31")
    after.add_argument("--holdout-start", default="2019-01-01")

    csep = sub.add_parser(
        "csep",
        help="CSEP N/L/T/S + Molchan tests on the issued fresh-shock daily p; never issued p",
    )
    csep.add_argument("--train-end", default="2018-12-31")
    csep.add_argument("--holdout-start", default="2019-01-01")

    foreshock = sub.add_parser(
        "foreshock",
        help="Catalog precursor: foreshock → fresh mainshock; never non-catalog, never issued p",
    )
    foreshock.add_argument("--train-end", default="2018-12-31")
    foreshock.add_argument("--holdout-start", default="2019-01-01")

    probe = sub.add_parser(
        "probe",
        help="Timeshift the M8 hint and test whether the model can name the box",
    )
    probe.add_argument("--train-end", default="2018-12-31")
    probe.add_argument("--holdout-start", default="2019-01-01")

    xfer = sub.add_parser("transfer", help="Train on one arc, score the other (Japan ↔ Chile)")
    xfer.add_argument("--train-end", default="2018-12-31")
    xfer.add_argument("--holdout-start", default="2019-01-01")

    leftover = sub.add_parser(
        "leftover",
        help="Climate + schedule + leftover of other sources; never writes issued p",
    )
    leftover.add_argument("--train-end", default="2018-12-31")
    leftover.add_argument("--holdout-start", default="2019-01-01")

    leftover_look = sub.add_parser(
        "leftover-look",
        help="Last 40 leftover-lookalike mornings and which-box-next; never issued p",
    )
    leftover_look.add_argument("--train-end", default="2018-12-31")
    leftover_look.add_argument("--holdout-start", default="2019-01-01")

    circus = sub.add_parser(
        "circus",
        help="Veto leftover if it tracks weekday/payday/HAARP/wiki foils; never issued p",
    )
    circus.add_argument("--train-end", default="2018-12-31")
    circus.add_argument("--holdout-start", default="2019-01-01")

    learn = sub.add_parser("learn", help="Brain + leftover + circus + transfer + promote; never writes issued p")
    learn.add_argument("--train-end", default="2018-12-31")
    learn.add_argument("--holdout-start", default="2019-01-01")

    strat = sub.add_parser("strategies", help="Walk-forward backtest of condition → box rules")
    strat.add_argument("--holdout-start", default="2019-01-01")

    pipe = sub.add_parser("pipeline", help="ingest → features → fit → predict → battery → score")
    pipe.add_argument("--start", required=True)
    pipe.add_argument("--end", required=True)
    pipe.add_argument("--minmag", type=float, default=4.5)
    pipe.add_argument("--train-end", default="2018-12-31")
    pipe.add_argument("--predict-start", default=None)
    pipe.add_argument("--horizon", type=int, default=30)
    pipe.add_argument("--skip-ingest", action="store_true")
    pipe.add_argument(
        "--hunt",
        action="store_true",
        help="Also run the weekly hunt scan and hunt-driven backtest",
    )

    analog = sub.add_parser(
        "analog",
        help="Earth-state anomaly, historical analogs, and fault-network stress; never issued p",
    )
    analog.add_argument("--train-end", default="2018-12-31")
    analog.add_argument("--holdout-start", default="2019-01-01")
    analog.add_argument("--no-write-features", action="store_true")
    anomaly = sub.add_parser("anomaly", help="Same as analog (Earth-state suite)")
    anomaly.add_argument("--train-end", default="2018-12-31")
    anomaly.add_argument("--holdout-start", default="2019-01-01")
    anomaly.add_argument("--no-write-features", action="store_true")
    graph = sub.add_parser("graph", help="Same as analog (Earth-state suite)")
    graph.add_argument("--train-end", default="2018-12-31")
    graph.add_argument("--holdout-start", default="2019-01-01")
    graph.add_argument("--no-write-features", action="store_true")

    lab = sub.add_parser("serve", help="Open the local lab UI (ledger, scoreboard, map)")
    lab.add_argument("--host", default="127.0.0.1")
    lab.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from eqpred.ingest.run import (
            WAVE1_SOURCES,
            WAVE2_SOURCES,
            WAVE3_SOURCES,
            WAVE4_SOURCES,
            ingest_wave1,
            ingest_wave2,
            ingest_wave3,
            ingest_wave4,
        )

        only = args.only
        known = set(WAVE1_SOURCES) | set(WAVE2_SOURCES) | set(WAVE3_SOURCES) | set(WAVE4_SOURCES)
        if only and only not in known:
            print(f"unknown source {only}; known: {', '.join(sorted(known))}")
            return 2
        written = {}
        if args.wave in {"1", "all"} and (only is None or only in WAVE1_SOURCES):
            written.update(
                ingest_wave1(
                    _ts(args.start),
                    _ts(args.end),
                    minmag=args.minmag,
                    only=only,
                    issued_boxes=args.issued_boxes,
                )
            )
        if args.wave in {"2", "all"} and (only is None or only in WAVE2_SOURCES):
            written.update(ingest_wave2(_ts(args.start), _ts(args.end), only=only))
        if args.wave in {"3", "all"} and (only is None or only in WAVE3_SOURCES):
            written.update(ingest_wave3(_ts(args.start), _ts(args.end), only=only))
        if args.wave in {"4", "all"} and (only is None or only in WAVE4_SOURCES):
            written.update(ingest_wave4(_ts(args.start), _ts(args.end), only=only))
        if only and not written:
            print(f"ingest wrote nothing for {only}")
            return 2
        print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
        return 0

    if args.cmd == "features":
        from eqpred.features.build import build_features

        frame = build_features(train_end=args.train_end)
        print(f"features {len(frame)} days -> data/features/global_daily.parquet")
        return 0

    if args.cmd == "fit":
        from eqpred.models.fit import fit_all

        etas, mod, quiet_cut, b, scale = fit_all(train_end=args.train_end)
        print("etas", etas)
        print("gr b", round(b, 4))
        print("modulator beta", dict(zip(["intercept", *mod.columns], mod.beta.round(4))))
        print("quiet-regime lambda_etas <=", round(quiet_cut, 4))
        print("issued ETAS scale", round(scale, 4), "(solar g unused on global p)")
        from eqpred.models.nowcast import load_nowcast

        _now, now_meta = load_nowcast()
        print("catalog nowcast", "on issued p" if now_meta.get("used_on_issued") else "held out", now_meta)
        from eqpred.models.coulomb import load_coulomb

        tide = load_coulomb()
        issued = [rid for rid, meta in (tide.get("regions") or {}).items() if meta.get("used_on_issued")]
        print("box coulomb", issued or "held out", tide.get("regions") or {})
        from eqpred.models.cusp import load_cusp

        cusp = load_cusp()
        cusp_on = [rid for rid, meta in (cusp.get("regions") or {}).items() if meta.get("used_on_issued")]
        print("box cusp", cusp_on or "held out", cusp.get("regions") or {})
        from eqpred.models.wave2 import load_wave2

        wave2 = load_wave2()
        wave2_on = {
            rid: [c.get("column") for c in (meta.get("columns") or []) if c.get("used_on_issued")]
            for rid, meta in (wave2.get("regions") or {}).items()
            if meta.get("used_on_issued")
        }
        print("wave2 residual", wave2_on or "held out")
        from eqpred.models.migration import load_migration

        mig = load_migration()
        mig_on = {
            rid: meta.get("column")
            for rid, meta in (mig.get("regions") or {}).items()
            if meta.get("used_on_issued")
        }
        print("box migration", mig_on or "held out")
        from eqpred.models.anti import load_anti

        anti = load_anti()
        print(
            "anti-prediction p_max",
            {rid: (meta.get("p_max"),) for rid, meta in (anti.get("regions") or {}).items()}
            or "unfit",
        )
        from eqpred.models.budget import load_budget

        print("alarm budget / month", load_budget().get("days_per_month") or {})
        from eqpred.models.registry import load_current

        pinned = load_current()
        print("registry", pinned["id"] if pinned else "unpinned")
        return 0

    if args.cmd == "predict":
        from eqpred.ingest.watchdog import StaleSourcesError
        from eqpred.models.predict import issue_predictions

        try:
            cards = issue_predictions(_ts(args.start), _ts(args.end), horizon_days=args.horizon)
        except StaleSourcesError as exc:
            print(json.dumps(exc.report, indent=2, default=str))
            print("refused: required sources stale; existing cards left on disk")
            return 2
        except RuntimeError as exc:
            if "existing cards left on disk" in str(exc):
                print(str(exc))
                return 2
            raise
        from collections import Counter

        heads = dict(Counter(card["model_id"] for card in cards))
        print(f"wrote {len(cards)} prediction records", heads)
        return 0

    if args.cmd == "registry":
        from eqpred.models.registry import ensure_registered, pin_unpinned_batches, registry_report

        entry = ensure_registered()
        pinned = pin_unpinned_batches(entry=entry)
        report = registry_report()
        report["pinned_batches"] = pinned
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "watchdog":
        from eqpred.ingest.watchdog import freshness_report

        as_of = _ts(args.as_of) if args.as_of else datetime.now(timezone.utc)
        report = freshness_report(as_of)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2

    if args.cmd == "sources":
        from eqpred.ingest.watchdog import coverage_report

        as_of = _ts(args.as_of) if args.as_of else datetime.now(timezone.utc)
        report = coverage_report(as_of)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] and not report["missing"] else 2

    if args.cmd == "battery":
        from eqpred.eval.battery import run_battery

        reports = run_battery(holdout_start=args.holdout_start)
        print(json.dumps(reports, indent=2, default=str))
        return 0

    if args.cmd == "score":
        from eqpred.eval.score import score_predictions, summarize_ledger

        ledger = score_predictions()
        print(json.dumps(summarize_ledger(ledger), indent=2))
        return 0

    if args.cmd == "human":
        from eqpred.models.human import write_human_card

        card = write_human_card(
            region_id=args.region,
            magnitude_min=args.mag,
            horizon_days=args.horizon,
            kind=args.kind,
            value=args.value,
            note=args.note,
            issued_at=args.issued_at,
        )
        print(json.dumps(card, indent=2, default=str))
        return 0

    if args.cmd == "eepas":
        from eqpred.eval.eepas import run_eepas

        report = run_eepas(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report.get("note", ""))
        print(json.dumps({k: report[k] for k in report if k != "regions"}, indent=2, default=str))
        return 0

    if args.cmd == "widenet":
        from eqpred.eval.widenet import run_widenet

        report = run_widenet(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report.get("note", ""))
        print(json.dumps({k: report[k] for k in report if k != "clips"}, indent=2, default=str))
        return 0

    if args.cmd == "das-foil":
        from eqpred.eval.das_foil import run_das_foil

        report = run_das_foil()
        print(report.get("note", ""))
        print(json.dumps({k: report[k] for k in report if k != "clips"}, indent=2, default=str))
        return 0

    if args.cmd == "issued-stack":
        from eqpred.eval.issued_stack import run_issued_stacks

        packed = run_issued_stacks(holdout_start=args.holdout_start, horizons=tuple(args.horizon))
        for report in packed["reports"]:
            print(report.get("note", ""))
            print(
                json.dumps(
                    {k: report[k] for k in report if k not in {"regions", "path"}},
                    indent=2,
                    default=str,
                )
            )
        return 0

    if args.cmd == "hunt":
        from eqpred.eval.hunt import run_hunt

        report = run_hunt(
            train_end=args.train_end,
            holdout_start=args.holdout_start,
            fresh=not args.etas_residual,
            sequence=args.sequence,
        )
        print(
            json.dumps(
                {
                    "n_knobs": report["n_knobs"],
                    "n_tests": report["n_tests"],
                    "n_discovered": report["n_discovered"],
                    "n_confirmed": report["n_confirmed"],
                    "n_confirmed_unconventional": report["n_confirmed_unconventional"],
                    "n_confirmed_sequence": report.get("n_confirmed_sequence"),
                    "sequence": report.get("sequence"),
                    "writes_issued_p": report.get("writes_issued_p"),
                    "looks_at_all_sources": report.get("looks_at_all_sources"),
                    "family_counts": report["family_counts"],
                    "confirmed": report["confirmed"][:40],
                    "honest": (
                        {
                            "n_knobs": report["honest"]["n_knobs"],
                            "n_tests": report["honest"]["n_tests"],
                            "n_confirmed": report["honest"]["n_confirmed"],
                            "n_beats_climate": report["honest"]["n_beats_climate"],
                            "n_confirmed_family": report["honest"]["n_confirmed_family"],
                            "n_confirmed_quiet": report["honest"]["n_confirmed_quiet"],
                            "n_confirmed_m8": report["honest"]["n_confirmed_m8"],
                            "writes_issued_p": report["honest"]["writes_issued_p"],
                        }
                        if report.get("honest")
                        else None
                    ),
                    "note": report["note"],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "backtest":
        from eqpred.eval.backtest import pack_from_hunt, paper_trade

        if args.from_hunt:
            report = pack_from_hunt(
                share=args.share,
                train_end=args.train_end,
                holdout_start=args.holdout_start,
            )
            print(json.dumps(report, indent=2, default=str))
            return 0
        if not args.driver:
            print("backtest needs --driver or --from-hunt")
            return 2
        trade = paper_trade(
            args.driver,
            lag_days=args.lag,
            region_id=args.region,
            share=args.share,
            train_end=args.train_end,
            holdout_start=args.holdout_start,
        )
        print(json.dumps(trade, indent=2, default=str))
        return 0

    if args.cmd == "mainshock":
        from eqpred.eval.mainshock import (
            run_last_week_mainshock,
            run_locked_thin_mainshock,
            run_mainshock,
        )

        if args.locked_thin:
            report = run_locked_thin_mainshock(train_end=args.train_end, holdout_start=args.holdout_start)
            locked = report.get("locked_thin") or {}
            print(locked.get("note") or report.get("note") or "Locked thin model scored.")
            print(
                json.dumps(
                    {
                        "writes_issued_p": False,
                        "mode": "locked_thin",
                        "n_features": locked.get("n_features"),
                        "n_locked": locked.get("n_locked"),
                        "n_beats_climatology": locked.get("n_beats_climatology"),
                        "week": locked.get("week"),
                        "regions": [
                            {
                                "label": r["label"],
                                "magnitude_min": r["magnitude_min"],
                                "brier": r.get("brier"),
                                "climatology_brier": r.get("climatology_brier"),
                                "brier_p_holm": r.get("brier_p_holm"),
                                "recipe": r.get("recipe"),
                                "beats_climatology": r.get("beats_climatology"),
                                "fresh_raw_p": (r.get("this_week") or {}).get("fresh_raw_p"),
                            }
                            for r in locked.get("regions") or []
                        ],
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0

        if args.last_week:
            report = run_last_week_mainshock(train_end=args.train_end, holdout_start=args.holdout_start)
            last = report.get("last_week") or {}
            print(last.get("note") or report.get("note") or "Last-week model scored.")
            print(
                json.dumps(
                    {
                        "writes_issued_p": False,
                        "mode": "last_week",
                        "n_features": last.get("n_features"),
                        "n_beats_climatology": last.get("n_beats_climatology"),
                        "week": last.get("week"),
                        "regions": [
                            {
                                "label": r["label"],
                                "magnitude_min": r["magnitude_min"],
                                "brier": r.get("brier"),
                                "climatology_brier": r.get("climatology_brier"),
                                "recipe": r.get("recipe"),
                                "fresh_raw_p": (r.get("this_week") or {}).get("fresh_raw_p"),
                            }
                            for r in last.get("regions") or []
                        ],
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0

        report = run_mainshock(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "n_features": report["n_features"],
                    "n_beats_climatology": report["n_beats_climatology"],
                    "score_kind": report.get("score_kind"),
                    "evaluation": report.get("evaluation"),
                    "experiments": [
                        {
                            "id": exp["id"],
                            "n_beats_climatology": exp["n_beats_climatology"],
                            "regions": [
                                {
                                    "label": r["label"],
                                    "brier": r.get("brier"),
                                    "climatology_brier": r.get("climatology_brier"),
                                    "recipe": r.get("recipe"),
                                }
                                for r in exp.get("regions") or []
                            ],
                        }
                        for exp in report.get("experiments") or []
                    ],
                    "shrunk": {
                        "n_features": (report.get("shrunk") or {}).get("n_features"),
                        "n_beats_climatology": (report.get("shrunk") or {}).get("n_beats_climatology"),
                        "writes_issued_p": False,
                    },
                    "alpine_sequence": {
                        "n_features": (report.get("alpine_sequence") or {}).get("n_features"),
                        "applied": (report.get("alpine_sequence") or {}).get("applied"),
                        "writes_issued_p": (report.get("alpine_sequence") or {}).get("writes_issued_p"),
                    },
                    "method_eval": {
                        "n_beats_climatology": (report.get("method_eval") or {}).get("n_beats_climatology"),
                        "writes_issued_p": False,
                    },
                    "kermadec_attribution": {
                        "verdict": (report.get("kermadec_attribution") or {}).get("verdict"),
                        "writes_issued_p": False,
                    },
                    "week": report["week"],
                    "regions": [
                        {
                            "label": r["label"],
                            "magnitude_min": r["magnitude_min"],
                            "brier": r.get("brier"),
                            "climatology_brier": r.get("climatology_brier"),
                            "brier_p_value": r.get("brier_p_value"),
                            "brier_p_holm": r.get("brier_p_holm"),
                            "brier_week": r.get("brier_week"),
                            "issued_from": r.get("issued_from"),
                            "n_score": r.get("n_score"),
                            "lift": r.get("lift"),
                            "fresh_rate_holdout": r.get("fresh_rate_holdout"),
                            "raw_rate_holdout": r.get("raw_rate_holdout"),
                        }
                        for r in report["regions"]
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "aftershock":
        from eqpred.eval.aftershock import run_aftershock

        report = run_aftershock(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": False,
                    "track": "aftershock_etas",
                    "issue_day": report.get("issue_day"),
                    "n_beats_climatology": report.get("n_beats_climatology"),
                    "week": [
                        {
                            "region": w["region"],
                            "chance_plain": w.get("chance_plain"),
                            "fresh_chance_plain": w.get("fresh_chance_plain"),
                        }
                        for w in report.get("week") or []
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "csep":
        from eqpred.eval.csep import load_csep, run_csep

        existing = load_csep()
        if existing and existing.get("boxes"):
            report = existing
        else:
            report = run_csep(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report.get("note") or "")
        print(
            json.dumps(
                {
                    "writes_issued_p": False,
                    "horizon_days": report.get("horizon_days"),
                    "n_pass": report.get("n_pass"),
                    "n_fail": report.get("n_fail"),
                    "spatial": (report.get("spatial") or {}).get("plain"),
                    "boxes": [
                        {
                            "region": b.get("region"),
                            "n_pass": b.get("n_pass"),
                            "n_fail": b.get("n_fail"),
                        }
                        for b in report.get("boxes") or []
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "foreshock":
        from eqpred.eval.foreshock import run_foreshock

        report = run_foreshock(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": False,
                    "track": "catalog_precursor",
                    "n_features": report.get("n_features"),
                    "n_beats_climatology": report.get("n_beats_climatology"),
                    "week": report.get("week"),
                    "regions": [
                        {
                            "label": r["label"],
                            "magnitude_min": r["magnitude_min"],
                            "brier": r.get("brier"),
                            "climatology_brier": r.get("climatology_brier"),
                            "recipe": r.get("recipe"),
                            "foreshock_raw_p": (r.get("this_week") or {}).get("foreshock_raw_p"),
                        }
                        for r in report.get("regions") or []
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "probe":
        from eqpred.eval.probe import run_probe

        report = run_probe(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "n_features": report["n_features"],
                    "m8": {
                        "looks_like_leakage": report["m8"].get("looks_like_leakage"),
                        "aligned_lift": report["m8"].get("aligned_lift"),
                        "shifts": [
                            {
                                "shift_days": row.get("shift_days"),
                                "lift": row.get("lift"),
                                "brier": row.get("brier"),
                                "n_hits": row.get("n_hits"),
                                "n_trades": row.get("n_trades"),
                            }
                            for row in report["m8"].get("shifts") or []
                        ],
                    },
                    "where_issued": {
                        k: report["where_issued"].get(k)
                        for k in (
                            "fresh_acc",
                            "majority_acc",
                            "majority_label",
                            "beats_majority",
                            "n_hold",
                        )
                    },
                    "where_transfer": {
                        k: report["where_transfer"].get(k)
                        for k in (
                            "fresh_acc",
                            "majority_acc",
                            "majority_label",
                            "beats_majority",
                            "n_hold",
                        )
                    },
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "brain":
        from eqpred.eval.brain import run_brain

        report = run_brain(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "looks_at_all_sources": report.get("looks_at_all_sources"),
                    "n_features": report["n_features"],
                    "locked_lift": report["locked_lift"],
                    "locked_brier": report["locked_brier"],
                    "locked_etas_brier": report["locked_etas_brier"],
                    "locked_climate_brier": report.get("locked_climate_brier"),
                    "locked_beats_climate": report.get("locked_beats_climate"),
                    "recent": {
                        "n_features": (report.get("recent") or {}).get("n_features"),
                        "locked_brier": (report.get("recent") or {}).get("locked_brier"),
                        "locked_climate_brier": (report.get("recent") or {}).get("locked_climate_brier"),
                    },
                    "quiet_start": {
                        "locked_lift": (report.get("quiet_start") or {}).get("locked_lift"),
                        "locked_brier": (report.get("quiet_start") or {}).get("locked_brier"),
                    },
                    "rare": {
                        "locked_brier": (report.get("rare") or {}).get("locked_brier"),
                        "locked_climate_brier": (report.get("rare") or {}).get("locked_climate_brier"),
                    },
                    "improved": report["improved"],
                    "walk_forward": {
                        "n_folds": report["walk_forward"]["n_folds"],
                        "mean_lift": report["walk_forward"]["mean_lift"],
                    },
                    "connections": report["connections"][:12],
                    "experiments": report["experiments"],
                    "journal_runs": report["journal_runs"],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "leftover":
        from eqpred.eval.leftover import run_leftover

        report = run_leftover(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "n_leftover_columns": report.get("n_leftover_columns"),
                    "n_beats_climate": report.get("n_beats_climate"),
                    "n_beats_schedule": report.get("n_beats_schedule"),
                    "attached_to_mainshock": report.get("attached_to_mainshock"),
                    "week": [
                        {
                            "label": w.get("label"),
                            "leftover_chance_plain": w.get("leftover_chance_plain"),
                            "schedule_chance_plain": w.get("schedule_chance_plain"),
                            "climatology_chance_plain": w.get("climatology_chance_plain"),
                        }
                        for w in report.get("week") or []
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "leftover-look":
        from eqpred.eval.leftover_look import run_leftover_look

        report = run_leftover_look(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report.get("sentence") or report.get("note"))
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "n_analogs": report.get("n_analogs"),
                    "which_box": report.get("which_box"),
                    "which_box_label": report.get("which_box_label"),
                    "which_box_rate": report.get("which_box_rate"),
                    "box_rates": report.get("box_rates"),
                    "issue_day": report.get("issue_day"),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "circus":
        from eqpred.eval.circus import run_circus

        report = run_circus(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "pass": report.get("pass"),
                    "n_fail": report.get("n_fail"),
                    "foils": report.get("foils"),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "transfer":
        from eqpred.eval.transfer import run_transfer

        report = run_transfer(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(json.dumps({k: report[k] for k in ("writes_issued_p", "n_pairs", "n_beats_etas", "pairs") if k in report}, indent=2, default=str))
        return 0

    if args.cmd == "learn":
        from eqpred.eval.learn import run_learn

        report = run_learn(train_end=args.train_end, holdout_start=args.holdout_start)
        print(report["note"])
        print(
            json.dumps(
                {
                    "writes_issued_p": report["writes_issued_p"],
                    "promote": report["promote"],
                    "transfer_beats": report["transfer"].get("n_beats_etas"),
                    "brain_lift": report["brain"].get("locked_lift"),
                    "leftover_beats_schedule": (report.get("leftover") or {}).get("n_beats_schedule"),
                    "circus_pass": (report.get("circus") or {}).get("pass"),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "strategies":
        from eqpred.eval.strategies import run_strategies

        reports = run_strategies(holdout_start=args.holdout_start)
        print(json.dumps(reports, indent=2, default=str))
        return 0

    if args.cmd == "pipeline":
        return _pipeline(args)

    if args.cmd in {"analog", "anomaly", "graph"}:
        from eqpred.eval.earth_state import run_earth_state

        report = run_earth_state(
            train_end=args.train_end,
            holdout_start=args.holdout_start,
            write_features=not args.no_write_features,
        )
        print(report.get("note") or "")
        print(
            json.dumps(
                {
                    "writes_issued_p": False,
                    "issue_day": report.get("issue_day"),
                    "n_state_columns": report.get("n_state_columns"),
                    "n_beats_climatology": report.get("n_beats_climatology"),
                    "anomaly": report.get("anomaly"),
                    "n_analogs": len(report.get("analogs") or []),
                    "graph_transfer_on": (report.get("graph") or {}).get("transfer_on"),
                    "regions": [
                        {
                            "label": r.get("label"),
                            "analog_brier": r.get("analog_brier"),
                            "climatology_brier": r.get("climatology_brier"),
                            "beats_climatology": r.get("beats_climatology"),
                            "this_week_analog_p": r.get("this_week_analog_p"),
                        }
                        for r in report.get("regions") or []
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    if args.cmd == "serve":
        from eqpred.ui.server import serve

        serve(host=args.host, port=args.port)
        return 0

    return 1


def _pipeline(args: argparse.Namespace) -> int:
    predict_start = args.predict_start
    if predict_start is None:
        predict_start = (datetime.fromisoformat(args.train_end) + timedelta(days=1)).date().isoformat()

    if not args.skip_ingest:
        main(["ingest", "--start", args.start, "--end", args.end, "--minmag", str(args.minmag)])
    main(["features", "--train-end", args.train_end])
    main(["fit", "--train-end", args.train_end])
    predict_rc = main(
        ["predict", "--start", predict_start, "--end", args.end, "--horizon", str(args.horizon)]
    )
    if predict_rc:
        return predict_rc
    main(["battery", "--holdout-start", predict_start])
    main(["score"])
    main(["mainshock", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["csep", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["aftershock", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["eepas", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["widenet", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["das-foil"])
    main(["strategies", "--holdout-start", predict_start])
    if args.hunt:
        main(["hunt", "--train-end", args.train_end, "--holdout-start", predict_start])
        main(["backtest", "--from-hunt", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["learn", "--train-end", args.train_end, "--holdout-start", predict_start])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
