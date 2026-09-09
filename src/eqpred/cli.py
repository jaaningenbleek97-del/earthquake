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

    ing = sub.add_parser("ingest", help="Wave 1, 2, or 3 connectors")
    ing.add_argument("--start", required=True)
    ing.add_argument("--end", required=True)
    ing.add_argument("--minmag", type=float, default=4.5)
    ing.add_argument("--wave", default="1", choices=["1", "2", "3", "all"])
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
    pred.add_argument("--horizon", type=int, default=7)

    bat = sub.add_parser("battery", help="Paper tests + 180-day solar time-shift null")
    bat.add_argument("--holdout-start", default="2019-01-01")

    sub.add_parser("score", help="Score issued cards against ComCat")

    human = sub.add_parser("human", help="Log a discretionary card (model_id=human); not used in fit")
    human.add_argument("--region", default="global")
    human.add_argument("--mag", type=float, default=7.0)
    human.add_argument("--horizon", type=int, default=7)
    human.add_argument("--kind", default="probability", choices=["probability", "alarm"])
    human.add_argument("--value", type=float, required=True)
    human.add_argument("--note", default="")
    human.add_argument("--issued-at", default=None, help="UTC issue morning; default is today")

    eepas = sub.add_parser("eepas", help="Paper-score the 90-day EEPAS foil vs ETAS; no issued p")
    eepas.add_argument("--train-end", default="2018-12-31")
    eepas.add_argument("--holdout-start", default="2019-01-01")

    stack = sub.add_parser("issued-stack", help="Paper-score issued fusion vs ETAS; no cards")
    stack.add_argument("--holdout-start", default="2019-01-01")
    stack.add_argument("--horizon", type=int, nargs="*", default=[1, 7, 90])

    watch = sub.add_parser("watchdog", help="Source freshness as-of a morning")
    watch.add_argument("--as-of", default=None, help="UTC timestamp; default is now")

    sub.add_parser("registry", help="Pin or list the issued model versions")

    hunt = sub.add_parser("hunt", help="Scan every feature knob for a next-week connection")
    hunt.add_argument("--train-end", default="2018-12-31")
    hunt.add_argument("--holdout-start", default="2019-01-01")
    hunt.add_argument(
        "--fresh",
        action="store_true",
        help="Hunt leftover knobs vs a declustered big-quake hit (no aftershock residual)",
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
        help="Leftover AI call for a fresh big quake (not aftershocks); backtest later years",
    )
    mainshock.add_argument("--train-end", default="2018-12-31")
    mainshock.add_argument("--holdout-start", default="2019-01-01")
    mainshock.add_argument(
        "--last-week",
        action="store_true",
        help="Only score leftover from the last 7 days of leftover sources (does not rewrite issued p)",
    )

    foreshock = sub.add_parser(
        "foreshock",
        help="Catalog precursor: foreshock → fresh mainshock; never leftover, never issued p",
    )
    foreshock.add_argument("--train-end", default="2018-12-31")
    foreshock.add_argument("--holdout-start", default="2019-01-01")

    probe = sub.add_parser(
        "probe",
        help="Timeshift the leftover M8 hint and test whether leftover can name the box",
    )
    probe.add_argument("--train-end", default="2018-12-31")
    probe.add_argument("--holdout-start", default="2019-01-01")

    xfer = sub.add_parser("transfer", help="Train on one arc, score the other (Japan ↔ Chile)")
    xfer.add_argument("--train-end", default="2018-12-31")
    xfer.add_argument("--holdout-start", default="2019-01-01")

    learn = sub.add_parser("learn", help="Brain + transfer + promote gate; never writes issued p")
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
    pipe.add_argument("--horizon", type=int, default=7)
    pipe.add_argument("--skip-ingest", action="store_true")

    lab = sub.add_parser("serve", help="Open the local lab UI (ledger, scoreboard, map)")
    lab.add_argument("--host", default="127.0.0.1")
    lab.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from eqpred.ingest.run import (
            WAVE1_SOURCES,
            WAVE2_SOURCES,
            WAVE3_SOURCES,
            ingest_wave1,
            ingest_wave2,
            ingest_wave3,
        )

        only = args.only
        known = set(WAVE1_SOURCES) | set(WAVE2_SOURCES) | set(WAVE3_SOURCES)
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

        report = run_hunt(train_end=args.train_end, holdout_start=args.holdout_start, fresh=args.fresh)
        print(
            json.dumps(
                {
                    "n_knobs": report["n_knobs"],
                    "n_tests": report["n_tests"],
                    "n_discovered": report["n_discovered"],
                    "n_confirmed": report["n_confirmed"],
                    "n_confirmed_unconventional": report["n_confirmed_unconventional"],
                    "family_counts": report["family_counts"],
                    "confirmed": report["confirmed"][:40],
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
        from eqpred.eval.mainshock import run_last_week_mainshock, run_mainshock

        if args.last_week:
            report = run_last_week_mainshock(train_end=args.train_end, holdout_start=args.holdout_start)
            last = report.get("last_week") or {}
            print(last.get("note") or report.get("note") or "Last-week leftover scored.")
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
                                "leftover_raw_p": (r.get("this_week") or {}).get("leftover_raw_p"),
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
                    "week": report["week"],
                    "regions": [
                        {
                            "label": r["label"],
                            "magnitude_min": r["magnitude_min"],
                            "brier": r.get("brier"),
                            "climatology_brier": r.get("climatology_brier"),
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
                            "leftover_acc",
                            "majority_acc",
                            "majority_label",
                            "beats_majority",
                            "n_hold",
                        )
                    },
                    "where_transfer": {
                        k: report["where_transfer"].get(k)
                        for k in (
                            "leftover_acc",
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
                    "n_features": report["n_features"],
                    "locked_lift": report["locked_lift"],
                    "locked_brier": report["locked_brier"],
                    "locked_etas_brier": report["locked_etas_brier"],
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
    main(["strategies", "--holdout-start", predict_start])
    main(["hunt", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["backtest", "--from-hunt", "--train-end", args.train_end, "--holdout-start", predict_start])
    main(["learn", "--train-end", args.train_end, "--holdout-start", predict_start])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
