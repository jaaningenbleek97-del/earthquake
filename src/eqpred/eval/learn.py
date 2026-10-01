from __future__ import annotations

"""Nightly learn: residual brain, leftover extra, circus, transfer, promote. Never writes issued p."""

from pathlib import Path
from typing import Any

from eqpred.eval.brain import run_brain
from eqpred.eval.circus import run_circus
from eqpred.eval.leftover import run_leftover
from eqpred.eval.leftover_look import run_leftover_look
from eqpred.eval.promote import run_promote
from eqpred.eval.transfer import run_transfer


def run_learn(
    train_end: str = "2018-12-31",
    holdout_start: str = "2019-01-01",
    root: Path | None = None,
    features=None,
    quakes=None,
) -> dict[str, Any]:
    brain = run_brain(
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
        features=features,
        quakes=quakes,
    )
    leftover = run_leftover(
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
        features=features,
        quakes=quakes,
    )
    leftover_look = run_leftover_look(
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
        features=features,
        quakes=quakes,
    )
    circus = run_circus(
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
        features=features,
        quakes=quakes,
        leftover=leftover,
        leftover_hats=leftover.get("_hats"),
    )
    transfer = run_transfer(
        train_end=train_end,
        holdout_start=holdout_start,
        root=root,
        features=features,
        quakes=quakes,
    )
    promote = run_promote(root=root, brain=brain)
    return {
        "writes_issued_p": False,
        "brain": brain,
        "leftover": leftover,
        "leftover_look": leftover_look,
        "circus": circus,
        "transfer": transfer,
        "promote": promote,
        "note": " ".join(
            part
            for part in (
                brain.get("note"),
                leftover.get("note"),
                leftover_look.get("note"),
                circus.get("note"),
                transfer.get("note"),
                promote.get("note"),
            )
            if part
        ),
    }
