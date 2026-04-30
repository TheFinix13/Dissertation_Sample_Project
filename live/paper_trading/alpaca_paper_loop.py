"""Phase A — shadow paper-trading loop via Alpaca.

This is a SCAFFOLD only. It does not make a real trading decision.

The implementation plan is documented in ``live/ROADMAP.md`` (Phase A).
The hard rules in ``live/README.md`` apply: no secrets in the repo, kill
switch via the ``live/KILL_SWITCH`` file, full audit trail under
``live/paper_trading/runs/<tag>/``.

When this script is filled in it will, on every trading day:
    1. read the trained 70-ticker probabilistic policy and its DeepAR-style
       forecaster from ``trained_models/``;
    2. pull the day's observation for each ticker from Alpaca's market-data
       API;
    3. query the policy and the forecaster to produce a
       ``(mean, std, raw action, scaled action)`` tuple per ticker;
    4. submit the corresponding paper-account order via Alpaca's paper
       trading endpoint;
    5. log the full trail to ``runs/<tag>/<YYYY-MM-DD>.jsonl``.

Until then this file exists so that the path is reserved, the imports
exist, and the dissertation's Section 7.2 reference is real.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LIVE_ROOT = Path(__file__).resolve().parent.parent
KILL_SWITCH = LIVE_ROOT / "KILL_SWITCH"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

REQUIRED_ENV = ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_PAPER_BASE_URL")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A — shadow paper-trading loop via Alpaca. "
            "Currently a scaffold that exits cleanly without trading."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="phaseA-shadow",
        help="Run tag, appended to the JSONL audit-trail directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a single iteration of the loop without contacting Alpaca. "
            "Always implied while this module is a scaffold."
        ),
    )
    return parser


def _check_kill_switch() -> bool:
    if KILL_SWITCH.exists():
        logging.warning("KILL_SWITCH present at %s — exiting without trading.", KILL_SWITCH)
        return True
    return False


def _check_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [paper-trading] %(levelname)s %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)

    logging.info("Phase A shadow paper-trading loop starting (tag=%s).", args.tag)
    if _check_kill_switch():
        return 0

    missing = _check_env()
    if missing:
        logging.warning(
            "Missing required environment variables: %s. The scaffold will exit cleanly.",
            ", ".join(missing),
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(
        "This script is a scaffold for Phase A of the live deployment roadmap. "
        "See live/ROADMAP.md for the checklist."
    )
    logging.info("Exiting without trading. Implementation tracked under Phase A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
