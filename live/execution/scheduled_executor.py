"""Phase C — scheduled executor with continuous risk gates.

This is a SCAFFOLD only. It does not place an order.

The implementation plan is documented in ``live/ROADMAP.md`` (Phase C).
The hard rules in ``live/README.md`` apply: no secrets in the repo, kill
switch via the ``live/KILL_SWITCH`` file, full audit trail under
``live/execution/runs/<tag>/``.

When this script is filled in it will, on every scheduled run:
    1. invoke the Phase B advisor to produce a recommendation;
    2. evaluate the recommendation against the non-negotiable pre-trade
       risk gates listed in ``README.md``;
    3. place the corresponding orders on the live account via the Alpaca
       trading API if and only if every gate passes;
    4. record the full audit trail under ``runs/<tag>/<YYYY-MM-DD>.jsonl``;
    5. halt and write an incident note on any breach of the drawdown
       circuit breaker.

This file exists so that the path is reserved, the imports exist, and the
dissertation's Section 7.2 reference is real.
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

REQUIRED_ENV = (
    "ALPACA_KEY_ID",
    "ALPACA_SECRET_KEY",
    "ALPACA_LIVE_BASE_URL",
    "MAX_SINGLE_NAME_PCT",
    "MAX_GROSS_PCT",
    "MAX_SECTOR_PCT",
    "MAX_DAILY_LOSS_PCT",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase C — scheduled executor with continuous risk gates. "
            "Currently a scaffold that exits cleanly without trading."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="phaseC-executor",
        help="Run tag, appended to the JSONL audit-trail directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Evaluate the risk gates and write the would-be orders to disk "
            "without placing them. Always implied while this module is a scaffold."
        ),
    )
    return parser


def _check_kill_switch() -> bool:
    if KILL_SWITCH.exists():
        logging.warning("KILL_SWITCH present at %s — refusing to start.", KILL_SWITCH)
        return True
    return False


def _check_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [executor] %(levelname)s %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)

    logging.info("Phase C scheduled executor starting (tag=%s).", args.tag)
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
        "This script is a scaffold for Phase C of the live deployment roadmap. "
        "It will NOT place an order until each pre-trade risk gate is implemented "
        "and the Phase B → Phase C go / no-go criterion has been cleared."
    )
    logging.info("Exiting without trading. Implementation tracked under Phase C.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
