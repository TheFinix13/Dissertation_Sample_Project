"""Phase B — nightly decision-support advisor against the real portfolio.

This is a SCAFFOLD only. It does not read the real account.

The implementation plan is documented in ``live/ROADMAP.md`` (Phase B).
The hard rules in ``live/README.md`` apply: read-only credentials, kill
switch via the ``live/KILL_SWITCH`` file, full audit trail under
``live/decision_support/runs/<tag>/``.

When this script is filled in it will, every trading day:
    1. pull current positions and prices from the live Alpaca account
       (read-only);
    2. run the trained probabilistic agent on the day's observation;
    3. emit a recommendation by email or Slack;
    4. record the recommendation alongside the human's eventual action in
       a side-by-side ledger for P&L attribution.

It must NEVER place an order. The human is the executor.
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
    "ADVISOR_NOTIFY_TARGET",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase B — nightly decision-support advisor. "
            "Currently a scaffold that exits cleanly without reading the account."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="phaseB-advisor",
        help="Run tag, appended to the JSONL audit-trail directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the notification step and write the recommendation only to disk. "
            "Always implied while this module is a scaffold."
        ),
    )
    return parser


def _check_kill_switch() -> bool:
    if KILL_SWITCH.exists():
        logging.warning("KILL_SWITCH present at %s — exiting without recommending.", KILL_SWITCH)
        return True
    return False


def _check_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [advisor] %(levelname)s %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)

    logging.info("Phase B nightly advisor starting (tag=%s).", args.tag)
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
        "This script is a scaffold for Phase B of the live deployment roadmap. "
        "It will NEVER place an order — the human is the executor."
    )
    logging.info("Exiting without recommending. Implementation tracked under Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
