# `live/` — Real-world deployment roadmap

This folder is **not part of the EEEM004 dissertation submission**. It is the
post-submission engineering scaffold for taking the trained probabilistic agents
from a backtest into a real, live portfolio.

The dissertation answers a *scientific* question:

> *Does feeding a deep RL policy its own forecaster's predictive uncertainty
> sit on a more attractive point of the return-versus-drawdown trade-off than
> uncertainty-blind alternatives, on a held-out test window with real macro
> shocks?*

This folder begins to answer a related but distinct *engineering* question:

> *Can the resulting agents be made useful, safely, against a real portfolio?*

The answer is staged across three phases (A → B → C), each gated by an
explicit go / no-go criterion taken from the dissertation's
preservation-against-high-watermark and Sharpe metrics. The phases are
described in [`ROADMAP.md`](./ROADMAP.md). Section 7.2 of the main
dissertation references this folder.

## Layout

```
live/
├── README.md                       # This file.
├── ROADMAP.md                      # Phase A / B / C checklist with go-no-go gates.
├── paper_trading/                  # Phase A — shadow paper-trading via Alpaca.
│   ├── README.md
│   └── alpaca_paper_loop.py        # Stub: daily polling loop.
├── decision_support/               # Phase B — recommendations to a human executor.
│   ├── README.md
│   └── nightly_advisor.py          # Stub: nightly email/Slack advisor.
└── execution/                      # Phase C — scheduled execution with risk gates.
    ├── README.md
    └── scheduled_executor.py       # Stub: cron-driven executor.
```

## Status

This folder currently holds **scaffolding only**. None of the scripts make a
real trading decision. They are placeholders that import the trained models
and the `_close_1d` data helper, and exit cleanly with a "to be implemented"
log line. Each phase will be populated incrementally after the September 2026
submission.

## Hard rules (apply to every phase)

1. **No real-capital trading from this folder until Phase C.** Phases A and B
   either run against a paper account or run as decision-support only.
2. **No secrets in the repository.** All API keys, account IDs and trading
   endpoints must be read from environment variables (e.g. `ALPACA_KEY_ID`,
   `ALPACA_SECRET_KEY`, `ALPACA_PAPER_BASE_URL`).
3. **Hard kill-switch.** Every script must check for the presence of a
   `live/KILL_SWITCH` file at the start of every loop iteration and exit
   cleanly if it exists.
4. **Pre-trade risk gates.** Before any order leaves the script (Phase C
   only): position-size cap, gross-exposure cap, sector-exposure cap,
   daily-loss cap, drawdown-against-high-watermark circuit breaker.
5. **Full audit trail.** Every observation, predicted (mean, std) pair, raw
   action, scaled action and resulting fill is logged on a per-day,
   per-ticker basis to a versioned JSONL trail under `live/<phase>/runs/`,
   so that any month of live operation can be replayed deterministically
   against the backtest.

## Why this is in `main` and not on a separate `live` git branch

The dissertation cites this folder from Section 7.2. The folder needs to live
on `main` so that anyone who clones the repository and reads the dissertation
can immediately see the roadmap. The implementation work for each phase is
expected to land on its own feature branch (e.g. `live/phase-a-paper-trading`)
and be merged back to `main` once each phase's go / no-go gate is cleared.
