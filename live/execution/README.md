# Phase C — Scheduled execution with continuous risk gates

This sub-folder is the engineering scaffold for Phase C of the
post-submission deployment roadmap (see `../ROADMAP.md`).

**Status: scaffold only.** `scheduled_executor.py` is a placeholder that
imports the trained models and exits cleanly with a "to be implemented"
log line. It does **not** place an order.

## What this phase does (when implemented)

A cron-driven (or systemd-timer-driven) executor that:

1. runs the advisor (Phase B logic) to produce a recommendation;
2. evaluates the recommendation against **non-negotiable pre-trade risk
   gates**;
3. places the corresponding orders on the **live** Alpaca account if and
   only if every gate passes;
4. records the full observation / prediction / action / fill audit trail
   under `runs/`.

Phase C runs on a small fraction of equity initially. Position sizing
scales up only after each successive month of operation continues to
satisfy the preservation-against-high-watermark and Sharpe gates.

## Pre-trade risk gates (non-negotiable)

| Gate | Default cap | Rationale |
|---|---|---|
| Position-size cap | `MAX_SINGLE_NAME_PCT` (e.g. 5% of equity) | Limits single-name exposure. |
| Gross-exposure cap | `MAX_GROSS_PCT` (e.g. 100% of equity) | Limits total leverage. |
| Sector-exposure cap | `MAX_SECTOR_PCT` (e.g. 30% of equity) | Limits sector concentration. |
| Daily-loss cap | `MAX_DAILY_LOSS_PCT` (e.g. -2%) | Halts trading after a bad day. |
| Drawdown circuit breaker | running 0.95 floor | Same as the dissertation's preservation-against-high-watermark gate. |

A breach of the drawdown circuit breaker **halts the executor**, requires
a written incident note in `runs/`, and forces a return to Phase B until
the cause is understood.

## Required environment variables

| Variable | Purpose |
|---|---|
| `ALPACA_KEY_ID` | Alpaca API key (live, trading scopes). |
| `ALPACA_SECRET_KEY` | Alpaca API secret (live, trading scopes). |
| `ALPACA_LIVE_BASE_URL` | Live trading endpoint. |
| `MAX_SINGLE_NAME_PCT` | Position-size cap, percent. |
| `MAX_GROSS_PCT` | Gross-exposure cap, percent. |
| `MAX_SECTOR_PCT` | Sector-exposure cap, percent. |
| `MAX_DAILY_LOSS_PCT` | Daily-loss cap, percent. |

## Hard rules

1. The executor refuses to start if any required environment variable is
   missing.
2. The executor refuses to start if the `live/KILL_SWITCH` file exists.
3. The executor refuses to start if the audit trail of the previous run
   is not closed (i.e. the previous run did not exit cleanly).
4. The executor halts and writes an incident note on any breach of the
   drawdown circuit breaker.
