# Phase A — Shadow paper-trading via Alpaca

This sub-folder is the engineering scaffold for Phase A of the
post-submission deployment roadmap (see `../ROADMAP.md`).

**Status: scaffold only.** `alpaca_paper_loop.py` is a placeholder that
imports the trained models and exits cleanly with a "to be implemented"
log line. It does **not** make a real trading decision.

## What this phase does (when implemented)

A daily polling loop that:

1. reads the trained 70-ticker probabilistic policy and its DeepAR-style
   forecaster from `trained_models/`;
2. pulls the day's observation for each ticker from Alpaca's market-data
   API;
3. queries the policy and the forecaster, producing a `(mean, std, raw
   action, scaled action)` tuple per ticker;
4. submits the corresponding paper-account order via Alpaca's paper trading
   endpoint;
5. records everything to a versioned JSONL audit trail under `runs/`.

## Required environment variables

| Variable | Purpose |
|---|---|
| `ALPACA_KEY_ID` | Alpaca API key (paper). |
| `ALPACA_SECRET_KEY` | Alpaca API secret (paper). |
| `ALPACA_PAPER_BASE_URL` | Paper trading endpoint, e.g. `https://paper-api.alpaca.markets`. |
| `LIVE_PAPER_TRADING_TICKERS` | Comma-separated list of tickers to run. Defaults to the 70-ticker dissertation universe if unset. |

## How to launch (once implemented)

```bash
python -m live.paper_trading.alpaca_paper_loop --tag phaseA-shadow-2026-09
```

The `--tag` argument is appended to every JSONL log line for retrieval.

## Out

JSONL audit trail under `live/paper_trading/runs/<tag>/` with one record
per ticker per trading day.
