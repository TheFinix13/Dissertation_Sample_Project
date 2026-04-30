# Phase B — Decision-support advisor against the real portfolio

This sub-folder is the engineering scaffold for Phase B of the
post-submission deployment roadmap (see `../ROADMAP.md`).

**Status: scaffold only.** `nightly_advisor.py` is a placeholder that
imports the trained models and exits cleanly with a "to be implemented"
log line. It does **not** read the real account.

## What this phase does (when implemented)

A nightly advisory loop that:

1. pulls current positions and prices from the **real** Alpaca account via
   the read-only data endpoints;
2. runs the trained agent on the day's observation;
3. emits a recommendation (target weight, scaled action, confidence) by
   email or Slack to the human operator;
4. records what the agent suggested, what the human did, and the resulting
   P&L attribution between the two, in a side-by-side ledger.

**Phase B never places an order.** The human is the executor.

## Required environment variables

| Variable | Purpose |
|---|---|
| `ALPACA_KEY_ID` | Alpaca API key (live, read-only scopes only). |
| `ALPACA_SECRET_KEY` | Alpaca API secret (live, read-only scopes only). |
| `ALPACA_LIVE_BASE_URL` | Live trading endpoint, e.g. `https://api.alpaca.markets`. |
| `ADVISOR_NOTIFY_TARGET` | Slack webhook URL or email address for the recommendation. |

## Recommendation contract

Every emitted recommendation is a JSON object of the shape:

```json
{
  "as_of": "2026-09-15",
  "ticker": "SPY",
  "target_weight": 0.045,
  "scaled_action": 0.012,
  "confidence": 0.83,
  "uncertainty": 0.17,
  "would_breach_caps": false,
  "audit_id": "phaseB-2026-09-15-SPY"
}
```

A recommendation is **suppressed** if it would breach the advisory
exposure caps; the suppression is logged in the audit trail.
