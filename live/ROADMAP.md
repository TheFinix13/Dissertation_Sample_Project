# Real-world deployment roadmap

Three phases, low risk to higher risk, each gated by an explicit go / no-go
criterion. The criteria are written so that they can be evaluated mechanically
from the audit trail produced by the previous phase.

The metrics referenced below are defined in Chapter 3 of the main
dissertation; the most important ones for these gates are:

- **Capital-preservation ratio against the running high-watermark.** The
  fraction of trading days the equity curve stays at or above 95% of its
  running peak. The dissertation requires this to sit above 0.95 in the
  headline window.
- **Sharpe ratio.** Annualised excess return divided by annualised
  volatility, computed on the live equity curve.
- **Maximum drawdown.** The largest peak-to-trough decline in equity over
  the window.

---

## Phase A — Shadow paper-trading via Alpaca

**Goal.** Confirm that the agent, run end-to-end against a real market data
feed and a paper-account order surface, reproduces the backtest behaviour
within Monte-Carlo noise. No real capital is touched.

**Checklist.**

- [ ] Provision an Alpaca paper-trading account and store credentials in
      environment variables (`ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`,
      `ALPACA_PAPER_BASE_URL`). No secrets in the repo.
- [ ] Wire the saved 70-ticker probabilistic policy and its DeepAR-style
      forecaster behind a daily polling loop in
      `live/paper_trading/alpaca_paper_loop.py`.
- [ ] Log every observation, predicted `(mean, std)` pair, raw action,
      scaled action and resulting fill on a per-day, per-ticker basis to
      a versioned JSONL trail under `live/paper_trading/runs/`.
- [ ] Run the loop in shadow mode for **at least two weeks of trading
      days**. The agent acts only on the paper account.
- [ ] Reconcile the live equity curve against a simultaneous backtest
      replay on the same observation stream, to confirm the live wiring
      reproduces the backtest behaviour up to fill-time and slippage.

**Go / no-go criterion to advance to Phase B.**

- Live capital-preservation ratio sits **above 0.95** across the shadow
  window, **and**
- live Sharpe ratio is **non-inferior to the matched-window backtest**
  within Monte-Carlo noise (overlapping bootstrap confidence intervals).

If either fails, stop and write a written incident note before retrying.

---

## Phase B — Decision-support advisor against the real portfolio

**Goal.** With Phase A's reconciliation in hand, run the agent against the
real portfolio as a *decision-support tool*. The human is the executor; the
agent is the advisor. No orders leave the script.

**Checklist.**

- [ ] Build `live/decision_support/nightly_advisor.py`: pull current
      positions and prices from Alpaca (live account, **read-only**), run
      the agent on the day's observation, and emit a recommendation
      (target weight, scaled action, confidence) by email or Slack.
- [ ] Operate as decision-support only for **at least one full quarter of
      trading days** (≈ 63 trading days).
- [ ] Maintain a side-by-side ledger that records, for every
      recommendation, what the agent suggested, what the human did, and
      the resulting P&L attribution between the two.
- [ ] Include an explicit kill-switch (the `live/KILL_SWITCH` file) and a
      hard exposure cap (no single-name position above a configurable
      percent of equity, no aggregate gross above a configurable percent
      of equity) — these caps are advisory in this phase, but the script
      refuses to *recommend* anything that would breach them.

**Go / no-go criterion to advance to Phase C.**

- Across the quarter the recommendations would have produced a Sharpe
  **non-inferior to the actual portfolio**, **and**
- maximum drawdown **below the running-high-watermark threshold** the
  dissertation reports (the same 0.95 floor as Phase A).

If either fails, stay in Phase B and revisit the model — the agent is not
yet trustworthy enough to drive execution.

---

## Phase C — Scheduled execution with continuous risk gates

**Goal.** Place real orders on the real account, automatically, behind a
non-negotiable layer of risk gates. Sized small initially and only scaled up
after each successive month of clean operation.

**Checklist.**

- [ ] Build `live/execution/scheduled_executor.py`: cron-driven (or
      systemd-timer-driven) entry that runs the advisor and, where it
      satisfies pre-trade risk gates, places the corresponding orders on
      the live account through the Alpaca trading API.
- [ ] Pre-trade risk gates are non-negotiable and applied before any order
      leaves the script:
  - position-size cap (no single-name position above X% of equity)
  - gross-exposure cap (sum of |position| / equity ≤ Y)
  - sector-exposure cap (per the GICS-like sector mapping used in
    Section 5.5 of the dissertation)
  - daily-loss cap (cumulative day P&L below `-Z%` halts trading)
  - drawdown-against-high-watermark circuit breaker (running 0.95
    floor; breach halts trading and requires a written incident note)
- [ ] Run only on a small fraction of equity initially. Size up only after
      each successive **month** of operation continues to satisfy the
      preservation-against-high-watermark and Sharpe gates.
- [ ] Maintain the same observation / prediction / action / fill audit
      trail as Phases A and B, so any month of live operation can be
      replayed against the backtest deterministically.

**Stop criterion (from any state in Phase C).**

- Any breach of the preservation-against-high-watermark gate halts the
  executor, requires a written incident note in `live/execution/runs/`,
  and forces a **return to Phase B** until the cause is understood.

---

## Out-of-scope for this roadmap

The following items are interesting but explicitly **not** part of the
post-submission deployment roadmap, and would each be their own project:

- multi-account / multi-portfolio support;
- intraday execution (the dissertation is daily-bar);
- options or derivatives execution (the dissertation is long-only equities);
- a public hosted decision-support service.

These are deliberately listed here to keep the roadmap honest about what it
covers and what it does not.
