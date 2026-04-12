# Supervisor Progress Report (Draft)

## Executive Summary

- Objective: test whether uncertainty-aware PPO improves risk-aware portfolio behavior.
- Status: baseline PPO, probabilistic PPO, and benchmarks are implemented and reproducible.
- Current result: probabilistic PPO shows stronger final value and preservation than baseline PPO on current setup.

## Data and Protocol

- Source: Yahoo Finance daily adjusted close via `yfinance` API.
- Market proxy: SPY (configured ticker) across protocol split in `experiments/configs/dissertation_protocol.json`.
- Benchmark checks: Buy-and-hold and all-cash.

## Current Results (Average Across Seeds)

- Baseline PPO final value: 985,463.41
- Probabilistic PPO final value: 1,618,577.16
- Baseline preservation ratio: 0.9811
- Probabilistic preservation ratio: 0.9965
- Baseline max drawdown: 0.0209
- Probabilistic max drawdown: 0.1833
- Baseline Sharpe: -0.4285
- Probabilistic Sharpe: 0.8511
- Buy-and-hold final value: 1,520,353.38
- Buy-and-hold max drawdown: 0.2450

## Interpretation (For Discussion)

- Probabilistic variant currently improves capital-preservation ratio and final value vs baseline PPO.
- Max drawdown trade-off is visible and should be discussed during viva.
- Result should be treated as provisional pending robustness tests and alternate tickers/event windows.

## What Is Ready by Monday

- Reproducible scripts:
  - `experiments/run_baseline.py`
  - `experiments/run_probabilistic_agent.py`
  - `experiments/run_benchmarks.py`
  - `reports/generate_dissertation_report.py`
- Supervisor chart: `reports/generated/charts/final_value_comparison.png`
- Dissertation summary: `reports/generated/dissertation_results.md`

## Next Tests Before Viva

- Multi-ticker tests (SPY, QQQ, sector ETFs).
- Sensitivity test on uncertainty threshold and trade scaling.
- Event-window analysis for shock periods.
- Ablation: PPO vs PPO+uncertainty-signal vs PPO+uncertainty-guard.

