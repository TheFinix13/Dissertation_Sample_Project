"""Rule-based stop-loss comparator for the dissertation protocol.

Two non-AI baselines are produced, both starting fully invested in the
underlying asset and following a trailing-stop / moving-average-re-entry rule:

    - stop_loss_5pct : trailing stop at -5% from peak, re-enter on MA20 > MA50
    - stop_loss_10pct: trailing stop at -10% from peak, re-enter on MA20 > MA50

The portfolio-value curve is fed through the same compute_metrics helper used
for every other agent, so the resulting JSON / CSV rows slot directly into
the dissertation comparison table.

Loops over all tickers resolved from --tickers; falls back to the legacy
single ticker when the flag is omitted.

Run examples:
    venv/bin/python experiments/runners/run_rule_baselines.py
    venv/bin/python experiments/runners/run_rule_baselines.py --tickers basket
    venv/bin/python experiments/runners/run_rule_baselines.py --tickers SPY,QQQ,IWM,XLK,XLF,XLE,XLV,XLU
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    add_common_cli,
    compute_metrics,
    fetch_close_prices,
    load_protocol,
    make_run_id,
    resolve_initial_balance,
    resolve_tickers,
)


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Trailing simple moving average; positions before the window are NaN."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    if window <= 0 or window > len(series):
        return out
    cumsum = np.cumsum(series, dtype=np.float64)
    out[window - 1:] = (cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:-window]))) / window
    return out


def rule_based_curve(
    prices: np.ndarray,
    *,
    drawdown_floor: float,
    fast_window: int = 20,
    slow_window: int = 50,
    initial_balance: float = 1_000_000.0,
    transaction_cost_rate: float = 0.001,
) -> list[float]:
    """Simulate the trailing-stop / MA-crossover re-entry policy.

    Logic at every close, after the slow-window warmup:
        - If currently invested and (V_t / running_peak) < (1 - drawdown_floor),
          exit fully at the close (sell all shares, debit transaction cost).
        - If currently in cash and MA(fast) > MA(slow), re-enter fully at the
          close (buy with all available cash, debit transaction cost) and
          reset the running peak.

    During the moving-average warmup period the policy holds the initial
    long position (matching buy-and-hold).
    """
    n = len(prices)
    fast = moving_average(prices, fast_window)
    slow = moving_average(prices, slow_window)

    p0 = float(prices[0])
    shares = initial_balance / max(p0, 1e-8)
    cash = 0.0
    in_market = True
    running_peak = initial_balance

    curve = [initial_balance]

    for t in range(1, n):
        p_t = float(prices[t])
        v_t = cash + shares * p_t

        if in_market and v_t > running_peak:
            running_peak = v_t

        if in_market:
            drawdown = 1.0 - (v_t / max(running_peak, 1e-8))
            if drawdown >= drawdown_floor:
                gross = shares * p_t
                fee = gross * transaction_cost_rate
                cash = max(gross - fee, 0.0)
                shares = 0.0
                in_market = False
                v_t = cash
        else:
            ma_ready = not (np.isnan(fast[t]) or np.isnan(slow[t]))
            if ma_ready and fast[t] > slow[t]:
                gross = cash
                fee = gross * transaction_cost_rate
                shares = max(gross - fee, 0.0) / max(p_t, 1e-8)
                cash = 0.0
                in_market = True
                running_peak = shares * p_t
                v_t = shares * p_t

        curve.append(v_t)

    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_cli(parser)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    protocol = load_protocol(root / "configs" / "dissertation_protocol.json")
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.tag)

    tickers = resolve_tickers(args.tickers, protocol)
    initial_balance = resolve_initial_balance(args, protocol)
    test_start, test_end = protocol["splits"]["test"]

    variants = [
        ("stop_loss_5pct", 0.05),
        ("stop_loss_10pct", 0.10),
    ]

    rows = []
    curves_payload: dict[str, dict[str, list[float]]] = {}

    for ticker in tickers:
        try:
            prices = fetch_close_prices(ticker, test_start, test_end)
        except ValueError as e:
            print(f"[WARN] {ticker}: {e}")
            continue
        curves_payload[ticker] = {}

        for agent_name, dd_floor in variants:
            curve = rule_based_curve(
                prices, drawdown_floor=dd_floor, initial_balance=initial_balance,
            )
            metrics = compute_metrics(curve)
            metrics["agent"] = agent_name
            metrics["ticker"] = ticker
            metrics["seed"] = -1
            metrics["drawdown_floor"] = dd_floor
            metrics["fast_window"] = 20
            metrics["slow_window"] = 50
            rows.append(metrics)
            curves_payload[ticker][agent_name] = curve
            print(
                f"{ticker:<5} {agent_name}: "
                f"final={metrics['final_portfolio_value']:.2f}, "
                f"sharpe={metrics['sharpe_ratio']:.4f}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"preservation={metrics['capital_preservation_rate_95pct_hwm']:.4f}"
            )

    if not rows:
        print("[ERROR] no results were produced; check tickers / network access.")
        return

    json_path = out_dir / f"rule_baseline_{run_id}.json"
    csv_path = out_dir / f"rule_baseline_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    curves_path = out_dir / f"stop_loss_curves_{run_id}.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves_payload, f)

    print(f"\nWrote rule-based baseline results:\n- {json_path}\n- {csv_path}\n- {curves_path}")


if __name__ == "__main__":
    main()
