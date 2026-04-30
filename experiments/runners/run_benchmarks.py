"""Buy-and-hold and all-cash benchmark runner.

Loops over all tickers resolved from --tickers (or the legacy single ticker
when the flag is omitted) and writes a row per (ticker, agent) pair so the
benchmark numbers slot directly into the multi-asset comparison tables.

Run examples:
    venv/bin/python experiments/runners/run_benchmarks.py
    venv/bin/python experiments/runners/run_benchmarks.py --tickers basket
    venv/bin/python experiments/runners/run_benchmarks.py --tickers SPY,QQQ,IWM
"""

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


def buy_and_hold_curve(prices: np.ndarray, initial_balance: float) -> list[float]:
    shares = initial_balance / max(float(prices[0]), 1e-8)
    curve = (shares * prices).astype(np.float64)
    return curve.tolist()


def equal_cash_curve(prices: np.ndarray, initial_balance: float) -> list[float]:
    return [initial_balance for _ in prices]


def main():
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

    rows = []
    for ticker in tickers:
        try:
            prices = fetch_close_prices(ticker, test_start, test_end)
        except ValueError as e:
            print(f"[WARN] {ticker}: {e}")
            continue
        for name, curve in [
            ("buy_and_hold", buy_and_hold_curve(prices, initial_balance)),
            ("all_cash",     equal_cash_curve(prices, initial_balance)),
        ]:
            metrics = compute_metrics(curve)
            metrics["agent"] = name
            metrics["ticker"] = ticker
            metrics["seed"] = -1
            rows.append(metrics)
            print(
                f"{ticker:<5} {name:<13}: "
                f"final={metrics['final_portfolio_value']:.2f}, "
                f"sharpe={metrics['sharpe_ratio']:.4f}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"preservation={metrics['capital_preservation_rate_95pct_hwm']:.4f}"
            )

    if not rows:
        print("[ERROR] no results were produced; check tickers / network access.")
        return

    json_path = out_dir / f"benchmarks_{run_id}.json"
    csv_path = out_dir / f"benchmarks_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote benchmark results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
