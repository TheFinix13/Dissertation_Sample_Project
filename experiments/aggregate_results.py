"""Aggregate per-cell results into median + IQR summaries.

Pools every JSON file under experiments/results/ that matches one of the
runner prefixes (baseline, probabilistic, benchmarks, rule_baseline,
walk_forward) and produces a tidy long-form summary keyed by
(agent, ticker, fold_id) with the median, lower quartile and upper quartile
across seeds for each metric.

The output is a CSV (default) and a JSON copy, both with one row per
(agent, ticker, fold, metric) cell, suitable for direct ingestion into
the dissertation's Results chapter.

Run examples:
    venv/bin/python experiments/aggregate_results.py
    venv/bin/python experiments/aggregate_results.py --prefixes walk_forward --tag headline
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import make_run_id

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

METRICS_OF_INTEREST = (
    "final_portfolio_value",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "var_95",
    "var_95_violation_rate",
    "capital_preservation_rate_95pct_hwm",
)


def _load_json_rows(prefix: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS.glob(f"{prefix}_*.json")):
        if "curves" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            row.setdefault("source_file", path.name)
            rows.append(row)
    return rows


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ticker" not in df.columns:
        df["ticker"] = "SPY"
    else:
        df["ticker"] = df["ticker"].fillna("SPY")
    if "fold_id" not in df.columns:
        df["fold_id"] = "test_legacy"
    else:
        df["fold_id"] = df["fold_id"].fillna("test_legacy")
    out_rows: list[dict] = []
    grouped = df.groupby(["agent", "ticker", "fold_id"], dropna=False)
    for (agent, ticker, fold_id), group in grouped:
        for metric in METRICS_OF_INTEREST:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            out_rows.append({
                "agent": agent,
                "ticker": ticker,
                "fold_id": fold_id,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std":  float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "median": float(values.median()),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prefixes",
        default="baseline,probabilistic,benchmarks,rule_baseline,walk_forward",
        help="Comma-separated list of result prefixes to include.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional tag suffix appended to the output filename.",
    )
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    all_rows: list[dict] = []
    for prefix in prefixes:
        all_rows.extend(_load_json_rows(prefix))

    if not all_rows:
        print(f"[ERROR] no rows found under {RESULTS}/ for prefixes {prefixes}.")
        return

    df = pd.DataFrame(all_rows)
    summary = _summarise(df)

    run_id = make_run_id(args.tag)
    csv_path = RESULTS / f"summary_{run_id}.csv"
    json_path = RESULTS / f"summary_{run_id}.json"

    summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    pivot = summary.pivot_table(
        index=["agent", "ticker", "fold_id"],
        columns="metric",
        values="median",
    )
    print("Median across seeds (head):")
    print(pivot.head(20).to_string())
    print(f"\nWrote summary:\n- {csv_path}\n- {json_path}")


if __name__ == "__main__":
    main()
