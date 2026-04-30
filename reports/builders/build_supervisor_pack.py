import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _latest_metric_csv(prefix: str, folder: Path) -> Path:
    files = sorted(
        p
        for p in folder.glob(f"{prefix}_*.csv")
        if not p.name.startswith(f"{prefix}_curve_")
    )
    if not files:
        raise FileNotFoundError(f"No metric CSV files found for prefix: {prefix}")
    return files[-1]


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _avg(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return sum(vals) / max(len(vals), 1)


def build_chart(baseline: list[dict], probabilistic: list[dict], buy_hold_value: float, chart_path: Path):
    labels = ["BaselinePPO", "ProbabilisticPPO", "BuyAndHold"]
    values = [
        _avg(baseline, "final_portfolio_value"),
        _avg(probabilistic, "final_portfolio_value"),
        buy_hold_value,
    ]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values)
    plt.title("Final Portfolio Value Comparison (Test Window)")
    plt.ylabel("USD")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root.parent / "experiments" / "results"
    out_dir = root / "generated"
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = _latest_metric_csv("baseline", results_dir)
    probabilistic_csv = _latest_metric_csv("probabilistic", results_dir)
    benchmark_csv = _latest_metric_csv("benchmarks", results_dir)

    baseline_rows = _read_csv(baseline_csv)
    probabilistic_rows = _read_csv(probabilistic_csv)
    benchmark_rows = _read_csv(benchmark_csv)
    buy_hold_row = next((r for r in benchmark_rows if r["agent"] == "buy_and_hold"), None)
    buy_hold_value = float(buy_hold_row["final_portfolio_value"]) if buy_hold_row else 0.0
    buy_hold_mdd = float(buy_hold_row["max_drawdown"]) if buy_hold_row else 0.0

    chart_path = charts_dir / "final_value_comparison.png"
    build_chart(baseline_rows, probabilistic_rows, buy_hold_value, chart_path)

    baseline_pres = _avg(baseline_rows, "capital_preservation_rate_95pct_hwm")
    probabilistic_pres = _avg(probabilistic_rows, "capital_preservation_rate_95pct_hwm")
    baseline_mdd = _avg(baseline_rows, "max_drawdown")
    probabilistic_mdd = _avg(probabilistic_rows, "max_drawdown")
    baseline_sharpe = _avg(baseline_rows, "sharpe_ratio")
    probabilistic_sharpe = _avg(probabilistic_rows, "sharpe_ratio")
    baseline_final = _avg(baseline_rows, "final_portfolio_value")
    probabilistic_final = _avg(probabilistic_rows, "final_portfolio_value")

    report_text = f"""# Supervisor Progress Report (Draft)

## Executive Summary
- Objective: test whether uncertainty-aware PPO improves risk-aware portfolio behavior.
- Status: baseline PPO, probabilistic PPO, and benchmarks are implemented and reproducible.
- Current result: probabilistic PPO shows stronger final value and preservation than baseline PPO on current setup.

## Data and Protocol
- Source: Yahoo Finance daily adjusted close via `yfinance` API.
- Market proxy: SPY (configured ticker) across protocol split in `experiments/configs/dissertation_protocol.json`.
- Benchmark checks: Buy-and-hold and all-cash.

## Current Results (Average Across Seeds)
- Baseline PPO final value: {baseline_final:,.2f}
- Probabilistic PPO final value: {probabilistic_final:,.2f}
- Baseline preservation ratio: {baseline_pres:.4f}
- Probabilistic preservation ratio: {probabilistic_pres:.4f}
- Baseline max drawdown: {baseline_mdd:.4f}
- Probabilistic max drawdown: {probabilistic_mdd:.4f}
- Baseline Sharpe: {baseline_sharpe:.4f}
- Probabilistic Sharpe: {probabilistic_sharpe:.4f}
- Buy-and-hold final value: {buy_hold_value:,.2f}
- Buy-and-hold max drawdown: {buy_hold_mdd:.4f}

## Interpretation (For Discussion)
- Probabilistic variant currently improves capital-preservation ratio and final value vs baseline PPO.
- Max drawdown trade-off is visible and should be discussed during viva.
- Result should be treated as provisional pending robustness tests and alternate tickers/event windows.

## What Is Ready by Monday
- Reproducible scripts:
  - `experiments/runners/run_baseline.py`
  - `experiments/runners/run_probabilistic_agent.py`
  - `experiments/runners/run_benchmarks.py`
  - `reports/builders/generate_dissertation_report.py`
- Supervisor chart: `reports/generated/charts/final_value_comparison.png`
- Dissertation summary: `reports/generated/dissertation_results.md`

## Next Tests Before Viva
- Multi-ticker tests (SPY, QQQ, sector ETFs).
- Sensitivity test on uncertainty threshold and trade scaling.
- Event-window analysis for shock periods.
- Ablation: PPO vs PPO+uncertainty-signal vs PPO+uncertainty-guard.
"""
    out_path = out_dir / "supervisor_progress_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote supervisor pack:\n- {out_path}\n- {chart_path}")


if __name__ == "__main__":
    main()
