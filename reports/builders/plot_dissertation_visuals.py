from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def _latest_metric_csv(prefix: str, folder: Path) -> Path:
    files = sorted(
        p
        for p in folder.glob(f"{prefix}_*.csv")
        if not p.name.startswith(f"{prefix}_curve_")
    )
    if not files:
        raise FileNotFoundError(f"No metric CSV files found for prefix: {prefix}")
    return files[-1]


def _run_id_from_metrics(path: Path, prefix: str) -> str:
    # example: baseline_20260411T230000Z.csv -> 20260411T230000Z
    name = path.name
    return name.replace(f"{prefix}_", "").replace(".csv", "")


def _load_curves(results_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No curve files found for pattern: {pattern}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _plot_equity_curves(base_df: pd.DataFrame, prob_df: pd.DataFrame, out_path: Path) -> None:
    b = base_df.groupby("date", as_index=False)["portfolio_value"].mean()
    p = prob_df.groupby("date", as_index=False)["portfolio_value"].mean()
    plt.figure(figsize=(10, 4))
    plt.plot(b["date"], b["portfolio_value"], label="Baseline PPO", linewidth=1.8)
    plt.plot(p["date"], p["portfolio_value"], label="Probabilistic PPO", linewidth=1.8)
    plt.xticks(rotation=45)
    plt.ylabel("Portfolio Value (USD)")
    plt.title("Agent Equity Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_uncertainty(prob_df: pd.DataFrame, out_path: Path) -> None:
    p = prob_df.groupby("date", as_index=False)["uncertainty"].mean()
    plt.figure(figsize=(10, 3))
    plt.plot(p["date"], p["uncertainty"], color="orange", linewidth=1.6)
    plt.xticks(rotation=45)
    plt.ylabel("Uncertainty Score")
    plt.title("Probabilistic Uncertainty Signal Over Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_market_data(ticker: str, start: str, end: str, out_path: Path) -> None:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker} [{start} to {end}]")
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Close"], linewidth=1.6)
    plt.ylabel("Adjusted Close")
    plt.title(f"{ticker} Dataset Used For Experiments")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_intraday_realtime_proxy(ticker: str, out_path: Path) -> None:
    intraday = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if intraday.empty:
        return
    plt.figure(figsize=(10, 3))
    plt.plot(intraday.index, intraday["Close"], linewidth=1.4)
    plt.ylabel("Price")
    plt.title(f"{ticker} Intraday (1m) Real-time Proxy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    root = Path(__file__).resolve().parent.parent
    repo_root = root.parent
    results_dir = repo_root / "experiments" / "results"
    charts_dir = root / "generated" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = _latest_metric_csv("baseline", results_dir)
    probabilistic_csv = _latest_metric_csv("probabilistic", results_dir)
    run_id = _run_id_from_metrics(baseline_csv, "baseline")
    prob_run_id = _run_id_from_metrics(probabilistic_csv, "probabilistic")

    base_curve = _load_curves(results_dir, f"baseline_curve_{run_id}_seed*.csv")
    prob_curve = _load_curves(results_dir, f"probabilistic_curve_{prob_run_id}_seed*.csv")

    _plot_equity_curves(base_curve, prob_curve, charts_dir / "equity_curve_comparison.png")
    _plot_uncertainty(prob_curve, charts_dir / "uncertainty_signal.png")
    _plot_market_data("SPY", "2022-01-01", "2025-12-31", charts_dir / "dataset_spy_close.png")
    _plot_intraday_realtime_proxy("SPY", charts_dir / "spy_intraday_realtime_proxy.png")

    print(
        "Wrote charts:\n"
        f"- {charts_dir / 'equity_curve_comparison.png'}\n"
        f"- {charts_dir / 'uncertainty_signal.png'}\n"
        f"- {charts_dir / 'dataset_spy_close.png'}\n"
        f"- {charts_dir / 'spy_intraday_realtime_proxy.png'}"
    )


if __name__ == "__main__":
    main()
