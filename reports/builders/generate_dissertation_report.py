import json
from pathlib import Path


def _latest(pattern: str, folder: Path) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[-1]


def _avg(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(len(rows), 1)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root.parent / "experiments" / "results"
    template_path = root / "templates" / "dissertation_results_template.md"

    baseline_path = _latest("baseline_*.json", results_dir)
    probabilistic_path = _latest("probabilistic_*.json", results_dir)
    benchmark_path = _latest("benchmarks_*.json", results_dir)

    baseline_rows = json.loads(baseline_path.read_text(encoding="utf-8"))
    prob_rows = json.loads(probabilistic_path.read_text(encoding="utf-8"))
    benchmark_rows = json.loads(benchmark_path.read_text(encoding="utf-8"))
    template = template_path.read_text(encoding="utf-8")

    baseline_final = _avg(baseline_rows, "final_portfolio_value")
    baseline_sharpe = _avg(baseline_rows, "sharpe_ratio")
    baseline_mdd = _avg(baseline_rows, "max_drawdown")
    baseline_var = _avg(baseline_rows, "var_95_violation_rate")
    baseline_pres = _avg(baseline_rows, "capital_preservation_rate_95pct_hwm")

    prob_final = _avg(prob_rows, "final_portfolio_value")
    prob_sharpe = _avg(prob_rows, "sharpe_ratio")
    prob_mdd = _avg(prob_rows, "max_drawdown")
    prob_var = _avg(prob_rows, "var_95_violation_rate")
    prob_pres = _avg(prob_rows, "capital_preservation_rate_95pct_hwm")
    bench_lookup = {row["agent"]: row for row in benchmark_rows}
    buy_hold = bench_lookup.get("buy_and_hold", {})

    report = (
        template.replace("{{baseline_final_value}}", f"{baseline_final:,.2f}")
        .replace("{{baseline_sharpe}}", f"{baseline_sharpe:.4f}")
        .replace("{{baseline_max_drawdown}}", f"{baseline_mdd:.4f}")
        .replace("{{baseline_var_violations}}", f"{baseline_var:.4f}")
        .replace("{{baseline_preservation}}", f"{baseline_pres:.4f}")
        .replace("{{prob_final_value}}", f"{prob_final:,.2f}")
        .replace("{{prob_sharpe}}", f"{prob_sharpe:.4f}")
        .replace("{{prob_max_drawdown}}", f"{prob_mdd:.4f}")
        .replace("{{prob_var_violations}}", f"{prob_var:.4f}")
        .replace("{{prob_preservation}}", f"{prob_pres:.4f}")
        .replace("{{preservation_delta}}", f"{(prob_pres - baseline_pres):.4f}")
        .replace("{{drawdown_delta}}", f"{(baseline_mdd - prob_mdd):.4f}")
        .replace(
            "{{decision}}",
            (
                "Probabilistic agent improves preservation over PPO baseline"
                if prob_pres > baseline_pres
                else "No clear probabilistic improvement over PPO baseline"
            ),
        )
    )
    report += (
        "\n\n## Benchmark Check\n"
        f"- Buy-and-hold final value: {float(buy_hold.get('final_portfolio_value', 0.0)):,.2f}\n"
        f"- Buy-and-hold max drawdown: {float(buy_hold.get('max_drawdown', 0.0)):.4f}\n"
    )

    output = root / "generated" / "dissertation_results.md"
    output.write_text(report, encoding="utf-8")
    print(f"Report generated at: {output}")


if __name__ == "__main__":
    main()
