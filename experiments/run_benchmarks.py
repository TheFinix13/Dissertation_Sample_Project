import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from common import compute_metrics, fetch_close_prices, load_protocol


def buy_and_hold_curve(prices: np.ndarray, initial_balance: float = 1_000_000.0) -> list[float]:
    shares = initial_balance / max(float(prices[0]), 1e-8)
    curve = (shares * prices).astype(np.float64)
    return curve.tolist()


def equal_cash_curve(prices: np.ndarray, initial_balance: float = 1_000_000.0) -> list[float]:
    return [initial_balance for _ in prices]


def main():
    root = Path(__file__).resolve().parent
    protocol = load_protocol(root / "configs" / "dissertation_protocol.json")
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_start, test_end = protocol["splits"]["test"]
    prices = fetch_close_prices(protocol["data"]["tickers"][0], test_start, test_end)

    rows = []
    for name, curve in [
        ("buy_and_hold", buy_and_hold_curve(prices)),
        ("all_cash", equal_cash_curve(prices)),
    ]:
        metrics = compute_metrics(curve)
        metrics["agent"] = name
        metrics["seed"] = -1
        rows.append(metrics)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"benchmarks_{timestamp}.json"
    csv_path = out_dir / f"benchmarks_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
