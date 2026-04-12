import csv
import json
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import EnvConfig, StockEnv, compute_metrics, fetch_close_frame, load_protocol, set_global_seed


def _close_1d(price_df: pd.DataFrame) -> pd.Series:
    close = price_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype("float32")


def main():
    root = Path(__file__).resolve().parent
    protocol = load_protocol(root / "configs" / "dissertation_protocol.json")
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    test_start, test_end = protocol["splits"]["test"]
    price_df = fetch_close_frame(protocol["data"]["tickers"][0], test_start, test_end)
    close = _close_1d(price_df)
    prices = close.to_numpy(dtype="float32")

    rows = []
    for seed in protocol["seeds"]:
        set_global_seed(seed)
        env_cfg = EnvConfig()

        def _make_env():
            return StockEnv(prices=prices, cfg=env_cfg)

        env = DummyVecEnv([_make_env])
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=protocol["baseline"]["timesteps"])

        eval_env = StockEnv(prices=prices, cfg=env_cfg)
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _, _ = eval_env.step(action)

        portfolio_values = eval_env.portfolio_values
        date_slice = close.index[env_cfg.lookback : env_cfg.lookback + len(portfolio_values)]
        curve_df = pd.DataFrame(
            {
                "date": [d.strftime("%Y-%m-%d") for d in date_slice],
                "portfolio_value": portfolio_values,
                "seed": seed,
                "agent": protocol["baseline"]["model_name"],
            }
        )
        curve_df.to_csv(out_dir / f"baseline_curve_{run_id}_seed{seed}.csv", index=False)
        metrics = compute_metrics(portfolio_values)
        metrics["seed"] = seed
        metrics["agent"] = protocol["baseline"]["model_name"]
        rows.append(metrics)

    json_path = out_dir / f"baseline_{run_id}.json"
    csv_path = out_dir / f"baseline_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = sorted(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote baseline results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
