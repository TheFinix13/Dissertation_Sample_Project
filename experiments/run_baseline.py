"""Baseline PPO runner — single fold (legacy: train + evaluate on the test window).

Loops over all (ticker, seed) combinations resolved from CLI flags; falls back
to the original single-ticker, three-seed behaviour when invoked with no flags
so existing dissertation numbers stay reproducible.

Run examples:
    venv/bin/python experiments/run_baseline.py
    venv/bin/python experiments/run_baseline.py --tickers basket
    venv/bin/python experiments/run_baseline.py --tickers SPY,QQQ --seeds extended --timesteps 50000 --tag full
"""

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import (
    EnvConfig,
    StockEnv,
    add_common_cli,
    close_1d,
    compute_metrics,
    fetch_close_frame,
    load_protocol,
    make_run_id,
    resolve_initial_balance,
    resolve_seeds,
    resolve_tickers,
    set_global_seed,
)


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
    seeds = resolve_seeds(args.seeds, protocol)
    timesteps = args.timesteps if args.timesteps is not None else protocol["baseline"]["timesteps"]
    initial_balance = resolve_initial_balance(args, protocol)

    test_start, test_end = protocol["splits"]["test"]
    model_name = protocol["baseline"]["model_name"]

    rows = []
    for ticker in tickers:
        try:
            price_df = fetch_close_frame(ticker, test_start, test_end)
        except ValueError as e:
            print(f"[WARN] {ticker}: {e}")
            continue
        close = close_1d(price_df)
        prices = close.to_numpy(dtype="float32")

        for seed in seeds:
            set_global_seed(seed)
            env_cfg = EnvConfig(initial_balance=initial_balance)

            def _make_env(prices=prices, env_cfg=env_cfg):
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
            model.learn(total_timesteps=timesteps)

            eval_env = StockEnv(prices=prices, cfg=env_cfg)
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, _, _ = eval_env.step(action)

            portfolio_values = eval_env.portfolio_values
            metrics = compute_metrics(portfolio_values)
            metrics["seed"] = seed
            metrics["ticker"] = ticker
            metrics["fold_id"] = "test_legacy"
            metrics["timesteps"] = timesteps
            metrics["agent"] = model_name
            rows.append(metrics)
            print(
                f"{ticker:<5} seed={seed:>3} ts={timesteps}: "
                f"final={metrics['final_portfolio_value']:.2f}, "
                f"sharpe={metrics['sharpe_ratio']:.4f}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"preservation={metrics['capital_preservation_rate_95pct_hwm']:.4f}"
            )

    if not rows:
        print("[ERROR] no results were produced; check tickers / network access.")
        return

    json_path = out_dir / f"baseline_{run_id}.json"
    csv_path = out_dir / f"baseline_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote baseline results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
