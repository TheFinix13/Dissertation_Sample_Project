"""Walk-forward evaluation across multiple tickers, seeds and folds.

Trains the baseline PPO and the probabilistic PPO on each fold's TRAIN window
and evaluates both on the corresponding TEST window — so this is a properly
out-of-time evaluation, unlike the legacy single-fold runners which trained
and evaluated on the test window.

For each (ticker, fold, seed) cell, the harness writes:
    - one row per agent into the aggregated JSON / CSV
    - the corresponding portfolio-value curve into a per-cell CSV

Run examples:
    # Quick smoke test (2 tickers, 3 seeds, 1 fold, 5k timesteps)
    venv/bin/python experiments/run_walk_forward.py \\
        --tickers SPY,QQQ --seeds default --folds wf_2022_2023 --timesteps 5000 --tag smoke

    # Headline run (3 tickers, default 3 seeds, all folds, 50k timesteps)
    venv/bin/python experiments/run_walk_forward.py \\
        --tickers SPY,QQQ,IWM --folds all --timesteps 50000 --tag headline

    # Full robustness sweep (8 tickers, 10 seeds, all folds, 50k timesteps).
    # CPU runtime: several hours; recommended on a Colab GPU runtime.
    venv/bin/python experiments/run_walk_forward.py \\
        --tickers basket --seeds extended --folds all --timesteps 50000 --tag full
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
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
    maybe_bootstrap_training_prices,
    resolve_folds,
    resolve_initial_balance,
    resolve_seeds,
    resolve_tickers,
    set_global_seed,
)
from run_probabilistic_agent import estimate_uncertainty


def _train_and_eval(
    *,
    train_prices,
    test_prices,
    train_uncertainty=None,
    test_uncertainty=None,
    seed: int,
    timesteps: int,
    cfg: EnvConfig,
):
    set_global_seed(seed)

    def _make_env(prices=train_prices, uncertainty=train_uncertainty, cfg=cfg):
        return StockEnv(prices=prices, uncertainty=uncertainty, cfg=cfg)

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

    eval_env = StockEnv(prices=test_prices, uncertainty=test_uncertainty, cfg=cfg)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _, _ = eval_env.step(action)
    return eval_env.portfolio_values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_cli(parser)
    parser.add_argument(
        "--agents", default="baseline,probabilistic",
        help="Which agents to run, comma-separated. Subset of "
             "{baseline, probabilistic}. Default: both.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    protocol = load_protocol(root / "configs" / "dissertation_protocol.json")
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "wf_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.tag)

    tickers = resolve_tickers(args.tickers, protocol)
    seeds = resolve_seeds(args.seeds, protocol)
    folds = resolve_folds(args.folds, protocol)
    timesteps = (
        args.timesteps if args.timesteps is not None
        else protocol["probabilistic_agent"]["timesteps"]
    )
    initial_balance = resolve_initial_balance(args, protocol)
    bootstrap_paths = int(args.bootstrap_paths)
    requested_agents = {a.strip().lower() for a in args.agents.split(",") if a.strip()}

    baseline_name = protocol["baseline"]["model_name"]
    prob_name = protocol["probabilistic_agent"]["model_name"]
    prob_overrides = dict(
        uncertainty_stop_quantile=protocol["probabilistic_agent"]["uncertainty_quantile_stop"],
        min_trade_scale=protocol["probabilistic_agent"]["position_scale_floor"],
        initial_balance=initial_balance,
    )

    rows = []
    total_cells = len(tickers) * len(folds) * len(seeds)
    cell_idx = 0
    print(f"Walk-forward sweep: {len(tickers)} tickers x {len(folds)} folds x "
          f"{len(seeds)} seeds = {total_cells} cells; timesteps={timesteps}, "
          f"bootstrap_paths={bootstrap_paths}, agents={sorted(requested_agents)}")

    for ticker in tickers:
        for fold in folds:
            train_start, train_end = fold["train"]
            test_start, test_end = fold["test"]
            try:
                train_df = fetch_close_frame(ticker, train_start, train_end)
                test_df = fetch_close_frame(ticker, test_start, test_end)
            except ValueError as e:
                print(f"[WARN] {ticker} fold={fold['fold_id']}: {e}")
                continue

            train_close = close_1d(train_df)
            test_close = close_1d(test_df)
            train_prices = train_close.to_numpy(dtype="float32")
            test_prices = test_close.to_numpy(dtype="float32")

            test_uncertainty = estimate_uncertainty(test_prices)

            for seed in seeds:
                cell_idx += 1
                cell_label = (
                    f"[{cell_idx}/{total_cells}] {ticker:<5} fold={fold['fold_id']:<14} seed={seed:>3}"
                )

                train_prices_aug = maybe_bootstrap_training_prices(
                    train_prices, num_paths=bootstrap_paths, protocol=protocol, seed=seed,
                )
                if bootstrap_paths > 0:
                    train_uncertainty = estimate_uncertainty(train_prices_aug)
                else:
                    train_uncertainty = estimate_uncertainty(train_prices)

                if "baseline" in requested_agents:
                    base_cfg = EnvConfig(initial_balance=initial_balance)
                    base_curve = _train_and_eval(
                        train_prices=train_prices_aug,
                        test_prices=test_prices,
                        train_uncertainty=None,
                        test_uncertainty=None,
                        seed=seed,
                        timesteps=timesteps,
                        cfg=base_cfg,
                    )
                    base_metrics = compute_metrics(base_curve)
                    base_metrics.update({
                        "agent": baseline_name,
                        "ticker": ticker,
                        "fold_id": fold["fold_id"],
                        "seed": seed,
                        "timesteps": timesteps,
                        "bootstrap_paths": bootstrap_paths,
                        "train_window": f"{train_start}/{train_end}",
                        "test_window":  f"{test_start}/{test_end}",
                    })
                    rows.append(base_metrics)
                    pd.DataFrame({
                        "date": [d.strftime("%Y-%m-%d") for d in test_close.index[: len(base_curve)]],
                        "portfolio_value": base_curve,
                        "agent": baseline_name,
                        "ticker": ticker,
                        "fold_id": fold["fold_id"],
                        "seed": seed,
                    }).to_csv(
                        curves_dir
                        / f"baseline_{ticker}_{fold['fold_id']}_seed{seed}_{run_id}.csv",
                        index=False,
                    )
                    print(f"{cell_label} baseline:      "
                          f"final={base_metrics['final_portfolio_value']:.0f}, "
                          f"sharpe={base_metrics['sharpe_ratio']:+.4f}, "
                          f"mdd={base_metrics['max_drawdown']:.4f}")

                if "probabilistic" in requested_agents:
                    prob_cfg = EnvConfig(**prob_overrides)
                    prob_curve = _train_and_eval(
                        train_prices=train_prices_aug,
                        test_prices=test_prices,
                        train_uncertainty=train_uncertainty,
                        test_uncertainty=test_uncertainty,
                        seed=seed,
                        timesteps=timesteps,
                        cfg=prob_cfg,
                    )
                    prob_metrics = compute_metrics(prob_curve)
                    prob_metrics.update({
                        "agent": prob_name,
                        "ticker": ticker,
                        "fold_id": fold["fold_id"],
                        "seed": seed,
                        "timesteps": timesteps,
                        "bootstrap_paths": bootstrap_paths,
                        "train_window": f"{train_start}/{train_end}",
                        "test_window":  f"{test_start}/{test_end}",
                    })
                    rows.append(prob_metrics)
                    pd.DataFrame({
                        "date": [d.strftime("%Y-%m-%d") for d in test_close.index[: len(prob_curve)]],
                        "portfolio_value": prob_curve,
                        "agent": prob_name,
                        "ticker": ticker,
                        "fold_id": fold["fold_id"],
                        "seed": seed,
                    }).to_csv(
                        curves_dir
                        / f"probabilistic_{ticker}_{fold['fold_id']}_seed{seed}_{run_id}.csv",
                        index=False,
                    )
                    print(f"{cell_label} probabilistic: "
                          f"final={prob_metrics['final_portfolio_value']:.0f}, "
                          f"sharpe={prob_metrics['sharpe_ratio']:+.4f}, "
                          f"mdd={prob_metrics['max_drawdown']:.4f}")

    if not rows:
        print("[ERROR] no results were produced; check tickers / folds / network access.")
        return

    json_path = out_dir / f"walk_forward_{run_id}.json"
    csv_path = out_dir / f"walk_forward_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote walk-forward results:\n- {json_path}\n- {csv_path}\n- per-cell curves in {curves_dir}/")


if __name__ == "__main__":
    main()
