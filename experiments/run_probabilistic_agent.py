import csv
import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import EnvConfig, StockEnv, compute_metrics, fetch_close_frame, load_protocol, set_global_seed


def _close_1d(price_df: pd.DataFrame) -> pd.Series:
    close = price_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype("float32")


class ProbabilisticLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        mean = self.fc_mean(last)
        log_var = self.fc_logvar(last)
        return mean, log_var


def gaussian_nll(y_true, mean, log_var):
    var = torch.exp(log_var) + 1e-6
    return 0.5 * (torch.log(var) + ((y_true - mean) ** 2) / var).mean()


def build_sequences(data: np.ndarray, seq_len: int):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    x = np.asarray(x, dtype=np.float32)[:, :, None]
    y = np.asarray(y, dtype=np.float32)[:, None]
    return x, y


def estimate_uncertainty(prices: np.ndarray, seq_len: int = 20, epochs: int = 20) -> np.ndarray:
    flat_prices = np.asarray(prices, dtype=np.float32).reshape(-1)
    returns = np.diff(np.log(np.maximum(flat_prices, 1e-8))).astype(np.float32)
    x, y = build_sequences(returns, seq_len=seq_len)
    xt = torch.tensor(x)
    yt = torch.tensor(y)

    model = ProbabilisticLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mean, log_var = model(xt)
        loss = gaussian_nll(yt, mean, log_var)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, log_var = model(xt)
        std = torch.exp(0.5 * log_var).squeeze(-1).numpy()

    padded = np.zeros(len(prices), dtype=np.float32)
    values = np.clip(std, 1e-6, None)
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    start = seq_len + 1
    padded[start : start + len(values)] = values
    if start > 0:
        padded[:start] = values[0]
    return padded


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
    uncertainty = estimate_uncertainty(prices)

    rows = []
    for seed in protocol["seeds"]:
        set_global_seed(seed)
        env_cfg = EnvConfig(
            uncertainty_stop_quantile=protocol["probabilistic_agent"]["uncertainty_quantile_stop"],
            min_trade_scale=protocol["probabilistic_agent"]["position_scale_floor"],
        )

        def _make_env():
            return StockEnv(prices=prices, uncertainty=uncertainty, cfg=env_cfg)

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
        model.learn(total_timesteps=protocol["probabilistic_agent"]["timesteps"])

        eval_env = StockEnv(prices=prices, uncertainty=uncertainty, cfg=env_cfg)
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
                "agent": protocol["probabilistic_agent"]["model_name"],
                "uncertainty": uncertainty[env_cfg.lookback : env_cfg.lookback + len(portfolio_values)],
            }
        )
        curve_df.to_csv(out_dir / f"probabilistic_curve_{run_id}_seed{seed}.csv", index=False)
        metrics = compute_metrics(portfolio_values)
        metrics["seed"] = seed
        metrics["agent"] = protocol["probabilistic_agent"]["model_name"]
        rows.append(metrics)

    json_path = out_dir / f"probabilistic_{run_id}.json"
    csv_path = out_dir / f"probabilistic_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote probabilistic results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
