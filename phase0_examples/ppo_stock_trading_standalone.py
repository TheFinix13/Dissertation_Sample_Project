"""
Phase 0.2: Standalone PPO Stock Trading Example

A self-contained PPO agent for stock trading - no FinRL dependency.
Uses: yfinance (data), stable-baselines3 (PPO), gymnasium (env).

Use this if FinRL has import/dependency issues. Same learning objective.

Usage:
    source venv/bin/activate
    python phase0_examples/ppo_stock_trading_standalone.py
"""

import os
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"
warnings.filterwarnings("ignore")


# ============== Simple Stock Trading Environment ==============
class SimpleStockTradingEnv(gym.Env):
    """
    Single-stock trading env: hold/sell/buy based on price and returns.
    State: normalized prices, returns, position.
    Action: [-1, 1] = sell to hold / hold to buy.
    """

    def __init__(self, prices: np.ndarray, lookback=10, initial_balance=1e6):
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float32).ravel()
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.n_steps = len(prices) - lookback - 1

        # State: last `lookback` returns + current position (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback + 1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.entry_price = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        s = self.step_idx + self.lookback
        if s + 1 >= len(self.prices):
            returns = np.zeros(self.lookback, dtype=np.float32)
        else:
            rets = np.diff(np.log(self.prices[s - self.lookback : s + 2]))
            returns = rets[: self.lookback].astype(np.float32)
        position = np.array([self.shares * self.prices[s] / (self.balance + 1e-8)], dtype=np.float32)
        return np.concatenate([returns, position])

    def step(self, action):
        s = self.step_idx + self.lookback
        price = self.prices[s]
        next_price = self.prices[s + 1] if s + 1 < len(self.prices) else price

        # Action: positive = buy, negative = sell
        trade_pct = float(np.clip(action[0], -1, 1))
        trade_value = self.balance * 0.1 * trade_pct  # max 10% of balance per step

        if trade_value > 0:  # Buy
            new_shares = trade_value / price
            self.shares += new_shares
            self.balance -= trade_value
        else:  # Sell
            sell_value = min(-trade_value, self.shares * price)
            self.shares -= sell_value / price
            self.balance += sell_value

        self.step_idx += 1
        portfolio_value = self.balance + self.shares * next_price
        reward = np.log(portfolio_value / self.initial_balance) * 100  # scale
        terminated = self.step_idx >= self.n_steps - 1
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}


# ============== Data & Training ==============
def generate_synthetic_prices(n_days=500, seed=42):
    """Generate synthetic price series for offline/testing use."""
    np.random.seed(seed)
    returns = 0.0003 + 0.01 * np.random.randn(n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices.astype(np.float32)


def fetch_data(ticker="AAPL", start="2019-01-01", end="2021-12-31"):
    """Download OHLCV from Yahoo Finance. Falls back to synthetic if unavailable."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient data")
        prices = np.asarray(df["Close"].values, dtype=np.float32).ravel()
        return prices
    except Exception as e:
        print(f"  (Yahoo Finance unavailable: {e})")
        print("  Using synthetic price data instead.")
        return generate_synthetic_prices(500)


def make_env(prices):
    def _init():
        return SimpleStockTradingEnv(prices, lookback=10)

    return _init


def main():
    print("=" * 60)
    print("Standalone PPO Stock Trading Example")
    print("=" * 60)

    prices = fetch_data("AAPL", "2019-01-01", "2021-12-31")
    print(f"Loaded {len(prices)} days of price data")

    env = DummyVecEnv([make_env(prices)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        verbose=1,
    )

    print("\nTraining PPO for 5,000 steps...")
    model.learn(total_timesteps=5_000)
    model.save("trained_models/ppo_standalone_demo")

    print("\n" + "=" * 60)
    print("Done! Model saved to trained_models/ppo_standalone_demo")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)
    main()
