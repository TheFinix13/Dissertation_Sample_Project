import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from gymnasium import spaces


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_protocol(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_close_prices(ticker: str, start: str, end: str) -> np.ndarray:
    df = fetch_close_frame(ticker=ticker, start=start, end=end)
    return np.asarray(df["Close"].values, dtype=np.float32).ravel()


def fetch_close_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} [{start} -> {end}]")
    return df


def compute_metrics(portfolio_values: list[float], risk_free_rate_daily: float = 0.0) -> dict:
    pv = np.asarray(portfolio_values, dtype=np.float64)
    returns = np.diff(np.log(np.maximum(pv, 1e-8)))
    if len(returns) == 0:
        returns = np.array([0.0])

    final_portfolio_value = float(pv[-1])
    annualized_return = float(np.expm1(np.mean(returns) * 252))
    annualized_volatility = float(np.std(returns) * np.sqrt(252))
    sharpe = float(
        ((np.mean(returns) - risk_free_rate_daily) / (np.std(returns) + 1e-8))
        * np.sqrt(252)
    )

    running_max = np.maximum.accumulate(pv)
    drawdowns = 1.0 - (pv / np.maximum(running_max, 1e-8))
    max_drawdown = float(np.max(drawdowns))
    hwm = float(np.max(pv))
    preservation_rate = float(final_portfolio_value / hwm) if hwm > 0 else 0.0

    var_95 = float(np.quantile(returns, 0.05))
    var_violations = float(np.mean(returns < var_95))

    return {
        "final_portfolio_value": final_portfolio_value,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_95_violation_rate": var_violations,
        "capital_preservation_rate_95pct_hwm": preservation_rate,
        "meets_95pct_preservation_goal": bool(preservation_rate >= 0.95),
    }


@dataclass
class EnvConfig:
    lookback: int = 20
    initial_balance: float = 1_000_000.0
    max_trade_fraction: float = 0.10
    transaction_cost_rate: float = 0.001
    uncertainty_stop_quantile: float = 0.80
    min_trade_scale: float = 0.10


class StockEnv(gym.Env):
    def __init__(
        self,
        prices: np.ndarray,
        uncertainty: np.ndarray | None = None,
        cfg: EnvConfig = EnvConfig(),
    ):
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float32).ravel()
        self.uncertainty = (
            np.asarray(uncertainty, dtype=np.float32).ravel()
            if uncertainty is not None
            else np.zeros_like(self.prices, dtype=np.float32)
        )
        self.cfg = cfg
        self.n_steps = len(self.prices) - cfg.lookback - 1
        self.uncertainty_threshold = float(
            np.quantile(self.uncertainty, self.cfg.uncertainty_stop_quantile)
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.lookback + 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        start_price = float(self.prices[self.cfg.lookback])
        starting_equity = float(self.cfg.initial_balance)
        initial_invested = starting_equity * 0.5
        self.shares = initial_invested / max(start_price, 1e-8)
        self.balance = starting_equity - initial_invested
        self.portfolio_values = [self.cfg.initial_balance]
        self.trade_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        s = self.step_idx + self.cfg.lookback
        rets = np.diff(np.log(self.prices[s - self.cfg.lookback : s + 1]))
        rets = rets.astype(np.float32)
        position = np.array(
            [self.shares * self.prices[s] / (self.balance + 1e-8)], dtype=np.float32
        )
        uncertainty = np.array([self.uncertainty[s]], dtype=np.float32)
        return np.concatenate([rets, position, uncertainty]).astype(np.float32)

    def step(self, action):
        s = self.step_idx + self.cfg.lookback
        price = float(self.prices[s])
        next_price = float(self.prices[s + 1]) if s + 1 < len(self.prices) else price

        uncertainty_level = float(self.uncertainty[s])
        trade_scale = 1.0 - uncertainty_level
        trade_scale = max(trade_scale, self.cfg.min_trade_scale)

        trade_pct = float(np.clip(action[0], -1, 1))
        trade_value = self.balance * self.cfg.max_trade_fraction * trade_pct * trade_scale
        if uncertainty_level >= self.uncertainty_threshold and trade_value > 0:
            # High uncertainty regime: block new risk-on buys.
            trade_value = 0.0

        if trade_value > 0:
            fee = abs(trade_value) * self.cfg.transaction_cost_rate
            new_shares = trade_value / max(price, 1e-6)
            self.shares += new_shares
            self.balance -= trade_value + fee
            self.trade_count += 1
        else:
            sell_value = min(-trade_value, self.shares * price)
            fee = abs(sell_value) * self.cfg.transaction_cost_rate
            self.shares -= sell_value / max(price, 1e-6)
            self.balance += max(sell_value - fee, 0.0)
            if sell_value > 0:
                self.trade_count += 1

        self.step_idx += 1
        portfolio_value = self.balance + self.shares * next_price
        self.portfolio_values.append(portfolio_value)
        prev_portfolio_value = self.portfolio_values[-2]
        reward = math.log(
            max(portfolio_value, 1e-8) / max(prev_portfolio_value, 1e-8)
        ) * 100
        terminated = self.step_idx >= self.n_steps - 1
        return self._get_obs(), reward, terminated, False, {}
