import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def close_1d(price_df: pd.DataFrame) -> pd.Series:
    """Return the Close column as a 1-D float32 Series even when yfinance gives MultiIndex columns."""
    close = price_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype("float32")


# --------------------------------------------------------------------------- #
# CLI / protocol helpers
# --------------------------------------------------------------------------- #
def _parse_tickers(value: str | None, protocol: dict) -> list[str]:
    """Resolve --tickers into a concrete list.

    Accepted values:
        - None or "" : use the legacy single ticker (protocol.data.tickers[0]).
        - "basket"   : use protocol.data.ticker_basket (the eight-ticker formal basket).
        - "all"      : alias for 'basket'.
        - <named_group> : any key under protocol.data.named_groups, e.g.
          "fiyins_portfolio", "fiyins_stocks", "fiyins_etfs".
          Case-insensitive.
        - comma-separated list, e.g. "SPY,QQQ,XLK" : use exactly those tickers.
    """
    if value is None or value.strip() == "":
        return [protocol["data"]["tickers"][0]]
    key = value.strip().lower()
    if key in {"basket", "all"}:
        basket = protocol["data"].get("ticker_basket")
        if not basket:
            raise ValueError("Protocol has no 'data.ticker_basket' field.")
        return list(basket)
    named_groups = protocol["data"].get("named_groups", {}) or {}
    named_groups_lower = {k.lower(): v for k, v in named_groups.items()}
    if key in named_groups_lower:
        return list(named_groups_lower[key])
    return [t.strip().upper() for t in value.split(",") if t.strip()]


def _parse_seeds(value: str | None, protocol: dict) -> list[int]:
    """Resolve --seeds into a concrete list of integers."""
    if value is None or value.strip() == "":
        return list(protocol["seeds"])
    if value.strip().lower() == "default":
        return list(protocol["seeds"])
    if value.strip().lower() == "extended":
        ext = protocol.get("seeds_extended")
        if not ext:
            raise ValueError("Protocol has no 'seeds_extended' field.")
        return list(ext)
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def _parse_folds(value: str | None, protocol: dict) -> list[dict]:
    """Resolve --folds into a concrete list of fold dicts.

    Each fold dict has shape:
        {"fold_id": str, "train": [start, end], "test": [start, end], "notes": str}

    Accepted values:
        - None or "" or "test" : single fold spanning the legacy splits.test window
          (training and test ranges are taken from protocol.splits).
        - "all" : every entry of protocol.walk_forward_folds.
        - comma-separated list of fold_ids, e.g. "wf_2022_2023,wf_2024_2025".
    """
    if value is None or value.strip() == "" or value.strip().lower() == "test":
        return [{
            "fold_id": "test",
            "train": list(protocol["splits"]["train"]),
            "test":  list(protocol["splits"]["test"]),
            "notes": "Legacy single-fold test window from protocol.splits.test.",
        }]
    folds = protocol.get("walk_forward_folds", [])
    by_id = {f["fold_id"]: f for f in folds}
    if value.strip().lower() == "all":
        return list(folds)
    requested = [s.strip() for s in value.split(",") if s.strip()]
    missing = [r for r in requested if r not in by_id]
    if missing:
        raise ValueError(f"Unknown walk-forward fold ids: {missing}. "
                         f"Available: {list(by_id.keys())}")
    return [by_id[r] for r in requested]


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    """Standard CLI flags shared by every runner."""
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated tickers, or 'basket' for the full multi-asset basket. "
             "Default: legacy single ticker (protocol.data.tickers[0]).",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seeds, or 'default' / 'extended' for the protocol lists.",
    )
    parser.add_argument(
        "--folds", default=None,
        help="Comma-separated walk-forward fold ids, or 'test' / 'all'. "
             "Default: legacy single test window.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="PPO training timesteps. Default: legacy 'baseline.timesteps' / 'probabilistic_agent.timesteps' from the protocol.",
    )
    parser.add_argument(
        "--initial-balance", type=float, default=None,
        help="Starting capital in USD. Default: protocol.initial_balance or $1,000,000.",
    )
    parser.add_argument(
        "--bootstrap-paths", type=int, default=0,
        help="Number of synthetic block-bootstrapped training paths to concatenate "
             "to the real training-window prices (0 = disabled).",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional tag suffix appended to output filenames "
             "(useful for distinguishing 10k vs 50k runs etc.).",
    )


def make_run_id(tag: str | None = None) -> str:
    """UTC timestamp with optional tag suffix, suitable for output filenames."""
    from datetime import UTC, datetime
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{tag}" if tag else stamp


def resolve_initial_balance(args: argparse.Namespace, protocol: dict) -> float:
    if getattr(args, "initial_balance", None) is not None:
        return float(args.initial_balance)
    return float(protocol.get("initial_balance", 1_000_000.0))


# Public aliases so runner code reads cleanly.
resolve_tickers = _parse_tickers
resolve_seeds = _parse_seeds
resolve_folds = _parse_folds


# --------------------------------------------------------------------------- #
# Block-bootstrap training-data augmentation
# (Politis & Romano, 1994 — stationary block bootstrap.)
# --------------------------------------------------------------------------- #
def stationary_block_bootstrap(
    series: np.ndarray,
    *,
    expected_block_length: float,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one bootstrap path of `length` samples from `series`.

    The stationary block bootstrap (Politis & Romano, 1994) preserves
    short-range temporal structure by sampling overlapping blocks of
    geometrically distributed length. With expected block length L the
    per-step probability of starting a new block is p = 1/L.
    """
    n = len(series)
    if n == 0:
        raise ValueError("Cannot bootstrap from empty series.")
    p = 1.0 / max(expected_block_length, 1.0)
    out = np.empty(length, dtype=series.dtype)
    idx = int(rng.integers(0, n))
    for t in range(length):
        out[t] = series[idx]
        if rng.random() < p:
            idx = int(rng.integers(0, n))
        else:
            idx = (idx + 1) % n
    return out


def synthesize_bootstrap_prices(
    real_prices: np.ndarray,
    *,
    num_paths: int,
    expected_block_length: float,
    seed: int,
) -> np.ndarray:
    """Concatenate the real price path with `num_paths` bootstrap paths.

    Bootstraps the *log returns* of `real_prices` (preserving the empirical
    return distribution and short-range autocorrelation), then re-integrates
    into a price path that starts from the real path's last close. This is
    the recommended way to expand a training set without leaving the
    empirical distribution.
    """
    if num_paths <= 0:
        return np.asarray(real_prices, dtype=np.float32).ravel()

    real = np.asarray(real_prices, dtype=np.float32).ravel()
    rng = np.random.default_rng(int(seed))

    log_returns = np.diff(np.log(np.maximum(real, 1e-8)))
    last_real = float(real[-1])

    paths = [real]
    for _ in range(num_paths):
        sampled_returns = stationary_block_bootstrap(
            log_returns,
            expected_block_length=expected_block_length,
            length=len(log_returns),
            rng=rng,
        )
        levels = np.empty(len(sampled_returns) + 1, dtype=np.float32)
        levels[0] = last_real
        levels[1:] = last_real * np.exp(np.cumsum(sampled_returns))
        last_real = float(levels[-1])
        paths.append(levels)

    return np.concatenate(paths).astype(np.float32)


def maybe_bootstrap_training_prices(
    real_prices: np.ndarray,
    *,
    num_paths: int,
    protocol: dict,
    seed: int,
) -> np.ndarray:
    """Apply bootstrap augmentation if `num_paths > 0`, else passthrough."""
    if num_paths <= 0:
        return real_prices
    cfg = protocol.get("bootstrap", {})
    block_len = float(cfg.get("expected_block_length", 20))
    return synthesize_bootstrap_prices(
        real_prices,
        num_paths=num_paths,
        expected_block_length=block_len,
        seed=seed,
    )


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
