"""
Phase 0.2: FinRL PPO Example
Run a standard PPO agent on sample stock data using FinRL + Stable-Baselines3.

Uses import workaround to avoid FinRL's alpaca dependency for training-only use.

Usage:
    source venv/bin/activate
    python phase0_examples/finrl_ppo_example.py
"""

import os
import sys
import warnings

# Workaround: stub finrl.trade before any finrl import to avoid alpaca_trade_api
import importlib.util
import types
_spec = importlib.util.find_spec("finrl")
_finrl = types.ModuleType("finrl")
_finrl.__path__ = _spec.submodule_search_locations
_finrl_trade = types.ModuleType("finrl.trade")
_finrl_trade.trade = lambda *a, **k: None
sys.modules["finrl"] = _finrl
sys.modules["finrl.trade"] = _finrl_trade

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"
warnings.filterwarnings("ignore", category=UserWarning)

# Now safe to import FinRL components
from finrl.config import INDICATORS, PPO_PARAMS
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.train import train

# Small subset for fast demo
TICKERS = ["AAPL", "MSFT", "GOOGL"]
TRAIN_START = "2019-01-01"
TRAIN_END = "2020-12-31"
OUTPUT_DIR = "trained_models/ppo_demo"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FinRL PPO Example - Stock Trading with Reinforcement Learning")
    print("=" * 60)
    print(f"Tickers: {TICKERS}")
    print(f"Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    train(
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        ticker_list=TICKERS,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=StockTradingEnv,
        model_name="ppo",
        cwd=OUTPUT_DIR,
        agent_params=PPO_PARAMS,
        total_timesteps=10_000,
        if_vix=False,
        kwargs={},
    )

    print("\n" + "=" * 60)
    print("Training complete! Model saved to", OUTPUT_DIR)
    print("=" * 60)
