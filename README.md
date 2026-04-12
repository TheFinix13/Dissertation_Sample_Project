# Probabilistic DRL for Portfolio Risk Analysis

Phase 0 setup for the EEEM004 research project.

## Quick Start

```bash
cd portfolio-risk-drl
source venv/bin/activate

# 1. Run PPO stock trading example (Stable-Baselines3)
python phase0_examples/ppo_stock_trading_standalone.py

# 2. Run DeepAR-style probabilistic forecasting example
python phase0_examples/deepar_style_example.py
```

## Project Structure

```
portfolio-risk-drl/
├── venv/                    # Python virtual environment
├── phase0_examples/
│   ├── ppo_stock_trading_standalone.py   # PPO agent (SB3) - works
│   ├── deepar_style_example.py           # Probabilistic LSTM - works
│   └── finrl_ppo_example.py              # FinRL+PPO (needs extra deps)
├── trained_models/          # Saved PPO models
├── requirements.txt
└── README.md
```

## Phase 0 Status

| Step | Status | Notes |
|------|--------|-------|
| 0.1 Virtual env + packages | ✅ | pandas, numpy, SB3, PyTorch, yfinance |
| 0.2 PPO on sample data | ✅ | Use `ppo_stock_trading_standalone.py` |
| 0.3 DeepAR-style example | ✅ | Use `deepar_style_example.py` |

### GluonTS / FinRL Notes

- **GluonTS**: Requires `scipy<1.16` which has no pre-built wheel for Python 3.14. Use Python 3.10–3.11 and `pip install gluonts[torch]` for full GluonTS DeepAR.
- **FinRL**: Has many optional dependencies (alpaca, wrds, elegantrl). The standalone PPO example achieves the same learning objective without FinRL.

## Next Steps (Phase 1)

1. Integrate DeepAR-style outputs (mean, variance) into PPO state
2. Add risk thresholds based on predicted uncertainty
3. Benchmark on S&P 500 / NASDAQ 100 (2009–2025)

## Phase 1 Experiment Pipeline

```bash
source venv/bin/activate

# 1) Baseline PPO with deterministic seeds
python experiments/run_baseline.py

# 2) Probabilistic DeepAR-style uncertainty + PPO
python experiments/run_probabilistic_agent.py

# 3) Benchmarks for sanity checks
python experiments/run_benchmarks.py

# 4) Generate dissertation-ready summary report
python reports/generate_dissertation_report.py

# 5) Build supervisor pack (progress report + chart)
python reports/build_supervisor_pack.py

# 6) Build richer visualizations (equity curves, uncertainty, dataset, intraday)
python reports/plot_dissertation_visuals.py
```

- Protocol config: `experiments/configs/dissertation_protocol.json`
- Artifacts: `experiments/results/`
- Report output: `reports/generated/dissertation_results.md`
