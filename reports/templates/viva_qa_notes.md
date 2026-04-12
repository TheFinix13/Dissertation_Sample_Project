# Viva Q&A Notes (Draft)

## 1) Why probabilistic RL instead of standard PPO?
- Standard PPO uses point estimates and can over-trust noisy signals.
- Probabilistic layer estimates uncertainty; high uncertainty triggers safer behavior.

## 2) How is uncertainty measured?
- Forecast model outputs mean and variance for next-step return.
- Variance (or standard deviation) is normalized into an uncertainty score.
- This score is injected into the RL state and used for risk guard decisions.

## 3) What is the main hypothesis?
- Adding uncertainty-aware signals and guardrails improves capital preservation relative to baseline PPO.

## 4) What are your baselines?
- PPO baseline (same environment, no uncertainty signal/guard).
- Buy-and-hold benchmark.
- All-cash benchmark.

## 5) What are your key metrics?
- Final portfolio value
- Max drawdown
- Sharpe ratio
- VaR violation rate
- Capital preservation ratio relative to high-watermark (target >= 0.95)

## 6) What are the limitations?
- Daily granularity and single-source market data in current phase.
- Limited assets in initial runs.
- Further robustness checks needed for multi-event and multi-sector generalization.

## 7) Why not full live trading?
- Dissertation scope prioritizes scientific evaluation and reproducibility over production execution.

## 8) How do you validate reproducibility?
- Fixed seeds and scripted runs.
- Artifact generation in `experiments/results` and `reports/generated`.
