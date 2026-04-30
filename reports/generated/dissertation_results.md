# Dissertation Experiment Report

## Objective
Evaluate whether probabilistic uncertainty integration improves downside-risk behavior compared to baseline PPO.

## Protocol
- Config: `experiments/configs/dissertation_protocol.json`
- Baseline runner: `experiments/runners/run_baseline.py`
- Probabilistic runner: `experiments/runners/run_probabilistic_agent.py`

## Results Snapshot
| Agent | Final Value | Sharpe | Max Drawdown | VaR Violation Rate | Preservation Rate |
|---|---:|---:|---:|---:|---:|
| baseline_ppo | 989,429.52 | -0.2288 | 0.0327 | 0.0144 | 0.9728 |
| probabilistic_ppo | 1,462,518.31 | 0.4803 | 0.1985 | 0.0499 | 0.9738 |

## Interpretation
- Improvement in capital-preservation objective: 0.0010
- Improvement in max drawdown: -0.1657
- Decision: Probabilistic agent improves preservation over PPO baseline


## Benchmark Check
- Buy-and-hold final value: 1,369,337.75
- Buy-and-hold max drawdown: 0.2526
