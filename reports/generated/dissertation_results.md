# Dissertation Experiment Report

## Objective
Evaluate whether probabilistic uncertainty integration improves downside-risk behavior compared to baseline PPO.

## Protocol
- Config: `experiments/configs/dissertation_protocol.json`
- Baseline runner: `experiments/run_baseline.py`
- Probabilistic runner: `experiments/run_probabilistic_agent.py`

## Results Snapshot
| Agent | Final Value | Sharpe | Max Drawdown | VaR Violation Rate | Preservation Rate |
|---|---:|---:|---:|---:|---:|
| baseline_ppo | 985,463.41 | -0.4285 | 0.0209 | 0.0105 | 0.9811 |
| probabilistic_ppo | 1,618,577.16 | 0.8511 | 0.1833 | 0.0500 | 0.9965 |

## Interpretation
- Improvement in capital-preservation objective: 0.0154
- Improvement in max drawdown: -0.1624
- Decision: Probabilistic agent improves preservation over PPO baseline


## Benchmark Check
- Buy-and-hold final value: 1,520,353.38
- Buy-and-hold max drawdown: 0.2450
