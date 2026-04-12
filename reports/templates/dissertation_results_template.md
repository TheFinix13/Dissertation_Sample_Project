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
| baseline_ppo | {{baseline_final_value}} | {{baseline_sharpe}} | {{baseline_max_drawdown}} | {{baseline_var_violations}} | {{baseline_preservation}} |
| probabilistic_ppo | {{prob_final_value}} | {{prob_sharpe}} | {{prob_max_drawdown}} | {{prob_var_violations}} | {{prob_preservation}} |

## Interpretation
- Improvement in capital-preservation objective: {{preservation_delta}}
- Improvement in max drawdown: {{drawdown_delta}}
- Decision: {{decision}}
