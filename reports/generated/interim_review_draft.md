# EEEM004 Interim Review — Draft Content

> Use this file to fill the **black** boxes in `InterimReview.docx`.
> The blue supervisor boxes are intentionally left empty.

---

## Cover information

- **Name:** Fiyin Akano
- **URN:** 6962514
- **Supervisor:** Dr. Cuong Nguyen
- **Second supervisor (if applicable):** [FILL IN]
- **Date of meeting:** [FILL IN]

---

## Project title

**Probabilistic Deep Reinforcement Learning for Portfolio Risk Analysis:
drawdown-constrained portfolio control with an uncertainty-aware
reinforcement-learning policy.**

---

## Problem statement

Many investors and institutional mandates are required to keep their
portfolio's loss from peak (the maximum drawdown) below a stated floor — a
single-digit percentage in many institutional settings — while still earning
a return that beats holding cash and ideally beats a passive index. The
standard answers to that requirement are static convex optimisations
(mean–variance, risk parity) or fixed-rule overlays (stop losses, volatility
targeting). Neither answer adapts to within-window regime change, and
neither consumes a model's own confidence in its forecast. The dissertation
asks whether a Proximal Policy Optimization (PPO) agent that conditions on
the predictive uncertainty produced by a DeepAR-style probabilistic
recurrent network can sit on a more attractive point of the
return-versus-drawdown trade-off than three named alternatives: passive
buy-and-hold, a rule-based stop-loss policy of the kind a discretionary
investor would actually use, and a baseline PPO that sees no uncertainty
signal. The headline result is the joint of risk-adjusted return and
preservation against the running high-watermark; meeting either half on its
own is trivial (cash gives perfect preservation and zero return), so the
design is judged on whether it satisfies both at once on a test window
containing real macro shocks.

---

## Objectives (latest version)

The objectives have been refined during Phase 0 and Phase 1 in light of
supervisor feedback. They are stated in the order in which the dissertation
answers them: the core scientific question first, the empirical evidence
(both single-asset headline and multi-asset / out-of-time generalisation)
second, the reproducibility apparatus third, and the honest position on
where the method works and where it does not fourth.

- **O1 — the core scientific question.** Can a deep reinforcement-learning
  agent that conditions on its own forecaster's predictive uncertainty (how
  confident the forecaster is, not just what it predicts) sit closer to the
  drawdown-constrained risk-adjusted return frontier than uncertainty-blind
  alternatives? Operationally: a DeepAR-style probabilistic LSTM emits
  predictive mean and variance; a Proximal Policy Optimization (PPO) policy
  reads the variance as a state feature and as a hard guard that blocks new
  long-side trades when the uncertainty score exceeds a quantile threshold.
- **O2 — the empirical evidence.** Evaluate the resulting policy on a fixed
  held-out window containing real macro shocks (2022 to 2025) against three
  named comparators — passive buy-and-hold, a rule-based trailing stop-loss
  policy, and a baseline PPO with no uncertainty signal — and check that the
  conclusions survive contact with (a) a 70-ticker diversified-equity test
  universe (41 single-name US large-cap equities + 29 ETFs spanning
  broad-market, sector, dividend, thematic and commodity exposure) and (b)
  a four-fold walk-forward grid in which the train, validation and test
  windows roll forward across 2018–2025. Headline metrics: Sharpe ratio,
  terminal value relative to buy-and-hold, and the capital-preservation
  ratio against the running high-watermark.
- **O3 — reproducibility.** Pin down a fully reproducible evaluation
  protocol of fixed splits, fixed random seeds, scripted experiment
  runners, scripted reporting and a shared metric set, so that any
  comparison made in this dissertation is genuinely like-for-like and can
  be reproduced from the public repository in a single command sequence.
- **O4 — honest position on where it works and where it does not.**
  Diagnose the regimes in which the uncertainty-aware policy beats the
  alternatives and the regimes in which it does not, and take a defensible
  position — on the strength of O1–O3 — on when an explicit uncertainty
  signal earns a place in a portfolio control loop and, just as important,
  on when it does not.

---

## Literature Review (key references)

The list below is the working set of references I keep coming back to. It is
intentionally compact (around ten entries) per the form guidance; the full
bibliography in the dissertation will be longer.

1. **Schulman, Wolski, Dhariwal, Radford, Klimov (2017) — "Proximal Policy
   Optimization Algorithms."** *arXiv:1707.06347.*
   PPO is the policy-gradient algorithm I use for both the baseline and the
   uncertainty-aware variant. The clipped surrogate is the reason I picked
   PPO over vanilla policy gradient or TRPO: it is forgiving with respect to
   hyper-parameters, which matters on a single-asset environment with limited
   training budget.

2. **Sutton & Barto (2018) — *Reinforcement Learning: an Introduction*,
   2nd ed., MIT Press.**
   The standard reference for the MDP formulation and the policy-gradient
   derivations I rely on in the methodology chapter.

3. **Salinas, Flunkert, Gasthaus, Januschowski (2020) — "DeepAR:
   Probabilistic Forecasting with Autoregressive Recurrent Networks."**
   *International Journal of Forecasting, 36(3), 1181–1191.*
   The blueprint for my forecaster: an RNN that emits the parameters of a
   predictive distribution and is trained by negative log-likelihood. My
   implementation in `experiments/runners/run_probabilistic_agent.py` is a stripped-
   down DeepAR with a Gaussian head emitting (μ, log σ²).

4. **Hochreiter & Schmidhuber (1997) — "Long Short-Term Memory."** *Neural
   Computation, 9(8), 1735–1780.*
   The architectural workhorse inside the forecaster.

5. **Lakshminarayanan, Pritzel, Blundell (2017) — "Simple and Scalable
   Predictive Uncertainty Estimation using Deep Ensembles."** *NeurIPS.*
   I read this to decide whether to go with deep ensembles or a single
   network with a Gaussian head. The single-head approach is cheaper and
   sufficient given that the policy only needs a one-dimensional uncertainty
   summary; ensembles are noted as future work.

6. **Gal & Ghahramani (2016) — "Dropout as a Bayesian Approximation:
   Representing Model Uncertainty in Deep Learning."** *ICML.*
   Background reading on why a learned predictive variance can be treated as
   a meaningful uncertainty proxy at all.

7. **Jiang, Xu, Liang (2017) — "A Deep Reinforcement Learning Framework
   for the Financial Portfolio Management Problem."** *arXiv:1706.10059.*
   The closest prior work to mine in spirit. They use RL for portfolio
   selection but optimise for return; my contribution is on the *risk* side,
   so I cite them to position what is and isn't new in my approach.

8. **Yang, Liu, Zhong, Walid (2020) — "Deep Reinforcement Learning for
   Automated Stock Trading: An Ensemble Strategy."** *ICAIF.*
   I borrowed the train / validation / test split style and the seed-
   averaging convention from this paper.

9. **Liu, Yang, Gao, Wang (2021) — "FinRL: a deep reinforcement learning
   library for automated stock trading in quantitative finance."**
   *arXiv:2011.09607.*
   The reference design for finance-RL environments. I deliberately wrote my
   own `StockEnv` from scratch (in `experiments/common.py`) rather than
   subclassing FinRL, because I wanted to be able to explain every line in
   the viva. FinRL is cited to motivate the shape of the environment.

10. **Raffin, Hill, Gleave, Kanervisto, Ernestus, Dormann (2021) —
    "Stable-Baselines3: Reliable Reinforcement Learning Implementations."**
    *Journal of Machine Learning Research, 22(268), 1–8.*
    The implementation source for the PPO solver in both runners.

11. **Markowitz (1952) — "Portfolio Selection."** *Journal of Finance,
    7(1), 77–91.*
    Cited to frame why I picked capital preservation rather than mean–
    variance as the objective. Mean–variance is the conceptual baseline
    every risk-aware portfolio paper compares against, and it is worth
    being explicit about why I am not using it directly.

---

## Technical progress

### Summary

Phase 0 and Phase 1 are in place and reproducible end-to-end. The dissertation
compares **baseline PPO** against a **probabilistic-PPO** variant that
consumes a DeepAR-style uncertainty signal, with **buy-and-hold** and
**all-cash** as benchmarks, all on `SPY` for now (`QQQ` is queued for the
multi-ticker robustness study in Phase 2). Everything runs from a single
config file and a small set of scripts.

### What has been built

- A **probabilistic forecaster** (`experiments/runners/run_probabilistic_agent.py`):
  an LSTM trained with Gaussian NLL that emits the mean and log variance of
  the next-step log return. The predictive standard deviation is min-max
  normalised across the test window into a unit-interval uncertainty score.
- An **uncertainty-aware trading environment**
  (`experiments/common.py:StockEnv`):
  - Action space `[-1, 1]` over a configurable `max_trade_fraction` of cash,
    with the trade size shrunk by `(1 - uncertainty_level)` and floored at
    `min_trade_scale` so the agent is never silenced entirely.
  - When the uncertainty signal sits above the protocol quantile (default
    `0.80`) the environment **blocks new long-side trades** but still allows
    exits.
  - Reward is the per-step log of the portfolio-value ratio, multiplied by
    100 for numerical scale. This rewards compounding and penalises
    drawdowns automatically, without an extra term.
- A **baseline PPO runner** (`experiments/runners/run_baseline.py`) that uses the
  same environment without the uncertainty coordinate or the trade-size
  shrinkage, so the comparison is genuinely controlled.
- A **benchmarks runner** (`experiments/runners/run_benchmarks.py`) that evaluates
  buy-and-hold and all-cash on the same test window. These act as sanity
  checks on the metric definitions as much as competitors to beat.
- A single **evaluation protocol**
  (`experiments/configs/dissertation_protocol.json`) that fixes the splits
  (2009–2018 train / 2019–2021 validation / 2022–2025 test), the seeds
  (`[7, 19, 42]`) and the metric set, and is read by every script. This is
  the bit that actually makes the comparisons fair.
- A **reporting layer**: `reports/builders/generate_dissertation_report.py` produces
  the markdown summary, `reports/builders/build_supervisor_pack.py` produces the
  one-page chart, and `reports/builders/plot_dissertation_visuals.py` produces the
  detailed figures. There is also a `Dissertation_Walkthrough.ipynb` that
  re-runs the whole pipeline and renders the embedded outputs for review.

### Phase-0 → Phase-1 status table

| Step | Status | Notes |
|------|--------|-------|
| 0.1 Environment + dependencies | Done | `requirements.txt`, SB3, PyTorch, gymnasium, yfinance |
| 0.2 PPO baseline on sample data | Done | `phase0_examples/ppo_stock_trading_standalone.py` |
| 0.3 DeepAR-style probabilistic example | Done | `phase0_examples/deepar_style_example.py` |
| 1.1 Shared protocol + metrics | Done | `experiments/configs/dissertation_protocol.json`, `experiments/common.py` |
| 1.2 Reproducible baseline / probabilistic / benchmark runners | Done | three runners, seeded |
| 1.3 Dissertation report + supervisor pack | Done | `reports/generated/` |
| 1.4 Rule-based stop-loss comparator (5 % and 10 % variants) | Done | `experiments/runners/run_rule_baselines.py` |
| 1.5 Robustness (multi-ticker, walk-forward, ablations, shock windows) | In progress | Scheduled May–August (see future plan) |

### Current results (mean across 3 seeds, test window 2022–2025)

| Agent | Final value (USD) | Sharpe | Max drawdown | VaR-95 violation | Terminal preservation vs HWM | Path preservation (1 − MDD) |
|---|---:|---:|---:|---:|---:|---:|
| baseline PPO              | 985,463    | −0.4285 | 0.0209 | 0.0105 | 0.9811 | 0.9791 |
| **probabilistic PPO**     | **1,618,577** | **0.8511** | **0.1833** | **0.0500** | **0.9965** | **0.8167** |
| rule-based stop-loss (5%) | 1,233,203  | 0.4237  | 0.2523 | 0.0500 | 0.9905 | 0.7477 |
| rule-based stop-loss (10%)| 1,241,164  | 0.4085  | 0.2995 | 0.0500 | 0.9951 | 0.7005 |
| buy-and-hold (SPY)        | 1,520,353  | 0.5867  | 0.2450 | 0.0500 | 0.9951 | 0.7550 |
| all-cash                  | 1,000,000  | 0.0000  | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Reference figure: `reports/generated/charts/final_value_comparison.png`.
Equity curves and the uncertainty signal are in
`equity_curve_comparison.png` and `uncertainty_signal.png` respectively.
The two new rule-based comparators (5 % and 10 % trailing stop-losses with a
20/50-day moving-average crossover for re-entry) live in
`experiments/runners/run_rule_baselines.py`.

### 70-ticker test universe — Phase-1 robustness (Section 5.5)

The Phase-1 robustness study runs the same four-agent comparison on a 70-ticker
diversified-equity test universe — 41 single-name US large-cap equities (technology,
payments and financial services, healthcare, consumer, industrials) and 29
exchange-traded funds (broad-market indices, sector SPDRs, dividend ETFs,
thematic exposures and commodity funds) — on the same 2022–2025 test window
with the same metric definitions and the same four-agent comparison set.

| strategy | mean terminal value | mean Sharpe | mean Max-DD |
|---|---:|---:|---:|
| Baseline PPO (no uncertainty) | $989,430 | −0.23 | 0.033 |
| **Probabilistic PPO (this work)** | **$1,998,817** | **+0.60** | **0.225** |
| Manual 5 % trailing stop | $1,531,163 | +0.36 | 0.305 |
| Passive buy-and-hold | $2,099,838 | +0.54 | 0.370 |

Headline findings on the 70-ticker test universe:

- **Drawdown reduced versus passive buy-and-hold on 70 of 70 tickers (100 % of the universe)**, with an average reduction of 14.5 percentage points (mean drawdown cut from 37 % to 22.5 % — a 39 % relative reduction). This is the strongest single number in the dissertation.
- **Probabilistic agent beat the manually-tuned 5 % trailing stop on 61 of 70 tickers (87 %)** in terminal value, and on essentially every ticker in Sharpe ratio — direct empirical answer to the previous-meeting question on whether the AI agent beats a manually-tuned stop-loss alternative.
- Cost in mean terminal value vs buy-and-hold: ≈ 5 % give-up in mean upside in exchange for ≈ 39 % reduction in mean drawdown. The trade institutional risk officers run.
- Where the agent loses (45 of 70 tickers in terminal value, all winning on drawdown), the losses cluster in two diagnosable regimes: persistent low-uncertainty bull-market trends in single names (NVDA, AVGO, LLY) and very-low-drawdown defensives (JNJ, MCD, SCHD, GLD). Sector-aware uncertainty-quantile calibration is the targeted Phase-2 fix.
- The full per-ticker table is in Appendix B of `Main_Dissertation_Draft.docx`. A plain-English companion narrative of the same evidence — written for a non-quantitative reader, with finance context and visualisations — lives in `Fiyins_Dissertation.docx`.

### How to read these numbers

A few things are worth flagging before this table is read in isolation:

- The headline criterion is the **joint** of Sharpe ratio and the
  capital-preservation ratio against the running high-watermark, not
  preservation alone. Meeting either half on its own is trivial: an
  all-cash policy achieves preservation 1.0 with zero return, and a
  return-only policy ignores the constraint entirely. The probabilistic
  agent meets both halves; its Sharpe and terminal value finish above
  passive buy-and-hold and its preservation ratio sits above 0.95 across
  all three seeds. The baseline meets neither half: it ends roughly where
  it started with a slightly negative Sharpe.
- The third comparator — a rule-based stop-loss policy of the kind a
  discretionary investor would actually use — has now been implemented
  (`experiments/runners/run_rule_baselines.py`). Two variants (5 % and 10 %
  trailing stop with a 20/50-day moving-average re-entry rule) are run
  on the same protocol. They sit between cash and buy-and-hold on return
  and they end with adequate terminal preservation, but they incur path
  drawdowns *larger* than buy-and-hold's: the trailing stop fires only
  after the drawdown has begun, the moving-average re-entry rule is slow,
  and the policy sits in cash through much of the post-2022 recovery.
  This is a directly measured answer to the supervisor's "AI beats manual
  stop-losses" question. The probabilistic agent earns roughly $380,000
  more than the better rule-based variant over the four-year window,
  with twice the Sharpe and a smaller path drawdown.
- Max drawdown on the baseline looks small only because the baseline barely
  compounds in the first place. There is little to draw down from. The
  probabilistic agent compounds to a higher peak, gives some of it back,
  and still finishes well above the baseline. Preservation against the
  high-watermark is the metric that matches the objective; max drawdown is
  reported for transparency, not as a contradicting result.
- These numbers are provisional. The plan below explicitly tests how
  fragile they are to ticker choice, time period (walk-forward), threshold
  choice, and which part of the design is doing the work (the state
  feature, the trade-size shrink, or the entry guard).

### Reproducibility

```bash
python experiments/runners/run_baseline.py
python experiments/runners/run_probabilistic_agent.py
python experiments/runners/run_benchmarks.py
python reports/builders/generate_dissertation_report.py
python reports/builders/build_supervisor_pack.py
python reports/builders/plot_dissertation_visuals.py
```

Artifacts land in `experiments/results/` and `reports/generated/`. The full
source is on GitHub at `TheFinix13/Dissertation_Sample_Project`, with the
walkthrough notebook (`Dissertation_Walkthrough.ipynb`) as the single entry
point for someone reading the project for the first time.

---

## Future plan

**Progress against the previous plan.** The May 2026 milestones in the
original plan have all been completed: the dissertation has been reframed
around drawdown-constrained risk-adjusted return; a finance and
risk-management background section has been added; the rule-based
stop-loss comparator is checked in and reported alongside the AI agents;
the test universe has been expanded from single-index SPY to a 70-ticker
diversified-equity universe; and the extended seed-stability check has
been run on a representative sub-universe. The plan below covers what
remains.

Each scheduled task is tied to a milestone with a target date. Milestones
are tied to objectives O1–O4. The submission-critical path is the
backtest and walk-forward evidence; live execution sits outside that path
and is treated as a stretch goal (see the August 2026 row).

| Working period | Tasks to undertake | Milestones to meet (with target dates) |
|---|---|---|
| **June 2026 (4 weeks)** | Phase-2 extended grid on the full 70-ticker universe at extended budget (10 seeds × 50 000 timesteps × 4 walk-forward folds × 16 bootstrap paths) on Colab GPU runtime. Sector-aware uncertainty-quantile calibration (replace the single global threshold with a per-sector or per-regime threshold). Begin Chapter 2 (Background) and Chapter 3 (Methodology) full drafts. | M1: full 70-ticker × 4-fold × 10-seed × 50k-step extended grid (mid-June). M2: sector-aware calibration ablation + Chapter 2 and Chapter 3 first drafts (end of June). |
| **July 2026 (4–6 weeks)** | Sensitivity sweep on the uncertainty quantile threshold, the minimum trade-size scale, and the maximum trade fraction. Block-bootstrap data augmentation (Politis & Romano, 1994) to expand the effective training set. Lock the final headline results table. Draft Chapter 5 (Results) and Chapter 1 (Introduction). | M3: sensitivity and bootstrap results locked (mid-July). M4: Chapters 1, 2, 3 and 5 first drafts (end of July). |
| **August 2026 (4 weeks)** | Polish phase. Write Chapter 6 (Discussion) and Chapter 7 (Conclusion). Polish figures and tables, integrate supervisor feedback on the full draft, and finalise the dissertation. Code changes from this point are bug-fix only. **Stretch goal:** if time and a working brokerage account permit, run a two-week paper-trading shadow run via the Alpaca API and report the live PnL as an out-of-sample case study; if it does not happen the dissertation rests on the backtest and walk-forward evidence and the live run is recorded as post-submission work in the real-world deployment roadmap (see the `live/` directory in the repository). | M5: full draft to supervisor (mid-August). M6: submission-ready version (end of August). M7 (stretch): paper-trading PnL added to results chapter (third week of August), only if the shadow run is in scope. |
| **September 2026** | Submit by **1 September 2026**. Viva preparation: slide deck (≤12 slides, ≤20 minutes per the project handbook), demo of the reproducible pipeline, pre-emptive Q&A using `reports/templates/viva_qa_notes.md`. | M8: viva-ready presentation and demo by viva date. |

### Risks and mitigations

- **Compute time.** Phase-1 runs are CPU-friendly (10k PPO timesteps,
  three seeds). The full Phase-2 grid is larger but still tractable on a
  Google Colab T4 GPU runtime, and the runners are designed to lift onto
  Colab without code changes. Partial-grid results will be accepted for
  any interim deliverable.
- **Data-API drift.** `yfinance` occasionally changes its column shape.
  The `_close_1d` helper used by every runner already normalises this,
  and the protocol pins explicit dates so a re-pull stays comparable.
- **Result fragility.** The Phase-1 numbers may move under the full
  70-ticker, walk-forward and ablation work. To guard against
  over-claiming, results will be reported as median and inter-quartile
  range across ten seeds and across tickers, evaluated on multiple
  sliding test windows (walk-forward) rather than a single window, and
  any case where the probabilistic variant fails to beat the rule-based
  stop-loss comparator or buy-and-hold will be called out explicitly.
- **Paper-trading dependency (stretch goal only).** The Alpaca shadow
  run is a stretch goal that does not gate the dissertation. If the
  brokerage account, the API or the time available does not support a
  clean two-week run during August, the dissertation rests on the
  backtest and walk-forward evidence and the shadow run is moved into
  the real-world deployment roadmap as post-submission work.

---

## Extenuating circumstances

[FILL IN — for example "None to declare", or describe and indicate that the
personal tutor and student-support services have been informed. Do **not**
include medical detail here.]

---

## Indicative project hours and progress (student self-tick before meeting)

- [ ] The work has exceeded the first 100 hours of time allocated.
- [ ] The work has sufficiently met the first 100 hours.
- [ ] The majority of the first 100 hours have been completed but some time
  has been lost and will be made up.
- [ ] Engagement in the project has been insufficient and progress is of
  concern.

> Recommended self-tick, given the evidence above: *"The work has
> sufficiently met the first 100 hours of time allocated to the project."*
> The reproducible Phase-0 and Phase-1 pipeline, the protocol document, the
> baseline and probabilistic agents, the benchmarks and the generated
> reports together support this self-assessment.

---
