# What this dissertation actually does, in plain English

> **For anyone who is not a reinforcement-learning person and not a finance person.**
> If you only have 5 minutes, read this. The dissertation itself is the academic version of the same content.

---

## The problem in one paragraph

If you put $1,000,000 into the US stock market in January 2022 and held on, by January 2025 you would have had ~$1.52M. Good. But on the way there, your account would have shown $750,000 at one point — a 25% drop from peak. For a normal person that 25% drop is scary. For a pension fund, an endowment, or an institutional asset manager, that 25% drop **breaks contractual constraints** and forces them to sell at the worst possible moment. The financial industry calls this a "drawdown". Most pensions and endowments operate under explicit drawdown rules: don't lose more than X% from the peak, ever. Buy-and-hold violates those rules constantly. Manually setting a "stop loss" (sell if it drops 5%) violates them too — it sells late and buys back even later. **The question this dissertation answers is: can a small AI model do better than either of those two options?**

---

## The solution in one paragraph

We built a two-piece AI agent. **Piece 1** is a small neural network that watches recent price moves and outputs not just a prediction, but a confidence number. When it's not confident, it says so. **Piece 2** is a reinforcement-learning policy that reads that confidence number and decides how much money to put into the market. When confidence is low, the policy automatically reduces position size; when confidence is high, it goes back in. The whole thing is trained to maximise return *subject to* a 95%-of-peak floor — i.e., never let the account drop more than 5% below its all-time high without taking action. This is the exact constraint a real pension fund operates under. **The novelty is not the neural network or the reinforcement learning — those exist in the literature — it is the explicit, formal coupling of the two for drawdown control with confidence-aware position sizing.**

---

## Does it work? Three results, in dollars and percentages.

### Result 1 — On the broad US market (SPY ETF, 2022–2025)
Starting from $1,000,000:

| Strategy | Final value | Worst drawdown |
|---|---:|---:|
| Buy-and-hold (just hold the ETF) | $1.52M | -25% |
| Manual 5% stop-loss rule | $1.23M | -25% (still!) |
| **Probabilistic AI agent (this dissertation)** | **$1.62M** | **-18%** |

The AI made $100K more than buy-and-hold *and* took 7 percentage points less drawdown. The manual stop-loss rule made $290K *less* than buy-and-hold and didn't even reduce drawdown.

### Result 2 — On a 70-ticker diversified-equity test universe (the dissertation's headline)

Same test window. Mean across 70 tickers — 41 single-name US large-cap stocks (technology, payments and financial services, healthcare, consumer, industrials) and 29 ETFs (broad-market, sector, dividend, thematic, commodities) — each starting from $1M:

| Strategy | Mean final value | Mean worst drawdown |
|---|---:|---:|
| Buy-and-hold | $2.10M | -37% |
| Manual 5% stop-loss | $1.53M | -31% |
| **Probabilistic AI agent** | **$2.00M** | **-23%** |

The big finding: **the AI cut average max-drawdown by 14 percentage points (from -37% to -23%) at a cost of only ~5% in mean terminal value**. It cut drawdown on **70 of 70 tickers** — every single one. It beat the manually-tuned stop-loss alternative on **61 of 70 tickers** (87%) in terminal value, and on essentially every ticker in risk-adjusted return. This is a real, defensible improvement on a heterogeneous, real-world equity universe rather than on a single index ETF.

---

## Why drawdown control is the right thing to optimise

Three reasons, each one taken from a body of finance literature that is older than RL.

1. **Real money is run under drawdown rules, not Sharpe ratios.** Endowments operate under spending policies that are tied to how far the portfolio is below its high-water mark. Hedge funds charge their performance fee only above the high-water mark. Pension funds have funded-ratio targets that effectively cap drawdown. If the academic objective is "useful in finance", the academic objective should be drawdown control. (Chekhlov, Uryasev & Zabarankin 2005 formalised this with CDaR; Markowitz 1952 wrote the original variance-based version; both are in the dissertation §2.1.)

2. **Buy-and-hold's 25% drawdown actually happens, and it actually breaks people.** Behavioural finance has measured this for 40 years: real investors sell at the bottom, not because they are stupid, but because their constraints (margin calls, redemption requests, pension funding ratios) force them to. A 25% drawdown is the *minimum* risk on the broad US market in 2022. A drawdown-aware overlay is not a fancy optimisation — it is the difference between staying invested and being forced out.

3. **Manual stop-loss rules don't work, but people use them anyway.** The reason people use them is that there's no good alternative. The dissertation's contribution is the alternative: a confidence-aware overlay that does what the stop-loss tries to do, but smarter — it reduces position size *before* the drawdown rather than after, and it scales the reduction to confidence rather than firing all-or-nothing.

---

## What is "AI" in this project? Specifically.

There are two AI components, both small.

- **The forecaster** is an LSTM neural network with about 9,500 parameters. It reads the last 20 daily returns of a stock and outputs a mean prediction *and* a variance. The variance is the model's own admission of uncertainty. This is "DeepAR-style probabilistic forecasting" (Salinas et al. 2020).
- **The trader** is a Proximal Policy Optimisation (PPO) reinforcement-learning agent. It reads (today's price, recent price history, the forecaster's confidence number) and outputs an action: buy, hold, or sell, scaled by a fraction. PPO is a 2017 algorithm from OpenAI; it is the standard choice for continuous-action control problems.

Total parameter count is well under 50,000. A real production trading system would have 100x more. The model is small *on purpose* — the dissertation is about the methodology, not the size of the network.

---

## What is *not* claimed (honesty)

- **This is not a complete trading system.** It is the risk-control layer of one. The stock-picking is assumed; the layer being studied is what to do with positions you've already chosen.
- **The novelty is modest.** Probabilistic forecasting exists. Drawdown-constrained optimisation exists. RL-based trading exists. The contribution is putting the three together into one explicit, formal, reproducible pipeline and measuring the result. It is one paper's worth of contribution, not a thesis chapter's worth of breakthrough.
- **2022–2025 is a 4-year test window.** A four-fold walk-forward grid on a four-ticker subset of the universe (96 trainings) has finished and shows the agent beats the baseline on all 16 (ticker, fold) cells out-of-time. The full 70-ticker × 4-fold walk-forward grid is the GPU-only Phase-2 deliverable.
- **Three random seeds is a small ensemble at the Phase-1 budget.** Extended 10-seed × 50 000-step runs on a representative eight-ticker sub-universe have finished on CPU and show inter-quartile range under 3% of starting capital on six of eight tickers. The full 70-ticker × 10-seed extended grid is scheduled on Colab GPU in May–June.
- **Single-asset environment.** The agent runs on one ticker at a time. A true multi-asset version that watches the running peak of the *whole portfolio* is the natural next step and is in the future-work plan.

---

## What you can show Dr Nguyen, today

- **`reports/generated/exports/Main_Dissertation_Draft.docx`** — the full dissertation. The sections that directly answer the supervisor's previous-meeting feedback are §1.2 (problem statement, now formal), §2.1 (finance background with notation), §3.1.5 (explicit objective function — this was missing before), §5.5 (70-ticker test-universe aggregate), §5.5.1 (extended-budget seed-stability check), Appendix B (full 70-row per-ticker table), §6.3 (honest discussion of where the agent fails), §6.4 (walk-forward out-of-time evidence).
- **`reports/generated/exports/InterimReview.docx`** — the formal Interim Review document, ready to share.
- **`reports/generated/exports/Fiyins_Dissertation.docx`** — companion document; the same 70-ticker evidence as §5.5 / Appendix B of the academic dissertation, but in plain English with finance context for a non-quantitative reader.
- **`reports/generated/charts/fiyins_portfolio_results.png`** — one image showing the per-ticker results across all 70 tickers.
- **`reports/generated/charts/fiyins_portfolio_winloss.png`** — one image showing where the AI wins and loses across 70 tickers.
- **`reports/generated/findings_plain_english.md`** — this document.
- **`reports/generated/supervisor_handout.md`** — a one-page meeting brief.
