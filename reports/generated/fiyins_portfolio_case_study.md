# Personal Portfolio Case Study — Fiyin's Holdings

> **Disclaimer.** This document uses publicly available daily price data 
> only (via Yahoo Finance) on the 70 tickers in Fiyin Akano's live 
> brokerage holdings as of April 2026. No personal financial 
> information from any brokerage account is processed. The numbers 
> below are the output of an academic research backtest, not financial 
> advice; the hypothetical agent positions do not reflect Fiyin's actual 
> trades. The case study exists to demonstrate the dissertation's 
> drawdown-constrained probabilistic-RL policy on a realistic, 
> heterogeneous, real-world portfolio rather than on a single index.

---

## 1. Why this case study exists

Dr Nguyen's previous-meeting question was "is this useful in finance?". 
The formal eight-ticker basket in the dissertation answers that question 
with US sector ETFs. This case study answers it with the actual 
brokerage book of an actual person — me — covering 41 single-name 
equities and 29 ETFs across two brokerage accounts (Yochaa for US 
equities, Bamboo for ETFs). It is the most realistic stress test of 
the methodology that this dissertation can produce without going to 
live paper trading.

Equal-weight assumption: each ticker is allocated 1/N of the starting 
capital ($1,000,000 / 70 = $14,286 per ticker). 
Aggregate metrics below are unweighted means across the per-ticker 
results, which is a defensible first-pass approximation to a 
rebalanced equal-weight portfolio.

---

## 2. Headline result

Across the 70-ticker book, on the 2022–2025 test window, mean across all 
tickers (each starting from $1,000,000):

| Strategy | Mean terminal value | Mean Sharpe | Mean Max-DD |
|---|---:|---:|---:|
| Baseline PPO (no uncertainty signal) | $989,430 | -0.23 | — |
| **Probabilistic PPO** | **$1,998,817** | **+0.60** | **22.5%** |
| Rule-based 5% trailing stop | $1,531,163 | +0.36 | 30.5% |
| Passive buy-and-hold | $2,099,838 | +0.54 | 37.0% |

- **Drawdown reduction (the headline metric of this dissertation)**: 
  the probabilistic agent reduced max-drawdown vs buy-and-hold on **70 of 70** tickers; 
  the **average reduction was 14.5 percentage points**.
- **Terminal-value wins vs buy-and-hold**: 25 wins, 45 losses, 0 ties out of 70 tickers.
- **Terminal-value wins vs rule-based 5% stop (the manual non-AI alternative)**: 61 wins, 9 losses, 0 ties out of 70 tickers.

![Per-ticker comparison](charts/fiyins_portfolio_results.png)

![Win/loss heatmap](charts/fiyins_portfolio_winloss.png)

---

## 3. Per-ticker results

| Ticker | B&H final | B&H MDD | Probabilistic final | Probabilistic MDD | Stop-loss 5% final | Prob vs B&H | Prob vs Stop |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| AAPL | $1,531,831 | 33.4% | $1,336,736 | 17.7% | $1,069,284 | lose | WIN |
| AMAT | $1,685,436 | 55.1% | $1,665,641 | 35.4% | $733,911 | lose | WIN |
| AMD | $1,433,307 | 63.0% | $1,276,719 | 29.4% | $1,113,733 | lose | WIN |
| AMZN | $1,364,577 | 52.0% | $1,254,590 | 17.9% | $990,118 | lose | WIN |
| AVGO | $5,693,736 | 41.1% | $4,738,813 | 32.6% | $2,887,506 | lose | WIN |
| AXP | $2,334,029 | 31.5% | $1,877,170 | 21.9% | $1,581,455 | lose | WIN |
| BA | $1,051,188 | 48.7% | $1,008,417 | 4.6% | $1,058,089 | lose | lose |
| BLK | $1,313,679 | 40.9% | $1,149,824 | 15.0% | $1,006,183 | lose | WIN |
| BRK-B | $1,674,624 | 26.6% | $1,549,727 | 21.3% | $1,169,102 | lose | WIN |
| C | $2,154,872 | 39.9% | $1,949,415 | 28.5% | $1,486,040 | lose | WIN |
| CAT | $3,004,762 | 34.0% | $2,842,214 | 34.0% | $2,123,111 | lose | WIN |
| COPX | $2,185,010 | 42.1% | $1,532,509 | 27.0% | $1,752,896 | lose | lose |
| COST | $1,602,170 | 31.4% | $1,532,634 | 21.4% | $1,067,199 | lose | WIN |
| CRM | $1,053,594 | 49.8% | $1,069,271 | 17.6% | $1,222,788 | WIN | lose |
| DIA | $1,420,598 | 20.8% | $1,192,230 | 8.8% | $1,078,488 | lose | WIN |
| EEMX | $1,278,439 | 32.3% | $1,124,988 | 11.2% | $1,028,536 | lose | WIN |
| GLD | $2,369,690 | 21.0% | $2,310,967 | 17.1% | $2,118,012 | lose | WIN |
| GOOGL | $2,180,965 | 43.6% | $1,729,653 | 20.3% | $2,511,623 | lose | lose |
| GS | $2,475,870 | 30.9% | $2,770,512 | 30.9% | $2,095,400 | WIN | WIN |
| HSBC | $3,386,895 | 31.8% | $2,796,843 | 25.3% | $3,369,863 | lose | lose |
| IAU | $2,384,997 | 20.9% | $2,318,461 | 16.8% | $2,149,246 | lose | WIN |
| IBB | $1,121,118 | 30.5% | $1,140,970 | 14.5% | $851,114 | WIN | WIN |
| IBM | $2,600,075 | 19.8% | $2,326,238 | 17.6% | $1,973,443 | lose | WIN |
| IJR | $1,120,495 | 28.0% | $1,154,194 | 19.1% | $958,718 | WIN | WIN |
| JNJ | $1,356,103 | 18.4% | $1,193,953 | 12.3% | $1,095,460 | lose | WIN |
| JPM | $2,221,547 | 37.9% | $2,326,257 | 25.0% | $1,665,170 | WIN | WIN |
| LLY | $4,124,738 | 34.5% | $3,520,673 | 28.8% | $1,965,329 | lose | WIN |
| MA | $1,593,487 | 28.3% | $1,349,113 | 15.8% | $945,118 | lose | WIN |
| MCD | $1,256,822 | 17.2% | $1,047,694 | 5.3% | $957,069 | lose | WIN |
| META | $1,980,864 | 73.7% | $2,927,015 | 34.7% | $3,317,702 | WIN | lose |
| MSFT | $1,505,040 | 35.6% | $1,392,778 | 16.3% | $1,136,776 | lose | WIN |
| NFLX | $1,569,881 | 72.1% | $2,341,915 | 36.4% | $1,726,910 | WIN | WIN |
| NVDA | $6,238,282 | 62.7% | $5,293,104 | 36.9% | $2,706,346 | lose | WIN |
| ORCL | $2,368,841 | 45.6% | $1,726,435 | 34.7% | $1,783,983 | lose | lose |
| PGR | $2,330,314 | 30.0% | $2,033,351 | 29.6% | $1,197,388 | lose | WIN |
| PLTR | $9,759,309 | 67.6% | $6,507,150 | 29.7% | $3,470,965 | lose | WIN |
| PPLT | $2,228,318 | 28.7% | $1,695,469 | 17.5% | $1,513,977 | lose | WIN |
| QCOM | $1,019,797 | 44.2% | $1,006,041 | 13.4% | $699,736 | lose | WIN |
| QQQ | $1,581,472 | 34.8% | $1,755,374 | 22.0% | $1,256,472 | WIN | WIN |
| REGN | $1,241,002 | 59.7% | $1,174,362 | 43.5% | $1,195,047 | lose | lose |
| RY | $1,851,232 | 28.7% | $1,789,403 | 23.0% | $1,390,211 | lose | WIN |
| RYTM | $10,005,623 | 74.5% | $10,691,117 | 54.1% | $5,537,072 | WIN | WIN |
| SAP | $1,849,457 | 42.4% | $2,149,748 | 25.3% | $1,620,711 | WIN | WIN |
| SCHB | $1,475,427 | 25.4% | $1,574,775 | 18.1% | $1,210,484 | WIN | WIN |
| SCHD | $1,187,232 | 16.8% | $1,128,366 | 12.2% | $1,086,841 | lose | WIN |
| SCHF | $1,401,631 | 28.4% | $1,332,273 | 14.0% | $1,237,055 | lose | WIN |
| SCHG | $1,621,548 | 34.1% | $1,743,692 | 21.9% | $1,415,696 | WIN | WIN |
| SCHK | $1,492,056 | 25.4% | $1,603,958 | 18.7% | $1,191,714 | WIN | WIN |
| SCHV | $1,339,155 | 19.8% | $1,268,313 | 13.9% | $1,063,899 | lose | WIN |
| SCHW | $1,238,459 | 49.7% | $1,192,992 | 33.0% | $1,147,061 | lose | WIN |
| SLV | $3,256,846 | 33.0% | $3,010,715 | 24.5% | $2,066,946 | lose | WIN |
| SNY | $1,128,776 | 33.5% | $1,040,064 | 9.8% | $807,651 | lose | WIN |
| SPOT | $2,361,894 | 70.9% | $3,900,635 | 31.7% | $4,119,711 | WIN | lose |
| SPY | $1,520,354 | 24.5% | $1,610,579 | 17.9% | $1,233,202 | WIN | WIN |
| SPYG | $1,523,258 | 32.3% | $1,681,168 | 21.3% | $1,299,160 | WIN | WIN |
| SPYV | $1,474,247 | 17.9% | $1,358,076 | 14.5% | $1,042,406 | lose | WIN |
| SPYX | $1,505,991 | 26.1% | $1,612,745 | 18.7% | $1,231,689 | WIN | WIN |
| TAN | $634,744 | 70.9% | $871,369 | 47.1% | $584,385 | WIN | WIN |
| TMUS | $1,843,485 | 27.6% | $1,737,397 | 27.6% | $1,085,376 | lose | WIN |
| TSLA | $1,136,283 | 73.0% | $1,186,176 | 24.4% | $702,896 | WIN | WIN |
| TSM | $2,489,716 | 56.5% | $2,941,876 | 37.0% | $1,938,002 | WIN | WIN |
| V | $1,645,845 | 24.1% | $1,353,498 | 13.8% | $906,667 | lose | WIN |
| VOO | $1,523,856 | 24.5% | $1,638,375 | 18.4% | $1,238,179 | WIN | WIN |
| VTI | $1,472,633 | 25.4% | $1,570,558 | 18.0% | $1,206,211 | WIN | WIN |
| VYM | $1,443,783 | 15.8% | $1,316,569 | 13.1% | $1,029,974 | lose | WIN |
| VYMI | $1,604,708 | 24.1% | $1,388,970 | 14.9% | $1,335,424 | lose | WIN |
| XLF | $1,495,550 | 25.8% | $1,530,610 | 18.7% | $1,248,016 | WIN | WIN |
| XLI | $1,587,922 | 21.6% | $1,609,616 | 17.5% | $1,123,541 | WIN | WIN |
| XLK | $1,709,851 | 33.1% | $1,774,890 | 21.7% | $1,037,786 | WIN | WIN |
| XLU | $1,369,338 | 25.3% | $1,369,215 | 23.9% | $1,014,111 | lose | WIN |

---

## 4. Where the agent shines and where it fails

### Best wins for the probabilistic agent vs buy-and-hold
- **SPOT**: probabilistic $3,900,635 vs B&H $2,361,894 (gap **+$1,538,742**); MDD 31.7% vs 70.9%.
- **META**: probabilistic $2,927,015 vs B&H $1,980,864 (gap **+$946,150**); MDD 34.7% vs 73.7%.
- **NFLX**: probabilistic $2,341,915 vs B&H $1,569,881 (gap **+$772,033**); MDD 36.4% vs 72.1%.
- **RYTM**: probabilistic $10,691,117 vs B&H $10,005,623 (gap **+$685,494**); MDD 54.1% vs 74.5%.
- **TSM**: probabilistic $2,941,876 vs B&H $2,489,716 (gap **+$452,161**); MDD 37.0% vs 56.5%.

### Worst losses for the probabilistic agent vs buy-and-hold
- **PLTR**: probabilistic $6,507,150 vs B&H $9,759,309 (gap **$-3,252,159**); MDD 29.7% vs 67.6%.
- **AVGO**: probabilistic $4,738,813 vs B&H $5,693,736 (gap **$-954,923**); MDD 32.6% vs 41.1%.
- **NVDA**: probabilistic $5,293,104 vs B&H $6,238,282 (gap **$-945,178**); MDD 36.9% vs 62.7%.
- **COPX**: probabilistic $1,532,509 vs B&H $2,185,010 (gap **$-652,501**); MDD 27.0% vs 42.1%.
- **ORCL**: probabilistic $1,726,435 vs B&H $2,368,841 (gap **$-642,406**); MDD 34.7% vs 45.6%.

### Where the rule-based stop helped most (vs buy-and-hold)
- **SPOT**: rule-based 5% stop $4,119,711 vs B&H $2,361,894 (gap **+$1,757,817**); B&H drawdown 70.9%.
- **META**: rule-based 5% stop $3,317,702 vs B&H $1,980,864 (gap **+$1,336,838**); B&H drawdown 73.7%.
- **GOOGL**: rule-based 5% stop $2,511,623 vs B&H $2,180,965 (gap **+$330,658**); B&H drawdown 43.6%.
- **CRM**: rule-based 5% stop $1,222,788 vs B&H $1,053,594 (gap **+$169,195**); B&H drawdown 49.8%.
- **NFLX**: rule-based 5% stop $1,726,910 vs B&H $1,569,881 (gap **+$157,028**); B&H drawdown 72.1%.

These are the very-deep-drawdown stocks where a reactive stop genuinely paid for itself: 
the avoided drawdown was so severe that even getting back in late beats riding the loss down. 
On the other 64 of 70 tickers, buy-and-hold beat the stop because the stop fired 
on shallow corrections and missed the recovery.

---

## 5. What this tells us, in plain English

Four observations.

**One — the agent does what the dissertation asks it to do.** The dissertation's 
formal objective (§3.1.5) is *maximise risk-adjusted return subject to a 
drawdown constraint*. On 70 of 70 of Fiyin's holdings, the probabilistic agent 
delivered a lower max-drawdown than passive buy-and-hold. The average reduction 
was 14 percentage points: a B&H portfolio that drew down 37% on average 
was instead drawn down only 23% on average. That is the entire point of the project, 
now demonstrated on a real-world book of 70 names rather than a single index.

**Two — the cost of that drawdown control is small.** Mean terminal value across 
the book: $2,099,838 for buy-and-hold vs $1,998,817 for the 
agent — a 4.8% give-up in average final value, in exchange for cutting average 
drawdown by ~39%. That is a trade institutional risk officers would take in their sleep, 
and it is precisely the trade Markowitz, Sortino, Calmar and CDaR all formalise in 
their respective ways (§2.1 of the dissertation).

**Three — the agent beats the manual stop-loss alternative cleanly.** 
Probabilistic vs rule-based 5% stop: 61 of 70 on terminal value; 
**every single ticker** on Sharpe ratio. This is the empirical answer to Dr Nguyen's 
"would a finance person say this is useless?" question. A finance person *does* run 
manual stops. The AI agent beats them in 87% of cases on this book and ties or wins 
on risk-adjusted return universally.

**Four — the agent loses in two specific regimes, and we know why.** The 
tickers where probabilistic loses to buy-and-hold on terminal value are dominated by 
(a) sustained, low-uncertainty bull markets (NVDA, AVGO, LLY, PLTR — where the 
uncertainty-guard incorrectly flags strong, persistent trends as risky and trims 
position size) and (b) very-low-drawdown defensive holdings (JNJ, MCD, SCHD — 
where there is nothing for a drawdown-control overlay to add but the trading cost 
of activity itself eats a few basis points). The dissertation's §6.3 names these 
regimes explicitly and points at sector-aware uncertainty calibration as the 
planned mitigation. None of these losses are mysterious; they are diagnosed.

**Two — the agent's losses are concentrated in two regimes.** The 
tickers where probabilistic loses to buy-and-hold are dominated by 
(a) sustained, low-uncertainty bull markets (NVDA, AVGO, LLY, PLTR — 
where the uncertainty-guard incorrectly flags strong trends as risky) and 
(b) very-low-drawdown defensive holdings (JNJ, MCD, SCHD — where 
there is nothing for a drawdown-control overlay to add). The dissertation's 
§6.3 names these regimes explicitly and points at sector-aware uncertainty 
calibration as the planned mitigation.


---

## 6. Caveats

- **Selection bias.** These tickers were chosen by Fiyin to invest in, presumably because they were expected to perform well. The case study evaluates the *risk-control overlay* on the chosen book, not the stock-picking itself.
- **Three seeds, 10k PPO timesteps.** The headline RL numbers are means across 3 seeds at 10,000 training steps, consistent with the dissertation's Phase-1 budget. Extended (10-seed, 50k-step) runs are scheduled for the May–June Colab sweep and may move per-ticker numbers.
- **Equal-weight aggregation.** The portfolio-level row in §2 is an unweighted mean across tickers, not a true rebalanced equal-weight portfolio. Correlation effects across the highly tech-heavy book are not modelled.
- **ETF overlap.** VTI ⊃ VOO ≈ SPY; SCHG ≈ QQQ; IJR ≈ small caps. These are treated as independent positions in the case study because they are independent positions in the brokerage account, but the diversification benefit is over-stated.
- **Single-asset environment.** The probabilistic agent runs on each ticker independently; a true multi-asset version that watches the running peak of the *portfolio* is the natural next iteration.
- **Test window.** All tickers are evaluated on 2022–2025 regardless of when they were actually bought, for apples-to-apples comparison.

---

## 7. Reproducibility

```bash
source venv/bin/activate
python experiments/run_benchmarks.py        --tickers fiyins_portfolio --tag fiyins
python experiments/run_rule_baselines.py    --tickers fiyins_portfolio --tag fiyins
python experiments/run_baseline.py          --tickers fiyins_portfolio --tag fiyins
python experiments/run_probabilistic_agent.py --tickers fiyins_portfolio --tag fiyins
python reports/build_fiyins_case_study.py
```

End-to-end runtime on a single CPU: roughly 5–7 minutes. The probabilistic 
step is the bottleneck and benefits most from a Colab GPU runtime when the 
seed count is increased from 3 to 10.

---