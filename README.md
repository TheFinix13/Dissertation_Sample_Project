# Probabilistic DRL for Portfolio Risk Analysis

EEEM004 research project: an uncertainty-aware PPO policy for
capital preservation under regime stress.

## For supervisors — two notebooks, two registers

* **Single-ticker walkthrough (CPU/Colab, narrative):** [`notebooks/Dissertation_Walkthrough.ipynb`](notebooks/Dissertation_Walkthrough.ipynb) — open in Colab and *Run all*. Loads the SPY test-window dataset, builds the DeepAR-style probabilistic forecaster, prints the uncertainty values, states the mathematics that differentiates the probabilistic agent from the baseline PPO, and renders the comparison tables and equity-curve plots. Ships with executed outputs so the notebook is readable without running anything.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheFinix13/Dissertation_Sample_Project/blob/main/Dissertation_Walkthrough.ipynb)

* **Heavy experiments runner (Colab GPU only):** [`notebooks/extended_grid_colab.ipynb`](notebooks/extended_grid_colab.ipynb) — *Runtime → T4 GPU → Run all*. Clones the repo, smoke-tests the GPU, runs the full 70-ticker × 10-seed × 50 000-step extended grid + walk-forward folds + bootstrap, rebuilds **both** Word documents (academic dissertation and personal-portfolio companion) with the new numbers, then offers a one-click zip download. ~5–7 hours on T4, ~2 hours on A100. *Do not run on a CPU laptop.*

For a local run instead of Colab:

```bash
git clone https://github.com/TheFinix13/Dissertation_Sample_Project.git
cd Dissertation_Sample_Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook Dissertation_Walkthrough.ipynb
```

The interim review draft (Surrey form) lives at
`reports/generated/interim_review_draft.md`.

## Generated artifacts (under `reports/generated/exports/`)

- `Main_Dissertation_Draft.docx` — the **academic** Master's dissertation. Headline robustness evidence is the four-agent comparison on a 70-ticker diversified-equity test universe (Section 5.5) with the full per-ticker table in Appendix B; supplementary studies are an extended seed-stability check on a representative eight-ticker sub-universe (Section 5.5.1) and a four-fold walk-forward grid on a four-ticker subset (Section 6.4). Title page, abstract, 7 chapters, references, two appendices.
- `Fiyins_Dissertation.docx` — the **plain-English companion**. Same 70-ticker evidence as Section 5.5 / Appendix B of the academic dissertation, but written for a non-quantitative reader with finance context, full per-ticker commentary and visualisations. Cover, executive summary, 10 chapters, reproducibility appendix.
- `InterimReview.docx` — the formal Surrey Interim Review form.
- `Dissertation_Walkthrough.pdf` — fully-rendered PDF of the supervisor walkthrough notebook (with embedded outputs, tables and plots).
- `equations/` — individual PNGs for every equation in the docx.

To regenerate every document:

```bash
venv/bin/python reports/builders/build_main_dissertation_docx.py       # academic dissertation
venv/bin/python reports/builders/build_fiyins_dissertation_docx.py     # personal-portfolio companion
venv/bin/python reports/builders/build_interim_review_docx.py          # interim review form
venv/bin/python -m nbconvert --to webpdf --allow-chromium-download \
  --output reports/generated/exports/Dissertation_Walkthrough.pdf \
  Dissertation_Walkthrough.ipynb
```

Heaviest experiments (70-ticker × 10-seed × 50k-step extended grid, walk-forward
across all four folds, bootstrap-augmented training) live in
`notebooks/extended_grid_colab.ipynb` and run on a Colab T4/A100 GPU runtime — see
"Phase-2 (Colab GPU) pipeline" below.

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
├── experiments/
│   ├── runners/             # CLI entry points (run_baseline.py, run_probabilistic_agent.py, ...)
│   ├── common.py            # Shared library: env, metrics, data, training helpers
│   ├── aggregate_results.py # Pools per-cell JSON results into median + IQR summaries
│   ├── configs/             # dissertation_protocol.json
│   └── results/             # Per-cell JSON + CSV outputs from runners
├── reports/
│   ├── builders/            # All build_*.py / generate_*.py / plot_*.py scripts
│   ├── generated/           # Outputs (markdown, charts/, exports/)
│   └── templates/           # Markdown templates and viva notes
├── notebooks/               # Dissertation_Walkthrough + extended_grid_colab
├── trained_models/          # Saved PPO models
├── live/                    # Post-submission deployment scaffold (Phase A/B/C, see live/ROADMAP.md)
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

## Phase 1 Experiment Pipeline (single-fold, legacy headline)

```bash
source venv/bin/activate

# 1) Baseline PPO with deterministic seeds
python experiments/runners/run_baseline.py

# 2) Probabilistic DeepAR-style uncertainty + PPO
python experiments/runners/run_probabilistic_agent.py

# 3) Buy-and-hold and all-cash benchmarks
python experiments/runners/run_benchmarks.py

# 4) Rule-based trailing stop-loss comparator (5 % and 10 % variants)
python experiments/runners/run_rule_baselines.py

# 5) Markdown summary, supervisor pack, plots
python reports/builders/generate_dissertation_report.py
python reports/builders/build_supervisor_pack.py
python reports/builders/plot_dissertation_visuals.py

# 6) Word documents (dissertation + interim review)
python reports/builders/build_main_dissertation_docx.py
python reports/builders/build_interim_review_docx.py
```

- Protocol config: `experiments/configs/dissertation_protocol.json`
- Artifacts: `experiments/results/`
- Report output: `reports/generated/dissertation_results.md`

## Phase 2 Experiment Pipeline (70-ticker test universe, walk-forward, multi-seed)

Every runner now accepts `--tickers`, `--seeds`, `--folds`, `--timesteps`,
`--initial-balance`, `--bootstrap-paths` and `--tag` flags. The dissertation's
headline test universe is the **70-ticker diversified-equity universe**
materialised as the named group `fiyins_portfolio` under `data.named_groups` in
`experiments/configs/dissertation_protocol.json`: 41 single-name US large-cap
equities (technology, payments and financial services, healthcare, consumer,
industrials) plus 29 ETFs (broad-market indices, sector SPDRs, dividend,
thematic, commodities). The runner CLI also still accepts the legacy `basket`
group (eight-ticker sub-universe used for the Section 5.5.1 extended seed-stability
check) and arbitrary comma-separated ticker lists.

### Two documents — academic dissertation and plain-English companion

The deliverables split into **two** Word documents and **one** Colab notebook
for the heaviest experiments:

| File                                                             | Scope                                                                                                          | Built by |
|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------|
| `reports/generated/exports/Main_Dissertation_Draft.docx`         | The academic EEEM004 dissertation. 70-ticker test universe headline (Section 5.5 + Appendix B), eight-ticker sub-universe extended seed-stability check (Section 5.5.1), four-ticker walk-forward grid (Section 6.4). | `python reports/builders/build_main_dissertation_docx.py` |
| `reports/generated/exports/Fiyins_Dissertation.docx`             | Plain-English companion to the 70-ticker evidence. Same numbers as Section 5.5 / Appendix B of the academic dissertation, but written for a non-quantitative reader with finance context, full per-ticker commentary and visualisations. | `python reports/builders/build_fiyins_dissertation_docx.py` |
| `notebooks/extended_grid_colab.ipynb`                            | The heaviest experiments — full 70-ticker × 10-seed × 50k-step extended grid + 4-fold walk-forward + bootstrap. *GPU-only.* | Open in Colab → *Run all*. |

### Phase-1 (CPU) pipeline — runs on a laptop in 25–35 minutes

```bash
# Phase-1 budget run on the 70-ticker test universe (3 seeds × 10k steps; ≈ 25–35 min CPU)
python experiments/runners/run_benchmarks.py        --tickers fiyins_portfolio --tag fiyins70
python experiments/runners/run_rule_baselines.py    --tickers fiyins_portfolio --tag fiyins70
python experiments/runners/run_baseline.py          --tickers fiyins_portfolio --tag fiyins70
python experiments/runners/run_probabilistic_agent.py --tickers fiyins_portfolio --tag fiyins70

# Walk-forward subset (96 trainings, ≈ 6–8 hours CPU)
python experiments/runners/run_walk_forward.py --tickers SPY,QQQ,XLK,XLF

# Extended seed-stability check on representative sub-universe (80 trainings, ≈ 4–5 hours CPU)
python experiments/runners/run_probabilistic_agent.py --tickers basket --seeds extended --timesteps 50000 --tag extbasket

# Build everything — academic dissertation, companion document, interim review,
# case-study charts, plain-English summary.
python reports/builders/build_fiyins_case_study.py            # tables + PNG charts
python reports/builders/build_fiyins_dissertation_docx.py     # Fiyins_Dissertation.docx
python reports/builders/build_main_dissertation_docx.py       # Main_Dissertation_Draft.docx
python reports/builders/build_interim_review_docx.py          # InterimReview.docx
```

Outputs land at:

- `reports/generated/exports/Fiyins_Dissertation.docx`
- `reports/generated/exports/Main_Dissertation_Draft.docx`
- `reports/generated/exports/InterimReview.docx`
- `reports/generated/exports/FiyinsPortfolio_CaseStudy.docx` *(legacy short-form case study; the canonical home for the personal-portfolio analysis is `Fiyins_Dissertation.docx`)*
- `reports/generated/charts/fiyins_portfolio_results.png`
- `reports/generated/charts/fiyins_portfolio_winloss.png`

### Phase-2 (Colab GPU) pipeline — heavy lifting only

Anything that takes more than ~1 hour on CPU lives in
`notebooks/extended_grid_colab.ipynb`. Runtime preset: *T4 GPU* for the headline
70-ticker grid (~5–7 h), *A100* if you also want the full 70-ticker walk-forward
(~12–14 h on A100). The notebook is intentionally compact (8 numbered sections);
*Run all* clones the repository, smoke-tests the GPU, executes the heaviest
experiments below in sequence, rebuilds both Word documents with the new numbers
and offers a one-click zip download of every output.

| Section in the notebook                                          | Experiment                                                                                                | Approx wall-time on T4 |
|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------|
| Section 3 — heaviest run                                                 | 70-ticker × 10-seed × 50 000-PPO-step extended grid + 32-path stationary block bootstrap                  | 5–7 hours              |
| Section 4 — walk-forward at extended budget                              | 4-ticker × 4-fold × 10-seed × 50 000-step walk-forward (out-of-time evaluation, CPU-feasible subset extended to 10 seeds) | 3–4 hours              |
| Section 5 — *(A100 only)* full 70-ticker walk-forward                    | 70-ticker × 4-fold × 10-seed × 50 000-step × 16-bootstrap-path × 2-agent grid                              | 12–14 hours on A100    |
| Section 6 — rebuild `Main_Dissertation_Draft.docx` + `Fiyins_Dissertation.docx`| Aggregates every result JSON and refreshes both Word documents with extended-budget median + IQR bands.   | < 1 minute             |

For local Colab launch:

```bash
# In any Colab session, after opening the notebook:
#   Runtime → Change runtime type → Hardware accelerator: T4 GPU
#   Runtime → Run all
```

Or to drive the same heavy run from the command line (e.g. on a leased GPU node):

```bash
python experiments/runners/run_extended_grid.py \
    --tickers fiyins_portfolio --seeds extended --folds all \
    --timesteps 50000 --bootstrap-paths 16 --tag colab_70_extended
```

Approximate wall-time: ≈ 5–7 hours on a Colab T4, ≈ 2–3 days on CPU. The runner
writes per-cell JSON files plus an aggregate summary CSV which both the
academic dissertation builder and the personal-portfolio builder consume
automatically the next time they run.

## CLI flag reference (every runner)

| Flag | Default | Notes |
|---|---|---|
| `--tickers` | legacy single ticker | Comma-separated, a CLI alias such as `basket` (8-ticker sub-universe) or a named group from `data.named_groups` in the protocol — `fiyins_portfolio` (the 70-ticker test universe), `fiyins_stocks` (41 single names) or `fiyins_etfs` (29 ETFs). |
| `--seeds` | `[7, 19, 42]` | Comma-separated, or `default` / `extended` (10 seeds). |
| `--folds` | legacy single test window | Comma-separated fold ids from `walk_forward_folds`, or `all`. (Walk-forward runner only honours this.) |
| `--timesteps` | from protocol | PPO training budget per cell. |
| `--initial-balance` | $1,000,000 | Starting capital in USD; metric ratios (Sharpe, MDD, preservation) are unit-free. |
| `--bootstrap-paths` | 0 | Politis & Romano (1994) stationary block-bootstrap synthetic training paths. |
| `--tag` | none | Optional suffix appended to output filenames. |
| `--agents` | `baseline,probabilistic` | Walk-forward only; subset to run. |
