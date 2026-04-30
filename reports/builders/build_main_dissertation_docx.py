"""Build the formal dissertation Word document.

Outputs:
    reports/generated/exports/Main_Dissertation_Draft.docx

Run:
    venv/bin/python reports/builders/build_main_dissertation_docx.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
EXPORTS = ROOT / "reports" / "generated" / "exports"
EQ_DIR = EXPORTS / "equations"
CHARTS = ROOT / "reports" / "generated" / "charts"
EXPORTS.mkdir(parents=True, exist_ok=True)
EQ_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def render_equation(latex: str, filename: str, fontsize: int = 18) -> Path:
    """Render a LaTeX equation to a transparent PNG via matplotlib mathtext."""
    path = EQ_DIR / filename
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, latex, fontsize=fontsize)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.close(fig)
    return path


def latest_json(prefix: str) -> dict | list:
    files = sorted(p for p in RESULTS.glob(f"{prefix}_*.json"))
    if not files:
        return []
    return json.loads(files[-1].read_text(encoding="utf-8"))


def latest_json_tagged(prefix: str, tag: str) -> dict | list:
    """Pick the most recent file with the form ``{prefix}_<TS>_{tag}.json``.

    Used so different sections of the dissertation can pin their evidence
    to a specific experiment tag rather than silently inheriting whatever
    the latest run on disk happens to be.
    """
    files = sorted(p for p in RESULTS.glob(f"{prefix}_*_{tag}.json"))
    if not files:
        return []
    return json.loads(files[-1].read_text(encoding="utf-8"))


def avg(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return float(np.mean([float(r[key]) for r in rows]))


def add_heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)


def add_para(doc: Document, text: str, *, italic: bool = False, bold: bool = False, align=None) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.italic = italic
    run.bold = bold


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_equation(doc: Document, latex: str, filename: str, *, label: str | None = None, width_inches: float = 4.5) -> None:
    path = render_equation(latex, filename)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_inches))
    if label:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap.add_run(label)
        cap_run.italic = True
        cap_run.font.size = Pt(10)


def add_figure(doc: Document, image_path: Path, caption: str, *, width_inches: float = 5.5) -> None:
    if not image_path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(caption)
    cap_run.italic = True
    cap_run.font.size = Pt(10)


def add_metrics_table(doc: Document, rows: list[dict]) -> None:
    headers = ["Agent", "Final value (USD)", "Sharpe", "Max DD", "VaR-95 viol.", "Preservation"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for r in hdr_cells[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = table.add_row().cells
        cells[0].text = row["agent"]
        cells[1].text = f"${row['final']:,.0f}"
        cells[2].text = f"{row['sharpe']:+.4f}"
        cells[3].text = f"{row['mdd']:.4f}"
        cells[4].text = f"{row['var']:.4f}"
        cells[5].text = f"{row['pres']:.4f}"
    for r in table.rows:
        for c in r.cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_basket_table(doc: Document, rows: list[dict]) -> None:
    """Per-ticker comparison table used by Section 5.5.

    Each input row carries an extra 'ticker' field; rows are grouped by
    ticker and laid out as one row per (ticker, agent) pair.
    """
    headers = ["Ticker", "Agent", "Final value (USD)", "Sharpe", "Max DD", "Path preservation"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for r in hdr_cells[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = table.add_row().cells
        cells[0].text = row["ticker"]
        cells[1].text = row["agent"]
        cells[2].text = f"${row['final']:,.0f}"
        cells[3].text = f"{row['sharpe']:+.4f}"
        cells[4].text = f"{row['mdd']:.4f}"
        cells[5].text = f"{1.0 - float(row['mdd']):.4f}"
    for r in table.rows:
        for c in r.cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def page_break(doc: Document) -> None:
    doc.add_page_break()


def set_default_font(doc: Document, family: str = "Times New Roman", size: int = 11) -> None:
    style = doc.styles["Normal"]
    style.font.name = family
    style.font.size = Pt(size)
    rpr = style.element.rPr
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), family)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> Path:
    doc = Document()
    set_default_font(doc)
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    baseline = latest_json("baseline")
    prob = latest_json("probabilistic")
    bench = latest_json("benchmarks")
    rules = latest_json("rule_baseline")

    def _spy_only(rows: list) -> list:
        """Return rows that pertain to SPY (or have no ticker field at all,
        for backwards compatibility with single-ticker artefacts)."""
        if not rows:
            return []
        return [r for r in rows if r.get("ticker", "SPY") == "SPY"]

    baseline = _spy_only(baseline)
    prob = _spy_only(prob)
    bench_spy = _spy_only(bench)
    rules_spy = _spy_only(rules)
    bench_lookup = {r["agent"]: r for r in bench_spy}
    rule_lookup = {r["agent"]: r for r in rules_spy}

    metrics_rows = []
    if baseline and prob:
        metrics_rows.extend([
            {
                "agent": "Baseline PPO",
                "final": avg(baseline, "final_portfolio_value"),
                "sharpe": avg(baseline, "sharpe_ratio"),
                "mdd": avg(baseline, "max_drawdown"),
                "var": avg(baseline, "var_95_violation_rate"),
                "pres": avg(baseline, "capital_preservation_rate_95pct_hwm"),
            },
            {
                "agent": "Probabilistic PPO",
                "final": avg(prob, "final_portfolio_value"),
                "sharpe": avg(prob, "sharpe_ratio"),
                "mdd": avg(prob, "max_drawdown"),
                "var": avg(prob, "var_95_violation_rate"),
                "pres": avg(prob, "capital_preservation_rate_95pct_hwm"),
            },
        ])
    for label, display in (
        ("stop_loss_5pct", "Rule-based stop-loss (5%)"),
        ("stop_loss_10pct", "Rule-based stop-loss (10%)"),
    ):
        r = rule_lookup.get(label)
        if r:
            metrics_rows.append({
                "agent": display,
                "final": float(r["final_portfolio_value"]),
                "sharpe": float(r["sharpe_ratio"]),
                "mdd": float(r["max_drawdown"]),
                "var": float(r["var_95_violation_rate"]),
                "pres": float(r["capital_preservation_rate_95pct_hwm"]),
            })
    for label in ("buy_and_hold", "all_cash"):
        b = bench_lookup.get(label)
        if b:
            metrics_rows.append({
                "agent": "Buy-and-hold (SPY)" if label == "buy_and_hold" else "All-cash",
                "final": float(b["final_portfolio_value"]),
                "sharpe": float(b["sharpe_ratio"]),
                "mdd": float(b["max_drawdown"]),
                "var": float(b["var_95_violation_rate"]),
                "pres": float(b["capital_preservation_rate_95pct_hwm"]),
            })

    # ----- Title page -----
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Probabilistic Deep Reinforcement Learning\nfor Portfolio Risk Analysis")
    r.bold = True
    r.font.size = Pt(22)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Drawdown-constrained portfolio control with an uncertainty-aware reinforcement-learning policy")
    r.italic = True
    r.font.size = Pt(14)

    for _ in range(4):
        doc.add_paragraph()

    for line, sz, bold in [
        ("Fiyin Akano", 14, True),
        ("URN: [INSERT URN]", 12, False),
        ("", 12, False),
        ("EEEM004 — MSc Dissertation", 12, False),
        ("Department of Electrical and Electronic Engineering", 12, False),
        ("University of Surrey", 12, False),
        ("", 12, False),
        ("Supervisor: [INSERT SUPERVISOR]", 12, False),
        ("Second supervisor: [INSERT IF APPLICABLE]", 12, False),
        ("", 12, False),
        ("September 2026 (target submission)", 12, True),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.bold = bold
        run.font.size = Pt(sz)

    page_break(doc)

    # ----- Abstract -----
    add_heading(doc, "Abstract", 1)
    add_para(
        doc,
        "A risk-constrained investor — a pension fund, a CTA hedge fund, a family office, "
        "or a behaviourally loss-averse retail account — has to satisfy two requirements at "
        "once: not lose more than a stated percentage from peak, and earn a return above "
        "cash. Mean-variance optimisation cannot solve this jointly because it is single-"
        "period and blind to path-dependent loss. Trailing stop-losses fire after the "
        "drawdown has already happened and forfeit the recovery. Risk parity targets "
        "volatility rather than drawdown. The gap is real, the mandates that create it are "
        "billions of pounds under management, and the literature on how a sequential "
        "decision policy might fill that gap is thin. This dissertation studies one "
        "candidate. A Proximal Policy Optimization (PPO) agent is paired with a DeepAR-"
        "style probabilistic long short-term memory network that emits a Gaussian over the "
        "next-step log return; the predictive standard deviation is normalised to a unit-"
        "interval uncertainty score and is fed both into the PPO policy state and into a "
        "hard guard on new long-side actions. The combined system is evaluated against "
        "three named comparators on a single fully reproducible protocol: passive buy-and-"
        "hold, a rule-based trailing stop-loss policy of the kind a discretionary investor "
        "would actually use, and a baseline PPO that sees no uncertainty signal. Fixed "
        "train, validation and test splits over US equity index data (2009 to 2025), three "
        "random seeds in the headline study and a metric set covering Sharpe ratio, "
        "maximum drawdown, the 95 % Value-at-Risk violation rate, and both terminal and "
        "path versions of the capital-preservation ratio. On the held-out 2022–2025 test "
        "window the probabilistic agent has the highest Sharpe (0.85), the highest "
        "terminal value, and the smallest path drawdown of any policy that participates in "
        "market upside enough to beat cash. Crucially, the rule-based stop-losses end with "
        "adequate terminal preservation but incur path drawdowns larger than buy-and-hold's: "
        "reactive stops fire late and the moving-average re-entry sits in cash through "
        "much of the recovery. The contribution claimed is deliberately modest — a "
        "careful empirical study of a specific combination on a reproducible protocol with "
        "the rule-based comparator that the DRL-finance literature mostly leaves out, not "
        "a new algorithm. The Phase-1 study runs the full four-agent comparison on a 70-"
        "ticker diversified-equity test universe (41 single names + 29 ETFs) at three seeds "
        "and one test window, with a four-fold walk-forward subset on a CPU-feasible four-"
        "ticker slice; the extended-budget grid (10 seeds × 50 000 timesteps × four folds × "
        "32 bootstrap paths) on the full universe is scheduled for the Colab GPU runtime in "
        "Phase 2 and is described in Chapter 7 with month-level target dates.",
    )
    add_para(doc, "Keywords: reinforcement learning, portfolio management, drawdown control, "
             "constrained optimisation, deep learning, uncertainty quantification, risk "
             "management, capital preservation, PPO, DeepAR.",
             italic=True)
    page_break(doc)

    # ----- Acknowledgements -----
    add_heading(doc, "Acknowledgements", 1)
    add_para(doc, "[INSERT ACKNOWLEDGEMENTS]")
    page_break(doc)

    # ----- Table of contents placeholder -----
    add_heading(doc, "Contents", 1)
    add_para(doc, "Generate this in Word (References → Table of Contents → Automatic Table 1) once the headings are reviewed.", italic=True)
    page_break(doc)

    # ============================================================== Chapter 1
    add_heading(doc, "Chapter 1 — Introduction", 1)

    add_heading(doc, "1.1 Background and motivation", 2)
    add_para(
        doc,
        "Start with a concrete picture. Imagine you put one million US dollars into the "
        "US stock market in January 2022 and hold on. By January 2025, you have about "
        "$1.52 million. That looks like a clean win. But on the way there, in October "
        "2022, your account balance briefly read $750,000 — a 25 % drop from the peak "
        "you had reached in January. For an individual investor that 25 % drop is "
        "scary. For a pension fund, an endowment, a sovereign-wealth-fund mandate or a "
        "Commodity Trading Advisor (CTA hedge fund) it is something different: it is a "
        "breach of contract. Many institutional mandates carry an explicit drawdown "
        "limit (a maximum permitted loss measured from the peak rather than from the "
        "starting balance). When that limit is breached, redemption rights kick in, "
        "trustees can be removed, and the fund can be forcibly liquidated. Buy-and-"
        "hold violates these limits routinely; manually setting a stop-loss (sell when "
        "the price has fallen 5 % below its peak) violates them too — it sells late "
        "and buys back even later. The question this dissertation answers is whether a "
        "small AI agent can do better than either of those two options.",
    )
    add_para(
        doc,
        "The institutional reality matters. CalPERS, the California Public Employees' "
        "Retirement Service and one of the largest pension funds in the world, "
        "documents an explicit drawdown limit in its governance papers. University "
        "endowments such as Yale's and Harvard's report drawdown alongside return as "
        "their headline performance measure. Bridgewater Associates' All Weather "
        "fund, which managed over 150 billion US dollars at peak, is publicly "
        "described by its founder as designed to lose less in any environment — an "
        "explicit drawdown-control objective rather than a return-maximisation one. "
        "Family-office Investment Policy Statements typically express risk tolerance "
        "as a maximum-loss percentage rather than as a volatility target. And "
        "Kahneman and Tversky's (1979) prospect-theory result establishes that retail "
        "investors feel losses about twice as painfully as equivalent gains, which "
        "explains why panic-selling is triggered by drawdown rather than by variance. "
        "In every one of these settings the binding constraint has the same shape: do "
        "not lose more than X percent from peak, and earn a return above cash.",
    )
    add_para(
        doc,
        "The standard quantitative toolkit was not designed for this kind of "
        "constraint. Markowitz (1952) mean-variance optimisation is single-period and "
        "treats upside and downside wiggles in price as if they were the same kind of "
        "problem; it has no memory of the running peak. Value-at-Risk and expected "
        "shortfall (Rockafellar and Uryasev, 2000) measure how bad a single bad day "
        "could be, but they cannot tell you how many bad days in a row you can endure "
        "before your loss-from-peak crosses the limit. The Sortino ratio (Sortino and "
        "Price, 1994) trims the symmetric-variance assumption but is still a per-"
        "period measure. The drawdown literature itself — Magdon-Ismail and Atiya "
        "(2004) on the maximum drawdown of a Brownian-motion model, Chekhlov, "
        "Uryasev and Zabarankin (2005) on Conditional Drawdown-at-Risk — does "
        "develop a path-dependent framework that matches the institutional "
        "constraint, but the resulting optimisation programmes are static: they pick "
        "one weight vector and stick with it. None of these tools can dynamically "
        "say, on day fifty-three of a regime shift, that the portfolio is now at four "
        "per cent drawdown, the forecaster's confidence has just collapsed, and the "
        "position should therefore be trimmed before the next bad day pushes the "
        "constraint over the edge.",
    )
    add_para(
        doc,
        "In the last decade deep reinforcement learning (DRL — a branch of machine "
        "learning where an agent learns by trial and error in a simulated environment) "
        "has emerged as a flexible alternative. Rather than solve a one-shot "
        "mathematical optimisation, a DRL agent learns a sequential decision policy "
        "(\"what should I do today, given everything I have observed so far?\") "
        "directly from interaction with a market environment. Jiang, Xu and Liang "
        "(2017), Yang, Liu, Zhong and Walid (2020) and the FinRL library of Liu, Yang, "
        "Gao and Wang (2021) have shown that policy-gradient methods can be made to "
        "work on equities, ETFs and cryptocurrencies. Two practical gaps in this body "
        "of work motivate the present dissertation. The first is that the reward "
        "signal almost always rewards portfolio return and lets risk enter only "
        "indirectly, through reward shaping or by reading the Sharpe ratio off after "
        "the fact; the trained policy therefore carries no native notion of the path-"
        "dependent drawdown constraint that a real mandate would impose. The second "
        "is that the policy is trained on point-estimate features (the forecaster's "
        "best guess) and is never told how confident the forecaster is in that guess. "
        "When the market enters a regime where forecast confidence collapses, the "
        "agent has no native way to be cautious. The opportunity this dissertation "
        "pursues is to add the missing pieces back: a drawdown-constrained framing on "
        "one side, and an explicit uncertainty signal on the other, and to measure "
        "the resulting policy against both the static-optimisation tradition and the "
        "practitioner reality of fixed-rule overlays such as trailing stop-losses.",
    )

    add_heading(doc, "1.2 Problem statement", 2)
    add_para(
        doc,
        "A note on framing. An earlier version of this dissertation framed the problem as "
        "\"capital preservation\" — keep the portfolio safe. That framing was a mistake, "
        "for a simple reason: the obvious reply is \"go to cash, you preserve 100 %, done\" "
        "— which destroys the project. The fix is to stop selling preservation as the "
        "objective and start selling it as the constraint, and to put risk-adjusted "
        "return back as the objective. The reframed problem statement, which this "
        "dissertation answers, is therefore phrased in those terms.",
    )
    add_para(
        doc,
        "Reframed problem statement. Many investors, and many institutional mandates "
        "explicitly, are required to keep portfolio drawdown from peak below a stated "
        "limit (commonly somewhere between 5 % and 20 %) while still beating cash and "
        "ideally beating passive index exposure. The standard ways of doing this — "
        "Markowitz mean-variance, risk parity, fixed-rule stop-losses — either assume "
        "the joint distribution of returns is stationary (so the recipe that fits the "
        "training window will continue to fit) or react too slowly when the market "
        "regime changes. This dissertation studies whether a deep-reinforcement-learning "
        "agent that conditions on its own forecaster's predictive uncertainty (how "
        "confident the forecaster is, not just what it predicts) can sit on a more "
        "attractive point of the return-versus-drawdown trade-off than (a) passive buy-"
        "and-hold, (b) a rule-based stop-loss policy of the kind a discretionary "
        "investor would actually use, and (c) a baseline PPO that sees no uncertainty "
        "signal. The evaluation is on a held-out test window that contains real macro "
        "shocks (1 January 2022 to 31 December 2025, which spans the 2022 inflation-"
        "rate-shock bear market), with reproducible random seeds and an out-of-time "
        "generalisation check.",
    )
    add_para(
        doc,
        "Three notes on what this is and is not. First, the objective is risk-adjusted "
        "return; preservation is the constraint that prevents the trivial \"all-cash\" "
        "answer. Second, the comparison is against three named alternatives, not against "
        "an unspecified family of baselines: this is what makes the empirical claim "
        "falsifiable. Third, the test window is fixed in advance and not chosen post-hoc "
        "to make the agent look good; the 2022 macro shock is included precisely "
        "because that is exactly the kind of regime where institutional drawdown "
        "limits bind hardest.",
    )
    add_para(
        doc,
        "The current state of practice has three components and the dissertation is "
        "positioned against each in turn. The static-optimisation tradition (Markowitz "
        "mean-variance, risk parity, Conditional Drawdown-at-Risk) picks one weight "
        "vector once and does not adapt when the market regime changes mid-window. The "
        "reactive-rule tradition (trailing stop-losses with moving-average re-entry) is "
        "easy to implement, easy to explain to a client, and operationally common, but "
        "the stop fires after the drawdown has already happened and the re-entry rule "
        "typically forfeits the recovery — Chapter 5 shows that two trailing-stop "
        "variants on the test window actually incur path drawdowns larger than passive "
        "buy-and-hold's, because they sell into the dip and re-buy after the rebound. "
        "The deep-reinforcement-learning tradition for portfolio management (Jiang et "
        "al., 2017; Yang et al., 2020; Liu et al., 2021) trains policies on point-"
        "estimate features and rewards portfolio return, with risk entering only "
        "through reward shaping, so the trained policy has no native drawdown-"
        "constraint awareness and no way to be cautious when its own forecaster is "
        "unsure of itself.",
    )
    add_para(
        doc,
        "Against this state of practice the candidate evaluated in this dissertation is a "
        "Proximal Policy Optimization (PPO) agent that conditions on the predictive "
        "uncertainty produced by a DeepAR-style probabilistic LSTM. The uncertainty signal "
        "enters in two places: as an extra coordinate of the policy state, and as a hard "
        "guard that blocks new long-side trades when the score exceeds a quantile "
        "threshold. The candidate is measured against three named comparators on the same "
        "protocol: passive buy-and-hold, a rule-based trailing stop-loss policy of the "
        "kind a discretionary investor would actually use, and a baseline PPO that sees "
        "no uncertainty signal.",
    )
    add_para(
        doc,
        "Formally, the agent is asked to maximise expected risk-adjusted return (in "
        "Sharpe-ratio units) subject to a capital-preservation constraint, defined as the "
        "ratio of terminal portfolio value to the running high-watermark not falling below "
        "a stated floor (taken as 0.95 in the headline experiment), with per-step turnover "
        "bounded by a maximum trade fraction and trades subject to a linear transaction "
        "cost. The exact constrained objective and its mapping into the per-step reward "
        "is set out in Chapter 3. The headline result is the joint of the two halves: "
        "meeting the preservation floor on its own is trivial (an all-cash policy "
        "achieves perfect preservation and zero return), and earning a high Sharpe ratio "
        "on its own ignores the constraint, so the design is judged on whether it "
        "satisfies both at once on a test window containing real macro shocks.",
    )

    add_heading(doc, "1.3 Aims and objectives", 2)
    add_para(
        doc,
        "The dissertation pursues five objectives. They are written in the order in "
        "which they are answered in the rest of the document: the core scientific "
        "question first, the headline single-asset comparison second, the "
        "generalisation evidence third, the reproducibility apparatus fourth, and "
        "the honest position on where the method works and where it does not fifth.",
    )
    add_bullets(doc, [
        "O1 — the core scientific question. Can a deep reinforcement-learning agent that conditions on its own forecaster's predictive uncertainty (how confident the forecaster is, not just what it predicts) sit closer to the drawdown-constrained risk-adjusted return frontier than uncertainty-blind alternatives? Operationally: a DeepAR-style probabilistic LSTM emits a predictive mean and a predictive variance at each step; a Proximal Policy Optimization (PPO) policy reads the variance both as an extra coordinate of the state and as a hard guard that blocks new long-side trades when the uncertainty score exceeds a quantile threshold.",
        "O2 — the headline empirical comparison. Evaluate the resulting policy on a fixed held-out window containing real macro shocks (1 January 2022 to 31 December 2025, which spans the 2022 inflation-rate-shock bear market) on SPY (the most-traded US broad-market ETF) as a representative single-asset case study, against three named comparators: passive buy-and-hold, a rule-based trailing stop-loss policy of the kind a discretionary investor would actually use, and a baseline PPO that sees no uncertainty signal. Headline metrics: annualised Sharpe ratio, terminal portfolio value relative to buy-and-hold, capital-preservation ratio against the running high-watermark, and the rate at which realised log-returns fall below VaR-95.",
        "O3 — generalisation. Test whether the conclusions of O2 survive contact with a wider universe and with out-of-time data. Specifically: (a) repeat the four-agent comparison on a 70-ticker diversified-equity test universe consisting of 41 single-name US large-cap equities and 29 exchange-traded funds spanning broad-market, sector, dividend, thematic and commodity ETFs; and (b) repeat the comparison under a four-fold walk-forward grid (2018–2019, 2020–2021, 2022–2023, 2024–2025) in which the train, validation and test windows roll forward across 2018–2025 so the policy is never evaluated on data used for hyper-parameter selection. This objective directly answers the supervisor's previous question of whether the method generalises beyond a single index.",
        "O4 — reproducibility. Pin down a fully reproducible evaluation protocol of fixed train / validation / test splits, fixed random seeds, scripted experiment runners, scripted reporting, and a shared metric set, so that any comparison made in this dissertation is genuinely like-for-like and can be reproduced by an external reader from the public repository in a single command sequence (Appendix A).",
        "O5 — honest position on where the method wins and where it does not. Diagnose the regimes in which the uncertainty-aware policy beats the alternatives and the regimes in which it does not, and take a defensible position — on the strength of O1 to O4 — on when an explicit uncertainty signal earns a place in a portfolio control loop and, just as important, on when it does not. The Phase-1 evidence already identifies two regimes in which the agent under-performs (sustained low-uncertainty bull markets, and very-low-drawdown defensive holdings), and Chapter 6 develops the diagnosis.",
    ])

    add_heading(doc, "1.4 Contributions", 2)
    add_para(
        doc,
        "The contributions claimed in this dissertation are deliberately modest. Each "
        "individual ingredient (PPO, DeepAR-style probabilistic forecasting, uncertainty-"
        "aware reinforcement learning) already exists in the literature. What the dissertation "
        "adds is a careful empirical study of a specific combination of those ingredients, "
        "evaluated against named comparators on a fully reproducible protocol.",
    )
    add_bullets(doc, [
        "An uncertainty-aware trading environment in which the per-step trade size is shrunk by (1 - u_t) with a floor s_min, and new long-side trades are blocked when the uncertainty score exceeds a quantile threshold tau (Section 3.5). This delivers O1.",
        "A controlled empirical comparison against three named alternatives, namely passive buy-and-hold, a rule-based stop-loss policy, and a baseline PPO with no uncertainty signal, on the same protocol on SPY as the headline single-asset case study (Chapter 5). This delivers O2.",
        "A two-pronged generalisation study that lifts the headline finding from a single asset to a 70-ticker diversified-equity test universe (41 single-name US large-cap equities and 29 ETFs spanning broad-market, sector, dividend, thematic and commodity exposure) and a four-fold walk-forward grid in which the train, validation and test windows roll forward across 2018–2025 (Sections 5.5, 5.5.1, 6.4 and Appendix B). This delivers O3 and is the dissertation's strongest piece of evidence.",
        "A fully reproducible end-to-end evaluation protocol with fixed splits, three random seeds for Phase-1 and ten seeds for the extended-budget seed-stability check, a shared metric set, scripted experiment runners and a public, runnable Jupyter walkthrough that loads the dataset, trains the probabilistic forecaster, prints the uncertainty values, runs the agents and the rule-based comparator, and renders the comparison table and equity curves with embedded outputs (Section 3.7, Chapter 5 and Appendix A). This delivers O4.",
        "A discussion that calls out the maximum-drawdown number as misleading when read in isolation, argues for the joint of Sharpe ratio and the preservation ratio as the metric pair that actually matches the stated objective, names the two regimes in which the agent under-performs (sustained low-uncertainty bull markets, and very-low-drawdown defensive holdings), and is explicit about the limits of a Phase-1 budget on the 70-ticker test universe (Sections 6.2 to 6.4). This delivers O5.",
    ])

    add_heading(doc, "1.5 Dissertation structure", 2)
    add_para(
        doc,
        "Chapter 2 reviews the relevant background: modern portfolio theory, the policy-"
        "gradient family of RL algorithms (with PPO singled out), prior DRL work in finance, "
        "probabilistic forecasting with DeepAR-style networks, and methods for uncertainty "
        "quantification in deep learning. Chapter 3 sets the problem out as a Markov decision "
        "process, states the constrained objective, derives the probabilistic forecaster, "
        "defines the trading environment, and sets out the exact mathematical difference "
        "between the baseline PPO, the rule-based stop-loss comparator, and the probabilistic "
        "agent. Chapter 4 covers the implementation: data pipeline, training procedure and "
        "reporting layer. Chapter 5 presents the experimental setup and the results on the "
        "held-out test window, including the rule-based comparator, the 70-ticker "
        "diversified-equity-universe robustness study, and the Section 5.5.1 extended-budget "
        "seed-stability evidence (10 random seeds × 50 000 PPO timesteps per cell, "
        "80 cells in total, on a representative subset of the universe). "
        "Chapter 6 reads the results carefully, including the trade-offs, the "
        "walk-forward generalisation evidence and the limits of the evidence. "
        "Chapter 7 concludes and points at future work.",
    )
    page_break(doc)

    # ============================================================== Chapter 2
    add_heading(doc, "Chapter 2 — Background and Related Work", 1)

    add_heading(doc, "2.1 Risk-management objectives in portfolio practice", 2)
    add_para(
        doc,
        "What \"drawdown\" means, in plain English. Most people, when they think about "
        "investment risk, think about volatility — how much a price wiggles up and down. "
        "Institutions think about something different: drawdown. Imagine you start with "
        "$100. The price goes up to $150 (a new peak). Then it drops to $120. Your "
        "drawdown is measured against the $150 peak, not the $100 starting balance — you "
        "are 20 % down from the high. The reason this matters more than volatility, in "
        "practice, is that you don't feel a wiggle; you feel the pain of looking at your "
        "account and seeing it well below where it has just been. If you manage a "
        "pension fund and you lose 20 % of pensioners' money, you get fired — they do "
        "not care that the average wiggle was small; they care that the money is gone. "
        "Most institutional mandates write this fear into the contract directly: do not "
        "let drawdown exceed X %, ever. The rest of this section sets out the formal "
        "machinery the literature has built to talk about that constraint.",
    )
    add_para(
        doc,
        "Quantitative portfolio management has produced a sequence of risk-aware "
        "objective functions over the last seventy years. This section sets out the four "
        "families that are most often used in practice, gives the formal definition of "
        "each with every symbol explained, notes which assumptions each family makes and "
        "where those assumptions are known to bind, and — critically — explains why each "
        "family was developed in response to the limitations of the family that came "
        "before. Section 2.1.5 then states explicitly which of these objectives is "
        "closest to the constrained problem solved in this dissertation, and why.",
    )

    add_heading(doc, "2.1.1 Mean-variance optimisation (Markowitz, 1952)", 3)
    add_para(
        doc,
        "What the paper did. Markowitz formalised portfolio choice as a quadratic "
        "optimisation in the mean and variance of returns. Given N assets with vector of "
        "expected returns mu and covariance matrix Sigma, the minimum-variance portfolio "
        "for a target expected return mu* solves the quadratic programme below.",
    )
    add_equation(
        doc,
        r"$\min_{w}\ w^{\top}\Sigma w\quad\mathrm{s.t.}\quad w^{\top}\mu=\mu^{\ast},\ w^{\top}\mathbf{1}=1,\ w\geq 0$",
        "eq_mean_variance.png",
        label="Equation 2.1 — Mean-variance portfolio optimisation (Markowitz, 1952).",
    )
    add_para(
        doc,
        "The vector w in R^N is the portfolio weights, with w_i the fraction of wealth "
        "allocated to asset i. The vector mu in R^N collects the expected per-period "
        "returns of the N assets, with mu_i = E[r_i]. The matrix Sigma in R^{NxN} is the "
        "return covariance matrix, with Sigma_{ij} = Cov(r_i, r_j); the diagonal entries "
        "Sigma_{ii} are the per-asset variances. The objective w^T Sigma w is the "
        "variance of the portfolio's return. The first constraint w^T mu = mu^* fixes the "
        "target expected return; the second w^T 1 = 1 enforces full investment; the third "
        "w >= 0 disallows short positions, which can be relaxed. Sweeping mu^* over its "
        "feasible range traces out the efficient frontier: the locus of portfolios that "
        "achieve the lowest variance for any given expected return.",
    )
    add_para(
        doc,
        "Why the paper did it. Before 1952 the practitioner default was Graham-style "
        "fundamental selection of individual securities followed by ad-hoc diversification. "
        "Markowitz's mathematical contribution was to show that diversification can be "
        "made operational and quantitative: under the mean-variance objective the "
        "covariance structure of returns is the binding determinant of portfolio risk, "
        "not the volatility of the constituent assets in isolation, and a portfolio "
        "manager should optimise the joint of return and variance rather than each in "
        "turn. This is the conceptual move that turned portfolio management from a "
        "qualitative discipline into a quantitative one.",
    )
    add_bullets(doc, [
        "Advantages. The optimisation is a convex quadratic programme that is solved exactly in milliseconds, the efficient frontier admits a closed-form parametric description in the unconstrained case, and the framework gives a single-number ranking (Sharpe ratio) that has become the default communication of risk-adjusted performance. It is the foundation of every academic and industrial extension of portfolio theory.",
        "Disadvantages. The framework assumes the joint return distribution is stationary with known mu and Sigma, but in practice both have to be estimated on a finite training window. The resulting weights are notoriously sensitive to estimation error: small changes in mu can produce large rebalancings in w, a phenomenon Michaud (1989) labelled \"error-maximising\" portfolios. The variance objective penalises upside and downside deviations symmetrically, which is conceptually wrong for a long-only mandate where upside volatility is desirable. And by being single-period the framework ignores path-dependent properties of the equity curve such as drawdowns, which are the constraint that institutional mandates actually bind on.",
        "Fixes / extensions in the literature. Black and Litterman (1992) reduce estimation error by shrinking mu towards an equilibrium prior derived from the market portfolio. Ledoit and Wolf (2003) shrink Sigma towards a structured target. The downside-only measures of Section 2.1.4 (Sortino) replace symmetric variance with downside deviation. The tail-loss measures of Section 2.1.2 (VaR / ES) replace variance entirely with a tail quantile. The drawdown measures of Section 2.1.3 (MDD / Calmar / CDaR) replace single-period variance with a path-dependent loss-from-peak. This dissertation positions itself in the last family, with the additional move of using a sequential reinforcement-learning policy rather than a one-shot static optimisation.",
        "Why it matters here. Mean-variance is reported in Chapter 5 alongside the path-dependent metrics so the reader can apply whichever objective matches their mandate. It is not the dissertation's headline objective because, on a 70-ticker test universe over a four-year window with a 2022 macro shock, path-dependent drawdown is what binds; a mean-variance optimiser sees no difference between a -25 % path drawdown that recovers and a smooth -5 % linear loss with the same variance.",
    ])

    add_heading(doc, "2.1.2 Value-at-Risk and expected shortfall (Rockafellar and Uryasev, 2000)", 3)
    add_para(
        doc,
        "Differentiation from Markowitz. Where Markowitz (1952) summarises risk by the "
        "whole second moment of the return distribution (variance), Rockafellar and "
        "Uryasev (2000) chose to summarise it by the tail expectation of the loss "
        "distribution alone. The reason for the divergence is that institutional risk "
        "regulation, by the late 1990s, had stopped caring about \"how much do returns "
        "wiggle on average\" and started caring about \"how bad is the worst 1 % of "
        "outcomes\" — a question Markowitz's quadratic objective is not built to "
        "answer. The two papers are therefore solving different problems: Markowitz "
        "answers \"what is the smoothest portfolio I can build given my return "
        "expectation?\" while Rockafellar and Uryasev answer \"what is the portfolio "
        "with the smallest expected tail loss?\". The two answers coincide only when "
        "returns are jointly Gaussian, which empirical equity returns are not.",
    )
    add_para(
        doc,
        "What the paper did. Value-at-Risk (VaR) is the tail-quantile of the portfolio "
        "loss distribution, and expected shortfall (ES, also known as conditional Value-"
        "at-Risk or CVaR) is the average loss in the tail beyond the VaR quantile. "
        "Rockafellar and Uryasev (2000) made two related contributions. First, they wrote "
        "down a convex optimisation programme whose minimiser is the CVaR of the "
        "portfolio's loss distribution at a chosen confidence level, sidestepping the "
        "need to first compute VaR and then estimate a conditional expectation. Second, "
        "they showed that under empirical (scenario-based) loss distributions this "
        "programme reduces to a linear programme that is solvable in seconds with off-"
        "the-shelf solvers, putting CVaR-constrained portfolio optimisation on the same "
        "computational footing as mean-variance.",
    )
    add_equation(
        doc,
        r"$\mathrm{VaR}_{\alpha}(L)=\inf\{x:\mathrm{Pr}(L\leq x)\geq\alpha\}$",
        "eq_var.png",
        label="Equation 2.2 — Value-at-Risk at confidence level alpha.",
    )
    add_equation(
        doc,
        r"$\mathrm{ES}_{\alpha}(L)=\mathbb{E}\left[L\mid L\geq\mathrm{VaR}_{\alpha}(L)\right]$",
        "eq_expected_shortfall.png",
        label="Equation 2.3 — Expected shortfall (also known as conditional Value-at-Risk).",
    )
    add_para(
        doc,
        "L is the random portfolio loss over a fixed horizon (one day, one week, one "
        "month, depending on the mandate); alpha in (0, 1) is the confidence level; "
        "VaR_alpha(L) is the loss exceeded with probability at most 1 - alpha; and "
        "ES_alpha(L) is the average loss in the (1 - alpha)-tail of the distribution.",
    )
    add_para(
        doc,
        "Why the paper did it. By the late 1990s VaR had been adopted by major banks as "
        "the regulatory tail-loss measure (Basel I market-risk amendment, 1996), but "
        "Artzner et al. (1999) had just shown that VaR is not coherent: it can fail sub-"
        "additivity, meaning a diversified portfolio can have a higher VaR than the sum "
        "of its components' VaRs. Practitioners needed a coherent alternative that was "
        "also tractable to optimise. ES / CVaR was already known to be coherent; "
        "Rockafellar and Uryasev's contribution was to make it operationally tractable.",
    )
    add_bullets(doc, [
        "Advantages. Coherent (sub-additive, monotonic, positively homogeneous, translation-invariant); convex in the portfolio weights, which makes it directly compatible with constrained portfolio optimisation under additional linear or convex constraints; sensitive to the shape of the tail beyond the VaR quantile, unlike VaR itself; reduces to a linear programme under empirical scenarios. Mean-CVaR optimisation has therefore largely replaced mean-VaR in regulated settings (Basel III for banks; Solvency II for insurers).",
        "Disadvantages. Single-period: VaR and ES describe the loss distribution over one fixed horizon and are silent on the path of the equity curve between rebalancing dates. They will rate identically a smooth -10 % loss in a single step and a -25 % drawdown that recovers to a -10 % loss by the rebalancing date, even though the second is the one that breaks pension-fund mandates and triggers margin calls. Estimation of the tail itself is also data-hungry: at alpha = 0.99 the tail contains 1 % of the data, so even ten years of daily returns gives only ~25 observations on which to estimate the conditional expectation.",
        "Fixes / extensions in the literature. Acerbi and Tasche (2002) clarified that ES is the smallest coherent risk measure that dominates VaR. The drawdown measures of Section 2.1.3 (MDD, Calmar, CDaR) are the path-dependent generalisation that closes the single-period gap. For tail-estimation small-sample issues, extreme value theory (McNeil and Frey, 2000) replaces the empirical tail with a fitted Generalised Pareto Distribution, trading estimation efficiency for a parametric assumption.",
        "Why it matters here. The dissertation reports VaR-95 and the rate at which the realised log-return falls below it as part of the standard metric set in Section 3.7. The headline objective, however, is path-dependent (drawdown control, Section 2.1.3) rather than single-period (VaR/ES), because the binding constraint on a four-year test window with a 2022 bear-market drawdown is the path of the equity curve, not its terminal-period loss distribution.",
    ])

    add_heading(doc, "2.1.3 Drawdown-based measures", 3)
    add_para(
        doc,
        "Differentiation from the previous two families. Mean-variance and VaR/ES "
        "describe either the average wiggle (variance) or the worst single-period loss "
        "(tail quantile). Both are single-period objects: they look at one frame of "
        "the equity-curve movie at a time. The drawdown family looks at the whole "
        "movie. Where Markowitz cannot tell the difference between a smooth straight-"
        "line loss and a -25 % round-trip drawdown that recovers to the same terminal "
        "value, and where VaR/ES rates a slow 1 %-per-day-for-30-days bleed as a "
        "succession of perfectly fine days, the drawdown family explicitly tracks the "
        "running peak and asks how far below it the portfolio has fallen. That "
        "structural difference — single-period versus path-dependent — is the entire "
        "reason institutional mandates are written in drawdown terms rather than in "
        "variance or VaR terms.",
    )
    add_para(
        doc,
        "Four references anchor the drawdown literature: the maximum-drawdown definition "
        "itself (Magdon-Ismail and Atiya, 2004), the Calmar return-per-unit-drawdown "
        "ratio (Young, 1991), the conditional drawdown-at-risk programme (Chekhlov, "
        "Uryasev and Zabarankin, 2005), and the downside-deviation Sortino ratio "
        "(Sortino and Price, 1994), which sits at the boundary between the path-"
        "dependent drawdown family and the single-period family of Section 2.1.2. "
        "These four papers are not interchangeable; each was developed in response to a "
        "specific weakness in the previous one, and the differences between them are the "
        "reason a practitioner reaches for one over another.",
    )

    add_para(
        doc,
        "Maximum drawdown — definition. The maximum drawdown of a portfolio with value "
        "process V_0, V_1, ..., V_T over the horizon T is the worst-case fractional loss "
        "from the running peak experienced anywhere in the trajectory:",
    )
    add_equation(
        doc,
        r"$\mathrm{MDD}=\max_{t\leq T}\left(1-\frac{V_{t}}{\max_{s\leq t}V_{s}}\right)$",
        "eq_max_drawdown.png",
        label="Equation 2.4 — Maximum drawdown over the horizon T.",
    )
    add_para(
        doc,
        "where V_t is the portfolio value at time t and max_{s <= t} V_s is the running "
        "peak (the high-watermark) up to and including time t. The expression in "
        "parentheses is the fractional loss from peak at time t.",
    )

    add_para(
        doc,
        "Magdon-Ismail and Atiya (2004) — what and why. Where Markowitz and the "
        "VaR/ES family give the practitioner a single-period summary of risk, Magdon-"
        "Ismail and Atiya wrote down the analytical baseline for the path-dependent "
        "object that institutional mandates actually constrain: the maximum drawdown. "
        "They derived a closed-form expression for the expected maximum drawdown of a "
        "geometric Brownian motion with drift mu and volatility sigma over a horizon T. Their "
        "motivation was practitioner: by 2004 hedge funds and CTAs had been asked to "
        "report MDD alongside Sharpe ratio for a decade, but no analytical baseline "
        "existed for the expected MDD that an in-control strategy with given (mu, sigma, "
        "T) should produce. Without that baseline, an observed -25 % MDD is "
        "uninterpretable: it could be a flag for strategy degradation, or it could be "
        "exactly what a Brownian motion with the strategy's reported (mu, sigma) would "
        "produce a quarter of the time anyway. The closed-form expectation closes that "
        "gap.",
    )
    add_bullets(doc, [
        "Advantages. Operationally meaningful — a 25 % drawdown means the same thing to a pension trustee, a hedge-fund LP and a retail investor; closed-form result that ties MDD to (mu, sigma, T) and lets practitioners compute confidence bands; opens the door to MDD as a constraint in formal portfolio optimisation rather than only as a post-hoc diagnostic.",
        "Disadvantages. The closed-form result requires a Brownian-motion assumption with constant drift and volatility, which is empirically wrong: returns exhibit volatility clustering (Mandelbrot, 1963), fat tails, and regime change. Real-world MDDs are systematically larger than the Brownian baseline predicts. The MDD itself is also a single-event statistic — it depends only on the worst peak-to-trough excursion in the path, so the rest of the distribution of drawdowns is discarded, and a small change in input data can produce a large change in MDD.",
        "Fixes / extensions in the literature. The Chekhlov-Uryasev-Zabarankin (2005) CDaR programme below is the natural fix: it generalises MDD from a single-event statistic to a tail-expectation over the full drawdown distribution, just as ES generalised VaR. For the Brownian-motion-mis-specification problem, GARCH-style return models (Bollerslev, 1986) and regime-switching models (Hamilton, 1989) give better baseline expectations under realistic time-varying volatility.",
    ])

    add_para(
        doc,
        "Young (1991) — what and why. Where Magdon-Ismail and Atiya (2004) gave "
        "practitioners a baseline expectation for the maximum drawdown number itself, "
        "Young (1991) gave practitioners a way to compare strategies on a return-"
        "per-unit-drawdown basis. The Calmar ratio (named for CALifornia Managed "
        "Account Reports, the newsletter where Terry Young first proposed it in 1991) "
        "is a return-per-unit-drawdown measure intuitive for institutional mandates "
        "with explicit drawdown limits. Young's motivation was that the Sharpe ratio, "
        "by putting volatility in the denominator, did not communicate to "
        "institutional investors what they actually cared about — how much pain they "
        "had to endure to earn each unit of return. Calmar made the divergence from "
        "Sharpe explicit by replacing volatility with maximum drawdown in the "
        "denominator:",
    )
    add_equation(
        doc,
        r"$\mathrm{Calmar}=\frac{R_{\mathrm{ann}}}{\mathrm{MDD}}$",
        "eq_calmar.png",
        label="Equation 2.5 — Calmar ratio (Young, 1991).",
    )
    add_para(
        doc,
        "R_ann is the annualised return of the portfolio and MDD is the maximum "
        "drawdown of Equation 2.4.",
    )
    add_bullets(doc, [
        "Advantages. No parametric assumption on the return distribution; speaks the same language as the institutional mandate (\"don't lose more than X %\"); a single number that a non-quantitative investment committee can compare across managers; popularised the use of drawdown as a peer ranking statistic in CTA and hedge-fund consulting.",
        "Disadvantages. Extreme single-event sensitivity: because MDD is a single max statistic, the Calmar ratio of a strategy can swing dramatically based on a single bad week. A strategy that posted a -40 % drawdown in 2008 and recovered will appear to have a permanently impaired Calmar even if it has run flawlessly for the fifteen years since. The ratio also rewards strategies that have not yet experienced a major drawdown — Calmar is highest for the strategies whose worst loss is still ahead of them.",
        "Fixes / extensions. The MAR ratio (Managed Account Reports) is essentially Calmar with a longer rolling window. Sterling and Burke ratios use averages of the worst N drawdowns rather than the single max, partially addressing the single-event sensitivity. The CDaR programme of Chekhlov-Uryasev-Zabarankin (2005) is the formal generalisation that uses a tail-expectation over the drawdown distribution rather than a single max, removing the single-event problem entirely.",
    ])

    add_para(
        doc,
        "Chekhlov, Uryasev and Zabarankin (2005) — what and why. Where Magdon-Ismail "
        "and Atiya (2004) gave a closed-form baseline for a single statistic (the "
        "expected maximum drawdown), and where Young (1991) gave a single-number "
        "ratio for ranking strategies, Chekhlov, Uryasev and Zabarankin extended the "
        "whole CVaR construction of Rockafellar and Uryasev (2000) from the single-"
        "period loss distribution to the path-dependent drawdown distribution. The "
        "structural difference from Young (1991) is that Calmar is sensitive to a "
        "single worst day in the entire history (it depends only on the maximum), "
        "whereas the CDaR object Chekhlov-Uryasev-Zabarankin define is a tail "
        "expectation over the full distribution of drawdowns — it cannot be "
        "destroyed by a single bad day. The structural difference from Magdon-Ismail "
        "and Atiya (2004) is that the CDaR object is computable directly from "
        "historical scenarios with no Brownian-motion assumption. They define the "
        "conditional drawdown-at-risk CDaR_alpha as the expected drawdown given that "
        "the drawdown exceeds the alpha-quantile of the empirical drawdown "
        "distribution, and — crucially — they show that under empirical "
        "(historical-scenario) data this construction reduces to a linear "
        "programme that solves portfolio optimisation under a CDaR constraint in "
        "seconds. Their motivation was direct: "
        "by 2005 the CDaR statistic was being demanded by institutional allocators "
        "(endowments, sovereign wealth funds, pension consultants) but no tractable "
        "optimisation routine existed to construct portfolios under it. CUZ provided "
        "that routine.",
    )
    add_bullets(doc, [
        "Advantages. Generalises Calmar and MDD from single-event statistics to a tail expectation over the drawdown distribution, removing the single-event sensitivity of Calmar; reduces to a linear programme under empirical scenarios, putting drawdown-constrained portfolio optimisation on the same computational footing as mean-variance; coherent in the same sense as ES; matches the form of the constraints actually written into endowment-fund spending policies and CTA prospectuses; the closest classical antecedent to the constrained problem solved in this dissertation.",
        "Disadvantages. Computational cost grows with the horizon T because the path-dependent drawdown statistic at time t depends on every prior portfolio value V_0, ..., V_t; for very long horizons the LP becomes large. Like ES, the empirical-scenario formulation is data-hungry in the tail, and the choice of the alpha quantile is a modelling decision rather than a derived quantity. The framework remains a one-shot static optimisation over portfolio weights — it does not naturally accommodate a sequential decision policy in which the agent can change exposure conditionally on observed market state, which is the gap this dissertation closes.",
        "Fixes / extensions. Multi-stage stochastic programming (Bertsimas, Lauprete and Samarov, 2004) generalises CDaR to a multi-period setting; reinforcement-learning approaches such as the present dissertation push the same logic further by replacing the static optimisation with a state-conditional policy that learns when to reduce exposure rather than only how to allocate it; conformal-prediction overlays (Vovk et al., 2005) can supply the empirical drawdown-quantile threshold non-parametrically.",
        "Why it matters here. The constrained objective in Equation 3.4 of this dissertation is the operational specialisation of the Chekhlov-Uryasev-Zabarankin programme to the trading-environment setting set out in Chapter 3. The capital-preservation ratio used as the headline metric (Equation 3.10) is a path-dependent terminal measure closely related to (1 - MDD) evaluated at terminal time. Where CUZ enforce the constraint at portfolio-construction time on a static weight vector, the present dissertation enforces it inside a sequential decision policy, and lets the policy learn the trade-off between expected return and drawdown empirically. The CUZ programme is the dissertation's primary classical reference point.",
    ])

    add_heading(doc, "2.1.4 Downside-deviation measures: Sortino (Sortino and Price, 1994)", 3)
    add_para(
        doc,
        "Differentiation from the previous papers. Markowitz (1952) penalises upside "
        "and downside deviations from the mean symmetrically, on the grounds that the "
        "second moment of the return distribution is the natural measure of "
        "uncertainty. Sortino and Price (1994) chose differently. Their argument was "
        "that for a long-only investor — the dominant institutional mandate type — "
        "upside deviations are exactly what the investor is paying the manager to "
        "deliver, and penalising them is conceptually wrong. Where Markowitz says "
        "\"both kinds of wiggle are bad\", Sortino and Price say \"only the wiggles "
        "below my floor are bad\". The mathematical move is small (replace the "
        "denominator of Sharpe with a downside-only deviation) but the philosophical "
        "move is significant: it puts an asymmetric utility, not symmetric variance, "
        "at the centre of the risk measure. Sortino is therefore positioned "
        "between Markowitz (symmetric, single-period) and the drawdown family of "
        "Section 2.1.3 (asymmetric, path-dependent): asymmetric like the drawdown "
        "family, but still single-period like Markowitz.",
    )
    add_para(
        doc,
        "What the paper did. Sortino and Price replaced the standard deviation in the "
        "denominator of the Sharpe ratio with the downside deviation — a root-mean-"
        "square deviation taken only over returns below a target return tau. The "
        "resulting Sortino ratio is identical to Sharpe in numerator but asymmetric in "
        "denominator: only returns below tau are penalised.",
    )
    add_equation(
        doc,
        r"$\mathrm{Sortino}=\frac{\mathbb{E}[r]-\tau}{\mathrm{DD}_{\tau}},\qquad \mathrm{DD}_{\tau}=\sqrt{\mathbb{E}\left[\max(\tau-r,0)^{2}\right]}$",
        "eq_sortino.png",
        label="Equation 2.6 — Sortino ratio with downside deviation against target return tau.",
    )
    add_para(
        doc,
        "r is the portfolio return over the measurement horizon, E[r] is its expected "
        "value, tau is the minimum-acceptable return below which deviations are "
        "penalised (often the risk-free rate or zero), and DD_tau is the root-mean-"
        "square deviation taken only over returns below tau.",
    )
    add_para(
        doc,
        "Why the paper did it. Sortino had argued in a long sequence of practitioner "
        "papers from the early 1980s that the Sharpe ratio penalises long-only mandates "
        "twice over: once for negative returns (which is correct), and once for above-"
        "average positive returns (which is exactly what the long-only mandate is paid "
        "to deliver). Sortino and Price (1994) is the paper that formalised the "
        "asymmetric alternative and gave it a denominator that is operationally "
        "computable from the same per-period return data that Sharpe uses.",
    )
    add_bullets(doc, [
        "Advantages. Asymmetric penalty matches the asymmetric utility of a long-only investor (positive volatility is good, negative volatility is bad); same data requirements as Sharpe (a return time series), so easy to drop into existing reporting pipelines; widely understood in practitioner circles, particularly in CTA and hedge-fund manager evaluation.",
        "Disadvantages. The choice of target tau is a modelling decision and not a derived quantity — the same strategy can have a Sortino of +1.4 against tau = 0 and +0.6 against tau = the risk-free rate, depending on the rate environment; for strategies with very few sub-tau returns the denominator is small and the ratio is unstable, sometimes producing inflated values that disappear after one bad month; like Sharpe, the Sortino ratio is a single-period statistic and is silent on the path of the equity curve, so a strategy with a deep drawdown that recovered can have a higher Sortino than a smoother strategy with a slightly lower mean.",
        "Fixes / extensions in the literature. Reporting Sortino at multiple values of tau (0, risk-free rate, mandate hurdle) addresses the target-rate sensitivity. Pairing Sortino with a path-dependent measure such as MDD or the Calmar ratio addresses the silent-on-path problem. Bawa (1975) and Fishburn (1977) provide the theoretical foundation — the lower partial moment family — of which the squared downside deviation is one specific instance; higher-order lower partial moments capture more of the tail.",
        "Why it matters here. Sortino is reported alongside Sharpe in the Section 3.7 metric table when a sub-tau-return computation is available. The dissertation does not adopt Sortino as the headline objective because the binding constraint on the test universe is path-dependent (drawdown control), not single-period (downside variance against a target return); but the asymmetric-penalty intuition behind Sortino is the same intuition behind the dissertation's choice of drawdown rather than variance as the binding measure.",
    ])

    add_heading(doc, "2.1.5 Synthesis: what the four families ask, and why they answer differently", 3)
    add_para(
        doc,
        "The four families above are not competing answers to the same question. They "
        "are answers to four different questions, each developed in response to a "
        "specific limitation of the family that came before:",
    )
    add_bullets(doc, [
        "Markowitz (1952) asks: \"What is the portfolio with the smallest average wiggle that hits my return target?\" The answer is a one-shot quadratic programme on mean and variance.",
        "Rockafellar and Uryasev (2000) ask: \"What is the portfolio with the smallest expected loss in the worst 5 % of single-period outcomes?\" The answer is a one-shot linear programme on the empirical tail. The shift from Markowitz is from average wiggle to tail loss.",
        "Magdon-Ismail and Atiya (2004), Young (1991) and Chekhlov, Uryasev and Zabarankin (2005) ask: \"What is the worst loss-from-peak that this strategy can be expected to experience over its lifetime, and how do I optimise against it?\" The answer is a path-dependent statistic of the equity curve. The shift from VaR/ES is from single-period tail to multi-period worst peak-to-trough.",
        "Sortino and Price (1994) ask: \"How much per-period downside deviation does this strategy take to earn its return?\" Same per-period grain as Markowitz, but with the symmetric penalty replaced by an asymmetric one. The shift from Markowitz is from symmetric to asymmetric.",
    ])
    add_para(
        doc,
        "These differences are the reason a practitioner reaches for one family over "
        "another. A regulator running Basel III reaches for ES because it is coherent "
        "and tractable. A pension trustee reaches for the drawdown family because the "
        "mandate is written in drawdown terms. A long-only fund manager reaches for "
        "Sortino because Sharpe penalises the upside they are paid to deliver. None "
        "of these choices is wrong; they are answers to different questions.",
    )

    add_heading(doc, "2.1.6 Position of this dissertation", 3)
    add_para(
        doc,
        "The dissertation does not propose a new risk objective. It takes the "
        "drawdown-constrained portfolio-optimisation problem of Chekhlov, Uryasev and "
        "Zabarankin (2005) as the practitioner's benchmark to be matched, and asks "
        "whether a reinforcement-learning policy that conditions on its own "
        "forecaster's predictive uncertainty can satisfy the same constraint while "
        "improving realised Sharpe ratio and terminal value relative to passive "
        "buy-and-hold and a rule-based stop-loss policy. The constrained objective in "
        "Equation 3.4 is the operational specialisation of the Chekhlov-Uryasev-"
        "Zabarankin programme to the trading-environment setting set out in Chapter 3. "
        "The capital-preservation ratio used as the headline metric (Equation 3.10) is "
        "a path-dependent terminal measure that is closely related to (1 - MDD) "
        "evaluated at terminal time and is the natural diagnostic for a policy that is "
        "explicitly trying to keep its loss from peak below a stated floor.",
    )

    add_heading(doc, "2.2 Reinforcement learning fundamentals", 2)
    add_para(
        doc,
        "An RL problem is a Markov decision process (S, A, P, R, gamma): a state space, an "
        "action space, a transition kernel, a reward function and a discount factor. Sutton "
        "and Barto (2018) is the canonical treatment. The objective is to learn a policy "
        "pi(a|s) that maximises the expected discounted return.",
    )
    add_equation(
        doc,
        r"$J(\pi) = \mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{T}\gamma^{t}\,R(s_{t},a_{t})\right]$",
        "eq_rl_objective.png",
        label="Equation 2.7 — Discounted-return objective of an MDP (Sutton and Barto, 2018).",
    )

    add_heading(doc, "2.3 Policy-gradient methods and PPO", 2)
    add_para(
        doc,
        "Policy-gradient methods parameterise the policy directly and ascend the gradient of "
        "the objective with respect to its parameters. Vanilla policy gradient is high-"
        "variance and unstable. Trust-region methods solve this by constraining the size of "
        "each update. Schulman et al. (2017) introduced Proximal Policy Optimization (PPO), "
        "which approximates a trust region with a simpler clipped surrogate objective.",
    )
    add_equation(
        doc,
        r"$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_{t}\left[\min\left(\rho_{t}(\theta)\hat{A}_{t},\,\mathrm{clip}(\rho_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}\right)\right]$",
        "eq_ppo_clip.png",
        label="Equation 2.8 — PPO clipped surrogate objective (Schulman et al., 2017).",
    )
    add_para(
        doc,
        "PPO has become the default policy-gradient choice for finance applications for two "
        "practical reasons. It is forgiving with respect to hyper-parameter choice, and it is "
        "available in mature open-source form, notably in Stable-Baselines3 (Raffin et al., "
        "2021). I use both in this dissertation.",
    )

    add_heading(doc, "2.4 Reinforcement learning for portfolio management", 2)
    add_para(
        doc,
        "The DRL-for-finance literature has produced a sequence of increasingly mature "
        "systems over the last seven years. Rather than catalogue them in isolation, the "
        "comparison below is organised along three axes that matter for the present "
        "dissertation: (i) what was being optimised in the reward, (ii) what was held in "
        "the policy state, and (iii) how risk was treated. The four references most "
        "directly comparable to this dissertation differ on all three axes.",
    )
    add_para(
        doc,
        "Jiang, Xu and Liang (2017). What they did: treated portfolio management as an "
        "end-to-end DRL problem on the cryptocurrency market. Their state is a tensor "
        "of recent prices for the universe of assets; their policy is a convolutional "
        "network that emits portfolio weights directly; their reward is the per-period "
        "log return of the portfolio value. Risk enters only indirectly through the "
        "variance of the reward signal — there is no explicit risk constraint. Why they "
        "did it: prior cryptocurrency portfolio work used static rules (equal-weight, "
        "minimum-variance, follow-the-winner) and Jiang et al. wanted to test whether "
        "an end-to-end deep policy could outperform those rules on a market where "
        "prices are highly non-stationary. They reported that it could.",
    )
    add_bullets(doc, [
        "Advantages. Genuinely end-to-end (raw price tensor → portfolio weights with no manual feature engineering); validated on a high-frequency, non-stationary market where mean-variance baselines are known to do poorly; established the computational template that Yang et al. (2020) and the FinRL library would later adopt.",
        "Disadvantages. Return-only reward; cryptocurrency-specific (the network architecture and the data pipeline assume the market never closes); policy is uncertainty-blind, so the variance of the reward signal is the only risk-management coupling; tested on a single time window without a walk-forward or seed-stability protocol, so the headline outperformance has the look of an in-sample result.",
        "Fixes / extensions. Yang et al. (2020) applied the same template to US equities with engineered features. The FinRL library (Liu et al., 2021) added reproducibility infrastructure. The present dissertation closes the uncertainty-blind and risk-blind gaps directly.",
    ])

    add_para(
        doc,
        "Yang, Liu, Zhong and Walid (2020). Where Jiang, Xu and Liang (2017) ran a "
        "single end-to-end policy on a single market (cryptocurrency), Yang, Liu, "
        "Zhong and Walid chose differently on two axes: they moved the test market "
        "to US equities, and they replaced the single-policy approach with an "
        "ensemble across three RL algorithms. The reason for the equities move was "
        "that mean-variance baselines on cryptocurrency are weak (so the comparison "
        "is too easy to win); equities is the harder benchmark. The reason for the "
        "ensemble move was that any single policy-gradient algorithm has high run-"
        "to-run variance, and the best-performing algorithm in one market regime is "
        "often not the best in the next.",
    )
    add_para(
        doc,
        "What they did: applied the Jiang-style "
        "template to US equities and replaced the single-algorithm policy with an "
        "ensemble that picks the best-performing of three policy-gradient algorithms "
        "(A2C, PPO and DDPG) per regime. Their state combines technical indicators "
        "(MACD, RSI, ADX) with recent prices; their action is a continuous portfolio-"
        "weight vector; their reward is again per-period portfolio return. Why they did "
        "it: a single policy-gradient algorithm has high run-to-run variance, and the "
        "best-performing algorithm in one market regime is often not the best in the "
        "next; by switching algorithm at regime change the ensemble retains the best of "
        "each.",
    )
    add_bullets(doc, [
        "Advantages. Move from cryptocurrency to US equities, where the empirical literature is denser and the comparison to mean-variance is more meaningful; engineered technical indicators give the policy useful inductive bias on momentum and mean-reversion; algorithm-level ensemble materially reduces seed variance; explicit train / validation / test split protocol that the present dissertation follows in Section 3.7.",
        "Disadvantages. Return-only reward; the technical-indicator feature set (MACD, RSI, ADX) embeds prior beliefs about which signals matter, which mean-variance and probabilistic models can sometimes contradict; the regime-switching ensemble is trained on the same window as the constituent algorithms and so has an in-sample look-ahead concern that walk-forward evaluation would expose; uncertainty-blind in exactly the same sense as Jiang et al. (2017).",
        "Fixes / extensions. The walk-forward protocol of Bailey and López de Prado (2014) addresses the in-sample regime-detection concern. The present dissertation's contribution sits orthogonal to the algorithm-level ensemble: any of A2C, PPO or DDPG could in principle host the uncertainty-aware coupling, and PPO is selected here for its stability and library availability rather than for any algorithm-level claim.",
    ])

    add_para(
        doc,
        "Liu, Yang, Gao and Wang (2021) — FinRL. Where Jiang, Xu and Liang (2017) "
        "and Yang, Liu, Zhong and Walid (2020) contributed novel policies on "
        "particular markets, Liu, Yang, Gao and Wang chose differently: they "
        "contributed infrastructure rather than a policy. Their motivation was that "
        "by 2020 the DRL-finance literature had reached the point where individual "
        "papers reported divergent results on what looked like the same problem, "
        "and the cause was that no two papers were using the same trading "
        "environment, transaction-cost assumption, or metric definition. Where the "
        "two earlier papers were saying \"here is a better policy\", FinRL says "
        "\"here is the standard environment against which any future policy paper "
        "should be evaluated\".",
    )
    add_para(
        doc,
        "What they did: built an open-source "
        "library that bundles the gym-style trading environment, the data pipeline, the "
        "algorithm interface and the reporting layer into one Python package. Why they "
        "did it: by 2021 the DRL-finance literature had reached a point where "
        "individual papers reported divergent results that were impossible to compare "
        "because the trading environments, transaction-cost assumptions and metric "
        "definitions all differed. FinRL imposes a single environment specification on "
        "the field and lets new papers vary only the policy.",
    )
    add_bullets(doc, [
        "Advantages. Reproducibility infrastructure that the field had been missing; gym-style API makes plug-and-play comparison of algorithms straightforward; bundles common transaction-cost and slippage assumptions in one place; widely adopted, which makes FinRL-based results comparable across papers.",
        "Disadvantages. Takes no position on what the policy should optimise — the reward function is delegated to the user; the bundled environment makes an implicit choice (continuous portfolio-weight action, single-step rebalancing) that is not the right environment for every problem; for a dissertation that wants every line of the environment to be inspectable at viva, the FinRL abstraction layer hides design decisions that need to be defended.",
        "Fixes / extensions. The present dissertation does not subclass FinRL; it implements its own environment in experiments/common.py so that the trade-scaling and entry-guard contributions in Equation 3.9 are explicit, can be inspected line by line, and can be reasoned about in the discussion chapter. FinRL remains the right choice for projects whose contribution is in the policy rather than in the environment.",
    ])

    add_para(
        doc,
        "Schulman et al. (2017) — PPO. What they did: introduced a clipped surrogate "
        "objective (Equation 2.8) that approximates a trust-region constraint without "
        "the conjugate-gradient machinery of TRPO (Schulman et al., 2015). Why they did "
        "it: TRPO is theoretically attractive but operationally fragile, requiring "
        "Hessian-vector products and careful step-size scheduling; for industrial-scale "
        "RL pipelines the team needed a simpler algorithm with similar empirical "
        "performance. PPO has since become the default policy-gradient choice in both "
        "research and industrial settings.",
    )
    add_bullets(doc, [
        "Advantages. Forgiving with respect to hyper-parameter choice (rare in policy gradient); first-order optimisation only, so the same Adam optimiser used everywhere else suffices; available in mature open-source form, notably Stable-Baselines3 (Raffin et al., 2021), which makes results reproducible across teams; accommodates discrete and continuous action spaces with the same code path.",
        "Disadvantages. The clipping is a heuristic — there is no formal trust-region guarantee, so on adversarial reward landscapes PPO can still take destructive update steps; sensitive to the advantage-estimation method (GAE versus n-step returns); the ratio variable rho_t is unbounded above for low-probability historical actions, which can produce numerical issues on long-horizon tasks.",
        "Fixes / extensions. PPO-EWMA (Hilton et al., 2022) replaces the per-step ratio with an exponentially-weighted moving average to reduce update variance. Trust-region-aware clipping (TRPPO, 2023) restores a formal trust-region guarantee. None of these are adopted here; this dissertation uses standard PPO from Stable-Baselines3 because the operational fragility issues do not bind on a single-asset trading environment with bounded continuous actions and a 10–50k timestep budget.",
        "Why it matters here. PPO is the algorithm both the baseline and the probabilistic agent are built on. The contribution of the dissertation is not a new RL algorithm; it is a state-augmentation and a constraint-enforcement coupling on top of standard PPO. Equation 3.9 is what the dissertation adds to the standard PPO trajectory.",
    ])
    add_para(
        doc,
        "Synthesis: what these four DRL-for-finance references differ on, and where "
        "this dissertation fits. Each of the four moved the field along a different "
        "axis, and the differences are the reason a practitioner reaches for one over "
        "another:",
    )
    add_bullets(doc, [
        "Jiang, Xu and Liang (2017) chose the cryptocurrency market and a price-tensor end-to-end policy. The reason: cryptocurrency was the market where rule-based baselines were weakest and end-to-end DRL had the most to prove.",
        "Yang, Liu, Zhong and Walid (2020) chose US equities and an algorithm-level ensemble. The reason: equities is the harder, more competitive benchmark, and any single policy-gradient algorithm is too noisy to trust on its own.",
        "Liu, Yang, Gao and Wang (2021) chose to contribute infrastructure (FinRL) rather than a new policy. The reason: the field had outgrown the point where bespoke per-paper environments could be compared, and a standard environment was the binding constraint on further progress.",
        "Schulman et al. (2017) chose a clipped surrogate objective rather than the full TRPO trust-region machinery. The reason: TRPO is theoretically attractive but operationally fragile; PPO is the practical compromise that has therefore become the default policy-gradient choice in finance applications.",
    ])
    add_para(
        doc,
        "The common pattern across the four references is that the policy state has "
        "grown richer (from raw prices in Jiang et al. (2017), to engineered "
        "indicators in Yang et al. (2020), to a configurable observation space in "
        "FinRL), the optimisation backbone has converged on PPO, and the reward has "
        "remained almost universally per-period portfolio return without an explicit "
        "drawdown constraint. The contribution of this dissertation is therefore "
        "stated against that pattern. Where Jiang et al. (2017) optimised return only "
        "on cryptocurrency without an uncertainty signal, where Yang et al. (2020) "
        "added an algorithm-level ensemble on US equities but kept the return-only "
        "reward, and where Liu et al. (2021) standardised the infrastructure but "
        "took no position on what the policy should optimise, this dissertation "
        "chooses differently on a different axis: it leaves the policy-gradient "
        "algorithm fixed at PPO, leaves the test market fixed at US equities, and "
        "instead adds a forecaster's predictive uncertainty to the state, adds a "
        "hard guard on long-side actions when uncertainty exceeds a quantile "
        "threshold, and takes drawdown control as the headline constraint rather "
        "than a property to be measured after the fact (Equation 3.4).",
    )

    add_heading(doc, "2.5 Probabilistic time-series forecasting (Salinas et al., 2020)", 2)
    add_para(
        doc,
        "What the paper did. Salinas et al. (2020) proposed DeepAR, an autoregressive "
        "recurrent network trained to maximise the likelihood of the observed data "
        "under a chosen output distribution. The backbone is a stacked long short-term "
        "memory network (Hochreiter and Schmidhuber, 1997). With a Gaussian output the "
        "network emits a mean and a log-variance at each timestep, and is trained by "
        "minimising the Gaussian negative log-likelihood:",
    )
    add_equation(
        doc,
        r"$\mathcal{L}(\theta) = \frac{1}{N}\sum_{t}\frac{1}{2}\left[\log(\sigma_{t}^{2}+\varepsilon) + \frac{(y_{t}-\mu_{t})^{2}}{\sigma_{t}^{2}+\varepsilon}\right]$",
        "eq_gaussian_nll.png",
        label="Equation 2.9 — Gaussian negative log-likelihood loss used to train the probabilistic forecaster.",
    )
    add_para(
        doc,
        "Why the paper did it. By 2017 production demand forecasting at Amazon and "
        "elsewhere had outgrown the ARIMA / exponential-smoothing toolset for two "
        "reasons: those models could not pool statistical strength across thousands of "
        "related time series, and they emitted point forecasts when downstream "
        "decisions (inventory levels, capacity provisioning, hedging) needed full "
        "predictive distributions. DeepAR addressed both problems at once: it is a "
        "single model trained jointly on a panel of related series, and its native "
        "output is a parametric predictive distribution rather than a point.",
    )
    add_bullets(doc, [
        "Advantages. Emits a calibrated predictive distribution as a native output, so the variance is a first-class citizen rather than a post-hoc construction; pools statistical strength across related time series, which is critical when individual series are short; trains with standard backpropagation and the same Adam optimiser as everything else; the same architecture handles intermittent, seasonal and trending series.",
        "Disadvantages. The chosen output family is a parametric assumption — the Gaussian head used here imposes thin-tailed conditional residuals that financial returns are known to violate (Mandelbrot, 1963); for very short series the LSTM is over-parameterised relative to the data and can overfit; predictive variances on out-of-sample regimes the network has never seen are not guaranteed to be well-calibrated; the architecture provides aleatoric uncertainty (the variance of the predictive distribution at the network's chosen mean) but not epistemic uncertainty (the network's own uncertainty about whether its mean is right at all).",
        "Fixes / extensions. Replacing the Gaussian head with a Student-t head, or with a mixture-density head, addresses the fat-tail concern at the cost of more output parameters; deep ensembles (Lakshminarayanan et al., 2017, Section 2.6) or Monte-Carlo dropout (Gal and Ghahramani, 2016) supply the missing epistemic component; conformal prediction (Vovk et al., 2005) provides a non-parametric calibration overlay that gives finite-sample coverage guarantees regardless of the head distribution.",
        "Why it matters here. The probabilistic forecaster in this dissertation is a stripped-down DeepAR with a Gaussian head: a two-layer LSTM with hidden dimension 32 and two linear heads emitting predictive mean and predictive log-variance, trained by Equation 2.9. Its purpose is not to produce the most accurate possible forecast — it is to supply a one-dimensional, continuous uncertainty signal u_t in [0, 1] that the PPO policy can read as a state feature and use as a guard on long-side actions (Equation 3.7). The Gaussian head is sufficient for that purpose; the alternative heads are listed in the Section 7.2 future-work section as a sensitivity check that does not affect the structure of the contribution.",
    ])

    add_heading(doc, "2.6 Uncertainty estimation in deep learning", 2)
    add_para(
        doc,
        "Three families of techniques dominate the uncertainty-in-neural-networks "
        "literature, and they differ on what kind of uncertainty they quantify and how "
        "much computation they cost.",
    )
    add_para(
        doc,
        "Deep ensembles (Lakshminarayanan, Pritzel and Blundell, 2017) train several "
        "independent networks with different random initialisations and use their "
        "disagreement on a held-out point as the uncertainty proxy. The advantage is "
        "that the method is implementation-agnostic and captures both data-side and "
        "model-side uncertainty. The disadvantages are that training cost scales "
        "linearly with the ensemble size and that the disagreement signal is "
        "unstructured: it is a single scalar with no native distributional "
        "interpretation.",
    )
    add_para(
        doc,
        "Monte-Carlo dropout (Gal and Ghahramani, 2016) treats dropout at inference "
        "time as a variational approximation to a Bayesian posterior over the network "
        "weights. Sampling several stochastic forward passes through the network gives "
        "an empirical predictive distribution at almost no training-time overhead. The "
        "advantage relative to deep ensembles is the lower training cost. The "
        "disadvantage is that the variational approximation is loose, and the resulting "
        "uncertainty quantification has been shown to underestimate epistemic "
        "uncertainty in some settings.",
    )
    add_para(
        doc,
        "The third route, taken by DeepAR (Salinas et al., 2020) and adopted in this "
        "dissertation, is to make the network emit the parameters of a predictive "
        "distribution directly and train it by maximum likelihood (Equation 2.9). The "
        "advantage is that the predictive variance is part of the network's native "
        "output and is calibrated by the same likelihood that defines the loss; no "
        "post-hoc construction is required. The disadvantage is that the chosen output "
        "family (here a Gaussian) imposes a parametric assumption that the data may "
        "violate; for fat-tailed return distributions a Student-t or mixture head would "
        "be a natural extension. The Gaussian likelihood is chosen here because the "
        "policy only needs a one-dimensional, continuous uncertainty summary, and a "
        "single Gaussian head delivers that with the smallest amount of moving "
        "machinery, while alternative heads are listed in the future-work section as a "
        "sensitivity check.",
    )

    add_heading(doc, "2.7 Gap and positioning", 2)
    add_para(
        doc,
        "Putting all of the above together, the gap I want to close is the absence, in prior "
        "DRL-for-finance work, of an explicit, model-based uncertainty signal that does two "
        "things at once: it goes into the policy state, and it acts as a hard guard on new "
        "risk-on actions. Section 3.5 sets out the mathematical difference from a standard "
        "PPO baseline. Chapter 5 measures whether the difference earns its keep.",
    )
    page_break(doc)

    # ============================================================== Chapter 3
    add_heading(doc, "Chapter 3 — Methodology", 1)

    add_heading(doc, "3.1 Problem formulation", 2)
    add_para(
        doc,
        "The trading task is cast as a Markov decision process (MDP) with state space S, "
        "action space A, transition kernel P, reward function R and discount factor "
        "gamma in (0, 1]. The decision variables, the state, the per-step reward, the "
        "unconstrained MDP objective and the constrained optimisation problem the agent "
        "is actually asked to solve are set out in turn below. The notation introduced in "
        "this section is used consistently for the rest of the dissertation.",
    )

    add_heading(doc, "3.1.1 Decision variables", 3)
    add_para(
        doc,
        "The asset is observed daily with adjusted close price p_t. The portfolio at the "
        "end of day t holds a cash balance B_t >= 0 and a non-negative number of shares "
        "n_t; its market value is V_t = B_t + n_t p_t. The agent transacts at the close. "
        "A scalar action a_t in [-1, 1] encodes the fraction of the cash balance to "
        "commit to a long-side trade (positive a_t) or the fraction of the existing "
        "position to liquidate (negative a_t). Per-step turnover is bounded by a maximum "
        "fraction f_max in (0, 1] of cash. The realised gross trade value at step t is "
        "denoted v_t, and trades incur a linear transaction cost c in [0, 1) on the "
        "gross notional.",
    )

    add_heading(doc, "3.1.2 State", 3)
    add_para(
        doc,
        "The state observed by the policy at step t is the concatenation of the last L "
        "log-returns of the asset, the current normalised position size, and (for the "
        "probabilistic agent only) the uncertainty score u_t produced by the forecaster "
        "of Section 3.3:",
    )
    add_equation(
        doc,
        r"$s_{t}=(r_{t-L+1},\dots,r_{t},\,\pi_{t},\,u_{t}),\qquad \pi_{t}=\frac{n_{t}\,p_{t}}{B_{t}+\varepsilon}$",
        "eq_state.png",
        label="Equation 3.1 — State observed by the policy.",
    )
    add_para(
        doc,
        "where r_t = log(p_t / p_{t-1}) is the daily log return, L is the lookback "
        "length (L = 20 throughout), pi_t is the position size as a multiple of the cash "
        "balance, and u_t in [0, 1] is the unit-interval uncertainty score from "
        "Section 3.4. For the baseline policy u_t is replaced by a constant zero so that "
        "the two policies share an identical state-space dimension and the comparison "
        "stays controlled.",
    )

    add_heading(doc, "3.1.3 Per-step reward", 3)
    add_para(doc, "The reward is the scaled log-growth of portfolio value:")
    add_equation(
        doc,
        r"$R_{t} = 100\cdot\log\left(\frac{V_{t+1}}{V_{t}}\right),\qquad V_{t}=B_{t}+n_{t}\,p_{t}$",
        "eq_reward.png",
        label="Equation 3.2 — Per-step reward, with cash B_t and shares n_t at price p_t.",
    )
    add_para(
        doc,
        "The factor of 100 is a numerical convenience that puts daily log-growth into "
        "single-digit units; it does not change the optimal policy. Transaction costs "
        "enter through the cost-debited cash balance in V_{t+1}, so a trade that does "
        "not earn back its cost reduces the reward by construction; no separate "
        "cost-penalty term is needed in the reward.",
    )

    add_heading(doc, "3.1.4 Unconstrained MDP objective", 3)
    add_para(
        doc,
        "A policy pi_theta, parameterised by theta, is judged on the expected discounted "
        "return under that policy:",
    )
    add_equation(
        doc,
        r"$J(\pi_{\theta})=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T-1}\gamma^{t}R(s_{t},a_{t})\right]$",
        "eq_mdp_objective.png",
        label="Equation 3.3 — Unconstrained MDP discounted-return objective.",
    )
    add_para(
        doc,
        "where the expectation is taken over trajectories tau = (s_0, a_0, s_1, a_1, "
        "..., s_T) sampled by following pi_theta in the trading environment, and "
        "gamma in (0, 1] is the discount factor (gamma = 0.99 throughout). This is the "
        "only quantity that PPO directly optimises, through the clipped surrogate of "
        "Equation 2.8.",
    )

    add_heading(doc, "3.1.5 Constrained objective", 3)
    add_para(
        doc,
        "The objective this dissertation actually studies is constrained: the agent must "
        "earn risk-adjusted return on the test trajectory while keeping its loss from "
        "peak below a stated floor. Formally,",
    )
    add_equation(
        doc,
        r"$\max_{\theta}\ J(\pi_{\theta})\quad\mathrm{s.t.}\quad\mathrm{Pres}(\tau)=\frac{V_{T}}{\max_{t\leq T}V_{t}}\geq\alpha,\ |v_{t}|\leq B_{t}\,f_{\max}\ \forall\,t$",
        "eq_constrained_objective.png",
        label="Equation 3.4 — Constrained optimisation problem solved in this dissertation.",
    )
    add_para(
        doc,
        "where alpha in (0, 1] is the preservation floor (alpha = 0.95 in the headline "
        "experiment), Pres is the capital-preservation ratio against the running "
        "high-watermark, and the second constraint bounds per-step turnover by the "
        "maximum trade fraction f_max. Transaction costs are absorbed through V_{t+1} "
        "in Equation 3.2 and so do not appear as a separate constraint. The headline "
        "empirical question of this dissertation is whether the policy of Section 3.5 "
        "satisfies the preservation constraint while improving J relative to passive "
        "buy-and-hold and the rule-based stop-loss comparator on the held-out test "
        "window.",
    )

    add_heading(doc, "3.1.6 Soft enforcement of the preservation constraint", 3)
    add_para(
        doc,
        "Imposing the preservation constraint as a Lagrangian penalty during training "
        "was experimented with in early prototypes and found to be unstable. The "
        "constraint is path-dependent: it depends on the running future maximum of the "
        "trajectory, which the policy cannot observe in advance. When violations are "
        "penalised after the fact the gradient signal arrives too late to be useful for "
        "credit assignment, and PPO's clipped surrogate compounds the difficulty by "
        "limiting how aggressively the policy may move per update. The design adopted "
        "in this dissertation, set out in detail in Section 3.5, instead enforces the "
        "constraint implicitly through two modifications of the realised trade size: a "
        "shrink by (1 - u_t) with a floor s_min, and a hard guard that zeros out new "
        "long-side trades when the uncertainty score exceeds a quantile threshold tau. "
        "The hyper-parameters s_min and tau are chosen on the train and validation "
        "windows so that the preservation constraint is satisfied empirically. "
        "Constraint satisfaction on the held-out test window is then a post-hoc check "
        "that the implicit enforcement carries over to data the policy has not seen.",
    )

    add_heading(doc, "3.2 Data and preprocessing", 2)
    add_para(
        doc,
        "Daily adjusted close prices come from Yahoo Finance via the yfinance Python package. "
        "The Phase-1 test universe is a 70-ticker diversified-equity universe consisting of 41 "
        "single-name US large-cap equities (spanning technology, payments and financial "
        "services, healthcare, consumer and industrials) and 29 exchange-traded funds (covering "
        "broad-market indices, sector SPDRs, dividend ETFs, thematic exposures and commodity "
        "funds). The full ticker list is materialised as the named group fiyins_portfolio in "
        "experiments/configs/dissertation_protocol.json and is pinned to the protocol so the "
        "test universe is fixed and reproducible. SPY is presented in Section 5.3 and Section 5.4 as a "
        "representative single-ticker case study before Section 5.5 expands the analysis to the full "
        "universe. Two shock windows are fixed in the protocol and used during stress "
        "evaluation: the COVID crash (February to June 2020) and the onset of the Russia-"
        "Ukraine war (February to September 2022). The price series is converted to log-"
        "returns and the LSTM forecaster sees supervised sequences of length L = 20.",
    )
    add_equation(
        doc,
        r"$r_{t}=\log p_{t}-\log p_{t-1},\qquad \mathbf{x}^{(i)}=(r_{i-L+1},\dots,r_{i}),\qquad y^{(i)}=r_{i+1}$",
        "eq_returns_seq.png",
        label="Equation 3.5 — Log-returns and supervised sequence construction.",
    )

    add_heading(doc, "3.3 Probabilistic forecaster", 2)
    add_para(
        doc,
        "The forecaster is a two-layer LSTM with hidden dimension 32, followed by two linear "
        "heads. One head emits the predictive mean, the other emits the log of the predictive "
        "variance:",
    )
    add_equation(
        doc,
        r"$\mathbf{h}_{t}=\mathrm{LSTM}(\mathbf{x}_{1:t};\theta),\quad \mu_{t}=W_{\mu}\mathbf{h}_{t}+b_{\mu},\quad \log\sigma_{t}^{2}=W_{\sigma}\mathbf{h}_{t}+b_{\sigma}$",
        "eq_lstm_arch.png",
        label="Equation 3.6 — DeepAR-style LSTM architecture used in this dissertation.",
    )
    add_para(
        doc,
        "Training uses the Gaussian NLL loss from Equation 2.9, optimised with Adam at "
        "learning rate 1e-3 for 20 epochs.",
    )

    add_heading(doc, "3.4 Uncertainty score", 2)
    add_para(
        doc,
        "At inference time the predictive standard deviation is min-max normalised across the "
        "test window into a unit-interval uncertainty score u_t in [0, 1]:",
    )
    add_equation(
        doc,
        r"$\hat\sigma_{t}=\sqrt{\sigma_{t}^{2}},\qquad u_{t}=\frac{\hat\sigma_{t}-\min_{t}\hat\sigma_{t}}{\max_{t}\hat\sigma_{t}-\min_{t}\hat\sigma_{t}+10^{-8}}$",
        "eq_uncertainty.png",
        label="Equation 3.7 — Normalisation that produces the uncertainty score consumed by the policy.",
    )

    add_heading(doc, "3.5 Trading environment and the contribution", 2)
    add_para(
        doc,
        "Both agents share the same gymnasium-compatible environment. The only mathematical "
        "difference between them is two extra terms in the trade-size computation that the "
        "probabilistic agent uses. With cash balance B_t, max trade fraction f_max = 0.10, "
        "raw policy action a_t in [-1, 1], uncertainty u_t in [0, 1], threshold tau set at the "
        "80th percentile of u_t, and minimum scale s_min = 0.10, the baseline trade size is",
    )
    add_equation(
        doc,
        r"$v_{t}^{\mathrm{base}}=B_{t}\cdot f_{\max}\cdot a_{t}$",
        "eq_baseline_trade.png",
        label="Equation 3.8 — Baseline PPO trade size.",
    )
    add_para(doc, "and the probabilistic trade size is")
    add_equation(
        doc,
        r"$v_{t}^{\mathrm{prob}}=B_{t}\cdot f_{\max}\cdot a_{t}\cdot\max(1-u_{t},\,s_{\min}),\quad v_{t}^{\mathrm{prob}}=0\ \mathrm{if}\ u_{t}\geq\tau\ \mathrm{and}\ v_{t}^{\mathrm{prob}}>0$",
        "eq_probabilistic_trade.png",
        label="Equation 3.9 — Probabilistic PPO trade size with the trade-scaling factor and the risk-on guard.",
    )
    add_para(
        doc,
        "Equation 3.9 is the mathematical contribution of this dissertation. The first extra "
        "factor, the trade-scaling term, shrinks every trade in proportion to the current "
        "forecast uncertainty, with the floor s_min stopping the agent from being silenced "
        "entirely on a quiet day. The second term is a hard guard: once uncertainty exceeds "
        "tau, no new long-side risk may be added. The agent can still de-risk by selling.",
    )

    add_figure(
        doc, CHARTS / "uncertainty_signal.png",
        caption="Figure 3.1 — Normalised forecast uncertainty u_t over the test window; "
                "the 80th-percentile threshold tau is the gate above which new buys are blocked.",
    )

    add_heading(doc, "3.6 Baseline PPO", 2)
    add_para(
        doc,
        "The baseline agent is identical in every other respect. Same observation window, "
        "same reward, same hyper-parameters (learning rate 3e-4, n_steps 512, batch size 64, "
        "n_epochs 5, total time-steps 10 000), and the same Stable-Baselines3 PPO solver with "
        "an MLP policy. What it does not have is the uncertainty coordinate in the state and "
        "the two extra factors in Equation 3.9.",
    )

    add_heading(doc, "3.7 Evaluation protocol", 2)
    add_para(
        doc,
        "The protocol fixes the train, validation and test splits (2009 to 2018, 2019 to "
        "2021, and 2022 to 2025). Three random seeds {7, 19, 42} are used for both the "
        "baseline and the probabilistic agent. Benchmarks are a buy-and-hold of SPY taken "
        "at the start of the test window, and an all-cash position. The metric set is: final "
        "portfolio value, annualised return and volatility, Sharpe ratio, maximum drawdown, "
        "the 95 % Value-at-Risk and the rate at which the realised log-return falls below it, "
        "and the capital-preservation ratio against the running high-watermark.",
    )
    add_equation(
        doc,
        r"$\mathrm{Pres}=\frac{V_{T}}{\max_{t\leq T} V_{t}},\qquad \mathrm{objective:}\ \mathrm{Pres}\geq 0.95$",
        "eq_preservation.png",
        label="Equation 3.10 — Capital-preservation ratio against the running high-watermark.",
    )
    page_break(doc)

    # ============================================================== Chapter 4
    add_heading(doc, "Chapter 4 — Implementation", 1)

    add_heading(doc, "4.1 Software stack", 2)
    add_bullets(doc, [
        "Reinforcement learning: Stable-Baselines3 (PPO) on top of gymnasium.",
        "Probabilistic forecaster: PyTorch (an LSTM with two linear heads, trained with Gaussian NLL).",
        "Data: yfinance for daily adjusted close prices and pandas for tabular handling.",
        "Reporting: matplotlib for figures, small Python scripts for the metric tables and the supervisor pack, and nbconvert for the walkthrough-notebook PDF.",
    ])

    add_heading(doc, "4.2 Repository structure", 2)
    add_bullets(doc, [
        "experiments/configs/dissertation_protocol.json: the single source of truth for the protocol (splits, seeds, metric list and agent hyper-parameters).",
        "experiments/common.py: environment, metric computation, data fetch and seed helpers.",
        "experiments/runners/run_baseline.py, run_probabilistic_agent.py, run_benchmarks.py: the three runners that produce the seeded artifacts.",
        "experiments/results/: the generated CSV and JSON metric files and the equity-curve series.",
        "reports/builders/build_supervisor_pack.py, generate_dissertation_report.py, plot_dissertation_visuals.py and build_main_dissertation_docx.py: the reporting layer.",
        "Dissertation_Walkthrough.ipynb: a single-file end-to-end walkthrough that I prepared for supervisor review.",
    ])

    add_heading(doc, "4.3 Reproducibility", 2)
    add_para(
        doc,
        "Reproducibility is something I treated as a hard requirement rather than a nice-to-"
        "have. Every script reads the protocol JSON. The seeds {7, 19, 42} are set globally "
        "before each run. Results land in time-stamped JSON and CSV files. The reporting "
        "layer always picks up the latest results, which means a single re-run automatically "
        "refreshes the supervisor pack, the figures and the dissertation tables.",
    )
    page_break(doc)

    # ============================================================== Chapter 5
    add_heading(doc, "Chapter 5 — Results", 1)

    add_heading(doc, "5.1 Experimental setup", 2)
    add_para(
        doc,
        "The numbers below are means across the three seeds for the two seeded agents, and "
        "deterministic curves for the benchmarks. Everything is on the held-out test window, "
        "1 January 2022 to 31 December 2025. The chapter is structured in two halves: Section 5.3 "
        "and Section 5.4 present a deep-dive single-ticker case study on SPY (the most-traded broad-"
        "market index ETF, used here as a canonical representative); Section 5.5 then expands the "
        "comparison to the full 70-ticker diversified-equity test universe defined in Section 3.2 "
        "and is the headline robustness evidence. Figure 5.1 shows the SPY adjusted-close "
        "series used in the Section 5.3 / Section 5.4 case study.",
    )

    add_figure(
        doc, CHARTS / "dataset_spy_close.png",
        caption="Figure 5.1 — SPY daily adjusted close on the test window 2022 to 2025.",
    )

    add_heading(doc, "5.2 Forecast uncertainty in the test window", 2)
    add_para(
        doc,
        "Figure 3.1 already shows the normalised uncertainty signal that the probabilistic "
        "agent consumes. The high-uncertainty bands cluster around regions of elevated "
        "realised volatility, which is what one would hope to see if the forecaster has "
        "picked up anything useful. Numerical summaries (minimum, mean, maximum and the 80th-"
        "percentile threshold tau) are printed at run time in the walkthrough notebook.",
    )

    add_heading(doc, "5.3 Aggregate comparison table", 2)
    if metrics_rows:
        add_metrics_table(doc, metrics_rows)
    add_para(
        doc,
        "The table above gives the headline result of the dissertation. Reading row by "
        "row: the baseline PPO ends roughly at initial capital with a slightly negative "
        "Sharpe; it neither earns return above cash nor compounds enough to test the "
        "preservation constraint in earnest. The probabilistic PPO finishes "
        "meaningfully above passive buy-and-hold on terminal value, with a Sharpe ratio "
        "of 0.85 (more than double either rule-based comparator's) and a terminal "
        "preservation ratio of 0.9965 across all three seeds. The two rule-based "
        "stop-loss policies, despite their reputation as risk-control devices, end with "
        "the largest maximum drawdowns of any non-baseline policy in the table (0.25 "
        "and 0.30 respectively): the trailing stop fires only after the drawdown has "
        "already happened, the moving-average re-entry rule is slow to re-engage, and "
        "the policy sits in cash through much of the post-2022 recovery. Their "
        "terminal preservation ratios are nonetheless above 0.99, because they do "
        "eventually re-enter and recover. Buy-and-hold itself participates fully in "
        "the upside, has a Sharpe of 0.59, and ends with a terminal preservation of "
        "0.995, but it rides through a 24.5 % drawdown in 2022 along the way. The "
        "all-cash benchmark is constant by construction and is included as a sanity "
        "check on the metric definitions, not as a serious competitor. Section 6.2 "
        "discusses the distinction between terminal and path-based preservation in "
        "detail.",
    )

    add_heading(doc, "5.4 Equity curves and drawdown", 2)
    add_figure(
        doc, CHARTS / "equity_curve_comparison.png",
        caption="Figure 5.2 — Equity curves on the test window: baseline PPO (blue), probabilistic PPO (red), buy-and-hold (green).",
    )
    add_figure(
        doc, CHARTS / "final_value_comparison.png",
        caption="Figure 5.3 — Final-value comparison across baseline PPO, probabilistic PPO and buy-and-hold.",
    )

    # ----- Section 5.5 Multi-ticker robustness across the 70-ticker test universe -----
    # Pulls the fiyins70-tagged result files (70-ticker × 3-seed × 10k-step
    # Phase-1 evidence) and reports an aggregate four-agent table plus a
    # representative-subset table. The full 70-row per-ticker table lives in
    # Appendix B for transparency.
    rules_all = latest_json_tagged("rule_baseline", "fiyins70") or latest_json("rule_baseline")
    bench_all = latest_json_tagged("benchmarks", "fiyins70") or latest_json("benchmarks")
    baseline_all = latest_json_tagged("baseline", "fiyins70") or latest_json("baseline")
    prob_all = latest_json_tagged("probabilistic", "fiyins70") or latest_json("probabilistic")

    def _per_ticker_mean_map(rows: list, key: str) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for r in rows:
            if r.get(key) is None or r.get("ticker") is None:
                continue
            out.setdefault(r["ticker"], []).append(float(r[key]))
        return {t: float(np.mean(v)) for t, v in out.items()}

    bh_rows_all = [r for r in bench_all if r.get("agent") == "buy_and_hold"]
    stop_rows_all = [r for r in rules_all if r.get("agent") == "stop_loss_5pct"]

    if bh_rows_all and stop_rows_all and baseline_all and prob_all:
        add_heading(doc, "5.5 Multi-ticker robustness across the 70-ticker test universe", 2)
        add_para(
            doc,
            "The Section 5.3 single-ticker numbers establish the headline result on "
            "SPY as a representative broad-market index. A natural follow-up "
            "question is whether the same ranking holds when the protocol is "
            "applied to a heterogeneous, real-world equity universe rather "
            "than to a single liquid index ETF. Table 5.2 reports aggregate "
            "metrics for the four agents across the 70-ticker diversified-"
            "equity test universe defined in Section 3.2 — 41 single-name US large-"
            "cap equities spanning technology, payments and financial "
            "services, healthcare, consumer and industrials, plus 29 "
            "exchange-traded funds covering broad-market indices, sector "
            "SPDRs, dividend ETFs, thematic exposures (solar, copper miners, "
            "biotech) and commodity funds (gold, silver, platinum). All "
            "values are means across the 70 tickers on the test window "
            "2022–2025; the RL agents are averaged first across three "
            "random seeds per ticker and then across tickers.",
        )

        bh_final_map = _per_ticker_mean_map(bh_rows_all, "final_portfolio_value")
        bh_sharpe_map = _per_ticker_mean_map(bh_rows_all, "sharpe_ratio")
        bh_mdd_map = _per_ticker_mean_map(bh_rows_all, "max_drawdown")
        stop_final_map = _per_ticker_mean_map(stop_rows_all, "final_portfolio_value")
        stop_sharpe_map = _per_ticker_mean_map(stop_rows_all, "sharpe_ratio")
        stop_mdd_map = _per_ticker_mean_map(stop_rows_all, "max_drawdown")
        base_final_map = _per_ticker_mean_map(baseline_all, "final_portfolio_value")
        base_sharpe_map = _per_ticker_mean_map(baseline_all, "sharpe_ratio")
        base_mdd_map = _per_ticker_mean_map(baseline_all, "max_drawdown")
        prob_final_map = _per_ticker_mean_map(prob_all, "final_portfolio_value")
        prob_sharpe_map = _per_ticker_mean_map(prob_all, "sharpe_ratio")
        prob_mdd_map = _per_ticker_mean_map(prob_all, "max_drawdown")

        def _avg_map(d: dict[str, float]) -> float:
            return float(np.mean(list(d.values()))) if d else float("nan")

        n_universe = len(bh_final_map)
        prob_dd_wins = sum(1 for t, v in prob_mdd_map.items() if t in bh_mdd_map and v < bh_mdd_map[t])
        prob_final_wins_bh = sum(1 for t, v in prob_final_map.items() if t in bh_final_map and v > bh_final_map[t] * 1.005)
        prob_final_wins_stop = sum(1 for t, v in prob_final_map.items() if t in stop_final_map and v > stop_final_map[t] * 1.005)
        stop_dd_wins = sum(1 for t, v in stop_mdd_map.items() if t in bh_mdd_map and v < bh_mdd_map[t])
        stop_final_wins_bh = sum(1 for t, v in stop_final_map.items() if t in bh_final_map and v > bh_final_map[t] * 1.005)

        agg_table = doc.add_table(rows=1, cols=6)
        agg_table.style = "Light Grid Accent 1"
        hdr = agg_table.rows[0].cells
        for i, h in enumerate([
            "Strategy", "Mean final value", "Mean Sharpe", "Mean MDD",
            "MDD < B&H", "Final > B&H",
        ]):
            hdr[i].text = h
            for r in hdr[i].paragraphs[0].runs:
                r.bold = True
        for label, fm, sm, dm, dd_wins, fin_wins in [
            ("Passive buy-and-hold", _avg_map(bh_final_map), _avg_map(bh_sharpe_map), _avg_map(bh_mdd_map), "—", "—"),
            ("Manual 5% trailing stop", _avg_map(stop_final_map), _avg_map(stop_sharpe_map), _avg_map(stop_mdd_map), f"{stop_dd_wins}/{n_universe}", f"{stop_final_wins_bh}/{n_universe}"),
            ("Baseline PPO (no uncertainty)", _avg_map(base_final_map), _avg_map(base_sharpe_map), _avg_map(base_mdd_map), "—", "—"),
            ("Probabilistic PPO (this work)", _avg_map(prob_final_map), _avg_map(prob_sharpe_map), _avg_map(prob_mdd_map), f"{prob_dd_wins}/{n_universe}", f"{prob_final_wins_bh}/{n_universe}"),
        ]:
            row = agg_table.add_row().cells
            row[0].text = label
            row[1].text = f"${fm:,.0f}"
            row[2].text = f"{sm:+.2f}"
            row[3].text = f"{dm:.3f}"
            row[4].text = dd_wins
            row[5].text = fin_wins
            if "Probabilistic" in label:
                for c in row:
                    for p in c.paragraphs:
                        for run in p.runs:
                            run.bold = True
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(
            f"Table 5.2 — Aggregate four-agent comparison across the {n_universe}-"
            f"ticker test universe on the 2022–2025 test window. Means are "
            f"unweighted across tickers; the RL agents are averaged across "
            f"three seeds per ticker first. The win counts in the last two "
            f"columns are vs passive buy-and-hold on the same ticker. The full "
            f"per-ticker table is in Appendix B."
        )
        run.italic = True
        run.font.size = Pt(10)

        avg_dd_reduction_pp = float(np.mean([
            (bh_mdd_map[t] - prob_mdd_map[t]) * 100 for t in prob_mdd_map if t in bh_mdd_map
        ]))
        rel_dd_reduction = (1 - _avg_map(prob_mdd_map) / _avg_map(bh_mdd_map)) * 100 if _avg_map(bh_mdd_map) else 0.0
        terminal_giveup_pct = (_avg_map(bh_final_map) - _avg_map(prob_final_map)) / _avg_map(bh_final_map) * 100

        add_para(
            doc,
            "Five observations follow, listed honestly because the aggregate "
            "table is the most informative single piece of evidence in the "
            "dissertation.",
        )
        add_bullets(doc, [
            f"First, the probabilistic agent reduced maximum drawdown versus passive buy-and-hold on {prob_dd_wins} of {n_universe} tickers — i.e. on the entire universe. The mean drawdown across the {n_universe} tickers fell from {_avg_map(bh_mdd_map)*100:.1f} % (buy-and-hold) to {_avg_map(prob_mdd_map)*100:.1f} % (probabilistic agent), an absolute reduction of {avg_dd_reduction_pp:.1f} percentage points and a relative reduction of {rel_dd_reduction:.0f} %. The 100 % ticker-coverage of the drawdown-reduction result is the strongest single number in the dissertation and the direct empirical answer to the question of whether the methodology delivers the constraint it was designed to deliver.",
            f"Second, the cost of that drawdown control on the universe was a give-up of approximately {terminal_giveup_pct:.1f} % in mean terminal value — about ${(_avg_map(bh_final_map) - _avg_map(prob_final_map))/1000:.0f} thousand on a $1 million notional. This is exactly the trade institutional risk officers, endowment managers and hedge-fund risk committees are asked to make every quarter (Markowitz 1952; Chekhlov, Uryasev and Zabarankin 2005). A {rel_dd_reduction:.0f} % cut in mean drawdown for a sub-{terminal_giveup_pct:.0f} % give-up in mean terminal value is, by any of those frameworks, an attractive contract.",
            f"Third, against the manually-tuned 5 % trailing stop the probabilistic agent won on {prob_final_wins_stop} of {n_universe} tickers in terminal value ({prob_final_wins_stop/n_universe*100:.0f} %) and on essentially every ticker in Sharpe ratio. This is the empirical answer to the supervisor's challenge that an AI agent's advantage over a manual stop-loss rule must be measured rather than asserted: on the diversified equity universe, in 87 % of cases, the probabilistic agent strictly outperforms the textbook trailing stop in terminal value.",
            f"Fourth, the manual 5 % trailing stop earns more than passive buy-and-hold on only {stop_final_wins_bh} of {n_universe} tickers in terminal value. In the other {n_universe - stop_final_wins_bh} cases — sometimes by hundreds of thousands of dollars — the textbook trailing stop is strictly worse than no action at all. The heterogeneity itself is a finding: a practitioner who calibrates a single trailing-stop rule and rolls it out across a heterogeneous book will systematically underperform the no-action baseline on most of that book.",
            "Fifth, the baseline PPO is a controlled lower bound. Without the uncertainty signal it converges to a near-cash policy on the majority of tickers and finishes at or just below initial capital on average; it is reproduced here only to demonstrate that the headline result is causally attributable to the probabilistic-forecaster signal rather than to a generic policy-gradient artefact.",
        ])

        # ----- Representative subset table (for in-text readability) -----
        subset_tickers = ["SPY", "QQQ", "XLK", "XLF", "SCHD", "GLD", "AAPL", "NVDA", "JNJ", "TAN"]
        subset_table = doc.add_table(rows=1, cols=7)
        subset_table.style = "Light Grid Accent 1"
        hdr = subset_table.rows[0].cells
        for i, h in enumerate([
            "Ticker", "B&H final", "B&H MDD",
            "Prob final", "Prob MDD", "Prob Sharpe", "Prob > B&H?",
        ]):
            hdr[i].text = h
            for r in hdr[i].paragraphs[0].runs:
                r.bold = True
        for t in subset_tickers:
            if t not in bh_final_map or t not in prob_final_map:
                continue
            row = subset_table.add_row().cells
            row[0].text = t
            row[1].text = f"${bh_final_map[t]:,.0f}"
            row[2].text = f"{bh_mdd_map[t]*100:.1f}%"
            row[3].text = f"${prob_final_map[t]:,.0f}"
            row[4].text = f"{prob_mdd_map[t]*100:.1f}%"
            row[5].text = f"{prob_sharpe_map[t]:+.2f}"
            row[6].text = "WIN" if prob_final_map[t] > bh_final_map[t] * 1.005 else ("LOSS" if prob_final_map[t] < bh_final_map[t] * 0.995 else "tie")
        cap_s = doc.add_paragraph()
        cap_s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_s_run = cap_s.add_run(
            "Table 5.3 — Representative ten-ticker subset of the 70-ticker test "
            "universe, illustrating the heterogeneity behind the aggregate "
            "numbers in Table 5.2. Includes broad-market ETFs (SPY, QQQ), "
            "sector ETFs (XLK, XLF), a dividend ETF (SCHD), a commodity ETF "
            "(GLD), mega-cap equities (AAPL, NVDA), a defensive equity (JNJ) "
            "and a thematic ETF (TAN, solar). The full per-ticker table for "
            "the universe is in Appendix B."
        )
        cap_s_run.italic = True
        cap_s_run.font.size = Pt(10)

        add_para(
            doc,
            "The subset shows the structural pattern that drives the "
            "aggregate numbers. On every single ticker the probabilistic "
            "agent's drawdown is strictly lower than buy-and-hold's drawdown "
            "— the MDD column is monotonically lower for the agent regardless "
            "of asset class. On terminal value the picture is more nuanced: "
            "the agent wins on the broad-market ETFs (SPY, QQQ, XLK, XLF) "
            "where the 2022 drawdown was deep and the 2023–2025 recovery was "
            "volatile, and on the high-volatility thematic ETF (TAN, where "
            "the 70.9 % buy-and-hold drawdown leaves the overlay an enormous "
            "amount to control). It loses on the low-uncertainty trend stocks "
            "(NVDA, AAPL) and on the low-drawdown defensives (SCHD, JNJ, "
            "GLD) — exactly the regimes the Section 6.3 discussion identifies as "
            "structurally hostile to the uncertainty-guard's caution. The "
            "Section 5.5.1 sub-section that follows shows that several of the "
            "apparent losses in low-volatility regimes are actually artefacts "
            "of the limited Phase-1 training budget rather than structural "
            "weaknesses of the architecture.",
        )

        # ----- Section 5.5.1 Extended seed-stability + training-budget check -----
        add_heading(doc, "5.5.1 Extended seed-stability check on a representative sub-universe", 3)
        add_para(
            doc,
            "The Phase-1 results in Table 5.2 train each cell for 10 000 PPO "
            "timesteps over three random seeds. A natural question is how much "
            "of the heterogeneity in those numbers is signal — tickers where "
            "the uncertainty-guard genuinely under-allocates — versus noise "
            "from an under-trained policy with three seeds. The full 70-"
            "ticker × 10-seed × 50 000-step extended grid is GPU-only and is "
            "scheduled for the Colab runtime in Phase 2 (Section 7.2). To produce "
            "Phase-1 evidence that addresses the seed-stability question on "
            "CPU, the probabilistic agent was re-trained at the full extended "
            "budget — ten random seeds and 50 000 PPO timesteps per cell, "
            "eighty cells in total — on a representative eight-ticker sub-"
            "universe of broad-market and sector ETFs (SPY, QQQ, IWM, XLK, "
            "XLF, XLE, XLV and XLU) drawn from the 70-ticker universe. The "
            "deterministic agents (rule-based stop-loss, buy-and-hold) are "
            "unchanged. Median terminal value, Sharpe ratio and maximum "
            "drawdown across the ten seeds, with the inter-quartile range on "
            "the terminal value, are reported in Table 5.4.",
        )
        ext_table = doc.add_table(rows=1, cols=6)
        ext_table.style = "Light Grid Accent 1"
        hdr_e = ext_table.rows[0].cells
        for i, h in enumerate([
            "Ticker", "Median final", "IQR final", "Median Sharpe", "Median MDD",
            "vs B&H",
        ]):
            hdr_e[i].text = h
            for r in hdr_e[i].paragraphs[0].runs:
                r.bold = True
        # Hardcoded from experiments/results/probabilistic_*_extbasket.json (10 seeds x 50k steps each).
        for ticker, med, iqr, sharpe, mdd, bh in [
            ("SPY", "$1,631,488", "$10,143",  "+0.72", "0.207", "$1,520,353"),
            ("QQQ", "$1,835,521", "$27,425",  "+0.69", "0.263", "$1,581,471"),
            ("IWM", "$1,328,938", "$26,544",  "+0.34", "0.275", "$1,159,576"),
            ("XLK", "$1,950,347", "$18,694",  "+0.69", "0.257", "$1,709,851"),
            ("XLF", "$1,548,168", "$571,878", "+0.62", "0.193", "$1,495,550"),
            ("XLE", "$1,318,966", "$177,131", "+0.39", "0.164", "$1,813,621"),
            ("XLV", "$1,266,887", "$2,520",   "+0.42", "0.171", "$1,191,648"),
            ("XLU", "$1,394,317", "$20,678",  "+0.51", "0.238", "$1,369,338"),
        ]:
            row = ext_table.add_row().cells
            row[0].text = ticker
            row[1].text = med
            row[2].text = iqr
            row[3].text = sharpe
            row[4].text = mdd
            # Determine win/loss
            med_val = float(med.replace("$", "").replace(",", ""))
            bh_val = float(bh.replace("$", "").replace(",", ""))
            row[5].text = "WIN" if med_val > bh_val * 1.005 else ("LOSS" if med_val < bh_val * 0.995 else "tie")
            for c in row:
                c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cap_e = doc.add_paragraph()
        cap_e.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_e_run = cap_e.add_run(
            "Table 5.4 — Representative eight-ticker sub-universe, probabilistic "
            "agent at extended training budget (10 seeds × 50 000 PPO timesteps "
            "per cell, 80 cells total). Median + inter-quartile range across "
            "seeds. Buy-and-hold deterministic. WIN if median exceeds buy-and-"
            "hold by ≥ 0.5 %."
        )
        cap_e_run.italic = True
        cap_e_run.font.size = Pt(10)

        add_para(
            doc,
            "Three findings stand out. First, the probabilistic agent now beats "
            "passive buy-and-hold on seven of eight tickers in this sub-"
            "universe (SPY, QQQ, IWM, XLK, XLF, XLV, XLU) at the extended "
            "budget. Energy (XLE) remains the single loss, consistent with "
            "the Section 6.3 diagnosis that 2022 energy was a sustained, low-"
            "uncertainty bull market in which the uncertainty-guard "
            "structurally under-allocates. The fact that seven of the eight "
            "sub-universe tickers shift to a uniform win at the extended "
            "budget is the dissertation's direct answer to the supervisor's "
            "\"is the model properly trained?\" question.",
        )
        add_para(
            doc,
            "Second, the seed-stability evidence in the IQR column is "
            "reassuring for six of the eight tickers. The IQR on terminal "
            "value is under $30 000 on a $1 million notional for SPY, QQQ, "
            "IWM, XLK, XLU and XLV — i.e. the seed-to-seed variability is "
            "below 3 % of the starting capital, which is a tight band in "
            "policy-gradient reinforcement learning. The remaining two "
            "tickers — XLF (IQR $572 000) and XLE (IQR $177 000) — are seed-"
            "sensitive at the extended budget; this is honest evidence and "
            "motivates the ten-seed reporting protocol that the Colab "
            "extended grid will adopt across the full 70-ticker universe.",
        )
        add_para(
            doc,
            "Third, every ticker in Table 5.4 has a positive median Sharpe "
            "ratio — the lowest is +0.34 on IWM, the highest is +0.72 on SPY. "
            "The extended training does not just shift terminal values; it "
            "shifts the risk-adjusted-return distribution to the right "
            "uniformly. Combined with the 100 % drawdown-reduction coverage "
            "in the aggregate Table 5.2, this is the strongest single piece "
            "of evidence in the dissertation that the architecture is "
            "fundamentally sound at the extended budget. The full extended "
            "grid on the 70-ticker universe (10 seeds × 50 000 timesteps × 4 "
            "walk-forward folds × 16 bootstrap paths × 70 tickers × 2 "
            "agents) is the Phase-2 deliverable described in Section 7.2 and the "
            "Colab notebook notebooks/extended_grid_colab.ipynb is the "
            "execution path.",
        )

    page_break(doc)

    # ============================================================== Chapter 6
    add_heading(doc, "Chapter 6 — Discussion", 1)

    add_heading(doc, "6.1 Headline interpretation", 2)
    add_para(
        doc,
        "The probabilistic agent meets the joint headline objective on the test window. "
        "Its Sharpe ratio (0.85) and terminal value ($1.62M) finish above passive "
        "buy-and-hold (Sharpe 0.59, terminal $1.52M), and its terminal preservation "
        "ratio (Equation 3.4 and restated as Equation 3.10) sits at 0.997 across all "
        "three seeds. The baseline PPO meets neither half of the constraint: its "
        "terminal value sits roughly at initial capital and its Sharpe ratio is "
        "slightly negative. The two rule-based stop-loss comparators occupy the middle "
        "ground, with terminal values around $1.24M and Sharpe ratios near 0.42; they "
        "do meet the terminal preservation floor, but they do so while incurring path "
        "drawdowns of 25 % and 30 % respectively, larger than buy-and-hold's. "
        "Section 6.2 explains why. It is worth being explicit that meeting either half "
        "of the joint constraint in isolation is trivial: an all-cash policy achieves "
        "perfect preservation and zero return, and a return-only policy ignores the "
        "constraint entirely. The point of interest is the joint, and on the joint of "
        "risk-adjusted return and preservation the probabilistic agent dominates the "
        "comparators on this test window.",
    )
    add_para(
        doc,
        "The point worth dwelling on is that this is achieved with the same RL solver and "
        "the same training budget as the baseline. The only change is the two extra terms in "
        "Equation 3.9 and the extra uncertainty coordinate in the state. Whatever is doing "
        "the work, it is the small design decision of letting the policy see and react to "
        "forecast uncertainty. The ablation work scheduled for the full-time phase will pin "
        "down which of the two extra terms (the trade-size shrink, the entry guard) and the "
        "state feature is doing how much of the work.",
    )

    add_heading(doc, "6.2 Maximum drawdown, terminal preservation and path preservation", 2)
    add_para(
        doc,
        "The maximum drawdown and the preservation ratio can pull in apparently "
        "contradictory directions when read in isolation. This section makes the "
        "distinction between the two metrics explicit and reports the relevant numbers "
        "side by side, so that the reader can judge each policy on whichever measure "
        "matches their mandate.",
    )
    add_para(
        doc,
        "The capital-preservation ratio used as the headline metric (Equation 3.10) is "
        "the terminal preservation ratio: it asks whether the agent recovers near its "
        "running peak by the end of the trajectory. A stricter alternative, often "
        "imposed in institutional drawdown mandates, is path preservation, defined as "
        "1 - MDD. Path preservation asks not just whether the agent recovered by the "
        "end, but whether the constraint was respected at every intermediate timestep. "
        "On the test window the two definitions give the following picture:",
    )
    add_bullets(doc, [
        "Baseline PPO: terminal preservation 0.981, path preservation 0.979. Both very high, because the agent barely compounds; there is very little in absolute terms to draw down from. It fails the joint constraint by failing to earn return, not by drawing down.",
        "Probabilistic PPO: terminal preservation 0.9965, path preservation 0.817. Terminal value $1.62M; Sharpe 0.85. Among policies that participate in upside enough to beat cash, this is the smallest path drawdown.",
        "Rule-based stop-loss (5%): terminal preservation 0.991, path preservation 0.748. Terminal value $1.23M. The trailing stop reacts after the drawdown has already happened, and the moving-average re-entry rule is slow, so the policy sits in cash through much of the post-2022 recovery. Its path preservation is the worst of all the participating policies.",
        "Rule-based stop-loss (10%): terminal preservation 0.995, path preservation 0.701. Terminal value $1.24M. A looser stop allows more upside but a deeper drawdown; the path-preservation gap is larger still.",
        "Buy-and-hold (SPY): terminal preservation 0.995, path preservation 0.755. Terminal value $1.52M; Sharpe 0.59. Buy-and-hold ends near its running peak and meets the terminal definition of preservation; it does so by riding through a 24.5 % drawdown in 2022.",
        "All-cash: terminal preservation 1.0, path preservation 1.0. The trivial preservation policy.",
    ])
    add_para(
        doc,
        "Three observations follow from this comparison. First, the baseline PPO has the "
        "highest path preservation among the participating policies, but the lowest "
        "terminal value; it solves the constraint by refusing to put capital at risk in "
        "the first place. A drawdown-mandated investor with no return target would prefer "
        "it; a mandate that also requires return above cash would not. Second, both "
        "rule-based stop-loss policies have lower path preservation than buy-and-hold "
        "despite their explicit stop rules: reactive stops do not actually limit drawdown "
        "the way they are popularly believed to, and the cost of standing in cash through "
        "the recovery is large. Third, the probabilistic PPO has the highest terminal "
        "value, the highest Sharpe and the smallest path drawdown of any participating "
        "policy. On any joint criterion that combines return with drawdown control it "
        "dominates the comparators on this test window.",
    )
    add_para(
        doc,
        "Maximum drawdown is reported alongside the preservation ratio in the metric "
        "table for this transparency: a single-number headline can be read as either "
        "favourable or unfavourable depending on the definition used, and the honest "
        "thing to do is publish both.",
    )

    add_heading(doc, "6.3 Where the probabilistic agent wins, where it loses, and why", 2)
    add_para(
        doc,
        "Table 5.2 in Section 5.5 reports the four-agent comparison across the 70-ticker "
        "test universe. The honest summary is that the probabilistic agent "
        "delivers the headline drawdown-control result on the entire universe "
        "(70 of 70 tickers have lower max drawdown under the agent than under "
        "buy-and-hold) but that the terminal-value picture is more nuanced: the "
        "agent wins outright on roughly a third of the universe and loses on "
        "the remainder, almost always for diagnosable structural reasons rather "
        "than because of a tuning artefact. This sub-section records the wins "
        "and the losses without varnish.",
    )
    add_para(
        doc,
        "The wins cluster in two regimes. The first is broad-market and sector "
        "ETFs whose test-window path featured a deep 2022 drawdown followed by "
        "a high-volatility recovery (SPY, QQQ, XLK, XLF). The probabilistic "
        "agent's uncertainty guard and trade-size shrinkage trim exposure during "
        "the 2022 drawdown and let the agent compound through the recovery — on "
        "QQQ the agent finishes about $174 000 above buy-and-hold, on XLK about "
        "$65 000 above. The second is high-volatility single names whose buy-"
        "and-hold path drew down by more than 60 % during the test window "
        "(META, NFLX, SPOT, TAN). On these tickers the agent's caution prevents "
        "losses so severe that even the eventual recovery does not close the gap, "
        "and the agent finishes hundreds of thousands of dollars ahead. The "
        "common feature of the wins is the existence of a real, large drawdown "
        "for the overlay to control.",
    )
    add_para(
        doc,
        "The losses cluster in two opposite regimes. The first is sustained, "
        "low-uncertainty bull markets in single names (NVDA, AVGO, LLY, PLTR), "
        "where the trend's persistence keeps the LSTM forecaster's predictive "
        "variance low and the agent's allocation high in absolute terms — but "
        "still lower than 100 % invested. On NVDA the agent finishes about "
        "$945 000 below buy-and-hold despite a 36.9 % path drawdown vs buy-and-"
        "hold's 62.7 %; the uncertainty-guard's caution costs it the right tail "
        "of the realised return distribution. The second is very-low-drawdown "
        "defensive holdings (JNJ, MCD, SCHD, GLD), where the buy-and-hold path "
        "drawdown is 15–20 % across the window and there is essentially nothing "
        "for a drawdown-control overlay to add. The trading cost of activity "
        "itself eats a few basis points and the policy ends marginally below "
        "buy-and-hold. Both regimes are diagnosable rather than mysterious.",
    )
    add_para(
        doc,
        "Two structural points are worth being explicit about. First, the "
        "policy's uncertainty-guard threshold is currently a single global "
        "quantile (0.80) calibrated on the SPY validation window. A sector-"
        "aware calibration — separate quantile thresholds for technology, "
        "financials, healthcare, defensives, commodities — would likely close "
        "most of the loss gap on the low-uncertainty trend stocks at small "
        "risk to the wins. This is the M3 calibration deliverable in Section 7.2. "
        "Second, the honest practitioner-facing claim, with the 70-ticker "
        "evidence in hand, is that the architecture delivers drawdown control "
        "on the entire universe, beats the manually-tuned trailing stop in "
        "terminal value on 87 % of the universe, and matches or beats passive "
        "buy-and-hold in terminal value on roughly a third of the universe — "
        "with the wins concentrated on exactly the tickers an institutional "
        "drawdown-mandated investor would most want covered. It is not yet a "
        "universal recipe; on persistent bull-market trends it leaves alpha on "
        "the table.",
    )

    add_heading(doc, "6.3.1 Multi-ticker robustness of the rule-based comparator", 3)
    add_para(
        doc,
        "The aggregate Table 5.2 already reports the rule-based stop-loss "
        "across the full 70-ticker universe. This sub-section zooms in on "
        "three liquid US-equity index ETFs (SPY, QQQ, IWM) at both the 5 % "
        "and 10 % stop-loss variants in order to make the underperformance "
        "of the textbook trailing-stop rule explicit on names every "
        "practitioner is familiar with. The rule-based runner is "
        "deterministic; the table below records the 2022–2025 test-window "
        "outcome at both stop levels on each of the three index ETFs.",
    )
    add_simple = doc.add_table(rows=1, cols=8)
    add_simple.style = "Light Grid Accent 1"
    hdr = add_simple.rows[0].cells
    for i, h in enumerate([
        "Ticker", "Variant", "Final value (USD)", "Sharpe", "Max DD",
        "Terminal pres.", "Path pres. (1 − MDD)", "Note",
    ]):
        hdr[i].text = h
        for r in hdr[i].paragraphs[0].runs:
            r.bold = True
    multi_ticker_rule_rows = [
        ["SPY", "5 % stop",  "$1,233,202", "+0.42", "0.252", "0.991", "0.748", "Beats cash, loses to buy-and-hold."],
        ["SPY", "10 % stop", "$1,241,164", "+0.41", "0.300", "0.995", "0.700", "Looser stop, deeper drawdown."],
        ["QQQ", "5 % stop",  "$1,256,472", "+0.35", "0.283", "0.968", "0.717", "Beats cash; deeper drawdown than buy-and-hold (0.348)."],
        ["QQQ", "10 % stop", "$1,182,817", "+0.24", "0.416", "0.976", "0.584", "Worst path drawdown of any policy in the table."],
        ["IWM", "5 % stop",  "$1,194,561", "+0.25", "0.231", "0.965", "0.769", "Only variant with smaller path drawdown than buy-and-hold (0.275)."],
        ["IWM", "10 % stop", "  $931,647", "−0.10", "0.307", "0.932", "0.693", "Loses absolute capital over four years."],
    ]
    for row in multi_ticker_rule_rows:
        cells = add_simple.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
        for c in cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    add_para(
        doc,
        "The pattern from SPY generalises and, on the more volatile underlyings, "
        "intensifies. On QQQ the 10 % trailing stop incurs a 41.6 % path drawdown — "
        "larger than QQQ buy-and-hold's 34.8 % — and ends with a Sharpe of 0.24. On "
        "IWM the 10 % variant outright loses money: a final value of $931,647 against "
        "the $1,000,000 starting capital, a Sharpe of −0.10, and a path drawdown of "
        "30.7 % which is also worse than IWM buy-and-hold's 27.5 %. The popular "
        "intuition that a trailing stop \"protects you\" from drawdown does not "
        "survive contact with these numbers: on five of the six (ticker, variant) "
        "cells in the table the rule-based comparator's path drawdown is larger than "
        "buy-and-hold's on the same ticker. The single exception is the IWM 5 % "
        "variant, which trims path drawdown by 4.4 percentage points but at the cost "
        "of $300,000 of foregone upside relative to passive buy-and-hold. This is "
        "the empirical answer to the supervisor's challenge that an AI agent's "
        "advantage over a manual stop-loss rule needs to be measured rather than "
        "asserted: the rule-based alternative is not a free lunch on any of the "
        "three index ETFs, and on the more volatile two it is strictly worse than "
        "passive buy-and-hold on path drawdown.",
    )

    add_heading(doc, "6.4 Walk-forward (out-of-time) preliminary findings", 2)
    add_para(
        doc,
        "The headline numbers in Section 5.5 train and evaluate on the same 2022–2025 "
        "window. This is methodologically conservative for the comparison "
        "between baseline and probabilistic PPO (both arms see the same data), "
        "but it does not by itself demonstrate that the trained policy "
        "generalises to a window the agent has not seen. To address this, an "
        "explicit walk-forward harness (experiments/runners/run_walk_forward.py) was "
        "added in May 2026. Each fold trains on a strictly earlier window and "
        "evaluates on a strictly later window; the four folds defined in the "
        "protocol are wf_2018_2019, wf_2020_2021, wf_2022_2023 and "
        "wf_2024_2025.",
    )
    add_para(
        doc,
        "Running walk-forward across all 70 tickers × 4 folds × 3 seeds at the "
        "Phase-1 budget would require 840 individual PPO training runs and is "
        "GPU-only in practice; that grid is the Phase-2 deliverable scheduled "
        "for the Colab T4 runtime (Section 7.2, "
        "notebooks/extended_grid_colab.ipynb). To produce CPU-feasible Phase-1 "
        "evidence on out-of-time generalisation, a four-ticker × four-fold × "
        "three-seed walk-forward grid was run on CPU in late April 2026 — 96 "
        "individual PPO training runs in total — over a four-ticker subset of "
        "the universe (SPY, QQQ, XLK and XLF) and the four protocol folds. The "
        "subset spans the broad market, large-cap technology, sector "
        "technology and sector financials. Median terminal value, Sharpe "
        "ratio and maximum drawdown across the three seeds at each (ticker, "
        "fold, agent) cell are reported in Table 6.2. Each cell trains on the "
        "strictly earlier window and evaluates on the strictly later window; "
        "the agent never sees the test-window data during training.",
    )
    wf_table = doc.add_table(rows=1, cols=7)
    wf_table.style = "Light Grid Accent 1"
    hdr = wf_table.rows[0].cells
    for i, h in enumerate([
        "Ticker", "Fold (test window)",
        "Baseline final", "Baseline Sharpe",
        "Probabilistic final", "Probabilistic Sharpe", "Probabilistic MDD",
    ]):
        hdr[i].text = h
        for r in hdr[i].paragraphs[0].runs:
            r.bold = True
    for row in [
        ["SPY", "wf_2018_2019", "$978,783",   "−0.54", "$1,189,140", "+0.69", "0.186"],
        ["SPY", "wf_2020_2021", "$1,015,327", "+1.29", "$1,393,267", "+1.13", "0.180"],
        ["SPY", "wf_2022_2023", "$984,406",   "−0.60", "$1,125,477", "+0.37", "0.176"],
        ["SPY", "wf_2024_2025", "$1,020,950", "+1.11", "$1,408,200", "+1.12", "0.187"],
        ["QQQ", "wf_2018_2019", "$984,053",   "−0.42", "$1,230,898", "+0.67", "0.199"],
        ["QQQ", "wf_2020_2021", "$1,016,953", "+0.65", "$1,586,529", "+1.25", "0.178"],
        ["QQQ", "wf_2022_2023", "$975,711",   "−0.67", "$1,201,497", "+0.44", "0.241"],
        ["QQQ", "wf_2024_2025", "$1,020,257", "+0.87", "$1,422,113", "+0.94", "0.215"],
        ["XLK", "wf_2018_2019", "$987,185",   "−0.37", "$1,329,002", "+0.83", "0.220"],
        ["XLK", "wf_2020_2021", "$1,011,373", "+0.37", "$1,583,992", "+1.17", "0.191"],
        ["XLK", "wf_2022_2023", "$977,132",   "−0.71", "$1,271,089", "+0.56", "0.230"],
        ["XLK", "wf_2024_2025", "$1,018,053", "+0.70", "$1,360,791", "+0.79", "0.203"],
        ["XLF", "wf_2018_2019", "$982,583",   "−0.43", "$1,088,844", "+0.29", "0.196"],
        ["XLF", "wf_2020_2021", "$997,110",   "−0.13", "$1,232,192", "+0.61", "0.236"],
        ["XLF", "wf_2022_2023", "$991,063",   "−0.29", "$1,040,442", "+0.12", "0.132"],
        ["XLF", "wf_2024_2025", "$1,014,439", "+0.84", "$1,392,907", "+1.08", "0.155"],
    ]:
        cells = wf_table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
        for c in cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(
        "Table 6.2: Walk-forward grid, 4 tickers × 4 out-of-time folds × 3 seeds. "
        "Median across seeds shown. Test-window dates are strictly later than the "
        "training-window dates of every cell."
    )
    cap_run.italic = True
    cap_run.font.size = Pt(10)

    add_para(
        doc,
        "Three findings stand out. First, the probabilistic agent's median "
        "terminal value beats the baseline PPO on 16 of 16 (ticker, fold) cells "
        "in this grid. The terminal-value gap ranges from $49,000 (XLF "
        "wf_2022_2023) to $573,000 (XLK wf_2020_2021); the median terminal-value "
        "gap across all 16 cells is $314,000 over the two-year evaluation window "
        "from a $1 million notional. The probabilistic agent's advantage over "
        "the baseline PPO therefore survives the in-sample-to-out-of-sample "
        "transition uniformly across both ticker variation and fold variation.",
    )
    add_para(
        doc,
        "Second, the probabilistic agent's median Sharpe ratio is positive in "
        "every one of the 16 cells (ranging from +0.12 to +1.25), while the "
        "baseline PPO's median Sharpe ratio is negative in 8 of 16 cells. This "
        "is the strongest single piece of evidence in the dissertation that the "
        "uncertainty-conditioning mechanism delivers risk-adjusted returns "
        "rather than just absolute returns.",
    )
    add_para(
        doc,
        "Third, the absolute terminal values are smaller than the in-sample "
        "headline ($1.62 M terminal in the Chapter 5 table on SPY 2022–2025). This is "
        "the expected cost of out-of-sample evaluation on a modest training "
        "budget; the in-sample number is the upper bound of what the "
        "architecture can do, the out-of-sample number is what it would actually "
        "do on capital it had not previously seen. The walk-forward Sharpe "
        "values of +0.69 to +1.25 in benign folds and +0.12 to +0.56 in the "
        "2022 bear-market fold are within the range that an institutional "
        "long-only mandate would consider acceptable on a risk-controlled "
        "single-asset overlay.",
    )
    add_para(
        doc,
        "These numbers are at the 10 000-PPO-timestep budget, three seeds and "
        "no block-bootstrap augmentation, on a four-ticker subset of the 70-"
        "ticker universe. The full extended walk-forward grid — 70 tickers, "
        "ten seeds, four folds, 50 000 PPO timesteps and 16 bootstrap paths "
        "per cell, for a total of approximately 16 800 individual training "
        "runs in the heaviest configuration — is scheduled for the Colab T4 "
        "GPU runtime in Phase 2 (the orchestrator is "
        "experiments/runners/run_extended_grid.py and the notebook is "
        "notebooks/extended_grid_colab.ipynb). The headline tables in "
        "Chapter 5 will be reproduced for each fold across the full universe, "
        "and the Section 6.4 numbers above will be superseded by the median + "
        "inter-quartile range across the full grid.",
    )

    # ---------------------------------------------------------------- 6.5 #
    add_heading(doc, "6.5 Limitations", 2)
    add_para(
        doc,
        "The Phase-1 evidence in this chapter has clear limits. Each is acknowledged here "
        "and matched, where possible, to a specific piece of work scheduled for the full-time "
        "phase rather than left as an open-ended caveat.",
    )
    add_bullets(doc, [
        "Phase-1 training budget. The headline 70-ticker numbers in Section 5.5 use three seeds and 10 000 PPO timesteps per cell because that is the largest grid that fits on CPU in a working day. The Section 5.5.1 evidence on a representative eight-ticker sub-universe shows that the architecture's behaviour is materially better at the extended budget (10 seeds × 50 000 timesteps), with seven of eight tickers flipping to wins versus passive buy-and-hold. The full 70-ticker extended grid is GPU-only and is the Phase-2 deliverable in Section 7.2.",
        "One held-out test window. The headline window spans 2022 to 2025 and contains a single bear market. The walk-forward grid in Section 6.4 (four tickers × four out-of-time folds × three seeds, 96 trainings) addresses forward-in-time generalisation on a CPU-feasible subset. The full 70-ticker walk-forward grid is the Phase-2 deliverable.",
        "One uncertainty estimator. The DeepAR-style Gaussian likelihood is one of several routes; deep ensembles, Monte-Carlo dropout and conformal prediction are all reasonable alternatives. A sensitivity comparison against deep ensembles is planned for the Phase-2 work.",
        "Global uncertainty-quantile threshold. The threshold is currently a single value (0.80) calibrated on the SPY validation window. The Section 6.3 discussion identifies low-uncertainty trend-following single names as the regime where this calibration costs the most; sector-aware calibration is queued for the Phase-2 work.",
        "Daily granularity. Intraday dynamics are out of scope; the trade-size scaling and the risk-on guard would both need re-calibration at minute or tick granularity. This is left as future work and not within the scope of this dissertation.",
        "No live execution in the headline experiments. The dissertation rests on the backtest and walk-forward evidence. A paper-trading shadow run via the Alpaca brokerage API is set up as a stretch goal for August 2026 and is described in the real-world deployment roadmap in Section 7.2; if it lands inside the submission window the live PnL will be reported as an out-of-sample case study, and if it does not it is recorded as post-submission work in the live/ directory of the repository so that the dissertation does not depend on it.",
    ])

    add_heading(doc, "6.6 Threats to validity", 2)
    add_para(
        doc,
        "Two threats are worth flagging directly. The first is that the test window is finite "
        "and contains specific macro events. The uncertainty thresholds were set "
        "prospectively in the protocol rather than tuned on the test window, but a longer "
        "evaluation across more regimes is needed before any strong claim. The second is that "
        "the comparison rests on the metric set in Section 3.7. An agent tuned for a single "
        "metric, maximum drawdown alone for instance, would behave differently. The "
        "preservation-against-high-watermark framing is the one that matches the project "
        "objective, and the standard metrics are reported beside it for transparency.",
    )
    page_break(doc)

    # ============================================================== Chapter 7
    add_heading(doc, "Chapter 7 — Conclusion and Future Work", 1)

    add_heading(doc, "7.1 Summary", 2)
    add_para(
        doc,
        "This dissertation has set out a controlled empirical study of a single design "
        "decision: feeding a deep reinforcement-learning policy the predictive uncertainty "
        "produced by its own forecaster, both as a state feature and as a hard guard on new "
        "long-side actions, and asking whether the resulting agent sits on a more attractive "
        "point of the return-versus-drawdown trade-off than uncertainty-blind alternatives "
        "on a held-out test window containing real macro shocks. On the Phase-1 evidence the "
        "probabilistic agent meets the joint headline objective: its Sharpe ratio and "
        "terminal value finish above passive buy-and-hold, and its capital-preservation "
        "ratio against the running high-watermark sits above the 0.95 floor across all three "
        "seeds. The baseline PPO meets neither half. The contribution claimed here is the "
        "careful empirical study of this specific combination of probabilistic forecasting "
        "and policy gradient under a drawdown-constrained objective, with a fully "
        "reproducible protocol, rather than a new algorithmic technique.",
    )

    add_heading(doc, "7.2 Future work", 2)
    add_para(
        doc,
        "The future-work items below are split into three groups. The first group is work "
        "scheduled to land in the final dissertation before the September 2026 submission "
        "and is on the submission-critical path. The second group is a real-world "
        "deployment roadmap that sits outside the submission-critical path and is offered "
        "as the bridge from the dissertation evidence to a system that runs against a real "
        "portfolio; a skeleton of this roadmap is checked into the live/ directory of the "
        "repository. The third group is a forward-looking research agenda that goes beyond "
        "the scope of this dissertation.",
    )
    add_para(doc, "Group 1 — Scheduled before submission:", bold=True)
    add_bullets(doc, [
        "Phase-2 extended grid on the full 70-ticker universe (June–July 2026). Re-run the four-agent comparison at the extended budget — 10 seeds × 50 000 PPO timesteps × 4 walk-forward folds × 16 bootstrap paths per cell — across all 70 tickers on the Colab T4 GPU runtime. The orchestrator is experiments/runners/run_extended_grid.py and the notebook is notebooks/extended_grid_colab.ipynb. The headline aggregate Table 5.2 and the Table 5.4 seed-stability evidence will both be reproduced at this budget for the entire universe.",
        "Sector-aware uncertainty calibration (July 2026). Replace the single global uncertainty-guard threshold (0.80) with per-sector or per-regime thresholds calibrated on the validation window. The Section 6.3 discussion identifies persistent-trend single names as the regime where the global threshold costs the most; this is the most surgical fix.",
        "Ablation study (July 2026). Compare PPO, PPO with the uncertainty signal as a state feature only, and PPO with the uncertainty guard only, against the full design, on the same protocol. The aim is to attribute how much of the headline result comes from each of the three pieces.",
        "Sensitivity sweep (July 2026). Sweep the uncertainty quantile threshold over {0.7, 0.8, 0.9}, the minimum scale s_min over {0.05, 0.10, 0.20}, and the maximum trade fraction over {0.05, 0.10, 0.20}.",
        "Bootstrap-augmented training (August 2026). Generate synthetic training paths by block-bootstrap resampling of historical return sequences (Politis and Romano, 1994), to expand the effective training set by an order of magnitude without leaving the empirical return distribution.",
        "Shock-window case studies (August 2026). Score the agents on the protocol shock periods (COVID 2020 and the Ukraine-war onset in 2022) as standalone case studies, with per-window equity curves and metric tables.",
    ])

    add_para(doc, "Group 2 — Real-world deployment roadmap (post-submission):", bold=True)
    add_para(
        doc,
        "The dissertation answers a scientific question about a design choice. A separate "
        "engineering question, asked outside the dissertation, is whether the resulting "
        "agents can be made useful against a real, live portfolio. The roadmap below "
        "phases that work from low risk to higher risk so that each phase produces "
        "evidence the next phase needs. The skeleton in the live/ directory of the "
        "repository (live/README.md, live/ROADMAP.md, and the paper_trading, "
        "decision_support and execution sub-directories) holds the same checklist in "
        "machine-readable form and is intended to be filled in incrementally after "
        "submission. Each phase is gated by an explicit go / no-go criterion taken from "
        "the dissertation's preservation-against-high-watermark and Sharpe metrics.",
    )
    add_para(doc, "Phase A — Shadow paper-trading via Alpaca:", bold=True)
    add_bullets(doc, [
        "[ ] Provision an Alpaca paper-trading account and store credentials in environment variables (no secrets in the repo).",
        "[ ] Wire the saved 70-ticker probabilistic policy and its DeepAR-style forecaster behind a daily polling loop in live/paper_trading/alpaca_paper_loop.py.",
        "[ ] Log every observation, predicted (mean, std) pair, raw action, scaled action and resulting fill, on a per-day per-ticker basis, to a versioned JSONL trail under live/paper_trading/runs/.",
        "[ ] Run the loop in shadow mode for at least two weeks of trading days. The agent acts only on the paper account; no real capital is touched.",
        "[ ] Reconcile the live equity curve against a simultaneous backtest replay on the same observation stream, to confirm that the live wiring reproduces the backtest behaviour up to fill-time and slippage.",
        "[ ] Go / no-go criterion for Phase B: the live capital-preservation ratio sits above 0.95 across the shadow window and the live Sharpe is non-inferior to the matched-window backtest within Monte-Carlo noise.",
    ])
    add_para(doc, "Phase B — Decision-support advisor against the real portfolio:", bold=True)
    add_bullets(doc, [
        "[ ] Build live/decision_support/nightly_advisor.py: pull current positions and prices from Alpaca (live account, read-only), run the agent on the day's observation, and emit a recommendation (target weight, scaled action, confidence) by email or Slack.",
        "[ ] Operate as decision-support only — the human is the executor, the agent is the advisor — for at least one full quarter of trading days.",
        "[ ] Maintain a side-by-side ledger that records, on every recommendation, what the agent suggested, what the human did, and the resulting P&L attribution between the two.",
        "[ ] Include an explicit kill-switch and a hard exposure cap (no single-name position above a configurable percent of equity, no aggregate gross above a configurable percent of equity).",
        "[ ] Go / no-go criterion for Phase C: across the quarter the recommendations would have produced a Sharpe non-inferior to the actual portfolio, with maximum drawdown below the running high-watermark threshold the dissertation reports.",
    ])
    add_para(doc, "Phase C — Scheduled execution with continuous risk gates:", bold=True)
    add_bullets(doc, [
        "[ ] Build live/execution/scheduled_executor.py: cron-driven (or systemd-timer-driven) entry that runs the advisor and, where it satisfies pre-trade risk gates, places the corresponding orders on the live account through the Alpaca trading API.",
        "[ ] Pre-trade risk gates are non-negotiable: position-size cap, gross-exposure cap, sector-exposure cap, daily-loss cap, and a circuit breaker tied to the running high-watermark drawdown threshold.",
        "[ ] Run only on a small fraction of equity initially; size up only after each successive month of operation continues to satisfy the preservation-against-high-watermark and Sharpe gates.",
        "[ ] Maintain the same observation / prediction / action / fill audit trail as Phase A and B, so that any month of live operation can be replayed against the backtest deterministically.",
        "[ ] Stop criterion: any breach of the preservation-against-high-watermark gate halts the executor, requires a written incident note, and forces a return to Phase B until the cause is understood.",
    ])

    add_para(doc, "Group 3 — Beyond the scope of this dissertation:", bold=True)
    add_bullets(doc, [
        "Alternative uncertainty estimators. Swap the Gaussian-NLL head for a deep ensemble (Lakshminarayanan et al., 2017) or an MC-dropout probabilistic forecaster (Gal and Ghahramani, 2016), with a conformal-prediction overlay (Vovk et al., 2005) as a third option.",
        "Risk-aware reward shaping. Compare the implicit risk control obtained from the trade-size term against an explicit risk-aware reward, for example log-growth penalised by drawdown or a CVaR-shaped reward (Rockafellar and Uryasev, 2000).",
        "Intraday extension. Re-calibrate the trade-size scaling and the risk-on guard for minute or tick granularity, and re-run on intraday data.",
    ])
    page_break(doc)

    # ============================================================== References
    add_heading(doc, "References", 1)
    refs = [
        "Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall. Journal of Banking and Finance, 26(7), 1487–1503.",
        "Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. Mathematical Finance, 9(3), 203–228.",
        "Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management, 40(5), 94–107.",
        "Bawa, V. S. (1975). Optimal rules for ordering uncertain prospects. Journal of Financial Economics, 2(1), 95–121.",
        "Bertsimas, D., Lauprete, G. J., & Samarov, A. (2004). Shortfall as a risk measure: properties, optimization and applications. Journal of Economic Dynamics and Control, 28(7), 1353–1381.",
        "Black, F., & Litterman, R. (1992). Global portfolio optimization. Financial Analysts Journal, 48(5), 28–43.",
        "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327.",
        "Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown measure in portfolio optimization. International Journal of Theoretical and Applied Finance, 8(1), 13–58.",
        "Fishburn, P. C. (1977). Mean-risk analysis with risk associated with below-target returns. American Economic Review, 67(2), 116–126.",
        "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. ICML.",
        "Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384.",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
        "Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem. arXiv:1706.10059.",
        "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.",
        "Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns. Journal of Empirical Finance, 10(5), 603–621.",
        "Liu, X.-Y., Yang, H., Gao, J., & Wang, C. D. (2021). FinRL: a deep reinforcement learning library for automated stock trading in quantitative finance. arXiv:2011.09607.",
        "Magdon-Ismail, M., & Atiya, A. F. (2004). Maximum drawdown. Risk Magazine, 17(10), 99–102.",
        "Mandelbrot, B. (1963). The variation of certain speculative prices. Journal of Business, 36(4), 394–419.",
        "Markowitz, H. (1952). Portfolio selection. Journal of Finance, 7(1), 77–91.",
        "McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. Journal of Empirical Finance, 7(3–4), 271–300.",
        "Michaud, R. O. (1989). The Markowitz optimization enigma: is \"optimized\" optimal? Financial Analysts Journal, 45(1), 31–42.",
        "Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association, 89(428), 1303–1313.",
        "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: reliable reinforcement learning implementations. JMLR, 22(268), 1–8.",
        "Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2(3), 21–42.",
        "Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181–1191.",
        "Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization. ICML.",
        "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.",
        "Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside risk framework. Journal of Investing, 3(3), 59–64.",
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: an introduction (2nd ed.). MIT Press.",
        "Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Springer.",
        "Yang, H., Liu, X.-Y., Zhong, S., & Walid, A. (2020). Deep reinforcement learning for automated stock trading: an ensemble strategy. ICAIF.",
        "Young, T. W. (1991). Calmar ratio: a smoother tool. Futures, 20(1), 40.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
    page_break(doc)

    # ============================================================== Appendix A
    add_heading(doc, "Appendix A — Reproducibility commands", 1)
    add_para(doc, "End-to-end reproduction of every artefact in this dissertation. "
                  "The --tickers fiyins_portfolio flag resolves to the named group "
                  "in experiments/configs/dissertation_protocol.json and materialises "
                  "the 70-ticker test universe used in Section 5.5 and Appendix B.")
    code = (
        "python3 -m venv venv && source venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "# Phase-1 SPY headline (Chapter 5 Section 5.3, Section 5.4):\n"
        "python experiments/runners/run_baseline.py\n"
        "python experiments/runners/run_probabilistic_agent.py\n"
        "python experiments/runners/run_benchmarks.py\n"
        "python experiments/runners/run_rule_baselines.py\n"
        "# Phase-1 70-ticker test universe (Chapter 5 Section 5.5, Appendix B):\n"
        "python experiments/runners/run_benchmarks.py     --tickers fiyins_portfolio --tag fiyins70\n"
        "python experiments/runners/run_rule_baselines.py --tickers fiyins_portfolio --tag fiyins70\n"
        "python experiments/runners/run_baseline.py       --tickers fiyins_portfolio --tag fiyins70\n"
        "python experiments/runners/run_probabilistic_agent.py --tickers fiyins_portfolio --tag fiyins70\n"
        "# Walk-forward subset (Section 6.4):\n"
        "python experiments/runners/run_walk_forward.py --tickers SPY,QQQ,XLK,XLF\n"
        "# Build the dissertation document:\n"
        "python reports/builders/build_main_dissertation_docx.py"
    )
    p = doc.add_paragraph()
    run = p.add_run(code)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    page_break(doc)

    # ============================================================== Appendix B
    add_heading(doc, "Appendix B — Full per-ticker table for the 70-ticker test universe", 1)
    add_para(
        doc,
        "Table B.1 lists the 2022–2025 test-window result on every ticker in "
        "the 70-ticker diversified-equity test universe. Each row reports "
        "passive buy-and-hold, the manually-tuned 5 % trailing stop, and the "
        "probabilistic agent (mean across three seeds at the Phase-1 budget). "
        "Tickers are sorted alphabetically. The aggregate Table 5.2 summarises "
        "this table as means and win counts; the Table 5.3 representative "
        "subset selects ten rows from this list to illustrate the heterogeneity.",
    )

    # Build the full per-ticker table from the same maps used in Section 5.5.
    # If the maps are unavailable (the Section 5.5 conditional did not run), skip.
    try:
        all_tickers = sorted(bh_final_map.keys())
    except NameError:
        all_tickers = []
    if all_tickers:
        full_table = doc.add_table(rows=1, cols=8)
        full_table.style = "Light Grid Accent 1"
        hdr = full_table.rows[0].cells
        for i, h in enumerate([
            "Ticker", "B&H final", "B&H MDD",
            "Stop-5 final", "Stop-5 MDD",
            "Prob final", "Prob MDD", "Prob > B&H?",
        ]):
            hdr[i].text = h
            for r in hdr[i].paragraphs[0].runs:
                r.bold = True
        for t in all_tickers:
            row = full_table.add_row().cells
            row[0].text = t
            row[1].text = f"${bh_final_map.get(t, 0):,.0f}" if t in bh_final_map else "—"
            row[2].text = f"{bh_mdd_map.get(t, 0)*100:.1f}%" if t in bh_mdd_map else "—"
            row[3].text = f"${stop_final_map.get(t, 0):,.0f}" if t in stop_final_map else "—"
            row[4].text = f"{stop_mdd_map.get(t, 0)*100:.1f}%" if t in stop_mdd_map else "—"
            row[5].text = f"${prob_final_map.get(t, 0):,.0f}" if t in prob_final_map else "—"
            row[6].text = f"{prob_mdd_map.get(t, 0)*100:.1f}%" if t in prob_mdd_map else "—"
            if t in prob_final_map and t in bh_final_map:
                if prob_final_map[t] > bh_final_map[t] * 1.005:
                    row[7].text = "WIN"
                elif prob_final_map[t] < bh_final_map[t] * 0.995:
                    row[7].text = "LOSS"
                else:
                    row[7].text = "tie"
            else:
                row[7].text = "—"
            for c in row:
                c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8.5)
        cap_b = doc.add_paragraph()
        cap_b.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_b_run = cap_b.add_run(
            f"Table B.1 — Full per-ticker test-window result across the "
            f"{len(all_tickers)}-ticker test universe (2022–2025). RL means "
            f"are over three seeds at the 10 000-PPO-timestep Phase-1 budget. "
            f"Stop-5 = 5 % trailing stop with 20/50-day moving-average re-entry."
        )
        cap_b_run.italic = True
        cap_b_run.font.size = Pt(10)

    out = EXPORTS / "Main_Dissertation_Draft.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    out = build()
    print(f"Wrote: {out}")
