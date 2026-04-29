"""Extended-grid runner.

A thin orchestrator that invokes the four existing runners back-to-back with
a single set of CLI flags, so the entire 'do the heavy thing' workflow is one
command. Every flag is forwarded to the underlying runners; sensible defaults
target the dissertation's full extended grid (10 seeds, 50 000 steps, all four
walk-forward folds, 16 bootstrap paths, 70-ticker portfolio).

This is the script the README's Colab section calls. On CPU it is a slow run
(2–3 days for the full grid); on a Colab T4 it is roughly 4–6 hours.

Examples:

    # The full 70-ticker × 10-seed × 4-fold × 50k-step × 16-bootstrap-path grid
    python experiments/run_extended_grid.py \\
        --tickers fiyins_portfolio --seeds extended --folds all \\
        --timesteps 50000 --bootstrap-paths 16 --tag colab_70_extended

    # A faster basket sanity check (8 tickers, 10 seeds, 50k steps, single fold)
    python experiments/run_extended_grid.py \\
        --tickers basket --seeds extended --timesteps 50000 \\
        --skip-walk-forward --tag basket_extended

The runner does not duplicate any logic: it shells out to the underlying
runners (run_benchmarks.py, run_rule_baselines.py, run_walk_forward.py and
the legacy single-fold runners) and forwards every flag. Failures in one
stage do not abort the overall pipeline; each stage's exit status is logged.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _run(label: str, argv: list[str]) -> tuple[int, float]:
    """Invoke a python script and stream its output. Returns (exit_code, secs)."""
    print(f"\n{'=' * 78}\n>>> {label}\n>>> {' '.join(shlex.quote(a) for a in argv)}\n{'=' * 78}")
    t0 = time.time()
    try:
        result = subprocess.run(argv, cwd=str(PROJECT_ROOT), check=False)
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f">>> {label}: {status} after {elapsed:.0f} s")
        return result.returncode, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        print(f">>> {label}: EXCEPTION after {elapsed:.0f} s: {exc}")
        return 1, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Forwarded to every underlying runner.
    parser.add_argument("--tickers", default="fiyins_portfolio",
                        help="Ticker spec (named group, comma-separated, or 'basket').")
    parser.add_argument("--seeds", default="extended",
                        help="Seed spec ('default', 'extended', or comma-separated).")
    parser.add_argument("--folds", default="all",
                        help="Walk-forward fold spec ('all' or comma-separated fold ids).")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="PPO training budget per cell.")
    parser.add_argument("--bootstrap-paths", type=int, default=16,
                        help="Politis & Romano (1994) stationary block-bootstrap paths.")
    parser.add_argument("--initial-balance", type=float, default=1_000_000.0,
                        help="Starting capital per ticker.")
    parser.add_argument("--tag", default="extended",
                        help="Output filename suffix.")
    parser.add_argument("--agents", default="baseline,probabilistic",
                        help="Walk-forward agent subset.")
    # Skip flags so subsets of the pipeline can be run.
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-rule", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-probabilistic", action="store_true")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")

    args = parser.parse_args()

    py = sys.executable

    common = [
        "--tickers", args.tickers,
        "--initial-balance", str(args.initial_balance),
        "--tag", args.tag,
    ]
    rl_common = common + [
        "--seeds", args.seeds,
        "--timesteps", str(args.timesteps),
    ]
    prob_common = rl_common + ["--bootstrap-paths", str(args.bootstrap_paths)]
    wf_common = common + [
        "--seeds", args.seeds,
        "--folds", args.folds,
        "--timesteps", str(args.timesteps),
        "--bootstrap-paths", str(args.bootstrap_paths),
        "--agents", args.agents,
    ]

    summary: list[tuple[str, int, float]] = []
    if not args.skip_benchmarks:
        rc, secs = _run("Benchmarks", [py, "experiments/run_benchmarks.py", *common])
        summary.append(("benchmarks", rc, secs))
    if not args.skip_rule:
        rc, secs = _run("Rule-based stop-loss", [py, "experiments/run_rule_baselines.py", *common])
        summary.append(("rule_baseline", rc, secs))
    if not args.skip_baseline:
        rc, secs = _run("Baseline PPO (single-fold extended)", [py, "experiments/run_baseline.py", *rl_common])
        summary.append(("baseline_single_fold", rc, secs))
    if not args.skip_probabilistic:
        rc, secs = _run("Probabilistic PPO (single-fold extended)",
                        [py, "experiments/run_probabilistic_agent.py", *prob_common])
        summary.append(("probabilistic_single_fold", rc, secs))
    if not args.skip_walk_forward:
        rc, secs = _run("Walk-forward grid (both agents, all folds)",
                        [py, "experiments/run_walk_forward.py", *wf_common])
        summary.append(("walk_forward", rc, secs))
    if not args.skip_aggregate:
        rc, secs = _run("Aggregate results", [py, "experiments/aggregate_results.py", "--tag", args.tag])
        summary.append(("aggregate", rc, secs))

    # ----- Final summary -----
    print(f"\n{'#' * 78}\n# Extended grid summary (tag={args.tag})\n{'#' * 78}")
    total_secs = sum(s for _, _, s in summary)
    any_failed = False
    for label, rc, secs in summary:
        status = "OK" if rc == 0 else f"FAILED ({rc})"
        any_failed = any_failed or (rc != 0)
        print(f"  {label:<32s} {status:<12s} {secs:>8.0f} s")
    print(f"  {'TOTAL':<32s} {'':<12s} {total_secs:>8.0f} s "
          f"({total_secs / 60:.1f} min, {total_secs / 3600:.2f} h)")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
