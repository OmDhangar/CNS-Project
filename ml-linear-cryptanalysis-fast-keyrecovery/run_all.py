"""Run the whole pipeline end to end.

    python run_all.py               # full run (slow; ~1-2 h on a CPU)
    python run_all.py --quick       # reduced trial counts for a smoke test

Stages are skipped if their outputs already exist unless --force is given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(label, args, cwd=ROOT):
    print("\n" + "=" * 78)
    print(f">>> {label}")
    print("=" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run([PY] + args, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"stage failed: {label} (exit {proc.returncode})")
    print(f"--- {label}: {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    q = args.quick
    run("self-tests: cipher and mask algebra", ["-m", "tests.test_cipher_and_masks"])
    run("linear approximation table", ["-m", "src.linear_analysis.approximations",
                                       "--measure", "4194304" if q else "16777216"])
    run("self-tests: attacks", ["-m", "tests.test_attacks"])
    run("verify Matsui's DES approximations", ["-m", "src.linear_analysis.known_masks_des",
                                               "--measure", "4194304" if q else "8388608"])
    run("exp1: reproduce the paper's framework",
        ["experiments/exp1_reproduce_baseline.py",
         "--trials", "50" if q else "300",
         "--multi-trials", "10" if q else "40"])
    run("calibrate the stopping rule (independent seeds)",
        ["experiments/calibrate_stopping.py", "--trials", "10" if q else "30"])
    run("exp2: guided vs exhaustive (headline)",
        ["experiments/exp2_guided_vs_bruteforce.py",
         "--trials", "10" if q else "100"] + (["--skip-skopt"] if q else []))
    run("exp3: budget sensitivity",
        ["experiments/exp3_budget_sensitivity.py", "--trials", "10" if q else "60"])
    run("exp4: Phase 2, the same search on real 8-round DES",
        ["experiments/exp4_phase2_des_reduced_round.py", "--config", "l6",
         "--trials", "10" if q else "60",
         "--calibrate-trials", "5" if q else "20"])
    run("figures", ["experiments/make_plots.py"])
    run("assemble report", ["report/build_report.py"])
    print("\nall stages complete; see results/ and report/project_report.md")


if __name__ == "__main__":
    main()
