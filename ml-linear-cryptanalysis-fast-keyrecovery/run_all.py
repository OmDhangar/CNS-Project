"""Run the whole pipeline end to end.

    python run_all.py --preset demo        # ~10 min, for showing the pipeline live
    python run_all.py --preset standard    # ~45 min, usable error bars
    python run_all.py --preset full        # ~3 h, the trial counts the report quotes

A note on trial counts, because it is easy to read the presets backwards.
Raising the trial count does not make the method perform worse -- it makes the
*measurement* honest.  At 40 trials per point the success curve wobbles by
+/-20 percentage points near its steep region, which is how an earlier run
produced 87.5% and 50.0% for the same configuration.  The numbers in
``report/project_report.md`` come from the ``full`` preset for that reason.
``demo`` exists to show the machinery working in front of an audience, not to
produce numbers worth quoting.

Trained distinguishers, the verified approximation table and the calibrated
stopping rule are cached under ``artifacts/``, so a second run skips the
expensive setup.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# preset -> (approx-measure, exp1 one-bit, exp1 sweep, calibration, exp2,
#            exp3, exp4, exp4-calibration, exp5)
PRESETS = {
    "demo":     (1 << 21, 40, 8, 8, 8, 10, 8, 5, 5),
    "standard": (1 << 22, 100, 50, 25, 40, 40, 30, 12, 12),
    "full":     (1 << 24, 300, 150, 50, 150, 100, 60, 25, 25),
}


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    ap.add_argument("--skip-skopt", action="store_true",
                    help="skopt is ~50 s per attack; skip it to save time")
    ap.add_argument("--skip-des", action="store_true",
                    help="skip Phase 2 (its distinguisher takes ~20 min to train "
                         "the first time)")
    args = ap.parse_args()

    (measure, one_bit, sweep, calib, exp2, exp3,
     exp4, exp4_calib, exp5) = PRESETS[args.preset]
    print(f"preset: {args.preset}")

    run("self-tests: cipher and mask algebra", ["-m", "tests.test_cipher_and_masks"])
    run("linear approximation table",
        ["-m", "src.linear_analysis.approximations", "--measure", str(measure)])
    run("self-tests: attacks", ["-m", "tests.test_attacks"])
    run("verify Matsui's DES approximations",
        ["-m", "src.linear_analysis.known_masks_des", "--measure", str(measure)])
    run("exp1: reproduce the paper's framework",
        ["experiments/exp1_reproduce_baseline.py",
         "--trials", str(one_bit), "--multi-trials", str(sweep)])
    run("calibrate the stopping rule (independent seeds)",
        ["experiments/calibrate_stopping.py", "--trials", str(calib)])
    run("exp2: guided vs exhaustive (headline)",
        ["experiments/exp2_guided_vs_bruteforce.py", "--trials", str(exp2)]
        + (["--skip-skopt"] if args.skip_skopt else []))
    run("exp3: budget sensitivity (the headline curve)",
        ["experiments/exp3_budget_sensitivity.py", "--trials", str(exp3)])
    if not args.skip_des:
        run("exp4: Phase 2, the same search on real 8-round DES",
            ["experiments/exp4_phase2_des_reduced_round.py", "--config", "l6",
             "--trials", str(exp4), "--calibrate-trials", str(exp4_calib)])
    run("exp5: does a covering opening design help? (it does not)",
        ["experiments/exp5_opening_design.py", "--trials", str(exp5)])
    run("figures", ["experiments/make_plots.py"])
    run("assemble report", ["report/build_report.py"])
    print("\nall stages complete; see results/ and report/project_report.md")


if __name__ == "__main__":
    main()
