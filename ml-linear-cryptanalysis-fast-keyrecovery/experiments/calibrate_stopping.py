"""Calibrate the Phase-1 guided search's stopping rule on independent trials.

The stopping threshold must not be tuned on the trials the headline experiment
reports, so this script runs on its own seed range, writes what it chooses to
``artifacts/stopping_calibration.json``, and exp2/exp3 read it from there while
running on different seeds.

Method: run the WKR search once per trial with stopping *disabled* and the full
|GK| budget, recording after every query the best-so-far candidate, the
best-so-far robust z-score, and the maximum posterior.  Every candidate
stopping rule and every fixed budget can then be replayed offline against those
trajectories, so the whole sweep costs no extra attack runs -- and because the
diagnostics are recorded inside the real search loop, a calibration run and a
production run with the same seed follow identical trajectories.

Two different questions are answered, and they have different answers:

* **agreement** -- does the search return what exhaustive Algorithm 2 would
  have returned?  This is the pure search-quality question.
* **success**  -- does the search return the actual subkey?  This is the
  cryptanalytic question, and it is the one the paper reports.  It can exceed
  agreement, because on trials where the true key is *not* the global argmax
  the exhaustive scan is wrong by construction while a search that stops early
  can still be holding the right answer.

The same helpers calibrate the Phase-2 DES configuration inside
``exp4_phase2_des_reduced_round.py``: the rule is configuration-dependent (how
fast the posterior concentrates depends on the bias, the data complexity and
the shape of the wrong-key response), so each configuration calibrates its own.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import (CALIBRATION_PATH, DEFAULT_NT, DEFAULT_T, POSTERIOR_GRID,
                    ROOT, announce_choice, build, ensure_results,
                    operating_points, print_operating_points, print_sweep,
                    record_trajectories, select_threshold, sweep_threshold,
                    write_csv)

from src.attacks.wrong_key_profile import WrongKeyProfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--nt", type=int, default=DEFAULT_NT)
    ap.add_argument("--t", type=int, default=DEFAULT_T)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--tolerance-pp", type=float, default=1.0,
                    help="allowed success-rate loss vs exhaustive, in percentage points")
    ap.add_argument("--reselect", action="store_true",
                    help="re-pick the threshold from the cached sweep, no attacks")
    args = ap.parse_args()

    if args.reselect:
        with open(CALIBRATION_PATH, encoding="utf-8") as fh:
            cal = json.load(fh)
        prows, ex_rate = cal["sweep_posterior"], cal["exhaustive_success"]
        print_sweep(prows, "stop_posterior")
        chosen = select_threshold(prows, ex_rate, args.tolerance_pp)
        announce_choice(chosen, ex_rate, args.tolerance_pp)
        cal.update({k: chosen[k] for k in
                    ("stop_posterior", "agreement", "success", "queries_median")})
        with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
            json.dump(cal, fh, indent=2)
        print(f"updated {CALIBRATION_PATH}")
        return

    ensure_results()
    appr, setup, net, meta = build(t=args.t, verbose=False)
    profile = WrongKeyProfile(setup)
    space = setup.n_candidates

    print(f"calibrating on {args.trials} independent trials, N*t = {args.nt:,}, "
          f"t = {args.t}, |GK| = {space}")
    trajs, ex_rate = record_trajectories(setup, net, args.t, args.nt, profile,
                                         args.trials, args.seed)
    print(f"  exhaustive Algorithm 2 success on this set: {ex_rate * 100:.0f}%")

    # ---------------- fixed-budget replay ----------------
    budget_rows = []
    for frac in (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
                 0.30, 0.40, 0.50, 0.75, 1.00):
        b = max(1, int(round(frac * space)))
        budget_rows.append({
            "budget_frac": frac, "budget": b,
            "success": float(np.mean([t["succeeds"][min(b, len(t["succeeds"])) - 1]
                                      for t in trajs])),
            "agreement": float(np.mean([t["agrees"][min(b, len(t["agrees"])) - 1]
                                        for t in trajs])),
        })
    print()
    print("fixed-budget replay (no early stopping):")
    print(f"{'budget':>9}{'% of |GK|':>11}{'success':>10}{'agreement':>11}")
    print("-" * 41)
    for r in budget_rows:
        print(f"{r['budget']:>9}{r['budget_frac'] * 100:>10.1f}%"
              f"{r['success'] * 100:>9.1f}%{r['agreement'] * 100:>10.1f}%")
    print(f"{'(exhaustive)':>20}{ex_rate * 100:>9.1f}%{100.0:>10.1f}%")

    # ---------------- stopping-rule sweeps ----------------
    zrows = sweep_threshold(trajs, "best_z", np.arange(3.0, 9.01, 0.25),
                            space, "stop_z")
    print()
    print("early-stopping sweep A: extreme-value rule on the best robust z-score")
    print_sweep(zrows, "stop_z")

    prows = sweep_threshold(trajs, "max_post", np.asarray(POSTERIOR_GRID),
                            space, "stop_posterior")
    print()
    print("early-stopping sweep B: posterior-concentration rule")
    print_sweep(prows, "stop_posterior")

    points = operating_points(prows)
    print()
    print("operating points -- cheapest threshold reaching each agreement target:")
    print_operating_points(points)
    write_csv(os.path.join(ROOT, "results", "calibration_operating_points.csv"), points)

    chosen = select_threshold(prows, ex_rate, args.tolerance_pp)
    announce_choice(chosen, ex_rate, args.tolerance_pp)

    write_csv(os.path.join(ROOT, "results", "calibration_stopping_z.csv"), zrows)
    write_csv(os.path.join(ROOT, "results", "calibration_stopping_posterior.csv"), prows)
    write_csv(os.path.join(ROOT, "results", "calibration_budget.csv"), budget_rows)
    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
        json.dump({"stop_posterior": chosen["stop_posterior"],
                   "nt": args.nt, "t": args.t,
                   "trials": args.trials, "seed": args.seed,
                   "exhaustive_success": ex_rate,
                   "agreement": chosen["agreement"], "success": chosen["success"],
                   "queries_median": chosen["queries_median"],
                   "sweep_z": zrows, "sweep_posterior": prows,
                   "budget_sweep": budget_rows}, fh, indent=2)
    print(f"wrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
