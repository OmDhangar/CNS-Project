"""Experiment 2 (headline): guided search vs. exhaustive Algorithm 2.

Every method scores candidates through the *same* trained distinguisher, the
same CRD rule and the same plaintext/ciphertext data, so the only thing that
differs is the order in which candidates are visited and when the search stops.

Methods
-------
exhaustive              Algorithm 2 of the paper, verbatim -- the control.
sequential-earlystop    index-order scan + our stopping rule (isolates the
                        contribution of stopping early, on its own).
random                  uniform sampling at a matched budget (isolates the
                        contribution of the model, on its own).
guided-skopt            scikit-optimize GP search over the 12 guessed bits.
guided-wkr              ours: wrong-key-response-guided Bayesian search.

Usage::

    python experiments/exp2_guided_vs_bruteforce.py --trials 100
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import (DEFAULT_NT, DEFAULT_T, build, ensure_results, print_table,
                    read_calibration, summarise, write_csv)

from src.attacks.multi_bit_bruteforce import run_exhaustive
from src.attacks.multi_bit_guided import (RandomSearch, SequentialSearch,
                                          SkoptSearch, WKRSearch)
from src.attacks.wrong_key_profile import WrongKeyProfile, check_key_independence
from src.distinguisher.crd import CandidateScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--nt", type=int, default=DEFAULT_NT, help="N * t plaintexts")
    ap.add_argument("--t", type=int, default=DEFAULT_T, help="sample dimension")
    ap.add_argument("--budget-frac", type=float, default=None,
                    help="budget for the budgeted controls, as a fraction of "
                         "|GK|; default matches the calibrated median of the "
                         "guided search")
    ap.add_argument("--seed", type=int, default=20250917)
    ap.add_argument("--skip-skopt", action="store_true",
                    help="skip the scikit-optimize comparison entirely")
    ap.add_argument("--skopt-budget", type=int, default=100,
                    help="scikit-optimize gets its own, much smaller budget: it "
                         "refits a Gaussian process on every observation, which "
                         "is O(n^3), so it cannot be run at the same budget as "
                         "the others")
    ap.add_argument("--skopt-trials", type=int, default=10,
                    help="and its own, much smaller trial count, for the same "
                         "reason; reported separately in the table")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    res_dir = ensure_results()
    appr, setup, net, meta = build(t=args.t, force_train=args.force_train)
    profile = WrongKeyProfile(setup)
    space = setup.n_candidates

    # The guided search uses the stopping rule calibrated on an independent
    # seed range; the budgeted controls get a budget matched to the median
    # number of queries that rule was measured to use, so "random at the same
    # cost" really is the same cost.
    cal = read_calibration()
    stop_posterior = cal.get("stop_posterior", WKRSearch.DEFAULT_STOP_POSTERIOR)
    if args.budget_frac is not None:
        budget = int(round(args.budget_frac * space))
    else:
        budget = int(round(cal.get("queries_median", 0.25 * space)))
    budget = max(32, min(space, budget))

    print(setup.describe())
    print(f"  distinguisher ND_{appr.rounds}^{args.t}: validation accuracy "
          f"{meta['val_acc'] * 100:.3f}% (fixed-p Bayes reference "
          f"{meta['bayes_acc'] * 100:.3f}%)")
    print(f"  data per attack N*t = {args.nt:,}  (N = {args.nt // args.t}, t = {args.t})")
    print(f"  guided search stopping rule: posterior >= {stop_posterior:.2f} "
          f"(calibrated on {cal.get('trials', '?')} independent trials)")
    print(f"  matched budget for the budgeted controls: {budget} "
          f"({budget / space * 100:.1f}% of |GK|)")
    if not args.skip_skopt:
        print(f"  scikit-optimize: budget {args.skopt_budget}, "
              f"{args.skopt_trials} trials (its GP refit dominates its runtime)")
    print()
    print(profile.summary())
    print(f"  rho independent of the key: front={check_key_independence(setup.front_lut)}, "
          f"back={check_key_independence(setup.back_lut)}")
    print()

    rng = np.random.default_rng(args.seed)
    per_trial = []
    buckets = {k: [] for k in
               ["exhaustive", "sequential-earlystop", "random", "guided-skopt", "guided-wkr"]}
    trajectories = []

    t_start = time.time()
    for trial in range(args.trials):
        data = setup.generate(rng, args.nt)
        scorer = CandidateScorer(net, setup, data, args.t, cache=True)

        r_ex = run_exhaustive(scorer, setup, data)
        r_sq = SequentialSearch(setup, budget=space).run(scorer, data)
        r_rd = RandomSearch(setup, budget=budget, seed=trial).run(scorer, data)
        r_wk = WKRSearch(setup, profile=profile, budget=space,
                         stop_posterior=stop_posterior, seed=trial).run(scorer, data)
        runs = [r_ex, r_sq, r_rd, r_wk]
        if not args.skip_skopt and trial < args.skopt_trials:
            runs.append(SkoptSearch(setup, budget=min(args.skopt_budget, budget),
                                    seed=trial).run(scorer, data))

        for r in runs:
            buckets[r.method].append(r)
            per_trial.append({
                "trial": trial, "method": r.method,
                "success": int(r.success),
                "agrees_with_exhaustive": int(r.best_candidate == r_ex.best_candidate),
                "best_candidate": r.best_candidate,
                "true_candidate": r.true_candidate,
                "gamma_correct": int(r.gamma_correct),
                "n_evals": r.n_evals, "seconds": r.seconds,
                "stopped_by": r.extra.get("stopped_by", ""),
                "duplicate_proposals": r.extra.get("duplicate_proposals", 0),
                "rank_of_true": r_ex.rank_of_true(),
            })
        trajectories.append({
            "trial": trial,
            "true_candidate": int(data.true_candidate),
            "exhaustive_best": int(r_ex.best_candidate),
            "exhaustive_success": bool(r_ex.success),
            "wkr_visited": [int(c) for c in r_wk.visited],
            "wkr_best_so_far": [int(c) for c in r_wk.extra["best_so_far"]],
            "random_visited": [int(c) for c in r_rd.visited],
            "random_best_so_far": [int(c) for c in r_rd.extra["best_so_far"]],
        })

        if (trial + 1) % 5 == 0 or trial == args.trials - 1:
            el = time.time() - t_start
            print(f"  trial {trial + 1}/{args.trials}  "
                  f"({el:.0f}s elapsed, {el / (trial + 1):.1f}s/trial)")

    print()
    rows = [summarise(k, v, space) for k, v in buckets.items() if v]
    ex_rate = next(r["success_rate"] for r in rows if r["method"] == "exhaustive")
    ex_by_trial = {x["trial"]: x["success"]
                   for x in per_trial if x["method"] == "exhaustive"}
    for r in rows:
        mine = [x for x in per_trial if x["method"] == r["method"]]
        # skopt runs on a subset of the trials, so compare it against the
        # exhaustive success rate on that same subset, not on all trials.
        matched = float(np.mean([ex_by_trial[x["trial"]] for x in mine]))
        r["exhaustive_success_on_same_trials"] = matched
        r["success_delta_pp"] = (r["success_rate"] - matched) * 100
        r["agreement_with_exhaustive"] = float(np.mean(
            [x["agrees_with_exhaustive"] for x in mine]))
        r["speedup_vs_exhaustive"] = (
            next(q["seconds_mean"] for q in rows if q["method"] == "exhaustive")
            / r["seconds_mean"]) if r["seconds_mean"] > 0 else float("nan")
    print_table(rows)
    print()
    for r in rows:
        print(f"  {r['method']:<24} agrees with exhaustive argmax "
              f"{r['agreement_with_exhaustive'] * 100:5.1f}% | "
              f"success delta {r['success_delta_pp']:+5.1f} pp | "
              f"wall-clock speedup {r['speedup_vs_exhaustive']:.2f}x")

    write_csv(os.path.join(res_dir, "exp2_summary.csv"), rows)
    write_csv(os.path.join(res_dir, "exp2_per_trial.csv"), per_trial)
    with open(os.path.join(res_dir, "exp2_trajectories.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": {"trials": args.trials, "nt": args.nt, "t": args.t,
                              "budget": budget, "space": space,
                              "val_acc": meta["val_acc"], "p_r": appr.p},
                   "trajectories": trajectories}, fh)
    print(f"\nwrote results to {res_dir}")


if __name__ == "__main__":
    main()
