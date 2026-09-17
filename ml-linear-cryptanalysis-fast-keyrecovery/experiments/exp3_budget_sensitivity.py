"""Experiment 3: success rate as a function of the query budget.

This produces the project's headline figure -- how much of the 4096-candidate
space can be skipped, and what it costs in success rate.

Each trial runs every search strategy once with the full |GK| budget and
stopping disabled, recording the best-so-far candidate after every query.  A
fixed budget B is then replayed offline by reading position B-1 of that
trajectory, so the whole curve comes from one pass per method rather than one
run per (method, budget) pair.

Because every strategy in a trial queries the same candidate set eventually,
the scorer's memo cache is shared across strategies within a trial (a harness
speed-up only: the reported evaluation counts are per-run and cache-independent,
see :class:`CandidateScorer`).  Wall-clock numbers therefore come from exp2,
not from here.

Usage::

    python experiments/exp3_budget_sensitivity.py --trials 60
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import DEFAULT_NT, DEFAULT_T, build, ensure_results, write_csv

from src.attacks.multi_bit_bruteforce import run_exhaustive
from src.attacks.multi_bit_guided import RandomSearch, SequentialSearch, WKRSearch
from src.attacks.wrong_key_profile import WrongKeyProfile
from src.distinguisher.crd import CandidateScorer

BUDGET_FRACS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
                0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--nt", type=int, default=DEFAULT_NT)
    ap.add_argument("--t", type=int, default=DEFAULT_T)
    ap.add_argument("--seed", type=int, default=777_001)
    ap.add_argument("--skopt", action="store_true",
                    help="also sweep scikit-optimize (slow: GP fitting is O(n^3))")
    ap.add_argument("--skopt-budget-frac", type=float, default=0.10)
    args = ap.parse_args()

    res = ensure_results()
    appr, setup, net, meta = build(t=args.t, verbose=False)
    profile = WrongKeyProfile(setup)
    space = setup.n_candidates
    rng = np.random.default_rng(args.seed)

    print(setup.describe())
    print(f"  N*t = {args.nt:,}, t = {args.t}, trials = {args.trials}")
    print()

    methods = ["guided-wkr", "random", "sequential"]
    trajs = {m: [] for m in methods}
    ex_success, ex_best = [], []

    for trial in range(args.trials):
        data = setup.generate(rng, args.nt)
        scorer = CandidateScorer(net, setup, data, args.t, cache=True,
                                 persistent_cache=True)
        r_ex = run_exhaustive(scorer, setup, data)
        ex_success.append(int(r_ex.success))
        ex_best.append(int(r_ex.best_candidate))

        runs = {
            "guided-wkr": WKRSearch(setup, profile=profile, budget=space,
                                    stop_z=None, stop_posterior=None,
                                    seed=trial).run(scorer, data),
            "random": RandomSearch(setup, budget=space, seed=trial).run(scorer, data),
            "sequential": SequentialSearch(setup, budget=space, stop_z=None
                                           ).run(scorer, data),
        }
        for m, r in runs.items():
            best = np.asarray(r.extra["best_so_far"])
            trajs[m].append({
                "succeeds": (best == r.true_candidate).astype(np.int8),
                "agrees": (best == r_ex.best_candidate).astype(np.int8),
            })
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial + 1}/{args.trials}")

    ex_rate = float(np.mean(ex_success))
    print(f"\nexhaustive Algorithm 2 success on this set: {ex_rate * 100:.1f}% "
          f"({sum(ex_success)}/{args.trials})")

    rows = []
    for frac in BUDGET_FRACS:
        b = max(1, int(round(frac * space)))
        row = {"budget_frac": frac, "budget": b,
               "exhaustive_success": ex_rate}
        for m in methods:
            row[f"{m}_success"] = float(np.mean(
                [t["succeeds"][min(b, len(t["succeeds"])) - 1] for t in trajs[m]]))
            row[f"{m}_agreement"] = float(np.mean(
                [t["agrees"][min(b, len(t["agrees"])) - 1] for t in trajs[m]]))
        rows.append(row)

    print()
    hdr = (f"{'budget':>8}{'% |GK|':>9}" + "".join(f"{m:>20}" for m in methods))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['budget']:>8}{r['budget_frac'] * 100:>8.1f}%"
              + "".join(f"{r[f'{m}_success'] * 100:>19.1f}%" for m in methods))
    print(f"{'exhaustive':>17}{ex_rate * 100:>19.1f}%")

    # The headline number: smallest budget at which the guided search matches
    # the exhaustive success rate to within one percentage point.
    hit = [r for r in rows if r["guided-wkr_success"] >= ex_rate - 0.01]
    if hit:
        h = min(hit, key=lambda r: r["budget"])
        print(f"\nguided-wkr reaches exhaustive success (within 1 pp) at "
              f"{h['budget']} queries = {h['budget_frac'] * 100:.1f}% of |GK|")
    for m in ("random", "sequential"):
        hit = [r for r in rows if r[f"{m}_success"] >= ex_rate - 0.01]
        if hit:
            h = min(hit, key=lambda r: r["budget"])
            print(f"  {m:<12} needs {h['budget']} queries "
                  f"= {h['budget_frac'] * 100:.1f}% of |GK|")
        else:
            print(f"  {m:<12} never reaches it below |GK|")

    write_csv(os.path.join(res, "exp3_budget_sensitivity.csv"), rows)
    with open(os.path.join(res, "exp3_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"trials": args.trials, "nt": args.nt, "t": args.t,
                   "space": space, "exhaustive_success": ex_rate,
                   "val_acc": meta["val_acc"], "p_r": appr.p}, fh, indent=2)
    print(f"\nwrote results to {res}")


if __name__ == "__main__":
    main()
