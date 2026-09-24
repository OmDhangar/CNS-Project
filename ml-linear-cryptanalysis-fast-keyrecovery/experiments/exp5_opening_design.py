"""Experiment 5: does a covering opening design make the search stop sooner?

Motivation
----------
The measured weakness of the guided search is not finding the key -- that is
cheap -- but *becoming confident* that it has been found.  The suspected cause
was structural: a hypothesis that no observation has touched keeps its prior, so
if the search only ever probes high-posterior candidates, most of the space
stays untouched and the posterior cannot concentrate without near-exhaustive
testing.

That suggested an obvious fix.  Because the wrong-key response factorises, a
probe set can be chosen **offline** so that every one of the 4096 hypotheses is
within a given response level of some probe (see
``WrongKeyProfile.covering_design``).  Spend the opening queries covering the
space, and every hypothesis has evidence against it; then refine.

This experiment tests that idea against the plain random opening, at several
coverage levels, on a seed range independent of the reported trials.  Both
metrics are reported, because they answer different questions:

* **fixed budget** -- of the two, which finds the key faster?
* **stopping**     -- of the two, which knows it has found the key sooner?

Usage::

    python experiments/exp5_opening_design.py --trials 25
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import (DEFAULT_NT, DEFAULT_T, POSTERIOR_GRID, build, ensure_results,
                    select_threshold, sweep_threshold, write_csv)

from src.attacks.multi_bit_bruteforce import run_exhaustive
from src.attacks.multi_bit_guided import WKRSearch
from src.attacks.wrong_key_profile import WrongKeyProfile
from src.distinguisher.crd import CandidateScorer

BUDGET_FRACS = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.50, 1.00)


def trace(setup, net, profile, t, nt, trials, seed, design, cover_level):
    rng = np.random.default_rng(seed)
    space = setup.n_candidates
    trajs, ex = [], 0
    for trial in range(trials):
        data = setup.generate(rng, nt)
        sc = CandidateScorer(net, setup, data, t, cache=True, persistent_cache=True)
        r_ex = run_exhaustive(sc, setup, data)
        ex += r_ex.success
        r = WKRSearch(setup, profile=profile, budget=space, stop_z=None,
                      stop_posterior=None, seed=trial, design=design,
                      cover_level=cover_level).run(sc, data, record_diagnostics=True)
        best = np.asarray(r.extra["best_so_far"])
        trajs.append({
            "max_post": np.asarray(r.extra["diag_max_posterior"]),
            "best_z": np.asarray(r.extra["diag_best_z"]),
            "agrees": (best == r_ex.best_candidate).astype(int),
            "succeeds": (best == r.true_candidate).astype(int),
            "min_queries": 32,
        })
    return trajs, ex / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--nt", type=int, default=DEFAULT_NT)
    ap.add_argument("--t", type=int, default=DEFAULT_T)
    ap.add_argument("--seed", type=int, default=2_000_003)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25, 0.375, 0.5])
    args = ap.parse_args()

    res = ensure_results()
    appr, setup, net, _ = build(t=args.t, verbose=False)
    profile = WrongKeyProfile(setup)
    space = setup.n_candidates

    print(f"{args.trials} trials, N*t = {args.nt:,}, |GK| = {space}\n")
    print("offline covering designs available:")
    for lv in args.levels:
        _, info = profile.covering_design(level=lv)
        print(f"   level {lv:.3f} -> {info['n_probes']:5d} probes "
              f"({info['fraction_of_space'] * 100:5.1f}% of |GK|)")
    print()

    configs = [("random", None)] + [("covering", lv) for lv in args.levels]
    rows, budget_rows = [], []
    for design, lv in configs:
        name = design if lv is None else f"covering@{lv:g}"
        trajs, ex = trace(setup, net, profile, args.t, args.nt,
                          args.trials, args.seed, design, lv or 0.25)
        sweep = sweep_threshold(trajs, "max_post", np.asarray(POSTERIOR_GRID),
                                space, "stop_posterior")
        ch = select_threshold(sweep, ex, 1.0)
        rows.append({
            "design": name, "trials": args.trials, "exhaustive_success": ex,
            "stop_posterior": ch["stop_posterior"],
            "queries_median": ch["queries_median"],
            "queries_frac_median": ch["queries_frac_median"],
            "queries_p95": ch["queries_p95"],
            "success": ch["success"], "agreement": ch["agreement"],
        })
        for frac in BUDGET_FRACS:
            b = max(1, int(round(frac * space)))
            budget_rows.append({
                "design": name, "budget_frac": frac, "budget": b,
                "success": float(np.mean([t["succeeds"][min(b, len(t["succeeds"])) - 1]
                                          for t in trajs])),
                "agreement": float(np.mean([t["agrees"][min(b, len(t["agrees"])) - 1]
                                            for t in trajs])),
            })
        print(f"  {name:<16} done (exhaustive success {ex * 100:.1f}%)")

    print("\nFIXED BUDGET -- which finds the key faster? (agreement with exhaustive)")
    hdr = f"{'budget':>8}{'% |GK|':>9}" + "".join(f"{r['design']:>16}" for r in rows)
    print(hdr)
    print("-" * len(hdr))
    for frac in BUDGET_FRACS:
        line = f"{max(1, int(round(frac * space))):>8}{frac * 100:>8.1f}%"
        for r in rows:
            v = next(b for b in budget_rows
                     if b["design"] == r["design"] and b["budget_frac"] == frac)
            line += f"{v['agreement'] * 100:>15.1f}%"
        print(line)

    print("\nSTOPPING -- which knows it has found the key sooner?")
    print(f"{'design':<16}{'threshold':>11}{'queries(med)':>14}{'% |GK|':>9}"
          f"{'queries(p95)':>14}{'success':>10}{'agreement':>11}")
    print("-" * 85)
    for r in rows:
        print(f"{r['design']:<16}{r['stop_posterior']:>11.3f}"
              f"{r['queries_median']:>14.0f}{r['queries_frac_median'] * 100:>8.1f}%"
              f"{r['queries_p95']:>14.0f}{r['success'] * 100:>9.1f}%"
              f"{r['agreement'] * 100:>10.1f}%")

    best = min(rows, key=lambda r: r["queries_median"])
    print(f"\ncheapest to stop: {best['design']} "
          f"({best['queries_frac_median'] * 100:.1f}% of |GK|)")

    write_csv(os.path.join(res, "exp5_opening_design.csv"), rows)
    write_csv(os.path.join(res, "exp5_opening_design_budget.csv"), budget_rows)
    print(f"\nwrote results to {res}")


if __name__ == "__main__":
    main()
