"""Experiment 4 (Phase 2): the same guided search, on real reduced-round DES.

Nothing in ``src/attacks`` changes between Phase 1 and Phase 2.  ``AttackSetup``
takes the cipher module as a parameter, so pointing the whole pipeline at DES is
a matter of handing it :mod:`src.ciphers.des` and one of Matsui's approximations
instead of TinyDES-24 and a searched one.

Two configurations are available:

``--config l6``  the paper's own geometry: 8-round DES, the 6-round expression
                 ``L6``, guessing ``K0[18..23]`` (S-box 5) and ``K7[42..47]``
                 (S-box 1), |GK| = 2^12 = 4096.  This is the row the paper
                 reports at 80.2%-96.5% success for N*t = 4.0-6.4 x 10^5.
``--config l3``  5-round DES with the 3-round expression ``L3``, whose bias is
                 2^-2.35.  Three orders of magnitude less data per attack, so
                 it runs in seconds; used as a smoke test of the whole Phase-2
                 path.

Matsui's ``L5`` is deliberately *not* offered: its input mask spans five
S-boxes, so the front guess would cost 30 bits rather than 6.  Only ``L3`` and
``L6`` have an end-geometry that gives the 6 + 6 bit, |GK| = 4096 attack -- a
concrete reason the paper builds its 8-round attack on ``L6`` specifically.

Usage::

    python experiments/exp4_phase2_des_reduced_round.py --config l6 --trials 60
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import (POSTERIOR_GRID, announce_choice, ensure_results,
                    print_sweep, print_table, record_trajectories,
                    select_threshold, summarise, sweep_threshold, write_csv)

from src.attacks.candidate_space import AttackSetup
from src.attacks.multi_bit_bruteforce import run_exhaustive
from src.attacks.multi_bit_guided import RandomSearch, SequentialSearch, WKRSearch
from src.attacks.wrong_key_profile import WrongKeyProfile, check_key_independence
from src.ciphers import des
from src.distinguisher.crd import CandidateScorer
from src.distinguisher.train import train_multi_bit
from src.linear_analysis import known_masks_des

CONFIGS = {
    # name: (approximation rounds, sample dimension t, default N*t)
    "l3": (3, 16, 320),
    "l6": (6, 80, 400_000),      # the paper's own 8-round DES parameters
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), default="l6")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--nt", type=int, default=None)
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--stop-posterior", type=float, default=None,
                    help="skip calibration and use this threshold directly")
    ap.add_argument("--calibrate-trials", type=int, default=15,
                    help="trials used to calibrate the stopping rule, on a seed "
                         "range disjoint from the reported ones")
    ap.add_argument("--tolerance-pp", type=float, default=1.0)
    ap.add_argument("--budget-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=424_242)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    res = ensure_results()
    rounds, t_def, nt_def = CONFIGS[args.config]
    t = args.t or t_def
    nt = args.nt or nt_def

    print("=" * 78)
    print(f"Phase 2: real DES, {rounds + 2}-round attack with Matsui's L{rounds}")
    print("=" * 78)
    appr = known_masks_des.get_approximation(rounds, n_measure=1 << 22)
    print(appr.describe())
    if rounds in known_masks_des.PUBLISHED_P:
        print(f"  published probability  = {known_masks_des.PUBLISHED_P[rounds]:.6f}")
    print()

    setup = AttackSetup(appr, cipher=des)
    print(setup.describe())
    print(f"  -> front guess = K_0 bits "
          f"{[42 - 6 * setup.j_front + i for i in range(6)]}, "
          f"back guess = K_{setup.cipher_rounds - 1} bits "
          f"{[42 - 6 * setup.j_back + i for i in range(6)]}")
    print()

    net, meta = train_multi_bit(setup, t=t, epochs=args.epochs,
                                samples_per_epoch=args.samples_per_epoch,
                                force=args.force_train, verbose=True)
    print()
    profile = WrongKeyProfile(setup)
    print(profile.summary())
    print(f"  rho independent of the key: front={check_key_independence(setup.front_lut)}, "
          f"back={check_key_independence(setup.back_lut)}")
    print()

    space = setup.n_candidates
    print(f"  N*t = {nt:,} (N = {nt // t}, t = {t}), trials = {args.trials}")

    stop_posterior = args.stop_posterior
    cal_sweep = None
    if stop_posterior is None:
        print(f"\n  calibrating the stopping rule on {args.calibrate_trials} "
              f"independent trials (seed range {args.seed + 500_000})")
        trajs, cal_ex = record_trajectories(setup, net, t, nt, profile,
                                            args.calibrate_trials,
                                            args.seed + 500_000)
        cal_sweep = sweep_threshold(trajs, "max_post", POSTERIOR_GRID, space,
                                    "stop_posterior")
        print_sweep(cal_sweep, "stop_posterior")
        chosen = select_threshold(cal_sweep, cal_ex, args.tolerance_pp)
        announce_choice(chosen, cal_ex, args.tolerance_pp)
        stop_posterior = chosen["stop_posterior"]
        write_csv(os.path.join(res, f"exp4_des_{args.config}_calibration.csv"),
                  cal_sweep)

    budget = int(round(args.budget_frac * space))
    print(f"\n  guided stopping rule: posterior >= {stop_posterior}")
    print(f"  matched budget for the controls: {budget} ({budget / space * 100:.0f}% of |GK|)")
    print()

    rng = np.random.default_rng(args.seed)
    buckets = {k: [] for k in ["exhaustive", "sequential-earlystop", "random", "guided-wkr"]}
    per_trial = []
    for trial in range(args.trials):
        data = setup.generate(rng, nt)
        scorer = CandidateScorer(net, setup, data, t, cache=True)
        runs = [
            run_exhaustive(scorer, setup, data),
            SequentialSearch(setup, budget=space).run(scorer, data),
            RandomSearch(setup, budget=budget, seed=trial).run(scorer, data),
            WKRSearch(setup, profile=profile, budget=space,
                      stop_posterior=stop_posterior, seed=trial).run(scorer, data),
        ]
        ex = runs[0]
        for r in runs:
            buckets[r.method].append(r)
            per_trial.append({
                "trial": trial, "method": r.method, "success": int(r.success),
                "agrees_with_exhaustive": int(r.best_candidate == ex.best_candidate),
                "n_evals": r.n_evals, "seconds": r.seconds,
                "stopped_by": r.extra.get("stopped_by", ""),
                "rank_of_true": ex.rank_of_true(),
            })
        if (trial + 1) % 5 == 0:
            print(f"  trial {trial + 1}/{args.trials}")

    print()
    rows = [summarise(k, v, space) for k, v in buckets.items() if v]
    ex_rate = next(r["success_rate"] for r in rows if r["method"] == "exhaustive")
    ex_secs = next(r["seconds_mean"] for r in rows if r["method"] == "exhaustive")
    for r in rows:
        r["success_delta_pp"] = (r["success_rate"] - ex_rate) * 100
        r["agreement_with_exhaustive"] = float(np.mean(
            [x["agrees_with_exhaustive"] for x in per_trial if x["method"] == r["method"]]))
        r["speedup_vs_exhaustive"] = ex_secs / r["seconds_mean"] if r["seconds_mean"] else float("nan")
    print_table(rows)
    print()
    for r in rows:
        print(f"  {r['method']:<24} agrees {r['agreement_with_exhaustive'] * 100:5.1f}% | "
              f"success delta {r['success_delta_pp']:+5.1f} pp | "
              f"speedup {r['speedup_vs_exhaustive']:.2f}x")

    tag = args.config
    write_csv(os.path.join(res, f"exp4_des_{tag}_summary.csv"), rows)
    write_csv(os.path.join(res, f"exp4_des_{tag}_per_trial.csv"), per_trial)
    with open(os.path.join(res, f"exp4_des_{tag}_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": tag, "rounds": rounds, "cipher_rounds": setup.cipher_rounds,
                   "t": t, "nt": nt, "trials": args.trials, "space": space,
                   "p_measured": appr.p_measured, "val_acc": meta["val_acc"],
                   "bayes_acc": meta["bayes_acc"],
                   "j_front": setup.j_front, "j_back": setup.j_back,
                   "stop_posterior": stop_posterior,
                   "exhaustive_success": ex_rate}, fh, indent=2)
    print(f"\nwrote results to {res}")


if __name__ == "__main__":
    main()
