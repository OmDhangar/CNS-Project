"""Experiment 1: reproduce the paper's framework before trying to improve it.

Three checks, in the order the paper presents them:

1.  **Distinguishers.**  Train ND_r^t in both of the paper's data formats and
    report accuracy against the Bayes-optimal accuracy at the measured p_r.
    The paper's own usability bar is 51%; the reference says how much of the
    available signal the network actually extracts.  (Accuracy alone turns out
    not to be a sufficient check -- see the report, Sect. 4.3 -- because the
    CRD sums logits rather than using the network's decision.)

2.  **One-bit key recovery** (paper's Sect. "ML-aided one bit key recovery",
    Table 5).  Recover gamma.K with the CRD rule and compare against Matsui's
    Algorithm 1 on the *same* number of plaintexts, which is exactly the
    comparison the paper makes.

3.  **Multi-bit key recovery** (Algorithm 2, Table's 8-round DES row).  Sweep
    N*t and report the success rate, reproducing the shape of the paper's
    "80.2% / 88.4% / 93.6% / 96.5%" sweep and fixing the operating point that
    exp2 and exp3 then use.

Usage::

    python experiments/exp1_reproduce_baseline.py --trials 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import DEFAULT_T, build, ensure_results, write_csv

from src.attacks.candidate_space import AttackSetup
from src.attacks.multi_bit_bruteforce import run_exhaustive
from src.distinguisher import data_gen
from src.distinguisher.crd import CandidateScorer, ClassicalScorer, one_bit_crd
from src.distinguisher.train import train_one_bit
from src.linear_analysis.approximations import get_approximation


# ---------------------------------------------------------------------------
# 2. one-bit key recovery
# ---------------------------------------------------------------------------

def one_bit_attack_ml(net, appr, rng, n_samples, t, trials):
    """Paper's one-bit framework, in two variants.

    Returns (success with the paper's literal Step-4 rule,
             success with the antisymmetrised CRD -- see crd.one_bit_crd).
    """
    ok_plain = ok_sym = 0
    for _ in range(trials):
        omega = data_gen.one_bit_omega(appr, rng, 1, n_samples * t).reshape(n_samples, t)
        # The attacker observes x = alpha.P ^ beta.C = omega ^ gamma.K.
        gamma = int(rng.integers(0, 2))
        x = (omega ^ gamma).astype(np.float32)
        # Step 4: score > 0 means the samples look like omega, i.e. gamma.K = 0.
        ok_plain += ((0 if one_bit_crd(net, x, symmetric=False) > 0 else 1) == gamma)
        ok_sym += ((0 if one_bit_crd(net, x, symmetric=True) > 0 else 1) == gamma)
    return ok_plain / trials, ok_sym / trials


def one_bit_attack_matsui(appr, rng, n_plaintexts, trials):
    """Matsui's Algorithm 1 on the same number of plaintexts."""
    ok = 0
    for _ in range(trials):
        omega = data_gen.one_bit_omega(appr, rng, 1, n_plaintexts).reshape(-1)
        gamma = int(rng.integers(0, 2))
        x = omega ^ gamma
        t_count = int(np.count_nonzero(x == 0))
        # Pr[x = 0] = p_r if gamma.K = 0, else 1 - p_r.
        if appr.p > 0.5:
            guess = 0 if t_count > n_plaintexts / 2 else 1
        else:
            guess = 1 if t_count > n_plaintexts / 2 else 0
        ok += (guess == gamma)
    return ok / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200,
                    help="trials for the one-bit attacks")
    ap.add_argument("--multi-trials", type=int, default=40,
                    help="trials per point of the multi-bit N*t sweep")
    ap.add_argument("--t", type=int, default=DEFAULT_T)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    res = ensure_results()
    rng = np.random.default_rng(args.seed)

    # ---------------- 1. approximations + distinguishers ----------------
    print("=" * 78)
    print("1.  Linear approximations of TinyDES-24 and the distinguishers built on them")
    print("=" * 78)
    appr_rows = []
    for r in (3, 4, 5, 6):
        a = get_approximation(r)
        print(a.describe())
        appr_rows.append({
            "rounds": r, "a_in": a.a_in, "b_in": a.b_in, "a_out": a.a_out,
            "b_out": a.b_out, "p_theory": a.p_theory, "p_measured": a.p_measured,
            "bias": a.bias, "log2_bias": float(np.log2(a.bias)),
            "key_masks": " ".join(f"0x{k:05x}" for k in a.key_masks),
        })
        print()
    write_csv(os.path.join(res, "exp1_approximations.csv"), appr_rows)

    # ---------------- 2. one-bit key recovery ----------------
    print("=" * 78)
    print("2.  One-bit key recovery: ML-aided (Eq. 6 + CRD) vs Matsui's Algorithm 1")
    print("=" * 78)
    one_rows = []
    # The paper performs its one-bit attacks with N*t = (p_r - 1/2)^-2, so N is
    # chosen here to put N*t as close to bias^-2 as the chosen t allows.
    plan = {3: (8, 5), 4: (16, 160), 5: (32, 430), 6: (64, 1355)}
    for r, (t, n_samples) in plan.items():
        a = get_approximation(r)
        net, meta = train_one_bit(a, t, force=args.force_train, verbose=False)
        nt = n_samples * t
        ml, ml_sym = one_bit_attack_ml(net, a, rng, n_samples, t, args.trials)
        cl = one_bit_attack_matsui(a, rng, nt, args.trials)
        print(f"  L{r}: t = {t:4d}, N = {n_samples:4d}, N*t = {nt:6d} "
              f"(bias^-2 = {1 / a.bias ** 2:8.0f})   "
              f"ND acc {meta['val_acc'] * 100:5.2f}% "
              f"(ref {meta['bayes_acc'] * 100:5.2f}%)   "
              f"ML Eq.2 {ml * 100:5.1f}%   ML sym {ml_sym * 100:5.1f}%   "
              f"Matsui {cl * 100:5.1f}%")
        one_rows.append({
            "rounds": r, "t": t, "N": n_samples, "N_times_t": nt,
            "nd_accuracy": meta["val_acc"], "nd_bayes_ceiling": meta["bayes_acc"],
            "success_ml": ml, "success_ml_symmetric": ml_sym,
            "success_matsui": cl, "trials": args.trials,
        })
    write_csv(os.path.join(res, "exp1_one_bit.csv"), one_rows)

    # ---------------- 3. multi-bit key recovery ----------------
    print()
    print("=" * 78)
    print("3.  Multi-bit key recovery (Algorithm 2) on 8-round TinyDES-24, |GK| = 4096")
    print("=" * 78)
    appr, setup, net, meta = build(t=args.t, force_train=args.force_train, verbose=False)
    print(setup.describe())
    print(f"  ND_{appr.rounds}^{args.t} accuracy {meta['val_acc'] * 100:.3f}% "
          f"(fixed-p Bayes reference {meta['bayes_acc'] * 100:.3f}%)")
    print()
    print(f"{'N*t':>10}{'N':>7}{'ML-aided Alg.2':>17}{'Matsui Alg.2':>15}"
          f"{'mean rank':>12}{'time':>9}")
    print("-" * 70)
    multi_rows = []
    for nt in (524_288, 655_360, 786_432, 917_504, 1_048_576):
        ml_ok, cl_ok, ranks, secs = 0, 0, [], []
        for _ in range(args.multi_trials):
            data = setup.generate(rng, nt)
            sc = CandidateScorer(net, setup, data, args.t, cache=True)
            r = run_exhaustive(sc, setup, data)
            ml_ok += r.success
            ranks.append(r.rank_of_true())
            secs.append(r.seconds)
            cs = ClassicalScorer(setup, data)
            rc = run_exhaustive(cs, setup, data)
            cl_ok += rc.success
        n = args.multi_trials
        print(f"{nt:>10,}{nt // args.t:>7}{ml_ok / n * 100:>16.1f}%"
              f"{cl_ok / n * 100:>14.1f}%{np.mean(ranks):>12.1f}"
              f"{np.mean(secs):>8.2f}s")
        multi_rows.append({
            "N_times_t": nt, "N": nt // args.t, "t": args.t,
            "trials": n, "success_ml": ml_ok / n, "success_matsui": cl_ok / n,
            "mean_rank_of_true": float(np.mean(ranks)),
            "median_rank_of_true": float(np.median(ranks)),
            "seconds_per_attack": float(np.mean(secs)),
        })
    write_csv(os.path.join(res, "exp1_multi_bit.csv"), multi_rows)
    print(f"\nwrote results to {res}")


if __name__ == "__main__":
    main()
