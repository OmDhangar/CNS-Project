"""Self-tests for the attack layer.

Run with:  python -m tests.test_attacks

These are the invariants that make the head-to-head comparison meaningful:

* the correct candidate reproduces the approximation's own bias, and wrong
  candidates do not;
* the fast scorer path is bit-identical to the readable definition;
* a full-budget guided search with stopping disabled agrees with exhaustive
  Algorithm 2 on *every* trial -- if it ever disagreed, the search would be
  losing candidates rather than merely reordering them;
* the evaluation counter counts distinct candidates per run and is unaffected
  by the memo cache;
* the wrong-key response really is key-independent, and really does predict
  the shape of the measured score surface.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from src.attacks.candidate_space import AttackSetup                    # noqa: E402
from src.attacks.multi_bit_bruteforce import run_exhaustive            # noqa: E402
from src.attacks.multi_bit_guided import (RandomSearch, SequentialSearch,  # noqa: E402
                                          WKRSearch)
from src.attacks.wrong_key_profile import (WrongKeyProfile,            # noqa: E402
                                           check_key_independence)
from src.distinguisher.crd import CandidateScorer                      # noqa: E402
from src.distinguisher.train import train_multi_bit                    # noqa: E402
from src.linear_analysis.approximations import get_approximation       # noqa: E402

FAILURES = []
T = 256
NT = 131_072          # small: these are correctness tests, not measurements


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' :: ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


def setup_and_net():
    appr = get_approximation(6)
    setup = AttackSetup(appr)
    net, _ = train_multi_bit(setup, t=T, verbose=False)
    return appr, setup, net


def test_correct_candidate_bias(setup, appr):
    rng = np.random.default_rng(3)
    hits = tot = 0
    wrong_hits = wrong_tot = 0
    for _ in range(20):
        d = setup.generate(rng, 200_000)
        w = setup.omega_bits(d)
        hits += int(np.count_nonzero(w == 0))
        tot += w.size
        for c in rng.integers(0, setup.n_candidates, size=3):
            c = int(c)
            if c == d.true_candidate:
                continue
            ww = setup.transform_bits(d, c) ^ np.uint8(d.gamma_key_bit)
            wrong_hits += int(np.count_nonzero(ww == 0))
            wrong_tot += ww.size
    p_right, p_wrong = hits / tot, wrong_hits / wrong_tot
    check("correct candidate reproduces the approximation bias",
          abs(p_right - appr.p) < 2e-3,
          f"measured {p_right:.5f} vs approximation {appr.p:.5f}")
    check("wrong candidates are unbiased",
          abs(p_wrong - 0.5) < 1e-3, f"measured {p_wrong:.5f}")


def test_fast_path(setup, net):
    rng = np.random.default_rng(4)
    d = setup.generate(rng, NT)
    sc = CandidateScorer(net, setup, d, T, cache=False)
    ok = True
    worst = 0.0
    for c in (0, 1, 777, 2048, 4095):
        bits = setup.transform_bits(d, c)[: (NT // T) * T].reshape(-1, T).astype(np.float32)
        with torch.no_grad():
            l0 = net.logit(bits).numpy()
            l1 = net.logit(1 - bits).numpy()
        ref = max(l0.sum(), l1.sum()) / math.log(2)
        worst = max(worst, abs(sc(c).score - ref))
        ok &= abs(sc(c).score - ref) < 1e-3
    check("scorer fast path matches the from-scratch definition", ok,
          f"max |difference| = {worst:.2e}")


def test_eval_counter(setup, net):
    rng = np.random.default_rng(5)
    d = setup.generate(rng, NT)
    sc = CandidateScorer(net, setup, d, T, cache=True, persistent_cache=True)
    sc.reset()
    for c in [7, 7, 7, 9]:
        sc(c)
    check("n_evals counts distinct candidates, not calls",
          sc.n_evals == 2 and sc.n_calls == 4, f"evals={sc.n_evals} calls={sc.n_calls}")
    sc.reset()
    sc(7)
    check("a persistent cache does not suppress the next run's eval count",
          sc.n_evals == 1, f"evals={sc.n_evals}")


def test_full_budget_agrees(setup, net):
    """With the full budget and no stopping, every search must find the argmax."""
    rng = np.random.default_rng(6)
    profile = WrongKeyProfile(setup)
    space = setup.n_candidates
    ok_w = ok_r = ok_s = True
    evals_ok = True
    for _ in range(3):
        d = setup.generate(rng, NT)
        sc = CandidateScorer(net, setup, d, T, cache=True, persistent_cache=True)
        ex = run_exhaustive(sc, setup, d)
        w = WKRSearch(setup, profile=profile, budget=space, stop_z=None,
                      stop_posterior=None, seed=0).run(sc, d)
        r = RandomSearch(setup, budget=space, seed=0).run(sc, d)
        s = SequentialSearch(setup, budget=space, stop_z=None).run(sc, d)
        ok_w &= w.best_candidate == ex.best_candidate
        ok_r &= r.best_candidate == ex.best_candidate
        ok_s &= s.best_candidate == ex.best_candidate
        evals_ok &= (ex.n_evals == space and w.n_evals == space
                     and r.n_evals == space and s.n_evals == space)
    check("full-budget guided search reproduces the exhaustive argmax", ok_w)
    check("full-budget random search reproduces the exhaustive argmax", ok_r)
    check("full-budget sequential scan reproduces the exhaustive argmax", ok_s)
    check("all full-budget searches use exactly |GK| evaluations", evals_ok)


def test_budget_is_respected(setup, net):
    rng = np.random.default_rng(7)
    profile = WrongKeyProfile(setup)
    d = setup.generate(rng, NT)
    sc = CandidateScorer(net, setup, d, T, cache=True)
    ok = True
    for b in (40, 137, 500):
        w = WKRSearch(setup, profile=profile, budget=b, stop_z=None,
                      stop_posterior=None, seed=1).run(sc, d)
        ok &= w.n_evals == b and len(set(w.visited)) == b
    check("the guided search never exceeds its budget and never repeats", ok)


def test_wrong_key_profile(setup, net):
    check("rho_front is independent of the key", check_key_independence(setup.front_lut))
    check("rho_back is independent of the key", check_key_independence(setup.back_lut))

    profile = WrongKeyProfile(setup)
    check("rho(0) = 1 on both sides",
          profile.rho_front[0] == 1.0 and profile.rho_back[0] == 1.0)

    # The response vector must equal a direct recomputation from the LUTs.
    kf, kb = 13, 41
    cand = setup.join(kf, kb)
    g = profile.response_vector(cand)
    h = setup.join(20, 5)
    hf, hb = setup.split(h)
    direct = abs(profile.rho_front[hf ^ kf] * profile.rho_back[hb ^ kb])
    check("response_vector indexes the right hypothesis",
          abs(g[h] - direct) < 1e-12, f"{g[h]:.6f} vs {direct:.6f}")

    # And it must predict the shape of the real score surface.
    #
    # Note which statistic is checked.  A Pearson correlation over all 4096
    # cells is not it: the profile predicts exactly zero for 1699 of them and
    # below 0.1 for 2153 more, so that number is dominated by the noise in
    # cells the search never relies on, and it moves around a lot with the
    # trial count.  What the search actually uses is (a) that measured score
    # rises monotonically with the predicted response, and (b) that the
    # prediction is accurate on the strongly-coupled cells.  Both are checked.
    rng = np.random.default_rng(8)
    acc = np.zeros(setup.n_candidates)
    n = 20
    for _ in range(n):
        d = setup.generate(rng, 262_144)
        sc = CandidateScorer(net, setup, d, T, cache=False)
        s, _ = sc.full_landscape()
        mu = np.median(s)
        sigma = 1.4826 * np.median(np.abs(s - mu))
        z = (s - mu) / sigma
        # re-index by XOR difference from the true key
        tf_, tb_ = setup.split(d.true_candidate)
        idx = np.arange(setup.n_candidates)
        df, db = (idx & 63) ^ tf_, (idx >> 6) ^ tb_
        acc += z[(db << 6) | df]
    acc /= n
    pred = profile.response_vector(0)

    strong = pred >= 0.25
    r_strong = float(np.corrcoef(acc[strong], pred[strong])[0, 1])
    check("the offline profile predicts the score of the cells the search uses",
          r_strong > 0.8,
          f"Pearson r = {r_strong:.3f} on the {int(strong.sum())} cells with "
          f"predicted response >= 0.25, over {n} attacks")

    edges = [0.0, 0.1, 0.2, 0.3, 0.45, 1.01]
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pred >= lo) & (pred < hi)
        if sel.sum():
            means.append(float(acc[sel].mean()))
    monotone = all(b > a for a, b in zip(means, means[1:]))
    check("measured score rises monotonically with the predicted response",
          monotone, " < ".join(f"{m:+.2f}" for m in means))

    check("the predicted peak is at zero difference",
          int(np.argmax(acc)) == 0,
          f"argmax at candidate difference {int(np.argmax(acc))}, "
          f"z = {acc.max():.2f}")


def test_logit_calibration(setup, net):
    """The CRD sums logits, so the logit -- not just its sign -- must be good.

    For these Bernoulli problems the sufficient statistic is the popcount, so
    the exact log-likelihood ratio is affine in it.  Any logit variance that
    the popcount does not explain is noise the CRD pays for, and the fraction
    explained is (the square of) the signal-to-noise ratio the attack retains
    relative to Matsui's count.  A network can sit at the Bayes accuracy and
    still fail this, which is why it is checked separately.
    """
    rng = np.random.default_rng(9)
    x = rng.integers(0, 2, size=(40000, setup_dim(net))).astype(np.float32)
    with torch.no_grad():
        lg = net.logit(x).numpy()
    k = x.sum(1)
    coef = np.polyfit(k, lg, 1)
    resid = lg - np.polyval(coef, k)
    r2 = 1.0 - resid.var() / lg.var()
    check("the distinguisher's logit is close to affine in the popcount",
          r2 > 0.85, f"R^2 = {r2:.4f}, SNR retained {np.sqrt(max(r2, 0)):.3f}")


def setup_dim(net):
    return net.dim


def main():
    appr, setup, net = setup_and_net()
    print(f"configuration: {setup.cipher_name}, {setup.cipher_rounds} rounds, "
          f"|GK| = {setup.n_candidates}, t = {T}, N*t = {NT}\n")
    test_correct_candidate_bias(setup, appr)
    test_logit_calibration(setup, net)
    test_fast_path(setup, net)
    test_eval_counter(setup, net)
    test_budget_is_respected(setup, net)
    test_full_budget_agrees(setup, net)
    test_wrong_key_profile(setup, net)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
