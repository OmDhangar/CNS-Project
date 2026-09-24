"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from src.attacks.candidate_space import AttackSetup            # noqa: E402
from src.distinguisher.train import train_multi_bit            # noqa: E402
from src.linear_analysis.approximations import get_approximation  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------------------------
# The Phase-1 attack configuration.
#
# Chosen to mirror the paper's 8-round DES attack exactly in shape:
#   * an 8-round cipher attacked with a 6-round linear approximation,
#   * six guessed subkey bits at each end (paper: K0[18..23], K7[42..47]),
#   * |GK| = 2**12 = 4096 candidates,
#   * a data complexity in the same "few times bias^-2" regime, giving a
#     success rate in the same band as the paper's 78.6%-80.2%.
# ---------------------------------------------------------------------------
DEFAULT_ROUNDS = 6
DEFAULT_T = 256
DEFAULT_NT = 524_288


ARTIFACTS = os.path.join(ROOT, "artifacts")
CALIBRATION_PATH = os.path.join(ARTIFACTS, "stopping_calibration.json")


def ensure_results():
    os.makedirs(RESULTS, exist_ok=True)
    return RESULTS


def read_calibration():
    """The stopping rule chosen by experiments/calibrate_stopping.py."""
    import json

    if not os.path.exists(CALIBRATION_PATH):
        return {}
    with open(CALIBRATION_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def build(rounds=DEFAULT_ROUNDS, t=DEFAULT_T, force_train=False, verbose=True):
    """Approximation + attack geometry + trained distinguisher."""
    appr = get_approximation(rounds)
    setup = AttackSetup(appr)
    net, meta = train_multi_bit(setup, t=t, force=force_train, verbose=verbose)
    return appr, setup, net, meta


def write_csv(path, rows, fieldnames=None):
    import csv

    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Stopping-rule calibration, shared by calibrate_stopping.py and exp4.
#
# The rule is configuration-dependent -- it is a threshold on how concentrated
# the posterior has to get, and how fast the posterior concentrates depends on
# the bias of the approximation, on the data complexity and on the shape of the
# wrong-key response.  So every attack configuration calibrates its own, always
# on a seed range disjoint from the one the reported trials use.
# ---------------------------------------------------------------------------

POSTERIOR_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
                  0.95, 0.975, 0.99, 0.995, 0.999, 0.9999)


def record_trajectories(setup, net, t, nt, profile, trials, seed, verbose=True):
    """Run the guided search with stopping off, recording per-query diagnostics."""
    import numpy as _np

    from src.attacks.multi_bit_bruteforce import run_exhaustive
    from src.attacks.multi_bit_guided import WKRSearch
    from src.distinguisher.crd import CandidateScorer

    rng = _np.random.default_rng(seed)
    space = setup.n_candidates
    trajs, ex_success = [], 0
    for trial in range(trials):
        data = setup.generate(rng, nt)
        scorer = CandidateScorer(net, setup, data, t, cache=True, persistent_cache=True)
        r_ex = run_exhaustive(scorer, setup, data)
        ex_success += r_ex.success
        r = WKRSearch(setup, profile=profile, budget=space, stop_z=None,
                      stop_posterior=None, seed=trial
                      ).run(scorer, data, record_diagnostics=True)
        best = _np.asarray(r.extra["best_so_far"])
        trajs.append({
            "best_z": _np.asarray(r.extra["diag_best_z"]),
            "max_post": _np.asarray(r.extra["diag_max_posterior"]),
            "agrees": (best == r_ex.best_candidate).astype(int),
            "succeeds": (best == r.true_candidate).astype(int),
            "min_queries": WKRSearch(setup, profile=profile).min_queries,
        })
        if verbose and (trial + 1) % 5 == 0:
            print(f"    calibration trial {trial + 1}/{trials}")
    return trajs, ex_success / trials


def sweep_threshold(trajs, field, thresholds, space, name):
    import numpy as _np

    rows = []
    for th in thresholds:
        used, agree, succ = [], [], []
        for tr in trajs:
            idx = _np.flatnonzero(_np.asarray(tr[field]) >= th)
            idx = idx[idx + 1 >= tr["min_queries"]]
            k = int(idx[0]) if len(idx) else len(tr[field]) - 1
            used.append(k + 1)
            agree.append(tr["agrees"][k])
            succ.append(tr["succeeds"][k])
        rows.append({
            name: float(th),
            "queries_median": float(_np.median(used)),
            "queries_mean": float(_np.mean(used)),
            "queries_p95": float(_np.percentile(used, 95)),
            "queries_frac_median": float(_np.median(used) / space),
            "agreement": float(_np.mean(agree)),
            "success": float(_np.mean(succ)),
        })
    return rows


TARGET_AGREEMENT = 0.95


def select_threshold(rows, ex_rate, tolerance_pp, name="stop_posterior",
                     target_agreement=TARGET_AGREEMENT):
    """Cheapest threshold that still reproduces the exhaustive scan's answer.

    Selection is on **agreement**, not on success rate, and the reason is
    variance rather than preference.  Agreement is a *paired* statistic: on each
    trial it compares the search's answer with what the exhaustive scan returned
    on that same trial, with the same data and the same network.  Success
    compares two separately estimated rates, so the criterion "success within
    1 pp of exhaustive" differences two noisy numbers, and on 25 trials a single
    flipped trial moves the selected threshold from 0.90 to 0.99 -- which in
    turn moves the reported cost from 23% of the candidate space to 100%.  We
    measured exactly that swing between two seed sets before switching.

    Agreement is also the conservative choice: a rule that reproduces the
    scan's answer 95% of the time cannot lose much success rate relative to it.
    Success is still reported as the outcome, it is just not what the threshold
    is tuned on.
    """
    ok = [r for r in rows if r["agreement"] >= target_agreement]
    if not ok:                      # nothing meets the bar: take the strictest
        return max(rows, key=lambda r: r["agreement"])
    return min(ok, key=lambda r: r["queries_median"])


def print_sweep(rows, name):
    print(f"{name:>16}{'queries(med)':>14}{'% of |GK|':>11}"
          f"{'queries(p95)':>14}{'agreement':>11}{'success':>10}")
    print("-" * 76)
    for r in rows:
        print(f"{r[name]:>16.4f}{r['queries_median']:>14.0f}"
              f"{r['queries_frac_median'] * 100:>10.1f}%{r['queries_p95']:>14.0f}"
              f"{r['agreement'] * 100:>10.1f}%{r['success'] * 100:>9.1f}%")


def operating_points(rows, targets=(0.80, 0.90, 0.95, 0.99), name="stop_posterior"):
    """The cheapest threshold reaching each agreement target.

    A single tuned point is a misleading way to report this.  The cost of
    stopping is very sensitive to the agreement demanded -- on real DES the
    difference between "92% agreement" and "95% agreement" is 10% of the
    candidate space versus 40% -- so the honest presentation is the curve, with
    several targets marked, and the reader choosing the trade-off.
    """
    out = []
    for target in targets:
        ok = [r for r in rows if r["agreement"] >= target]
        if ok:
            best = min(ok, key=lambda r: r["queries_median"])
            out.append({"target_agreement": target, "reached": True, **best})
        else:
            out.append({"target_agreement": target, "reached": False,
                        **max(rows, key=lambda r: r["agreement"])})
    return out


def print_operating_points(points, name="stop_posterior"):
    print(f"{'target':>8}{name:>17}{'queries(med)':>14}{'% of |GK|':>11}"
          f"{'agreement':>11}{'success':>10}")
    print("-" * 71)
    for p in points:
        mark = "" if p["reached"] else "  (not reached)"
        print(f"{p['target_agreement'] * 100:>7.0f}%{p[name]:>17.4f}"
              f"{p['queries_median']:>14.0f}{p['queries_frac_median'] * 100:>10.1f}%"
              f"{p['agreement'] * 100:>10.1f}%{p['success'] * 100:>9.1f}%{mark}")


def announce_choice(chosen, ex_rate, tolerance_pp, name="stop_posterior"):
    print(f"\nchosen {name} = {chosen[name]:.4f}")
    print(f"  success   {chosen['success'] * 100:.1f}%  "
          f"(exhaustive {ex_rate * 100:.1f}%, tolerance {tolerance_pp:.0f} pp)")
    print(f"  agreement {chosen['agreement'] * 100:.1f}%")
    print(f"  queries   median {chosen['queries_median']:.0f} "
          f"({chosen['queries_frac_median'] * 100:.1f}% of |GK|), "
          f"p95 {chosen['queries_p95']:.0f}")


def summarise(name, results, space):
    """One row of the head-to-head table."""
    ok = np.array([r.success for r in results], dtype=float)
    ev = np.array([r.n_evals for r in results], dtype=float)
    sec = np.array([r.seconds for r in results], dtype=float)
    return {
        "method": name,
        "trials": len(results),
        "success_rate": float(ok.mean()),
        "success_ci95": float(1.96 * np.sqrt(ok.mean() * (1 - ok.mean()) / max(1, len(ok)))),
        "evals_median": float(np.median(ev)),
        "evals_mean": float(ev.mean()),
        "evals_p95": float(np.percentile(ev, 95)),
        "evals_frac_median": float(np.median(ev) / space),
        "seconds_mean": float(sec.mean()),
        "seconds_median": float(np.median(sec)),
    }


def print_table(rows):
    hdr = (f"{'method':<24}{'trials':>7}{'success':>10}{'evals(med)':>12}"
           f"{'% of |GK|':>11}{'evals(p95)':>12}{'time/attack':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<24}{r['trials']:>7}"
              f"{r['success_rate'] * 100:>9.1f}%"
              f"{r['evals_median']:>12.0f}"
              f"{r['evals_frac_median'] * 100:>10.1f}%"
              f"{r['evals_p95']:>12.0f}"
              f"{r['seconds_mean']:>12.2f}s")
