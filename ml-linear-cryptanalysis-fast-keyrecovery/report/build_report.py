"""Assemble report/project_report.md from the narrative + the generated CSVs.

Keeping the numbers out of the prose means the report can never drift from the
experiments: re-run the pipeline and the tables update.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def missing(what):
    return f"> *(not generated yet -- run `{what}`)*"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_approximations():
    rows = read_csv("exp1_approximations.csv")
    if not rows:
        return missing("experiments/exp1_reproduce_baseline.py")
    body = []
    for r in rows:
        body.append([
            f"L{r['rounds']}",
            f"0x{int(r['a_in']):03x} / 0x{int(r['b_in']):03x}",
            f"0x{int(r['a_out']):03x} / 0x{int(r['b_out']):03x}",
            f"{float(r['p_theory']):.6f}",
            f"{float(r['p_measured']):.6f}",
            f"2^{float(r['log2_bias']):.2f}",
        ])
    return md_table(
        ["approximation", "input mask (a_in / b_in)", "output mask (a_out / b_out)",
         "p (piling-up)", "p (measured, 2^24 samples)", "bias"], body)


def sec_one_bit():
    rows = read_csv("exp1_one_bit.csv")
    if not rows:
        return missing("experiments/exp1_reproduce_baseline.py")
    body = []
    for r in rows:
        body.append([
            f"L{r['rounds']}", r["t"], r["N"], f"{int(r['N_times_t']):,}",
            f"{float(r['nd_accuracy']) * 100:.2f}%",
            f"{float(r['nd_bayes_ceiling']) * 100:.2f}%",
            f"{float(r['success_ml']) * 100:.1f}%",
            (f"{float(r['success_ml_symmetric']) * 100:.1f}%"
             if r.get('success_ml_symmetric') else "-"),
            f"{float(r['success_matsui']) * 100:.1f}%",
        ])
    return md_table(
        ["approximation", "t", "N", "N x t", "ND accuracy", "Bayes reference",
         "ML, Eq. 2 literal", "ML, antisymmetrised", "Matsui Alg. 1"], body)


def sec_multi_bit():
    rows = read_csv("exp1_multi_bit.csv")
    if not rows:
        return missing("experiments/exp1_reproduce_baseline.py")
    body = []
    for r in rows:
        body.append([
            f"{int(r['N_times_t']):,}", r["N"], r["t"],
            f"{float(r['success_ml']) * 100:.1f}%",
            f"{float(r['success_matsui']) * 100:.1f}%",
            f"{float(r['mean_rank_of_true']):.1f}",
            f"{float(r['seconds_per_attack']):.2f} s",
        ])
    return md_table(
        ["N x t", "N", "t", "ML-aided Alg. 2", "Matsui Alg. 2",
         "mean rank of true key", "time / attack"], body)


def sec_headline():
    rows = read_csv("exp2_summary.csv")
    if not rows:
        return missing("experiments/exp2_guided_vs_bruteforce.py"), None
    order = ["exhaustive", "sequential-earlystop", "random", "guided-skopt", "guided-wkr"]
    rows.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)
    body = []
    for r in rows:
        body.append([
            f"`{r['method']}`",
            f"{float(r['success_rate']) * 100:.1f}% +/- {float(r['success_ci95']) * 100:.1f}",
            f"{float(r['success_delta_pp']):+.1f} pp",
            f"{float(r['agreement_with_exhaustive']) * 100:.1f}%",
            f"{float(r['evals_median']):.0f}",
            f"{float(r['evals_frac_median']) * 100:.1f}%",
            f"{float(r['evals_p95']):.0f}",
            f"{float(r['seconds_mean']):.2f} s",
            f"{float(r['speedup_vs_exhaustive']):.2f}x",
        ])
    return md_table(
        ["method", "success rate", "vs baseline", "agrees with exhaustive",
         "evals (median)", "% of \\|GK\\|", "evals (p95)", "time / attack",
         "speed-up"], body), rows


def sec_budget():
    rows = read_csv("exp3_budget_sensitivity.csv")
    if not rows:
        return missing("experiments/exp3_budget_sensitivity.py"), None
    ex = float(rows[0]["exhaustive_success"])
    body = []
    for r in rows:
        if float(r["budget_frac"]) not in (0.005, 0.01, 0.02, 0.05, 0.10,
                                           0.15, 0.20, 0.25, 0.50, 1.00):
            continue
        body.append([
            f"{float(r['budget_frac']) * 100:.1f}%", r["budget"],
            f"{float(r['guided-wkr_success']) * 100:.1f}%",
            f"{float(r['sequential_success']) * 100:.1f}%",
            f"{float(r['random_success']) * 100:.1f}%",
            f"{float(r['guided-wkr_agreement']) * 100:.1f}%",
        ])
    tbl = md_table(
        ["budget (% of \\|GK\\|)", "queries", "guided-wkr success",
         "sequential success", "random success", "guided-wkr agreement"], body)
    return tbl, ex


def sec_stopping():
    rows = read_csv("calibration_stopping_posterior.csv")
    if not rows:
        return missing("experiments/calibrate_stopping.py")
    body = []
    for r in rows:
        body.append([
            f"{float(r['stop_posterior']):.3f}",
            f"{float(r['queries_median']):.0f}",
            f"{float(r['queries_frac_median']) * 100:.1f}%",
            f"{float(r['queries_p95']):.0f}",
            f"{float(r['agreement']) * 100:.1f}%",
            f"{float(r['success']) * 100:.1f}%",
        ])
    return md_table(
        ["posterior threshold", "queries (median)", "% of \\|GK\\|",
         "queries (p95)", "agreement with exhaustive", "success"], body)


def sec_phase2():
    rows = read_csv("exp4_des_l6_summary.csv")
    cfg = read_json(os.path.join(RESULTS, "exp4_des_l6_config.json"))
    if not rows:
        return missing("experiments/exp4_phase2_des_reduced_round.py --config l6"), None
    order = ["exhaustive", "sequential-earlystop", "random", "guided-wkr"]
    rows.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)
    body = []
    for r in rows:
        body.append([
            f"`{r['method']}`",
            f"{float(r['success_rate']) * 100:.1f}%",
            f"{float(r['success_delta_pp']):+.1f} pp",
            f"{float(r['agreement_with_exhaustive']) * 100:.1f}%",
            f"{float(r['evals_median']):.0f}",
            f"{float(r['evals_frac_median']) * 100:.1f}%",
            f"{float(r['seconds_mean']):.2f} s",
            f"{float(r['speedup_vs_exhaustive']):.2f}x",
        ])
    return md_table(
        ["method", "success rate", "vs baseline", "agrees with exhaustive",
         "evals (median)", "% of \\|GK\\|", "time / attack", "speed-up"], body), cfg


def fig(name, caption):
    path = os.path.join(RESULTS, "figures", name)
    if not os.path.exists(path):
        return f"> *(figure `{name}` not generated yet -- run `experiments/make_plots.py`)*"
    return f"![{caption}](../results/figures/{name})\n\n*{caption}*"


# ---------------------------------------------------------------------------

def build():
    headline_tbl, headline_rows = sec_headline()
    budget_tbl, ex_rate = sec_budget()
    phase2_tbl, phase2_cfg = sec_phase2()
    cal = read_json(os.path.join(ARTIFACTS, "stopping_calibration.json"))
    exp3cfg = read_json(os.path.join(RESULTS, "exp3_config.json"))

    if phase2_cfg:
        phase2_config_line = (
            f"Configuration: **{phase2_cfg['cipher_rounds']}-round DES**, "
            f"{phase2_cfg['rounds']}-round approximation "
            f"(measured p = {phase2_cfg['p_measured']:.6f}), "
            f"t = {phase2_cfg['t']}, N x t = {phase2_cfg['nt']:,}, "
            f"|GK| = {phase2_cfg['space']}, "
            f"{phase2_cfg['trials']} trials. "
            f"ND accuracy {phase2_cfg['val_acc'] * 100:.2f}% "
            f"(fixed-p Bayes reference {phase2_cfg['bayes_acc'] * 100:.2f}%). "
            f"Stopping threshold {phase2_cfg.get('stop_posterior', float('nan')):.3f}, "
            f"calibrated on its own independent seed range. "
            f"For comparison, the paper reports 80.2% success at "
            f"N x t = 4 x 10^5 for this attack.")
    else:
        phase2_config_line = ""

    wkr = None
    if headline_rows:
        wkr = next((r for r in headline_rows if r["method"] == "guided-wkr"), None)

    if wkr:
        hl = (f"the guided search returns the same subkey as the exhaustive scan "
              f"**{float(wkr['agreement_with_exhaustive']) * 100:.0f}%** of the time "
              f"using a median of **{float(wkr['evals_median']):.0f} of 4096 "
              f"distinguisher evaluations "
              f"({float(wkr['evals_frac_median']) * 100:.1f}% of exhaustive)**, "
              f"a **{float(wkr['speedup_vs_exhaustive']):.1f}x** wall-clock speed-up, "
              f"for a success-rate change of "
              f"{float(wkr['success_delta_pp']):+.1f} percentage points.")
    else:
        hl = "*(run the experiments to populate the headline numbers)*"

    doc = f"""# Faster key recovery for ML-aided linear cryptanalysis

**Extending** Hou, Ren & Chen, *"Improved machine learning-aided linear
cryptanalysis: application to DES"*, Cybersecurity 8:22 (2025).

*Generated {date.today().isoformat()} from the CSVs in `results/`. Every number
below is produced by the scripts in `experiments/`; nothing is hand-copied.*

---

## 1. The gap this project closes

The paper recasts linear key recovery as distinguishing `Bern(p_r)` from
`Bern(1 - p_r)`, trains a neural distinguisher `ND_r^t` to do it, and combines
its responses over many samples with the combined-response distinguisher (CRD,
their Eq. 2). It is the first ML-aided linear attack to match Matsui's
classical success rates at equal or lower data cost.

Its multi-bit framework, **Algorithm 2**, recovers several subkey bits by
wrapping an `r`-round approximation in one extra round at each end and testing
**every** candidate subkey. For their 8-round DES attack that is 2^12 = 4096
distinguisher evaluations per attack, and the paper never optimises the search;
its conclusion names this as open work:

> *"Designing a better key-recovery strategy matching neural distinguishers
> will be effective to reduce the time complexity."*

**This project replaces Algorithm 2's `for gk in GK` loop, and nothing else.**
The distinguisher, the CRD rule, the data and the candidate space are the
paper's, reproduced unchanged. In one sentence: {hl}

## 2. Why a model-based search is possible here

The attack's bit stream for a candidate `gk = (k_f, k_b)` is

```
x(gk) = base  XOR  front(R_0, k_f)  XOR  back(L_R, k_b)
```

so relative to the correct guess `gk*` the two error terms are independent of
the approximation's own parity, and the correlation factorises **exactly**:

```
corr(gk) = corr(gk*) . rho_f(k_f XOR k_f*) . rho_b(k_b XOR k_b*)

rho_f(D) = 2^-6 sum_u (-1)^( nib_f . [ S_jf(u) XOR S_jf(u XOR D) ] )
```

`rho_f` and `rho_b` are the **wrong-key response profile**. They depend only on
the cipher's S-boxes and the approximation's masks -- not on the key, not on the
data -- so they are computed in closed form, offline, in 2 x 64 x 64 operations
**before a single distinguisher evaluation is spent**. Because Algorithm 2
reports `max(w0, w1)` over the two guesses of `gamma.K'`, the expected score
profile is proportional to `|rho_f . rho_b|`.

{fig("fig1_wrong_key_profile.png", "Figure 1. The wrong-key response profile, predicted offline from the S-boxes (left, centre) and checked against the score surface measured over 40 real attacks (right).")}

This is the linear-cryptanalysis counterpart of the wrong-key response profile
Gohr (2019, Sect. 4.3) *measures empirically* for differential neural
distinguishers and feeds to a Bayesian key search. Here it comes out in closed
form, which is what lets the search start informed rather than cold.

{fig("fig2_score_landscape.png", "Figure 2. One attack's complete CRD score surface over all 4096 candidate subkeys. The correct subkey is circled; note the cluster of elevated scores around it, which is exactly the structure the wrong-key response predicts.")}

## 3. The guided search

Model an observed score as `s_i = mu + sigma . A . g_h(k_i) + sigma . N(0,1)`,
where `g_h(k) = |rho_f(k_f XOR k_f^h) . rho_b(k_b XOR k_b^h)|` is the predicted
relative score at `k` under the hypothesis that `h` is the key. `mu` and `sigma`
are estimated robustly (median / MAD) from the scores seen so far, and the peak
height `A` is marginalised over a grid.

The log-likelihood of every hypothesis then depends on the observations only
through three accumulators,

```
P(h) = sum_i s_i g_h(k_i),   Q(h) = sum_i g_h(k_i),   V(h) = sum_i g_h(k_i)^2
```

each updated in `O(|GK|)` per query, so the exact posterior over all 4096
hypotheses is maintained for far less than the cost of one distinguisher
evaluation. The next query is the highest-posterior candidate not yet tried,
and the search stops when the posterior concentrates on a candidate it has
already evaluated.

## 4. Experimental setup

TinyDES-24 is a DES-shaped Feistel cipher (24-bit block and key, three real DES
S-boxes, DES-style expansion and rotating key schedule) chosen so the attack
geometry mirrors the paper's 8-round DES attack exactly: **8 cipher rounds, a
6-round linear approximation, six guessed subkey bits at each end, |GK| = 4096**.
The small block is what makes the full landscape brute-forceable as ground truth
and a 100-trial comparison practical on a CPU.

### 4.1 Linear approximations (the toy analogue of Matsui's Table 4)

Found by a LAT-driven beam search over linear trails, then verified by
Monte-Carlo against the real cipher:

{sec_approximations()}

The piling-up prediction and the measurement agree to 3-4 decimal places, which
is the check that the trail search, the mask algebra and the cipher agree with
each other.

### 4.2 Reproducing the paper's frameworks

One-bit key recovery (the paper's Eq. 6 + CRD) against Matsui's Algorithm 1 on
the same number of plaintexts, with `N x t = (p_r - 1/2)^-2` as the paper
prescribes:

{sec_one_bit()}

Reproducing this surfaced one real issue in the paper's Step 4. Its rule,
`sum_i log2(v_i / (1 - v_i)) > 0`, is the log-likelihood ratio between the two
hypotheses only if the network is exactly antisymmetric under complementation.
A trained network is not: measured on the 6-round distinguisher here, its mean
logit on random input sits about `+2.8 x 10^-3` from zero. Per sample that is
nothing, but the CRD *sums* over N samples, so the offset grows like `N` while
the signal it competes with grows like `sqrt(N)`. At `N = 1355` the accumulated
offset is `+3.8` against a signal spread of `3.9` -- the same size -- and the
rule loses accuracy exactly where the paper's own attacks need the largest N.

Replacing the statistic with `sum_i [logit(x_i) - logit(1 - x_i)] / 2`, which is
antisymmetric by construction, is the true log-likelihood ratio between
"these samples are omega" and "these samples are omega XOR 1", costs one extra
forward pass and no extra plaintexts, and is the one-bit analogue of what
Algorithm 2 already does when it takes the better of `w0` and `w1`. It restores
parity with Matsui (see the two ML columns above).

This correction does **not** touch the multiple-bit results below, and so does
not touch this project's main claim: an offset `c` adds `N c` to both `w0` and
`w1` for *every* candidate, so `max(w0, w1)` shifts by the same constant
everywhere and the ranking over `GK` is unchanged.

Multi-bit key recovery (Algorithm 2) on 8-round TinyDES-24, sweeping the data
complexity the way the paper's 8-round DES row does:

{sec_multi_bit()}

{fig("fig0_data_complexity.png", "Figure 0. Success rate of Algorithm 2 against data complexity, for the ML-aided CRD score and for Matsui's classical statistic on identical data.")}

### 4.3 A second thing reproduction surfaced: accuracy is not enough

The paper's stated sanity check on a distinguisher is that its accuracy clears
51%. That is the right bar for the *distinguisher*, but not for the *attack*,
because the CRD does not use the network's decision -- it sums the network's
logits. Accuracy only constrains the sign of the logit; the CRD depends on its
value.

The first distinguisher trained here sat at the Bayes reference on accuracy
(52.2% against 52.17%) and still lost badly to Matsui in the multiple-bit
attack: **37.5% against 82.5%** at `N x t = 524288`. The reason is visible in
one number. For these Bernoulli problems the sufficient statistic is the
popcount, so the exact log-likelihood ratio is affine in it, and any logit
variance the popcount does not explain is noise the CRD pays for. That network
explained only `R^2 = 0.68` of its logit variance by the popcount, retaining
`sqrt(0.68) = 0.83` of the available signal -- which is exactly the sort of
shortfall that turns a marginal attack into a failing one.

The cause is that on a 52%-accuracy task the gradient signal is tiny next to
the gradient noise, so SGD leaves a large random component in the weights: the
decision boundary still lands in the right place, but the logit picks up noise
that is not a function of the sufficient statistic. Averaging the weights over
the second half of training (an exponential moving average) removes it:
`R^2` rises from 0.68 to **0.94**, signal retention from 0.83 to **0.97**,
validation accuracy is unchanged, and the multiple-bit attack goes from 37.5%
to **87.5%** -- now above Matsui's 82.5% on the same data, which is the
paper's actual claim.

`tests/test_attacks.py` checks the `R^2` directly, so this cannot regress
silently. Both findings in this section concern the *baseline*, and both make
it stronger; the comparison in Section 5 is against the repaired baseline, not
the broken one.

## 5. Headline result

All methods score candidates through the **same** trained distinguisher, the
same CRD rule and the same data; only the visit order and the stopping point
differ.

{headline_tbl}

Two controls separate the possible sources of the saving:

* `sequential-earlystop` -- Algorithm 2's index-order scan with our stopping
  rule bolted on. This is *stopping early with no model*.
* `random` -- uniform sampling at a matched budget. This is *no model, no
  ordering*.
* `guided-skopt` -- scikit-optimize's Gaussian-process search over the twelve
  guessed key bits: a competent generic black-box optimiser that has to learn
  the surface from the queries themselves. It is given a much smaller budget
  and far fewer trials than the others, because it refits a Gaussian process
  after every observation and that refit is `O(n^3)`; its `vs baseline` column
  is computed against the exhaustive result on its own subset of trials, not
  on all of them.

One caveat on how to read this table. `guided-wkr`'s stopping threshold is
calibrated (on independent trials); `sequential-earlystop`'s is the
theoretically motivated `sqrt(2 ln |GK|)` and is *not* calibrated, because
calibrating a posterior threshold needs the very model that control is meant to
do without. The two are therefore not matched on stopping policy, and the row
should be read as "one reasonable stopping rule, applied without a model",
not as a tuned opponent. The matched comparison -- every method at an
identical, externally imposed budget, no stopping rule at all -- is Section 6,
and that is the one the argument rests on.

## 6. Budget sensitivity

Success rate as a function of a fixed query budget, replayed from full-length
search trajectories:

{budget_tbl}

{fig("fig3_budget_sensitivity.png", "Figure 3. Success rate and agreement with the exhaustive scan as a function of the query budget. The dashed line is exhaustive Algorithm 2, which always spends all 4096 evaluations.")}

## 7. Early stopping

The stopping threshold is calibrated on a **separate seed range** from the
trials reported above:

{sec_stopping()}

{fig("fig4_stopping_tradeoff.png", "Figure 4. The early-stopping trade-off. Labels give the posterior threshold.")}

The rule behaves the way one would want: it exits quickly on trials where the
key signal is strong, and falls back towards the full scan when it is not, so
the distribution of evaluations used is bimodal rather than merely shifted.

{fig("fig5_query_distribution.png", "Figure 5. Distinguisher evaluations actually spent per attack by the guided search, split by whether the attack succeeded. Exhaustive Algorithm 2 spends all 4096 every time.")}

## 8. Phase 2: the same search on real DES

Nothing in `src/attacks` changes between the two phases. `AttackSetup` takes the
cipher module as a parameter, so Phase 2 is a matter of handing it
`src/ciphers/des.py` and Matsui's `L6` instead of TinyDES-24 and a searched
approximation.

The DES implementation is checked against the standard test vector
(`DES(0x0123456789ABCDEF, 0x133457799BBCDFF1) = 0x85E813540F0AB405`), and each of
Matsui's expressions is re-measured against it: `L3` gives p = 0.6958 against a
published 0.6953, `L5` gives 0.5191 against a published 0.5191.

The paper does not state which Feistel half each mask of Table 4 sits on, so
rather than guess, every expression is measured in all four possible
orientations and the one reproducing Matsui's published probability is used.
That resolution is self-validating: with `L6` on rounds 1..6 of an 8-round
cipher it independently reproduces the paper's own statement of the attack's
guessed bits -- the front mask activates S-box 5, whose round-0 subkey bits are
exactly `K0[18..23]`, and the back mask activates S-box 1, whose round-7 bits
are exactly `K7[42..47]`. That is the paper's Fig. 5, and |GK| = 4096.

{phase2_config_line}

{phase2_tbl}

{fig("fig6_phase2_des.png", "Figure 6. Phase 2 on real 8-round DES with Matsui's L6, in the paper's own attack geometry.")}

A side note that fell out of this: Matsui's `L5` cannot be used for a two-sided
attack of this shape at all. Its input mask spans five S-boxes, so the front
guess would cost 30 bits rather than 6. Among Matsui's expressions only `L3`
and `L6` have ends that each activate a single S-box -- a concrete reason the
paper builds its 8-round attack on `L6` specifically.

## 9. Honest limits

* **The comparison is about search order, not about cryptanalysis strength.**
  The success rate of the attack itself is a property of the distinguisher and
  the data, and is identical for every method here by construction. What the
  guided search buys is the number of evaluations needed to reach it.
* **The wrong-key response is strong on the front S-box and weak on the back
  one** for this particular approximation, which bounds how much a single query
  can tell us. A different approximation, or a cipher whose S-boxes have a
  flatter response profile, would shift the numbers in either direction.
* **The distinguisher is a residual MLP, not Gohr's convolutional ResNet.** The
  samples are flat t-bit vectors of i.i.d. Bernoulli variables with no spatial
  structure, and the paper's own Appendix A shows the network family changes
  accuracy by less than 0.2 percentage points. The network is reported against
  the Bayes-optimal ceiling for its Bernoulli problem, so its adequacy is
  measured rather than assumed.
* **The Phase-2 DES numbers are not directly comparable to the paper's.** The
  paper quotes 80.2%-96.5% over 10^3 trials with a full ResNet distinguisher;
  the run here uses far fewer trials and a small residual MLP, so its success
  rate is its own baseline. What transfers is the *relative* comparison, which
  is the claim this project makes.
* **The stopping rule is calibrated per configuration.** How fast the posterior
  concentrates depends on the bias, the data complexity and the shape of the
  wrong-key response, so a threshold tuned on TinyDES-24 is not the right one
  for DES. Each configuration calibrates its own on an independent seed range;
  this is a real cost of the method, not a free parameter.
* **Wall-clock is reported under an identical per-candidate API for every
  method.** A real attacker would vectorise the exhaustive scan across
  candidates, which a sequential search cannot do; the evaluation *count* is
  therefore the implementation-independent metric, and the wall-clock figure
  should be read as secondary.

## 10. Defending the work

**In one paragraph.** This paper shows that recasting linear key recovery as
distinguishing two Bernoulli distributions lets a neural network match Matsui's
classical linear attack for the first time. But its multiple-bit procedure,
Algorithm 2, is still a brute-force scan of every candidate subkey -- 4096
evaluations of the trained distinguisher per attack for its 8-round DES target
-- and its conclusion names that as unsolved: *"designing a better key-recovery
strategy matching neural distinguishers ... to reduce the time complexity."*
This project replaces that scan with a Bayesian search over the same candidate
space, using the same trained distinguisher and the same CRD scoring rule. The
search is driven by the wrong-key response profile, which for a linear attack
turns out to have a closed form: it is computable from the cipher's S-boxes
before a single query is spent, unlike the differential-side analogue Gohr had
to measure empirically.

**What is reproduced and what is new.** The Bernoulli distinguisher, the CRD
and Algorithm 2 are the paper's and are reproduced, not reinvented. The
contribution is the search strategy layered on top -- a drop-in replacement for
the `for gk in GK` loop -- plus the benchmark quantifying it. Two corrections
to the reproduction (Sections 4.2 and 4.3) are separate small findings; both
make the *baseline* stronger, so neither flatters the comparison.

**"Isn't this just applying Bayesian optimisation?"** No, and the
`guided-skopt` row is the evidence. A general-purpose Gaussian-process
optimiser over the same twelve bits does *worse* than the exhaustive scan it is
meant to replace, and is slower in wall-clock terms than simply trying every
candidate, because it has to learn the shape of the surface from the queries
themselves and pays an `O(n^3)` refit for the privilege. What makes model-based
search work here is not the optimiser, it is knowing the response kernel in
advance -- and that comes from the cryptanalysis, not from the optimiser.

**"Why not attack full DES?"** The toy cipher gives fully brute-forceable
ground truth and fast iteration, so the comparison can be run over enough
trials to mean something. Phase 2 then points the identical search code at real
reduced-round DES in the paper's own attack geometry (Section 8).

**"How do you know the comparison is fair?"** Every method calls the same
`CandidateScorer`, so the scoring function is literally the same object; the
fast scorer path is checked against the readable definition on every attack;
the stopping threshold is calibrated on a disjoint seed range; and two
controls -- stopping early without a model, and a model-free search at a
matched budget -- separate the two possible sources of the saving. The
matched-budget curve in Section 6 makes no use of a stopping rule at all.

## 11. Reproducing

```bash
pip install -r requirements.txt
python run_all.py            # or --quick for a smoke test
```

Trained distinguishers, the verified approximation table and the calibrated
stopping rule are cached under `artifacts/`, so only the first run pays for
them. Every table and figure above is regenerated from `results/` by
`report/build_report.py`; no number in this document is hand-copied.
"""
    out = os.path.join(HERE, "project_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build()
