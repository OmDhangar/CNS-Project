# Faster key recovery for ML-aided linear cryptanalysis

An extension of

> Zezhou Hou, Jiongjiong Ren, Shaozhen Chen.
> **"Improved machine learning-aided linear cryptanalysis: application to DES."**
> *Cybersecurity* 8:22 (2025). <https://doi.org/10.1186/s42400-024-00327-4>

---

## 1. What the paper does

Linear cryptanalysis (Matsui, 1993) recovers key bits from a linear approximate
expression `α·P ⊕ β·C = γ·K` that holds with probability `p_r ≠ ½`. The paper's
insight is to recast this as **distinguishing two Bernoulli distributions**: if
`γ·K = 1` then `α·P ⊕ β·C ~ Bern(p_r)`, otherwise `Bern(1 − p_r)`. A neural
network is trained to tell the two apart from a `t`-bit sample, and a
**combined-response distinguisher** (CRD, their Eq. 2) sums log-likelihood
ratios over many samples to make the final call. It is the first ML-aided
linear attack to match Matsui's classical success rate at equal or lower data
cost.

Their multi-bit framework (**Algorithm 2**) extends this to several subkey bits
by wrapping an `r`-round approximation in one extra round at each end and
**exhaustively testing every candidate subkey** `gk ∈ GK`. For their 8-round DES
attack that means running the trained distinguisher 2¹² = 4096 times per attack,
a stated time complexity of `2¹² × N × t / 8` DES evaluations.

## 2. What this project adds

The paper's conclusion names the gap explicitly:

> *"Designing a better key-recovery strategy matching neural distinguishers will
> be effective to reduce the time complexity."*

This project closes that gap. It **reproduces** the Bernoulli-distinguisher and
CRD scoring unchanged, and **replaces only the `for gk in GK` loop** with a
sequential, model-based search.

The key observation is that the CRD score surface over `GK` is not a black box.
Writing the attack's bit stream as

```
x(gk) = base ⊕ front(R₀, k_f) ⊕ back(L_R, k_b)
```

the correlation of a wrong guess factorises exactly:

```
corr(gk) = corr(gk*) · ρ_f(k_f ⊕ k_f*) · ρ_b(k_b ⊕ k_b*)
```

where `ρ_f` and `ρ_b` — the **wrong-key response profile** — depend only on the
cipher's S-boxes and the approximation's masks. They are **computed in closed
form, offline, before a single distinguisher evaluation is spent**. That turns
key search into Bayesian inference over 4096 hypotheses with a known response
kernel, which is the linear-cryptanalysis analogue of the empirical wrong-key
response profile Gohr (2019, §4.3) feeds to his differential-side key search.

The search maintains the exact posterior over all 4096 hypotheses at a cost far
below one distinguisher evaluation and queries the current
maximum-a-posteriori candidate. It is run to a **fixed budget**: an optional
early-stopping rule is implemented and calibrated, but we measured it to be
worse than simply choosing a budget at matched quality, so the budget curve is
what the results lead with (see `report/FACULTY_SUMMARY.md`, Sect. 6).

## 3. Target ciphers

**Phase 1 — TinyDES-24** (`src/ciphers/toy_feistel.py`), a DES-shaped Feistel
cipher chosen so the attack geometry mirrors the paper's 8-round DES attack
exactly:

| | paper (DES) | this project (TinyDES-24) |
|---|---|---|
| block / key | 64 / 56 bits | 24 / 24 bits |
| cipher rounds attacked | 8 | 8 |
| approximation | `L6`, 6 rounds | `L6`, 6 rounds (searched, then verified) |
| guessed subkey bits | `K₀[18..23]`, `K₇[42..47]` | `K₀` front S-box, `K₇` back S-box |
| candidate space \|GK\| | 2¹² = 4096 | 2¹² = 4096 |
| bias of the approximation | 2⁻⁷·⁸⁷ (measured here) | 2⁻⁸·²⁰ (measured over 2²⁴ samples) |

The small block means the full 4096-candidate landscape can be brute-forced as
ground truth and every trial re-run cheaply, which is what makes a ≥100-trial
comparison practical on a CPU.

**Phase 2 — real DES** (`src/ciphers/des.py`), validated against the standard
test vector `DES(0x0123456789ABCDEF, 0x133457799BBCDFF1) = 0x85E813540F0AB405`,
attacked with Matsui's own `L6` in exactly the paper's Fig. 5 geometry. Nothing
in `src/attacks` changes between the phases: `AttackSetup` takes the cipher
module as a parameter, so Phase 2 hands it `des` instead of `toy_feistel`.

Matsui's expressions are re-measured against this implementation rather than
taken on faith — `L3` gives p = 0.6958 against a published 0.6953, `L5` gives
0.5191 against 0.5191 — and the paper's unstated L/R orientation is resolved by
measurement. That resolution is self-validating: it independently reproduces
the paper's own statement that the 8-round attack guesses `K0[18..23]` (S-box 5)
and `K7[42..47]` (S-box 1).

## 4. Layout

```
src/
  ciphers/
    toy_feistel.py                TinyDES-24: vectorised encrypt/decrypt, mask algebra
    des.py                        real DES, reduced-round, same interface
  linear_analysis/
    bias_search.py                LAT + beam search over linear trails + Monte-Carlo verification
    approximations.py             cached, verified approximation table (the toy "Table 4")
    known_masks_des.py            Matsui's L3/L5/L6, orientation resolved by measurement
  distinguisher/
    data_gen.py                   the paper's Eq. 6 and Eq. 7 formats + the Bayes reference
    model.py                      ND_r^t: residual network, online training
    train.py                      training drivers with on-disk caching
    crd.py                        Eq. 2, the shared CandidateScorer, Matsui's classical scorer
  attacks/
    candidate_space.py            attack geometry + the shared partial transform (cipher-agnostic)
    wrong_key_profile.py          closed-form wrong-key response (the project's core idea)
    multi_bit_bruteforce.py       Algorithm 2, verbatim  <- the baseline
    multi_bit_guided.py           guided searches        <- the contribution
experiments/
  common.py                       shared plumbing + the stopping-rule calibration helpers
  exp1_reproduce_baseline.py      reproduce the paper's framework and fix the operating point
  calibrate_stopping.py           calibrate the stopping rule on an independent seed range
  exp2_guided_vs_bruteforce.py    headline head-to-head
  exp3_budget_sensitivity.py      success rate vs query budget (the headline figure)
  exp4_phase2_des_reduced_round.py  the same search, on real 8-round DES
  exp5_opening_design.py          does a covering opening design help? (it does not)
  make_plots.py                   render every figure from the CSVs
tests/
  test_cipher_and_masks.py        roundtrip, bijectivity, mask algebra, exact LAT
  test_attacks.py                 scorer exactness, eval counting, full-budget invariants,
                                  and that the offline profile predicts the real surface
results/                          CSVs + figures (auto-generated)
report/
  FACULTY_SUMMARY.md              evaluation-facing: the algorithm, the methodology,
                                  every result table, improvements and negative results
  approach_and_findings.md        plain-language walkthrough and an honest account
                                  of what we got wrong along the way
  project_report.md               the results write-up (assembled from the CSVs)
```

## 5. Reproducing

```bash
pip install -r requirements.txt
python run_all.py --preset demo       # ~10 min
python run_all.py --preset full       # ~3 h, the reported trial counts
```

or stage by stage:

```bash
python -m tests.test_cipher_and_masks              # cipher + mask invariants
python -m src.linear_analysis.approximations       # rebuild the approximation table
python -m tests.test_attacks                       # attack-layer invariants
python -m src.linear_analysis.known_masks_des      # re-verify Matsui's DES masks
python experiments/exp1_reproduce_baseline.py
python experiments/calibrate_stopping.py
python experiments/exp2_guided_vs_bruteforce.py --trials 100
python experiments/exp3_budget_sensitivity.py --trials 60
python experiments/exp4_phase2_des_reduced_round.py --config l6 --trials 60
python experiments/make_plots.py
python report/build_report.py
```

Trained distinguishers and the verified approximation table are cached in
`artifacts/`, so only the first run pays for them.

## 6. Methodological commitments

* The baseline and every guided search call the **same** `CandidateScorer`, so
  the only difference between them is the order of visits and the stopping
  point. Any measured difference is attributable to the search strategy alone.
* The readable definition of the partial transform (`transform_bits`) and the
  fast path used inside the scorer are checked against each other on every
  attack (`AttackSetup.self_test`).
* The stopping threshold is calibrated on a **separate seed range** from the
  trials the headline experiments report.
* Two controls separate the two possible sources of any saving:
  `sequential-earlystop` (right stopping rule, no model) and `random`
  (no model, matched budget). A generic Bayesian optimiser
  (`scikit-optimize`'s GP) is included as a third comparison.
* The trained distinguisher is reported against the **Bayes-optimal ceiling**
  for its Bernoulli problem, so "the network is good enough" is a measured
  claim rather than an assumption.

* `report/FACULTY_SUMMARY.md` -- the evaluation-facing document: the algorithm in
  pseudocode, the methodology commitments, all result tables against the paper's
  own numbers, the improvements we made to the paper's pipeline, and the
  negative results.
* `report/approach_and_findings.md` -- the readable overview: what the paper does,
  what we added, comparison tables against the paper's own numbers, and a section
  on the mistakes we made (two of which were real flaws in our first reproduction
  of the paper's own pipeline).
* `report/project_report.md` -- the results document, regenerated from `results/`
  so no number in it is hand-copied.
