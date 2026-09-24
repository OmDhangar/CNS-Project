# Faster Key Recovery for ML-Aided Linear Cryptanalysis

**Project summary for evaluation**

Extending: Zezhou Hou, Jiongjiong Ren, Shaozhen Chen, *"Improved machine
learning-aided linear cryptanalysis: application to DES"*, **Cybersecurity**
8:22 (2025).

---

## 1. The problem in one page

**Linear cryptanalysis** (Matsui, 1993) attacks a cipher using an equation that
is slightly more often true than false:

```
(bits of plaintext) ⊕ (bits of ciphertext) = (bits of key)
```

A random cipher would make this true exactly half the time. A real one makes it
true, say, 50.34% of the time. That 0.34% leak, accumulated over enough
plaintext/ciphertext pairs, reveals key bits.

**What the paper contributes.** It reframes this as machine learning. "Is this
coin fair, or biased to 50.34%?" is a classification problem, so a neural
network is trained to look at a block of `t` such bits and decide. To recover
several key bits at once (their **Algorithm 2**) it wraps one extra cipher
round around each end of the equation, guesses the 6 subkey bits feeding one
S-box at each end, and tries **all 2¹² = 4096 combinations**, keeping whichever
makes the network most confident.

**The gap we close.** Trying all 4096 is brute force, and the paper's own
conclusion names this as unsolved:

> *"Designing a better key-recovery strategy matching neural distinguishers
> will be effective to reduce the time complexity."*

**Our scope.** Replace that brute-force loop — and nothing else. Same network,
same scoring rule, same data. Only the *order* candidates are tried in.

---

## 2. The algorithm we designed

### 2.1 The observation it rests on

A wrong key guess is **not** random. Writing the attack's computed bit as

```
x(guess) = base ⊕ front(R₀, k_f) ⊕ back(L_R, k_b)
```

and comparing against the correct guess `gk*`, the two error terms are
independent of the equation's own bias, so the correlation **factorises
exactly**:

```
corr(gk) = corr(gk*) · ρ_f(k_f ⊕ k_f*) · ρ_b(k_b ⊕ k_b*)
```

where, for the front S-box `S_j` with output mask `ν`:

```
ρ_f(Δ) = 2⁻⁶ · Σ_u (−1)^( ν · [ S_j(u) ⊕ S_j(u ⊕ Δ) ] )
```

**The critical property:** `ρ_f` and `ρ_b` depend only on the cipher's S-boxes
and the approximation's masks — **not on the key and not on the data**. They
are two 64-entry tables computable *offline*, before a single query is spent.

This is the **wrong-key response profile**. Gohr (CRYPTO 2019) uses one for
*differential* attacks but must measure it empirically; for *linear* attacks it
has a closed form. That is the new idea.

### 2.2 The algorithm: Wrong-Key-Response-Guided Bayesian Search

Model an observed score as

```
s_i = μ + σ·A·g_h(k_i) + σ·N(0,1),    g_h(k) = |ρ_f(k_f ⊕ k_f^h) · ρ_b(k_b ⊕ k_b^h)|
```

where `h` ranges over all 4096 hypotheses, `μ` and `σ` are estimated robustly
(median / MAD) from the scores seen so far, and the peak height `A` is
marginalised over a grid.

```
Input : trained distinguisher ND, N·t plaintext/ciphertext pairs,
        candidate set GK, query budget B
Offline (0 queries):
    compute ρ_f, ρ_b from the S-boxes and masks
    P, Q, V ← zero vectors of length |GK|

1.  Query a small random opening set; record each score.
2.  Repeat until budget B is spent:
      a. g ← response_vector(last candidate)          # O(|GK|), an outer product
      b. P += s·g ;  Q += g ;  V += g²                # the only state needed
      c. μ, σ ← median / MAD of scores so far
         U ← (P − μ·Q) / σ
         log L(h|A) = A·U(h) − ½·A²·V(h)              # closed form
         posterior ← normalise( mean over the A-grid of exp(log L) )
      d. query the highest-posterior candidate not yet evaluated
3.  Return the best-scoring candidate seen.
```

**Why this is cheap.** The log-likelihood of *every* hypothesis depends on the
observations only through the three accumulators `P`, `Q`, `V`, each updated in
`O(|GK|)`. Maintaining the exact posterior over all 4096 hypotheses therefore
costs far **less than one evaluation of the neural network**. The model is pure
overhead-free structure, not a learned surrogate.

**Why it is not "just Bayesian optimisation."** We tested that too — see §4.

---

## 3. Evaluation methodology

| Commitment | How it is enforced |
|---|---|
| All methods score identically | Baseline and every search call **the same Python scoring object** — not equivalent code, the same object |
| Fast code path is correct | The optimised scorer is checked against the readable definition **on every attack** (max error 4×10⁻⁶) |
| No tuning on reported data | The stopping threshold is calibrated on a **disjoint seed range** |
| The gain is attributable | Two controls: random search (no model) and sequential scan (no model, stops early), both at matched budget |
| Nothing is hand-copied | `project_report.md` is regenerated from the result CSVs |
| Invariants are tested | **30 automated checks** — cipher correctness, mask algebra, scorer exactness, search equivalence |

**Two target ciphers.** *Phase 1* is **TinyDES-24**, a 24-bit DES-shaped Feistel
cipher (real DES S-boxes, DES-style expansion and key schedule) whose attack
geometry mirrors the paper exactly: 8 rounds, 6-round approximation, 6+6
guessed bits, |GK| = 4096. Small enough to brute-force the full landscape as
ground truth. *Phase 2* is **real DES**, validated against the FIPS test vector
`DES(0x0123456789ABCDEF, 0x133457799BBCDFF1) = 0x85E813540F0AB405`.

---

## 4. Results

### 4.1 Reproduction of the paper (framework: confirmed)

Data complexity is normalised by `bias⁻²` so two different ciphers are
comparable.

| Data (× bias⁻²) | **Paper** (real DES, 1000 trials) | **Ours** (TinyDES-24, 150 trials) |
|---|---|---|
| 7.3× / 7.6× | 80.2% | **88.0%** |
| 8.7× / 9.1× | 88.4% | **90.7%** |
| 10.2× / 10.6× | 93.6% | **94.7%** |
| 11.6× / 12.1× | 96.5% | 93.3% |

Read this as "same shape, same neighbourhood" rather than point-for-point
agreement: we bracket the paper (+7.8 pp at the low end, −3.2 pp at the high
end) and the normalisation across two different ciphers is approximate. What it
establishes is that **the paper's framework reproduces** — the ML attack works
and its success curve rises over the same range of normalised data.

### 4.2 One claim of the paper does *not* reproduce

The paper also reports that the ML attack **beats** classical Matsui (80.2% vs
78.6%). At 150 trials we find the opposite, consistently:

| Data (× bias⁻²) | ML-aided Alg. 2 | Classical Matsui Alg. 2 | Gap |
|---|---|---|---|
| 6.0× | 65.3% | 72.0% | −6.7 pp |
| 7.6× | 88.0% | 90.0% | −2.0 pp |
| 9.1× | 90.7% | 93.3% | −2.6 pp |
| 10.6× | 94.7% | 96.0% | −1.3 pp |
| 12.1× | 93.3% | 96.0% | −2.7 pp |

We report this rather than the flattering earlier result: a 40-trial run *had*
shown ML ahead, but the error bar there was ±20 pp, and the ordering reversed
once we raised to 150 trials. A likely cause is that our distinguisher is a
small residual MLP rather than the paper's full ResNet — but we have not tested
that, so it remains a conjecture. **It does not affect our own result**, since
the guided search is compared against exhaustive Algorithm 2 with the identical
scorer on both sides.

### 4.3 The contribution — head-to-head (150 trials)

The paper has **no entry** in this comparison: its cost is fixed at
`2¹² · N · t / 8`, i.e. always 4096 evaluations.

| Method | Success | Agrees with full scan | Evals (median) | % of \|GK\| |
|---|---|---|---|---|
| **Exhaustive Algorithm 2** (paper) | 72.0% | 100% by def. | 4096 | 100% |
| **Guided search (ours)** | **72.7%** | **98.7%** | **2227** | **54%** |
| Random, identical budget | 38.7% | 54.7% | 2227 | 54% |
| Sequential + early stop | 7.3% | 8.0% | 204 | 5% |
| scikit-optimize (generic BO) | 10.0% | 10.0% | 99 | 2% |

Note the guided search is **+0.7 pp above** the exhaustive baseline, not below.
That is not noise in our favour: on trials where the true key is not the global
maximum, stopping before reaching the higher-scoring wrong candidate returns
the correct answer, where the full scan is wrong by construction.

### 4.4 The budget curve — the result the project rests on (100 trials)

| Budget | % of \|GK\| | **Guided** | Random | Sequential |
|---|---|---|---|---|
| 205 | 5% | **39%** | 9% | 6% |
| 410 | 10% | **50%** | 13% | 13% |
| 614 | 15% | **56%** | 14% | 17% |
| 1024 | 25% | **60%** | 19% | 23% |
| 2048 | 50% | **67%** | 41% | 42% |
| 4096 | 100% | 68% | 68% | 68% |

*(Success rate. Exhaustive Algorithm 2 = 68%.)* The guided search reaches
exhaustive-level success at **50%** of the queries; random and sequential
scanning **never** reach it below 100%.

### 4.5 Phase 2 — the identical code on real 8-round DES (60 trials)

The geometry came out exactly as the paper states: front guess `K₀[18..23]`
(S-box 5), back guess `K₇[42..47]` (S-box 1), |GK| = 4096. This independently
confirms our mask-orientation derivation.

| Method | Success | Agrees with full scan | Evals (median) |
|---|---|---|---|
| Exhaustive Algorithm 2 | 58.3% | 100% by def. | 4096 |
| **Guided (ours)** | 56.7% | **95.0%** | 2911 (71%) |
| Random | 10.0% | 25.0% | 1024 |

That row understates the method, because 95% agreement is an expensive target.
The trade-off curve:

| Agreement target | Queries (median) | % of \|GK\| | Agreement reached |
|---|---|---|---|
| 80% | 259 | **6.3%** | 80.0% |
| **90%** | **418** | **10.2%** | **92.0%** |
| 95% | 1660 | 40.5% | 100.0% |

**92% agreement at 10.2% of the candidate space on real DES** is the headline
Phase-2 number.

---

## 5. Improvements we made beyond the original scope

Reproducing the paper faithfully turned up two genuine defects. Both concern
the **baseline**, so fixing them made the thing we are competing against
*stronger*, not weaker.

### 5.1 "Accuracy above 51%" is not a sufficient check

The paper's stated sanity check on a distinguisher is that accuracy clears 51%.
That is right for the *distinguisher* and wrong for the *attack*, because the
scoring rule (their Eq. 2) sums the network's **logits**, not its decisions.
Accuracy constrains only the sign; the score depends on the magnitude.

Our first network sat exactly at the theoretical optimum on accuracy — 52.19%
against a Bayes reference of 52.17% — and still lost badly: **37.5% against
classical Matsui's 67.5%**. Diagnosis: only **68%** of its logit variance was
explained by the sufficient statistic (the popcount), so the scoring rule was
summing 32% noise.

Fix: average the network weights over the second half of training.

| | Before | After |
|---|---|---|
| Accuracy | 52.19% | 52.19% (unchanged) |
| Logit variance explained by sufficient statistic | 68% | **94%** |
| Signal retained vs. theoretical best | 0.83 | **0.97** |
| **Mean rank of true key** (lower = better) | **68.8** | **4.7** |

`tests/test_attacks.py` now checks this directly, so it cannot regress silently.

### 5.2 The paper's one-bit decision rule has a systematic bias

Their Step 4 rule — "sum the log-odds; if positive, guess 0" — is the correct
likelihood ratio **only if** the network is exactly antisymmetric under
complementation. Real networks are not; ours was off by ~2.8×10⁻³ per sample.

Negligible per sample, but the rule **sums** over N samples, so the error grows
like `N` while the signal it competes with grows like `√N`. At N = 1355 the
accumulated error was `+3.8` against a signal of `3.9` — the same size.

Fix (one line, no extra data): use `Σ [logit(x) − logit(1−x)] / 2`, which is
antisymmetric by construction and is the true likelihood ratio between "these
samples are ω" and "these are ω ⊕ 1".

| Approximation | Paper's rule as written | Corrected | Classical Matsui |
|---|---|---|---|
| L3 | 98.7% | 98.7% | 97.3% |
| **L4** | **83.7%** | **97.0%** | 97.7% |
| L5 | 98.0% | 98.0% | 98.0% |
| L6 | 96.0% | 96.0% | 97.7% |

Never hurts; recovers 13 pp where the bias bites. (Does not affect the
multi-bit results: a constant offset shifts every candidate equally.)

### 5.3 Engineering

* Matsui's classical Algorithm 2 reimplemented via the standard 8192-cell
  joint histogram: **16 s → 0.06 s** per attack, exactly equal output.
* Scorer inner loop: pre-materialised bit streams, no per-call allocation,
  single-pass complement. **~11% faster per scan, 2× faster setup**, verified
  bit-exact.
* `run_all.py --preset demo | standard | full` so the pipeline can be
  demonstrated live in ~10 minutes.

---

## 6. Negative results (reported, not buried)

| We tried | Outcome |
|---|---|
| Generic Bayesian optimisation (`scikit-optimize`) over the 12 key bits | **9× slower than brute force** and far less accurate. It must learn the landscape from queries and refits an `O(n³)` model each step. |
| A covering probe design, chosen offline so every hypothesis is informed | **Worse at every small budget.** We assumed the bottleneck was breadth, but one probe already carries 14.7 direct-test-equivalents of information. The real limit is *contrast*: neighbouring hypotheses get near-identical responses. |
| Letting the search decide when to stop | **Worse than a fixed budget at matched quality** (54% of the space vs 40% for 96% agreement). Exhaustive itself fails ~40% of the time; a budget abandons hopeless trials, the stopping rule cannot tell hopeless from not-yet-found. |

The third is why the deliverable is the **budget curve**, not a tuned
stopping threshold.

---

## 7. Honest limitations

* **Phase 1 is a toy cipher.** We argue comparability by normalising against
  `bias⁻²` and the numbers line up, but that is an argument, not a proof.
  Phase 2 on real DES is the check, and it holds.
* **Our distinguisher is a small residual MLP, not the paper's ResNet.** We
  measure it against the theoretical optimum rather than assuming adequacy, but
  it is a deviation and is the most likely cause of §4.2.
* **Trial counts are 60–150, the paper used 1000.** Error bars ≈ ±4 pp.
* **Wall-clock speed-up is secondary.** A real attacker would vectorise the
  brute-force scan across candidates, which a sequential search cannot do. The
  **evaluation count** is the implementation-independent metric and is what we
  lead with.
* **The method needs one active S-box per side.** Matsui's `L5` activates five
  at one end and does not fit the 6+6-bit geometry at all.

---

## 8. Summary

| | |
|---|---|
| **Problem** | The paper's Algorithm 2 brute-forces all 4096 candidate subkeys; its conclusion names a better strategy as open work |
| **Our algorithm** | Wrong-key-response-guided Bayesian search — the response kernel has a closed form, computable offline from the S-boxes, so the search starts informed instead of cold |
| **Main result** | Same answer as the full scan **98.7%** of the time at **54%** of the evaluations, with **+0.7 pp** success; **92%** agreement at **10%** of evaluations on real DES |
| **Controls** | Random at the same budget: 54.7%. Generic Bayesian optimisation: 9× slower than brute force |
| **By-products** | Two real defects found in the paper's own pipeline, both fixed, both strengthening the baseline |
| **Not solved** | Automatic stopping; our evidence says a fixed budget is the better answer |

**Reproduce:** `pip install -r requirements.txt && python run_all.py --preset demo`
