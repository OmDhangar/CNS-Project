# How we got here: approach, results, and the things we got wrong

*A plain-language companion to `project_report.md`. That document is generated
from the result CSVs and holds the final tables; this one explains the
reasoning, and is honest about the wrong turns, because several of them are
more instructive than the result.*

**Status:** complete. Every stage has run to completion at the trial counts
stated beside each table — 150 trials for the reproduction sweep, 150 for the
head-to-head, 100 for the budget curve, 60 for the real-DES run. Where an
earlier draft of this document quoted a smaller run and got a different answer,
that is called out explicitly rather than quietly overwritten.

---

## 1. The problem, in plain language

Linear cryptanalysis breaks a cipher by finding an equation that is *slightly*
more often true than false:

```
(some bits of the plaintext) XOR (some bits of the ciphertext) = (some bits of the key)
```

If that equation held exactly half the time, it would tell you nothing. The
attack works because it holds, say, 50.34% of the time instead of 50%. Collect
enough plaintext/ciphertext pairs, count how often the left side is zero, and
the small imbalance leaks a key bit.

**What the paper adds.** Hou, Ren & Chen noticed this can be restated as a
machine-learning problem. "Is this coin fair, or is it biased to 50.34%?" is a
classification task. So they train a neural network to look at a block of `t`
such bits and say which coin produced it. To recover *several* key bits at
once, they wrap an extra cipher round around each end of the equation, guess the
6 key bits feeding one S-box at each end, and try **every one of the 4096
possible guesses**, keeping whichever guess makes the network most confident.
That is their Algorithm 2.

**The gap.** Trying all 4096 guesses is brute force. The paper says so itself,
in its conclusion:

> *"Designing a better key-recovery strategy matching neural distinguishers will
> be effective to reduce the time complexity."*

**Our job.** Replace that brute-force loop — and nothing else. Same network,
same scoring rule, same data. Only the *order* in which guesses are tried, and
when we stop trying.

---

## 2. The idea that made it work

A search can only be smarter than brute force if the thing it is searching has
structure. So the first question was: does a *wrong* key guess look completely
random, or does it leak something?

It leaks. And the leak has an exact formula.

When you make a guess, the bit you compute is:

```
x(guess) = (something you know) XOR (front S-box term) XOR (back S-box term)
```

Compare that against the correct guess, and the two error terms are independent
of the equation's own bias. The consequence is that the strength of a wrong
guess **multiplies out**:

```
strength(guess) = strength(correct guess)  x  rho_f(front error)  x  rho_b(back error)
```

where `rho_f` and `rho_b` measure "how much does this S-box still behave the
same way if I get its key bits wrong by this much?"

The important part: **`rho_f` and `rho_b` depend only on the cipher's S-boxes
and the equation's masks.** Not on the key. Not on the data. They are 64x64
tables you can compute on paper, before running a single attack.

So the search is not a black box. We know the shape of the landscape in
advance; we just don't know where its peak is. That turns key search into
Bayesian inference over 4096 hypotheses with a known response kernel — cheap
enough that maintaining the exact posterior over all 4096 costs far less than
one evaluation of the neural network.

**Why this is not "just applying Bayesian optimisation."** We tested that too
(`scikit-optimize`, a standard Gaussian-process optimiser, on the same 12 bits).
It performs *worse than the brute-force scan it is meant to replace*, and is
slower in wall-clock time, because it has to learn the landscape's shape from
the queries themselves and pays an expensive model refit for every observation.
Knowing the response kernel in advance is what makes the method work, and that
knowledge comes from the cryptanalysis, not from the optimiser.

**Prior art we are echoing.** Gohr (2019) uses a "wrong-key response profile"
for differential attacks — but he has to *measure* his empirically. For linear
attacks it turns out to have a closed form. That is the small new idea here.

---

## 3. What we built

| Piece | What it is |
|---|---|
| **TinyDES-24** | A miniature DES (24-bit block) with the same anatomy: Feistel rounds, expansion, real DES S-boxes, rotating key schedule. Small enough to brute-force the whole 4096-candidate landscape as ground truth, so we can check answers. |
| **Trail search** | Finds the best linear equations for TinyDES-24 automatically, the way Matsui found his for DES. |
| **The paper's pipeline** | Their two data formats, their network, their scoring rule, and Algorithm 2 verbatim. This is the **baseline we are trying to beat**, so it had to be right. |
| **The guided search** | Our contribution: the drop-in replacement for the brute-force loop. |
| **Real DES** | A full DES implementation, checked against the official test vector, so the identical search code can be pointed at the paper's actual target. |
| **Tests** | 30 automated checks (cipher correctness, mask algebra, scorer exactness, search invariants). |

**Design rule we held to throughout:** the brute-force baseline and the guided
search call *the same scoring object*. Not "equivalent code" — literally the
same Python object. So any difference between them is caused by search order
alone, and cannot be an artefact of one being implemented better than the other.

---

## 4. Results

### 4.1 Did we faithfully reproduce the paper? The framework yes, one claim no.

Success rates on two different ciphers cannot be compared directly, because the
attack's difficulty depends on how weak the equation is. The standard way to
make them comparable is to measure data in units of `bias^-2` — the natural
scale of the problem.

| Data (in units of bias⁻²) | **Paper** (real DES, 1000 trials) | **Us** (TinyDES-24, 150 trials) |
|---|---|---|
| 6.0x | — | 65.3% |
| **7.3x / 7.6x** | **80.2%** | **88.0%** |
| **8.7x / 9.1x** | **88.4%** | **90.7%** |
| **10.2x / 10.6x** | **93.6%** | **94.7%** |
| 11.6x / 12.1x | 96.5% | 93.3% |

Read this as "same shape, same neighbourhood" rather than point-for-point
agreement: we bracket the paper (+7.8 pp at the low end, −3.2 pp at the high
end), and normalising across two different ciphers is approximate. What it
establishes is that the reproduction of the framework is sound.

**Their second claim does *not* reproduce, and we should say so.** The paper
also argues that the ML attack slightly *beats* Matsui's classical method on
the same data (80.2% against 78.6%). At 150 trials per point we find the
opposite, consistently:

| Data | ML-aided Algorithm 2 | Classical Matsui Algorithm 2 | Gap |
|---|---|---|---|
| 524,288 (6.0x) | 65.3% | 72.0% | −6.7 pp |
| 655,360 (7.6x) | 88.0% | 90.0% | −2.0 pp |
| 786,432 (9.1x) | 90.7% | 93.3% | −2.6 pp |
| 917,504 (10.6x) | 94.7% | 96.0% | −1.3 pp |
| 1,048,576 (12.1x) | 93.3% | 96.0% | −2.7 pp |

*(150 trials per row, about ±4 pp.)* The ML-aided attack is **1–7 pp below**
classical at every data point, with the gap narrowing as data grows.

An earlier draft of this document reported the opposite, because a 40-trial run
showed ML ahead at one point (92.5% vs 90.0%). That was noise: at 40 trials the
error bar is about ±20 pp near the steep part of the curve, and the same
configuration measured twice gave 87.5% and 50.0%. Raising to 150 trials made
the ordering stable and it went the other way. We are reporting the 150-trial
result.

A plausible cause is that our distinguisher is a small residual MLP rather than
the paper's full ResNet — but we have not tested that, so it stays a
conjecture rather than an explanation. What we can say is that **this does not
touch the project's own result**: the guided search is compared against
exhaustive Algorithm 2, and both sides of that comparison use the identical
scorer, so a distinguisher that is a little weak weakens both equally.

### 4.2 The optimisation: what we actually gained

The paper has **no entry in this table**, because it never optimised the search.
Its cost is fixed at `2^12 x N x t / 8` — always 4096 network evaluations. So
the comparison is against its own exhaustive scan, with the same network, the
same scoring rule and the same data on both sides.

| Metric | Paper's Algorithm 2 | **Our guided search** | Difference |
|---|---|---|---|
| Network evaluations per attack | **4096, every time** | **2227** (fixed budget) | **1.8x fewer** |
| Returns the same key as the full scan | 100% by definition | **98.7%** | −1.3 pp |
| Success rate | 72.0% | **72.7%** | **+0.7 pp** |

*(150 trials.)* The small success *gain* is not noise in our favour and is worth
understanding: on trials where the true key is not the global maximum, a search
that stops before reaching the higher-scoring wrong candidate still returns the
right answer, where the full scan is wrong by construction.

**The budget curve is the real result** (100 trials), because it lets the
reader pick the trade-off rather than accepting one tuned point:

| Query budget | % of 4096 | **Guided** | Random | Sequential |
|---|---|---|---|---|
| 205 | 5% | **39%** | 9% | 6% |
| 410 | 10% | **50%** | 13% | 13% |
| 614 | 15% | **56%** | 14% | 17% |
| 1024 | 25% | **60%** | 19% | 23% |
| 2048 | 50% | **67%** | 41% | 42% |
| 4096 | 100% | 68% | 68% | 68% |

*(Success rate; exhaustive Algorithm 2 = 68% at 100%.)* The guided search
reaches exhaustive-level success at **50%** of the queries. Random and
sequential scanning **never** reach it below 100%.

### 4.3 Is the gain real, or an artefact?

Everything in this table used the same network, the same data and the same
scoring object; only the visit order differs.

| Method | Success | Agrees with full scan | Evals (median) | What it tests |
|---|---|---|---|---|
| Exhaustive Algorithm 2 | 72.0% | 100% by def. | 4096 | the baseline |
| **Guided (ours)** | **72.7%** | **98.7%** | 2227 | — |
| Random, identical budget | 38.7% | 54.7% | 2227 | is the gain just "fewer queries"? **No** |
| Sequential + early stop | 7.3% | 8.0% | 204 | is the gain just "stopping early"? **No** |
| scikit-optimize (generic BO) | 10.0% | 10.0% | 99 | is the gain just "using an optimiser"? **No** |

The `scikit-optimize` row deserves a second look: it is not merely less
accurate, it runs at **0.11x** the speed of brute force — that is, a standard
Gaussian-process optimiser is roughly **nine times slower than simply trying
all 4096 candidates**, because it refits its model after every observation.
Knowing the response kernel in advance is what makes model-based search viable
here, and that knowledge comes from the cryptanalysis, not from the optimiser.

### 4.4 Phase 2: the same code on real DES

The identical search, pointed at real 8-round DES with Matsui's `L6`. The
geometry came out exactly as the paper states it — front guess `K0[18..23]`
(S-box 5), back guess `K7[42..47]` (S-box 1), |GK| = 4096 — which is an
independent confirmation that our mask-orientation reasoning was right.

| Method | Success | Agrees with full scan | Evals (median) |
|---|---|---|---|
| Exhaustive Algorithm 2 | 58.3% | 100% by def. | 4096 |
| **Guided (ours)** | 56.7% (−1.7 pp) | **95.0%** | 2911 (71%) |
| Random | 10.0% | 25.0% | 1024 |

*(60 trials.)* That single row understates it, because the 95% agreement target
is expensive. The trade-off curve on DES:

| Agreement target | Queries (median) | % of \|GK\| | Agreement reached |
|---|---|---|---|
| 80% | 259 | **6.3%** | 80.0% |
| **90%** | **418** | **10.2%** | **92.0%** |
| 95% | 1660 | 40.5% | 100.0% |

**92% agreement at 10% of the candidate space** is the number to carry away
from Phase 2. The jump to 40% buys the last eight percentage points.

---

## 5. Where we were right

1. **The core bet paid off.** The guess that the CRD score landscape has
   exploitable structure was correct, and the structure turned out to be
   stronger and cleaner than expected — an exact closed form rather than
   something that needed to be learned.
2. **Predicting the landscape offline works.** The formula, computed with zero
   attack queries, predicts the measured score surface with a correlation of
   **0.93** on the cells the search actually relies on, and the measured score
   rises monotonically with the prediction (`+0.18 < +0.32 < +0.59 < +1.20 < +2.23`).
3. **Insisting on a shared scoring object.** This removed an entire category of
   "your baseline was just slower" objections before it could arise.
4. **Checking theory against measurement everywhere.** Every linear equation is
   predicted by theory *and* measured by simulation. This caught a real bug
   (below) that would otherwise have silently poisoned everything downstream.
5. **Building the toy cipher first.** It let us run hundreds of attacks with
   known ground truth. On real DES alone, each experiment would have been too
   slow to iterate on.

---

## 6. Where we were wrong

This is the part worth reading. Six mistakes, roughly in order of how much they
cost.

### 6.1 We "reproduced" the paper's pipeline and it was badly broken — twice

**Mistake A — the network was accurate but useless.**

Our first distinguisher hit the theoretical optimum on accuracy: 52.19% against
a best-possible 52.17%. By the paper's own stated check ("accuracy above 51%"),
it was fine. It was not fine. In the multi-bit attack it scored **37.5%** where
classical Matsui got 67.5% — far worse than the method it was supposed to beat.

The reason: **the scoring rule does not use the network's decision, it sums the
network's confidence values.** Accuracy only constrains whether the confidence
lands on the right side of the line; the score depends on *how far*. We measured
how much of the network's output was explained by the quantity that actually
matters (the number of 1-bits in the sample) and got only **68%**. The other
32% was noise, and the scoring rule was adding up that noise across thousands of
samples.

The cause is that on a 52%-accuracy task the learning signal is tiny compared to
the randomness in training, so the network's decision boundary lands in the
right place while its confidence values stay noisy. Averaging the network's
weights over the second half of training removes that noise. Measured, on
identical data and the same random seed:

| | Before fix | After fix |
|---|---|---|
| Accuracy | 52.19% | 52.19% (unchanged) |
| Output explained by the sufficient statistic | 68% | **94%** |
| Signal retained vs. the theoretical best | 0.83 | **0.97** |
| **Average rank of the true key (lower is better)** | **68.8** | **4.7** |
| Success rate at the same data | 37.5% | 50.0% |

**The lesson: "accuracy above 51%" is not a sufficient check for this
framework.** It is the right check for a distinguisher and the wrong check for
the attack built on it. We now test the confidence calibration directly, so this
cannot regress silently.

**Mistake B — the paper's one-bit decision rule has a systematic bias.**

The paper's rule is "add up the confidence scores; if the total is positive,
guess 0." That is only correct if the network is perfectly symmetric — if
flipping every input bit flips the output by exactly the same amount. Real
networks are not. Ours was off by about `0.0028` per sample.

Per sample that is nothing. But the rule *adds up* thousands of samples, so the
error grows in proportion to the sample count while the signal it competes with
only grows like the square root. At N = 1355 samples the accumulated error was
`+3.8` against a signal of `3.9` — the same size. The rule was, at that point,
roughly a coin flip dressed up as a measurement.

The fix is one line and costs no extra data: compare the sample against its own
mirror image instead of against zero. Measured over 300 trials:

| Equation | Paper's rule as written | Corrected rule | Classical Matsui |
|---|---|---|---|
| L3 | 98.7% | 98.7% | 97.3% |
| **L4** | **83.7%** | **97.0%** | 97.7% |
| L5 | 98.0% | 98.0% | 98.0% |
| L6 | 96.0% | 96.0% | 97.7% |

The correction never hurts, and where the bias bites it recovers 13 percentage
points. Note this does **not** affect our main result: in the multi-bit attack
the offset shifts every candidate by the same amount, so the ranking is
unchanged.

**Both of these mistakes made the baseline stronger.** We are beating a
repaired version of the paper's method, not a broken one.

### 6.2 A bit-ordering bug that theory caught

Our first linear-equation search produced equations that theory said had a
strong bias and measurement said had none. The cause was a six-bit field being
read in the wrong order in one helper function. Only a coincidence of symmetry
made one equation appear to work, which is exactly how such a bug survives
casual testing.

**What saved us:** insisting that every equation be independently *predicted*
and *measured*. Theory said 0.658203, measurement said 0.658095 — that kind of
agreement is the check. Without it we would have built everything on nonsense
and the final numbers would have looked plausible.

### 6.3 We assumed the search could know when to stop

Our first stopping rule asked for 90% confidence that the best candidate found
so far was the global best. It essentially never triggered. The reason is not a
bug — it is correct reasoning we had not thought through. If 4000 candidates
remain untested, one good observation genuinely *cannot* prove the answer has
been found.

We then tried the obvious structural fix, and it also failed. Because the
wrong-key response factorises, a probe set can be chosen **offline** so that
every one of the 4096 hypotheses is within a given response of some probe (256
probes reach every hypothesis at 0.25; verified, not assumed). The idea was
that an untouched hypothesis keeps its prior, so covering the space first
should let the posterior concentrate. It does not. Measured against a plain
random opening, the covering design is *worse at every small budget*:

| Budget | random | covering@0.25 | covering@0.375 | covering@0.5 |
|---|---|---|---|---|
| 5% | **52%** | 8% | 0% | 8% |
| 7.5% | **56%** | 32% | 4% | 8% |
| 25% | 88% | **92%** | 88% | 20% |

The reasoning was wrong in an interesting way. We assumed the bottleneck was
*breadth*, but one probe already carries about **14.7 direct-test-equivalents**
of information (we measured `sum_h g_h^2 = 14.7`). Breadth was never the
problem. The problem is **contrast**: nearby hypotheses receive nearly
identical responses from every probe, so covering tells you which
*neighbourhood* the key is in but cannot separate candidates inside it — and
the adaptive search already localises quickly, so the 256-probe opening is
simply spent.

Then a third measurement settled the question. **At matched agreement, the
stopping rule is worse than simply choosing a budget.** It reaches 96%
agreement at a median of 54% of the space; a fixed budget reaches 96% at 40%.
Exhaustive Algorithm 2 only succeeds ~60% of the time at this data complexity,
so ~40% of trials contain no findable key; a budget gives up on those, while
the stopping rule cannot tell "hopeless" from "not yet found" and keeps
querying exactly where there is nothing to find.

**Conclusion: we do not lead with early stopping.** The deliverable is the
fixed-budget curve. The stopping rule is built, calibrated and reported as an
optional extra, with its full trade-off curve rather than one tuned number.

### 6.4 We assumed a standard optimiser would be a reasonable starting point

The original plan was to use `scikit-optimize` as the primary method. It turned
out to be unusable: it re-fits a Gaussian process after every observation, which
gets rapidly more expensive as observations accumulate, and on this problem it
is **slower than simply trying all 4096 candidates**. It also kept proposing
guesses it had already tested.

This was a wrong assumption that became a useful finding — it is now the
strongest evidence that the cryptanalytic prior, not the optimisation
machinery, is what does the work.

### 6.5 We measured the right thing the wrong way

Our first test of "does the offline prediction match reality?" computed a
correlation across all 4096 cells and got 0.42 — weak enough to look like a
failure. But 1699 of those cells are *predicted to be exactly zero*, so that
number was dominated by noise in cells the search never consults. Restricted to
the 60 cells the search actually uses, the correlation is **0.93**.

The claim was right; the statistic was wrong. Worth stating plainly because the
temptation in that situation is to quietly lower the threshold until the test
passes, which would have hidden a genuinely strong result behind a weak one.

### 6.6 We ran too few trials near the steep part of the curve

We measured the same configuration twice and got 87.5% and 50.0%. Not a bug: at
that data level the success curve climbs from 50% to 80% for a 25% increase in
data, so it is hypersensitive, and 40 trials is not enough there. The paper
used 1000.

**Now resolved.** Everything reported above was re-run at 150 trials (sweep and
head-to-head), 100 (budget curve) and 60 (DES), bringing the error bar to about
±4 pp. The ordering that flipped — whether the ML attack beats classical
Matsui — settled the other way once the noise was removed, and Section 4.1
reports the corrected result rather than the flattering one.

The lesson is worth stating plainly because it nearly cost us a wrong
conclusion in a submitted document: **a difference smaller than the error bar
is not a finding**, and at 40 trials the error bar here was 5x larger than the
effect we thought we had measured.

## 7. Honest assessment against the original goal

| Target | Result | Verdict |
|---|---|---|
| Reproduce the paper's framework | Matches at equal normalised data (88.0% vs 80.2% at ~7.5x bias⁻²) | ✅ met |
| Reproduce the paper's "ML beats classical" claim | ML is 1–7 pp *below* classical at 150 trials | ❌ not met |
| Cut the search well below \|GK\| | 98.7% agreement at 54% of queries; 92% agreement at 10% on DES | ✅ met |
| Lose ≤1–2 pp of success rate | **+0.7 pp** (a small gain, not a loss) | ✅ met |
| Search *decides for itself* when to stop | Built and calibrated, but worse than a fixed budget at matched agreement | ❌ not met |
| Show the gain is not just early stopping | Random at the identical budget: 54.7% vs 98.7% | ✅ met |
| Show the gain is not just "using an optimiser" | Generic Bayesian optimiser is 9x *slower* than brute force | ✅ met |
| Same scoring path for all methods | Literally the same object; verified by test on every attack | ✅ met |
| Run on real DES | 8-round DES, Matsui L6, the paper's exact guessed bits | ✅ met |

**The honest one-line summary:** knowing *where to look* is solved — the search
reproduces the full scan's answer 98.7% of the time at roughly half the cost,
and 92% of the time at a tenth of the cost on real DES. Knowing *when to stop
looking* is not solved, and we now have evidence that it is not worth solving
the way we tried: a fixed budget beats the adaptive rule at matched quality.

## 8. What we would do next

1. **Close the ML-vs-classical gap.** The most likely cause is our small
   residual MLP standing in for the paper's ResNet. Training a larger
   distinguisher and re-running the sweep would test that directly, and it is
   the one open question that bears on the paper's own claim.
2. **Test whether the response formula generalises.** It was derived for
   approximations with one active S-box per side. Matsui's `L5` activates five
   at one end and does not fit — mapping out when the method applies is a real
   limitation worth charting.
3. **Attack stopping from the decision-theoretic side, not the search side.**
   Our evidence says the problem is not the search order but the question:
   "am I certain?" is expensive, while "is this budget spent well?" is cheap.
   A cost-aware rule that gives up early on trials that look hopeless is the
   more promising direction than more clever probing.

## 9. Things a reader should be sceptical about

* Our success rates are on a **toy cipher**. We argue comparability by
  normalising against `bias^-2` and the numbers line up well, but that is an
  argument, not a proof. The real-DES run is what would settle it.
* **Trial counts are 60-150** (the paper used 1000), so error bars are
  about +/-4 pp. This was 40 in an earlier draft and it changed a
  conclusion; see 6.6.
* The **wall-clock speed-up should be read as secondary**. A real attacker would
  vectorise the brute-force scan across candidates, which a sequential search
  cannot do. The count of network evaluations is the implementation-independent
  metric and is the one we lead with.
* Our distinguisher is a **small residual network, not the paper's full ResNet**.
  We justify this by measuring it against the theoretical optimum rather than
  assuming it is good enough — but it is a deviation.
* The `sequential-earlystop` control in the main table uses an **uncalibrated**
  stopping threshold while our method's is calibrated. They are not matched on
  that axis. The matched comparison is the budget curve in 4.2, and that is the
  one the argument should rest on.
