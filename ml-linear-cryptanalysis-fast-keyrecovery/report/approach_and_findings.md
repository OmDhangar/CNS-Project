# How we got here: approach, results, and the things we got wrong

*A plain-language companion to `project_report.md`. That document is generated
from the result CSVs and holds the final tables; this one explains the
reasoning, and is honest about the wrong turns, because several of them are
more instructive than the result.*

**Status at time of writing:** the reproduction and the optimisation are both
built and measured. The final large-trial runs (exp2, exp3, and the real-DES
Phase 2) were still executing; every number below is from a completed run and
says how many trials it rests on.

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

### 4.1 Did we faithfully reproduce the paper? Yes.

Success rates on two different ciphers cannot be compared directly, because the
attack's difficulty depends on how weak the equation is. The standard way to
make them comparable is to measure data in units of `bias^-2` — the natural
scale of the problem.

| Data (in units of bias⁻²) | **Paper** (real DES, 1000 trials) | **Us** (TinyDES-24, 40 trials) |
|---|---|---|
| 6.0x | — | 50.0% |
| **7.3x / 7.6x** | **80.2%** | **80.0%** |
| **8.7x / 9.1x** | **88.4%** | **92.5%** |
| **10.2x / 10.6x** | **93.6%** | **97.5%** |
| 11.6x / 12.1x | 96.5% | **100.0%** |

At matched difficulty, they got 80.2% and we got 80.0%. The curve has the same
shape and sits in the same place. The reproduction is sound.

**Their second claim also reproduces.** The paper argues the ML attack is
slightly *better* than Matsui's classical method on the same data. We see the
same crossover:

| Data | ML-aided Algorithm 2 | Classical Matsui Algorithm 2 | Winner |
|---|---|---|---|
| 524,288 (6.0x) | 50.0% | 67.5% | classical |
| 655,360 (7.6x) | 80.0% | 85.0% | classical |
| 786,432 (9.1x) | **92.5%** | 90.0% | **ML** |
| 917,504 (10.6x) | 97.5% | 97.5% | tie |
| 1,048,576 (12.1x) | 100.0% | 100.0% | tie |

*(40 trials per row.)* The ML method starts behind and pulls ahead once there is
enough data — the paper reports the same crossover, at 80.2% vs 78.6%.

### 4.2 The optimisation: what we actually gained

The paper has **no entry in this table**, because it never optimised the search.
Its cost is fixed at `2^12 x N x t / 8` — always 4096 network evaluations. So
the comparison is against its own exhaustive scan.

| Metric | Paper's Algorithm 2 | **Our guided search** | Difference |
|---|---|---|---|
| Network evaluations per attack | **4096, every time** | **614** (median) | **6.7x fewer** |
| Returns the same key as the full scan | 100% by definition | **93.3%** | −6.7 pp |
| Success rate | 66.7% | 63.3% | **−3.4 pp** |
| Fraction of the search space skipped | 0% | **~85%** | — |

*(30 independent trials.)* Full curve:

| Query budget | % of 4096 | Success | Agrees with full scan |
|---|---|---|---|
| 205 | 5% | 43.3% | 56.7% |
| 307 | 7.5% | 56.7% | 70.0% |
| **614** | **15%** | **63.3%** | **93.3%** |
| 1024 | 25% | 63.3% | 96.7% |
| 4096 | 100% | 66.7% | 100% |

### 4.3 Is the gain real, or an artefact?

Two controls, both at the same budget, isolate the two possible explanations.

| Method (same 15% budget, same network, same data) | Agrees with full scan | What it tests |
|---|---|---|
| **Guided search (ours)** | **93.3%** | — |
| Random sampling | ~20% | Is the gain just "stopping early"? **No.** |
| scikit-optimize (generic Bayesian optimisation) | fails, and is *slower* than brute force | Is the gain just "applying an optimiser"? **No.** |

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
bug — it is correct reasoning that we had not thought through. If 4000
candidates remain untested, one good observation genuinely *cannot* prove the
answer has been found. Demanding certainty means testing nearly everything,
which defeats the purpose.

This forced an honest reframing: **the real result is "success rate at a given
budget", not "the search decides for itself when to quit."** Finding the key is
cheap; *proving* you have found it is not. This is still the weakest part of the
work (see below).

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

We measured the same configuration twice and got 87.5% and 50.0%. Not a bug:
at that data level the success curve climbs from 50% to 80% for a 25% increase
in data, so it is hypersensitive, and 40 trials is simply not enough there. The
paper used 1000.

**This is unresolved.** The numbers in the tables above carry roughly ±8
percentage points of uncertainty, and more near the steep region. Raising the
trial count is the first item on the to-do list.

---

## 7. Honest assessment against the original goal

| Target | Result | Verdict |
|---|---|---|
| Reproduce the paper's framework | Matches at equal normalised data (80.0% vs 80.2%) | ✅ met |
| Find the key using ≤15–20% of the evaluations | 15% of queries, 93.3% agreement | ✅ met |
| Lose ≤1–2 pp of success rate | −3.4 pp at that budget | ⚠️ slightly over |
| Search *decides for itself* when to stop, within 1 pp | needs 37.7% of queries | ❌ above target |
| Show the gain is not just early stopping | random search fails at the same budget | ✅ met |
| Show the gain is not just "using an optimiser" | generic optimiser is worse than brute force | ✅ met |
| Same scoring path for all methods | literally the same object; verified by test | ✅ met |
| Run on real DES | implemented and validated; final run pending | 🔄 in progress |

**The honest one-line summary:** knowing *where to look* is solved — roughly 85%
of the search can be skipped. Knowing *when to stop looking* is not solved, and
currently costs about 38% of the search to do safely.

---

## 8. What we would do next

1. **Raise the trial count to 150–200.** The cheapest fix; the current error
   bars are the main thing limiting what can be claimed.
2. **Finish the real-DES run.** The identical search code on 8-round DES in the
   paper's exact geometry (guessing `K0[18..23]` and `K7[42..47]`). This is what
   would let a number be placed directly beside the paper's own table.
3. **Attack the stopping problem properly.** The most promising idea is a
   two-stage search: first a cheap set of probes chosen offline so that every
   possible key is "near" one of them under the response formula, then focused
   refinement around whichever probe scores highest. That converts the
   stopping question from "am I certain?" into "has the coarse pass found
   anything?", which is a much cheaper question.
4. **Test whether the response formula generalises.** It was derived for
   one-active-S-box-per-side equations. Matsui's `L5`, for example, activates
   five S-boxes at one end and does not fit — understanding when the method
   applies is a real limitation worth mapping out.

---

## 9. Things a reader should be sceptical about

* Our success rates are on a **toy cipher**. We argue comparability by
  normalising against `bias^-2` and the numbers line up well, but that is an
  argument, not a proof. The real-DES run is what would settle it.
* **40 trials** is too few (the paper used 1000). See 6.6.
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
