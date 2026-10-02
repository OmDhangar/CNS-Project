# Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search

**Abstract**—Although neural distinguishers have recently achieved parity with Matsui's classical linear cryptanalysis on round-reduced block ciphers, their multiple-bits key-recovery procedure (e.g., Algorithm 2 in Hou, Ren, and Chen, *Cybersecurity* 2025) relies on an exhaustive brute-force search over all $L$ candidate subkeys. For an 8-round attack on the Data Encryption Standard (DES), this requires $L = 2^{12} = 4096$ full-budget neural evaluations, leaving candidate search reduction as a critical open problem. In this paper, we demonstrate that wrong key guesses in linear cryptanalysis are not statistically independent or uniform, but exhibit a structured correlation profile that factorises exactly into front and back S-box response kernels: $\text{corr}(gk) = \text{corr}(gk^*) \cdot \rho_f(k_f \oplus k_f^*) \cdot \rho_b(k_b \oplus k_b^*)$. Crucially, $\rho_f$ and $\rho_b$ depend exclusively on the cipher's S-boxes and linear approximation masks, allowing their exact shape to be computed in closed form offline before expending a single plaintext query. We exploit this insight to replace the exhaustive search loop with a *Wrong-Key-Response-Guided Bayesian Search* that dynamically maintains the exact posterior over all 4096 hypotheses with an $O(L)$ update costing less than a single neural evaluation. Validated across empirical attacks on TinyDES-24 and real 8-round DES (verified against FIPS test vectors), the guided search returns the identical candidate subkey as the exhaustive scan 98.7% of the time using only 54% of the evaluations on TinyDES-24 (+0.7 pp higher success rate), and reaches 92.0% agreement using only 10.2% of evaluations (418 queries) on real DES. Furthermore, our investigation identifies and repairs two foundational defects in existing neural scoring pipelines: logit variance insufficiency (remedied via weight averaging, lifting signal retention from 0.83 to 0.97) and logit asymmetry in one-bit decision rules (recovering up to 13.3 percentage points in key recovery).

**Index Terms**—Linear cryptanalysis, machine learning, neural distinguisher, key recovery, time complexity, wrong-key response profile, Bayesian search, DES.

---

## I. Introduction

Linear cryptanalysis, originally introduced by Matsui in [1], [2], exploits linear approximations connecting plaintext, ciphertext, and key bits with probability $p \neq 1/2$. Recently, machine learning techniques have emerged as powerful tools in symmetric cryptanalysis. In a foundational advance, Hou, Ren, and Chen (hereafter "HRC") [3] recast linear cryptanalysis as a binary classification problem between two Bernoulli distributions ($\text{Bern}(p)$ vs. $\text{Bern}(1-p)$), training a neural distinguisher ($ND$) to classify blocks of parity bits and accumulating evidence via a Combined-Response Distinguisher (CRD). Their framework is the first ML-aided linear attack to match classical Matsui success rates at equivalent data complexity.

However, recovering multiple subkey bits simultaneously (Algorithm 2 in [3]) requires wrapping the approximation in an additional cipher round at both the plaintext and ciphertext boundaries. For an 8-round attack on the Data Encryption Standard (DES), this creates a candidate subkey space of size $L = 2^{12} = 4096$. HRC evaluate every candidate key exhaustively against the complete plaintext budget $N \cdot t$, incurring a time complexity of $2^{12} \times N \times t / 8$ DES encryptions. Recognizing this inefficiency, HRC explicitly stated in their conclusion:
> *"Designing a better key-recovery strategy matching neural distinguishers will be effective to reduce the time complexity."* [3]

This paper directly addresses this open challenge. Rather than treating candidate scoring as an unstructured black-box optimization problem, we uncover a fundamental algebraic property of linear cryptanalysis: **the score response of an incorrect subkey guess is deterministic and factorisable**. Specifically, when an incorrect candidate subkey is tested, the resulting parity sequence has a reduced correlation determined by the differential properties of the target S-boxes under the approximation masks. We prove that this *wrong-key response profile* ($\rho_f, \rho_b$) can be computed in closed form offline in $O(1)$ time relative to the online attack budget.

Using this offline kernel, we design a **Wrong-Key-Response-Guided Bayesian Search**. The algorithm begins with a small opening sample and sequentially queries the candidate subkey with the highest posterior probability. Because the likelihood update reduces to maintaining three summary accumulators ($P, Q, V$), the per-query Bayesian bookkeeping requires only $O(L)$ operations—far less than the floating-point operations of a single neural network forward pass.

### Contributions
1. **Closed-Form Linear Response Profiles**: We derive the exact factorisation of wrong-key correlations for two-sided linear attacks, providing a closed-form offline calculation for arbitrary S-boxes and linear masks.
2. **Efficient Bayesian Search Algorithm**: We replace the $O(L)$ brute-force loop with an informed Bayesian search that requires zero neural retrainings or pipeline changes, operating as a strict drop-in replacement for Algorithm 2.
3. **Rigorous Empirical Validation**: We evaluate our method on both **TinyDES-24** (a 24-bit DES-like Feistel cipher enabling 100+ trial CPU ground truth) and **Real 8-round DES** (verified with FIPS test vectors and Matsui's $L_6$ approximation). On real DES, our algorithm reaches 92.0% agreement with the exhaustive scan while evaluating only 10.2% of the candidate space.
4. **Correction of Baseline Pipeline Defects**: We uncover and fix two subtle flaws in prior ML linear cryptanalysis implementations: (i) SGD logit noise that degrades CRD scoring despite high validation accuracy (resolved via Exponential Moving Average weight averaging, increasing $R^2$ from 0.68 to 0.94), and (ii) logit asymmetry in one-bit decision rules (resolved via antisymmetric scoring, recovering up to 13.3 pp).
5. **Negative Results and Design Boundaries**: We evaluate and explain why generic Bayesian optimization (e.g., Gaussian Processes via `scikit-optimize`) is $9\times$ slower than brute force, why covering probe designs fail due to contrast limits, and why fixed-budget modes systematically outperform calibrated stopping thresholds.

---

## II. Related Work

### A. Machine Learning in Cryptanalysis
The integration of deep learning with cryptanalysis gained major traction when Gohr [4] designed deep convolutional residual networks as differential distinguishers for Speck32/64, coupling them with a Bayesian key-guessing framework. Bao et al. [5] extended this via neutral bits to enhance key recovery efficiency. In linear cryptanalysis, Hou et al. [6] and Zhou et al. [7] explored early neural approximations. Most recently, HRC [3] achieved a breakthrough by formulating linear cryptanalysis as distinguishing Bernoulli parameters, matching Matsui's success rate on reduced-round DES.

### B. Wrong-Key Response Profiles
In differential cryptanalysis, Gohr [4] observed that neural distinguishers output elevated scores for key guesses adjacent to the true key. However, because differential transitions through complex rounds lack simple analytical descriptions, Gohr had to measure this profile *empirically* by running extensive pre-computations. In contrast, for linear cryptanalysis, we demonstrate that the linear correlation through active S-boxes yields an exact, closed-form profile directly computable from the S-box Look-Up Table (LUT) and linear masks.

### C. Complexity Reduction in Classical Linear Cryptanalysis
Junod [8] and Selçuk [9] established the formal success probability and sample complexity of Matsui's attacks. Collard, Standaert, and Quisquater [10] showed that Matsui's classical counting statistic across all subkey candidates can be computed simultaneously using a Fast Walsh–Hadamard Transform (FWHT). While FWHT accelerates classical parity counting across the full candidate space, ML-aided linear cryptanalysis evaluates a neural network on transformed samples, where candidate evaluation cannot be trivially expressed as a single global Walsh transform. Hence, reducing the number of candidate evaluations evaluated by the neural network is essential.

---

## III. The Base Framework and Its Computational Bottleneck

### A. ML-Aided Linear Key Recovery (HRC Algorithm 2)
Let $E_K$ be an $R$-round Feistel cipher. Matsui's linear cryptanalysis relies on an $(R-2)$-round linear approximation over rounds $1$ to $R-2$:
$$\alpha \cdot P_{1} \oplus \beta \cdot C_{R-1} = \gamma \cdot K' \quad \text{with probability } p_r \neq \frac{1}{2}$$
where $P_1$ is the state after round 0, $C_{R-1}$ is the state before round $R-1$, and $\gamma \cdot K'$ is a parity of intermediate key bits.

In an 8-round attack, the attacker extends the approximation by one round at the plaintext end (Round 0, guessing front subkey bits $k_f$) and one round at the ciphertext end (Round 7, guessing back subkey bits $k_b$). For each candidate subkey $gk = (k_f, k_b) \in GK$, the attacker partially encrypts/decrypts the input pairs $(P, C)$ to compute the computed bit:
$$x(gk) = \alpha_R \cdot R_0 \oplus \beta_L \cdot L_7 \oplus \alpha_L \cdot F(R_0, k_f) \oplus \beta_R \cdot F(L_7, k_b)$$

For the correct candidate $gk^* = (k_f^*, k_b^*)$, $x(gk^*) \sim \text{Bern}(p_r)$ if $\gamma \cdot K' = 0$ and $\text{Bern}(1 - p_r)$ if $\gamma \cdot K' = 1$. For an incorrect candidate, $x(gk)$ has diminished bias.

HRC train a neural network $ND_r^t$ on $t$-bit parity vectors. To score candidate $gk$ over $N$ batches of $t$ samples, HRC define the Combined-Response Distinguisher (CRD) scoring rule:
$$w_0(gk) = \sum_{i=1}^N \log_2 \frac{v_i(gk)}{1 - v_i(gk)}, \quad w_1(gk) = \sum_{i=1}^N \log_2 \frac{1 - v_i(gk)}{v_i(gk)} = -w_0(gk)$$
$$S(gk) = \max(w_0(gk), w_1(gk)) = |w_0(gk)|$$
where $v_i(gk) = ND_r^t(X_i(gk))$ is the neural network output probability for batch $i$.

### B. The Brute-Force Bottleneck
Algorithm 2 in [3] evaluates $S(gk)$ for all $L = |GK|$ candidates exhaustively:
```python
for gk in GK: # L = 4096 iterations
    compute S(gk) over all N * t samples
return argmax_{gk} S(gk)
```
For DES with 6 active key bits per S-box at each boundary, $|GK| = 2^6 \times 2^6 = 4096$. Scoring each candidate against $N \cdot t$ plaintexts results in $4096 \times N \times t$ neural distinguisher sample evaluations. Because $99.97\%$ of these candidate keys are incorrect, exhaustively evaluating all 4096 candidates over the full sample budget represents enormous computational waste.

---

## IV. Proposed Approach: Wrong-Key-Response-Guided Bayesian Key Search

```
                                OFFLINE (0 Queries)
       ┌──────────────────────────────────────────────────────────────────┐
       │ Compute S-box Correlation Kernels:                               │
       │ ρ_f(Δ_f) from S_jf and ν_f;   ρ_b(Δ_b) from S_jb and ν_b         │
       │ Predicted Profile: g_h(k) = |ρ_f(k_f ⊕ k_f^h) · ρ_b(k_b ⊕ k_b^h)|│
       └─────────────────────────────────┬────────────────────────────────┘
                                         │
                                ONLINE SEARCH LOOP
       ┌─────────────────────────────────▼────────────────────────────────┐
       │ 1. Evaluate small random opening set of candidates               │
       │ 2. Update Bayesian Accumulators in O(|GK|):                      │
       │      P(h) += s_i · g_h(k_i)                                      │
       │      Q(h) += g_h(k_i)                                            │
       │      V(h) += g_h(k_i)²                                           │
       │ 3. Compute Posterior:                                            │
       │      L(h|A) = exp( A·(P - μQ)/σ - ½ A² V )                       │
       │      posterior(h) ∝ ∫ L(h|A) p(A) dA                             │
       │ 4. Query next candidate: k_{next} = argmax_{k ∉ Evaluated} Post  │
       │ 5. Terminate when Budget B is reached (or threshold reached)     │
       └──────────────────────────────────────────────────────────────────┘
```
*Fig. 1. Architecture of the Wrong-Key-Response-Guided Bayesian Key Recovery Pipeline.*

### A. Closed-Form Wrong-Key Response Factorisation
Let $gk^* = (k_f^*, k_b^*)$ be the true subkey and $gk = (k_f, k_b)$ be an arbitrary candidate guess. Let $\Delta_f = k_f \oplus k_f^*$ and $\Delta_b = k_b \oplus k_b^*$.

The calculated parity bit $x(gk)$ can be written as:
$$x(gk) = x(gk^*) \oplus e_f(R_0, \Delta_f) \oplus e_b(L_7, \Delta_b)$$
where $e_f(R_0, \Delta_f) = \nu_f \cdot [S_{jf}(R_0 \oplus k_f^*) \oplus S_{jf}(R_0 \oplus k_f^* \oplus \Delta_f)]$, $\nu_f$ is the active S-box output mask, and $S_{jf}$ is the target S-box.

By Matsui's Piling-Up Lemma, assuming independence between the round inputs and cipher state:
$$\text{corr}(x(gk)) = \text{corr}(x(gk^*)) \cdot \rho_f(\Delta_f) \cdot \rho_b(\Delta_b)$$
where the individual S-box correlation response $\rho(\Delta)$ for an S-box $S$ with output mask $\nu$ is:
$$\rho(\Delta) = \frac{1}{2^m} \sum_{u \in \{0,1\}^m} (-1)^{\nu \cdot [S(u) \oplus S(u \oplus \Delta)]}$$

**Theorem 1 (Offline Computability).** *The correlation response $\rho_f(\Delta_f)$ and $\rho_b(\Delta_b)$ depends strictly on the cipher's S-box definitions and the linear masks $\nu_f, \nu_b$. It is entirely independent of the secret key $K$, plaintext data $P$, and ciphertext $C$.*

For 6-bit S-boxes, $\rho_f$ and $\rho_b$ are 64-element vectors precomputed in $2 \times 64 \times 64 = 8192$ operations ($< 1\text{ ms}$). Because CRD scores accumulate magnitude $|w_0|$, the expected score profile under hypothesis $h = (k_f^h, k_b^h)$ is:
$$g_h(k) = |\rho_f(k_f \oplus k_f^h) \cdot \rho_b(k_b \oplus k_b^h)|$$

### B. Fast Bayesian Posterior Formulation
We model the observed CRD score $s_i$ for tested candidate $k_i$ as:
$$s_i = \mu + \sigma A g_h(k_i) + \sigma \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 1)$$
where $\mu, \sigma$ represent the background noise floor (estimated robustly via median and Median Absolute Deviation (MAD) of evaluated scores), and $A$ is the unknown signal-to-noise ratio.

Given a set of $m$ evaluated candidates $\{(k_i, s_i)\}_{i=1}^m$, the log-likelihood of hypothesis $h$ conditioned on $A$ is:
$$\log \mathcal{L}(h \mid A) = -\frac{1}{2\sigma^2} \sum_{i=1}^m (s_i - \mu - \sigma A g_h(k_i))^2 = A \cdot U(h) - \frac{1}{2} A^2 \cdot V(h) + C$$
where:
$$P(h) = \sum_{i=1}^m s_i g_h(k_i), \quad Q(h) = \sum_{i=1}^m g_h(k_i), \quad V(h) = \sum_{i=1}^m g_h(k_i)^2, \quad U(h) = \frac{P(h) - \mu Q(h)}{\sigma}$$

**Computational Complexity:** When a new candidate $k_m$ is queried with score $s_m$:
1. $g_h(k_m)$ is computed for all $h \in GK$ as an outer product vector $\mathbf{g} = |\boldsymbol{\rho}_f \otimes \boldsymbol{\rho}_b|$.
2. Accumulators are updated in vectorised form:
   $$\mathbf{P} \leftarrow \mathbf{P} + s_m \mathbf{g}, \quad \mathbf{Q} \leftarrow \mathbf{Q} + \mathbf{g}, \quad \mathbf{V} \leftarrow \mathbf{V} + \mathbf{g}^2$$
3. Marginalising $A$ over a discrete grid $\mathcal{A}$ yields the posterior:
   $$\text{posterior}(h) \propto \sum_{A \in \mathcal{A}} \exp\left( A \cdot U(h) - \frac{1}{2} A^2 \cdot V(h) \right) p(A)$$

This complete update takes $O(|GK|)$ operations (a few thousand additions/multiplications in NumPy), executing in under $0.2\text{ ms}$ on CPU—negligible compared to neural forward passes.

---

## V. Experimental Setup and Target Ciphers

### A. Target Ciphers
We validate our approach across two Feistel cipher environments:

1. **TinyDES-24** (`src/ciphers/toy_feistel.py`):
   * Block size: 24 bits, Key size: 24 bits, Rounds: 8.
   * Utilizes real DES S-boxes ($S_1, S_2, S_3$), DES-style expansion $E$, and bit permutation $P$.
   * Attack geometry exactly mirrors HRC's 8-round attack: 6-round linear approximation ($L_6$, bias $2^{-8.20}$), 6 guessed subkey bits at Round 0 ($k_f$), 6 guessed subkey bits at Round 7 ($k_b$), $|GK| = 4096$.
   * Enables exhaustive 4096-candidate ground truth across 100–150 independent trials on standard CPU hardware.

2. **Real 8-Round DES** (`src/ciphers/des.py`):
   * Standard 64-bit block, 56-bit key.
   * Validated against the official FIPS 46-3 test vector:
     $$\text{DES}(\texttt{0x0123456789ABCDEF}, \texttt{0x133457799BBCDFF1}) = \texttt{0x85E813540F0AB405}$$
   * Attacked with Matsui's 6-round approximation $L_6$ (measured $p = 0.495753$).
   * Front guess: S-box 5, subkey bits $K_0[18..23]$; Back guess: S-box 1, subkey bits $K_7[42..47]$, $|GK| = 4096$.

### B. Control Baselines
To rigorously isolate the sources of performance gain, we evaluate:
1. `exhaustive`: Full scan of all 4096 candidates over all $N \cdot t$ samples (HRC Algorithm 2).
2. `guided-wkr-budget`: Our proposed WKR Bayesian search with a fixed query budget.
3. `guided-wkr`: Our proposed WKR Bayesian search with calibrated posterior early stopping.
4. `random`: Uniform random candidate selection at the identical query budget (no model, no ordering).
5. `sequential-earlystop`: Index-order scan with early stopping (stopping rule with no spatial model).
6. `guided-skopt`: Generic Bayesian optimization using Gaussian Process regression over the 12-dimensional key bit hypercube.

---

## VI. Experimental Results and Analysis

### A. Reproduction of the Base Paper
We first verified that our neural distinguisher and CRD pipeline faithfully reproduce HRC's baseline. Normalizing data complexity by $\text{bias}^{-2}$ demonstrates consistent recovery curves:

#### TABLE I: Reproduction of HRC Framework Baseline
| Normalised Data ($\times \text{bias}^{-2}$) | HRC Published (Real DES, 1000 trials) | Ours (TinyDES-24, 150 trials) |
| :---: | :---: | :---: |
| $7.3\times / 7.6\times$ | 80.2% | **88.0%** |
| $8.7\times / 9.1\times$ | 88.4% | **90.7%** |
| $10.2\times / 10.6\times$ | 93.6% | **94.7%** |
| $11.6\times / 12.1\times$ | 96.5% | 93.3% |

### B. Headline Result: Head-to-Head Comparison
All methods evaluated candidate subkeys through the **identical trained neural distinguisher and CRD scorer**.

#### TABLE II: Head-to-Head Performance Comparison on TinyDES-24 (150 Independent Trials)
| Method | Success Rate | vs. Baseline | Agreement with Full Scan | Median Evals | % of $\|GK\|$ | Eval Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exhaustive Algorithm 2** (HRC [3]) | 72.0% $\pm$ 7.2 | +0.0 pp | 100.0% (by def.) | 4096 | 100.0% | $1.00\times$ |
| **Guided WKR Budget (Ours)** | **72.7% $\pm$ 7.1** | **+0.7 pp** | **98.7%** | **2227** | **54.4%** | **$1.88\times$** |
| Guided WKR (Calibrated Stopping) | 72.7% $\pm$ 7.1 | +0.7 pp | 97.3% | 2634 | 64.3% | $1.61\times$ |
| Random (Matched Budget) | 38.7% $\pm$ 7.8 | -33.3 pp | 54.7% | 2227 | 54.4% | $1.88\times$ |
| Sequential + Early Stopping | 7.3% $\pm$ 4.2 | -64.7 pp | 8.0% | 204 | 5.0% | $15.02\times$ |
| Generic BO (`scikit-optimize` GP) | 10.0% $\pm$ 18.6 | -70.0 pp | 10.0% | 99 | 2.4% | $0.11\times$ |

**Key Findings:**
* **$98.7\%$ Agreement at $54\%$ Queries:** The guided search identifies the exact same key as the full 4096-candidate scan in 98.7% of trials while saving nearly half the evaluations.
* **$+0.7\text{ pp}$ Accuracy Gain:** On trials where the true key score is marginally below a random spurious peak, early termination based on the true key's high posterior prevents selecting the false global maximum.
* **Control Separation:** Random sampling at 54% budget achieves only 38.7% success (54.7% agreement), proving that the performance gain stems from the S-box response model rather than sample reduction alone.

### C. Budget Sensitivity and Trade-off Curve
By evaluating search trajectories across varying candidate budgets, we establish the fundamental budget curve:

#### TABLE III: Budget Sensitivity on TinyDES-24 (100 Trials)
| Budget (% of $\|GK\|$) | Candidate Queries | Guided WKR Success | Random Success | Sequential Success | Agreement with Full Scan |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.5% | 20 | 3.0% | 2.0% | 0.0% | 4.0% |
| 1.0% | 41 | 16.0% | 2.0% | 1.0% | 19.0% |
| 2.0% | 82 | 26.0% | 3.0% | 3.0% | 33.0% |
| 5.0% | 205 | 39.0% | 9.0% | 6.0% | 57.0% |
| 10.0% | 410 | 50.0% | 13.0% | 13.0% | 72.0% |
| 15.0% | 614 | 56.0% | 14.0% | 17.0% | 80.0% |
| 20.0% | 819 | 59.0% | 15.0% | 22.0% | 86.0% |
| 25.0% | 1024 | 60.0% | 19.0% | 23.0% | 87.0% |
| **50.0%** | **2048** | **67.0%** | **41.0%** | **42.0%** | **98.0%** |
| 100.0% | 4096 | 68.0% | 68.0% | 68.0% | 100.0% |

*(Full exhaustive scan achieves 68.0% success on this trial set).* Guided WKR reaches near-exhaustive recovery performance at only **25%–50%** of the candidate queries.

### D. Phase 2: Validation on Real 8-Round DES
We applied the exact same search logic to real 8-round DES with Matsui's $L_6$ approximation ($N \cdot t = 400,000$, 60 trials).

#### TABLE IV: Real 8-Round DES Attack Results
| Method | Success Rate | Agreement with Exhaustive | Median Queries | % of Candidate Space |
| :--- | :---: | :---: | :---: | :---: |
| Exhaustive Scan | 58.3% | 100.0% | 4096 | 100.0% |
| **Guided WKR** | **56.7%** | **95.0%** | **2911** | **71.1%** |
| Random Sampling | 10.0% | 25.0% | 1024 | 25.0% |

By sweeping the agreement target, the trade-off curve on real DES is revealed:

#### TABLE V: Real DES Agreement Trade-off
| Agreement Target | Median Queries | % of Space | Actual Agreement | Success Rate |
| :---: | :---: | :---: | :---: | :---: |
| 80% | 259 | **6.3%** | 80.0% | 72.0% |
| **90%** | **418** | **10.2%** | **92.0%** | **80.0%** |
| 95% | 1660 | 40.5% | 100.0% | 76.0% |

**Headline Real DES Result:** On real 8-round DES, our guided search achieves **92.0% agreement with the exhaustive scan using only 10.2% of the candidate evaluations (418 queries)**.

---

## VII. Discovery and Correction of Baseline Pipeline Defects

In faithfully reproducing HRC's pipeline, our investigation revealed two fundamental issues in prior ML linear cryptanalysis implementations. Fixing these issues strengthens the baseline itself:

### A. Defect 1: Distinguisher Logit Noise and Popcount Variance
HRC specify that a neural distinguisher is sufficient if its classification accuracy clears 51%. However, CRD scoring (Eq. 2 in [3]) sums the network's continuous **logits**, not binary predictions. Accuracy constrains only the sign; CRD score quality depends on logit magnitude fidelity.

On Bernoulli distinguishing tasks, the popcount is the mathematical sufficient statistic. Any logit variance not explained by the popcount represents pure noise accumulated during scoring. In our initial training, a network with 52.19% validation accuracy (matching the Bayes optimal 52.17%) explained only $R^2 = 0.68$ of its logit variance, retaining only $\sqrt{0.68} = 0.83$ of available signal. This caused multiple-bit attack success to plummet to 37.5% (versus Matsui's 82.5%).

**Solution:** Averaging neural network weights over the second half of training via Exponential Moving Average (EMA) eliminated SGD gradient noise without altering the decision boundary.

#### TABLE VI: Distinguisher Logit Fidelity Before and After EMA Weight Averaging
| Metric | Standard Training | EMA Weight Averaged |
| :--- | :---: | :---: |
| Validation Accuracy | 52.19% | 52.19% (Unchanged) |
| Logit Variance Explained by Popcount ($R^2$) | 0.68 | **0.94** |
| Signal Retention ($\sqrt{R^2}$) | 0.83 | **0.97** |
| Multiple-Bit Attack Success Rate | 37.5% | **87.5%** |
| Mean Rank of True Key | 68.8 | **4.7** |

### B. Defect 2: Asymmetry in One-Bit Decision Scoring
HRC's Step 4 rule—deciding $\gamma \cdot K' = 0$ if $\sum_i \log_2(v_i / (1 - v_i)) > 0$—is valid only if the network is perfectly antisymmetric under input complementation ($ND(1-x) = 1 - ND(x)$). Real trained networks exhibit a persistent offset (measured at $+2.8 \times 10^{-3}$ per sample).

While negligible per sample, summing over $N = 1355$ samples accumulates an error of $+3.8$, directly overwhelming the true signal ($\approx 3.9$).

**Solution:** We formulate an antisymmetric scoring rule:
$$\text{Score}_{\text{corrected}} = \frac{1}{2} \sum_{i=1}^N [\text{logit}(X_i) - \text{logit}(\mathbf{1} - X_i)]$$
This restores exact antisymmetry by construction, costs zero additional plaintexts, and recovers up to 13.3 percentage points in accuracy:

#### TABLE VII: One-Bit Key Recovery Success Rates
| Approximation | HRC Rule as Written | Corrected Antisymmetric Rule | Classical Matsui Alg. 1 |
| :---: | :---: | :---: | :---: |
| $L_3$ | 98.7% | **98.7%** | 97.3% |
| **$L_4$** | **83.7%** | **97.0% (+13.3 pp)** | 97.7% |
| $L_5$ | 98.0% | **98.0%** | 98.0% |
| $L_6$ | 96.0% | **96.0%** | 97.7% |

---

## VIII. Negative Results and Search Design Boundaries

We document three critical negative results that clarify why alternative acceleration techniques fail:

1. **Generic Bayesian Optimization (`scikit-optimize`) is 9× Slower than Brute Force:** Standard Gaussian Process regression treats the key bits as a 12-dimensional black-box hypercube. Refitting the $O(n^3)$ kernel covariance matrix after every query incurs extreme computational overhead (51.76 s per attack vs. 5.85 s for full brute force) while achieving only 10.0% success due to lack of domain-specific S-box structure.
2. **Covering Probe Designs Fail Due to Contrast Limits:** We tested choosing a pre-computed probe set to cover the hypothesis space within a fixed correlation radius. Surprisingly, this performed worse than random opening at small budgets. The bottleneck in linear cryptanalysis is not *breadth* (a single query provides $\sum_h g_h^2 = 14.7$ direct-test-equivalents), but *contrast*—neighbouring hypotheses produce highly similar responses that cannot be resolved without targeted local queries.
3. **Budget Mode Dominates Calibrated Early Stopping:** While a calibrated posterior stopping rule functions effectively, approximately 40% of trials at low data budgets contain no detectable key signal. A fixed budget gracefully truncates hopeless trials, whereas stopping rules over-query unresolvable instances.

---

## IX. Conclusion and Future Work

This paper resolves the primary open question posed by Hou, Ren, and Chen [3] by introducing **Wrong-Key-Response-Guided Bayesian Search** for machine-learning-aided linear cryptanalysis. By proving that wrong-key correlations factorise into closed-form S-box profiles ($\rho_f, \rho_b$), we convert an expensive $L = 4096$ neural search into an informed, $O(L)$ Bayesian search.

On real 8-round DES, our method achieves **92.0% agreement with the exhaustive attack while evaluating only 10.2% of the candidate space**, and reaches 98.7% agreement at 54% queries on TinyDES-24 without requiring any neural retraining or pipeline alterations.

**Future Directions:**
1. Combining WKR Bayesian search with Fast Walsh–Hadamard Transforms to explore hybrid neural-classical recovery.
2. Extending closed-form response kernels to larger block ciphers (e.g., AES-128 and Simon/Speck).
3. Multi-fidelity screening, utilizing distilled neural distinguishers for initial candidate filtering.

---

## References

[1] M. Matsui, “Linear cryptanalysis method for DES cipher,” in *Proc. EUROCRYPT 1993*, LNCS 765, Springer, pp. 386–397, 1993.  
[2] M. Matsui, “The first experimental cryptanalysis of the Data Encryption Standard,” in *Proc. CRYPTO 1994*, LNCS 839, Springer, pp. 1–11, 1994.  
[3] Z. Hou, J. Ren, and S. Chen, “Improved machine learning-aided linear cryptanalysis: application to DES,” *Cybersecurity*, vol. 8, no. 22, 2025.  
[4] A. Gohr, “Improving attacks on round-reduced Speck32/64 using deep learning,” in *Proc. CRYPTO 2019*, LNCS 11693, Springer, pp. 150–179, 2019.  
[5] Z. Bao, J. Guo, M. Liu, L. Ma, and Y. Tu, “Enhancing differential-neural cryptanalysis,” in *Proc. ASIACRYPT 2022*, LNCS 13793, Springer, pp. 318–347, 2022.  
[6] B. Hou, Y. Li, H. Zhao, and B. Wu, “Linear attack on round-reduced DES using deep learning,” in *Proc. ESORICS 2020*, LNCS 12282, Springer, pp. 131–145, 2020.  
[7] R. Zhou, M. Duan, Q. Wang, Q. Wu, S. Guo, L. Guo, and Z. Gong, “Neural-linear attack based on distribution data and its application on DES,” in *Proc. WCSE 2023*, pp. 64–73, 2023.  
[8] P. Junod, “On the complexity of Matsui's attack,” in *Proc. SAC 2001*, LNCS 2259, Springer, pp. 199–211, 2001.  
[9] A. A. Selçuk, “On probability of success in linear and differential cryptanalysis,” *Journal of Cryptology*, vol. 21, no. 1, pp. 131–147, 2008.  
[10] B. Collard, F.-X. Standaert, and J.-J. Quisquater, “Improving the time complexity of Matsui's linear cryptanalysis,” in *Proc. ICISC 2007*, LNCS 4817, Springer, pp. 77–88, 2007.
