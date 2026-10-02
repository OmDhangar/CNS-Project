# Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search

[![Coursework](https://img.shields.io/badge/Coursework-CNS--Project-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Paper](https://img.shields.io/badge/Research--Paper-RESEARCH__PAPER.md-orange.svg)](RESEARCH_PAPER.md)
[![Verification](https://img.shields.io/badge/Tests-30%20Passing-brightgreen.svg)](#)

This repository contains the complete research paper, mathematical framework, source code, neural network models, and experimental evaluation for:

> **"Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search"**  
> 📄 **Full Paper**: [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) *(also available at [`ml-linear-cryptanalysis-fast-keyrecovery/report/RESEARCH_PAPER.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/RESEARCH_PAPER.md))*

---

## 1. Overview & Problem Formulation (Paper Section I & III)

This research directly resolves the key-recovery computational bottleneck identified by:
> Zezhou Hou, Jiongjiong Ren, Shaozhen Chen, *"Improved machine learning-aided linear cryptanalysis: application to DES"*, **Cybersecurity** 8:22 (2025).

In an 8-round attack on the Data Encryption Standard (DES), recovering multiple subkey bits simultaneously requires guessing 6 subkey bits at Round 0 ($k_f$) and 6 subkey bits at Round 7 ($k_b$), creating a candidate space of size **$L = 2^{12} = 4096$**. 

* **The Problem**: Hou et al.'s **Algorithm 2** scores each of the 4096 candidate subkeys against the entire plaintext budget $N \cdot t$ before making any decision, incurring a time complexity of **$2^{12} \times N \times t / 8$ DES encryptions**.
* **The Goal**: Reduce the time complexity of the candidate key search without changing the neural distinguisher architecture, scoring rule, or dataset.

---

## 2. Every Approach Investigated in the Research Paper

The research paper systematically evaluates and compares **7 distinct approaches** across the baseline, proposed techniques, control baselines, and architectural repairs:

```
                                    KEY RECOVERY APPROACHES
 ┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
 │         PROPOSED SEARCH APPROACHES           │              BASELINE APPROACHES             │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ 1. Guided WKR Budget Search (Proposed Main)  │ 3. Exhaustive Scan (Hou et al. 2025 Alg. 2)  │
 │ 2. Guided WKR Calibrated Stopping (Proposed) │ 4. Classical Matsui Algorithm 2 (1994)       │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │         CONTROL & COMPARISON SEARCHES        │          PIPELINE REPAIR APPROACHES          │
 ├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
 │ 5. Uniform Random Sampling Control           │ 8. EMA Weight-Averaged Distinguisher         │
 │ 6. Sequential Scan + Early Stopping Control  │ 9. Antisymmetric Logit Differential Scoring  │
 │ 7. Generic Bayesian Optimization (skopt GP)  │                                              │
 │ 8. Covering Probe Design Search              │                                              │
 └──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

### Approach 1: Wrong-Key-Response-Guided Bayesian Search (Fixed Budget Mode) — *Primary Contribution*
* **Paper Section**: **Section IV-A, IV-B & VI-B** | **Code**: [`src/attacks/multi_bit_guided.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_guided.py)
* **Description**: Exploits the exact closed-form correlation factorisation of wrong key guesses:
  $$\text{corr}(gk) = \text{corr}(gk^*) \cdot \rho_f(k_f \oplus k_f^*) \cdot \rho_b(k_b \oplus k_b^*)$$
  where $\rho_f(\Delta)$ and $\rho_b(\Delta)$ are 64-entry tables precomputed offline in $<1\text{ ms}$ directly from S-box tables and approximation masks.
* **Mechanism**: Evaluates candidate keys sequentially against full plaintext batches. Maintains exact Bayesian accumulators ($\mathbf{P}, \mathbf{Q}, \mathbf{V}$) in $O(|GK|)$ time ($<0.2\text{ ms}$ per query), querying the Maximum A Posteriori (MAP) candidate until a fixed budget $B$ is reached.
* **Outcome**: 
  * **TinyDES-24**: Reaches **98.7% agreement** with the full scan and **+0.7 pp higher success rate (72.7%)** using only **54.4% of candidate queries (2227 / 4096)**.
  * **Real 8-Round DES**: Reaches **92.0% agreement** with the full scan using only **10.2% of queries (418 / 4096)**.

---

### Approach 2: Wrong-Key-Response-Guided Bayesian Search (Calibrated Early-Stopping Mode)
* **Paper Section**: **Section IV-B, VI-B & VII** | **Code**: [`src/attacks/multi_bit_guided.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_guided.py), [`experiments/calibrate_stopping.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/calibrate_stopping.py)
* **Description**: Instead of an externally fixed query budget, the search dynamically tracks posterior mass concentration $\max_h \text{posterior}(h)$ and halts as soon as the posterior on an already-evaluated candidate crosses a calibrated certainty threshold $\tau$.
* **Calibration**: $\tau$ is calibrated on a **strictly disjoint seed range** to avoid data snooping.
* **Outcome**: Reaches 97.3% agreement at 64.3% median queries on TinyDES-24. *(Paper finding: Fixed budget mode is preferred because ~40% of trials at low data budgets have no detectable key signal, which causes threshold stopping rules to over-query hopeless trials).*

---

### Approach 3: Exhaustive Candidate Scan (Hou et al. 2025 Algorithm 2 Baseline)
* **Paper Section**: **Section III-A & III-B** | **Code**: [`src/attacks/multi_bit_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_bruteforce.py)
* **Description**: The benchmark baseline from the target paper. Iterates through all $L = 4096$ candidate subkeys in arbitrary order, scoring each against the complete $N \cdot t$ plaintext dataset with the Combined-Response Distinguisher (CRD) $S(gk) = |w_0(gk)|$, returning $\text{argmax}_{gk} S(gk)$.
* **Outcome**: 72.0% success rate on TinyDES-24, 58.3% on real DES; always consumes **100% of candidate queries (4096 / 4096)**.

---

### Approach 4: Classical Matsui Algorithm 2 Baseline (1994)
* **Paper Section**: **Section II-C, VI-A & Table VII** | **Code**: [`src/distinguisher/crd.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/crd.py)
* **Description**: Matsui's classical linear key recovery algorithm using parity counters $T(gk) = |N(gk) - N/2|$. Implemented via an 8192-cell joint histogram for fast ground truth comparison.
* **Outcome**: Achieves 72.0%–96.0% success across normalized data complexities. Provides the classical cryptanalytic reference ceiling.

---

### Approach 5: Uniform Random Sampling Control
* **Paper Section**: **Section V-B & VI-B** | **Code**: [`experiments/exp2_guided_vs_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp2_guided_vs_bruteforce.py)
* **Description**: Evaluates candidates chosen uniformly at random without replacement at the **exact same query budget** as the guided search (no model, no search ordering).
* **Outcome**: At 54.4% budget (2227 queries), achieves only **38.7% success** (54.7% agreement), confirming that the guided search's 72.7% success is driven by the S-box response model rather than query count alone.

---

### Approach 6: Sequential Scan with Early Stopping Control
* **Paper Section**: **Section V-B & VI-B** | **Code**: [`src/attacks/multi_bit_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_bruteforce.py)
* **Description**: Tests candidates in static index order ($0, 1, 2, \dots$) and applies a theoretical score threshold stopping condition $z = \sqrt{2 \ln |GK|}$ (stopping early with no spatial model).
* **Outcome**: Achieves only **7.3% success** (8.0% agreement), demonstrating that naive early stopping without a wrong-key spatial model fails catastrophically.

---

### Approach 7: Generic Bayesian Optimization (`scikit-optimize` Gaussian Process) — *Negative Result*
* **Paper Section**: **Section V-B & VIII** | **Code**: [`experiments/exp2_guided_vs_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp2_guided_vs_bruteforce.py)
* **Description**: Applies off-the-shelf Gaussian Process (GP) regression over the 12-dimensional key bit hypercube ($k_1, \dots, k_{12} \in \{0,1\}^{12}$) without domain-specific S-box knowledge.
* **Outcome**: **9× slower than exhaustive brute force** (51.76 s/attack vs 5.85 s for brute force) due to $O(n^3)$ GP kernel refits after every query, achieving only **10.0% success**.

---

### Approach 8: Covering Probe Design Search — *Negative Result*
* **Paper Section**: **Section VIII & Table VI** | **Code**: [`experiments/exp5_opening_design.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp5_opening_design.py)
* **Description**: Pre-selects a minimal covering set of probe candidates offline (e.g., 256 probes covering the space at correlation radius 0.25) to evaluate during the opening phase before exploiting the posterior.
* **Outcome**: Performs worse than random opening at small budgets. *Paper finding: The bottleneck in linear cryptanalysis is not breadth ($\sum_h g_h^2 = 14.7$ direct-test-equivalents per probe), but contrast—neighbouring hypotheses produce nearly identical responses that require local exploitation.*

---

### Approach 9: Distinguisher Weight Averaging (EMA) — *Pipeline Repair 1*
* **Paper Section**: **Section VII-A & Table VI** | **Code**: [`src/distinguisher/model.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/model.py), [`src/distinguisher/train.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/train.py)
* **Description**: Replaces standard SGD checkpointing with an Exponential Moving Average (EMA) of network weights across the second half of training.
* **Outcome**: Removes gradient noise from logit magnitudes without changing the decision boundary. Popcount logit variance ($R^2$) increases from **0.68 to 0.94**, signal retention rises from **0.83 to 0.97**, and multi-bit attack success jumps from **37.5% to 87.5%**.

---

### Approach 10: Antisymmetric Logit Differential Scoring — *Pipeline Repair 2*
* **Paper Section**: **Section VII-B & Table VII** | **Code**: [`src/distinguisher/crd.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/crd.py)
* **Description**: Replaces the one-sided log-odds decision rule with an antisymmetric logit difference:
  $$\text{Score}_{\text{corrected}} = \frac{1}{2} \sum_{i=1}^N [\text{logit}(X_i) - \text{logit}(\mathbf{1} - X_i)]$$
* **Outcome**: Eliminates systematic network offset ($+2.8 \times 10^{-3}$/sample) that otherwise accumulates with $N$, recovering **+13.3 pp accuracy** on approximation $L_4$ (83.7% $\to$ 97.0%).

---

## 3. Comparative Results Table Across All Approaches (Paper Section VI-B)

| Approach | Category | Success Rate | Agreement with Full Scan | Median Queries | % of Space | Eval Speedup | Wall-Clock Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exhaustive Scan (Hou et al.)** | Baseline | 72.0% $\pm$ 7.2 | 100.0% | 4096 | 100.0% | $1.00\times$ | 5.85 s |
| **Guided WKR Budget (Ours)** | **Proposed** | **72.7% $\pm$ 7.1** | **98.7%** | **2227** | **54.4%** | **$1.88\times$** | **3.11 s** |
| **Guided WKR Calibrated Stop** | **Proposed** | 72.7% $\pm$ 7.1 | 97.3% | 2634 | 64.3% | $1.61\times$ | 3.64 s |
| **Uniform Random Control** | Control | 38.7% $\pm$ 7.8 | 54.7% | 2227 | 54.4% | $1.88\times$ | 2.18 s |
| **Sequential + Early Stop** | Control | 7.3% $\pm$ 4.2 | 8.0% | 204 | 5.0% | $15.02\times$ | 0.39 s |
| **Generic BO (`scikit-optimize`)**| Control (BO) | 10.0% $\pm$ 18.6 | 10.0% | 99 | 2.4% | $0.11\times$ | 51.76 s |
| **Real DES Guided WKR (90% target)** | **Real DES** | **80.0%** | **92.0%** | **418** | **10.2%** | **$9.80\times$** | **0.90 s** |

---

## 4. Paper to Code Mapping

| Paper Section | Topic & Content | Source Code | Result Artifact |
| :--- | :--- | :--- | :--- |
| **Section I & III** | Baseline & Complexity Analysis | [`src/attacks/multi_bit_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_bruteforce.py) | [`results/exp1_multi_bit.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp1_multi_bit.csv) |
| **Section IV-A** | Closed-Form $\rho_f, \rho_b$ Profiles | [`src/attacks/wrong_key_profile.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/wrong_key_profile.py) | [`results/figures/fig1_wrong_key_profile.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig1_wrong_key_profile.png) |
| **Section IV-B** | WKR Bayesian Search Algorithm | [`src/attacks/multi_bit_guided.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_guided.py) | [`results/exp2_summary.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp2_summary.csv) |
| **Section V** | Cipher & Mask Verification | [`src/ciphers/toy_feistel.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/ciphers/toy_feistel.py), [`src/ciphers/des.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/ciphers/des.py) | `tests/test_cipher_and_masks.py` |
| **Section VI-B** | Head-to-Head Benchmark | [`experiments/exp2_guided_vs_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp2_guided_vs_bruteforce.py) | [`results/exp2_per_trial.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp2_per_trial.csv) |
| **Section VI-C** | Budget Sensitivity Curve | [`experiments/exp3_budget_sensitivity.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp3_budget_sensitivity.py) | [`results/figures/fig3_budget_sensitivity.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig3_budget_sensitivity.png) |
| **Section VI-D** | Real 8-Round DES Attack | [`experiments/exp4_phase2_des_reduced_round.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp4_phase2_des_reduced_round.py) | [`results/figures/fig6_phase2_des.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig6_phase2_des.png) |
| **Section VII** | Distinguisher EMA & Antisymmetry | [`src/distinguisher/model.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/model.py), [`src/distinguisher/crd.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/crd.py) | [`results/exp1_one_bit.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp1_one_bit.csv) |
| **Section VIII** | Negative Results (BO & Probes) | [`experiments/exp5_opening_design.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp5_opening_design.py) | [`results/exp5_opening_design.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp5_opening_design.csv) |

---

## 5. Reproduction & Verification

```bash
cd ml-linear-cryptanalysis-fast-keyrecovery
pip install -r requirements.txt

# Run demo (~10 min, runs all pipeline stages):
python run_all.py --preset demo

# Run all 30 automated test assertions:
python -m tests.test_cipher_and_masks
python -m tests.test_attacks
```

---

## 6. Primary Evaluation Documents

* 📄 **[`RESEARCH_PAPER.md`](RESEARCH_PAPER.md)** — Full IEEE-formatted research paper (The document to be evaluated).
* 📋 **[`report/FACULTY_SUMMARY.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/FACULTY_SUMMARY.md)** — Faculty review summary.
* 🔍 **[`report/approach_and_findings.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/approach_and_findings.md)** — Detailed approach walkthrough and error analysis.
