# Faster Key Recovery for ML-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Search

[![Coursework](https://img.shields.io/badge/Coursework-CNS--Project-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Paper](https://img.shields.io/badge/Research--Paper-RESEARCH__PAPER.md-orange.svg)](report/RESEARCH_PAPER.md)
[![Verification](https://img.shields.io/badge/Tests-30%20Passing-brightgreen.svg)](#)

This directory contains the core implementation, experimental scripts, result artifacts, and evaluation documents for the research paper:

> **"Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search"**  
> 📄 **Primary Paper for Evaluation**: [`report/RESEARCH_PAPER.md`](report/RESEARCH_PAPER.md) *(also at repo root: [`../RESEARCH_PAPER.md`](../RESEARCH_PAPER.md))*

---

## 1. Problem Formulation (Paper Section I & III)

Linear cryptanalysis (Matsui, 1993) recovers subkey bits by finding linear approximations $\alpha \cdot P \oplus \beta \cdot C = \gamma \cdot K$ holding with probability $p_r \neq 1/2$. In *Cybersecurity* 2025, Hou, Ren, and Chen (HRC) recast this as distinguishing Bernoulli distributions using a neural network and Combined-Response Distinguisher (CRD).

### The Exhaustive Bottleneck
For an 8-round DES attack, recovering multiple subkey bits simultaneously (Algorithm 2 in HRC) wraps the approximation in one extra round at each boundary ($k_f$ at Round 0, $k_b$ at Round 7), creating **$L = 2^{12} = 4096$ candidate subkeys**. Algorithm 2 evaluates **every candidate exhaustively** on the full $N \cdot t$ plaintext dataset ($2^{12} \times N \times t / 8$ DES encryptions).

---

## 2. Every Approach Investigated in the Research Paper

The research paper rigorously develops, implements, and evaluates **7 distinct approaches**:

### 1. Wrong-Key-Response-Guided Bayesian Search (Fixed Budget Mode) — *Proposed Primary*
* **Paper Reference**: Section IV-A, IV-B, VI-B | **Source**: [`src/attacks/multi_bit_guided.py`](src/attacks/multi_bit_guided.py)
* **Mathematical Basis**: Wrong key guesses exhibit a deterministic correlation profile that factorises into closed-form S-box kernels:
  $$\text{corr}(gk) = \text{corr}(gk^*) \cdot \rho_f(k_f \oplus k_f^*) \cdot \rho_b(k_b \oplus k_b^*)$$
  where $\rho_f, \rho_b$ are computed offline in $<1\text{ ms}$ before spending any queries.
* **Mechanism**: Maintains Bayesian accumulators ($P, Q, V$) in $O(|GK|)$ time ($<0.2\text{ ms}$ per query), querying the candidate with the highest posterior until budget $B$ is reached.
* **Outcome**: Reaches **98.7% agreement** with full scan on TinyDES-24 using only **54.4% of queries (+0.7 pp success)**, and **92.0% agreement** on Real 8-round DES using only **10.2% of queries (418 / 4096)**.

### 2. Wrong-Key-Response-Guided Bayesian Search (Calibrated Stopping Mode) — *Proposed Variant*
* **Paper Reference**: Section IV-B, VI-B, VII | **Source**: [`src/attacks/multi_bit_guided.py`](src/attacks/multi_bit_guided.py), [`experiments/calibrate_stopping.py`](experiments/calibrate_stopping.py)
* **Mechanism**: Dynamically terminates when the posterior on an evaluated candidate crosses a threshold $\tau$ calibrated on an independent seed range.
* **Outcome**: Reaches 97.3% agreement at 64.3% queries on TinyDES-24. *(Paper finding: Fixed budget mode dominates calibrated stopping because stopping rules over-query hopeless trials where no key signal exists).*

### 3. Exhaustive Candidate Scan (Hou et al. 2025 Algorithm 2) — *Baseline*
* **Paper Reference**: Section III-A, III-B | **Source**: [`src/attacks/multi_bit_bruteforce.py`](src/attacks/multi_bit_bruteforce.py)
* **Mechanism**: Tests all 4096 candidate subkeys across all $N \cdot t$ samples using the Combined-Response Distinguisher (CRD) $S(gk) = |w_0(gk)|$.
* **Outcome**: 72.0% success rate on TinyDES-24, 58.3% on real DES; always spends 100% of candidate queries (4096 / 4096).

### 4. Classical Matsui Algorithm 2 — *Classical Reference*
* **Paper Reference**: Section II-C, VI-A | **Source**: [`src/distinguisher/crd.py`](src/distinguisher/crd.py)
* **Mechanism**: Classical linear cryptanalysis parity counter evaluated via an 8192-cell joint histogram.
* **Outcome**: Provides the classical cryptanalytic benchmark ceiling (72.0%–96.0% success).

### 5. Uniform Random Sampling Control — *Control 1*
* **Paper Reference**: Section V-B, VI-B | **Source**: [`experiments/exp2_guided_vs_bruteforce.py`](experiments/exp2_guided_vs_bruteforce.py)
* **Mechanism**: Uniform candidate sampling without replacement at the exact same query budget as Guided WKR (no model, no ordering).
* **Outcome**: Achieves only **38.7% success** (54.7% agreement) at 54.4% budget, proving the gain comes from the S-box response model rather than query reduction alone.

### 6. Sequential Scan with Early Stopping Control — *Control 2*
* **Paper Reference**: Section V-B, VI-B | **Source**: [`src/attacks/multi_bit_bruteforce.py`](src/attacks/multi_bit_bruteforce.py)
* **Mechanism**: Index-order candidate scan with early stopping rule $z = \sqrt{2 \ln |GK|}$ (stopping early without a spatial model).
* **Outcome**: Achieves only **7.3% success** (8.0% agreement), showing naive early stopping fails without domain structure.

### 7. Generic Bayesian Optimization (`scikit-optimize` GP) — *Negative Result 1*
* **Paper Reference**: Section V-B, VIII | **Source**: [`experiments/exp2_guided_vs_bruteforce.py`](experiments/exp2_guided_vs_bruteforce.py)
* **Mechanism**: Gaussian Process regression treating the 12 key bits as a black-box hypercube.
* **Outcome**: **9× slower than exhaustive scan** (51.76 s vs 5.85 s) due to $O(n^3)$ GP kernel refits; achieves only 10.0% success.

### 8. Covering Probe Design Search — *Negative Result 2*
* **Paper Reference**: Section VIII | **Source**: [`experiments/exp5_opening_design.py`](experiments/exp5_opening_design.py)
* **Mechanism**: Offline selection of a minimal covering set of probe candidates before Bayesian exploitation.
* **Outcome**: Performs worse than random opening at small budgets due to the contrast bottleneck in linear cryptanalysis.

### 9. Distinguisher Weight Averaging (EMA) — *Pipeline Repair 1*
* **Paper Reference**: Section VII-A | **Source**: [`src/distinguisher/model.py`](src/distinguisher/model.py)
* **Mechanism**: Averages weights over the second half of training to remove SGD logit noise.
* **Outcome**: Increases popcount logit variance explained ($R^2$) from **0.68 to 0.94**, boosting multi-bit attack success from **37.5% to 87.5%**.

### 10. Antisymmetric Logit Differential Scoring — *Pipeline Repair 2*
* **Paper Reference**: Section VII-B | **Source**: [`src/distinguisher/crd.py`](src/distinguisher/crd.py)
* **Mechanism**: Corrects the one-bit decision rule using $\frac{1}{2}\sum [\text{logit}(x) - \text{logit}(1-x)]$.
* **Outcome**: Eliminates systematic network offset (+2.8e-3/sample), recovering **+13.3 pp accuracy** on approximation $L_4$.

---

## 3. Comparative Results (Paper Table II)

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

## 4. Code & Paper Cross-Reference

| Paper Section | Topic | Code Implementation | Result Artifact |
| :--- | :--- | :--- | :--- |
| **Section I & III** | Baseline Algorithm 2 | [`src/attacks/multi_bit_bruteforce.py`](src/attacks/multi_bit_bruteforce.py) | [`results/exp1_multi_bit.csv`](results/exp1_multi_bit.csv) |
| **Section IV-A** | Closed-Form Profiles $\rho_f, \rho_b$ | [`src/attacks/wrong_key_profile.py`](src/attacks/wrong_key_profile.py) | [`results/figures/fig1_wrong_key_profile.png`](results/figures/fig1_wrong_key_profile.png) |
| **Section IV-B** | WKR Bayesian Search | [`src/attacks/multi_bit_guided.py`](src/attacks/multi_bit_guided.py) | [`results/exp2_summary.csv`](results/exp2_summary.csv) |
| **Section V** | Ciphers (TinyDES & DES) | [`src/ciphers/toy_feistel.py`](src/ciphers/toy_feistel.py), [`src/ciphers/des.py`](src/ciphers/des.py) | `tests/test_cipher_and_masks.py` |
| **Section VI-B** | Head-to-Head Comparison | [`experiments/exp2_guided_vs_bruteforce.py`](experiments/exp2_guided_vs_bruteforce.py) | [`results/exp2_per_trial.csv`](results/exp2_per_trial.csv) |
| **Section VI-C** | Budget Sensitivity Curve | [`experiments/exp3_budget_sensitivity.py`](experiments/exp3_budget_sensitivity.py) | [`results/figures/fig3_budget_sensitivity.png`](results/figures/fig3_budget_sensitivity.png) |
| **Section VI-D** | Real 8-Round DES Attack | [`experiments/exp4_phase2_des_reduced_round.py`](experiments/exp4_phase2_des_reduced_round.py) | [`results/figures/fig6_phase2_des.png`](results/figures/fig6_phase2_des.png) |
| **Section VII** | Distinguisher EMA & Antisymmetry | [`src/distinguisher/model.py`](src/distinguisher/model.py), [`src/distinguisher/crd.py`](src/distinguisher/crd.py) | [`results/exp1_one_bit.csv`](results/exp1_one_bit.csv) |
| **Section VIII** | Negative Results (BO & Probes) | [`experiments/exp5_opening_design.py`](experiments/exp5_opening_design.py) | [`results/exp5_opening_design.csv`](results/exp5_opening_design.csv) |

---

## 5. Quickstart & Reproduction

```bash
pip install -r requirements.txt

# Run demo (~10 min):
python run_all.py --preset demo

# Run all 30 automated test assertions:
python -m tests.test_cipher_and_masks
python -m tests.test_attacks
```
