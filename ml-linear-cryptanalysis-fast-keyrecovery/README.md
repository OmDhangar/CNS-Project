# Faster Key Recovery for ML-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Search

[![Coursework](https://img.shields.io/badge/Coursework-CNS--Project-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Paper](https://img.shields.io/badge/Research--Paper-RESEARCH__PAPER.md-orange.svg)](report/RESEARCH_PAPER.md)
[![Verification](https://img.shields.io/badge/Tests-30%20Passing-brightgreen.svg)](#)

This directory contains the core implementation, experimental scripts, result artifacts, and evaluation documents for the research paper:

> **"Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search"**  
> 📄 **Primary Paper for Evaluation**: [`report/RESEARCH_PAPER.md`](report/RESEARCH_PAPER.md) *(also available at repo root: [`../RESEARCH_PAPER.md`](../RESEARCH_PAPER.md))*

---

## 1. Context & Motivation (Paper Section I & III)

Linear cryptanalysis (Matsui, 1993) recovers key bits from a linear approximate expression $\alpha \cdot P \oplus \beta \cdot C = \gamma \cdot K$ holding with probability $p_r \neq 1/2$. 

In *Cybersecurity* 2025, Hou, Ren, and Chen (HRC) recast this as distinguishing two Bernoulli distributions ($\text{Bern}(p_r)$ vs $\text{Bern}(1-p_r)$) using a neural network and Combined-Response Distinguisher (CRD).

### The Bottleneck in HRC Algorithm 2
To recover multiple subkey bits simultaneously on an 8-round DES attack, Algorithm 2 wraps the approximation in an extra cipher round at each end and **brute-forces all $L = 2^{12} = 4096$ candidate subkeys**, incurring $2^{12} \times N \times t / 8$ DES encryptions. Their conclusion named this as open work:
> *"Designing a better key-recovery strategy matching neural distinguishers will be effective to reduce the time complexity."*

---

## 2. Our Core Contribution (Paper Section IV)

This project replaces the exhaustive candidate loop with an informed, model-based search while keeping the neural distinguisher, CRD rule, and dataset unchanged.

### The Closed-Form S-Box Response Kernel
A wrong key guess in linear cryptanalysis is **not random noise**. The correlation factorises into independent S-box response kernels:
$$\text{corr}(gk) = \text{corr}(gk^*) \cdot \rho_f(k_f \oplus k_f^*) \cdot \rho_b(k_b \oplus k_b^*)$$
where $\rho_f(\Delta)$ and $\rho_b(\Delta)$ depend **exclusively on the cipher's S-box lookup tables and linear masks**:
$$\rho(\Delta) = \frac{1}{2^m} \sum_{u \in \{0,1\}^m} (-1)^{\nu \cdot [S(u) \oplus S(u \oplus \Delta)]}$$

* **Precomputation Cost**: Computed offline in $2 \times 64 \times 64 = 8192$ operations ($<1\text{ ms}$) **before spending a single plaintext query**.
* **Bayesian Update**: Maintains accumulators $\mathbf{P}, \mathbf{Q}, \mathbf{V}$ in $O(|GK|)$ time ($<0.2\text{ ms}$ on CPU), far cheaper than a single neural network inference pass.

---

## 3. Results Summary (Paper Section VI)

### TinyDES-24 (8 rounds, $|GK|=4096$, 150 independent trials)
* **Exhaustive Scan (HRC)**: $72.0\% \pm 7.2$ success, 4096 evaluations ($1.00\times$).
* **Guided WKR (Ours)**: **$72.7\% \pm 7.1$ success (+0.7 pp)**, **$98.7\%$ agreement** with exhaustive scan using only **2227 evaluations (54.4% of space)** ($1.88\times$ speedup).
* **Random Sampling Control**: $38.7\%$ success ($54.7\%$ agreement) at the identical budget.
* **Generic BO (`scikit-optimize`) Control**: $10.0\%$ success, $9\times$ slower than brute force due to $O(n^3)$ GP kernel updates.

### Real 8-Round DES (FIPS test vector verified, 60 trials)
* **Exhaustive Scan**: $58.3\%$ success, 4096 evaluations.
* **Guided WKR (90% Agreement Target)**: **$80.0\%$ success**, **$92.0\%$ agreement** using only **418 evaluations (10.2% of space)**.
* **Guided WKR (95% Agreement Target)**: **$76.0\%$ success**, **$100.0\%$ agreement** using **1660 evaluations (40.5% of space)**.

### Repaired Baseline Pipeline Defects (Paper Section VII)
1. **SGD Logit Variance**: Replaced standard training with Exponential Moving Average (EMA) weight averaging, lifting logit variance explained by the sufficient statistic ($R^2$) from $0.68 \to 0.94$, signal retention from $0.83 \to 0.97$, and attack success from $37.5\% \to 87.5\%$.
2. **One-Bit Decision Asymmetry**: Replaced $\sum \log_2(v/(1-v))$ with antisymmetric $[\text{logit}(x) - \text{logit}(1-x)]/2$, eliminating systematic network bias and recovering $+13.3\text{ pp}$ on approximation $L_4$.

---

## 4. Repository Structure

```
ml-linear-cryptanalysis-fast-keyrecovery/
├── report/
│   ├── RESEARCH_PAPER.md           <- Complete IEEE-style Research Paper (Evaluated document)
│   ├── FACULTY_SUMMARY.md          <- Evaluation summary: algorithm, methodology, result tables
│   ├── approach_and_findings.md    <- Narrative walkthrough & error analysis
│   ├── project_report.md           <- Auto-generated report built from results/ CSVs
│   └── build_report.py             <- Script generating project_report.md
├── src/
│   ├── ciphers/
│   │   ├── toy_feistel.py          <- TinyDES-24: vectorised Feistel cipher + mask algebra
│   │   └── des.py                  <- Real DES (reduced-round), FIPS test vector verified
│   ├── linear_analysis/
│   │   ├── bias_search.py          <- LAT + beam search over linear trails + Monte Carlo check
│   │   ├── approximations.py       <- Approximation tables for TinyDES-24 (L3-L6)
│   │   └── known_masks_des.py      <- Matsui's L3/L5/L6 masks with orientation resolved
│   ├── distinguisher/
│   │   ├── model.py                <- PyTorch Residual MLP with EMA weight averaging
│   │   ├── crd.py                  <- CRD score rule, CandidateScorer, and Matsui classical scorer
│   │   ├── data_gen.py             <- Parity sample generator and Bayes-optimal references
│   │   └── train.py                <- Distinguisher training loop with disk caching
│   └── attacks/
│   │   ├── candidate_space.py      <- Multi-bit attack geometry and candidate representations
│   │   ├── wrong_key_profile.py    <- Closed-form S-box profile computation (rho_f, rho_b)
│   │   ├── multi_bit_bruteforce.py <- HRC Algorithm 2 (exhaustive baseline)
│   │   └── multi_bit_guided.py     <- Guided Bayesian search (our contribution)
├── experiments/
│   ├── exp1_reproduce_baseline.py  <- Baseline reproduction and operating point selection
│   ├── exp2_guided_vs_bruteforce.py<- Headline head-to-head evaluation (Table II)
│   ├── exp3_budget_sensitivity.py  <- Budget curve evaluation (Table III)
│   ├── exp4_phase2_des_reduced_round.py <- Real 8-round DES evaluation (Table IV & V)
│   ├── exp5_opening_design.py      <- Covering probe design evaluation (negative result)
│   ├── calibrate_stopping.py       <- Stopping threshold calibration on disjoint seeds
│   └── make_plots.py               <- Figure generation
├── tests/
│   ├── test_cipher_and_masks.py    <- 15 tests: DES FIPS vectors, mask algebra, LAT
│   └── test_attacks.py             <- 15 tests: Scorer bit-exactness, profile fidelity, R^2
├── results/                        <- CSV logs and generated figures (PNG)
├── requirements.txt                <- Python dependencies
└── run_all.py                      <- Master execution runner (--preset demo | standard | full)
```

---

## 5. Execution & Reproduction Guide

### Environment Setup
```bash
pip install -r requirements.txt
```

### Reproducing Experiments
```bash
# 1. Quick demo (~10 min, runs all pipeline stages on small trial count):
python run_all.py --preset demo

# 2. Standard run (~45 min):
python run_all.py --preset standard

# 3. Full reproduction (~3 hours, matches exact paper trial counts):
python run_all.py --preset full
```

### Step-by-Step Execution
```bash
# Verify invariants and cipher correctness:
python -m tests.test_cipher_and_masks
python -m tests.test_attacks

# Run individual experiments:
python experiments/exp1_reproduce_baseline.py
python experiments/calibrate_stopping.py
python experiments/exp2_guided_vs_bruteforce.py --trials 100
python experiments/exp3_budget_sensitivity.py --trials 60
python experiments/exp4_phase2_des_reduced_round.py --config l6 --trials 60
python experiments/exp5_opening_design.py --trials 50

# Generate figures and build report:
python experiments/make_plots.py
python report/build_report.py
```

---

## 6. Evaluation Verification Checklist

- [x] **Primary Research Paper**: Full IEEE manuscript in [`report/RESEARCH_PAPER.md`](report/RESEARCH_PAPER.md).
- [x] **Theoretical Derivation**: Closed-form S-box correlation profile proof ($\rho_f, \rho_b$) in Section IV-A.
- [x] **Implementation Fidelity**: Shared `CandidateScorer` object guarantees identical scoring between baseline and search.
- [x] **Cipher Invariants**: Real DES verified against FIPS 46-3 (`DES(0x0123456789ABCDEF, 0x133457799BBCDFF1) = 0x85E813540F0AB405`).
- [x] **Reproducibility**: Master script `run_all.py` and automated test suite with 30 passing assertions.
