# Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search

[![Coursework](https://img.shields.io/badge/Coursework-CNS--Project-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![Paper](https://img.shields.io/badge/Research--Paper-RESEARCH__PAPER.md-orange.svg)](RESEARCH_PAPER.md)
[![Verification](https://img.shields.io/badge/Tests-30%20Passing-brightgreen.svg)](#)

This repository contains the complete research paper, mathematical framework, source code, neural network models, and experimental evaluation for:

> **"Reducing Time Complexity in Machine-Learning-Aided Linear Cryptanalysis via Wrong-Key-Response-Guided Bayesian Key Search"**  
> 📄 **Full Paper**: [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) *(or in [`ml-linear-cryptanalysis-fast-keyrecovery/report/RESEARCH_PAPER.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/RESEARCH_PAPER.md))*

---

## 1. Executive Summary & Research Paper Alignment

This project resolves the primary open problem formulated in:
> Zezhou Hou, Jiongjiong Ren, Shaozhen Chen, *"Improved machine learning-aided linear cryptanalysis: application to DES"*, **Cybersecurity** 8:22 (2025).

### The Open Problem
HRC's multiple-bits key recovery procedure (**Algorithm 2**) brute-forces all $L = 2^{12} = 4096$ candidate subkeys for 8-round DES against the entire plaintext budget $N \cdot t$, incurring a time complexity of $2^{12} \times N \times t / 8$ DES encryptions. Their paper concluded with:
> *"Designing a better key-recovery strategy matching neural distinguishers will be effective to reduce the time complexity."*

### Our Algorithmic Solution
Wrong key guesses in linear cryptanalysis are **not random or uniform**—their correlation factorises into closed-form S-box response kernels:
$$\text{corr}(gk) = \text{corr}(gk^*) \cdot \rho_f(k_f \oplus k_f^*) \cdot \rho_b(k_b \oplus k_b^*)$$
where $\rho_f$ and $\rho_b$ depend **only on the cipher's S-boxes and linear masks**, computed offline in $<1\text{ ms}$ before spending a single plaintext query.

We replace the brute-force scan with a **Wrong-Key-Response-Guided Bayesian Search** that maintains exact posterior probabilities across all 4096 hypotheses with an $O(L)$ update taking $<0.2\text{ ms}$ on CPU (far less than one neural forward pass).

---

## 2. Headline Results (Paper Section VI)

| Cipher & Attack Target | Method | Success Rate | Agreement with Full Scan | Queries Used (Median) | Search Cost (% of Space) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TinyDES-24** (8-round, 150 trials) | **Exhaustive Scan (HRC)** | 72.0% $\pm$ 7.2 | 100.0% | 4096 | 100.0% | $1.00\times$ |
| | **Guided WKR (Ours)** | **72.7% $\pm$ 7.1** | **98.7%** | **2227** | **54.4%** | **$1.88\times$** |
| | Random (Matched Budget) | 38.7% $\pm$ 7.8 | 54.7% | 2227 | 54.4% | $1.88\times$ |
| | Generic BO (`scikit-optimize`) | 10.0% $\pm$ 18.6 | 10.0% | 99 | 2.4% | $0.11\times$ ($9\times$ slower!) |
| **Real 8-Round DES** (60 trials) | **Exhaustive Scan (HRC)** | 58.3% | 100.0% | 4096 | 100.0% | $1.00\times$ |
| | **Guided WKR (90% target)** | **80.0%** | **92.0%** | **418** | **10.2%** | **$9.80\times$** |
| | **Guided WKR (95% target)** | **76.0%** | **100.0%** | **1660** | **40.5%** | **$2.47\times$** |

### Key Takeaways
* **Real 8-Round DES**: Reaches **92.0% agreement** with the full 4096-candidate scan at **10.2% of evaluations (418 queries)**.
* **Accuracy Improvement**: On TinyDES-24, Guided WKR achieves **+0.7 pp higher success rate** than the full scan by avoiding spurious noise peaks.
* **Repaired Baseline Defects**:
  1. *Logit Noise*: Fixed SGD logit noise via EMA weight averaging ($R^2 = 0.68 \to 0.94$, signal retention $0.83 \to 0.97$, recovery rate $37.5\% \to 87.5\%$).
  2. *Asymmetric Scoring Offset*: Replaced one-bit decision rule with antisymmetric logit difference, recovering $+13.3\text{ pp}$ on $L_4$.

---

## 3. Paper to Codebase Navigation

| Paper Section | Topic | Codebase Implementation | Result Artifacts |
| :--- | :--- | :--- | :--- |
| **Section I & III** | Problem & Baseline | [`src/attacks/multi_bit_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_bruteforce.py) | [`results/exp1_multi_bit.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp1_multi_bit.csv) |
| **Section IV-A** | Closed-Form Profiles $\rho_f, \rho_b$ | [`src/attacks/wrong_key_profile.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/wrong_key_profile.py) | [`results/figures/fig1_wrong_key_profile.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig1_wrong_key_profile.png) |
| **Section IV-B** | WKR Bayesian Search | [`src/attacks/multi_bit_guided.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/attacks/multi_bit_guided.py) | [`results/exp2_summary.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp2_summary.csv) |
| **Section V** | Ciphers & Approximations | [`src/ciphers/toy_feistel.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/ciphers/toy_feistel.py), [`src/ciphers/des.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/ciphers/des.py) | `tests/test_cipher_and_masks.py` |
| **Section VI-B** | Head-to-Head Benchmark | [`experiments/exp2_guided_vs_bruteforce.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp2_guided_vs_bruteforce.py) | [`results/exp2_per_trial.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp2_per_trial.csv) |
| **Section VI-C** | Budget Sensitivity | [`experiments/exp3_budget_sensitivity.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp3_budget_sensitivity.py) | [`results/figures/fig3_budget_sensitivity.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig3_budget_sensitivity.png) |
| **Section VI-D** | Real 8-Round DES Attack | [`experiments/exp4_phase2_des_reduced_round.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp4_phase2_des_reduced_round.py) | [`results/figures/fig6_phase2_des.png`](ml-linear-cryptanalysis-fast-keyrecovery/results/figures/fig6_phase2_des.png) |
| **Section VII** | Baseline Repairs (EMA & Parity) | [`src/distinguisher/model.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/model.py), [`src/distinguisher/crd.py`](ml-linear-cryptanalysis-fast-keyrecovery/src/distinguisher/crd.py) | [`results/exp1_one_bit.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp1_one_bit.csv) |
| **Section VIII** | Negative Results (BO & Probes) | [`experiments/exp5_opening_design.py`](ml-linear-cryptanalysis-fast-keyrecovery/experiments/exp5_opening_design.py) | [`results/exp5_opening_design.csv`](ml-linear-cryptanalysis-fast-keyrecovery/results/exp5_opening_design.csv) |

---

## 4. Key Documentation for Evaluators

1. 📄 **[`RESEARCH_PAPER.md`](RESEARCH_PAPER.md)** — **The primary evaluation paper** in standard IEEE format with complete mathematical derivations, theorem proofs, and results.
2. 📋 **[`report/FACULTY_SUMMARY.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/FACULTY_SUMMARY.md)** — Faculty review summary detailing evaluation methodology, integrity checks, and result tables.
3. 🔍 **[`report/approach_and_findings.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/approach_and_findings.md)** — Narrative walkthrough of the cryptanalytic discovery, iterative development, and error analysis.
4. 📊 **[`report/project_report.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/project_report.md)** — Auto-generated results document built directly from `results/` CSV files.

---

## 5. Quickstart & Reproduction

### Prerequisites
* Python 3.10 or higher
* PyTorch, NumPy, SciPy, Matplotlib

```bash
cd ml-linear-cryptanalysis-fast-keyrecovery
pip install -r requirements.txt
```

### Running the Full Pipeline
```bash
# Run quick demo (~10 minutes, verifies all pipeline stages):
python run_all.py --preset demo

# Run standard evaluation (~45 minutes):
python run_all.py --preset standard

# Run full reproduction (3+ hours, exact trial counts reported in paper):
python run_all.py --preset full
```

### Running Automated Test Suite (30 Checks)
```bash
python -m tests.test_cipher_and_masks
python -m tests.test_attacks
```
The test suite validates:
* FIPS test-vector compliance for DES encryption and decryption.
* Bijectivity and exact linear approximation table (LAT) generation.
* Bit-exactness of fast-path and reference candidate scorers.
* S-box wrong-key response profile offline predictive fidelity.

---

## 6. License & Attribution

Coursework project for Cryptography & Network Security (CNS).  
Extending the work of Zezhou Hou, Jiongjiong Ren, and Shaozhen Chen (*Cybersecurity* 2025).
