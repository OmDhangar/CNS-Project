# CNS Project

Coursework repository for Cryptography & Network Security.

## Faster key recovery for ML-aided linear cryptanalysis

**→ [`ml-linear-cryptanalysis-fast-keyrecovery/`](ml-linear-cryptanalysis-fast-keyrecovery/)**

An extension of

> Zezhou Hou, Jiongjiong Ren, Shaozhen Chen.
> **"Improved machine learning-aided linear cryptanalysis: application to DES."**
> *Cybersecurity* 8:22 (2025). <https://doi.org/10.1186/s42400-024-00327-4>

The paper recovers subkey bits by training a neural network to distinguish two
Bernoulli distributions, then **exhaustively testing all 4096 candidate
subkeys** — its own conclusion names a better key-recovery strategy as open
work. This project replaces that brute-force loop, and nothing else: the same
network, the same scoring rule, the same data, only a different search order.

The idea it rests on is that a *wrong* key guess is not random. Its strength
factorises into two terms that depend only on the cipher's S-boxes and the
approximation's masks — so the shape of the search landscape can be computed in
closed form, **offline, before a single query is spent**.

**Headline:** the guided search returns the same subkey as the full 4096-candidate
scan **98.7%** of the time using **54%** of the evaluations — and **92%** of the
time using **10%** of them on real 8-round DES. Uniform random sampling at the
identical budget manages 54.7%; a generic Bayesian optimiser is *9x slower than
the brute-force scan it is meant to replace*.

### Where to start

| Document | What it is |
|---|---|
| [`report/FACULTY_SUMMARY.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/FACULTY_SUMMARY.md) | **Start here.** The algorithm stated precisely, the evaluation methodology, all result tables, the improvements made, and the negative results |
| [`report/approach_and_findings.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/approach_and_findings.md) | Plain-language walkthrough: the idea, comparison tables against the paper's own numbers, and an honest account of what we got wrong |
| [`report/project_report.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/project_report.md) | The results document, regenerated from `results/` so no number is hand-copied |
| [`README.md`](ml-linear-cryptanalysis-fast-keyrecovery/README.md) | Code layout, design commitments, how to reproduce |

### Reproducing

```bash
cd ml-linear-cryptanalysis-fast-keyrecovery
pip install -r requirements.txt
python run_all.py --preset demo       # ~10 min, shows the pipeline running
python run_all.py --preset full       # ~3 h, the trial counts the report quotes
```

Raising the trial count does not make the method perform worse — it makes the
measurement honest. At 40 trials per point the success curve wobbles by ±20
percentage points near its steep region; the reported numbers use 60–150.
