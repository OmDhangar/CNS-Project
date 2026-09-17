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

**Headline:** ~85% of the 4096-candidate search skipped, at a cost of a few
percentage points of success rate. A generic Bayesian optimiser on the same
problem is *worse than the brute-force scan it replaces*.

### Where to start

| Document | What it is |
|---|---|
| [`report/approach_and_findings.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/approach_and_findings.md) | **Read this first.** Plain-language walkthrough: the idea, comparison tables against the paper's own numbers, and an honest account of what we got wrong |
| [`report/project_report.md`](ml-linear-cryptanalysis-fast-keyrecovery/report/project_report.md) | The results document, regenerated from `results/` so no number is hand-copied |
| [`README.md`](ml-linear-cryptanalysis-fast-keyrecovery/README.md) | Code layout, design commitments, how to reproduce |

### Reproducing

```bash
cd ml-linear-cryptanalysis-fast-keyrecovery
pip install -r requirements.txt
python run_all.py            # or --quick for a smoke test
```
