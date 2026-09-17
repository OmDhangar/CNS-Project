"""Algorithm 2 of Hou, Ren & Chen (2025), implemented verbatim.

    Input : the trained neural distinguisher ND_r^t, N*t plaintext/ciphertext
            pairs, and the candidate subkey set GK.
    Output: the recovered subkey and the value of gamma.K'.

    score_max <- -inf
    for gk in GK:
        use gk to encrypt P to P' and decrypt C to C'
        form the N samples of dimension t from alpha.P' ^ beta.C' ^ gamma.K'
        w0 <- CRD of the samples with the guess gamma.K' = 0
        w1 <- CRD of the samples with the guess gamma.K' = 1
        if max(w0, w1) > score_max:
            score_max <- max(w0, w1);  RK <- gk;  gamma.K' <- argmax
    return RK, gamma.K'

This is the control group of the whole project: every candidate is tested, in
index order, with no early exit.  Its cost is exactly |GK| distinguisher
evaluations -- the paper's stated time complexity of 2**12 * N * t / 8 DES
executions for the 8-round attack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackResult:
    """Outcome of one key-recovery attack, in the form both searches report."""

    method: str
    best_candidate: int
    best_score: float
    gamma_guess: int
    true_candidate: int
    true_gamma: int
    n_evals: int                 # unique distinguisher evaluations used
    n_calls: int                 # including cache hits
    seconds: float
    candidate_space: int
    scores: np.ndarray = None    # full landscape if available (baseline only)
    visited: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.best_candidate == self.true_candidate

    @property
    def gamma_correct(self):
        return self.gamma_guess == self.true_gamma

    @property
    def query_fraction(self):
        return self.n_evals / self.candidate_space

    def rank_of_true(self):
        """Rank of the correct subkey in the score ordering (0 = top).

        Only defined when the full landscape is available.
        """
        if self.scores is None:
            return None
        order = np.argsort(-self.scores, kind="stable")
        return int(np.flatnonzero(order == self.true_candidate)[0])


def run_exhaustive(scorer, setup, data, keep_scores=True):
    """Algorithm 2, exhaustive over GK."""
    scorer.reset()
    n = setup.n_candidates
    scores = np.empty(n) if keep_scores else None

    t0 = time.perf_counter()
    score_max = -np.inf
    best, gamma = -1, 0
    for gk in range(n):
        res = scorer(gk)
        if keep_scores:
            scores[gk] = res.score
        if res.score > score_max:
            score_max = res.score
            best = gk
            gamma = res.gamma_guess
    elapsed = time.perf_counter() - t0

    return AttackResult(
        method="exhaustive",
        best_candidate=best,
        best_score=score_max,
        gamma_guess=gamma,
        true_candidate=int(data.true_candidate),
        true_gamma=int(data.gamma_key_bit),
        n_evals=scorer.n_evals,
        n_calls=scorer.n_calls,
        seconds=elapsed,
        candidate_space=n,
        scores=scores,
    )
