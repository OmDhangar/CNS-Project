"""Combined-response distinguisher (CRD), the paper's Eq. 2.

    v_crd = sum_i log2( v_i / (1 - v_i) )

Because ``log2(v / (1 - v)) = logit(v) / ln 2`` and our network emits logits
directly, the CRD is just the sum of the network's raw outputs rescaled -- no
sigmoid round-trip, so no saturation loss when v_i is close to 0 or 1.

This module also defines :class:`CandidateScorer`, the single object through
which *both* the exhaustive baseline and the guided search obtain a candidate's
score.  It owns the evaluation counter, which is the project's headline metric:
"distinguisher evaluations used to return an answer".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

LN2 = math.log(2.0)


def crd_from_logits(logits):
    """Eq. 2 applied to a vector of network logits."""
    return float(np.sum(logits)) / LN2


def one_bit_crd(model, samples, symmetric=True):
    """The one-bit key-recovery statistic of the paper's Step 4.

    The paper's literal rule is ``sum_i log2(v_i / (1 - v_i)) > 0  =>  gamma.K = 0``.
    That is the log-likelihood ratio between the two hypotheses only if the
    network is exactly antisymmetric under complementation, i.e. only if
    ``logit(1 - x) = -logit(x)``.  A trained network is not: measured on a
    6-round TinyDES-24 distinguisher, its mean logit on random input sits a few
    times 10^-3 away from zero.  Per sample that is negligible, but the CRD
    *sums* over N samples, so the offset grows like N while the signal it is
    competing with grows like sqrt(N).  At N of order 10^3 the offset is the
    same size as the signal and the rule starts losing accuracy.

    Setting ``symmetric`` uses

        sum_i [ logit(x_i) - logit(1 - x_i) ] / 2

    instead, which is the exact log-likelihood ratio between "these samples are
    omega" and "these samples are omega XOR 1", is antisymmetric by
    construction, and costs one extra forward pass and no extra plaintexts.
    It is the one-bit analogue of what Algorithm 2 already does in the
    multiple-bit setting when it takes the better of w0 and w1.

    Returns the statistic; positive means gamma.K = 0.
    """
    x = np.ascontiguousarray(samples, dtype=np.float32)
    if not symmetric:
        return crd_from_logits(model.logit(x).numpy())
    n = len(x)
    both = np.empty((2 * n, x.shape[1]), dtype=np.float32)
    both[:n] = x
    both[n:] = 1 - x
    with torch.no_grad():
        lg = model.forward(torch.from_numpy(both)).numpy()
    return float(np.sum(lg[:n] - lg[n:])) / (2 * LN2)


@dataclass
class ScoreResult:
    score: float          # max over the two guesses of gamma.K'
    gamma_guess: int      # the argmax, i.e. Algorithm 2's guess for gamma.K'
    score_gamma0: float
    score_gamma1: float


class CandidateScorer:
    """Scores a candidate subkey with the trained ND + CRD, counting calls.

    One call performs exactly the work Algorithm 2 does per candidate:
    partially encrypt/decrypt with the guess, form the N samples of dimension
    t, run the distinguisher, and combine the responses.  Because gamma.K' is
    unknown, both of its values are tried and the larger CRD kept -- that is
    the paper's `w0` / `w1` bookkeeping.
    """

    def __init__(self, model, setup, data, t, cache=True, batch_size=None,
                 persistent_cache=False):
        self.model = model
        self.setup = setup
        self.data = data
        self.t = int(t)
        self.n_samples = data.base.size // self.t
        if self.n_samples * self.t != data.base.size:
            raise ValueError(
                f"data size {data.base.size} is not a multiple of t = {self.t}"
            )
        self.batch_size = batch_size
        self.cache = {} if cache else None
        self.persistent_cache = bool(persistent_cache)
        # n_evals counts the DISTINCT candidates the current search asked about,
        # i.e. the number of distinguisher evaluations a real attacker would
        # perform.  It is deliberately independent of the memo cache, so an
        # experiment may share one cache across several search strategies (a
        # harness speed-up) without distorting the headline metric.
        self._seen = set()
        self.n_evals = 0
        self.n_calls = 0        # total calls, including repeats

        # Fast path: pre-materialise the two guessed F-terms and pre-allocate
        # the network's input buffer.  Verified against the readable
        # transform_bits definition by setup.self_test.
        setup.self_test(data)
        n_used = self.n_samples * self.t
        front, back = setup.precompute_streams(data)
        self._front = np.ascontiguousarray(front[:, :n_used])
        self._back = np.ascontiguousarray(back[:, :n_used])
        self._buf = np.empty((2 * self.n_samples, self.t), dtype=np.uint8)
        self._lo = self._buf[: self.n_samples].reshape(-1)
        self._hi = self._buf[self.n_samples:].reshape(-1)
        self._tensor_u8 = torch.from_numpy(self._buf)

    # -- bookkeeping -------------------------------------------------------
    def reset(self):
        self.n_evals = 0
        self.n_calls = 0
        self._seen.clear()
        if self.cache is not None and not self.persistent_cache:
            self.cache.clear()

    # -- scoring -----------------------------------------------------------
    def _raw(self, candidate):
        # Partially encrypt/decrypt with the guess and form the N samples of
        # dimension t.  The lower half of the buffer is the guess gamma.K' = 0,
        # the upper half its complement, i.e. the guess gamma.K' = 1.
        kf, kb = self.setup.split(candidate)
        np.bitwise_xor(self._front[kf], self._back[kb], out=self._lo)
        np.subtract(np.uint8(1), self._lo, out=self._hi)
        with torch.no_grad():
            logits = self.model.forward(self._tensor_u8.float()).numpy()
        s0 = crd_from_logits(logits[: self.n_samples])
        s1 = crd_from_logits(logits[self.n_samples:])
        if s0 >= s1:
            return ScoreResult(s0, 0, s0, s1)
        return ScoreResult(s1, 1, s0, s1)

    def __call__(self, candidate):
        candidate = int(candidate)
        self.n_calls += 1
        if candidate not in self._seen:
            self._seen.add(candidate)
            self.n_evals += 1
        if self.cache is not None:
            hit = self.cache.get(candidate)
            if hit is not None:
                return hit
        res = self._raw(candidate)
        if self.cache is not None:
            self.cache[candidate] = res
        return res

    def score(self, candidate):
        return self(candidate).score

    # -- convenience for analysis (NOT used inside the attacks) ------------
    def full_landscape(self):
        """All |GK| scores, vectorised.  Only for plots/diagnostics; calling it
        does not affect the evaluation counters used in the comparison."""
        saved = (self.n_evals, self.n_calls, set(self._seen))
        scores = np.empty(self.setup.n_candidates)
        gammas = np.empty(self.setup.n_candidates, dtype=np.int8)
        for c in range(self.setup.n_candidates):
            r = self.cache.get(c) if self.cache is not None else None
            if r is None:
                r = self._raw(c)
                if self.cache is not None:
                    self.cache[c] = r
            scores[c] = r.score
            gammas[c] = r.gamma_guess
        self.n_evals, self.n_calls, self._seen = saved[0], saved[1], saved[2]
        return scores, gammas


class ClassicalScorer:
    """Matsui's Algorithm 2 statistic |T - N/2|, for the classical baseline.

    Kept deliberately parallel to :class:`CandidateScorer` so the same attack
    driver can run either one.

    Implemented the textbook way: the counting statistic depends on the data
    only through the joint histogram of (base, g_front, g_back), which has just
    2 * 64 * 64 = 8192 cells.  Building it once turns each candidate's count
    into an 8192-term dot product instead of a pass over all N*t pairs, which
    is what makes a classical-versus-ML comparison over hundreds of trials
    affordable.  :meth:`self_test` checks it against the direct count.
    """

    def __init__(self, setup, data):
        self.setup = setup
        self.data = data
        self.n = data.base.size
        self._seen = set()
        self.n_evals = 0
        self.n_calls = 0

        idx = ((data.base.astype(np.int64) << 12)
               | (data.g_front.astype(np.int64) << 6)
               | data.g_back.astype(np.int64))
        self._hist = np.bincount(idx.ravel(), minlength=8192).astype(np.float64)
        # parity_table[v] for v = (base << 12) | (gf << 6) | gb, per candidate,
        # is base ^ front_lut[kf][gf] ^ back_lut[kb][gb]; the gf/gb part is an
        # outer XOR of the two 64-entry LUT rows.
        self.self_test()

    def _parity_table(self, candidate):
        kf, kb = self.setup.split(candidate)
        sub = (self.setup.front_lut[kf][:, None]
               ^ self.setup.back_lut[kb][None, :]).reshape(-1)   # 4096, base = 0
        return np.concatenate([sub, 1 - sub])                    # 8192

    def _count_zeros(self, candidate):
        return float(np.dot(self._hist, 1.0 - self._parity_table(candidate)))

    def self_test(self, candidates=(0, 1, 777, 4095)):
        for c in candidates:
            direct = float(np.count_nonzero(
                self.setup.transform_bits(self.data, c) == 0))
            if abs(self._count_zeros(c) - direct) > 0.5:
                raise AssertionError(
                    f"classical histogram disagrees with the direct count for {c}")
        return True

    def reset(self):
        self.n_evals = self.n_calls = 0
        self._seen.clear()

    def __call__(self, candidate):
        self.n_calls += 1
        if int(candidate) not in self._seen:
            self._seen.add(int(candidate))
            self.n_evals += 1
        dev = self._count_zeros(int(candidate)) - self.n / 2.0
        # Matsui guesses gamma.K' from the sign of the deviation and the sign
        # of (p_r - 1/2); the magnitude is the ranking statistic.
        p_gt_half = self.setup.appr.p > 0.5
        gamma = int(not ((dev > 0) == p_gt_half))
        return ScoreResult(abs(dev), gamma, dev, -dev)

    def score(self, candidate):
        return self(candidate).score
