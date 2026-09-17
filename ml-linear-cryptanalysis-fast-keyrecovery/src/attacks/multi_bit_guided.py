"""Guided key-recovery search -- this project's contribution.

Algorithm 2 of the paper scans every candidate subkey in ``GK`` and keeps the
one with the largest CRD score.  Its conclusion names the alternative as open
work: *"Designing a better key-recovery strategy matching neural distinguishers
will be effective to reduce the time complexity."*

This module replaces the ``for gk in GK`` loop with a sequential, model-based
search over the same candidate space, using the same trained distinguisher and
the same CRD scoring rule.  Four strategies are provided:

``wkr``         our main proposal: Bayesian search driven by the *wrong-key
                response profile*, which :mod:`wrong_key_profile` derives in
                closed form from the cipher's S-boxes before any query is spent.
``skopt``       scikit-optimize's Gaussian-process ``Optimizer`` over the 12
                binary decision variables; a generic black-box optimiser with
                no cryptanalytic prior.
``random``      uniform sampling without replacement -- the control that isolates
                the contribution of the model.
``sequential``  index-order scan plus the stopping rule -- the control that
                isolates the contribution of stopping early.

The WKR model
-------------
Writing ``g_h(k) = |rho_f(kf ^ kf_h) * rho_b(kb ^ kb_h)|`` for the predicted
relative score at ``k`` when the true key is ``h``, we model an observed score
as

    s_i = mu + sigma * A * g_h(k_i) + sigma * N(0, 1)

with ``mu`` and ``sigma`` estimated robustly (median / MAD) from the scores
seen so far and the peak height ``A`` marginalised over a grid.  The
log-likelihood of every hypothesis then depends on the observations only
through three accumulators,

    P(h) = sum_i s_i g_h(k_i),   Q(h) = sum_i g_h(k_i),   V(h) = sum_i g_h(k_i)^2

each updated in O(|GK|) per query, so maintaining the exact posterior over all
4096 hypotheses costs far less than a single distinguisher evaluation.

The next query is the highest-posterior hypothesis not yet evaluated (a
discrete probability-of-improvement rule); Thompson sampling is available as an
alternative.  The search stops when the posterior concentrates on an
already-evaluated hypothesis or when the query budget runs out.
"""

from __future__ import annotations

import time

import numpy as np

from .multi_bit_bruteforce import AttackResult
from .wrong_key_profile import WrongKeyProfile

# Grid of peak heights (in units of the robust scale) marginalised over.
DEFAULT_A_GRID = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 9.0])


class _Recorder:
    """Tracks the best-so-far trajectory, which gives exp3 its budget curve free."""

    def __init__(self):
        self.order = []            # candidates in query order
        self.scores = []
        self.best_so_far = []      # argmax-so-far after each query
        self._best_c = -1
        self._best_s = -np.inf

    def add(self, cand, score):
        self.order.append(int(cand))
        self.scores.append(float(score))
        if score > self._best_s:
            self._best_s = float(score)
            self._best_c = int(cand)
        self.best_so_far.append(self._best_c)


# --------------------------------------------------------------------------
# Our proposal
# --------------------------------------------------------------------------

class WKRSearch:
    """Wrong-key-response-guided Bayesian search over GK."""

    # Default stopping rule.  0.85 is what experiments/calibrate_stopping.py
    # selected on an independent seed range for the Phase-1 configuration: the
    # cheapest posterior threshold whose success rate stays within one
    # percentage point of exhaustive Algorithm 2.  Re-calibrate for a different
    # cipher, approximation or data complexity.
    DEFAULT_STOP_POSTERIOR = 0.85

    def __init__(self, setup, profile=None, n_init=8, budget=None,
                 stop_z=None, stop_posterior=DEFAULT_STOP_POSTERIOR,
                 min_queries=32, a_grid=None, acquisition="map", seed=0):
        self.setup = setup
        self.profile = profile if profile is not None else WrongKeyProfile(setup)
        self.n_cand = setup.n_candidates
        self.n_init = int(n_init)
        self.budget = int(budget) if budget else self.n_cand
        self.min_queries = int(min_queries)
        # Stopping rules.  ``"auto"`` selects the extreme-value threshold of the
        # project plan (Sect. 3.2): under the null that every remaining
        # candidate is wrong, the largest of |GK| scores sits about
        # sqrt(2 ln |GK|) robust standard deviations above the median.  ``None``
        # disables a rule; both can be active, in which case either one fires.
        self.stop_z = self._auto_z(self.n_cand) if stop_z == "auto" else (
            None if stop_z is None else float(stop_z))
        self.stop_posterior = None if stop_posterior is None else float(stop_posterior)
        self.a_grid = DEFAULT_A_GRID if a_grid is None else np.asarray(a_grid, float)
        self.acquisition = acquisition
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _auto_z(n_cand):
        return float(np.sqrt(2.0 * np.log(n_cand)))

    @staticmethod
    def _robust_scale(scores):
        s = np.asarray(scores)
        mu = float(np.median(s))
        mad = float(np.median(np.abs(s - mu)))
        sigma = 1.4826 * mad if mad > 0 else (float(s.std()) or 1.0)
        return mu, sigma

    # -- posterior ---------------------------------------------------------
    def _posterior(self, P, Q, V, scores):
        mu, sigma = self._robust_scale(scores)
        # U(h) = sum_i z_i g_h(k_i) with z = (s - mu) / sigma
        U = (P - mu * Q) / sigma
        # loglik(h | A) = -(sum z^2 - 2 A U + A^2 V) / 2 ; the sum z^2 term is
        # common to all h and drops out of the normalisation.  A is marginalised
        # over a grid with a flat prior.
        ll = (self.a_grid[:, None] * U[None, :]
              - 0.5 * (self.a_grid[:, None] ** 2) * V[None, :])
        m = ll.max()
        post = np.exp(ll - m).mean(axis=0)
        total = post.sum()
        return post / total if total > 0 else np.full(self.n_cand, 1.0 / self.n_cand)

    # -- main loop ---------------------------------------------------------
    def run(self, scorer, data, record_diagnostics=False):
        """Run the search.

        With ``record_diagnostics`` the per-query best-z and maximum posterior
        are recorded as well, which is what the stopping-rule calibration and
        the budget sweep replay offline.  Recording does not change the search
        itself, so a calibration run and a production run follow identical
        trajectories for the same seed.
        """
        scorer.reset()
        rec = _Recorder()
        diag_z, diag_post = [], []
        evaluated = np.zeros(self.n_cand, dtype=bool)
        P = np.zeros(self.n_cand)
        Q = np.zeros(self.n_cand)
        V = np.zeros(self.n_cand)
        scores = []
        best_c, best_s, best_gamma = -1, -np.inf, 0
        stopped_by = "budget"
        post = None

        t0 = time.perf_counter()
        init = self.rng.choice(self.n_cand, size=min(self.n_init, self.budget),
                               replace=False)
        queue = list(int(c) for c in init)

        while len(rec.order) < self.budget:
            if queue:
                cand = queue.pop(0)
                if evaluated[cand]:
                    continue
            else:
                if post is None:
                    post = self._posterior(P, Q, V, scores)
                cand = self._propose(post, evaluated)
                if cand is None:
                    stopped_by = "exhausted"
                    break

            res = scorer(cand)
            evaluated[cand] = True
            rec.add(cand, res.score)
            scores.append(res.score)
            g = self.profile.response_vector(cand)
            P += res.score * g
            Q += g
            V += g * g
            if res.score > best_s:
                best_s, best_c, best_gamma = res.score, cand, res.gamma_guess

            # The accumulators changed, so any cached posterior is stale.  It is
            # recomputed at most once per query and then reused both by the
            # stopping test and by the next iteration's proposal.
            post = None
            if record_diagnostics or self.stop_posterior is not None:
                post = self._posterior(P, Q, V, scores)

            mu, sigma = self._robust_scale(scores)
            best_z = (best_s - mu) / sigma
            if record_diagnostics:
                diag_z.append(float(best_z))
                diag_post.append(float(post.max()))

            if len(rec.order) >= max(self.min_queries, self.n_init):
                if self.stop_z is not None and best_z >= self.stop_z:
                    stopped_by = "extreme-value"
                    break
                if self.stop_posterior is not None:
                    top = int(np.argmax(post))
                    if post[top] >= self.stop_posterior and evaluated[top]:
                        stopped_by = "posterior"
                        break

        elapsed = time.perf_counter() - t0
        if post is None:
            post = self._posterior(P, Q, V, scores) if scores else None

        return AttackResult(
            method="guided-wkr",
            best_candidate=best_c,
            best_score=best_s,
            gamma_guess=best_gamma,
            true_candidate=int(data.true_candidate),
            true_gamma=int(data.gamma_key_bit),
            n_evals=scorer.n_evals,
            n_calls=scorer.n_calls,
            seconds=elapsed,
            candidate_space=self.n_cand,
            visited=rec.order,
            extra={"trajectory_scores": rec.scores,
                   "best_so_far": rec.best_so_far,
                   "stopped_by": stopped_by,
                   "diag_best_z": diag_z,
                   "diag_max_posterior": diag_post,
                   "max_posterior": float(post.max()) if post is not None else float("nan")},
        )

    def _propose(self, post, evaluated):
        if self.acquisition == "thompson":
            p = post.copy()
            p[evaluated] = 0.0
            tot = p.sum()
            if tot <= 0:
                free = np.flatnonzero(~evaluated)
                return int(self.rng.choice(free)) if len(free) else None
            return int(self.rng.choice(self.n_cand, p=p / tot))
        masked = np.where(evaluated, -np.inf, post)
        if not np.isfinite(masked).any():
            return None
        return int(np.argmax(masked))


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

class SequentialSearch:
    """Algorithm 2's index-order scan, but with our early-stopping rule bolted on.

    This control isolates the two possible sources of saving: stopping early,
    and choosing a good order.  Anything ``WKRSearch`` achieves beyond this is
    attributable to the wrong-key-response model.
    """

    def __init__(self, setup, budget=None, stop_z="auto", min_queries=32, seed=0):
        self.setup = setup
        self.n_cand = setup.n_candidates
        self.budget = int(budget) if budget else self.n_cand
        self.stop_z = WKRSearch._auto_z(self.n_cand) if stop_z == "auto" else (
            None if stop_z is None else float(stop_z))
        self.min_queries = int(min_queries)

    def run(self, scorer, data):
        scorer.reset()
        rec = _Recorder()
        best_c, best_s, best_gamma = -1, -np.inf, 0
        stopped_by = "budget"
        t0 = time.perf_counter()
        for cand in range(min(self.budget, self.n_cand)):
            res = scorer(cand)
            rec.add(cand, res.score)
            if res.score > best_s:
                best_s, best_c, best_gamma = res.score, cand, res.gamma_guess
            if len(rec.order) >= self.min_queries and self.stop_z is not None:
                mu, sigma = WKRSearch._robust_scale(rec.scores)
                if (best_s - mu) / sigma >= self.stop_z:
                    stopped_by = "extreme-value"
                    break
        elapsed = time.perf_counter() - t0
        return AttackResult(
            method="sequential-earlystop", best_candidate=best_c, best_score=best_s,
            gamma_guess=best_gamma, true_candidate=int(data.true_candidate),
            true_gamma=int(data.gamma_key_bit), n_evals=scorer.n_evals,
            n_calls=scorer.n_calls, seconds=elapsed, candidate_space=self.n_cand,
            visited=rec.order,
            extra={"trajectory_scores": rec.scores, "best_so_far": rec.best_so_far,
                   "stopped_by": stopped_by},
        )


class RandomSearch:
    """Uniform sampling without replacement, same budget, no model."""

    def __init__(self, setup, budget, seed=0):
        self.setup = setup
        self.n_cand = setup.n_candidates
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)

    def run(self, scorer, data):
        scorer.reset()
        rec = _Recorder()
        best_c, best_s, best_gamma = -1, -np.inf, 0
        t0 = time.perf_counter()
        order = self.rng.permutation(self.n_cand)[: self.budget]
        for cand in order:
            res = scorer(int(cand))
            rec.add(int(cand), res.score)
            if res.score > best_s:
                best_s, best_c, best_gamma = res.score, int(cand), res.gamma_guess
        elapsed = time.perf_counter() - t0
        return AttackResult(
            method="random", best_candidate=best_c, best_score=best_s,
            gamma_guess=best_gamma, true_candidate=int(data.true_candidate),
            true_gamma=int(data.gamma_key_bit), n_evals=scorer.n_evals,
            n_calls=scorer.n_calls, seconds=elapsed, candidate_space=self.n_cand,
            visited=rec.order,
            extra={"trajectory_scores": rec.scores, "best_so_far": rec.best_so_far,
                   "stopped_by": "budget"},
        )


class SkoptSearch:
    """scikit-optimize Gaussian-process search over the 12 guessed key bits.

    This is the project plan's first-choice tool.  It is a generic black-box
    optimiser: it knows nothing about the cipher, and has to learn the shape of
    the score surface from the queries themselves.
    """

    def __init__(self, setup, budget, n_init=48, seed=0, base_estimator="GP"):
        self.setup = setup
        self.n_cand = setup.n_candidates
        self.budget = int(budget)
        self.n_init = int(n_init)
        self.seed = int(seed)
        self.base_estimator = base_estimator

    def run(self, scorer, data):
        import warnings

        from skopt import Optimizer
        from skopt.space import Integer

        scorer.reset()
        rec = _Recorder()
        n_duplicates = 0
        seen = set()
        n_bits = 12
        space = [Integer(0, 1, name=f"b{i}") for i in range(n_bits)]
        opt = Optimizer(space, base_estimator=self.base_estimator,
                        n_initial_points=self.n_init, random_state=self.seed,
                        acq_func="EI")

        best_c, best_s, best_gamma = -1, -np.inf, 0
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # skopt warns every time its acquisition re-proposes a point it has
            # already evaluated.  On a 12-bit binary space that happens
            # constantly; we count it instead of printing it.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(self.budget):
                x = opt.ask()
                cand = 0
                for i, b in enumerate(x):
                    if int(b):
                        cand |= 1 << i
                if cand in seen:
                    n_duplicates += 1
                seen.add(cand)
                res = scorer(cand)
                opt.tell(x, -float(res.score))          # skopt minimises
                rec.add(cand, res.score)
                if res.score > best_s:
                    best_s, best_c, best_gamma = res.score, cand, res.gamma_guess
        elapsed = time.perf_counter() - t0

        return AttackResult(
            method="guided-skopt", best_candidate=best_c, best_score=best_s,
            gamma_guess=best_gamma, true_candidate=int(data.true_candidate),
            true_gamma=int(data.gamma_key_bit), n_evals=scorer.n_evals,
            n_calls=scorer.n_calls, seconds=elapsed, candidate_space=self.n_cand,
            visited=rec.order,
            extra={"trajectory_scores": rec.scores, "best_so_far": rec.best_so_far,
                   "stopped_by": "budget", "duplicate_proposals": n_duplicates},
        )


def make_search(method, setup, budget, seed=0, **kw):
    if method == "wkr":
        return WKRSearch(setup, budget=budget, seed=seed, **kw)
    if method == "random":
        return RandomSearch(setup, budget=budget, seed=seed)
    if method == "sequential":
        return SequentialSearch(setup, budget=budget, seed=seed,
                                stop_z=kw.get("stop_z", "auto"),
                                min_queries=kw.get("min_queries", 32))
    if method == "skopt":
        return SkoptSearch(setup, budget=budget, seed=seed,
                           n_init=kw.get("n_init", 48))
    raise ValueError(f"unknown search method {method!r}")
