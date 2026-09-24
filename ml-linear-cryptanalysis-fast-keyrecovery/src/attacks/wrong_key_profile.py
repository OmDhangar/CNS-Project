"""The wrong-key response profile of a linear key-recovery attack.

This is the piece of structure our guided search exploits, and the reason a
model-based search is viable here at all.

Setting
-------
The attack's bit stream for a candidate ``gk = (kf, kb)`` is

    x(gk) = base ^ front(R_0, kf) ^ back(L_R, kb)

so, writing ``gk*`` for the correct guess,

    x(gk) = x(gk*) ^ [front(kf) ^ front(kf*)] ^ [back(kb) ^ back(kb*)].

The two error terms depend on the data only through the S-box inputs, which are
uniform and (to an excellent approximation) independent of the approximation's
own parity.  Hence the correlation of the candidate's bit stream factorises:

    corr(gk) = corr(gk*) * rho_f(kf ^ kf*) * rho_b(kb ^ kb*)

with

    rho_f(D) = 2^-6 * sum_u (-1)^( nib_f . [ S_jf(u) ^ S_jf(u ^ D) ] )

and likewise for rho_b.  Two things make this useful:

1.  ``rho_f`` and ``rho_b`` depend only on the S-box and the approximation's
    masks -- **not on the key and not on the data**.  They are computed exactly,
    offline, in 2 * 64 * 64 operations, before a single distinguisher
    evaluation is spent.
2.  Because Algorithm 2 reports ``max(w0, w1)`` over the two guesses of
    gamma.K', the reported score tracks the *absolute* correlation, so the
    expected score profile is proportional to ``|rho_f * rho_b|``.

This is the linear-cryptanalysis analogue of the wrong-key response profile
that Gohr (2019, Sect. 4.3) measures empirically for his differential neural
distinguishers and feeds to a Bayesian key search.  Here it comes out in closed
form, which is what lets the search start informed rather than cold.
"""

from __future__ import annotations

import numpy as np


def half_profile(lut):
    """rho(D) for one side, from that side's 64x64 parity LUT.

    ``lut[k, g] = parity(nib . S(g ^ k))``.  The definition below averages over
    k as well as g; the result is provably independent of k, and
    :func:`check_key_independence` asserts it numerically.
    """
    rho = np.zeros(64)
    for d in range(64):
        agree = 1.0 - 2.0 * (lut[0] ^ lut[d]).mean()
        rho[d] = agree
    return rho


def check_key_independence(lut, atol=1e-12):
    """rho(D) must not depend on which key it is measured from."""
    for d in range(64):
        ref = 1.0 - 2.0 * (lut[0] ^ lut[d]).mean()
        for k in range(1, 64):
            v = 1.0 - 2.0 * (lut[k] ^ lut[k ^ d]).mean()
            if abs(v - ref) > atol:
                return False
    return True


class WrongKeyProfile:
    """Offline model of the expected score surface of an :class:`AttackSetup`."""

    def __init__(self, setup):
        self.setup = setup
        self.rho_front = half_profile(setup.front_lut)
        self.rho_back = half_profile(setup.back_lut)
        self.abs_front = np.abs(self.rho_front)
        self.abs_back = np.abs(self.rho_back)
        # xor_f[d, k] = |rho_front[d ^ k]| -- pre-shifted for fast lookup
        idx = np.arange(64)
        self.xor_front = self.abs_front[idx[:, None] ^ idx[None, :]]
        self.xor_back = self.abs_back[idx[:, None] ^ idx[None, :]]

    def response_vector(self, candidate):
        """g[h] = |rho_f(kf_h ^ kf_c) * rho_b(kb_h ^ kb_c)| for every hypothesis h.

        Returned flat in candidate order (h = kf | kb << 6), length 4096.
        """
        kf, kb = self.setup.split(candidate)
        return np.outer(self.xor_back[kb], self.xor_front[kf]).reshape(-1)

    def expected_surface(self, true_candidate):
        """The full predicted |score| surface if `true_candidate` were correct."""
        return self.response_vector(true_candidate)

    def coverage(self, threshold=0.3):
        """How many hypotheses one query meaningfully informs about."""
        g = self.response_vector(0)
        return int(np.count_nonzero(g >= threshold))

    # -- offline probe design ---------------------------------------------
    #
    # The posterior can only rule a hypothesis out if some observation carries
    # information about it.  A hypothesis nothing has touched keeps its prior,
    # so a search that only ever tests high-posterior candidates leaves most of
    # the space untouched and can never become confident without testing nearly
    # all of it -- which is precisely the weakness measured in the stopping-rule
    # calibration.
    #
    # The fix is to spend the first queries *covering* the space rather than
    # exploiting it: choose probes so that every hypothesis is within a decent
    # response of at least one probe.  Because the response factorises, so does
    # the covering problem: cover the 64 front differences and the 64 back
    # differences independently and take the product.  The whole design is
    # computed offline from the S-boxes, with no queries spent.

    @staticmethod
    def _cover_axis(abs_rho, threshold):
        """Smallest set S (greedily) with: for every k, some s in S has |rho(k^s)| >= threshold.

        The sets ``s XOR D``, where ``D = {d : |rho(d)| >= threshold}``, are
        translates of one another, so this is a covering-code problem on
        GF(2)^6 and greedy set cover is a good fit.
        """
        d_set = np.flatnonzero(abs_rho >= threshold)
        if len(d_set) == 0:
            return list(range(64))
        covers = [set(int(s) ^ int(d) for d in d_set) for s in range(64)]
        uncovered = set(range(64))
        chosen = []
        while uncovered:
            best_s = max(range(64), key=lambda s: len(covers[s] & uncovered))
            gain = len(covers[best_s] & uncovered)
            if gain == 0:                      # cannot happen unless d_set is empty
                chosen.extend(sorted(uncovered))
                break
            chosen.append(best_s)
            uncovered -= covers[best_s]
        return chosen

    def covering_design(self, level=0.25):
        """Probe set covering every hypothesis at response >= `level`.

        Thresholds for the two sides are chosen jointly to minimise the number
        of probes subject to ``threshold_front * threshold_back >= level``.
        Returns ``(probes, info)``.
        """
        vals_f = sorted({v for v in np.round(self.abs_front, 6) if v > 0}, reverse=True)
        vals_b = sorted({v for v in np.round(self.abs_back, 6) if v > 0}, reverse=True)
        best = None
        for tf in vals_f:
            for tb in vals_b:
                if tf * tb < level - 1e-12:
                    continue
                sf = self._cover_axis(self.abs_front, tf)
                sb = self._cover_axis(self.abs_back, tb)
                n = len(sf) * len(sb)
                if best is None or n < best[0]:
                    best = (n, tf, tb, sf, sb)
        if best is None:                        # level unreachable; fall back
            sf = self._cover_axis(self.abs_front, max(vals_f))
            sb = self._cover_axis(self.abs_back, max(vals_b))
            best = (len(sf) * len(sb), max(vals_f), max(vals_b), sf, sb)

        n, tf, tb, sf, sb = best
        probes = [self.setup.join(kf, kb) for kf in sf for kb in sb]
        info = {"level": level, "threshold_front": tf, "threshold_back": tb,
                "n_front": len(sf), "n_back": len(sb), "n_probes": len(probes),
                "fraction_of_space": len(probes) / self.setup.n_candidates}
        return probes, info

    def verify_covering(self, probes, level):
        """Every hypothesis really is within `level` of some probe."""
        best = np.zeros(self.setup.n_candidates)
        for p in probes:
            np.maximum(best, self.response_vector(p), out=best)
        return float(best.min()) >= level - 1e-9, float(best.min())

    def summary(self):
        nf = int(np.count_nonzero(self.abs_front >= 0.5))
        nb = int(np.count_nonzero(self.abs_back >= 0.5))
        return (f"WrongKeyProfile: |rho_f| >= 0.5 for {nf}/64 differences, "
                f"|rho_b| >= 0.5 for {nb}/64;\n"
                f"  one query informs {self.coverage(0.3)} of "
                f"{self.setup.n_candidates} hypotheses at the 0.3 level "
                f"({self.coverage(0.3) / self.setup.n_candidates * 100:.1f}%)")
