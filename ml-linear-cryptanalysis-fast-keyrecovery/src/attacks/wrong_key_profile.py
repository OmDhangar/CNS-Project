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

    def summary(self):
        nf = int(np.count_nonzero(self.abs_front >= 0.5))
        nb = int(np.count_nonzero(self.abs_back >= 0.5))
        return (f"WrongKeyProfile: |rho_f| >= 0.5 for {nf}/64 differences, "
                f"|rho_b| >= 0.5 for {nb}/64;\n"
                f"  one query informs {self.coverage(0.3)} of "
                f"{self.setup.n_candidates} hypotheses at the 0.3 level "
                f"({self.coverage(0.3) / self.setup.n_candidates * 100:.1f}%)")
