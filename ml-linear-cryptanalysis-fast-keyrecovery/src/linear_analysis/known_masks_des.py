"""Matsui's linear approximations of DES (the paper's Table 4), verified.

The paper states the expressions in the form

    L3: (+_{i in [7,18,24,29]} L0[i]) + R0[15]
        + (+_{i in [7,18,24,29]} L3[i]) + R3[15]  =  K0[22] + K2[22]

with LSB-0 bit numbering.  Which Feistel half each mask sits on depends on a
convention the paper does not spell out (whether the halves are counted before
or after the round swap), so rather than guess, every expression below is
*measured* against this repository's DES implementation in all four possible
orientations and the one that reproduces Matsui's published probability is
recorded.  :func:`verify` re-runs that check.

The resolved orientation is self-validating: it independently reproduces the
paper's own statement of which subkey bits its 8-round attack guesses.  With
``L6`` placed on rounds 1..6 of an 8-round cipher, the front mask activates
S-box 5 -- whose round-0 subkey bits are exactly ``K0[18..23]`` -- and the back
mask activates S-box 1 -- whose round-7 subkey bits are exactly
``K7[42..47]``.  Those are the two 6-bit guesses the paper names, giving
``|GK| = 2^12 = 4096``.

Published probabilities (Matsui 1993, 1994):

    L3   p = 1/2 + 1.56 x 2^-3  = 0.695312
    L5   p = 1/2 + 1.22 x 2^-6  = 0.519063
    L6   not stated numerically in the paper; measured here.
"""

from __future__ import annotations

import numpy as np

from ..ciphers import des
from .bias_search import LinearApproximation

M = des.mask_from_bits


def _appr(rounds, a_in, b_in, a_out, b_out, key_masks):
    return LinearApproximation(
        rounds=rounds, a_in=a_in, b_in=b_in, a_out=a_out, b_out=b_out,
        key_masks=tuple(key_masks), corr_theory=float("nan"), cipher=des,
    )


# --------------------------------------------------------------------------
# The expressions, in the orientation resolved by measurement.
#
# For L3 and L5 the input masks are as the paper writes them and the output
# masks are exchanged; for L6 both ends are exchanged.  (The pattern is just
# the round swap: it depends on the parity of the round count and on where the
# expression's two ends sit relative to it.)
# --------------------------------------------------------------------------

def l3():
    """3-round approximation. Matsui: p = 1/2 + 1.56 * 2^-3."""
    a = M([7, 18, 24, 29])
    b = M([15])
    return _appr(3, a_in=a, b_in=b, a_out=b, b_out=a,
                 key_masks=[M([22]), 0, M([22])])


def l5():
    """5-round approximation. Matsui: p = 1/2 + 1.22 * 2^-6."""
    a = M([15])
    b = M([7, 18, 24, 27, 28, 29, 30, 31])
    return _appr(5, a_in=a, b_in=b, a_out=b, b_out=a,
                 key_masks=[M([42, 43, 45, 46]), M([22]), 0, M([22]),
                            M([42, 43, 45, 46])])


def l6():
    """6-round approximation -- the one the paper's 8-round DES attack uses.

    The key masks are indexed by round *of the approximation*, matching the
    paper: its Table 4 writes K1[22] + K2[44] + K3[22] + K5[22], and its
    8-round attack (where the approximation sits on cipher rounds 1..6) states
    the recovered bit as K2[22] + K3[44] + K4[22] + K6[22] -- the same indices
    shifted by one.
    """
    return _appr(6,
                 a_in=0, b_in=M([7, 18, 24]),
                 a_out=M([15]), b_out=M([7, 18, 24, 29]),
                 key_masks=[0, M([22]), M([44]), M([22]), 0, M([22])])


TABLE = {3: l3, 5: l5, 6: l6}

# Only L6 has the geometry the two-sided attack needs: its two ends each
# activate a single S-box, so each end costs a 6-bit guess and |GK| = 4096.
# L3's ends activate one S-box as well but over only three rounds; L5's input
# mask spans five S-boxes, which would cost 30 guessed bits at the front -- a
# concrete reason the paper builds its 8-round attack on L6 specifically.
ATTACK_READY = (3, 6)

PUBLISHED_P = {3: 0.5 + 1.56 * 2 ** -3, 5: 0.5 + 1.22 * 2 ** -6}


def get_approximation(rounds, n_measure=1 << 22, seed=0, verbose=False):
    """Return the approximation with its probability measured on real DES."""
    from .bias_search import measure_probability

    appr = TABLE[rounds]()
    measure_probability(appr, n_samples=n_measure, seed=seed)
    if verbose:
        print(appr.describe())
    return appr


def verify(n_measure=1 << 22, seed=0, verbose=True):
    """Re-run the orientation check and the probability measurement."""
    ok = True
    for r, factory in sorted(TABLE.items()):
        appr = get_approximation(r, n_measure=n_measure, seed=seed)
        bias = abs(appr.p_measured - 0.5)
        line = (f"  L{r}: measured p = {appr.p_measured:.6f}  "
                f"bias = 2^{np.log2(bias):.2f}")
        if r in PUBLISHED_P:
            want = PUBLISHED_P[r]
            close = abs(appr.p_measured - want) < 5e-3
            ok &= close
            line += f"   published {want:.6f}   {'OK' if close else 'MISMATCH'}"
        else:
            # No published number: just require a bias well above the noise floor.
            noise = 3.0 / np.sqrt(n_measure)
            close = bias > noise
            ok &= close
            line += f"   (noise floor {noise:.5f})   {'OK' if close else 'MISMATCH'}"
        if verbose:
            print(line)
    return ok


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", type=int, default=1 << 23)
    args = ap.parse_args()
    print("Matsui's DES approximations, measured against this implementation:")
    good = verify(n_measure=args.measure)
    print("all consistent" if good else "SOME MISMATCHED")
    print()
    for r in sorted(TABLE):
        a = get_approximation(r, n_measure=1 << 20)
        print(a.describe())
        print(f"  front guess S-box {a.in_active_sbox}, "
              f"back guess S-box {a.out_active_sbox}")
        print()
