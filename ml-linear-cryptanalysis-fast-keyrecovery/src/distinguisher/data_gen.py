"""Training-data construction in the paper's two formats.

Eq. 6 (one-bit key recovery)::

    Y(x_1||...||x_t) = 1  if x_i = omega_i        (x_i ~ Bern(1 - p_r))
                     = 0  if x_i = omega_i ^ 1    (x_i ~ Bern(p_r))

Eq. 7 (multiple-bit key recovery)::

    Y(x_1||...||x_t) = 1  if x_i = omega_i        (x_i ~ Bern(1 - p_r))
                     = 0  if x_i ~ Bern(1/2)

where ``omega_i = alpha.P_i ^ beta.C_i ^ gamma.K``.  In both formats *one
sample uses one master key* and different samples use different master keys,
so N samples of dimension t consume N master keys and N*t plaintexts.
"""

from __future__ import annotations

import numpy as np

from ..ciphers import toy_feistel as tf


def _chunks(total, size):
    done = 0
    while done < total:
        n = min(size, total - done)
        yield done, n
        done += n


# --------------------------------------------------------------------------
# omega streams
# --------------------------------------------------------------------------

def one_bit_omega(appr, rng, n_samples, t, chunk=4096):
    """omega for the plain r-round cipher: one master key per row.

    Used by the one-bit key-recovery framework, where no key guessing happens
    and the approximation sits on rounds 0 .. r-1.
    """
    c = appr.cipher
    out = np.empty((n_samples, t), dtype=np.uint8)
    for off, n in _chunks(n_samples, chunk):
        mk = c.random_master_keys(rng, (n,))
        subkeys = c.key_schedule(mk, appr.rounds)
        sk_b = [k[:, None] for k in subkeys]
        l0, r0 = c.random_blocks(rng, (n, t))
        lc, rc = c.encrypt(l0, r0, sk_b)
        bit = appr.state_parity(l0, r0, lc, rc) ^ appr.key_parity(subkeys, 0)[:, None]
        out[off:off + n] = bit
    return out


def multi_bit_omega(setup, rng, n_samples, t, chunk=2048):
    """omega for the (r+2)-round attack geometry, evaluated at the *correct*
    subkey guess.  This deliberately goes through exactly the same
    ``transform_bits`` path the attack uses, so training and attack data cannot
    drift apart."""
    out = np.empty((n_samples, t), dtype=np.uint8)
    for off, n in _chunks(n_samples, chunk):
        data = setup.generate(rng, (n, t))
        out[off:off + n] = setup.omega_bits(data)
    return out


# --------------------------------------------------------------------------
# Labelled data sets
# --------------------------------------------------------------------------

def _split_labels(rng, n_samples):
    """A balanced 0/1 label vector."""
    y = np.zeros(n_samples, dtype=np.uint8)
    y[: n_samples // 2] = 1
    rng.shuffle(y)
    return y


def make_one_bit_dataset(appr, n_samples, t, rng, chunk=4096):
    """Paper's Eq. 6.  Returns (X float32 (n, t), Y float32 (n,))."""
    omega = one_bit_omega(appr, rng, n_samples, t, chunk=chunk)
    y = _split_labels(rng, n_samples)
    x = omega ^ (1 - y)[:, None]          # label 0 -> complement of omega
    return x.astype(np.float32), y.astype(np.float32)


def make_multi_bit_dataset(setup, n_samples, t, rng, chunk=2048):
    """Paper's Eq. 7.  Returns (X float32 (n, t), Y float32 (n,))."""
    y = _split_labels(rng, n_samples)
    n_pos = int(y.sum())
    x = np.empty((n_samples, t), dtype=np.uint8)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(y == 0)
    x[pos_idx] = multi_bit_omega(setup, rng, n_pos, t, chunk=chunk)
    x[neg_idx] = rng.integers(0, 2, size=(len(neg_idx), t), dtype=np.uint8)
    return x.astype(np.float32), y.astype(np.float32)


# --------------------------------------------------------------------------
# Reference (non-learned) distinguisher, for sanity-checking the network
# --------------------------------------------------------------------------

def optimal_accuracy(p_r, t, negatives_random=True):
    """Accuracy of the Bayes-optimal test on a single t-bit sample, at fixed p_r.

    For these Bernoulli problems the sufficient statistic is the popcount, so
    the optimal test is a threshold on it, and the exact accuracy is a sum of
    two binomial pmfs.  This is the reference the trained network is measured
    against.

    It is a reference, not a hard ceiling: the probability of a linear
    approximation is *key-dependent*, and ``p_r`` here is its average over
    keys.  Accuracy is convex in |p - 1/2|, so averaging the optimal accuracy
    over the true spread of per-key probabilities is slightly higher than the
    optimal accuracy at the average probability.  A trained network can
    therefore land a little above this number -- as it does, by around 0.2-0.4
    percentage points -- without anything being wrong.
    """
    from scipy.stats import binom

    q1 = 1.0 - p_r                       # Pr[x_i = 1] for label 1
    q0 = 0.5 if negatives_random else p_r
    k = np.arange(t + 1)
    p1 = binom.pmf(k, t, q1)
    p0 = binom.pmf(k, t, q0)
    # Optimal decision per popcount value: pick the larger likelihood.
    return float(0.5 * np.sum(np.maximum(p1, p0)))
