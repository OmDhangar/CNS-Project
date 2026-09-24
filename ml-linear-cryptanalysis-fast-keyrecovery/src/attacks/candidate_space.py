"""The candidate subkey space of the multi-bit key-recovery attack.

This module owns the one piece of machinery that the baseline (exhaustive
Algorithm 2) and our guided search *must* share, so that any measured
difference between them is attributable to the search order alone:

    transform_bits(data, candidate) -> the bit sequence alpha.P' ^ beta.C'

which is the ``partial_transform`` of the project plan (step 1 of the build
order) and corresponds to Fig. 1 / Fig. 5 of the paper: guess K_0, encrypt one
round; guess K_{r+1}, decrypt one round.

Geometry (mirrors Fig. 5 of the paper exactly)
----------------------------------------------
The cipher has R = r + 2 rounds; the r-round linear approximation sits on
rounds 1 .. r, i.e. between states 1 and r+1.

    L_1     = R_0
    R_1     = L_0 ^ F(R_0, K_0)
    R_{R-1} = L_R
    L_{R-1} = R_R ^ F(L_R, K_{R-1})

so the approximation's parity is

    x = a_in.L_1 ^ b_in.R_1 ^ a_out.L_{r+1} ^ b_out.R_{r+1}
      = [ a_in.R_0 ^ b_in.L_0 ^ a_out.R_R ^ b_out.L_R ]     <- key-independent
        ^ b_in .F(R_0, K_0)                                 <- 6 guessed bits
        ^ a_out.F(L_R, K_{R-1})                             <- 6 guessed bits

Because the trail search guarantees that b_in and a_out each activate exactly
one S-box, each of the two F-terms depends on only six subkey bits, giving a
candidate space of |GK| = 2^6 * 2^6 = 4096 -- the same size as the paper's
8-round DES attack on K0[18..23] and K7[42..47].

The two F-terms are pre-tabulated as 64x64 bit LUTs, so scoring a candidate
costs two gathers and two XORs over the data, independent of the round count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ciphers import toy_feistel as tf
from ..linear_analysis.bias_search import LinearApproximation

GUESS_BITS_PER_SIDE = tf.SBOX_IN_BITS          # 6
GUESS_BITS = 2 * GUESS_BITS_PER_SIDE           # 12
N_CANDIDATES = 1 << GUESS_BITS                 # 4096


@dataclass
class AttackData:
    """Known plaintext/ciphertext material, reduced to what the attack needs.

    Shapes: ``base``, ``g_front`` and ``g_back`` all share the data shape.  For
    a single key-recovery attack that is ``(n_pairs,)``; for training-data
    generation it is ``(n_samples, t)``, one master key per row, which is the
    format Eq. 6 / Eq. 7 of the paper prescribe.  ``master_key``,
    ``true_candidate`` and ``gamma_key_bit`` are then arrays shaped so they
    broadcast against the data (i.e. ``(n_samples, 1)``).
    """

    base: np.ndarray        # uint8, the key-independent parity, one bit per pair
    g_front: np.ndarray     # uint8 0..63, S-box j_front input taken from R_0
    g_back: np.ndarray      # uint8 0..63, S-box j_back  input taken from L_R
    master_key: object
    true_candidate: object
    gamma_key_bit: object   # gamma . K' -- the extra bit Algorithm 2 also returns
    n_pairs: int


class AttackSetup:
    """Everything that is fixed across all trials of one attack configuration."""

    def __init__(self, appr: LinearApproximation, cipher_rounds=None, first_round=1,
                 cipher=tf):
        """`cipher` is the cipher module (``toy_feistel`` or ``des``).

        Both expose the same small surface -- ``SBOX_FLAT``, ``sbox_input``,
        ``subkey_group``, ``active_sboxes_from_f_output_mask``, ``key_schedule``,
        ``encrypt``, ``mask_dot``, ``random_blocks``, ``random_master_keys`` --
        so every attack in this package is cipher-agnostic and Phase 2 reuses
        the Phase-1 search code unchanged.
        """
        self.cipher = cipher
        self.appr = appr
        self.first_round = first_round
        self.cipher_rounds = cipher_rounds if cipher_rounds is not None else appr.rounds + 2
        if self.cipher_rounds != appr.rounds + 2:
            raise ValueError("expected cipher_rounds == approximation rounds + 2")

        act_in = cipher.active_sboxes_from_f_output_mask(appr.b_in)
        act_out = cipher.active_sboxes_from_f_output_mask(appr.a_out)
        if len(act_in) != 1 or len(act_out) != 1:
            raise ValueError(
                "approximation must activate exactly one S-box at each end "
                f"(got {sorted(act_in)} at the front, {sorted(act_out)} at the back)")
        self.j_front, self.nib_front = next(iter(act_in.items()))
        self.j_back, self.nib_back = next(iter(act_out.items()))

        self.front_lut = self._build_lut(self.j_front, self.nib_front)
        self.back_lut = self._build_lut(self.j_back, self.nib_back)
        self.n_candidates = N_CANDIDATES

    # -- construction ------------------------------------------------------
    def _build_lut(self, j, nib):
        """lut[k, g] = parity( nib . S_j(g ^ k) ), for all 6-bit k and g."""
        c = self.cipher
        g = np.arange(64, dtype=np.uint32)
        lut = np.zeros((64, 64), dtype=np.uint8)
        for k in range(64):
            lut[k] = c.parity(c.SBOX_FLAT[j][g ^ np.uint32(k)] & np.uint32(nib))
        return lut

    # -- candidate encoding ------------------------------------------------
    @staticmethod
    def split(candidate):
        """candidate -> (k_front, k_back), both 6-bit.  Array-safe."""
        c = np.asarray(candidate)
        if c.ndim == 0:
            i = int(c)
            return i & 0x3F, (i >> GUESS_BITS_PER_SIDE) & 0x3F
        return c & 0x3F, (c >> GUESS_BITS_PER_SIDE) & 0x3F

    @staticmethod
    def join(k_front, k_back):
        return (int(k_front) & 0x3F) | ((int(k_back) & 0x3F) << GUESS_BITS_PER_SIDE)

    @staticmethod
    def to_bit_vector(candidate):
        """candidate -> length-12 vector of 0/1 (LSB first). Used by the surrogate."""
        c = int(candidate)
        return np.array([(c >> i) & 1 for i in range(GUESS_BITS)], dtype=np.float64)

    @staticmethod
    def from_bit_vector(bits):
        c = 0
        for i, b in enumerate(bits):
            if int(round(float(b))):
                c |= 1 << i
        return c

    # -- data generation ---------------------------------------------------
    def generate(self, rng, shape, master_key=None):
        """Produce known plaintext/ciphertext material.

        ``shape`` is either an int ``n_pairs`` (one attack under one key) or a
        tuple ``(n_samples, t)`` (training data: one master key per row).
        """
        c = self.cipher
        shape = (int(shape),) if np.isscalar(shape) else tuple(shape)
        key_shape = shape[:-1] if len(shape) > 1 else ()
        if master_key is None:
            master_key = c.random_master_keys(rng, key_shape)
        mk = np.asarray(master_key)

        subkeys = c.key_schedule(mk, self.cipher_rounds)
        extra = len(shape) - mk.ndim
        sk_b = [k.reshape(k.shape + (1,) * extra) for k in subkeys] if extra else subkeys

        l0, r0 = c.random_blocks(rng, shape)
        lc, rc = c.encrypt(l0, r0, sk_b)

        a = self.appr
        base = (c.mask_dot(a.a_in, r0) ^ c.mask_dot(a.b_in, l0)
                ^ c.mask_dot(a.a_out, rc) ^ c.mask_dot(a.b_out, lc))

        g_front = c.sbox_input(r0, self.j_front).astype(np.uint8)
        g_back = c.sbox_input(lc, self.j_back).astype(np.uint8)

        kf = np.asarray(c.subkey_group(subkeys[0], self.j_front)).astype(np.int64)
        kb = np.asarray(c.subkey_group(subkeys[self.cipher_rounds - 1],
                                       self.j_back)).astype(np.int64)
        cand = kf | (kb << GUESS_BITS_PER_SIDE)
        gamma = a.key_parity(subkeys, self.first_round)

        if mk.ndim == 0:
            master_key = int(mk)
            cand = int(cand)
            gamma = int(gamma)
        else:
            cand = cand.reshape(cand.shape + (1,) * extra)
            gamma = gamma.reshape(gamma.shape + (1,) * extra)
            master_key = mk

        return AttackData(
            base=base, g_front=g_front, g_back=g_back,
            master_key=master_key,
            true_candidate=cand,
            gamma_key_bit=gamma,
            n_pairs=int(np.prod(shape)),
        )

    # -- the shared partial transform -------------------------------------
    def transform_bits(self, data: AttackData, candidate):
        """alpha.P' ^ beta.C' for the given candidate subkey guess.

        This is the single scoring input used by BOTH the exhaustive baseline
        and the guided search.
        """
        kf, kb = self.split(candidate)
        return data.base ^ self.front_lut[kf, data.g_front] ^ self.back_lut[kb, data.g_back]

    # -- fast path used by the scorer --------------------------------------
    #
    # transform_bits above is the readable definition.  In the attacks it is
    # called 4096 times per trial on ~10**5-element arrays, where two fancy
    # gathers plus a float conversion dominate the runtime.  Since the result
    # depends on the data only through the triple (base, g_front, g_back) --
    # 2 * 64 * 64 = 8192 possibilities -- we pack that triple into one index
    # array once, and then a candidate's whole bit stream is a single
    # ``np.take`` from an 8192-entry float table.  ``self_test`` below asserts
    # the two paths agree.

    def precompute_streams(self, data: AttackData):
        """Materialise the two guessed F-terms for all 64 half-guesses each.

        ``front[kf]`` is the bit stream ``b_in . F(R_0, kf)`` over the whole data
        set, already XORed with the key-independent ``base`` term; ``back[kb]``
        is ``a_out . F(L_R, kb)``.  A candidate's bit stream is then a single
        XOR of two uint8 arrays.  This costs 128 gathers up front, is shared by
        both search strategies, and is what makes the per-candidate cost of the
        attacks dominated by the neural distinguisher rather than by numpy.
        """
        front = np.empty((64, data.base.size), dtype=np.uint8)
        back = np.empty((64, data.base.size), dtype=np.uint8)
        for k in range(64):
            np.bitwise_xor(self.front_lut[k][data.g_front], data.base, out=front[k])
            back[k] = self.back_lut[k][data.g_back]
        return front, back

    def self_test(self, data: AttackData, candidates=(0, 1, 777, 4095), streams=None):
        """Check the fast path against the readable definition.

        ``streams`` lets a caller that has already built them pass them in, so
        the check costs four XORs rather than a second full precompute.
        """
        front, back = streams if streams is not None else self.precompute_streams(data)
        for c in candidates:
            kf, kb = self.split(c)
            slow = self.transform_bits(data, c)
            fast = front[kf] ^ back[kb]
            if not np.array_equal(slow, fast):
                raise AssertionError(f"fast path disagrees for candidate {c}")
        return True

    def omega_bits(self, data: AttackData, candidate=None):
        """omega = alpha.P' ^ beta.C' ^ gamma.K'  (the paper's training target).

        With the *correct* candidate this is the r-round approximation's own
        parity, which is 0 with probability p_r.  Only usable when the key is
        known, i.e. when generating training data.
        """
        if candidate is None:
            candidate = data.true_candidate
        return self.transform_bits(data, candidate) ^ np.asarray(data.gamma_key_bit, dtype=np.uint8)

    @property
    def cipher_name(self):
        n = getattr(self.cipher, "__name__", "cipher").rsplit(".", 1)[-1]
        return "TinyDES-24" if n == "toy_feistel" else n.upper()

    def describe(self):
        a = self.appr
        return (
            f"AttackSetup: {self.cipher_rounds}-round {self.cipher_name}, "
            f"{a.rounds}-round approximation on rounds "
            f"{self.first_round}..{self.first_round + a.rounds - 1}\n"
            f"  front guess: K_0 bits feeding S-box {self.j_front} "
            f"(6 bits, output mask 0x{self.nib_front:x})\n"
            f"  back  guess: K_{self.cipher_rounds - 1} bits feeding S-box {self.j_back} "
            f"(6 bits, output mask 0x{self.nib_back:x})\n"
            f"  |GK| = {self.n_candidates}\n"
            f"  approximation p = {a.p:.6f}  (bias 2^{np.log2(a.bias):.2f})"
        )
