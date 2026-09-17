"""TinyDES-24: a DES-shaped Feistel toy cipher used for Phase 1 of the project.

Design goals (see report/project_report.md):

* Same *structural* anatomy as DES -- Feistel network, expansion, S-box layer,
  bit permutation, rotating key schedule -- so that Matsui-style linear
  cryptanalysis and the paper's key-recovery attack transfer verbatim.
* Small enough (24-bit block, 24-bit master key) that the linear-approximation
  search is exhaustive-ish and the full candidate-subkey space can be brute
  forced as ground truth.
* A candidate subkey space for the multi-bit attack of exactly 2**12 = 4096,
  identical in size to the paper's 8-round DES attack (K0[18..23], K7[42..47]).

Structure
---------
State: 24 bits split as L (12 bits) || R (12 bits), bit 0 = LSB.

    L_{i+1} = R_i
    R_{i+1} = L_i XOR F(R_i, K_i)

    F(x, K) = P( S( E(x) XOR K ) )

E : 12 -> 18 bits, DES-style overlapping expansion into three 6-bit groups.
S : three 6->4 bit S-boxes; we reuse the real DES S1, S2, S3 so that the
    linear-approximation table has realistic (non-degenerate) structure.
P : a 12-bit bit permutation (multiply-by-5 mod 12, a bijection that spreads
    each S-box's four output bits across several next-round S-box inputs).
Key schedule: DES-style.  The 24-bit master key is split into C (low 12 bits)
    and D (high 12 bits); both are rotated left by a per-round amount and an
    18-of-24 selection (PC-2 analogue) produces the round subkey.

Everything is vectorised over numpy arrays of uint32; both the plaintext
arrays and the subkey arrays may be per-element, which is what the paper's
training-data format needs (one master key per training *sample*).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

BLOCK_BITS = 24
HALF_BITS = 12
HALF_MASK = (1 << HALF_BITS) - 1
N_SBOX = 3
SBOX_IN_BITS = 6
SBOX_OUT_BITS = 4
EXPANDED_BITS = N_SBOX * SBOX_IN_BITS          # 18
SUBKEY_BITS = EXPANDED_BITS                    # 18
MASTER_KEY_BITS = 24
MASTER_KEY_MASK = (1 << MASTER_KEY_BITS) - 1

# --------------------------------------------------------------------------
# Expansion E : 12 -> 18
#
# E_TABLE[6*j + q] is the index of the source bit of R that becomes the q-th
# bit (q = 0 is the MOST significant) of the 6-bit input of S-box j.  Exactly
# like DES: each 4-bit nibble is padded with the neighbouring edge bits.
# --------------------------------------------------------------------------

_e = []
for _j in range(N_SBOX):
    _e.extend([
        (4 * _j - 1) % HALF_BITS,
        (4 * _j + 0) % HALF_BITS,
        (4 * _j + 1) % HALF_BITS,
        (4 * _j + 2) % HALF_BITS,
        (4 * _j + 3) % HALF_BITS,
        (4 * _j + 4) % HALF_BITS,
    ])
E_TABLE = tuple(_e)

# E_SRC is the same table re-indexed by *bit position inside the packed 18-bit
# word*: the 6-bit input of S-box j lives in bits 6j..6j+5, with bit 6j+5 the
# most significant.  So packed position m = 6j + b corresponds to E_TABLE entry
# 6j + (5 - b).  Everything that manipulates packed expanded words or masks on
# them (the E lookup table, and the mask fold used by the trail search) must go
# through E_SRC, never E_TABLE directly.
E_SRC = tuple(
    E_TABLE[6 * (m // 6) + (5 - (m % 6))] for m in range(N_SBOX * 6)
)

# --------------------------------------------------------------------------
# S-boxes: DES S1, S2, S3 (4 rows x 16 columns)
# --------------------------------------------------------------------------

_DES_S1 = [
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
]
_DES_S2 = [
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
    [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
    [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
    [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
]
_DES_S3 = [
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
    [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
    [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
    [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
]

_SBOX_ROWS = [_DES_S1, _DES_S2, _DES_S3]


def _flatten_sbox(rows):
    """DES addressing: outer bits (b5, b0) pick the row, inner bits b4..b1 the column."""
    flat = np.zeros(1 << SBOX_IN_BITS, dtype=np.uint32)
    for v in range(1 << SBOX_IN_BITS):
        row = ((v >> 5) & 1) * 2 + (v & 1)
        col = (v >> 1) & 0xF
        flat[v] = rows[row][col]
    return flat


SBOX_FLAT = [_flatten_sbox(r) for r in _SBOX_ROWS]

# --------------------------------------------------------------------------
# Permutation P on 12 bits: source bit k -> destination bit (5*k) % 12.
# gcd(5, 12) = 1, so this is a bijection, and it scatters the four output bits
# of every S-box across three different next-round nibbles.
# --------------------------------------------------------------------------

P_TABLE = tuple((5 * k) % HALF_BITS for k in range(HALF_BITS))

# --------------------------------------------------------------------------
# Key schedule tables
# --------------------------------------------------------------------------

# Per-round left rotation amounts for C and D (cycled if more rounds are used).
KS_SHIFTS = (1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1)

# PC-2 analogue: pick 18 of the 24 bits of C||D (C = bits 0..11, D = bits 12..23).
# Nine bits come from C and nine from D.  PC2_TABLE[m] is the source index
# (0..23) of bit m of the packed 18-bit subkey; bits 6j..6j+5 are XORed onto the
# 6-bit input of S-box j (same packing as E_SRC).
PC2_TABLE = (
    5, 0, 9, 18, 14, 22,      # -> S-box 0
    1, 11, 3, 20, 16, 12,     # -> S-box 1
    8, 6, 4, 23, 13, 19,      # -> S-box 2
)
assert len(PC2_TABLE) == SUBKEY_BITS
assert len(set(PC2_TABLE)) == SUBKEY_BITS


# --------------------------------------------------------------------------
# Lookup tables
# --------------------------------------------------------------------------

def _build_e_lut():
    lut = np.zeros(1 << HALF_BITS, dtype=np.uint32)
    for x in range(1 << HALF_BITS):
        out = 0
        for m in range(EXPANDED_BITS):
            out |= ((x >> E_SRC[m]) & 1) << m
        lut[x] = out
    return lut


def _build_p_lut():
    lut = np.zeros(1 << HALF_BITS, dtype=np.uint32)
    idx = np.arange(1 << HALF_BITS, dtype=np.uint32)
    for k in range(HALF_BITS):
        bit = (idx >> np.uint32(k)) & np.uint32(1)
        lut |= bit << np.uint32(P_TABLE[k])
    return lut


def _build_sp_lut():
    """Fused S-box layer + P permutation, indexed by the full 18-bit E(x) ^ K."""
    idx = np.arange(1 << EXPANDED_BITS, dtype=np.uint32)
    s_out = np.zeros_like(idx)
    for j in range(N_SBOX):
        g = (idx >> np.uint32(SBOX_IN_BITS * j)) & np.uint32(0x3F)
        s_out |= SBOX_FLAT[j][g] << np.uint32(SBOX_OUT_BITS * j)
    lut = np.zeros(1 << EXPANDED_BITS, dtype=np.uint32)
    for k in range(HALF_BITS):
        bit = (s_out >> np.uint32(k)) & np.uint32(1)
        lut |= bit << np.uint32(P_TABLE[k])
    return lut


E_LUT = _build_e_lut()
P_LUT = _build_p_lut()
SP_LUT = _build_sp_lut()


# --------------------------------------------------------------------------
# Core primitives
# --------------------------------------------------------------------------

def f_function(x, subkey):
    """F(x, K) = P(S(E(x) ^ K)).  Vectorised; x and subkey broadcast together."""
    return SP_LUT[E_LUT[x] ^ subkey]


def key_schedule(master_key, rounds):
    """Expand a 24-bit master key into `rounds` 18-bit subkeys.

    `master_key` may be a python int or a numpy array of uint32.  Returns a
    list of length `rounds`; element i has the same shape as `master_key`.
    """
    mk = np.asarray(master_key, dtype=np.uint32)
    c = mk & np.uint32(HALF_MASK)
    d = (mk >> np.uint32(HALF_BITS)) & np.uint32(HALF_MASK)
    subkeys = []
    for i in range(rounds):
        s = int(KS_SHIFTS[i % len(KS_SHIFTS)])
        c = ((c << np.uint32(s)) | (c >> np.uint32(HALF_BITS - s))) & np.uint32(HALF_MASK)
        d = ((d << np.uint32(s)) | (d >> np.uint32(HALF_BITS - s))) & np.uint32(HALF_MASK)
        cd = c | (d << np.uint32(HALF_BITS))
        k = np.zeros_like(cd)
        for m, src in enumerate(PC2_TABLE):
            k |= ((cd >> np.uint32(src)) & np.uint32(1)) << np.uint32(m)
        subkeys.append(k)
    return subkeys


def encrypt(left, right, subkeys):
    """Encrypt (L0, R0) with the given list of subkeys.  Returns (L_r, R_r)."""
    l = np.asarray(left, dtype=np.uint32)
    r = np.asarray(right, dtype=np.uint32)
    for k in subkeys:
        l, r = r, l ^ f_function(r, k)
    return l, r


def decrypt(left, right, subkeys):
    """Inverse of :func:`encrypt` for the same subkey list."""
    l = np.asarray(left, dtype=np.uint32)
    r = np.asarray(right, dtype=np.uint32)
    for k in reversed(subkeys):
        l, r = r ^ f_function(l, k), l
    return l, r


def encrypt_block(plaintext, master_key, rounds):
    """Convenience wrapper taking/returning packed 24-bit blocks."""
    p = np.asarray(plaintext, dtype=np.uint32)
    l = (p >> np.uint32(HALF_BITS)) & np.uint32(HALF_MASK)
    r = p & np.uint32(HALF_MASK)
    l, r = encrypt(l, r, key_schedule(master_key, rounds))
    return (l << np.uint32(HALF_BITS)) | r


# --------------------------------------------------------------------------
# Mask helpers (used by the linear-trail search and by the attacks)
# --------------------------------------------------------------------------

def parity(x):
    """Bitwise parity (XOR of all bits) of a uint32 array, returned as uint8."""
    v = np.asarray(x, dtype=np.uint32).copy()
    v ^= v >> np.uint32(16)
    v ^= v >> np.uint32(8)
    v ^= v >> np.uint32(4)
    v ^= v >> np.uint32(2)
    v ^= v >> np.uint32(1)
    return (v & np.uint32(1)).astype(np.uint8)


def mask_dot(mask, value):
    """Inner product over GF(2): parity(mask & value)."""
    return parity(np.asarray(value, dtype=np.uint32) & np.uint32(mask))


def fold_expansion_mask(rho):
    """Pull an 18-bit mask on E(x) back to the induced 12-bit mask on x.

    E duplicates bits, so a mask on E's output folds by XOR onto x.
    """
    out = 0
    for m in range(EXPANDED_BITS):
        if (rho >> m) & 1:
            out ^= 1 << E_SRC[m]
    return out


def f_output_mask_to_sbox_mask(mu):
    """Mask on F's output (post-P)  ->  mask on the S-box layer output."""
    nu = 0
    for k in range(HALF_BITS):
        if (mu >> P_TABLE[k]) & 1:
            nu |= 1 << k
    return nu


def sbox_mask_to_f_output_mask(nu):
    """Inverse of :func:`f_output_mask_to_sbox_mask`."""
    mu = 0
    for k in range(HALF_BITS):
        if (nu >> k) & 1:
            mu |= 1 << P_TABLE[k]
    return mu


def active_sboxes_from_f_output_mask(mu):
    """Which S-boxes a given F-output mask touches, and their 4-bit out-masks."""
    nu = f_output_mask_to_sbox_mask(mu)
    out = {}
    for j in range(N_SBOX):
        nib = (nu >> (SBOX_OUT_BITS * j)) & 0xF
        if nib:
            out[j] = nib
    return out


def subkey_group(subkey, j):
    """The 6 subkey bits feeding S-box j, as an integer 0..63."""
    return (np.asarray(subkey, dtype=np.uint32) >> np.uint32(SBOX_IN_BITS * j)) & np.uint32(0x3F)


def sbox_input(x, j):
    """The 6-bit input of S-box j taken from the F-function input x.

    Same name and semantics in :mod:`des`, so the attack machinery is
    cipher-agnostic.
    """
    return (E_LUT[np.asarray(x, dtype=np.uint32)]
            >> np.uint32(SBOX_IN_BITS * j)) & np.uint32(0x3F)


def f_partial_output(x, j, key_group):
    """The F-output contribution of S-box j alone, given only its 6 key bits.

    Returns a 12-bit value whose only set bits are the P-image of S_j's output
    nibble.  This is the piece of F that the multi-bit attack can evaluate
    after guessing just six subkey bits.
    """
    x = np.asarray(x, dtype=np.uint32)
    g = (E_LUT[x] >> np.uint32(SBOX_IN_BITS * j)) & np.uint32(0x3F)
    s = SBOX_FLAT[j][g ^ np.asarray(key_group, dtype=np.uint32)]
    return P_LUT[s << np.uint32(SBOX_OUT_BITS * j)]


def random_blocks(rng, size):
    """Uniform random (L, R) halves."""
    l = rng.integers(0, 1 << HALF_BITS, size=size, dtype=np.uint64).astype(np.uint32)
    r = rng.integers(0, 1 << HALF_BITS, size=size, dtype=np.uint64).astype(np.uint32)
    return l, r


def random_master_keys(rng, size):
    return rng.integers(0, 1 << MASTER_KEY_BITS, size=size, dtype=np.uint64).astype(np.uint32)
