"""DES with reduced-round support, vectorised over numpy arrays.

Phase 2 of the project points the *identical* guided-search code at real DES,
so this module exposes the same surface as :mod:`toy_feistel`:

    encrypt / decrypt / key_schedule / f_function
    E_LUT-style group extraction, SBOX_FLAT, mask helpers

Conventions
-----------
* Following the paper (Sect. "Brief description of Des"), the initial and final
  permutations IP / IP^-1 are omitted from the attack path: they are key-
  independent bijections and Matsui's expressions are stated on (L_0, R_0).
  :func:`encrypt_with_ip` is provided so the implementation can be checked
  against the FIPS test vector.
* Bit numbering is LSB-0, as in the paper: ``L[0]`` is the least significant
  bit.  The standard DES tables number bits 1..n from the *most* significant,
  so a standard position ``p`` of an ``n``-bit word is LSB-0 index ``n - p``.

Speed
-----
The expansion E has a useful structure in LSB-0 terms: the six bits feeding
S-box ``j`` are the contiguous field ``R[27-4j .. 32-4j] (mod 32)``, so E is
eight rotations rather than 48 bit extractions.  The S-box layer and P are
fused into eight 64-entry "SP-box" tables, exactly as in a fast software DES.
"""

from __future__ import annotations

import numpy as np

BLOCK_BITS = 64
HALF_BITS = 32
HALF_MASK = np.uint32(0xFFFFFFFF)
N_SBOX = 8
SBOX_IN_BITS = 6
SBOX_OUT_BITS = 4
EXPANDED_BITS = 48
SUBKEY_BITS = 48
MASTER_KEY_BITS = 64      # 56 effective; the 8 parity bits are ignored

# --------------------------------------------------------------------------
# Standard tables (1-based, MSB-first, exactly as in FIPS 46-3)
# --------------------------------------------------------------------------

IP = (
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7,
)

FP = (
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25,
)

E_TABLE = (
    32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1,
)

P_TABLE = (
    16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25,
)

PC1 = (
    57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4,
)

PC2 = (
    14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32,
)

KS_SHIFTS = (1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1)

_SBOX_ROWS = [
    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]],
]


def _flatten_sbox(rows):
    flat = np.zeros(64, dtype=np.uint32)
    for v in range(64):
        row = ((v >> 5) & 1) * 2 + (v & 1)
        col = (v >> 1) & 0xF
        flat[v] = rows[row][col]
    return flat


SBOX_FLAT = [_flatten_sbox(r) for r in _SBOX_ROWS]

# --------------------------------------------------------------------------
# LSB-0 derived tables
# --------------------------------------------------------------------------

# P as a bit permutation in LSB-0 form: P_LSB[k] is the destination index of
# S-layer output bit k.  Standard: P_TABLE[m] is the source position (1-based,
# MSB-first) of output position m+1.
P_LSB = [0] * 32
for _m, _src in enumerate(P_TABLE):
    P_LSB[32 - _src] = 32 - (_m + 1)
P_LSB = tuple(P_LSB)

# Rotation amount for extracting S-box j's 6-bit input from R (see docstring).
E_ROT = tuple((27 - 4 * j) % 32 for j in range(N_SBOX))


def _build_sp_luts():
    """SP[j][v] = P(S_j(v) placed in its nibble), as a 32-bit word."""
    out = []
    for j in range(N_SBOX):
        tab = np.zeros(64, dtype=np.uint32)
        for v in range(64):
            s = int(SBOX_FLAT[j][v])
            # S-box j's 4-bit output occupies S-layer output positions
            # 4j+1 .. 4j+4 (1-based, MSB-first) -> LSB-0 indices 31-4j .. 28-4j,
            # with the nibble's MSB at 31-4j.
            word = 0
            for b in range(4):                   # b = 0 is the nibble's MSB
                if (s >> (3 - b)) & 1:
                    word |= 1 << P_LSB[31 - 4 * j - b]
            tab[v] = word
        out.append(tab)
    return out


SP_LUT = _build_sp_luts()


# --------------------------------------------------------------------------
# Core primitives
# --------------------------------------------------------------------------

def _rotr32(x, n):
    n = int(n) % 32
    if n == 0:
        return x
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))) & HALF_MASK


def sbox_input(x, j):
    """The 6-bit input of S-box j taken from the F-function input x.

    Same name and semantics in :mod:`toy_feistel`, so the attack machinery is
    cipher-agnostic.
    """
    return _rotr32(np.asarray(x, dtype=np.uint32), E_ROT[j]) & np.uint32(0x3F)


def sbox_inputs(r):
    """The eight 6-bit S-box inputs of E(R), as a list of uint32 arrays."""
    return [sbox_input(r, j) for j in range(N_SBOX)]


def subkey_group(subkey, j):
    """The 6 subkey bits feeding S-box j, as an integer 0..63.

    The packed 48-bit subkey stores S-box j's six bits at positions
    42-6j .. 47-6j (LSB-0), with 47-6j the most significant, matching the
    order in which PC-2 produces them.
    """
    return (np.asarray(subkey, dtype=np.uint64)
            >> np.uint64(42 - 6 * j)) & np.uint64(0x3F)


def f_function(r, subkey):
    """F(R, K) = P(S(E(R) ^ K))."""
    r = np.asarray(r, dtype=np.uint32)
    k = np.asarray(subkey, dtype=np.uint64)
    out = np.zeros(np.broadcast_shapes(r.shape, k.shape), dtype=np.uint32)
    for j in range(N_SBOX):
        g = _rotr32(r, E_ROT[j]) & np.uint32(0x3F)
        kj = ((k >> np.uint64(42 - 6 * j)) & np.uint64(0x3F)).astype(np.uint32)
        out |= SP_LUT[j][g ^ kj]
    return out


def f_partial_output(r, j, key_group):
    """The F-output contribution of S-box j alone, given only its 6 key bits."""
    r = np.asarray(r, dtype=np.uint32)
    g = _rotr32(r, E_ROT[j]) & np.uint32(0x3F)
    return SP_LUT[j][g ^ np.asarray(key_group, dtype=np.uint32)]


def key_schedule(master_key, rounds):
    """Expand a 64-bit DES key into `rounds` packed 48-bit subkeys (uint64)."""
    mk = np.asarray(master_key, dtype=np.uint64)
    # PC-1: 64 -> 56, giving C (upper 28) and D (lower 28).
    cd = np.zeros_like(mk)
    for m, src in enumerate(PC1):
        bit = (mk >> np.uint64(64 - src)) & np.uint64(1)
        cd |= bit << np.uint64(55 - m)
    c = (cd >> np.uint64(28)) & np.uint64(0x0FFFFFFF)
    d = cd & np.uint64(0x0FFFFFFF)

    subkeys = []
    for i in range(rounds):
        s = int(KS_SHIFTS[i % len(KS_SHIFTS)])
        c = ((c << np.uint64(s)) | (c >> np.uint64(28 - s))) & np.uint64(0x0FFFFFFF)
        d = ((d << np.uint64(s)) | (d >> np.uint64(28 - s))) & np.uint64(0x0FFFFFFF)
        cd = (c << np.uint64(28)) | d
        k = np.zeros_like(cd)
        for m, src in enumerate(PC2):
            bit = (cd >> np.uint64(56 - src)) & np.uint64(1)
            k |= bit << np.uint64(47 - m)
        subkeys.append(k)
    return subkeys


def encrypt(left, right, subkeys):
    """Feistel rounds only (no IP / IP^-1), matching the paper's convention."""
    l = np.asarray(left, dtype=np.uint32)
    r = np.asarray(right, dtype=np.uint32)
    for k in subkeys:
        l, r = r, l ^ f_function(r, k)
    return l, r


def decrypt(left, right, subkeys):
    l = np.asarray(left, dtype=np.uint32)
    r = np.asarray(right, dtype=np.uint32)
    for k in reversed(subkeys):
        l, r = r ^ f_function(l, k), l
    return l, r


def _permute64(x, table):
    x = np.asarray(x, dtype=np.uint64)
    out = np.zeros_like(x)
    n = len(table)
    for m, src in enumerate(table):
        out |= ((x >> np.uint64(64 - src)) & np.uint64(1)) << np.uint64(n - 1 - m)
    return out


def encrypt_with_ip(plaintext, master_key, rounds=16):
    """Full standard DES including IP and IP^-1, for test-vector validation.

    Note the final swap: standard DES applies IP^-1 to (R_16 || L_16).
    """
    p = _permute64(plaintext, IP)
    l = ((p >> np.uint64(32)) & np.uint64(HALF_MASK)).astype(np.uint32)
    r = (p & np.uint64(HALF_MASK)).astype(np.uint32)
    l, r = encrypt(l, r, key_schedule(master_key, rounds))
    pre = (r.astype(np.uint64) << np.uint64(32)) | l.astype(np.uint64)
    return _permute64(pre, FP)


# --------------------------------------------------------------------------
# Mask helpers (same names/semantics as toy_feistel)
# --------------------------------------------------------------------------

def parity(x):
    v = np.asarray(x, dtype=np.uint64).copy()
    for sh in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(sh)
    return (v & np.uint64(1)).astype(np.uint8)


def mask_dot(mask, value):
    return parity(np.asarray(value, dtype=np.uint64) & np.uint64(mask))


def mask_from_bits(bits):
    """Build a mask from a list of LSB-0 bit indices, as the paper writes them."""
    m = 0
    for b in bits:
        m |= 1 << int(b)
    return m


def f_output_mask_to_sbox_mask(mu):
    """Mask on F's output (post-P) -> mask on the S-box layer output."""
    nu = 0
    for k in range(32):
        if (mu >> P_LSB[k]) & 1:
            nu |= 1 << k
    return nu


def sbox_mask_to_f_output_mask(nu):
    mu = 0
    for k in range(32):
        if (nu >> k) & 1:
            mu |= 1 << P_LSB[k]
    return mu


def active_sboxes_from_f_output_mask(mu):
    """{j: 4-bit output mask} for the S-boxes an F-output mask touches."""
    nu = f_output_mask_to_sbox_mask(mu)
    out = {}
    for j in range(N_SBOX):
        nib = 0
        for b in range(4):                      # b = 0 is the nibble's MSB
            if (nu >> (31 - 4 * j - b)) & 1:
                nib |= 1 << (3 - b)
        if nib:
            out[j] = nib
    return out


def random_blocks(rng, size):
    l = rng.integers(0, 1 << 32, size=size, dtype=np.uint64).astype(np.uint32)
    r = rng.integers(0, 1 << 32, size=size, dtype=np.uint64).astype(np.uint32)
    return l, r


def random_master_keys(rng, size):
    return rng.integers(0, 1 << 63, size=size, dtype=np.uint64) * np.uint64(2) \
        + rng.integers(0, 2, size=size, dtype=np.uint64)
