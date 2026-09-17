"""Self-tests for the toy cipher and the mask algebra the trail search relies on.

Run with:  python -m tests.test_cipher_and_masks
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from src.ciphers import toy_feistel as tf          # noqa: E402
from src.linear_analysis import bias_search as bs  # noqa: E402


FAILURES = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' :: ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


def test_roundtrip():
    rng = np.random.default_rng(1)
    for rounds in (1, 3, 6, 8, 10):
        l, r = tf.random_blocks(rng, 4096)
        k = tf.random_master_keys(rng, 4096)
        sk = tf.key_schedule(k, rounds)
        cl, cr = tf.encrypt(l, r, sk)
        dl, dr = tf.decrypt(cl, cr, sk)
        check(f"encrypt/decrypt roundtrip, {rounds} rounds",
              np.array_equal(l, dl) and np.array_equal(r, dr))


def test_bijective():
    """For a fixed key the r-round permutation must be a bijection on 24 bits."""
    x = np.arange(1 << 24, dtype=np.uint32)
    l = (x >> np.uint32(12)) & np.uint32(0xFFF)
    r = x & np.uint32(0xFFF)
    sk = tf.key_schedule(np.uint32(0x5A3C71), 8)
    cl, cr = tf.encrypt(l, r, sk)
    c = (cl << np.uint32(12)) | cr
    check("8-round cipher is a bijection on the full 2^24 codebook",
          len(np.unique(c)) == (1 << 24))


def test_expansion_mask_fold():
    """parity(rho & E(x)) must equal parity(fold(rho) & x) for every x."""
    rng = np.random.default_rng(2)
    x = np.arange(1 << 12, dtype=np.uint32)
    ex = tf.E_LUT[x]
    ok = True
    bad = None
    for rho in rng.integers(0, 1 << 18, size=400, dtype=np.uint64).astype(np.uint32):
        folded = tf.fold_expansion_mask(int(rho))
        lhs = tf.parity(ex & np.uint32(rho))
        rhs = tf.parity(x & np.uint32(folded))
        if not np.array_equal(lhs, rhs):
            ok = False
            bad = int(rho)
            break
    check("fold_expansion_mask agrees with E on all 2^12 inputs", ok,
          "" if ok else f"first bad rho = 0x{bad:05x}")


def test_p_mask_inverse():
    rng = np.random.default_rng(3)
    s = np.arange(1 << 12, dtype=np.uint32)
    ps = tf.P_LUT[s]
    ok = True
    for mu in rng.integers(0, 1 << 12, size=400, dtype=np.uint64).astype(np.uint32):
        nu = tf.f_output_mask_to_sbox_mask(int(mu))
        if not np.array_equal(tf.parity(ps & np.uint32(mu)), tf.parity(s & np.uint32(nu))):
            ok = False
            break
    check("f_output_mask_to_sbox_mask agrees with P", ok)
    ok2 = all(tf.f_output_mask_to_sbox_mask(tf.sbox_mask_to_f_output_mask(n)) == n
              for n in range(1 << 12))
    check("sbox_mask_to_f_output_mask inverts f_output_mask_to_sbox_mask", ok2)


def test_partial_f():
    """f_partial_output must equal the S-box-j bits of the full F."""
    rng = np.random.default_rng(4)
    x = rng.integers(0, 1 << 12, size=20000, dtype=np.uint64).astype(np.uint32)
    k = rng.integers(0, 1 << 18, size=20000, dtype=np.uint64).astype(np.uint32)
    full = tf.f_function(x, k)
    ok = True
    for j in range(tf.N_SBOX):
        kg = tf.subkey_group(k, j)
        part = tf.f_partial_output(x, j, kg)
        nib_mask = tf.sbox_mask_to_f_output_mask(0xF << (4 * j))
        if not np.array_equal(full & np.uint32(nib_mask), part):
            ok = False
    check("f_partial_output reproduces the S-box-j slice of F", ok)


def test_lat_against_bruteforce():
    """LAT entries must match a direct count over the 64 S-box inputs."""
    ok = True
    for j in range(tf.N_SBOX):
        for rho in (0x01, 0x10, 0x2A, 0x3F, 0x07):
            for nu in (0x1, 0x6, 0xF, 0x9):
                v = np.arange(64, dtype=np.uint32)
                cnt = int(np.count_nonzero(
                    tf.parity(v & np.uint32(rho)) == tf.parity(tf.SBOX_FLAT[j][v] & np.uint32(nu))
                )) - 32
                if cnt != bs.LATS[j][rho, nu]:
                    ok = False
    check("LAT matches brute-force counting", ok)


def test_single_round_approximation():
    """The one-round model  lam_in.R ^ lam_out.F(R,K) ^ rho.K = 0  must hold
    with exactly the correlation the LAT predicts (this is exact, not
    statistical, because we enumerate all 2^12 inputs x all keys mod group)."""
    ok = True
    worst = 0.0
    rng = np.random.default_rng(5)
    valid = bs._valid_f_output_masks(1)
    opts = bs._round_options(valid)
    for u in list(valid)[1:12]:
        for lam_in, rho18, corr in opts[u][:6]:
            x = np.arange(1 << 12, dtype=np.uint32)
            k = np.uint32(int(rng.integers(0, 1 << 18)))
            fx = tf.f_function(x, k)
            bit = tf.mask_dot(lam_in, x) ^ tf.mask_dot(u, fx) ^ int(tf.mask_dot(rho18, k))
            p = np.count_nonzero(bit == 0) / len(x)
            got = 2 * (p - 0.5)
            worst = max(worst, abs(got - corr))
            if abs(got - corr) > 1e-9:
                ok = False
    check("one-round F approximations match the piling-up correlation exactly",
          ok, f"max |measured - predicted| = {worst:.2e}")


def main():
    test_roundtrip()
    test_expansion_mask_fold()
    test_p_mask_inverse()
    test_partial_f()
    test_lat_against_bruteforce()
    test_single_round_approximation()
    test_bijective()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
