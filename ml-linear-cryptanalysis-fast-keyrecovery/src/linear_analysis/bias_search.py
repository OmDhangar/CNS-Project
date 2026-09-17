"""Search for the best linear approximations of TinyDES-24.

This is the toy-cipher analogue of Matsui's Table 4 (the L3..L6 expressions for
DES).  We build the linear approximation table (LAT) of every S-box, then run a
DP / beam search over linear *trails* through the Feistel network, exactly as
Matsui (1993, 1994) did for DES.

Trail parametrisation
---------------------
For a Feistel round  L_{i+1} = R_i,  R_{i+1} = L_i XOR F(R_i, K_i), writing the
state mask at position i as (a_i, b_i) on (L_i, R_i), the term a_i . L_i can
only cancel if  b_{i+1} = a_i.  Setting u_i := a_i this gives

    state mask at position i   = (u_i, u_{i-1})
    F output mask  in round i  = u_i
    F input  mask  in round i  = u_{i-1} XOR u_{i+1}

so a trail over rounds s .. e-1 is just the sequence u_{s-1}, u_s, ..., u_e and
the DP state is the sliding pair (u_{i-1}, u_i).

Restrictions (documented, and the same ones Matsui used for DES)
----------------------------------------------------------------
* At most `max_active` active S-boxes per round (default 1).
* u_{s-1} and u_e are constrained to have exactly one active S-box.  That is
  not a loss for us: it is precisely the condition that makes the two-sided
  key-recovery attack of Fig. 1/Fig. 5 of the paper cost 6 + 6 guessed subkey
  bits, i.e. a candidate space of 2**12 = 4096 like the paper's 8-round DES
  attack.

Correlations are combined with the piling-up lemma and then *verified
empirically* against the real cipher, so a wrong sign or index convention
cannot silently propagate into the attacks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..ciphers import toy_feistel as tf


# --------------------------------------------------------------------------
# Linear approximation tables
# --------------------------------------------------------------------------

def sbox_lat(j):
    """LAT[rho, nu] = #{v : rho.v == nu.S_j(v)} - 32, for 6-bit rho and 4-bit nu."""
    n_in = 1 << tf.SBOX_IN_BITS
    n_out = 1 << tf.SBOX_OUT_BITS
    v = np.arange(n_in, dtype=np.uint32)
    sv = tf.SBOX_FLAT[j][v]
    lat = np.zeros((n_in, n_out), dtype=np.int32)
    for rho in range(n_in):
        pin = tf.parity(v & np.uint32(rho))
        for nu in range(n_out):
            pout = tf.parity(sv & np.uint32(nu))
            lat[rho, nu] = int(np.count_nonzero(pin == pout)) - n_in // 2
    return lat


LATS = [sbox_lat(j) for j in range(tf.N_SBOX)]


def lat_correlation(j, rho, nu):
    """Correlation c = 2 * (Pr[rho.v = nu.S(v)] - 1/2), in [-1, 1]."""
    return LATS[j][rho, nu] / (1 << (tf.SBOX_IN_BITS - 1))


# --------------------------------------------------------------------------
# Valid F-output masks
# --------------------------------------------------------------------------

def _valid_f_output_masks(max_active):
    """F-output masks (post-P) whose pre-P form has <= max_active active nibbles.

    Returns {mask: n_active}.
    """
    out = {0: 0}
    for n in range(1, max_active + 1):
        for js in itertools.combinations(range(tf.N_SBOX), n):
            for nibs in itertools.product(range(1, 1 << tf.SBOX_OUT_BITS), repeat=n):
                nu = 0
                for j, nib in zip(js, nibs):
                    nu |= nib << (tf.SBOX_OUT_BITS * j)
                out[tf.sbox_mask_to_f_output_mask(nu)] = n
    return out


def _round_options(valid_u):
    """For every allowed F-output mask u, all one-round F approximations.

    Returns {u: [(lambda_in, rho18, correlation), ...]}, where lambda_in is the
    induced 12-bit mask on F's input and rho18 is the 18-bit mask on the round
    subkey (the subkey is XORed straight onto E(x), so the subkey mask equals
    the S-box-layer input mask).
    """
    opts = {}
    for u in valid_u:
        if u == 0:
            opts[u] = [(0, 0, 1.0)]
            continue
        active = tf.active_sboxes_from_f_output_mask(u)
        per_box = []
        for j, nib in active.items():
            choices = []
            for rho in range(1, 1 << tf.SBOX_IN_BITS):
                c = lat_correlation(j, rho, nib)
                if c != 0.0:
                    choices.append((j, rho, c))
            per_box.append(choices)
        combos = []
        for pick in itertools.product(*per_box):
            rho18 = 0
            corr = 1.0
            for j, rho, c in pick:
                rho18 |= rho << (tf.SBOX_IN_BITS * j)
                corr *= c
            combos.append((tf.fold_expansion_mask(rho18), rho18, corr))
        opts[u] = combos
    return opts


# --------------------------------------------------------------------------
# Trail representation
# --------------------------------------------------------------------------

@dataclass
class LinearApproximation:
    """An r-round linear approximation  a_in.L_in ^ b_in.R_in ^ a_out.L_out ^
    b_out.R_out ^ gamma.K = 0, holding with probability `p`.

    `key_masks[i]` is the 18-bit mask applied to the subkey of the i-th round
    *of the approximation* (i = 0 is the approximation's first round).
    """

    rounds: int
    a_in: int
    b_in: int
    a_out: int
    b_out: int
    key_masks: tuple
    corr_theory: float
    u_sequence: tuple = ()
    p_measured: float = float("nan")
    p_measured_n: int = 0
    # Which cipher the masks belong to.  Defaults to the toy cipher; Phase 2
    # binds :mod:`src.ciphers.des` here so the same class carries Matsui's
    # approximations.  Deliberately excluded from the JSON cache in
    # ``approximations._FIELDS``.
    cipher: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.cipher is None:
            self.cipher = tf

    # -- derived -----------------------------------------------------------
    @property
    def bias_theory(self):
        return abs(self.corr_theory) / 2.0

    @property
    def p_theory(self):
        return 0.5 + self.corr_theory / 2.0

    @property
    def p(self):
        """Best available probability estimate (measured if present)."""
        return self.p_measured if np.isfinite(self.p_measured) else self.p_theory

    @property
    def bias(self):
        return abs(self.p - 0.5)

    @property
    def in_active_sbox(self):
        """Index of the single S-box that b_in touches (front key guess)."""
        act = self.cipher.active_sboxes_from_f_output_mask(self.b_in)
        return next(iter(act)) if len(act) == 1 else None

    @property
    def out_active_sbox(self):
        """Index of the single S-box that a_out touches (back key guess)."""
        act = self.cipher.active_sboxes_from_f_output_mask(self.a_out)
        return next(iter(act)) if len(act) == 1 else None

    def state_parity(self, l_in, r_in, l_out, r_out):
        """a_in.L_in ^ b_in.R_in ^ a_out.L_out ^ b_out.R_out (the data part)."""
        c = self.cipher
        return (c.mask_dot(self.a_in, l_in) ^ c.mask_dot(self.b_in, r_in)
                ^ c.mask_dot(self.a_out, l_out) ^ c.mask_dot(self.b_out, r_out))

    def key_parity(self, subkeys, first_round):
        """gamma.K, given the cipher's full subkey list and the round index at
        which this approximation starts."""
        acc = None
        for i, km in enumerate(self.key_masks):
            if km == 0:
                continue
            bit = self.cipher.mask_dot(km, subkeys[first_round + i])
            acc = bit if acc is None else (acc ^ bit)
        if acc is None:
            return np.zeros(np.shape(subkeys[0]), dtype=np.uint8)
        return acc

    def describe(self):
        act_in = self.cipher.active_sboxes_from_f_output_mask(self.b_in)
        act_out = self.cipher.active_sboxes_from_f_output_mask(self.a_out)
        name = getattr(self.cipher, "__name__", "cipher").rsplit(".", 1)[-1]
        name = "TinyDES-24" if name == "toy_feistel" else name.upper()
        lines = [
            f"L{self.rounds}: {self.rounds}-round linear approximation of {name}",
            f"  a_in  (mask on L_in)  = 0x{self.a_in:03x}",
            f"  b_in  (mask on R_in)  = 0x{self.b_in:03x}   active S-box(es) {sorted(act_in)}",
            f"  a_out (mask on L_out) = 0x{self.a_out:03x}   active S-box(es) {sorted(act_out)}",
            f"  b_out (mask on R_out) = 0x{self.b_out:03x}",
            f"  key masks per round   = {[f'0x{k:05x}' for k in self.key_masks]}",
            f"  piling-up correlation = {self.corr_theory:+.6g}"
            f"   (p = {self.p_theory:.6f}, bias = {self.bias_theory:.6g}"
            f" = 2^{np.log2(self.bias_theory):.2f})" if self.bias_theory > 0 else "",
        ]
        if np.isfinite(self.p_measured):
            b = abs(self.p_measured - 0.5)
            lines.append(
                f"  MEASURED p            = {self.p_measured:.6f}"
                f"   (bias = {b:.6g} = 2^{np.log2(b):.2f}, {self.p_measured_n:,} samples)"
            )
        return "\n".join(x for x in lines if x)


# --------------------------------------------------------------------------
# Trail search
# --------------------------------------------------------------------------

def search_best_trails(rounds, max_active=1, beam=4000, top=10):
    """Beam-search the best `rounds`-round linear trails.

    Returns a list of :class:`LinearApproximation`, best (largest |corr|) first.
    """
    valid_u = _valid_f_output_masks(max_active)
    single_active = [u for u, n in valid_u.items() if n == 1]
    opts = _round_options(valid_u)

    # state: (u_prev, u_cur) -> (abs_corr, corr, key_masks tuple, u_seq tuple)
    states = {}
    for u_prev in single_active:
        for u_cur in valid_u:
            states[(u_prev, u_cur)] = (1.0, 1.0, (), (u_prev, u_cur))

    for step in range(rounds):
        last = step == rounds - 1
        nxt = {}
        for (u_prev, u_cur), (acorr, corr, kms, useq) in states.items():
            for lam_in, rho18, c in opts[u_cur]:
                u_next = lam_in ^ u_prev
                if last:
                    if u_next not in valid_u or valid_u[u_next] != 1:
                        continue
                else:
                    if u_next not in valid_u:
                        continue
                nc = corr * c
                if nc == 0.0:
                    continue
                key = (u_cur, u_next)
                cand = (abs(nc), nc, kms + (rho18,), useq + (u_next,))
                cur = nxt.get(key)
                if cur is None or cand[0] > cur[0]:
                    nxt[key] = cand
        if not nxt:
            return []
        # prune
        states = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:beam])

    # Collect: state (u_{e-1}, u_e); masks are
    #   input  state s: (a_s, b_s) = (u_s, u_{s-1})
    #   output state e: (a_e, b_e) = (u_e, u_{e-1})
    results = []
    for (u_em1, u_e), (acorr, corr, kms, useq) in states.items():
        u_sm1, u_s = useq[0], useq[1]
        appr = LinearApproximation(
            rounds=rounds,
            a_in=u_s, b_in=u_sm1,
            a_out=u_e, b_out=u_em1,
            key_masks=kms,
            corr_theory=corr,
            u_sequence=useq,
        )
        if appr.in_active_sbox is None or appr.out_active_sbox is None:
            continue
        results.append(appr)
    results.sort(key=lambda a: -abs(a.corr_theory))
    return results[:top]


# --------------------------------------------------------------------------
# Empirical verification
# --------------------------------------------------------------------------

def measure_probability(appr, n_samples=1 << 22, batch=1 << 20, seed=0,
                        first_round=0, total_rounds=None):
    """Measure Pr[approximation holds] on the real cipher by Monte-Carlo.

    A fresh random master key is drawn for every sample, matching how the
    attack data is generated (the approximation's probability is taken over
    both plaintexts and keys).
    """
    if total_rounds is None:
        total_rounds = first_round + appr.rounds
    c = appr.cipher
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < n_samples:
        m = min(batch, n_samples - done)
        keys = c.random_master_keys(rng, m)
        subkeys = c.key_schedule(keys, total_rounds)
        l, r = c.random_blocks(rng, m)
        li, ri = l, r
        # Run the approximation's own rounds.
        lo, ro = c.encrypt(li, ri, subkeys[first_round:first_round + appr.rounds])
        bit = appr.state_parity(li, ri, lo, ro) ^ appr.key_parity(subkeys, first_round)
        hits += int(np.count_nonzero(bit == 0))
        done += m
    appr.p_measured = hits / n_samples
    appr.p_measured_n = n_samples
    return appr.p_measured


def best_approximation(rounds, max_active=1, beam=4000, n_measure=1 << 22,
                       seed=0, verbose=True, n_candidates=6):
    """Search, then empirically verify, and return the best approximation."""
    cands = search_best_trails(rounds, max_active=max_active, beam=beam,
                               top=n_candidates)
    if not cands:
        raise RuntimeError(f"no {rounds}-round trail found")
    for a in cands:
        measure_probability(a, n_samples=n_measure, seed=seed)
    cands.sort(key=lambda a: -abs(a.p_measured - 0.5))
    if verbose:
        for i, a in enumerate(cands):
            print(f"--- candidate {i} ---")
            print(a.describe())
    return cands[0]


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="TinyDES-24 linear approximation search")
    ap.add_argument("--rounds", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--max-active", type=int, default=1)
    ap.add_argument("--measure", type=int, default=1 << 22)
    args = ap.parse_args()

    for r in args.rounds:
        print("=" * 72)
        print(f"{r}-round trails (<= {args.max_active} active S-box per round)")
        print("=" * 72)
        best_approximation(r, max_active=args.max_active, n_measure=args.measure)
        print()
