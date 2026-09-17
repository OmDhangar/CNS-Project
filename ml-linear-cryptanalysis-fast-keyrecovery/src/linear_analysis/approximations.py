"""Cached linear approximations of TinyDES-24 (the toy analogue of Matsui's Table 4).

The trail search in :mod:`bias_search` is deterministic, but re-running it plus
the Monte-Carlo verification on every experiment is wasteful, so the result is
cached as JSON under ``artifacts/``.  Delete that file (or pass
``refresh=True``) to recompute.
"""

from __future__ import annotations

import json
import os

from .bias_search import LinearApproximation, best_approximation

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "artifacts", "toy_approximations.json"))

_FIELDS = ("rounds", "a_in", "b_in", "a_out", "b_out", "key_masks",
           "corr_theory", "u_sequence", "p_measured", "p_measured_n")


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def _to_dict(a: LinearApproximation):
    return {f: (list(getattr(a, f)) if isinstance(getattr(a, f), tuple) else getattr(a, f))
            for f in _FIELDS}


def _from_dict(d):
    return LinearApproximation(
        rounds=d["rounds"], a_in=d["a_in"], b_in=d["b_in"],
        a_out=d["a_out"], b_out=d["b_out"],
        key_masks=tuple(d["key_masks"]), corr_theory=d["corr_theory"],
        u_sequence=tuple(d["u_sequence"]),
        p_measured=d["p_measured"], p_measured_n=d["p_measured_n"],
    )


def get_approximation(rounds, refresh=False, max_active=1, n_measure=1 << 24, verbose=False):
    """Return the best verified `rounds`-round approximation of TinyDES-24."""
    cache = _load_cache()
    key = str(rounds)
    if not refresh and key in cache:
        return _from_dict(cache[key])
    appr = best_approximation(rounds, max_active=max_active, n_measure=n_measure,
                              verbose=verbose)
    cache[key] = _to_dict(appr)
    _save_cache(cache)
    return appr


def build_all(rounds_list=(3, 4, 5, 6, 7), n_measure=1 << 24, refresh=True):
    out = {}
    for r in rounds_list:
        a = get_approximation(r, refresh=refresh, n_measure=n_measure)
        out[r] = a
        print(a.describe())
        print()
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--measure", type=int, default=1 << 24)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    build_all(args.rounds, n_measure=args.measure, refresh=args.refresh)
