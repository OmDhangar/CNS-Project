"""Training drivers for the neural distinguishers, with on-disk caching.

Trained networks land in ``artifacts/`` keyed by (framework, rounds, t, shape),
so experiments can be re-run without retraining.  Pass ``force=True`` (or the
``--force`` flag of the experiment scripts) to retrain.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import data_gen, model as model_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.normpath(os.path.join(_HERE, "..", "..", "artifacts"))


def _path(kind, rounds, t, width, depth, cipher="toy"):
    return os.path.join(
        ARTIFACTS, f"nd_{cipher}_{kind}_r{rounds}_t{t}_w{width}d{depth}.pt")


def _cipher_tag(obj):
    """Short cipher name for cache keys; obj is an AttackSetup or a cipher module."""
    cipher = getattr(obj, "cipher", obj)
    name = getattr(cipher, "__name__", "toy").rsplit(".", 1)[-1]
    return "toy" if name == "toy_feistel" else name


def _report(acc, ceiling, verbose):
    adv, adv_max = acc - 0.5, ceiling - 0.5
    if verbose:
        frac = (adv / adv_max * 100) if adv_max > 0 else float("nan")
        print(f"  val-acc {acc * 100:.3f}%   fixed-p Bayes reference "
              f"{ceiling * 100:.3f}%   advantage {adv * 100:.3f} of "
              f"{adv_max * 100:.3f} pp ({frac:.1f}%)")


def _finish(net, path, meta, acc, ceiling, verbose):
    meta["val_acc"] = acc
    meta["bayes_acc"] = ceiling
    meta["advantage"] = acc - 0.5
    meta["advantage_ceiling"] = ceiling - 0.5
    model_mod.save(net, path, meta)
    _report(acc, ceiling, verbose)
    return net, meta


def train_multi_bit(setup, t, samples_per_epoch=250_000, n_val=40_000, width=32,
                    depth=2, epochs=40, batch_size=512, lr=1e-3, seed=0,
                    force=False, verbose=True):
    """Train ND_r^t in the paper's Eq. 7 format (multi-bit key recovery)."""
    path = _path("multi", setup.appr.rounds, t, width, depth, _cipher_tag(setup))
    if os.path.exists(path) and not force:
        m, meta = model_mod.load(path)
        if verbose:
            print(f"[cache] {os.path.basename(path)}  val-acc "
                  f"{meta.get('val_acc', float('nan')) * 100:.3f}%")
        return m, meta

    t0 = time.time()
    rng = np.random.default_rng(seed)
    xva, yva = data_gen.make_multi_bit_dataset(setup, n_val, t, rng)

    def sample_fn(n, r):
        return data_gen.make_multi_bit_dataset(setup, n, t, r)

    net = model_mod.NeuralDistinguisher(t, width=width, depth=depth)
    acc = model_mod.train_online(net, sample_fn, xva, yva, epochs=epochs,
                                 samples_per_epoch=samples_per_epoch,
                                 batch_size=batch_size, lr=lr, verbose=verbose,
                                 seed=seed)
    ceiling = data_gen.optimal_accuracy(setup.appr.p, t, negatives_random=True)
    meta = {"kind": "multi", "cipher": _cipher_tag(setup),
            "rounds": setup.appr.rounds, "t": t,
            "samples_seen": samples_per_epoch * epochs, "n_val": n_val,
            "p_r": setup.appr.p, "train_seconds": time.time() - t0}
    return _finish(net, path, meta, acc, ceiling, verbose)


def train_one_bit(appr, t, samples_per_epoch=250_000, n_val=40_000, width=32,
                  depth=2, epochs=40, batch_size=512, lr=1e-3, seed=0,
                  force=False, verbose=True):
    """Train ND_r^t in the paper's Eq. 6 format (one-bit key recovery)."""
    path = _path("one", appr.rounds, t, width, depth, _cipher_tag(appr))
    if os.path.exists(path) and not force:
        m, meta = model_mod.load(path)
        if verbose:
            print(f"[cache] {os.path.basename(path)}  val-acc "
                  f"{meta.get('val_acc', float('nan')) * 100:.3f}%")
        return m, meta

    t0 = time.time()
    rng = np.random.default_rng(seed)
    xva, yva = data_gen.make_one_bit_dataset(appr, n_val, t, rng)

    def sample_fn(n, r):
        return data_gen.make_one_bit_dataset(appr, n, t, r)

    net = model_mod.NeuralDistinguisher(t, width=width, depth=depth)
    acc = model_mod.train_online(net, sample_fn, xva, yva, epochs=epochs,
                                 samples_per_epoch=samples_per_epoch,
                                 batch_size=batch_size, lr=lr, verbose=verbose,
                                 seed=seed)
    ceiling = data_gen.optimal_accuracy(appr.p, t, negatives_random=False)
    meta = {"kind": "one", "cipher": _cipher_tag(appr),
            "rounds": appr.rounds, "t": t,
            "samples_seen": samples_per_epoch * epochs, "n_val": n_val,
            "p_r": appr.p, "train_seconds": time.time() - t0}
    return _finish(net, path, meta, acc, ceiling, verbose)
