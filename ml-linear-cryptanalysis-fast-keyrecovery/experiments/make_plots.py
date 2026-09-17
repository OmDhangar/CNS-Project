"""Render every figure the report uses, from the CSVs the experiments write.

Usage::  python experiments/make_plots.py
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from common import ROOT, build, ensure_results   # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")

PALETTE = {
    "exhaustive": "#111827",
    "guided-wkr": "#2563eb",
    "random": "#d97706",
    "sequential": "#15803d",
    "guided-skopt": "#9333ea",
}


def _read(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------

def fig_budget_curve():
    rows = _read("exp3_budget_sensitivity.csv")
    if not rows:
        print("  (skipping budget curve: exp3 results missing)")
        return
    x = np.array([float(r["budget_frac"]) * 100 for r in rows])
    ex = float(rows[0]["exhaustive_success"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in ((axes[0], "success", "success rate (correct subkey)"),
                         (axes[1], "agreement", "agreement with exhaustive argmax")):
        ax.axhline(ex if key == "success" else 100.0, color=PALETTE["exhaustive"],
                   ls="--", lw=1.4,
                   label=f"exhaustive Algorithm 2 ({ex:.0f}%)" if key == "success"
                   else "exhaustive Algorithm 2")
        for m in ("guided-wkr", "sequential", "random"):
            col = f"{m}_{key}"
            if col not in rows[0]:
                continue
            y = np.array([float(r[col]) * 100 for r in rows])
            ax.plot(x, y, "o-", ms=3.5, lw=1.8, color=PALETTE[m], label=m)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 25, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        _style(ax, f"Budget sensitivity: {lab}",
               "query budget (% of |GK| = 4096, log scale)", f"{lab} (%)")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_budget_sensitivity.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_wrong_key_profile():
    """Predicted (offline) vs measured wrong-key response."""
    appr, setup, net, meta = build(verbose=False)
    from src.attacks.wrong_key_profile import WrongKeyProfile

    prof = WrongKeyProfile(setup)
    pred = np.abs(np.outer(prof.rho_back, prof.rho_front))    # [dkb, dkf]
    meas_path = os.path.join(ROOT, "artifacts", "wrong_key_profile.npy")

    ncols = 3 if os.path.exists(meas_path) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.0))

    im = axes[0].imshow(pred, cmap="magma", origin="lower", vmin=0, vmax=1)
    _style(axes[0], "Predicted |rho_f x rho_b|  (offline, 0 queries)",
           "front key difference  kf XOR kf*", "back key difference  kb XOR kb*")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].bar(np.arange(64) - 0.2, np.abs(prof.rho_front), width=0.4,
                color="#2563eb", label="|rho_f| (front S-box)")
    axes[1].bar(np.arange(64) + 0.2, np.abs(prof.rho_back), width=0.4,
                color="#d97706", label="|rho_b| (back S-box)")
    _style(axes[1], "Wrong-key response per side",
           "key difference (6 bits)", "|correlation retained|")
    axes[1].legend(fontsize=8, frameon=False)

    if ncols == 3:
        meas = np.load(meas_path)
        axes[2].scatter(pred.ravel(), meas.ravel(), s=6, alpha=0.35,
                        color="#2563eb", edgecolors="none")
        r = np.corrcoef(pred.ravel(), meas.ravel())[0, 1]
        _style(axes[2], f"Measured vs predicted (Pearson r = {r:.3f})",
               "predicted |rho_f x rho_b|", "measured mean z-score")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_wrong_key_profile.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_landscape_example():
    """One attack's full 64x64 CRD landscape, with the true key marked."""
    from src.attacks.multi_bit_bruteforce import run_exhaustive
    from src.distinguisher.crd import CandidateScorer

    appr, setup, net, meta = build(verbose=False)
    cfg_path = os.path.join(RESULTS, "exp3_config.json")
    nt = 917_504
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            nt = json.load(fh)["nt"]
    rng = np.random.default_rng(12345)
    data = setup.generate(rng, nt)
    sc = CandidateScorer(net, setup, data, 256, cache=True)
    r = run_exhaustive(sc, setup, data)
    s = r.scores
    mu = np.median(s)
    sigma = 1.4826 * np.median(np.abs(s - mu))
    z = ((s - mu) / sigma).reshape(64, 64)      # [kb, kf]
    kf, kb = data.true_candidate & 63, data.true_candidate >> 6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im = axes[0].imshow(z, cmap="magma", origin="lower")
    axes[0].plot(kf, kb, "o", mfc="none", mec="#22d3ee", mew=2, ms=14)
    _style(axes[0], "One attack: CRD score over all 4096 candidates",
           "guessed K_0 bits (front S-box)", "guessed K_7 bits (back S-box)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="robust z-score")

    axes[1].hist(z.ravel(), bins=80, color="#94a3b8", edgecolor="none")
    axes[1].axvline(z[kb, kf], color="#2563eb", lw=2,
                    label=f"correct subkey (z = {z[kb, kf]:.2f})")
    _style(axes[1], "Score distribution", "robust z-score", "candidates")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_score_landscape.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_stopping_tradeoff():
    rows = _read("calibration_stopping_posterior.csv")
    if not rows:
        print("  (skipping stopping trade-off: calibration results missing)")
        return
    q = np.array([float(r["queries_frac_median"]) * 100 for r in rows])
    ag = np.array([float(r["agreement"]) * 100 for r in rows])
    su = np.array([float(r["success"]) * 100 for r in rows])
    th = np.array([float(r["stop_posterior"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(q, ag, "o-", color=PALETTE["guided-wkr"], ms=4, label="agreement with exhaustive")
    ax.plot(q, su, "s-", color=PALETTE["random"], ms=4, label="success (correct subkey)")
    for i in range(0, len(th), 2):
        ax.annotate(f"{th[i]:g}", (q[i], ag[i]), fontsize=7,
                    textcoords="offset points", xytext=(3, -10), color="#475569")
    _style(ax, "Early stopping: posterior threshold trade-off",
           "median queries used (% of |GK|)", "rate (%)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_stopping_tradeoff.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_exp1_sweep():
    rows = _read("exp1_multi_bit.csv")
    if not rows:
        print("  (skipping exp1 sweep: results missing)")
        return
    x = np.array([float(r["N_times_t"]) for r in rows])
    ml = np.array([float(r["success_ml"]) * 100 for r in rows])
    cl = np.array([float(r["success_matsui"]) * 100 for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, ml, "o-", color=PALETTE["guided-wkr"], label="ML-aided Algorithm 2 (CRD)")
    ax.plot(x, cl, "s--", color=PALETTE["exhaustive"], label="Matsui's Algorithm 2")
    _style(ax, "Multi-bit key recovery on 8-round TinyDES-24",
           "data complexity N x t (plaintexts)", "success rate (%)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig0_data_complexity.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_query_distribution():
    """Where the guided search's budget actually goes, split by outcome."""
    rows = _read("exp2_per_trial.csv")
    if not rows:
        print("  (skipping query distribution: exp2 results missing)")
        return
    wkr = [r for r in rows if r["method"] == "guided-wkr"]
    if not wkr:
        return
    ok = np.array([int(r["n_evals"]) for r in wkr if r["success"] == "1"])
    bad = np.array([int(r["n_evals"]) for r in wkr if r["success"] != "1"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 4096, 41)
    axes[0].hist([ok, bad], bins=bins, stacked=True,
                 color=[PALETTE["guided-wkr"], "#cbd5e1"],
                 label=[f"attack succeeds (n={len(ok)})",
                        f"attack fails (n={len(bad)})"])
    axes[0].axvline(4096, color=PALETTE["exhaustive"], ls="--", lw=1.4,
                    label="exhaustive Algorithm 2 (always 4096)")
    _style(axes[0], "Distinguisher evaluations used per attack",
           "evaluations", "attacks")
    axes[0].legend(fontsize=8, frameon=False)

    data = [d for d in (ok, bad) if len(d)]
    labels = [l for l, d in zip(["succeeds", "fails"], (ok, bad)) if len(d)]
    axes[1].boxplot(data, labels=labels, vert=True, widths=0.5)
    axes[1].axhline(4096, color=PALETTE["exhaustive"], ls="--", lw=1.4)
    _style(axes[1], "Same data, by outcome", "", "evaluations")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig5_query_distribution.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def fig_phase2():
    rows = _read("exp4_des_l6_summary.csv")
    if not rows:
        print("  (skipping Phase-2 figure: exp4 results missing)")
        return
    order = ["exhaustive", "sequential-earlystop", "random", "guided-wkr"]
    rows = [r for m in order for r in rows if r["method"] == m]
    names = [r["method"] for r in rows]
    succ = [float(r["success_rate"]) * 100 for r in rows]
    frac = [float(r["evals_frac_median"]) * 100 for r in rows]
    cols = [PALETTE.get(m, "#64748b") for m in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))
    axes[0].bar(x, succ, color=cols)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    _style(axes[0], "Phase 2 (8-round DES, Matsui L6): success rate", "", "success (%)")
    axes[1].bar(x, frac, color=cols)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    _style(axes[1], "Median evaluations used", "", "% of |GK| = 4096")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig6_phase2_des.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  {out}")


def main():
    ensure_results()
    os.makedirs(FIGS, exist_ok=True)
    print("rendering figures:")
    fig_exp1_sweep()
    fig_wrong_key_profile()
    fig_landscape_example()
    fig_budget_curve()
    fig_stopping_tradeoff()
    fig_query_distribution()
    fig_phase2()


if __name__ == "__main__":
    main()
