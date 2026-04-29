#!/usr/bin/env python3
"""
generate_gdlll_plots.py — Plots for the G-DLLL benchmark.

Reads  results/gdlll_final.json  (or gdlll.json) and writes figures to
figures/.  Handles truncated / incomplete JSON gracefully so that
plots can be generated while the benchmark is still running.

Figures produced
  Per-family 5-panel:  deep_benchmark_{gaussian,qary,goldstein}.pdf
    (a) operation count  (b) equiv-swap count W  (c) time  (d) δ₀  (e) variance

  Multi-family summaries:
    gdlll_equiv_swaps.pdf   — W vs dimension (all deep selectors)
    gdlll_speedup.pdf       — W reduction G-DLLL vs Deep-Var (%)
    gdlll_delta0.pdf        — root-Hermite factor δ₀
    gdlll_time.pdf          — wall-clock time
"""

import json
import os
import re
import sys
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
RESULT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style (verbatim from generate_plots.py) ──────────────────────────────────
plt.rcParams["font.family"] = "Ubuntu"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

_CMAP = mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])
PALETTE = [_CMAP(x) for x in np.linspace(0, 1, 8)]

ALG_COLORS = {
    "LLL": "#7f7f7f",
    "Deep-Var": PALETTE[0],
    "SS-GG": "#d67474",
    "Thermal-Adaptive": PALETTE[7],
    "G-DLLL": PALETTE[4],
}
ALG_MARKERS = {
    "LLL": "v",
    "Deep-Var": "o",
    "SS-GG": "s",
    "Thermal-Adaptive": "^",
    "G-DLLL": "D",
}
ALG_DISPLAY = {
    "LLL": "LLL (fplll)",
    "Deep-Var": "Deep-Var",
    "SS-GG": "SS-GG",
    "Thermal-Adaptive": "Thermal-Adaptive",
    "G-DLLL": "G-DLLL",
}

RC_PARAMS = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}
plt.rcParams.update(RC_PARAMS)


def _style_ax(ax):
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)


def save_fig(fig, name):
    for ax in fig.get_axes():
        _style_ax(ax)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Load data ─────────────────────────────────────────────────────────────────
def load_gdlll():
    path = os.path.join(RESULT_DIR, "gdlll_final.json")
    if not os.path.isfile(path):
        path = os.path.join(RESULT_DIR, "gdlll.json")
    if not os.path.isfile(path):
        print("ERROR: no gdlll*.json found.", file=sys.stderr)
        sys.exit(1)
    raw = open(path).read()
    # Fix known issues: double commas, nan/inf literals
    raw = re.sub(r",\s*,", ",", raw)
    raw = re.sub(r":-?nan\b", ":null", raw)
    raw = re.sub(r":-?inf\b", ":null", raw)
    # Handle truncated JSON: close any open braces
    depth = raw.count("{") - raw.count("}")
    if depth > 0:
        raw = raw.rstrip().rstrip(",")
        raw += "}" * depth
        print(
            f"  Note: JSON was truncated (closed {depth} braces). "
            f"Plotting partial results."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: Cannot parse JSON even after fixup: {e}", file=sys.stderr)
        sys.exit(1)


def extract(data, family, metric):
    """Return (dims, alg→{mean, std}) for a given family."""
    fam_data = data.get("results", data).get(family, {})
    dims = sorted(int(k) for k in fam_data.keys())
    algs_seen = set()
    for d in dims:
        algs_seen.update(fam_data[str(d)].keys())

    means = {a: [] for a in algs_seen}
    stds = {a: [] for a in algs_seen}
    for d in dims:
        cell = fam_data[str(d)]
        for a in algs_seen:
            if a in cell:
                v = cell[a].get(f"mean_{metric}", None)
                s = cell[a].get(f"std_{metric}", None)
                means[a].append(v if v is not None else np.nan)
                stds[a].append(s if s is not None else np.nan)
            else:
                means[a].append(np.nan)
                stds[a].append(np.nan)
    return dims, means, stds


ALGS = ["LLL", "Deep-Var", "SS-GG", "Thermal-Adaptive", "G-DLLL"]
DEEP_ALGS = ["Deep-Var", "SS-GG", "Thermal-Adaptive", "G-DLLL"]
FAMILIES = ["gaussian", "qary", "goldstein-mayer"]
FAM_LABEL = {
    "gaussian": "Gaussian",
    "qary": r"$q$-ary",
    "goldstein-mayer": "Goldstein-Mayer",
}


# ── Figure 1 : Equivalent-swap count W vs dimension ──────────────────────────
def plot_equiv_swaps(data):
    n_fam = sum(1 for f in FAMILIES if f in data.get("results", data))
    fig, axes = plt.subplots(1, n_fam, figsize=(4.2 * n_fam, 3.5), sharey=False)
    if n_fam == 1:
        axes = [axes]

    col = 0
    for family in FAMILIES:
        if family not in data.get("results", data):
            continue
        ax = axes[col]
        col += 1
        dims, means, stds = extract(data, family, "equiv_swaps")
        x = np.array(dims)
        for a in DEEP_ALGS:
            if a not in means:
                continue
            y = np.array(means[a])
            err = np.array(stds[a])
            mask = ~np.isnan(y) & (y > 0)
            if not mask.any():
                continue
            ax.errorbar(
                x[mask],
                y[mask],
                yerr=err[mask],
                color=ALG_COLORS.get(a, "k"),
                marker=ALG_MARKERS.get(a, "o"),
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=ALG_DISPLAY.get(a, a),
            )
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(r"Mean equiv.\ swaps $W$") if col == 1 else None
        ax.set_title(FAM_LABEL[family])
        ax.set_yscale("log")
        ax.legend(loc="upper left")

    fig.suptitle(
        r"Equivalent-swap cost $W = \sum_s (k_s - j_s)$ " r"(lower is better)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "gdlll_equiv_swaps.pdf")


# ── Figure 2 : Relative W reduction (G-DLLL vs Deep-Var) ─────────────────────
def plot_speedup(data):
    n_fam = sum(1 for f in FAMILIES if f in data.get("results", data))
    fig, axes = plt.subplots(1, n_fam, figsize=(4.0 * n_fam, 3.2), sharey=True)
    if n_fam == 1:
        axes = [axes]

    ax0 = axes[0]
    col = 0
    for family in FAMILIES:
        if family not in data.get("results", data):
            continue
        ax = axes[col]
        col += 1
        dims, means, stds = extract(data, family, "equiv_swaps")
        x = np.array(dims)
        W_dv = np.array(means.get("Deep-Var", [np.nan] * len(dims)))
        W_gd = np.array(means.get("G-DLLL", [np.nan] * len(dims)))
        s_dv = np.array(stds.get("Deep-Var", [np.nan] * len(dims)))
        s_gd = np.array(stds.get("G-DLLL", [np.nan] * len(dims)))

        ratio = (W_dv - W_gd) / np.where(W_dv > 0, W_dv, np.nan) * 100.0
        d_ratio_err = (
            np.sqrt(
                (s_gd / np.where(W_dv > 0, W_dv, np.nan)) ** 2
                + (W_gd * s_dv / np.where(W_dv > 0, W_dv, np.nan) ** 2) ** 2
            )
            * 100.0
        )
        mask = ~np.isnan(ratio)

        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        if mask.any():
            ax.fill_between(
                x[mask],
                ratio[mask] - d_ratio_err[mask],
                ratio[mask] + d_ratio_err[mask],
                color=ALG_COLORS["G-DLLL"],
                alpha=0.2,
            )
            ax.plot(
                x[mask],
                ratio[mask],
                color=ALG_COLORS["G-DLLL"],
                marker="D",
                markersize=4,
                linewidth=1.4,
                label="G-DLLL vs Deep-Var",
            )
        ax.set_xlabel(r"Dimension $d$")
        if col == 1:
            ax.set_ylabel(r"W reduction (\%)")
        ax.set_title(FAM_LABEL[family])

    fig.suptitle(
        r"G-DLLL work reduction vs Deep-Var (higher $= $ better)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "gdlll_speedup.pdf")


# ── Figure 3 : Root-Hermite factor δ₀ vs dimension ───────────────────────────
def plot_delta0(data):
    n_fam = sum(1 for f in FAMILIES if f in data.get("results", data))
    fig, axes = plt.subplots(1, n_fam, figsize=(4.0 * n_fam, 3.2), sharey=False)
    if n_fam == 1:
        axes = [axes]

    col = 0
    for family in FAMILIES:
        if family not in data.get("results", data):
            continue
        ax = axes[col]
        col += 1
        dims, means, stds = extract(data, family, "delta0")
        x = np.array(dims)
        for a in ALGS:
            if a not in means:
                continue
            y = np.array(means[a])
            err = np.array(stds[a])
            mask = ~np.isnan(y) & (y > 0)
            if not mask.any():
                continue
            ax.errorbar(
                x[mask],
                y[mask],
                yerr=err[mask],
                color=ALG_COLORS.get(a, "k"),
                marker=ALG_MARKERS.get(a, "o"),
                markersize=4,
                linewidth=1.4,
                capsize=2,
                label=ALG_DISPLAY.get(a, a),
            )
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(r"Mean $\delta_0$") if col == 1 else None
        ax.set_title(FAM_LABEL[family])
        ax.legend(loc="upper right")

    fig.suptitle(
        r"Output quality: root-Hermite factor $\delta_0$ " r"(lower is better)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "gdlll_delta0.pdf")


# ── Figure 4 : Wall-clock time comparison ────────────────────────────────────
def plot_time(data):
    n_fam = sum(1 for f in FAMILIES if f in data.get("results", data))
    fig, axes = plt.subplots(1, n_fam, figsize=(4.0 * n_fam, 3.2), sharey=False)
    if n_fam == 1:
        axes = [axes]

    col = 0
    for family in FAMILIES:
        if family not in data.get("results", data):
            continue
        ax = axes[col]
        col += 1
        dims, means, stds = extract(data, family, "time")
        x = np.array(dims)
        for a in ALGS:
            if a not in means:
                continue
            y = np.array(means[a])
            mask = ~np.isnan(y)
            if not mask.any():
                continue
            ax.plot(
                x[mask],
                y[mask],
                color=ALG_COLORS.get(a, "k"),
                marker=ALG_MARKERS.get(a, "o"),
                markersize=4,
                linewidth=1.4,
                label=ALG_DISPLAY.get(a, a),
            )
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(r"Mean wall-clock time (s)") if col == 1 else None
        ax.set_title(FAM_LABEL[family])
        ax.set_yscale("log")
        ax.legend(loc="upper left")

    fig.suptitle(r"Wall-clock time comparison", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "gdlll_time.pdf")


FAM_SHORT = {"gaussian": "gaussian", "qary": "qary", "goldstein-mayer": "goldstein"}


# ── Per-family 4-panel benchmarks (deep_benchmark_*.pdf) ─────────────────────
def plot_per_family(data, family):
    """5-panel figure: (a) ops, (b) equiv-swaps W, (c) time, (d) δ₀, (e) variance."""
    tag = FAM_SHORT.get(family, family)
    results = data.get("results", data)
    if family not in results:
        print(f"  Skipping {family} (not in results)")
        return

    fam_data = results[family]
    dims = sorted(int(k) for k in fam_data.keys())
    if not dims:
        return
    n_lat = 30
    se_fac = 1.0 / np.sqrt(n_lat)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.45)
    ax_ops = fig.add_subplot(gs[0, 0:2])
    ax_W = fig.add_subplot(gs[0, 2:4])
    ax_time = fig.add_subplot(gs[0, 4:6])
    ax_d0 = fig.add_subplot(gs[1, 1:3])
    ax_var = fig.add_subplot(gs[1, 3:5])

    fig.suptitle(
        f"{FAM_LABEL[family]} benchmark " rf"($\delta=0.99$, $n={n_lat}$)",
        fontsize=11,
        fontweight="bold",
    )

    def _plot_metric(ax, metric, alg_list, ylabel, title, yscale="linear"):
        for a in alg_list:
            dims_a, means_a, stds_a = extract(data, family, metric)
            if a not in means_a:
                continue
            x = np.array(dims_a)
            y = np.array(means_a[a])
            err = np.array(stds_a[a]) * se_fac
            mask = ~np.isnan(y) & (y > 0) if yscale == "log" else ~np.isnan(y)
            if not mask.any():
                continue
            ax.plot(
                x[mask],
                y[mask],
                marker=ALG_MARKERS.get(a, "o"),
                color=ALG_COLORS.get(a, "k"),
                label=ALG_DISPLAY.get(a, a),
                linewidth=1.8,
                markersize=5,
            )
            ax.fill_between(
                x[mask],
                np.maximum(
                    y[mask] - err[mask],
                    1e-6 if yscale == "log" else y[mask] - err[mask],
                ),
                y[mask] + err[mask],
                color=ALG_COLORS.get(a, "k"),
                alpha=0.15,
            )
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if yscale == "log":
            ax.set_yscale("log")
        ax.legend(fontsize=6.5)

    _plot_metric(
        ax_ops,
        "ops",
        DEEP_ALGS,
        "Deep insertions $N$",
        r"(a) Operation count ($\pm 1$\,SE)",
    )
    _plot_metric(
        ax_W,
        "equiv_swaps",
        DEEP_ALGS,
        r"Equiv.\ swaps $W$",
        r"(b) Total cascade work $W$ ($\pm 1$\,SE)",
        yscale="log",
    )
    _plot_metric(
        ax_time,
        "time",
        DEEP_ALGS,
        "Wall-clock time (s)",
        r"(c) Selector time ($\pm 1$\,SE)",
        yscale="log",
    )

    # (d) δ₀ with jittered x
    offsets = np.linspace(-0.6, 0.6, len(ALGS))
    for j_idx, a in enumerate(ALGS):
        dims_a, means_a, stds_a = extract(data, family, "delta0")
        if a not in means_a:
            continue
        x = np.array(dims_a, dtype=float) + offsets[j_idx]
        y = np.array(means_a[a])
        err = np.array(stds_a[a]) * se_fac
        mask = ~np.isnan(y) & (y > 0)
        if not mask.any():
            continue
        ax_d0.errorbar(
            x[mask],
            y[mask],
            yerr=err[mask],
            fmt=ALG_MARKERS.get(a, "o") + "-",
            color=ALG_COLORS.get(a, "k"),
            label=ALG_DISPLAY.get(a, a),
            linewidth=1.5,
            markersize=4,
            capsize=3,
            capthick=1,
        )
    ax_d0.set_xlabel(r"Dimension $d$")
    ax_d0.set_ylabel(r"$\delta_0$")
    ax_d0.set_title(r"(d) Output quality ($\pm 1$\,SE)")
    ax_d0.legend(fontsize=6.5)

    _plot_metric(
        ax_var,
        "final_var",
        ALGS,
        r"Final $\sum p_i^2$",
        r"(e) Fixed-point variance ($\pm 1$\,SE)",
    )

    save_fig(fig, f"deep_benchmark_{tag}.pdf")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating G-DLLL figures …")
    data = load_gdlll()

    # Per-family 4-panel benchmarks (deep_benchmark_*.pdf)
    for family in FAMILIES:
        plot_per_family(data, family)

    # Multi-family summary plots
    plot_equiv_swaps(data)
    plot_speedup(data)
    plot_delta0(data)
    plot_time(data)

    print("Done. Figures written to figures/.")
