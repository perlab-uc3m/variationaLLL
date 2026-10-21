#!/usr/bin/env python3
"""Per-family benchmark figures for the revised manuscript.

Keeps the layout and style of the submitted version (generate_gdlll_plots.py:
five panels, green-to-blue palette, Ubuntu font, bold titles, +-1 SE bands)
and changes only what the revision requires:
  * LLL (fplll) is shown in every panel, as Reviewer 1 asked;
  * panel (a) is the insertion count N (log scale, since LLL is far larger);
  * LLL's W is its measured total move depth (lll_depth_*.json);
  * panel (e) is labelled as the final profile variance V/d;
  * statistics come from the stored per-sample values (SE with ddof=1).

Usage: python3 plot_review_figures.py RESULTS_DIR OUT_DIR
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "Ubuntu"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold", "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 300,
})
_CMAP = mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])
PALETTE = [_CMAP(x) for x in np.linspace(0, 1, 8)]
COLORS = {"LLL": "#7f7f7f", "Deep-Var": PALETTE[0], "SS-GG": "#d67474",
          "Thermal-Adaptive": PALETTE[7], "G-DLLL": PALETTE[4]}
MARKERS = {"LLL": "v", "Deep-Var": "o", "SS-GG": "s", "Thermal-Adaptive": "^", "G-DLLL": "D"}
DISPLAY = {"LLL": "LLL (fplll)", "Deep-Var": "Deep-Var", "SS-GG": "SS-GG",
           "Thermal-Adaptive": "Thermal-Adaptive", "G-DLLL": "G-DLLL"}
ALGS = ["LLL", "Deep-Var", "SS-GG", "Thermal-Adaptive", "G-DLLL"]
FAMS = {"gaussian": ("Gaussian", "gaussian"), "qary": (r"$q$-ary", "qary"),
        "goldstein-mayer": ("Goldstein-Mayer", "goldstein")}

data = {}
for fn in sorted(glob.glob(os.path.join(RES, "main_*.json"))):
    for fam, dims in json.load(open(fn))["results"].items():
        for d, algs in dims.items():
            for a, x in algs.items():
                data.setdefault(fam, {}).setdefault(int(d), {})[a] = dict(x)
for fn in sorted(glob.glob(os.path.join(RES, "lll_depth_*.json"))):
    for fam, dims in json.load(open(fn))["results"].items():
        for d, algs in dims.items():
            x = algs["LLL"]
            main = data[fam][int(d)]["LLL"]
            assert x["s_ops"] == main["s_ops"], (fam, d)  # same runs, same moves
            main["s_equiv_swaps"] = x["s_equiv_swaps"]


def stat(fam, a, key):
    dims = sorted(d for d in data[fam] if a in data[fam][d])
    m, e = [], []
    for d in dims:
        v = np.asarray(data[fam][d][a][key], float)
        m.append(v.mean())
        e.append(v.std(ddof=1) / np.sqrt(len(v)))
    return np.array(dims, float), np.array(m), np.array(e)


def style(ax):
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)


for fam, (label, tag) in FAMS.items():
    if fam not in data:
        continue
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.45)
    axes = {"ops": fig.add_subplot(gs[0, 0:2]), "W": fig.add_subplot(gs[0, 2:4]),
            "time": fig.add_subplot(gs[0, 4:6]), "d0": fig.add_subplot(gs[1, 1:3]),
            "var": fig.add_subplot(gs[1, 3:5])}
    fig.suptitle(f"{label} benchmark " + r"($\delta=0.99$, $n=30$)", fontsize=11,
                 fontweight="bold")

    def band(ax, key, ylabel, title, log):
        for a in ALGS:
            x, y, e = stat(fam, a, key)
            ax.plot(x, y, marker=MARKERS[a], color=COLORS[a], label=DISPLAY[a],
                    linewidth=1.8, markersize=5)
            lo = np.maximum(y - e, 1e-12) if log else y - e
            ax.fill_between(x, lo, y + e, color=COLORS[a], alpha=0.15)
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=6.5)

    band(axes["ops"], "s_ops", r"Insertions $N$", r"(a) Insertion count ($\pm 1$ SE)", True)
    band(axes["W"], "s_equiv_swaps", r"Equiv. swaps $W$",
         r"(b) Equivalent-swap count $W$ ($\pm 1$ SE)", True)
    band(axes["time"], "s_time", "Wall-clock time (s)", r"(c) Wall-clock time ($\pm 1$ SE)", True)
    ax = axes["d0"]
    offsets = np.linspace(-0.6, 0.6, len(ALGS))
    for k, a in enumerate(ALGS):
        x, y, e = stat(fam, a, "s_delta0")
        ax.errorbar(x + offsets[k], y, yerr=e, fmt=MARKERS[a] + "-", color=COLORS[a],
                    label=DISPLAY[a], linewidth=1.5, markersize=4, capsize=3, capthick=1)
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel(r"$\delta_0$")
    ax.set_title(r"(d) Output quality $\delta_0$ ($\pm 1$ SE)")
    ax.legend(fontsize=6.5)
    band(axes["var"], "s_final_var", r"Final variance $V/d$",
         r"(e) Final profile variance ($\pm 1$ SE)", False)
    for a_ in fig.get_axes():
        style(a_)
        dims = sorted(data[fam])
        a_.set_xticks(dims)
    fig.savefig(os.path.join(OUT, f"deep_benchmark_{tag}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", tag)
