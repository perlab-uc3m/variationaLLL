#!/usr/bin/env python3
"""Build the tables, figure, and summary numbers of Section 6 from the raw
per-sample benchmark output of gdlll_benchmark (variationaLLL).

Usage: python3 analyze_benchmarks.py RESULTS_DIR PAPER_DIR
Writes PAPER_DIR/figures/fig_benchmarks.pdf, PAPER_DIR/tables/tab_*.tex and
RESULTS_DIR/summary.json.
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

RES, PAPER = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)

ALGS = ["LLL", "Deep-Var", "SS-GG", "Thermal-Adaptive", "G-DLLL"]
FAMS = ["gaussian", "qary", "goldstein-mayer"]
FAMNAME = {"gaussian": "Gaussian", "qary": "$q$-ary", "goldstein-mayer": "G.--M."}


def load(pattern):
    out = {}
    for fn in sorted(glob.glob(os.path.join(RES, pattern))):
        data = json.load(open(fn))
        for fam, dims in data["results"].items():
            for d, algs in dims.items():
                for a, x in algs.items():
                    cell = out.setdefault(fam, {}).setdefault(int(d), {})
                    if a in cell:
                        raise ValueError(f"duplicate result for {fam} d={d} {a}: {fn}")
                    if a == "LLL":
                        x["_lll_depth_measured"] = data.get("meta", {}).get("lll_count_convention") == \
                            "batched_row_moves_and_observed_depth"
                    cell[a] = x
    return out


main = load("main_*.json")
fixed = {}
for fn in sorted(glob.glob(os.path.join(RES, "fixed_*_a*.json"))):
    a = float(fn.rsplit("_a", 1)[1][:-5])
    fixed.setdefault(a, {})
    for fam, dims in json.load(open(fn))["results"].items():
        for d, algs in dims.items():
            fixed[a].setdefault(fam, {})[int(d)] = algs["Thermal-Adaptive"]
scaled = {}
for fn in glob.glob(os.path.join(RES, "scale4_*.json")):
    for fam, dims in json.load(open(fn))["results"].items():
        for d, algs in dims.items():
            scaled.setdefault(fam, {})[int(d)] = algs["Thermal-Adaptive"]


def arr(x, key):
    return np.asarray(x[key], float)


def mean_se(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


def paired_ratio(a, b):
    """Geometric-mean ratio a/b and 95% t-interval, from per-sample logs."""
    lr = np.log(np.asarray(a, float) / np.asarray(b, float))
    m, s = lr.mean(), lr.std(ddof=1) / math.sqrt(len(lr))
    t = stats.t.ppf(0.975, len(lr) - 1)
    return math.exp(m), math.exp(m - t * s), math.exp(m + t * s)


def b1_ratio(xa, xb, d):
    """Geometric-mean ratio of ||b_1|| (A over B) with 95% CI, same bases."""
    lr = d * (np.log(arr(xa, "s_delta0")) - np.log(arr(xb, "s_delta0")))
    m, s = lr.mean(), lr.std(ddof=1) / math.sqrt(len(lr))
    t = stats.t.ppf(0.975, len(lr) - 1)
    return math.exp(m), math.exp(m - t * s), math.exp(m + t * s)


summary = {"checks": {}, "cells": {}}

# Historical files copied fplll's row-move counter into s_equiv_swaps.
# Only use LLL depths from instrumented verification runs, without replacing
# the archived timings. Require the same recorded path counts and outputs.
lll_depth = load("lll_depth_*.json")
lll_depth_runs = 0
for fam, dims in lll_depth.items():
    for d, cell in dims.items():
        new = cell["LLL"]
        old = main[fam][d]["LLL"]
        if new["s_ops"] != old["s_ops"]:
            raise ValueError(f"LLL verification row counts differ: {fam} d={d}")
        for metric in ("s_delta0", "s_final_var"):
            if not np.allclose(arr(new, metric), arr(old, metric), rtol=1e-8, atol=1e-11):
                raise ValueError(f"LLL verification outputs differ: {fam} d={d} {metric}")
        if new["limit_hits"] or new["non_lll_outputs"] or new["completed_runs"] != 30:
            raise ValueError(f"failed LLL depth verification: {fam} d={d}")
        if any(w < n for w, n in zip(new["s_equiv_swaps"], new["s_ops"])):
            raise ValueError(f"invalid LLL depths: {fam} d={d}")
        if not new.get("_lll_depth_measured"):
            raise ValueError("LLL depth verification lacks measurement metadata")
        old["s_equiv_swaps"] = new["s_equiv_swaps"]
        old["_lll_depth_measured"] = True
        lll_depth_runs += 30
summary["lll_depth_verification_runs"] = lll_depth_runs


# ------------------------------------------------------------ integrity checks
# Fail before writing tables or figures if any recorded run is incomplete.
# The power-precomputation comparison is part of the reported study too.
expected_dims = {"gaussian": {40, 60, 80, 100, 120},
                 "qary": {40, 60, 80, 100, 120},
                 "goldstein-mayer": {40, 60, 80}}
for fam, dims in expected_dims.items():
    if set(main.get(fam, {})) != dims:
        raise ValueError(f"missing or unexpected main dimensions for {fam}")
    for d in dims:
        if set(main[fam][d]) != set(ALGS):
            raise ValueError(f"missing or unexpected algorithms for {fam} d={d}")

fastpow = load("fastpow_gq.json")
max_kappa, limit_hits, non_lll, nruns = 0.0, 0, 0, 0
for src in [main] + list(fixed.values()) + [scaled, fastpow]:
    for fam in src:
        for d in src[fam]:
            cell = src[fam][d]
            items = cell.items() if isinstance(next(iter(cell.values())), dict) \
                else [("Thermal-Adaptive", cell)]
            for a, x in items:
                context = f"{fam} d={d} {a}"
                for key in ("s_ops", "s_equiv_swaps", "s_time", "s_delta0",
                            "s_final_var", "s_fallback", "s_limit", "s_lll", "s_alpha"):
                    values = np.asarray(x[key], float)
                    if len(values) != 30 or not np.all(np.isfinite(values)):
                        raise ValueError(f"invalid sample array {key}: {context}")
                if x["completed_runs"] != 30 or x["limit_hits"] or x["non_lll_outputs"]:
                    raise ValueError(f"incomplete or failed run: {context}")
                if any(x["s_limit"]) or not all(v == 1 for v in x["s_lll"]):
                    raise ValueError(f"failed per-sample check: {context}")
                for key in ("s_ops", "s_equiv_swaps", "s_time", "s_delta0"):
                    if np.any(arr(x, key) <= 0):
                        raise ValueError(f"nonpositive paired metric {key}: {context}")
                nruns += len(x["s_ops"])
                limit_hits += x["limit_hits"]
                non_lll += x["non_lll_outputs"]
                if a != "LLL":
                    kappa = x["max_kappa_f"]
                    if kappa is None or not math.isfinite(kappa) or not 0 <= kappa <= 1e-6:
                        raise ValueError(f"invalid terminal residual: {context}")
                    max_kappa = max(max_kappa, kappa)
summary["checks"] = dict(runs=nruns, limit_hits=limit_hits, non_lll_outputs=non_lll,
                         max_kappa_f=max_kappa)
print(summary["checks"])

# ----------------------------------------------------------------- main table
def fmt(v, se=None, digits=0):
    s = f"{v:,.{digits}f}".replace(",", "\\,")
    return s


rows = []
for metric, key in (("Insertion count $N$", "s_ops"),
                    ("Equivalent-swap count $W$", "s_equiv_swaps")):
    rows.append(r"\multicolumn{7}{l}{\emph{" + metric + r"}}\\")
    for fam in FAMS:
        first = True
        for d in sorted(main[fam]):
            c = main[fam][d]
            cells = [FAMNAME[fam] if first else "", str(d)]
            first = False
            for a in ALGS:
                available = a != "LLL" or key == "s_ops" or main[fam][d]["LLL"].get("_lll_depth_measured", False)
                cells.append(fmt(arr(c[a], key).mean()) if available else "--")
            rows.append(" & ".join(cells) + r" \\")
            summary["cells"].setdefault(fam, {})[d] = {
                a: dict(N=mean_se(arr(c[a], "s_ops")),
                        W=mean_se(arr(c[a], "s_equiv_swaps"))
                          if a != "LLL" or main[fam][d]["LLL"].get("_lll_depth_measured", False) else None,
                        time=mean_se(arr(c[a], "s_time")), delta0=mean_se(arr(c[a], "s_delta0")),
                        V=mean_se(arr(c[a], "s_final_var")),
                        fallback=mean_se(arr(c[a], "s_fallback")),
                        alpha=float(np.mean(c[a]["s_alpha"])),
                        alpha_min=float(np.min(c[a]["s_alpha"])),
                        alpha_max=float(np.max(c[a]["s_alpha"])))
                for a in ALGS}
        rows.append(r"\addlinespace")
    rows.append(r"\midrule")
rows = rows[1:-2]
with open(os.path.join(PAPER, "tables", "tab_counts.tex"), "w") as fh:
    fh.write("\n".join(rows) + "\n")

# ------------------------------------------------------- paired comparisons
prow = []


def paired_line(label, fam, d, xa, xb, alpha_txt):
    rN = paired_ratio(arr(xa, "s_ops"), arr(xb, "s_ops"))
    rW = paired_ratio(arr(xa, "s_equiv_swaps"), arr(xb, "s_equiv_swaps"))
    rT = paired_ratio(arr(xa, "s_time"), arr(xb, "s_time"))
    rb = b1_ratio(xa, xb, d)
    same = float(np.mean(np.abs(arr(xa, "s_delta0") - arr(xb, "s_delta0")) < 1e-12))
    rec = dict(N=rN, W=rW, time=rT, b1=rb, same_b1_fraction=same, alpha=alpha_txt)
    f = lambda r: f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]"
    fb = lambda r: f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
    line = f"{label} & {d} & {alpha_txt} & {f(rN)} & {f(rT)} & {fb(rb)} \\\\"
    return line, rec


summary["paired"] = {}
for fam in FAMS:
    if fam not in main:
        continue
    first = True
    for d in sorted(main[fam]):
        c = main[fam][d]
        if "Thermal-Adaptive" not in c:
            continue
        al = np.asarray(c["Thermal-Adaptive"]["s_alpha"])
        at = f"{al.mean():.2f}" if al.std() > 0.005 else f"{al.mean():.1f}"
        line, rec = paired_line(FAMNAME[fam] if first else "", fam, d,
                                c["Thermal-Adaptive"], c["SS-GG"], at)
        first = False
        summary["paired"].setdefault("TA_vs_SSGG", {}).setdefault(fam, {})[d] = rec
        prow.append(line)
    prow.append(r"\addlinespace")
# fixed-alpha and scale rows
frow = []
for a in sorted(fixed):
    for fam in FAMS:
        if fam not in fixed[a]:
            continue
        for d in sorted(fixed[a][fam]):
            if fam in main and d in main[fam]:
                line, rec = paired_line(FAMNAME[fam], fam, d, fixed[a][fam][d],
                                        main[fam][d]["SS-GG"], f"{a:g} (fixed)")
                summary["paired"].setdefault(f"fixed_{a:g}_vs_SSGG", {}).setdefault(fam, {})[d] = rec
                frow.append(line)
                # also adaptive vs fixed
                line2, rec2 = paired_line("", fam, d, main[fam][d]["Thermal-Adaptive"],
                                          fixed[a][fam][d], "")
                summary["paired"].setdefault(f"TA_vs_fixed_{a:g}", {}).setdefault(fam, {})[d] = rec2
for fam in scaled:
    for d in sorted(scaled[fam]):
        if fam in main and d in main[fam]:
            al = np.asarray(scaled[fam][d]["s_alpha"])
            line, rec = paired_line(FAMNAME[fam] + r", $4\mathbf B$", fam, d, scaled[fam][d],
                                    main[fam][d]["SS-GG"], f"{al.mean():.2f}")
            summary["paired"].setdefault("TA_scaled4_vs_SSGG", {}).setdefault(fam, {})[d] = rec
            frow.append(line)
            line2, rec2 = paired_line("", fam, d, scaled[fam][d], main[fam][d]["Thermal-Adaptive"], "")
            summary["paired"].setdefault("TA_scaled4_vs_TA", {}).setdefault(fam, {})[d] = rec2
if prow and prow[-1] == r"\addlinespace":
    prow.pop()
with open(os.path.join(PAPER, "tables", "tab_paired.tex"), "w") as fh:
    fh.write("\n".join(prow) + "\n")
with open(os.path.join(PAPER, "tables", "tab_fixed.tex"), "w") as fh:
    fh.write("\n".join(frow) + "\n")

# Deep-Var and G-DLLL vs SS-GG (for prose)
for other in ["Deep-Var", "G-DLLL", "LLL"]:
    for fam in main:
        for d in main[fam]:
            c = main[fam][d]
            if other in c and "SS-GG" in c:
                _, rec = paired_line("", fam, d, c[other], c["SS-GG"], "")
                if other == "LLL" and not main[fam][d]["LLL"].get("_lll_depth_measured", False):
                    rec["W"] = None
                summary["paired"].setdefault(f"{other}_vs_SSGG", {}).setdefault(fam, {})[d] = rec

fp = os.path.join(RES, "fastpow_gq.json")
if os.path.exists(fp):
    summary["fastpow"] = {}
    for fam, dims in json.load(open(fp))["results"].items():
        for d, algs in dims.items():
            x = algs["Thermal-Adaptive"]
            if fam in main and int(d) in main[fam]:
                y = main[fam][int(d)]["Thermal-Adaptive"]
                s_ = main[fam][int(d)]["SS-GG"]
                summary["fastpow"].setdefault(fam, {})[d] = dict(
                    identical_ops=x["s_ops"] == y["s_ops"],
                    identical_recorded_metrics=all(x[k] == y[k] for k in
                        ("s_ops", "s_equiv_swaps", "s_fallback", "s_delta0", "s_final_var")),
                    speedup=paired_ratio(arr(y, "s_time"), arr(x, "s_time")),
                    fast_vs_ssgg=paired_ratio(arr(x, "s_time"), arr(s_, "s_time")))

json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

# -------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "mathtext.fontset": "cm"})
COL = {"LLL": "#555555", "Deep-Var": "#1f77b4", "SS-GG": "#d62728",
       "Thermal-Adaptive": "#2ca02c", "G-DLLL": "#9467bd"}
MK = {"LLL": "s", "Deep-Var": "o", "SS-GG": "^", "Thermal-Adaptive": "D", "G-DLLL": "v"}
fams = [f for f in FAMS if f in main]
fig, axes = plt.subplots(len(fams), 3, figsize=(6.6, 1.95 * len(fams)), squeeze=False)
for i, fam in enumerate(fams):
    dims = sorted(d for d in main[fam] if all(a in main[fam][d] for a in ALGS))
    for a in ALGS:
        for j, (key, logy) in enumerate([("s_ops", True), ("s_time", True), ("s_delta0", False)]):
            ax = axes[i][j]
            m = [arr(main[fam][d][a], key).mean() for d in dims]
            se = [arr(main[fam][d][a], key).std(ddof=1) / math.sqrt(len(main[fam][d][a][key]))
                  for d in dims]
            off = (ALGS.index(a) - 2) * 0.6 if key == "s_delta0" else 0
            ax.errorbar(np.array(dims) + off, m, yerr=se, color=COL[a], marker=MK[a], ms=3,
                        lw=1, capsize=1.5, label=a if a != "LLL" else "LLL (fplll)")
            if logy:
                ax.set_yscale("log")
    axes[i][0].set_ylabel({"gaussian": "Gaussian", "qary": "$q$-ary", "goldstein-mayer": "Goldstein-Mayer"}[fam] + "\n" + "insertion count $N$")
    axes[i][1].set_ylabel("time [s]")
    axes[i][2].set_ylabel(r"root-Hermite factor $\delta_0$")
    for j in range(3):
        axes[i][j].set_xticks(dims)
        axes[i][j].grid(alpha=0.25, lw=0.5)
for j in range(3):
    axes[-1][j].set_xlabel("dimension $d$")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(PAPER, "figures", "fig_benchmarks.pdf"))
print("wrote figure and tables")
