#!/usr/bin/env python3
"""
generate_paper_tables.py — Regenerate the numeric tables (Tables 1, 2, 3)
from the canonical JSON result files.

Inputs (under ../results/):
  gdlll_final.json         : Tables 1 (tab:deep) and 2 (tab:gdlll)
  profile_universality.json: Table 3 (tab:universality)

Outputs:
  LaTeX snippets for Tables 1, 2, 3 written to ../figures/tables/.

Companion script: generate_gdlll_plots.py (shaded bands use the same JSON).

Notes:
  Standard errors are SE = std / sqrt(n) with n = 30.
  Table 4 (tab:variants) is driven by ad-hoc smaller runs and is
  manually maintained.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULT_DIR = os.path.join(BASE, "results")
OUT_DIR = os.path.join(BASE, "figures", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

N_PER_CELL = 30  # samples per benchmark cell in gdlll_final.json


# ── helpers ──────────────────────────────────────────────────────────────────


def se(std: float, n: int = N_PER_CELL) -> float:
    return std / math.sqrt(n)


def fmt_mean_se(mean: float, std: float, width: int = 0) -> str:
    s = se(std)
    # heuristic SE rounding: <10 -> 1 decimal, >=10 -> integer
    if s < 10:
        return f"{mean:.0f} ({s:.0f})" if s >= 1 else f"{mean:.0f} ({s:.1f})"
    return f"{mean:.0f} ({s:.0f})"


def pct(new: float, base: float) -> float:
    """Percentage reduction of `new` relative to `base`. Positive = fewer ops."""
    if base == 0:
        return 0.0
    return 100.0 * (base - new) / base


# ── Table 1: deep-insertion selectors ────────────────────────────────────────

FAMILIES = [
    ("gaussian", "Gaussian"),
    ("qary", "$q$-ary"),
    ("goldstein-mayer", "Goldstein--Mayer"),
]
DIMS_MAIN = ["40", "80", "120", "160"]
ALGS_T1 = ["Deep-Var", "SS-GG", "Thermal-Adaptive"]


def build_table_deep(results: dict) -> Tuple[str, List[dict]]:
    """Return (LaTeX body, list of per-cell dicts for cross-check)."""
    rows: List[str] = []
    records: List[dict] = []
    for fam_k, fam_tex in FAMILIES:
        for i, d in enumerate(DIMS_MAIN):
            cell = results[fam_k][d]
            vals = {a: cell[a] for a in ALGS_T1}
            N = {a: vals[a]["mean_ops"] for a in ALGS_T1}
            N_std = {a: vals[a]["std_ops"] for a in ALGS_T1}
            W = {a: vals[a]["mean_equiv_swaps"] for a in ALGS_T1}
            W_std = {a: vals[a]["std_equiv_swaps"] for a in ALGS_T1}
            delta_ss = pct(N["Thermal-Adaptive"], N["SS-GG"])

            fam_label = fam_tex if i == 0 else " " * len(fam_tex)
            # Goldstein--Mayer gets its own indent style in the paper
            if fam_k == "goldstein-mayer":
                row = ("Goldstein--Mayer" if i == 0 else "") + " " * (
                    18 - (17 if i == 0 else 0)
                )
                # match the paper layout: family label spans a line above dims
            sign = "$+$" if delta_ss >= 0 else "$-$"
            rows.append(
                f"{fam_label if fam_k != 'goldstein-mayer' or i>0 else 'Goldstein--Mayer'} "
                f"& {int(d):3d} "
                f"& {fmt_mean_se(N['Deep-Var'],  N_std['Deep-Var'])} "
                f"& {fmt_mean_se(N['SS-GG'],     N_std['SS-GG'])} "
                f"& {fmt_mean_se(N['Thermal-Adaptive'], N_std['Thermal-Adaptive'])} "
                f"& {sign}{abs(delta_ss):.0f}\\% "
                f"& {fmt_mean_se(W['Deep-Var'],  W_std['Deep-Var'])} "
                f"& {fmt_mean_se(W['SS-GG'],     W_std['SS-GG'])} "
                f"& {fmt_mean_se(W['Thermal-Adaptive'], W_std['Thermal-Adaptive'])} \\\\"
            )
            records.append(
                dict(
                    family=fam_k,
                    d=int(d),
                    N=N,
                    N_std=N_std,
                    W=W,
                    W_std=W_std,
                    delta_ss=delta_ss,
                )
            )
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])  # drop trailing addlinespace
    return body, records


# ── Table 2: G-DLLL vs Deep-Var / SS-GG ──────────────────────────────────────

ALGS_T2 = ["Deep-Var", "SS-GG", "G-DLLL"]


def build_table_gdlll(results: dict) -> Tuple[str, List[dict]]:
    rows: List[str] = []
    records: List[dict] = []
    for fam_k, fam_tex in FAMILIES:
        for i, d in enumerate(DIMS_MAIN):
            cell = results[fam_k][d]
            dv, sg, gd = cell["Deep-Var"], cell["SS-GG"], cell["G-DLLL"]
            dW = pct(gd["mean_equiv_swaps"], dv["mean_equiv_swaps"])
            fam_label = fam_tex if i == 0 else ""
            sign = "$+$" if dW >= 0 else "$-$"
            rows.append(
                f"{fam_label} & {int(d):3d} "
                f"& {dv['mean_ops']:.0f} "
                f"& {dv['mean_equiv_swaps']:.0f} "
                f"& {sg['mean_equiv_swaps']:.0f} "
                f"& {gd['mean_ops']:.0f} "
                f"& {gd['mean_equiv_swaps']:.0f} "
                f"& {sign}{abs(dW):.0f}\\% \\\\"
            )
            records.append(
                dict(
                    family=fam_k,
                    d=int(d),
                    dv_N=dv["mean_ops"],
                    dv_W=dv["mean_equiv_swaps"],
                    sg_W=sg["mean_equiv_swaps"],
                    gd_N=gd["mean_ops"],
                    gd_W=gd["mean_equiv_swaps"],
                    delta_W=dW,
                )
            )
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return body, records


# ── Table 3: cross-dimension Pearson correlations ────────────────────────────


def build_table_universality(uni: dict) -> Tuple[str, dict]:
    corr = uni["cross_dimension_correlations"]
    ns = uni["sample_sizes"]
    dims = [20, 30, 40, 50, 60]
    # Build upper-triangular matrix
    mat = {}
    for i, a in enumerate(dims):
        for b in dims[i + 1 :]:
            key = f"{a}_vs_{b}"
            mat[(a, b)] = corr[key]
    rows = []
    for i, a in enumerate(dims):
        cells = [f"$d{{=}}{a}$"]
        for j, b in enumerate(dims):
            if j < i:
                cells.append("")
            elif j == i:
                cells.append("---")
            else:
                cells.append(f"{mat[(a, b)]:.3f}")
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    info = dict(
        matrix=mat,
        sample_sizes={int(k): v for k, v in ns.items()},
        min_ge30=min(v for (a, b), v in mat.items() if a >= 30),
    )
    return body, info


# ── Derived numeric claims quoted in prose ───────────────────────────────────


def derive_claims(t1_recs: List[dict], t2_recs: List[dict]) -> Dict[str, str]:
    """Compute every numeric range that appears in the paper's prose."""
    c: Dict[str, str] = {}

    # Abstract / Section 4 G-DLLL W-reduction bands
    def gd_w_range(fam: str) -> Tuple[float, float]:
        xs = [r["delta_W"] for r in t2_recs if r["family"] == fam]
        return min(xs), max(xs)

    g_lo, g_hi = gd_w_range("gaussian")
    q_lo, q_hi = gd_w_range("qary")
    m_lo, m_hi = gd_w_range("goldstein-mayer")

    c["G-DLLL W reduction, Gaussian"] = f"{g_lo:.0f} to {g_hi:.0f}%"
    c["G-DLLL W reduction, q-ary"] = f"{q_lo:.0f} to {q_hi:.0f}%"
    c["G-DLLL W reduction, Goldstein--Mayer"] = f"{m_lo:.0f} to {m_hi:.0f}%"

    # "structured" = max(q-ary, GM) bands; "1 to 27%" claim lumps all
    all_struct = [
        r["delta_W"] for r in t2_recs if r["family"] in ("qary", "goldstein-mayer")
    ]
    c["G-DLLL W red., structured (q-ary ∪ GM)"] = (
        f"{min(all_struct):.0f} to {max(all_struct):.0f}%"
    )
    all_any = [r["delta_W"] for r in t2_recs]
    c["G-DLLL W red., all families (conclusions)"] = (
        f"{min(all_any):.0f} to {max(all_any):.0f}%"
    )

    # Table 1: Thermal-Adaptive N-reduction vs SS-GG on Gaussian d in {40,80,120,160}
    gauss_ta = [r["delta_ss"] for r in t1_recs if r["family"] == "gaussian"]
    per_d = ", ".join(
        "d={}:{:+.0f}%".format(r["d"], r["delta_ss"])
        for r in t1_recs
        if r["family"] == "gaussian"
    )
    c["Thermal vs SS-GG, Gaussian N red. (Sec 5)"] = (
        f"{min(gauss_ta):.0f} to {max(gauss_ta):.0f}%  (per-d: {per_d})"
    )

    # Goldstein--Mayer: W reduction of Thermal vs SS-GG on d in {80,120,160}
    gm_recs = [
        r
        for r in t1_recs
        if r["family"] == "goldstein-mayer" and r["d"] in (80, 120, 160)
    ]
    gm_W_red = [pct(r["W"]["Thermal-Adaptive"], r["W"]["SS-GG"]) for r in gm_recs]
    c["Thermal vs SS-GG, GM W reduction, d∈{80,120,160}"] = (
        f"{min(gm_W_red):.0f} to {max(gm_W_red):.0f}%"
    )

    return c


# ── TeX emitters ─────────────────────────────────────────────────────────────

TAB_DEEP_TPL = r"""% Auto-generated by code/generate_paper_tables.py — do not edit by hand.
\begin{table}[t]
\centering
\caption{Deep-insertion selectors: mean operation count $N$ and mean equivalent-swap count $W$ over $n=30$ independent lattices per cell, with standard errors in parentheses ($\delta = 0.99$, C++ implementation). $\Delta_{\mathrm{SS}}$ is Thermal-Adaptive's percentage reduction in $N$ relative to SS-GG (positive means fewer operations); the $d{=}40$ Goldstein--Mayer entry is well within one standard error of zero and should not be read as a real difference.}
\label{tab:deep}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrrr}
\toprule
Family & $d$ & Deep-Var $N$ & SS-GG $N$ & Therm.\ $N$ & $\Delta_{\mathrm{SS}}$ & Deep-Var $W$ & SS-GG $W$ & Therm.\ $W$ \\
\midrule
%BODY%
\bottomrule
\end{tabular}
\setlength{\tabcolsep}{6pt}
\end{table}
"""

TAB_GDLLL_TPL = r"""% Auto-generated by code/generate_paper_tables.py — do not edit by hand.
\begin{table}[t]
\centering
\caption{G-DLLL vs.\ Deep-Var and SS-GG: mean equivalent-swap count $W$
  and mean operation count $N$ ($\delta=0.99$, C++ implementation).
  $\Delta_W$ is the percentage reduction of G-DLLL over Deep-Var in $W$
  (total cascade depth).}
\label{tab:gdlll}
\small
\begin{tabular}{llrrrrrr}
\toprule
Family & $d$ & Deep-Var $N$ & Deep-Var $W$ & SS-GG $W$ & G-DLLL $N$ & G-DLLL $W$ & $\Delta_W$ \\
\midrule
%BODY%
\bottomrule
\end{tabular}
\end{table}
"""

TAB_UNIV_TPL = r"""% Auto-generated by code/generate_paper_tables.py — do not edit by hand.
\begin{table}[H]
\centering
\caption{Cross-dimension Pearson correlations of mean normalized post-LLL profiles. Sample sizes: %SAMPLES%.}
\label{tab:universality}
\small
\begin{tabular}{rrrrrr}
\toprule
 & $d{=}20$ & $d{=}30$ & $d{=}40$ & $d{=}50$ & $d{=}60$ \\
\midrule
%BODY%
\bottomrule
\end{tabular}
\end{table}
"""


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    with open(os.path.join(RESULT_DIR, "gdlll_final.json")) as f:
        gdlll = json.load(f)
    with open(os.path.join(RESULT_DIR, "profile_universality.json")) as f:
        uni = json.load(f)

    results = gdlll["results"]
    meta = gdlll["meta"]

    t1_body, t1_recs = build_table_deep(results)
    t2_body, t2_recs = build_table_gdlll(results)
    t3_body, t3_info = build_table_universality(uni)

    tab1 = TAB_DEEP_TPL.replace("%BODY%", t1_body)
    tab2 = TAB_GDLLL_TPL.replace("%BODY%", t2_body)
    samples_str = ", ".join(
        f"$n = {t3_info['sample_sizes'][d]}$ at $d{{=}}{d}$"
        for d in (20, 30, 40, 50, 60)
    )
    tab3 = TAB_UNIV_TPL.replace("%BODY%", t3_body).replace("%SAMPLES%", samples_str)

    with open(os.path.join(OUT_DIR, "tab_deep.tex"), "w") as f:
        f.write(tab1)
    with open(os.path.join(OUT_DIR, "tab_gdlll.tex"), "w") as f:
        f.write(tab2)
    with open(os.path.join(OUT_DIR, "tab_universality.tex"), "w") as f:
        f.write(tab3)

    claims = derive_claims(t1_recs, t2_recs)

    print("=" * 70)
    print(f"Pipeline: regenerated tables from gdlll_final.json (meta: {meta})")
    print("=" * 70)
    print(
        f"Wrote:\n  {os.path.relpath(os.path.join(OUT_DIR,'tab_deep.tex'),BASE)}"
        f"\n  {os.path.relpath(os.path.join(OUT_DIR,'tab_gdlll.tex'),BASE)}"
        f"\n  {os.path.relpath(os.path.join(OUT_DIR,'tab_universality.tex'),BASE)}"
    )
    print()
    print("Derived numeric claims:")
    for k, v in claims.items():
        print(f"  {k:55s} = {v}")
    print()
    print(
        f"Cross-dim universality min correlation for d>=30: {t3_info['min_ge30']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
