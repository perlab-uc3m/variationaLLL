# variationaLLL

Reference implementation of the deep-insertion LLL selectors studied in
*Variational and Majorization Principles in Lattice Reduction*
(Blanco-Romero & Almenares Mendoza). The four selectors share one greedy
loop and differ only in the score they maximise:

| Selector            | Score                                               |
|---------------------|-----------------------------------------------------|
| Deep-Var            | $\Delta V = -\Delta\sum p_i^2$                      |
| SS-GG               | $\Delta\sum r_i$                                    |
| Thermal-Adaptive    | $\Delta\sum r_i^{\alpha}$, $\alpha$ from initial CV |
| G-DLLL              | $\Delta V/(k-j)$                                    |

Each iteration starts by size-reducing the current basis. It then exhaustively
scans every deep insertion and accepts a candidate when its score drop exceeds
the relative threshold $\rho F$ (default $\rho=10^{-6}$). G-DLLL uses the exact
global
`argmax ΔV/(k-j)` rule. Candidate scores are updated in constant time as the
insertion point moves, so a full scan takes quadratic score work per iteration.
If no candidate clears the threshold but an adjacent Lovász violation remains,
the loop performs that adjacent swap. This makes the returned basis LLL-reduced
while preserving score monotonicity. The algorithm stops only when the basis is
LLL-reduced and no deep insertion decreases the selected potential by more than
the relative threshold. Thus the output has both an LLL certificate and the
global score-stationarity certificate used by X-GG methods.

The C++ benchmark in `src/` is the canonical implementation built on
[fplll](https://github.com/fplll/fplll). It reports subthreshold fallback swaps,
operation-limit hits, failed final LLL checks, and the mean and maximum terminal
score residual $\kappa_F$. The residual is recomputed by an exhaustive candidate
scan; a completed selector run must have $\kappa_F\leq\rho$. Statistics exclude
runs that reach the operation limit; the limit is a failure guard, not a stopping
rule.
The Python scripts in `scripts/` drive the runs and
produce the figures and tables referenced in the paper.

`Thermal-Band` is an experimental selector and is not part of `--main-only`.
It chooses the shallowest candidate whose thermal score drop is within
$\rho F$ of the best drop, then uses the largest-score adjacent violation as
its fallback. Run `gdlll_benchmark --compare-thermal-band` to compare it with
Thermal-Adaptive on the same generated bases.

`Thermal-TwoClock` uses the same tolerance band but chooses the candidate with
the largest decrease in the LLL potential. Its fallback uses the same rule.
This makes the thermal score the primary clock and the LLL potential the
tie-breaker, without adding another parameter. Run
`gdlll_benchmark --compare-thermal-two-clock` for the paired comparison.

## Build

```bash
cd src
bash install_fplll.sh
export PKG_CONFIG_PATH=$PWD/deps/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export LD_LIBRARY_PATH=$PWD/deps/lib:${LD_LIBRARY_PATH:-}
mkdir -p build && cd build
cmake .. && make
```

## Reproduce the revised paper

```bash
bash scripts/run_revision_benchmarks.sh
python3 scripts/analyze_benchmarks.py results ../paper
(cd results && python3 ../scripts/validate_theory.py)
```

The revision runner uses a fresh timestamped results directory (override with
`RESULTS_DIR`) and writes the review figure to `paper_out/figures/review/`;
its table rows are printed to stdout. The second command above checks or updates
the tables inline in `paper/main_review.tex` from the archived results and writes
the review figure to `paper/figures/review/`. Use a fresh results directory to
analyze a new run. The analysis rejects incomplete, failed, duplicate, or
nonfinite data before producing tables.
It requires NumPy, SciPy and Matplotlib. The theory checks require NumPy and use
Python's `Fraction` for exact GSO and `Decimal` for the new logarithmic checks.

The implementation uses fplll's numerical size-reduction tolerance 0.51. Its
exit checks reuse the GSO and score formulas and are numerical diagnostics.
`LLLReduction::n_swaps` counts **batched row moves**, not adjacent transpositions.
The updated baseline observes `MatGSO::move_row` to measure the separate depth
sum W. Historical `main_*.json` files copied N into the LLL `s_equiv_swaps`
field; the analysis never treats that field as a measured depth. Instrumented
`lll_depth_*.json` verification files can supply W after their row counts and
output metrics are checked against the archived runs. Their timings do not
replace the archived timing comparisons.

The operation cap is a failure guard. Reported experiments use 30 bases per
cell. A smoke test is:

```bash
src/build/gdlll_benchmark --dims 20 --nlat 3 --main-only --families gaussian,qary,goldstein-mayer --score-drop 1e-6 -o /tmp/lattice-smoke.json
```

## Layout

```
src/        C++ canonical implementation (gdlll_benchmark)
scripts/    Python: lattice generators, plot/table builders, run drivers
results/    JSON benchmark output (created on first run)
figures/    Plots (created on first run)
```

## License

MIT.

## Revision benchmarks (DCC revision, October 2026)

`scripts/run_revision_benchmarks.sh` reproduces every number of Section 6 of the
revised manuscript and writes raw per-sample JSON to `results/`;
`scripts/analyze_benchmarks.py results ../paper` refreshes the inline tables in
`paper/main_review.tex`, writes `paper/figures/review/fig_benchmarks.pdf`, and
updates `results/summary.json` (paired ratios with 95% intervals).
`scripts/validate_theory.py` checks the theoretical statements of the paper in
exact rational GSO and numerical logarithms, and writes `validate_theory.json`.

New driver options: `--algs i,j,...` (algorithm indices), `--thermal-only`,
`--fixed-alpha A` (thermal score with a fixed exponent), `--scale s` (multiply
every basis by `s`), and `--thermal-fastpow` (precomputed powers; algebraically
equivalent score evaluation). The seed of every basis now depends only on
(seed, family, d, index), and fplll's GMP generator is reseeded per basis, so
separate runs see identical bases. JSON output stores per-sample arrays
(`s_ops`, `s_equiv_swaps`, `s_time`, `s_delta0`, `s_alpha`, ...).

### Optional square-score specialization

`--fixed-alpha 2 --thermal-square-fast` scores the thermal objective
`sum(r_i^2)` with arithmetic operations instead of general powers. The flag is
opt-in so the archived paper benchmarks still use their original scoring path.
It preserves the selection rule in exact arithmetic; floating-point ties can
still affect a trajectory. For a paired SS-GG comparison on the same bases:

```bash
src/build/gdlll_benchmark --dims 80 --nlat 30 --nthreads 1 \
  --families gaussian --algs 2,3 --fixed-alpha 2 \
  --thermal-square-fast --seed 42 --score-drop 1e-6 -o /tmp/alpha2.json
```

The three `results/alpha2_square_gauss80_r*.json` files are repeated runs of
this command with fplll 5.4.5. Their per-basis move counts and recorded output
metrics match the archived fixed-alpha-2 data. On this local machine the paired
square-score/SS-GG time ratios were 0.87, 0.92, and 0.86. This is a focused
Gaussian timing check, not a claim of improvement for every input family.
