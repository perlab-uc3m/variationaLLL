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
scans every generalized-Lovász-admissible insertion and accepts only strict
descent in the relevant potential. G-DLLL uses the exact global
`argmax ΔV/(k-j)` rule. Candidate scores are updated in constant time as the
insertion point moves, so a full scan takes quadratic score work per iteration.
If no candidate gives strict descent but an adjacent Lovász violation remains,
the loop performs that adjacent swap. This makes the returned basis LLL-reduced
without changing the primary score.

The C++ benchmark in `src/` is the canonical implementation built on
[fplll](https://github.com/fplll/fplll). It reports zero-score fallback swaps,
operation-limit hits, and failed final LLL checks. Statistics exclude runs that
reach the operation limit. The Python scripts in `scripts/` drive the runs and
produce the figures and tables referenced in the paper.

## Build

```bash
cd src
bash install_fplll.sh
export PKG_CONFIG_PATH=$PWD/deps/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export LD_LIBRARY_PATH=$PWD/deps/lib:${LD_LIBRARY_PATH:-}
mkdir -p build && cd build
cmake .. && make
```

## Reproduce the paper

```bash
bash scripts/run_gdlll_final.sh         # writes results/gdlll_final.json
python scripts/generate_gdlll_plots.py  # writes figures/*.pdf
python scripts/generate_paper_tables.py # writes figures/tables/*.tex
```

The runner uses 30 lattices per cell and only the five algorithms reported in
the main comparison. It validates every required cell before replacing the
previous result file. A single run takes a few hours on 12 threads. For a quick
smoke test, invoke `src/build/gdlll_benchmark --dims 40 --nlat 4 --main-only
--families gaussian -o /tmp/q.json` directly.

## Layout

```
src/        C++ canonical implementation (gdlll_benchmark)
scripts/    Python: lattice generators, plot/table builders, run drivers
results/    JSON benchmark output (created on first run)
figures/    Plots and LaTeX table snippets (created on first run)
```

## License

MIT.
