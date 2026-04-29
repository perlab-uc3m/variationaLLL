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

The C++ benchmark in `src/` is the canonical implementation built on
[fplll](https://github.com/fplll/fplll); the Python scripts in `scripts/`
drive the runs and produce the figures and tables referenced in the paper.

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

A single run of the full suite takes a few hours on 12 threads. For a
quick smoke test, invoke `src/build/gdlll_benchmark --dims 40 --nlat 4
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
