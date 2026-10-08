#!/usr/bin/env bash
# Reproduce every benchmark of Section 6 of the revised manuscript.
# Writes raw per-sample JSON, prints table rows, and builds the review figure.
# Runtime: about 4 h on 2 cores (Goldstein-Mayer d=80 dominates).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
B="$ROOT/src/build/gdlll_benchmark"
# Use a fresh directory so reruns cannot silently mix with archived split jobs.
R="${RESULTS_DIR:-$ROOT/results/revision_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$R"
T=${THREADS:-2}; S=42
common="--seed $S --nthreads $T --score-drop 1e-6"
$B --families gaussian,qary --dims 40,60,80,100,120 --nlat 30 --main-only $common -o $R/main_gq.json
for a in 2 3; do $B --families gaussian --dims 40,80,120 --nlat 30 --thermal-only --fixed-alpha $a $common -o $R/fixed_gauss_a$a.json; done
$B --families qary --dims 40,80 --nlat 30 --thermal-only --fixed-alpha 2 $common -o $R/fixed_qary_a2.json
$B --families gaussian --dims 40,80,120 --nlat 30 --thermal-only --scale 4 $common -o $R/scale4_gauss.json
$B --families goldstein-mayer --dims 40,60 --nlat 30 --main-only $common -o $R/main_gm_4060.json
$B --families goldstein-mayer --dims 40,60 --nlat 30 --thermal-only --fixed-alpha 2 $common -o $R/fixed_gm_a2.json
$B --families goldstein-mayer --dims 80 --nlat 30 --main-only $common -o $R/main_gm_80.json
# Precomputed-power check (compare recorded counts and output metrics):
$B --families gaussian,qary --dims 40,80,120 --nlat 30 --thermal-only --thermal-fastpow $common -o $R/fastpow_gq.json
python3 "$SCRIPT_DIR/analyze_benchmarks.py" "$R" "$ROOT/paper_out"
