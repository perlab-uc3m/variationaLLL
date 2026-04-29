#!/usr/bin/env bash
# run_gdlll_final.sh — Launch the full G-DLLL benchmark suite.
# Runs 4 sequential invocations (different families/dims/nlat), then merges.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/src/build/gdlll_benchmark"
OUTDIR="$ROOT_DIR/results"
mkdir -p "$OUTDIR"
SEED=42
THREADS=12

echo "=== G-DLLL Final Benchmark Suite ==="
echo "Binary : $BINARY"
echo "Output : $OUTDIR"
echo "Threads: $THREADS"
echo "Seed   : $SEED"
echo ""

# --- Run 1: Gaussian (auto nlat: 144/144/144/72/72/36/36) ---
echo "[1/4] Gaussian  d=40..200  (auto nlat)"
"$BINARY" \
    --families gaussian \
    --dims 40,60,80,100,120,160,200 \
    --seed "$SEED" --nthreads "$THREADS" \
    -o "$OUTDIR/gdlll_part_gauss.json"
echo "      done."

# --- Run 2: Q-ary (auto nlat: 144/144/144/72/72/36) ---
echo "[2/4] Q-ary     d=40..160  (auto nlat)"
"$BINARY" \
    --families qary \
    --dims 40,60,80,100,120,160 \
    --seed "$SEED" --nthreads "$THREADS" \
    -o "$OUTDIR/gdlll_part_qary.json"
echo "      done."

# --- Run 3: Goldstein-Mayer  d=40,80  (nlat=48) ---
echo "[3/4] GM        d=40,80    nlat=48"
"$BINARY" \
    --families goldstein-mayer \
    --dims 40,80 \
    --nlat 48 \
    --seed "$SEED" --nthreads "$THREADS" \
    -o "$OUTDIR/gdlll_part_gm_low.json"
echo "      done."

# --- Run 4: Goldstein-Mayer  d=120  (nlat=24) ---
echo "[4/4] GM        d=120      nlat=24"
"$BINARY" \
    --families goldstein-mayer \
    --dims 120 \
    --nlat 24 \
    --seed "$SEED" --nthreads "$THREADS" \
    -o "$OUTDIR/gdlll_part_gm_120.json"
echo "      done."

# --- Merge ---
echo ""
echo "Merging results..."
python3 "$SCRIPT_DIR/merge_gdlll_results.py" \
    "$OUTDIR/gdlll_part_gauss.json" \
    "$OUTDIR/gdlll_part_qary.json" \
    "$OUTDIR/gdlll_part_gm_low.json" \
    "$OUTDIR/gdlll_part_gm_120.json" \
    -o "$OUTDIR/gdlll_final.json"

echo ""
echo "=== All done.  Output: $OUTDIR/gdlll_final.json ==="
