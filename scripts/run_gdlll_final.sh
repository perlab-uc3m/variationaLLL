#!/usr/bin/env bash
# Run the canonical benchmark cells used by the main paper figures and tables.
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

PENDING="$OUTDIR/gdlll_final.pending.json"
echo "Running 30 lattices per cell."
"$BINARY" \
    --families gaussian,qary,goldstein-mayer \
    --dims 40,60,80,100,120,160 \
    --nlat 30 --main-only \
    --seed "$SEED" --nthreads "$THREADS" \
    -o "$PENDING"

python3 "$SCRIPT_DIR/validate_gdlll_results.py" "$PENDING" --expected-runs 30
mv -- "$PENDING" "$OUTDIR/gdlll_final.json"

echo ""
echo "=== All done.  Output: $OUTDIR/gdlll_final.json ==="
