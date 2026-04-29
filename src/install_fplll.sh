#!/usr/bin/env bash
# install_fplll.sh — Build fplll from source into ./deps
# Prerequisites: git, autoconf, automake, libtool, pkg-config,
#                libgmp-dev, libmpfr-dev
set -euo pipefail

FPLLL_VERSION="5.4.5"
PREFIX="$(cd "$(dirname "$0")" && pwd)/deps"
JOBS=$(nproc 2>/dev/null || echo 4)

echo "=== fplll ${FPLLL_VERSION} → ${PREFIX} ==="

# ── Check prerequisites ──────────────────────────────────────────────
for cmd in git autoconf automake libtoolize pkg-config make g++; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Error: $cmd not found"; exit 1; }
done

for hdr in gmp.h mpfr.h; do
    echo "#include <$hdr>" | g++ -xc++ -E - >/dev/null 2>&1 || {
        echo "Error: $hdr not found.  Install lib${hdr%.h}-dev"; exit 1; }
done

# ── Build fplll ──────────────────────────────────────────────────────
mkdir -p "${PREFIX}/src"
cd "${PREFIX}/src"

if [ ! -d fplll ]; then
    git clone --depth 1 --branch "${FPLLL_VERSION}" \
        https://github.com/fplll/fplll.git
fi

cd fplll
./autogen.sh
./configure --prefix="${PREFIX}" CXXFLAGS="-O2 -march=native"
make -j"${JOBS}"
make install

echo ""
echo "=== fplll ${FPLLL_VERSION} installed ==="
echo "Before building the benchmark, run:"
echo "  export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
echo "  export LD_LIBRARY_PATH=${PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
