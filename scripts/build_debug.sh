#!/usr/bin/env bash
# build_debug.sh — Debug build script for QuantCore
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_debug"

echo "=== QuantCore Debug Build ==="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_ROOT}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DQUANTCORE_BUILD_TESTS=ON \
    -DQUANTCORE_BUILD_BENCHMARKS=ON \
    -DQUANTCORE_BUILD_EXAMPLES=ON \
    -DQUANTCORE_ENABLE_AVX512=ON \
    -DQUANTCORE_ENABLE_AVX2=ON \
    -DQUANTCORE_ENABLE_SSE42=ON \
    -DQUANTCORE_ENABLE_FAST_MATH=OFF

cmake --build . -j "$(nproc)"

echo ""
echo "=== Debug build complete ==="
echo "Binaries: ${BUILD_DIR}/quantcore/"
echo "Tests:    ${BUILD_DIR}/tests/"
echo ""
echo "Run tests:       cd ${BUILD_DIR} && ctest --output-on-failure"
echo "Run with gdb:    gdb ${BUILD_DIR}/tests/test_column"
