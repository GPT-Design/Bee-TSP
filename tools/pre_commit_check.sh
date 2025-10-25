#!/usr/bin/env bash
# Pre-commit check per Update_17.txt section 13.1
set -euo pipefail

echo "=== PRE-COMMIT SAFETY CHECKS ==="

# 1. Banned strings scan
echo "Running banned strings scan..."
if rg -nEi 'synthetic|mock|dummy|stub|normalized|scaled|approx|float_len|fast_eval|toy|test_eval|sampled_len|fake' bee_tsp; then
  echo "❌ BANNED TERM DETECTED in production code"
  exit 1
fi
echo "✅ No banned terms found"

# 2. Boundary bloat guard
echo "Running boundary parameter checks..."
if rg -nEi 'edge[_-]?cap\s*:\s*(1[1-9]|[2-9]\d)|halo(_pct)?\s*:\s*0\.(2[1-9]|[3-9]\d)|portals[_-]?per[_-]?node\s*:\s*([89]|\d{2,})' configs bee_tsp; then
  echo "❌ BOUNDARY BLOAT DETECTED - parameter values too high"
  exit 1
fi
echo "✅ Boundary parameters within limits"

# 3. Environment check
echo "Checking evaluation environment..."
if [ "${STRICT_TSPLIB_EVAL:-}" != "1" ]; then
  echo "⚠️  WARNING: STRICT_TSPLIB_EVAL not set to 1"
  echo "   Set: export STRICT_TSPLIB_EVAL=1"
fi

if [ "${OMP_NUM_THREADS:-}" != "1" ] || [ "${OPENBLAS_NUM_THREADS:-}" != "1" ] || [ "${MKL_NUM_THREADS:-}" != "1" ]; then
  echo "⚠️  WARNING: Threading environment not optimized"
  echo "   Set: export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
fi

echo "✅ Pre-commit checks completed"