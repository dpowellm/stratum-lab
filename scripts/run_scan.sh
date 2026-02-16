#!/usr/bin/env bash
set -euo pipefail

# ===================== CONFIGURATION =====================
STRUCTURAL_SCAN_DIR="${1:?Usage: ./run_scan.sh <structural-scan-dir> [output-dir]}"
OUTPUT_DIR="${2:-./data/scan-output}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
VLLM_MODEL="${STRATUM_VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TARGET="${SCAN_TARGET:-1000}"
CONCURRENT="${SCAN_CONCURRENT:-8}"
TIMEOUT="${SCAN_TIMEOUT:-600}"
PILOT_SIZE="${SCAN_PILOT_SIZE:-20}"

export STRATUM_VLLM_MODEL="$VLLM_MODEL"

# ===================== PREFLIGHT =====================
echo "=== Preflight ==="
echo -n "  vLLM endpoint ($VLLM_URL)... "
curl -sf "${VLLM_URL}/models" > /dev/null && echo "OK" || { echo "FAIL"; exit 1; }
echo -n "  Docker... "
docker info > /dev/null 2>&1 && echo "OK" || { echo "FAIL"; exit 1; }
echo -n "  Structural scans... "
SCAN_COUNT=$(find "$STRUCTURAL_SCAN_DIR" -name "*.json" | wc -l)
echo "${SCAN_COUNT} files"
[ "$SCAN_COUNT" -gt 0 ] || { echo "ERROR: no scan files"; exit 1; }
echo -n "  Runner image... "
docker image inspect stratum-lab-runner > /dev/null 2>&1 && echo "OK" || {
    echo "building..."
    stratum-lab build-image
}
echo ""

# ===================== SCAN =====================
echo "=== Starting scan: target=${TARGET}, concurrent=${CONCURRENT}, timeout=${TIMEOUT}s ==="
stratum-lab pipeline \
    --input-dir "$STRUCTURAL_SCAN_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --vllm-url "$VLLM_URL" \
    --concurrent "$CONCURRENT" \
    --timeout "$TIMEOUT" \
    --pilot \
    --pilot-size "$PILOT_SIZE" \
    --max-instrumentation-failure-rate 0.20 \
    --max-model-failure-rate 0.15 \
    --resume \
    2>&1 | tee "${OUTPUT_DIR}/scan.log"

# ===================== VALIDATE =====================
echo ""
echo "=== Post-scan validation ==="
python scripts/validate_scan.py "$OUTPUT_DIR"
