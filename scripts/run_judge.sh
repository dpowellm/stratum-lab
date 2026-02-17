#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_judge.sh — Shell wrapper for the LLM-as-judge pipeline
#
# Usage:
#   ./scripts/run_judge.sh                          # full run
#   ./scripts/run_judge.sh --dry-run                # cost estimate only
#   ./scripts/run_judge.sh --resume --max-repos 50  # resume partial run
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if present (for ANTHROPIC_API_KEY)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Defaults
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results/full_scan/results}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/full_scan}"

# Verify API key
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "  Set it in .env or export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

echo "=== Stratum Lab — LLM-as-Judge Pipeline ==="
echo "Results dir: $RESULTS_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo ""

cd "$PROJECT_DIR"

python -m stratum_lab.judge \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$ANTHROPIC_API_KEY" \
    "$@"
