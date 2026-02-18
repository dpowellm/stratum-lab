#!/usr/bin/env bash
# ============================================================================
# deploy.sh — Deploy fixed stratum-lab files to droplet and validate
#
# Usage:
#   bash deploy.sh [--droplet-dir DIR]
#
# Steps:
#   1. Copy ALL fixed files (including Dockerfile) to droplet directory
#   2. Rebuild Docker image
#   3. Run vLLM health check
#   4. Run single-repo smoke test
#   5. Validate smoke test results
#   6. Print mass scan command
# ============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROPLET_DIR="${1:-$HOME/stratum-lab}"

echo "=== Stratum Lab Deploy ==="
echo "  Script dir:  $SCRIPT_DIR"
echo "  Droplet dir: $DROPLET_DIR"
echo ""

# ── Preflight checks ─────────────────────────────────────────────────────

if [ -z "$VLLM_HOST" ]; then
    echo "FATAL: VLLM_HOST is not set. Export it before running." >&2
    echo "  export VLLM_HOST=https://your-vllm-server:8000" >&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "FATAL: docker not found in PATH" >&2
    exit 1
fi

# ── Step 1: Copy fixed files ─────────────────────────────────────────────

echo "[deploy] Step 1: Copying fixed files..."

mkdir -p "$DROPLET_DIR"
mkdir -p "$DROPLET_DIR/scripts"
mkdir -p "$DROPLET_DIR/stratum_patcher"

# Copy Dockerfile (Issue D fix — was previously missing)
if [ -f "$SCRIPT_DIR/fixed_files/Dockerfile" ]; then
    cp "$SCRIPT_DIR/fixed_files/Dockerfile" "$DROPLET_DIR/Dockerfile"
    echo "  Copied Dockerfile from fixed_files/"
elif [ -f "$SCRIPT_DIR/Dockerfile" ]; then
    cp "$SCRIPT_DIR/Dockerfile" "$DROPLET_DIR/Dockerfile"
    echo "  Copied Dockerfile from script dir"
fi

# Copy run_repo.sh
for src in "$SCRIPT_DIR/fixed_files/run_repo.sh" "$SCRIPT_DIR/run_repo.sh"; do
    if [ -f "$src" ]; then
        cp "$src" "$DROPLET_DIR/run_repo.sh"
        echo "  Copied run_repo.sh"
        break
    fi
done

# Copy scripts/
for script in orchestrate.sh synthetic_harness.py inject_main.py evaluate_tiers.py \
              scan_defensive_patterns.py build_behavioral_records.py aggregate_results.py \
              check_pilot_quality.py analyze_semantics.py compute_enrichments.py \
              mine_remediations.py compute_risk_model.py scan_data_topology.py; do
    for src in "$SCRIPT_DIR/fixed_files/scripts/$script" "$SCRIPT_DIR/scripts/$script"; do
        if [ -f "$src" ]; then
            cp "$src" "$DROPLET_DIR/scripts/$script"
            echo "  Copied scripts/$script"
            break
        fi
    done
done

# Copy stratum_patcher/ files
for patcher in llm_redirect.py sitecustomize.py event_logger.py openai_patch.py \
               litellm_patch.py crewai_patch.py langgraph_patch.py autogen_patch.py \
               langchain_patch.py runner.py __init__.py; do
    for src in "$SCRIPT_DIR/fixed_files/stratum_patcher/$patcher" "$SCRIPT_DIR/stratum_patcher/$patcher"; do
        if [ -f "$src" ]; then
            cp "$src" "$DROPLET_DIR/stratum_patcher/$patcher"
            echo "  Copied stratum_patcher/$patcher"
            break
        fi
    done
done

# Copy requirements
for req in requirements-docker.txt requirements.txt; do
    if [ -f "$SCRIPT_DIR/$req" ]; then
        cp "$SCRIPT_DIR/$req" "$DROPLET_DIR/$req"
        echo "  Copied $req"
    fi
done

echo ""

# ── Step 2: Rebuild Docker image ─────────────────────────────────────────

echo "[deploy] Step 2: Rebuilding Docker image..."

cd "$DROPLET_DIR"
docker build -t stratum-lab-base . 2>&1 | tail -20
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo "FATAL: Docker build failed (exit=$BUILD_EXIT)" >&2
    exit 1
fi

echo "  Docker image rebuilt successfully"
echo ""

# ── Step 3: vLLM health check ────────────────────────────────────────────

echo "[deploy] Step 3: vLLM health check..."

if curl -sf "$VLLM_HOST/health" > /dev/null 2>&1; then
    echo "  vLLM /health: OK"
elif curl -sf "$VLLM_HOST/v1/models" > /dev/null 2>&1; then
    echo "  vLLM /v1/models: OK"
else
    echo "FATAL: vLLM not reachable at $VLLM_HOST" >&2
    exit 1
fi

echo ""

# ── Step 4: Single-repo smoke test ───────────────────────────────────────

echo "[deploy] Step 4: Running smoke test..."

SMOKE_REPO="https://github.com/binbakhsh/QBO-CrewAI"
SMOKE_DIR=$(mktemp -d)
SMOKE_TIMEOUT=300
VLLM_MODEL="${STRATUM_VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

mkdir -p /tmp/pip-cache

timeout $((SMOKE_TIMEOUT + 60)) docker run --rm \
    --network=host \
    --entrypoint bash \
    --memory=4g \
    -v "$DROPLET_DIR/run_repo.sh:/app/run_repo.sh:ro" \
    -v "$DROPLET_DIR/scripts/synthetic_harness.py:/app/synthetic_harness.py:ro" \
    -v "$DROPLET_DIR/scripts/inject_main.py:/app/inject_main.py:ro" \
    -v "$SMOKE_DIR:/app/output" \
    -v "/tmp/pip-cache:/root/.cache/pip" \
    -e "STRATUM_EVENTS_FILE=/app/output/events_run_1.jsonl" \
    -e "STRATUM_RUN_NUMBER=1" \
    -e "STRATUM_VLLM_MODEL=$VLLM_MODEL" \
    -e "STRATUM_RUN_ID=smoke_test" \
    -e "STRATUM_REPO_ID=$SMOKE_REPO" \
    -e "STRATUM_FRAMEWORK=auto" \
    -e "STRATUM_CAPTURE_PROMPTS=1" \
    -e "VLLM_HOST=$VLLM_HOST" \
    -e "VLLM_TIMEOUT=60" \
    -e "RUN_NUMBER=1" \
    stratum-lab-base \
    /app/run_repo.sh "$SMOKE_REPO" "$VLLM_HOST" "/app/output" "$SMOKE_TIMEOUT" \
    > "$SMOKE_DIR/container.log" 2>&1
SMOKE_EXIT=$?

echo "  Smoke test exit code: $SMOKE_EXIT"
echo ""

# ── Step 5: Validate smoke test results ──────────────────────────────────

echo "[deploy] Step 5: Validating smoke test results..."

SMOKE_OK=true

# Check status
SMOKE_STATUS=$(cat "$SMOKE_DIR/status.txt" 2>/dev/null || echo "MISSING")
echo "  Status: $SMOKE_STATUS"

# Check events
EVENTS_FILE="$SMOKE_DIR/events_run_1.jsonl"
EVENT_COUNT=0
if [ -f "$EVENTS_FILE" ]; then
    EVENT_COUNT=$(wc -l < "$EVENTS_FILE" 2>/dev/null | tr -d ' ')
fi
echo "  Events: $EVENT_COUNT"

if [ "$EVENT_COUNT" -eq 0 ]; then
    echo "  WARNING: No events captured"
    SMOKE_OK=false
fi

# Check for double-prefix bug (openai/openai/ in events)
DOUBLE_PREFIX=0
if [ -f "$EVENTS_FILE" ]; then
    DOUBLE_PREFIX=$(grep -c 'openai/openai/' "$EVENTS_FILE" 2>/dev/null || echo "0")
fi
echo "  Double-prefix occurrences: $DOUBLE_PREFIX"
if [ "$DOUBLE_PREFIX" -gt 0 ]; then
    echo "  FAIL: Double-prefix bug detected!"
    SMOKE_OK=false
fi

# Check I/O capture (system_prompt_preview or last_user_message_preview in events)
IO_CAPTURE=0
if [ -f "$EVENTS_FILE" ]; then
    IO_CAPTURE=$(grep -c 'system_prompt_preview\|last_user_message_preview\|output_preview' "$EVENTS_FILE" 2>/dev/null || echo "0")
fi
echo "  I/O capture fields: $IO_CAPTURE"
if [ "$IO_CAPTURE" -eq 0 ]; then
    echo "  WARNING: No I/O capture fields found"
fi

# Check tier_detail in run_metadata
TIER_DETAIL=""
if [ -f "$SMOKE_DIR/run_metadata_1.json" ]; then
    TIER_DETAIL=$(python3 -c "import json; print(json.load(open('$SMOKE_DIR/run_metadata_1.json')).get('tier_detail', 'MISSING'))" 2>/dev/null || echo "MISSING")
fi
echo "  tier_detail in metadata: $TIER_DETAIL"

echo ""

if [ "$SMOKE_OK" = true ]; then
    echo "  SMOKE TEST PASSED"
else
    echo "  SMOKE TEST: some checks failed (see warnings above)"
    echo "  Review $SMOKE_DIR/container.log for details"
fi

# Clean up
rm -rf "$SMOKE_DIR"

echo ""

# ── Step 6: Print mass scan command ──────────────────────────────────────

echo "[deploy] Step 6: Mass scan command"
echo ""
echo "  Run this to start the mass scan:"
echo ""
echo "    export VLLM_HOST=$VLLM_HOST"
echo "    export STRATUM_VLLM_MODEL=${VLLM_MODEL}"
echo "    nohup bash $DROPLET_DIR/scripts/orchestrate.sh \\"
echo "        $DROPLET_DIR/repos.txt \\"
echo "        ~/scan_output \\"
echo "        --concurrency 3 \\"
echo "        --timeout 600 \\"
echo "        --vllm-model ${VLLM_MODEL} \\"
echo "        > ~/scan_stdout.log 2>&1 &"
echo ""
echo "=== Deploy Complete ==="
