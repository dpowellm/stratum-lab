#!/usr/bin/env bash
# ============================================================================
# orchestrate.sh — Run N concurrent stratum-lab Docker containers
#
# Usage:
#   orchestrate.sh <repo_list_file> <output_dir> \
#       [--concurrency N] [--timeout S] [--vllm-url URL] [--vllm-host HOST]
#       [--image NAME] [--vllm-model MODEL] [--script-dir DIR]
#
# repo_list_file: one GitHub URL per line (or TSV with score\tURL)
# output_dir:     base directory for per-repo output subdirectories
#
# Features:
#   - SHA256 hash directory names (first 12 chars) — collision-safe at 1000+ repos
#   - Resume support: skips repos with existing status.json
#   - scan_log.txt: one-line-per-repo append log with timestamp, hash, status, URL
#   - Progress tracking with running totals: Success/Partial/Tier2/Failed + Rate + ETA
#   - Passes through STRATUM_VLLM_MODEL to containers
#   - STRATUM_RUN_ID / STRATUM_REPO_ID set per container
#   - Writes scan_summary.csv on completion
#   - Runs aggregate_results.py and ecosystem_report.py if available
#   - SIGINT/SIGTERM trap for clean shutdown
#   - vLLM health check before each container launch
#   - Docker network support (stratum-net if available)
#   - Bind-mount scripts instead of COPY
#   - Docker system prune every 100 repos
#   - VLLM_TIMEOUT env var for containers
#
# No set -e — we handle errors explicitly.
# ============================================================================

# ── Defaults ─────────────────────────────────────────────────────────────
CONCURRENCY="${WORKERS:-5}"
TIMEOUT=600
VLLM_URL="${VLLM_HOST:-http://host.docker.internal:8000/v1}"
VLLM_HOST_ADDR=""
IMAGE="stratum-lab-base"
VLLM_MODEL="${STRATUM_VLLM_MODEL:-}"
VLLM_TIMEOUT=60
SCRIPT_DIR=""

# ── Parse arguments ──────────────────────────────────────────────────────
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --timeout)     TIMEOUT="$2"; shift 2 ;;
        --vllm-url)    VLLM_URL="$2"; shift 2 ;;
        --vllm-host)   VLLM_HOST_ADDR="$2"; shift 2 ;;
        --vllm-model)  VLLM_MODEL="$2"; shift 2 ;;
        --image)       IMAGE="$2"; shift 2 ;;
        --script-dir)  SCRIPT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: orchestrate.sh <repo_list> <output_dir> [options]"
            echo ""
            echo "Options:"
            echo "  --concurrency N     Max concurrent containers (default: 5)"
            echo "  --timeout S         Per-repo timeout in seconds (default: 600)"
            echo "  --vllm-url URL      Full vLLM URL with /v1 suffix (legacy)"
            echo "  --vllm-host HOST    vLLM host without /v1 suffix (preferred)"
            echo "  --vllm-model MODEL  Model name for vLLM"
            echo "  --image NAME        Docker image name (default: stratum-lab-base)"
            echo "  --script-dir DIR    Directory containing run_repo.sh and synthetic_harness.py"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1"); shift ;;
    esac
done

REPO_LIST="${POSITIONAL[0]:-}"
OUTPUT_DIR="${POSITIONAL[1]:-}"

if [ -z "$REPO_LIST" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: repo_list_file and output_dir are required" >&2
    echo "Usage: orchestrate.sh <repo_list> <output_dir> [options]" >&2
    exit 1
fi

if [ ! -f "$REPO_LIST" ]; then
    echo "Error: $REPO_LIST not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Resolve SCRIPT_DIR ───────────────────────────────────────────────────
# If --script-dir was not provided, default to the directory containing this script
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ── Resolve VLLM_HOST (no /v1 suffix) ───────────────────────────────────
# --vllm-host takes priority; otherwise strip /v1 from --vllm-url
if [ -n "$VLLM_HOST_ADDR" ]; then
    VLLM_HOST="$VLLM_HOST_ADDR"
else
    # Strip trailing /v1 or /v1/ from VLLM_URL to derive the host
    VLLM_HOST=$(echo "$VLLM_URL" | sed 's|/v1/*$||')
fi

# ── Locate bind-mount scripts ───────────────────────────────────────────
RUN_REPO_SH=""
SYNTHETIC_HARNESS_PY=""

# Check SCRIPT_DIR first, then parent directory, then cwd
for candidate_dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$(pwd)"; do
    if [ -z "$RUN_REPO_SH" ] && [ -f "$candidate_dir/run_repo.sh" ]; then
        RUN_REPO_SH="$(cd "$candidate_dir" && pwd)/run_repo.sh"
    fi
    if [ -z "$SYNTHETIC_HARNESS_PY" ] && [ -f "$candidate_dir/synthetic_harness.py" ]; then
        SYNTHETIC_HARNESS_PY="$(cd "$candidate_dir" && pwd)/synthetic_harness.py"
    fi
done

if [ -z "$RUN_REPO_SH" ]; then
    echo "Error: run_repo.sh not found in $SCRIPT_DIR or parent directories" >&2
    exit 1
fi
if [ -z "$SYNTHETIC_HARNESS_PY" ]; then
    echo "Warning: synthetic_harness.py not found — containers may fail" >&2
fi

# ── Helper: SHA256-based directory name (12 hex chars) ───────────────────
repo_hash() {
    echo -n "$1" | sha256sum | cut -c1-12
}

# ── Docker network detection ────────────────────────────────────────────
DOCKER_NETWORK_FLAG=""
if docker network inspect stratum-net > /dev/null 2>&1; then
    DOCKER_NETWORK_FLAG="--network=stratum-net"
    echo "[orchestrate] Using Docker network: stratum-net"
else
    echo "[orchestrate] Docker network 'stratum-net' not found, using default"
fi

# ── vLLM health check ──────────────────────────────────────────────────
check_vllm() {
    curl -s --max-time 5 "${VLLM_HOST}/v1/models" > /dev/null 2>&1
    return $?
}

wait_for_vllm() {
    VLLM_RETRIES=0
    while ! check_vllm; do
        VLLM_RETRIES=$((VLLM_RETRIES + 1))
        if [ $VLLM_RETRIES -gt 60 ]; then
            echo "ERROR: vLLM server unreachable for 5 minutes, aborting"
            exit 1
        fi
        echo "vLLM server unreachable, waiting 5s (attempt $VLLM_RETRIES)..."
        sleep 5
    done
}

# ── State tracking ───────────────────────────────────────────────────────
TOTAL=0; COMPLETED=0; SKIPPED=0; LAUNCHED=0; SHUTDOWN=false
COUNT_SUCCESS=0; COUNT_PARTIAL=0; COUNT_TIER2=0; COUNT_FAILED=0
SCAN_START_TIME=$(date +%s)

declare -A BG_PIDS         # bg_pid -> repo_url
declare -A BG_REPO_DIRS    # bg_pid -> output_dir_name
declare -A BG_START_TIMES  # bg_pid -> epoch seconds

# Generate a run UUID for this orchestration run
RUN_UUID=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "run-$(date +%s)")

# Count total repos
while IFS= read -r line; do
    line=$(echo "$line" | xargs 2>/dev/null || true)
    [ -z "$line" ] && continue
    [[ "$line" == \#* ]] && continue
    TOTAL=$((TOTAL + 1))
done < "$REPO_LIST"

echo "[orchestrate] $TOTAL repos, concurrency=$CONCURRENCY, timeout=${TIMEOUT}s"
echo "[orchestrate] Image: $IMAGE"
echo "[orchestrate] vLLM host: $VLLM_HOST"
[ -n "$VLLM_MODEL" ] && echo "[orchestrate] Model: $VLLM_MODEL"
echo "[orchestrate] Scripts: $SCRIPT_DIR"
echo "[orchestrate] run_repo.sh: $RUN_REPO_SH"
[ -n "$SYNTHETIC_HARNESS_PY" ] && echo "[orchestrate] synthetic_harness.py: $SYNTHETIC_HARNESS_PY"
echo "[orchestrate] Output: $OUTPUT_DIR"
echo "[orchestrate] Run ID: $RUN_UUID"
echo ""

# ── SIGINT / SIGTERM handler ─────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[orchestrate] Shutting down — waiting for running containers..."
    SHUTDOWN=true

    for pid in "${!BG_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    # Collect any remaining results
    collect_finished

    echo "[orchestrate] $COMPLETED completed, $SKIPPED skipped out of $TOTAL total"
    print_status_summary
    write_summary
    exit 130
}
trap cleanup SIGINT SIGTERM

# ── Progress display helper ──────────────────────────────────────────────
print_progress() {
    local now
    now=$(date +%s)
    local elapsed=$((now - SCAN_START_TIME))

    # Calculate success rate (of completed only)
    local rate=0
    if [ "$COMPLETED" -gt 0 ]; then
        rate=$((COUNT_SUCCESS * 100 / COMPLETED))
    fi

    # Calculate ETA
    local eta_str="--"
    if [ "$COMPLETED" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        local remaining=$((TOTAL - COMPLETED - SKIPPED))
        local per_repo
        per_repo=$(python3 -c "print(round($elapsed / $COMPLETED, 1))" 2>/dev/null || echo "0")
        local eta_seconds
        eta_seconds=$(python3 -c "print(round($remaining * $per_repo))" 2>/dev/null || echo "0")
        local eta_hours
        eta_hours=$(python3 -c "print(round($eta_seconds / 3600, 1))" 2>/dev/null || echo "0")
        eta_str="${eta_hours}h"
    fi

    printf "[%d/%d] Success: %d | Partial: %d | Tier2: %d | Failed: %d | Rate: %d%% | ETA: %s\n" \
        "$COMPLETED" "$TOTAL" "$COUNT_SUCCESS" "$COUNT_PARTIAL" "$COUNT_TIER2" "$COUNT_FAILED" "$rate" "$eta_str"
}

print_status_summary() {
    echo ""
    echo "Running totals:"
    echo "  Success: $COUNT_SUCCESS"
    echo "  Partial: $COUNT_PARTIAL"
    echo "  Tier2:   $COUNT_TIER2"
    echo "  Failed:  $COUNT_FAILED"
}

# ── Summary CSV writer ───────────────────────────────────────────────────
write_summary() {
    local csv="$OUTPUT_DIR/scan_summary.csv"
    echo "repo,status,tier,exit_code,entry_point,event_count,duration_seconds" > "$csv"

    for dir in "$OUTPUT_DIR"/*/; do
        [ -d "$dir" ] || continue
        local sf="$dir/status.json"
        if [ -f "$sf" ]; then
            python3 -c "
import json, sys, csv, io
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([
        d.get('repo',''),
        d.get('status','UNKNOWN'),
        d.get('tier', 1),
        d.get('exit_code',''),
        d.get('entry_point',''),
        d.get('event_count',0),
        d.get('duration_seconds',0),
    ])
    print(out.getvalue().strip())
except Exception:
    print('PARSE_ERROR,,,,,,')
" "$sf" >> "$csv" 2>/dev/null
        fi
    done

    echo "[orchestrate] Summary: $csv"
}

# ── Classify status for running totals ───────────────────────────────────
classify_status() {
    local status="$1"
    local tier="$2"
    case "$status" in
        SUCCESS)
            COUNT_SUCCESS=$((COUNT_SUCCESS + 1))
            ;;
        PARTIAL|PARTIAL_SUCCESS)
            COUNT_PARTIAL=$((COUNT_PARTIAL + 1))
            ;;
        *)
            if [ "$tier" = "2" ]; then
                COUNT_TIER2=$((COUNT_TIER2 + 1))
            else
                COUNT_FAILED=$((COUNT_FAILED + 1))
            fi
            ;;
    esac
}

# ── Collect finished jobs ────────────────────────────────────────────────
collect_finished() {
    for pid in "${!BG_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local exit_code=$?

            local repo_url="${BG_PIDS[$pid]}"
            local repo_dir="${BG_REPO_DIRS[$pid]}"
            local start_time="${BG_START_TIMES[$pid]}"
            local end_time
            end_time=$(date +%s)
            local elapsed=$((end_time - start_time))

            COMPLETED=$((COMPLETED + 1))

            # Determine status and tier from status.json or exit code
            local status="UNKNOWN"
            local tier="1"
            if [ -f "$OUTPUT_DIR/$repo_dir/status.json" ]; then
                status=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status','UNKNOWN'))
except Exception:
    print('ERROR')
" "$OUTPUT_DIR/$repo_dir/status.json" 2>/dev/null)
                tier=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('tier',1))
except Exception:
    print('1')
" "$OUTPUT_DIR/$repo_dir/status.json" 2>/dev/null)
            elif [ "$exit_code" -ne 0 ]; then
                status="ERROR"
            fi

            # Update running totals
            classify_status "$status" "$tier"

            # Progress line with running totals
            print_progress

            # Append to scan_log.txt
            local timestamp
            timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")
            echo "$timestamp $repo_dir $status $repo_url" >> "$OUTPUT_DIR/scan_log.txt"

            unset "BG_PIDS[$pid]"
            unset "BG_REPO_DIRS[$pid]"
            unset "BG_START_TIMES[$pid]"
        fi
    done
}

# ── Count active jobs ────────────────────────────────────────────────────
active_jobs() {
    local count=0
    for pid in "${!BG_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# ── Main loop ────────────────────────────────────────────────────────────
while IFS= read -r line; do
    if [ "$SHUTDOWN" = true ]; then
        break
    fi

    line=$(echo "$line" | xargs 2>/dev/null || true)
    [ -z "$line" ] && continue
    [[ "$line" == \#* ]] && continue

    # Support TSV format: score\tURL — take last field
    repo_url=$(echo "$line" | awk '{print $NF}')
    repo_dir=$(repo_hash "$repo_url")

    # Resume: skip if already scanned
    if [ -f "$OUTPUT_DIR/$repo_dir/status.json" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Wait until we have a free slot
    while [ "$(active_jobs)" -ge "$CONCURRENCY" ]; do
        collect_finished
        sleep 1
    done

    # Collect any that finished while reading the list
    collect_finished

    # vLLM health check before launching container
    wait_for_vllm

    # Launch container in background
    mkdir -p "$OUTPUT_DIR/$repo_dir"
    LAUNCHED=$((LAUNCHED + 1))

    # Build volume mount args
    MOUNT_ARGS=(
        -v "$RUN_REPO_SH:/app/run_repo.sh:ro"
        -v "$OUTPUT_DIR/$repo_dir:/app/output"
    )
    if [ -n "$SYNTHETIC_HARNESS_PY" ]; then
        MOUNT_ARGS+=(-v "$SYNTHETIC_HARNESS_PY:/app/synthetic_harness.py:ro")
    fi

    (
        docker run --rm \
            $DOCKER_NETWORK_FLAG \
            "${MOUNT_ARGS[@]}" \
            --add-host=host.docker.internal:host-gateway \
            -e "STRATUM_EVENTS_FILE=/app/output/stratum_events.jsonl" \
            -e "STRATUM_VLLM_MODEL=$VLLM_MODEL" \
            -e "STRATUM_RUN_ID=$RUN_UUID" \
            -e "STRATUM_REPO_ID=$repo_url" \
            -e "STRATUM_FRAMEWORK=auto" \
            -e "STRATUM_CAPTURE_PROMPTS=1" \
            -e "VLLM_HOST=$VLLM_HOST" \
            -e "VLLM_TIMEOUT=$VLLM_TIMEOUT" \
            "$IMAGE" \
            bash /app/run_repo.sh "$repo_url" "$VLLM_HOST" "/app/output" "$TIMEOUT" \
            >/dev/null 2>&1 || true
    ) &
    bg_pid=$!

    BG_PIDS[$bg_pid]="$repo_url"
    BG_REPO_DIRS[$bg_pid]="$repo_dir"
    BG_START_TIMES[$bg_pid]=$(date +%s)

    # Docker system prune every 100 repos to reclaim disk space
    if [ $((LAUNCHED % 100)) -eq 0 ]; then
        docker system prune -f > /dev/null 2>&1
    fi

done < "$REPO_LIST"

# Wait for all remaining containers
while [ "$(active_jobs)" -gt 0 ]; do
    collect_finished
    sleep 2
done
collect_finished

echo ""
echo "[orchestrate] Done: $COMPLETED completed, $SKIPPED skipped, $TOTAL total"
print_status_summary

write_summary

# Run aggregator if available
if [ -f "$SCRIPT_DIR/aggregate_results.py" ]; then
    echo ""
    echo "[orchestrate] Running aggregate_results.py..."
    python3 "$SCRIPT_DIR/aggregate_results.py" "$OUTPUT_DIR"
fi

# Run ecosystem report if available
if [ -f "$SCRIPT_DIR/ecosystem_report.py" ]; then
    echo ""
    echo "[orchestrate] Running ecosystem_report.py..."
    python3 "$SCRIPT_DIR/ecosystem_report.py" "$OUTPUT_DIR"
fi

# Print quick status distribution
echo ""
echo "Status distribution:"
if [ -f "$OUTPUT_DIR/scan_summary.csv" ]; then
    tail -n +2 "$OUTPUT_DIR/scan_summary.csv" | cut -d',' -f2 | sort | uniq -c | sort -rn
fi
