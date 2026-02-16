#!/usr/bin/env bash
# ============================================================================
# orchestrate.sh — Run N concurrent stratum-lab Docker containers
#
# Usage:
#   orchestrate.sh <repo_list_file> <output_dir> \
#       [--concurrency N] [--timeout S] [--vllm-url URL] [--image NAME]
#       [--vllm-model MODEL]
#
# repo_list_file: one GitHub URL per line (or TSV with score\tURL)
# output_dir:     base directory for per-repo output subdirectories
#
# Features:
#   - SHA256 hash directory names (first 12 chars) — collision-safe at 1000+ repos
#   - Resume support: skips repos with existing status.json
#   - scan_log.txt: one-line-per-repo append log with timestamp, hash, status, URL
#   - Progress tracking: [completed/total] dir_hash STATUS (Xs)
#   - Passes through STRATUM_VLLM_MODEL to containers
#   - STRATUM_RUN_ID / STRATUM_REPO_ID are set inside run_repo.sh, not here
#   - Writes scan_summary.csv on completion
#   - Runs aggregate_results.py if available
#   - SIGINT/SIGTERM trap for clean shutdown
#
# No set -e — we handle errors explicitly.
# ============================================================================

# ── Defaults ─────────────────────────────────────────────────────────────
CONCURRENCY="${WORKERS:-5}"
TIMEOUT=600
VLLM_URL="${VLLM_HOST:-http://host.docker.internal:8000/v1}"
IMAGE="stratum-lab:latest"
VLLM_MODEL="${STRATUM_VLLM_MODEL:-}"

# ── Parse arguments ──────────────────────────────────────────────────────
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --timeout)     TIMEOUT="$2"; shift 2 ;;
        --vllm-url)    VLLM_URL="$2"; shift 2 ;;
        --vllm-model)  VLLM_MODEL="$2"; shift 2 ;;
        --image)       IMAGE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: orchestrate.sh <repo_list> <output_dir> [--concurrency N] [--timeout S] [--vllm-url URL] [--image NAME] [--vllm-model MODEL]"
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

# ── Helper: SHA256-based directory name (12 hex chars) ───────────────────
repo_hash() {
    echo -n "$1" | sha256sum | cut -c1-12
}

# ── State tracking ───────────────────────────────────────────────────────
TOTAL=0; COMPLETED=0; SKIPPED=0; SHUTDOWN=false

declare -A BG_PIDS         # bg_pid -> repo_url
declare -A BG_REPO_DIRS    # bg_pid -> output_dir_name
declare -A BG_START_TIMES  # bg_pid -> epoch seconds

# Count total repos
while IFS= read -r line; do
    line=$(echo "$line" | xargs 2>/dev/null || true)
    [ -z "$line" ] && continue
    [[ "$line" == \#* ]] && continue
    TOTAL=$((TOTAL + 1))
done < "$REPO_LIST"

echo "[orchestrate] $TOTAL repos, concurrency=$CONCURRENCY, timeout=${TIMEOUT}s"
echo "[orchestrate] Image: $IMAGE"
echo "[orchestrate] vLLM: $VLLM_URL"
[ -n "$VLLM_MODEL" ] && echo "[orchestrate] Model: $VLLM_MODEL"
echo "[orchestrate] Output: $OUTPUT_DIR"
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
    write_summary
    exit 130
}
trap cleanup SIGINT SIGTERM

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

            # Determine status from status.json or exit code
            local status="UNKNOWN"
            if [ -f "$OUTPUT_DIR/$repo_dir/status.json" ]; then
                status=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status','UNKNOWN'))
except Exception:
    print('ERROR')
" "$OUTPUT_DIR/$repo_dir/status.json" 2>/dev/null)
            elif [ "$exit_code" -ne 0 ]; then
                status="ERROR"
            fi

            # Progress line
            printf "[%d/%d] %s %s (%ds)\n" "$COMPLETED" "$TOTAL" "$repo_dir" "$status" "$elapsed"

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

    # Launch container in background
    mkdir -p "$OUTPUT_DIR/$repo_dir"
    (
        docker run --rm \
            -v "$OUTPUT_DIR/$repo_dir:/app/output" \
            --add-host=host.docker.internal:host-gateway \
            -e "STRATUM_VLLM_MODEL=$VLLM_MODEL" \
            -e "STRATUM_EVENTS_FILE=/app/output/stratum_events.jsonl" \
            "$IMAGE" \
            "$repo_url" "$TIMEOUT" "$VLLM_URL" \
            >/dev/null 2>&1 || true
    ) &
    bg_pid=$!

    BG_PIDS[$bg_pid]="$repo_url"
    BG_REPO_DIRS[$bg_pid]="$repo_dir"
    BG_START_TIMES[$bg_pid]=$(date +%s)

done < "$REPO_LIST"

# Wait for all remaining containers
while [ "$(active_jobs)" -gt 0 ]; do
    collect_finished
    sleep 2
done
collect_finished

echo ""
echo "[orchestrate] Done: $COMPLETED completed, $SKIPPED skipped, $TOTAL total"

write_summary

# Run aggregator if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/aggregate_results.py" ]; then
    echo ""
    echo "[orchestrate] Running aggregate_results.py..."
    python3 "$SCRIPT_DIR/aggregate_results.py" "$OUTPUT_DIR"
fi

# Print quick status distribution
echo ""
echo "Status distribution:"
if [ -f "$OUTPUT_DIR/scan_summary.csv" ]; then
    tail -n +2 "$OUTPUT_DIR/scan_summary.csv" | cut -d',' -f2 | sort | uniq -c | sort -rn
fi
