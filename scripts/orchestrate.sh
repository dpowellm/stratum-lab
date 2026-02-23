#!/usr/bin/env bash
# ============================================================================
# orchestrate.sh — Two-phase orchestration for stratum-lab Docker containers
#
# Usage:
#   orchestrate.sh <repo_list_file> <output_dir> \
#       [--concurrency N] [--timeout S] [--vllm-url URL] [--vllm-host HOST]
#       [--image NAME] [--vllm-model MODEL] [--script-dir DIR]
#       [--phase2-runs N] [--pilot-size N] [--skip-pilot]
#
# repo_list_file: one GitHub URL per line (or TSV with score\tURL)
# output_dir:     base directory for per-repo output subdirectories
#
# Phases:
#   Pilot:  Run PILOT_SIZE repos through Phase 1 to validate patcher/vLLM
#   Phase 1 (Discovery): All repos x 1 run each
#   Phase 2 (Depth):     Successful repos x PHASE2_RUNS more runs
#   Phase 3 (Collection): Build behavioral records from all runs
#
# Features:
#   - SHA256 hash directory names (first 12 chars) — collision-safe at 1000+ repos
#   - Resume support: skips repos/runs with existing run_metadata_N.json
#   - scan_log.txt: one-line-per-repo append log with timestamp, hash, status, URL
#   - Progress tracking with phase context and running totals
#   - Passes through STRATUM_VLLM_MODEL to containers
#   - STRATUM_RUN_ID / STRATUM_REPO_ID set per container
#   - Writes scan_summary.csv on completion
#   - Runs aggregate_results.py and ecosystem_report.py if available
#   - SIGINT/SIGTERM trap for clean shutdown
#   - vLLM health check before each container launch
#   - --network=host for external vLLM access
#   - Bind-mount scripts instead of COPY
#   - Docker system prune every 100 repos
#   - VLLM_TIMEOUT env var for containers
#   - Pilot quality gate before full scan
#   - Two-phase orchestration with depth runs for successful repos
#
# No set -e — we handle errors explicitly.
# ============================================================================

# ── VLLM_HOST preflight check ────────────────────────────────────────
if [ -z "$VLLM_HOST" ]; then
    echo "FATAL: VLLM_HOST is not set. Export it before running." >&2
    echo "  export VLLM_HOST=http://your-vllm-server:8000" >&2
    exit 1
fi
# Health check
if ! curl -sf "$VLLM_HOST/health" > /dev/null 2>&1; then
    # Try /v1/models as fallback health check
    if ! curl -sf "$VLLM_HOST/v1/models" > /dev/null 2>&1; then
        echo "FATAL: vLLM not reachable at $VLLM_HOST" >&2
        exit 1
    fi
fi
echo "vLLM health check passed: $VLLM_HOST"

# ── Defaults ─────────────────────────────────────────────────────────────
CONCURRENCY="${WORKERS:-5}"
TIMEOUT=600
VLLM_URL="${VLLM_HOST:-http://host.docker.internal:8000/v1}"
VLLM_HOST_ADDR=""
IMAGE="stratum-lab-base"
VLLM_MODEL="${STRATUM_VLLM_MODEL:-}"
VLLM_TIMEOUT=60
SCRIPT_DIR=""

# Two-phase defaults
PHASE1_RUNS=1
PHASE2_RUNS=4
PHASE2_TOTAL=$((PHASE1_RUNS + PHASE2_RUNS))  # 5 total runs per repo
PILOT_SIZE=30
SKIP_PILOT=false

# ── Parse arguments ──────────────────────────────────────────────────────
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --vllm-url)     VLLM_URL="$2"; shift 2 ;;
        --vllm-host)    VLLM_HOST_ADDR="$2"; shift 2 ;;
        --vllm-model)   VLLM_MODEL="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --script-dir)   SCRIPT_DIR="$2"; shift 2 ;;
        --phase2-runs)  PHASE2_RUNS="$2"; PHASE2_TOTAL=$((PHASE1_RUNS + PHASE2_RUNS)); shift 2 ;;
        --pilot-size)   PILOT_SIZE="$2"; shift 2 ;;
        --skip-pilot)   SKIP_PILOT=true; shift ;;
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
            echo "  --phase2-runs N     Number of additional depth runs per successful repo (default: 4)"
            echo "  --pilot-size N      Number of repos for pilot quality gate (default: 30)"
            echo "  --skip-pilot        Skip the pilot quality gate"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1"); shift ;;
    esac
done

REPO_LIST="${POSITIONAL[0]:-}"
OUTPUT_DIR="${POSITIONAL[1]:-}"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" 2>/dev/null && pwd || (mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd))"

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
RESULTS_DIR="$OUTPUT_DIR/results"
mkdir -p "$RESULTS_DIR"

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
INJECT_MAIN_PY=""
for candidate_dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$(pwd)"; do
    if [ -z "$RUN_REPO_SH" ] && [ -f "$candidate_dir/run_repo.sh" ]; then
        RUN_REPO_SH="$(cd "$candidate_dir" && pwd)/run_repo.sh"
    fi
    if [ -z "$SYNTHETIC_HARNESS_PY" ] && [ -f "$candidate_dir/synthetic_harness.py" ]; then
        SYNTHETIC_HARNESS_PY="$(cd "$candidate_dir" && pwd)/synthetic_harness.py"
    fi
    if [ -z "$INJECT_MAIN_PY" ] && [ -f "$candidate_dir/inject_main.py" ]; then
        INJECT_MAIN_PY="$(cd "$candidate_dir" && pwd)/inject_main.py"
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

# ── Helper: Extract repo URL from a line (supports TSV and JSONL) ────────
extract_repo_url() {
    local line="$1"
    # Check if line starts with { — JSONL format
    if [[ "$line" == \{* ]]; then
        # Extract repo_url from JSON object
        local url
        url=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('repo_url',''))" 2>/dev/null)
        if [ -n "$url" ]; then
            echo "$url"
            return
        fi
    fi
    # Fallback: TSV format — take last field (score\tURL or just URL)
    echo "$line" | awk '{print $NF}'
}

# ── Helper: Post-run behavioral status classification ─────────────────
classify_behavioral_status() {
    local events_file="$1"
    if [ ! -f "$events_file" ]; then
        echo "NO_EVENTS_FILE"
        return
    fi
    python3 -c "
import json, sys
agent_starts = 0
llm_starts = 0
try:
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                et = evt.get('event_type', '')
                if et == 'agent.task_start':
                    agent_starts += 1
                elif et == 'llm.call_start':
                    llm_starts += 1
            except json.JSONDecodeError:
                continue
except Exception:
    pass

if agent_starts > 0 and llm_starts > 0:
    print('FULL_BEHAVIORAL')
elif llm_starts > 0:
    print('LLM_ONLY')
else:
    print('NO_BEHAVIORAL_DATA')
" "$events_file" 2>/dev/null || echo "CLASSIFY_ERROR"
}

# ── Pip cache + disk space ──────────────────────────────────────────────
mkdir -p /tmp/pip-cache

check_disk_space() {
    local avail_gb
    avail_gb=$(df -BG /var/lib/docker 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "${avail_gb:-999}" -lt 5 ]; then
        echo "[WARN] Low disk space (${avail_gb}G). Running docker system prune..."
        docker system prune -f > /dev/null 2>&1
    fi
}

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
CURRENT_PHASE="Init"
PHASE_TOTAL=0

declare -A BG_PIDS         # bg_pid -> repo_url
declare -A BG_REPO_DIRS    # bg_pid -> output_dir_name
declare -A BG_START_TIMES  # bg_pid -> epoch seconds
declare -A BG_RUN_NUMBERS  # bg_pid -> run number
declare -A BG_CONTAINERS   # bg_pid -> container_name

# Generate a run UUID for this orchestration run
RUN_UUID=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "run-$(date +%s)")

# Count total repos
while IFS= read -r line; do
    # Trim whitespace — but NOT with xargs (it strips double quotes from JSON)
    line="${line#"${line%%[![:space:]]*}"}"  # trim leading
    line="${line%"${line##*[![:space:]]}"}"  # trim trailing
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
echo "[orchestrate] Results: $RESULTS_DIR"
echo "[orchestrate] Run ID: $RUN_UUID"
echo "[orchestrate] Phase2 runs: $PHASE2_RUNS (total runs per successful repo: $PHASE2_TOTAL)"
echo "[orchestrate] Pilot size: $PILOT_SIZE (skip_pilot=$SKIP_PILOT)"
echo ""

# ── SIGINT / SIGTERM handler ─────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[orchestrate] Shutting down — stopping running containers..."
    SHUTDOWN=true

    # Stop the watchdog first
    stop_watchdog

    # Force-stop all tracked containers
    for pid in "${!BG_CONTAINERS[@]}"; do
        local cname="${BG_CONTAINERS[$pid]}"
        if [ -n "$cname" ]; then
            docker stop -t 10 "$cname" 2>/dev/null || true
            docker rm -f "$cname" 2>/dev/null || true
        fi
    done

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
        local remaining=$((PHASE_TOTAL - COMPLETED - SKIPPED))
        if [ "$remaining" -lt 0 ]; then
            remaining=0
        fi
        local per_repo
        per_repo=$(python3 -c "print(round($elapsed / $COMPLETED, 1))" 2>/dev/null || echo "0")
        local eta_seconds
        eta_seconds=$(python3 -c "print(round($remaining * $per_repo))" 2>/dev/null || echo "0")
        local eta_hours
        eta_hours=$(python3 -c "print(round($eta_seconds / 3600, 1))" 2>/dev/null || echo "0")
        eta_str="${eta_hours}h"
    fi

    printf "[%s: %d/%d] Success: %d | Partial: %d | Tier2: %d | Failed: %d | Rate: %d%% | ETA: %s\n" \
        "$CURRENT_PHASE" "$((COMPLETED + SKIPPED))" "$PHASE_TOTAL" "$COUNT_SUCCESS" "$COUNT_PARTIAL" "$COUNT_TIER2" "$COUNT_FAILED" "$rate" "$eta_str"
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

    for dir in "$RESULTS_DIR"/*/; do
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
        SUCCESS|TIER1_5_SUCCESS)
            COUNT_SUCCESS=$((COUNT_SUCCESS + 1))
            ;;
        PARTIAL|PARTIAL_SUCCESS)
            COUNT_PARTIAL=$((COUNT_PARTIAL + 1))
            ;;
        TIER2_SUCCESS)
            COUNT_TIER2=$((COUNT_TIER2 + 1))
            ;;
        *)
            COUNT_FAILED=$((COUNT_FAILED + 1))
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
            local run_number="${BG_RUN_NUMBERS[$pid]}"
            local start_time="${BG_START_TIMES[$pid]}"
            local end_time
            end_time=$(date +%s)
            local elapsed=$((end_time - start_time))

            COMPLETED=$((COMPLETED + 1))

            # Determine status and tier from status.json or exit code
            local status="UNKNOWN"
            local tier="1"
            if [ -f "$RESULTS_DIR/$repo_dir/status.json" ]; then
                status=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status','UNKNOWN'))
except Exception:
    print('ERROR')
" "$RESULTS_DIR/$repo_dir/status.json" 2>/dev/null)
                tier=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('tier',1))
except Exception:
    print('1')
" "$RESULTS_DIR/$repo_dir/status.json" 2>/dev/null)
            elif [ "$exit_code" -ne 0 ]; then
                status="ERROR"
            fi

            # Post-run behavioral status classification
            local behavioral_status="UNKNOWN"
            local events_file="$RESULTS_DIR/$repo_dir/events_run_${run_number}.jsonl"
            if [ -f "$events_file" ]; then
                behavioral_status=$(classify_behavioral_status "$events_file")
            fi

            # Update running totals
            classify_status "$status" "$tier"

            # Progress line with phase context
            print_progress

            # Read tier detail for logging
            local tier_num="0"
            if [ -f "$RESULTS_DIR/$repo_dir/tier_detail.txt" ]; then
                tier_num=$(cat "$RESULTS_DIR/$repo_dir/tier_detail.txt" | tr -d ' \n')
            elif [ -f "$RESULTS_DIR/$repo_dir/tier.txt" ]; then
                tier_num=$(cat "$RESULTS_DIR/$repo_dir/tier.txt" | tr -d ' \n')
            fi

            # Append to scan_log.txt with tier and behavioral status
            local timestamp
            timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S")
            echo "$timestamp $repo_dir run=$run_number status=$status tier=$tier_num behavioral=$behavioral_status $repo_url" >> "$OUTPUT_DIR/scan_log.txt"

            # Ensure container is stopped (belt-and-suspenders with subshell cleanup)
            local cname="${BG_CONTAINERS[$pid]:-}"
            if [ -n "$cname" ] && docker inspect "$cname" >/dev/null 2>&1; then
                docker stop -t 5 "$cname" 2>/dev/null || true
                docker rm -f "$cname" 2>/dev/null || true
            fi

            unset "BG_PIDS[$pid]"
            unset "BG_REPO_DIRS[$pid]"
            unset "BG_START_TIMES[$pid]"
            unset "BG_RUN_NUMBERS[$pid]"
            unset "BG_CONTAINERS[$pid]"
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

# ── Wait for all background jobs to finish ───────────────────────────────
drain_jobs() {
    while [ "$(active_jobs)" -gt 0 ]; do
        collect_finished
        sleep 2
    done
    collect_finished
}

# ── Container watchdog — kills any container exceeding max allowed age ─
WATCHDOG_PID=""

container_watchdog() {
    local max_age=$((TIMEOUT + 180))
    while true; do
        sleep 60
        local now
        now=$(date +%s)
        # Find stratum containers by name prefix
        docker ps --filter "name=stratum-" --format "{{.ID}} {{.Names}}" 2>/dev/null | while IFS=' ' read -r cid cname; do
            [ -z "$cid" ] && continue
            local started
            started=$(docker inspect --format '{{.State.StartedAt}}' "$cid" 2>/dev/null) || continue
            local start_epoch
            start_epoch=$(date -d "$started" +%s 2>/dev/null) || continue
            local age=$((now - start_epoch))
            if [ "$age" -gt "$max_age" ]; then
                echo "[watchdog] Container $cname ($cid) running for ${age}s (limit: ${max_age}s) — killing"
                echo "[watchdog] $(date -u +%Y-%m-%dT%H:%M:%S) Killed $cname age=${age}s limit=${max_age}s" >> "$OUTPUT_DIR/watchdog.log"
                docker stop -t 10 "$cid" 2>/dev/null || true
                sleep 12
                docker kill "$cid" 2>/dev/null || true
                docker rm -f "$cid" 2>/dev/null || true
            fi
        done
    done
}

start_watchdog() {
    container_watchdog &
    WATCHDOG_PID=$!
    echo "[orchestrate] Container watchdog started (PID=$WATCHDOG_PID, max_age=$((TIMEOUT + 180))s)"
}

stop_watchdog() {
    if [ -n "$WATCHDOG_PID" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
    fi
}

# ── Launch a container for a single repo+run ─────────────────────────────
launch_container() {
    local repo_url="$1"
    local repo_dir="$2"
    local run_number="$3"

    mkdir -p "$RESULTS_DIR/$repo_dir"
    LAUNCHED=$((LAUNCHED + 1))

    # Disk space check every 50 launches
    if [ $((LAUNCHED % 50)) -eq 0 ]; then
        check_disk_space
    fi

    # Build volume mount args
    local MOUNT_ARGS=(
        -v "$RUN_REPO_SH:/app/run_repo.sh:ro"
        -v "$RESULTS_DIR/$repo_dir:/app/output"
        -v "/tmp/pip-cache:/root/.cache/pip"
    )
    if [ -n "$SYNTHETIC_HARNESS_PY" ]; then
        MOUNT_ARGS+=(-v "$SYNTHETIC_HARNESS_PY:/app/synthetic_harness.py:ro")
    fi
    if [ -n "$INJECT_MAIN_PY" ]; then
        MOUNT_ARGS+=(-v "$INJECT_MAIN_PY:/app/inject_main.py:ro")
    fi

    # Unique container name for tracking and cleanup
    local container_name="stratum-${repo_dir}-r${run_number}"

    # Remove any leftover container with this name (e.g., from a crashed prior run)
    docker rm -f "$container_name" 2>/dev/null || true

    (
        timeout --foreground --kill-after=30 $((TIMEOUT + 120)) \
            docker run --rm \
            --name "$container_name" \
            --init \
            --network=host \
            --entrypoint bash \
            --memory=4g \
            --cpus=2 \
            --stop-timeout=30 \
            "${MOUNT_ARGS[@]}" \
            -e "STRATUM_EVENTS_FILE=/app/output/events_run_${run_number}.jsonl" \
            -e "STRATUM_RUN_NUMBER=$run_number" \
            -e "STRATUM_VLLM_MODEL=$VLLM_MODEL" \
            -e "STRATUM_RUN_ID=$RUN_UUID" \
            -e "STRATUM_REPO_ID=$repo_url" \
            -e "STRATUM_FRAMEWORK=auto" \
            -e "STRATUM_CAPTURE_PROMPTS=1" \
            -e "VLLM_HOST=$VLLM_HOST" \
            -e "VLLM_TIMEOUT=$VLLM_TIMEOUT" \
            -e "RUN_NUMBER=$run_number" \
            "$IMAGE" \
            /app/run_repo.sh "$repo_url" "$VLLM_HOST" "/app/output" "$TIMEOUT" \
            > "$RESULTS_DIR/$repo_dir/container.log" 2>&1
        local rc=$?

        # If timeout fired (124=SIGTERM, 137=SIGKILL) or any abnormal exit,
        # the container may still be running — force-stop it.
        if [ $rc -ne 0 ]; then
            docker stop -t 10 "$container_name" 2>/dev/null || true
            docker rm -f "$container_name" 2>/dev/null || true
        fi
    ) &
    local bg_pid=$!

    BG_PIDS[$bg_pid]="$repo_url"
    BG_REPO_DIRS[$bg_pid]="$repo_dir"
    BG_START_TIMES[$bg_pid]=$(date +%s)
    BG_RUN_NUMBERS[$bg_pid]="$run_number"
    BG_CONTAINERS[$bg_pid]="$container_name"

    # Docker system prune every 100 repos to reclaim disk space
    if [ $((LAUNCHED % 100)) -eq 0 ]; then
        docker system prune -f > /dev/null 2>&1
    fi
}

# ── Run a list of repos through Phase 1 (1 run each) ────────────────────
run_phase1_for_list() {
    local repo_file="$1"
    local phase1_count=0

    while IFS= read -r line; do
        if [ "$SHUTDOWN" = true ]; then
            break
        fi

        # Trim whitespace — but NOT with xargs (it strips double quotes from JSON)
        line="${line#"${line%%[![:space:]]*}"}"  # trim leading
        line="${line%"${line##*[![:space:]]}"}"  # trim trailing
        [ -z "$line" ] && continue
        [[ "$line" == \#* ]] && continue

        # Support both TSV and JSONL input formats
        local repo_url
        repo_url=$(extract_repo_url "$line")
        [ -z "$repo_url" ] && continue

        local repo_dir
        repo_dir=$(repo_hash "$repo_url")

        # Resume: skip if run 1 already completed
        if [ -f "$RESULTS_DIR/$repo_dir/run_metadata_1.json" ]; then
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

        # Periodic vLLM health check every 50 repos
        phase1_count=$((phase1_count + 1))
        if [ $((phase1_count % 50)) -eq 0 ]; then
            echo "[orchestrate] Periodic vLLM health check (repo #$phase1_count)..."
            local health_retries=0
            while ! check_vllm; do
                health_retries=$((health_retries + 1))
                if [ $health_retries -gt 3 ]; then
                    echo "FATAL: vLLM unreachable after 3 retries during periodic check, aborting" >&2
                    SHUTDOWN=true
                    break
                fi
                echo "vLLM unreachable, retrying in 30s (attempt $health_retries/3)..."
                sleep 30
            done
            if [ "$SHUTDOWN" = true ]; then
                break
            fi
        fi

        # vLLM health check before launching container
        wait_for_vllm

        # Launch container with run_number=1
        launch_container "$repo_url" "$repo_dir" 1
    done < "$repo_file"

    # Wait for all remaining containers
    drain_jobs
}

# ============================================================================
# PILOT QUALITY GATE
# ============================================================================
if [ "$SKIP_PILOT" = false ] && [ "$TOTAL" -gt "$PILOT_SIZE" ]; then
    echo "=== PILOT: $PILOT_SIZE repos ==="
    head -"$PILOT_SIZE" "$REPO_LIST" > "$OUTPUT_DIR/pilot_repos.txt"

    # Reset counters for pilot
    COMPLETED=0; SKIPPED=0
    COUNT_SUCCESS=0; COUNT_PARTIAL=0; COUNT_TIER2=0; COUNT_FAILED=0
    CURRENT_PHASE="Pilot"
    PHASE_TOTAL=$PILOT_SIZE

    run_phase1_for_list "$OUTPUT_DIR/pilot_repos.txt"

    echo ""
    echo "[orchestrate] Pilot complete: $COMPLETED completed, $SKIPPED skipped"
    print_status_summary

    # Check pilot quality
    if [ -f "$SCRIPT_DIR/check_pilot_quality.py" ]; then
        echo ""
        echo "[orchestrate] Running pilot quality check..."
        python3 "$SCRIPT_DIR/check_pilot_quality.py" \
            --results-dir "$RESULTS_DIR" \
            --instr-threshold 0.20 \
            --model-threshold 0.15
        if [ $? -ne 0 ]; then
            echo "PILOT FAILED -- fix patcher/vLLM before running full scan"
            exit 1
        fi
        echo "PILOT PASSED -- proceeding to full scan"
    else
        echo "[orchestrate] Warning: check_pilot_quality.py not found in $SCRIPT_DIR, skipping quality gate"
    fi
    echo ""
elif [ "$SKIP_PILOT" = true ]; then
    echo "[orchestrate] Pilot skipped (--skip-pilot)"
    echo ""
else
    echo "[orchestrate] Pilot skipped (total repos $TOTAL <= pilot size $PILOT_SIZE)"
    echo ""
fi

# ============================================================================
# PHASE 1: DISCOVERY (all repos x 1 run each)
# ============================================================================
echo "=== PHASE 1: DISCOVERY ($TOTAL repos x $PHASE1_RUNS run) ==="

# Start the container watchdog safety net
start_watchdog

# Reset counters for Phase 1 (pilot repos with existing metadata will be skipped via resume)
COMPLETED=0; SKIPPED=0; LAUNCHED=0
COUNT_SUCCESS=0; COUNT_PARTIAL=0; COUNT_TIER2=0; COUNT_FAILED=0
CURRENT_PHASE="Phase1"
PHASE_TOTAL=$TOTAL

run_phase1_for_list "$REPO_LIST"

echo ""
echo "=== PHASE 1 COMPLETE ==="
echo "[orchestrate] Phase 1: $COMPLETED completed, $SKIPPED skipped out of $TOTAL total"
print_status_summary

# Identify successful repos
find "$RESULTS_DIR" -name "status.json" -exec grep -l '"status"' {} \; 2>/dev/null | while read -r sf; do
    local_status=$(python3 -c "
import json, sys
try:
    s = json.load(open(sys.argv[1])).get('status','')
    if s in ('SUCCESS','PARTIAL_SUCCESS','TIER1_5_SUCCESS','TIER2_SUCCESS','PARTIAL'):
        print('OK')
except Exception:
    pass
" "$sf" 2>/dev/null)
    if [ "$local_status" = "OK" ]; then
        dirname "$sf" | xargs basename
    fi
done > "$OUTPUT_DIR/successful_hashes.txt" 2>/dev/null

# Fallback: also check status.txt files for backwards compatibility
if [ -f "$OUTPUT_DIR/successful_hashes.txt" ]; then
    find "$RESULTS_DIR" -name "status.txt" 2>/dev/null | while read -r st; do
        if grep -q "SUCCESS\|PARTIAL_SUCCESS\|TIER1_5_SUCCESS\|TIER2_SUCCESS" "$st" 2>/dev/null; then
            dirname "$st" | xargs basename
        fi
    done >> "$OUTPUT_DIR/successful_hashes.txt" 2>/dev/null
    # Deduplicate
    sort -u "$OUTPUT_DIR/successful_hashes.txt" -o "$OUTPUT_DIR/successful_hashes.txt"
fi

# ── Recover repos with events but no status (timeout-killed containers) ──
for repo_dir in "$RESULTS_DIR"/*/; do
    [ -d "$repo_dir" ] || continue
    REPO_HASH=$(basename "$repo_dir")

    # Skip if already successful
    if grep -qx "$REPO_HASH" "$OUTPUT_DIR/successful_hashes.txt" 2>/dev/null; then
        continue
    fi

    # Check for orphaned events
    for ef in "$repo_dir"/events_run_*.jsonl; do
        [ -f "$ef" ] && [ -s "$ef" ] || continue
        local_beh=$(classify_behavioral_status "$ef")
        if [ "$local_beh" = "FULL_BEHAVIORAL" ] || [ "$local_beh" = "LLM_ONLY" ]; then
            echo "$REPO_HASH" >> "$OUTPUT_DIR/successful_hashes.txt"
            echo "[orchestrate] Recovered orphaned events: $REPO_HASH (behavioral=$local_beh)"

            # Write minimal recovery metadata so build_behavioral_records can process
            if [ ! -f "$repo_dir/run_metadata_1.json" ]; then
                RECOVERED_URL=$(grep "$REPO_HASH" "$OUTPUT_DIR/scan_log.txt" 2>/dev/null | tail -1 | awk '{print $NF}')
                EVT_COUNT=$(wc -l < "$ef" 2>/dev/null | tr -d ' ')
                python3 -c "
import json, sys
from datetime import datetime, timezone
json.dump({
    'repo_url': sys.argv[1],
    'repo_full_name': '',
    'run_number': 1,
    'run_id': '',
    'scan_id': '$RUN_UUID',
    'exit_code': -1,
    'status': 'RECOVERED_FROM_TIMEOUT',
    'tier': 0,
    'tier_detail': 0,
    'entry_point': '',
    'framework_detected': 'unknown',
    'events_file': '$(basename "$ef")',
    'event_count': int(sys.argv[2]),
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'timeout_seconds': $TIMEOUT,
    'model_name': '${VLLM_MODEL:-}',
    'recovered': True,
}, open(sys.argv[3], 'w'), indent=2)
" "${RECOVERED_URL:-unknown}" "$EVT_COUNT" "$repo_dir/run_metadata_1.json" 2>/dev/null \
                    && echo "[orchestrate] Wrote recovery metadata: $REPO_HASH ($EVT_COUNT events)" || true
            fi
            break  # one events file is enough
        fi
    done
done
sort -u "$OUTPUT_DIR/successful_hashes.txt" -o "$OUTPUT_DIR/successful_hashes.txt" 2>/dev/null || true

SUCCESSFUL=0
if [ -f "$OUTPUT_DIR/successful_hashes.txt" ]; then
    SUCCESSFUL=$(wc -l < "$OUTPUT_DIR/successful_hashes.txt" | tr -d ' ')
fi
echo ""
echo "[orchestrate] Successful repos for Phase 2: $SUCCESSFUL"

# ============================================================================
# PHASE 2: DEPTH (successful repos x PHASE2_RUNS more runs)
# ============================================================================
if [ "$SUCCESSFUL" -gt 0 ] && [ "$PHASE2_RUNS" -gt 0 ]; then
    echo ""
    echo "=== PHASE 2: DEPTH ($SUCCESSFUL repos x $PHASE2_RUNS more runs) ==="

    # Reset counters for Phase 2
    COMPLETED=0; SKIPPED=0; LAUNCHED=0
    COUNT_SUCCESS=0; COUNT_PARTIAL=0; COUNT_TIER2=0; COUNT_FAILED=0
    CURRENT_PHASE="Phase2"
    PHASE_TOTAL=$((SUCCESSFUL * PHASE2_RUNS))

    while IFS= read -r hash; do
        if [ "$SHUTDOWN" = true ]; then
            break
        fi

        [ -z "$hash" ] && continue

        # Read repo URL from Phase 1 metadata
        local_repo_url=""
        if [ -f "$RESULTS_DIR/$hash/run_metadata_1.json" ]; then
            local_repo_url=$(python3 -c "import json; print(json.load(open('$RESULTS_DIR/$hash/run_metadata_1.json'))['repo_url'])" 2>/dev/null)
        fi
        if [ -z "$local_repo_url" ]; then
            # Fallback: try status.json
            if [ -f "$RESULTS_DIR/$hash/status.json" ]; then
                local_repo_url=$(python3 -c "import json; print(json.load(open('$RESULTS_DIR/$hash/status.json')).get('repo',''))" 2>/dev/null)
            fi
        fi
        if [ -z "$local_repo_url" ]; then
            echo "[orchestrate] Warning: cannot determine repo URL for hash $hash, skipping"
            continue
        fi

        for run in $(seq 2 "$PHASE2_TOTAL"); do
            if [ "$SHUTDOWN" = true ]; then
                break
            fi

            # Resume: skip if this run already completed
            if [ -f "$RESULTS_DIR/$hash/run_metadata_${run}.json" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Wait until we have a free slot
            while [ "$(active_jobs)" -ge "$CONCURRENCY" ]; do
                collect_finished
                sleep 1
            done

            # Collect any that finished
            collect_finished

            # vLLM health check before launching container
            wait_for_vllm

            # Launch container with run_number=$run
            launch_container "$local_repo_url" "$hash" "$run"
        done
    done < "$OUTPUT_DIR/successful_hashes.txt"

    # Wait for all remaining containers
    drain_jobs

    echo ""
    echo "=== PHASE 2 COMPLETE ==="
    echo "[orchestrate] Phase 2: $COMPLETED completed, $SKIPPED skipped out of $PHASE_TOTAL total runs"
    print_status_summary
else
    echo ""
    echo "[orchestrate] Skipping Phase 2: no successful repos or phase2-runs=0"
fi

# ============================================================================
# PHASE 3: PER-REPO SEMANTIC ANALYSIS
# ============================================================================
echo ""
echo "=== PHASE 3: SEMANTIC ANALYSIS ==="

if [ -f "$SCRIPT_DIR/analyze_semantics.py" ] && [ -n "$VLLM_HOST" ]; then
    for repo_dir in "$RESULTS_DIR"/*/; do
        [ -d "$repo_dir" ] || continue
        REPO_NAME=$(basename "$repo_dir")

        # Skip repos without events
        if ! ls "$repo_dir"/events_run_*.jsonl 1>/dev/null 2>&1; then
            continue
        fi

        # Skip if already analyzed
        if [ -f "$repo_dir/semantic_analysis.json" ]; then
            continue
        fi

        echo "[orchestrate] Phase 3: Running semantic analysis for $REPO_NAME..."
        python3 "$SCRIPT_DIR/analyze_semantics.py" \
            --results-dir "$repo_dir" \
            --vllm-url "$VLLM_HOST/v1" \
            --model "${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}" \
            --output "$repo_dir/semantic_analysis.json" \
            2>>"$repo_dir/semantic_analysis_stderr.log" || {
            echo '{"error": "semantic analysis failed", "aggregate_scores": {}}' > "$repo_dir/semantic_analysis.json"
            echo "[WARN] Semantic analysis failed for $REPO_NAME"
        }
    done
else
    echo "[orchestrate] Skipping Phase 3: analyze_semantics.py or VLLM_HOST not available"
fi

# ============================================================================
# PHASE 0b: DATA TOPOLOGY SCAN (tool registrations, state keys, permissions)
# ============================================================================
echo ""
echo "=== PHASE 0b: DATA TOPOLOGY SCAN ==="

if [ -f "$SCRIPT_DIR/scan_data_topology.py" ]; then
    for repo_dir in "$RESULTS_DIR"/*/; do
        [ -d "$repo_dir" ] || continue
        REPO_NAME=$(basename "$repo_dir")

        # Skip if already scanned
        if [ -f "$repo_dir/data_topology.json" ]; then
            continue
        fi

        # Look for cloned repo source (may be in repo_dir/repo or similar)
        REPO_SRC=""
        for candidate in "$repo_dir/repo" "$repo_dir/source" "$repo_dir"; do
            if [ -d "$candidate" ] && ls "$candidate"/*.py 1>/dev/null 2>&1; then
                REPO_SRC="$candidate"
                break
            fi
        done

        if [ -n "$REPO_SRC" ]; then
            echo "[Phase 0b] Scanning data topology for $REPO_NAME..."
            python3 "$SCRIPT_DIR/scan_data_topology.py" "$REPO_SRC" > "$repo_dir/data_topology.json" 2>/dev/null || echo '{"error":"scan_failed"}' > "$repo_dir/data_topology.json"
        fi
    done
else
    echo "[orchestrate] Skipping Phase 0b: scan_data_topology.py not found"
fi

# ============================================================================
# PHASE 4: COLLECTION (behavioral records)
# ============================================================================
echo ""
echo "=== PHASE 4: COLLECTION ==="

if [ -f "$SCRIPT_DIR/build_behavioral_records.py" ]; then
    mkdir -p "$OUTPUT_DIR/behavioral_records"
    python3 "$SCRIPT_DIR/build_behavioral_records.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$OUTPUT_DIR/behavioral_records/"
else
    echo "[orchestrate] Warning: build_behavioral_records.py not found in $SCRIPT_DIR, skipping collection"
fi

# ============================================================================
# PHASE 4b: RESEARCH ENRICHMENTS (privacy, permissions, cost, audit)
# ============================================================================
echo ""
echo "=== PHASE 4b: RESEARCH ENRICHMENTS ==="

if [ -f "$SCRIPT_DIR/compute_enrichments.py" ]; then
    for repo_dir in "$RESULTS_DIR"/*/; do
        [ -d "$repo_dir" ] || continue
        REPO_NAME=$(basename "$repo_dir")

        # Skip repos without events
        if ! ls "$repo_dir"/events_run_*.jsonl 1>/dev/null 2>&1; then
            continue
        fi

        # Skip if already enriched
        if [ -f "$repo_dir/enrichments.json" ]; then
            continue
        fi

        echo "[Phase 4b] Computing research enrichments for $REPO_NAME..."
        python3 "$SCRIPT_DIR/compute_enrichments.py" "$repo_dir" 2>/dev/null || echo "Warning: enrichment computation failed for $REPO_NAME"
    done
else
    echo "[orchestrate] Skipping Phase 4b: compute_enrichments.py not found"
fi

# ============================================================================
# PHASE 5: CROSS-REPO REMEDIATION MINING
# ============================================================================
echo ""
echo "=== PHASE 5: REMEDIATION MINING ==="

if [ -f "$SCRIPT_DIR/mine_remediations.py" ]; then
    echo "[orchestrate] Phase 5: Mining remediation evidence across corpus..."
    python3 "$SCRIPT_DIR/mine_remediations.py" \
        --scan-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/remediation_evidence.json" \
        2>>"$OUTPUT_DIR/remediation_mining_stderr.log" || {
        echo '{"error": "remediation mining failed", "findings": []}' > "$OUTPUT_DIR/remediation_evidence.json"
        echo "[WARN] Remediation mining failed"
    }
else
    echo "[orchestrate] Skipping Phase 5: mine_remediations.py not found"
fi

# ============================================================================
# PHASE 5b: CORPUS-WIDE RISK MODEL (DTMC learning)
# ============================================================================
echo ""
echo "=== PHASE 5b: CORPUS RISK MODEL ==="

if [ -f "$SCRIPT_DIR/compute_risk_model.py" ]; then
    CORPUS_DIR="$OUTPUT_DIR"
    ALL_RECORDS_DIR="$RESULTS_DIR"
    echo "[Phase 5b] Computing corpus risk model..."
    python3 "$SCRIPT_DIR/compute_risk_model.py" "$ALL_RECORDS_DIR" > "$CORPUS_DIR/risk_model.json" 2>/dev/null || echo '{"error":"model_failed"}' > "$CORPUS_DIR/risk_model.json"
else
    echo "[orchestrate] Skipping Phase 5b: compute_risk_model.py not found"
fi

# ── Stop the watchdog ──────────────────────────────────────────────────
stop_watchdog

# ── Final summary and post-processing ────────────────────────────────────
echo ""
echo "[orchestrate] Done: Phase 1 + Phase 2 complete"
print_status_summary

write_summary

# Run aggregator if available
if [ -f "$SCRIPT_DIR/aggregate_results.py" ]; then
    echo ""
    echo "[orchestrate] Running aggregate_results.py..."
    local br_flag=""
    if [ -d "$OUTPUT_DIR/behavioral_records" ]; then
        br_flag="--behavioral-records-dir $OUTPUT_DIR/behavioral_records"
    fi
    python3 "$SCRIPT_DIR/aggregate_results.py" "$RESULTS_DIR" $br_flag
fi

# ecosystem_report.py removed in v2 — replaced by build_behavioral_records.py (Phase 3)

# Print quick status distribution
echo ""
echo "Status distribution:"
if [ -f "$OUTPUT_DIR/scan_summary.csv" ]; then
    tail -n +2 "$OUTPUT_DIR/scan_summary.csv" | cut -d',' -f2 | sort | uniq -c | sort -rn
fi

# Print behavioral yield summary from scan_log.txt
echo ""
echo "Behavioral yield:"
if [ -f "$OUTPUT_DIR/scan_log.txt" ]; then
    echo "  FULL_BEHAVIORAL:    $(grep -c 'behavioral=FULL_BEHAVIORAL' "$OUTPUT_DIR/scan_log.txt" 2>/dev/null || echo 0)"
    echo "  LLM_ONLY:           $(grep -c 'behavioral=LLM_ONLY' "$OUTPUT_DIR/scan_log.txt" 2>/dev/null || echo 0)"
    echo "  NO_BEHAVIORAL_DATA: $(grep -c 'behavioral=NO_BEHAVIORAL_DATA' "$OUTPUT_DIR/scan_log.txt" 2>/dev/null || echo 0)"
fi

echo "=== SCAN COMPLETE ==="
