#!/usr/bin/env bash
# ============================================================================
# run_repo.sh -- Stratum Lab Docker ENTRYPOINT
#
# Two-tier execution harness for behavioral scanning of AI agent repos.
#   Tier 1: Native execution (clone -> install -> detect entry -> run)
#   Tier 2: Synthetic harness fallback (extract agent defs -> generate -> run)
#
# Usage:
#   bash /app/run_repo.sh "$REPO_URL" "$VLLM_HOST" "/app/output" 300
#
# Arguments:
#   $1  REPO_URL   - GitHub URL to clone
#   $2  VLLM_HOST  - vLLM base URL (e.g. http://host.docker.internal:8000)
#   $3  OUTPUT_DIR - Directory for all output artifacts
#   $4  TIMEOUT    - Max seconds per execution attempt (default: 300)
#
# CRITICAL: No set -e ANYWHERE. The retry loop MUST survive Python failures.
# ============================================================================


# ============================================================================
# ARGUMENTS
# ============================================================================

REPO_URL="${1:-}"
VLLM_HOST="${2:-}"
OUTPUT_DIR="${3:-/app/output}"
TIMEOUT="${4:-300}"

if [ -z "$REPO_URL" ]; then
    echo "Usage: run_repo.sh <REPO_URL> <VLLM_HOST> <OUTPUT_DIR> [TIMEOUT]" >&2
    exit 1
fi
if [ -z "$VLLM_HOST" ]; then
    echo "ERROR: VLLM_HOST is required" >&2
    exit 1
fi

export STRATUM_RUN_NUMBER="${RUN_NUMBER:-1}"
EVENTS_FILE="$OUTPUT_DIR/events_run_${STRATUM_RUN_NUMBER}.jsonl"

mkdir -p "$OUTPUT_DIR"

START_EPOCH=$(date +%s)

log() { echo "[stratum] $(date +%H:%M:%S) $*" >&2; }


# ============================================================================
# OUTPUT WRITERS
# ============================================================================

write_output() {
    local status="$1"
    local exit_code="${2:-0}"
    local entry_point="${3:-}"
    local tier="${4:-1}"

    # Write individual status files
    echo "$status"      > "$OUTPUT_DIR/status.txt"
    echo "$exit_code"   > "$OUTPUT_DIR/exit_code.txt"
    echo "$entry_point" > "$OUTPUT_DIR/entry_point.txt"
    echo "$tier"        > "$OUTPUT_DIR/tier.txt"

    # Count events if present
    local event_count=0
    if [ -f "$EVENTS_FILE" ]; then
        event_count=$(wc -l < "$EVENTS_FILE" 2>/dev/null || echo "0")
        event_count=$(echo "$event_count" | tr -d ' ')
    fi

    # Calculate duration
    local end_epoch
    end_epoch=$(date +%s)
    local duration=$(( end_epoch - START_EPOCH ))

    # Write machine-readable status.json
    python3 -c "
import json, sys, os
d = {
    'repo': sys.argv[1],
    'status': sys.argv[2],
    'exit_code': int(sys.argv[3]),
    'entry_point': sys.argv[4],
    'tier': int(sys.argv[5]),
    'event_count': int(sys.argv[6]),
    'duration_seconds': int(sys.argv[7]),
    'run_id': os.environ.get('STRATUM_RUN_ID', ''),
    'scan_id': os.environ.get('STRATUM_SCAN_ID', ''),
    'vllm_model': os.environ.get('STRATUM_VLLM_MODEL', ''),
    'error_log_tail': '',
}
# Read stderr tail for error context
for log_path in ['$OUTPUT_DIR/stderr.log', '$OUTPUT_DIR/tier2_stderr.log']:
    try:
        with open(log_path) as f:
            lines = f.readlines()
            if lines:
                d['error_log_tail'] = ''.join(lines[-20:])[:2000]
                break
    except: pass
with open(sys.argv[8], 'w') as f:
    json.dump(d, f, indent=2)
" "$REPO_URL" "$status" "$exit_code" "$entry_point" "$tier" \
  "$event_count" "$duration" "$OUTPUT_DIR/status.json" 2>/dev/null \
  || echo "{\"repo\":\"$REPO_URL\",\"status\":\"$status\",\"tier\":$tier}" > "$OUTPUT_DIR/status.json"
}


# ============================================================================
# PIP NAME MAPPING TABLE
# ============================================================================

declare -A PIP_NAME_MAP=(
    # Image / media
    [cv2]=opencv-python  [cv]=opencv-python  [PIL]=Pillow
    [skimage]=scikit-image

    # Data / science
    [sklearn]=scikit-learn  [yaml]=pyyaml  [Bio]=biopython
    [faiss]=faiss-cpu

    # Web scraping / parsing
    [bs4]=beautifulsoup4  [newspaper]=newspaper3k
    [trafilatura]=trafilatura  [lxml]=lxml

    # Dotenv / config
    [dotenv]=python-dotenv  [pydantic_settings]=pydantic-settings

    # Crypto / auth
    [Crypto]=pycryptodome  [jose]=python-jose  [jwt]=PyJWT

    # Date / string
    [dateutil]=python-dateutil  [Levenshtein]=python-Levenshtein

    # System / hardware
    [gi]=PyGObject  [usb]=pyusb  [serial]=pyserial
    [magic]=python-magic  [wx]=wxPython

    # Attributes / types
    [attr]=attrs

    # Documents
    [docx]=python-docx  [pptx]=python-pptx  [xlrd]=xlrd
    [openpyxl]=openpyxl  [markdown]=markdown

    # Vector databases
    [chromadb]=chromadb  [pinecone]=pinecone-client
    [weaviate]=weaviate-client  [qdrant_client]=qdrant-client

    # LangChain ecosystem
    [langchain_community]=langchain-community
    [langchain_openai]=langchain-openai
    [langchain_anthropic]=langchain-anthropic
    [langchain_experimental]=langchain-experimental
    [langchain_core]=langchain-core
    [langchain_text_splitters]=langchain-text-splitters

    # Search / tools
    [tavily]=tavily-python  [exa_py]=exa-py
    [duckduckgo_search]=duckduckgo-search

    # LLM providers / tools
    [litellm]=litellm  [tiktoken]=tiktoken  [cohere]=cohere
    [together]=together  [unstructured]=unstructured

    # CrewAI
    [crewai_tools]=crewai-tools

    # CLI / display
    [rich]=rich  [tqdm]=tqdm  [click]=click  [typer]=typer
    [fire]=fire  [colorama]=colorama  [termcolor]=termcolor
)

resolve_pip_name() {
    local module="$1"
    # 1. Explicit map lookup
    if [ -n "${PIP_NAME_MAP[$module]+x}" ]; then
        echo "${PIP_NAME_MAP[$module]}"; return
    fi
    # 2. Replace underscores with hyphens
    local hyphenated="${module//_/-}"
    if [ "$hyphenated" != "$module" ]; then
        echo "$hyphenated"; return
    fi
    # 3. Return as-is
    echo "$module"
}


# ============================================================================
# STEP 1: CLONE
# ============================================================================

log "Step 1: Cloning $REPO_URL ..."

if ! git clone --depth 1 "$REPO_URL" /tmp/repo 2>/dev/null; then
    log "FAIL: Clone failed"
    write_output "CLONE_FAILED" 1 "" 0
    exit 1
fi


# ============================================================================
# STEP 1.5: PHASE 0 — STATIC DEFENSIVE PATTERN SCAN + SOURCE SNAPSHOT
# ============================================================================

log "Phase 0: Scanning defensive patterns..."
python3 /app/scripts/scan_defensive_patterns.py /tmp/repo "$OUTPUT_DIR/defensive_patterns.json" 2>/dev/null || {
    echo '{"error": "defensive pattern scan failed", "patterns": [], "summary": {}}' > "$OUTPUT_DIR/defensive_patterns.json"
}

# Phase 0: Source snapshot (cap at 200 py files, exclude .git/venv/__pycache__/node_modules)
find /tmp/repo -name "*.py" -not -path "*/.git/*" -not -path "*/venv/*" \
    -not -path "*/__pycache__/*" -not -path "*/node_modules/*" | \
    head -200 | tar czf "$OUTPUT_DIR/source_snapshot.tar.gz" -T - 2>/dev/null || true

log "Phase 0: Complete"


# ============================================================================
# STEP 2: ENVIRONMENT SETUP
# ============================================================================

log "Step 2: Setting environment variables ..."

# -- LLM provider keys (placeholders to prevent KeyError crashes) --
export OPENAI_API_KEY="sk-placeholder"
export OPENAI_BASE_URL="${VLLM_HOST}/v1"
export OPENAI_API_BASE="${VLLM_HOST}/v1"
export ANTHROPIC_API_KEY="sk-placeholder"

# -- Search / tool API keys --
export TAVILY_API_KEY="tvly-placeholder"
export SERPER_API_KEY="placeholder"
export EXA_API_KEY="placeholder"
export GOOGLE_API_KEY="placeholder"
export BROWSERLESS_API_KEY="placeholder"
export SERPAPI_API_KEY="placeholder"
export BING_SUBSCRIPTION_KEY="placeholder"
export BING_SEARCH_URL="https://placeholder"

# -- Additional LLM providers --
export GROQ_API_KEY="placeholder"
export TOGETHER_API_KEY="placeholder"
export COHERE_API_KEY="placeholder"

# -- Hugging Face --
export HUGGINGFACEHUB_API_TOKEN="placeholder"
export HUGGING_FACE_HUB_TOKEN="placeholder"

# -- Vector databases --
export PINECONE_API_KEY="placeholder"
export WEAVIATE_API_KEY="placeholder"
export QDRANT_API_KEY="placeholder"

# -- Browser / scraping --
export FIRECRAWL_API_KEY="placeholder"
export BROWSERBASE_API_KEY="placeholder"
export BROWSERBASE_PROJECT_ID="placeholder"

# -- LangChain / LangSmith --
export LANGCHAIN_API_KEY="placeholder"
export LANGCHAIN_TRACING_V2="false"
export LANGSMITH_API_KEY="placeholder"

# -- Stratum identifiers --
# orchestrate.sh passes the scan-level UUID via -e STRATUM_RUN_ID.
# Preserve it as STRATUM_SCAN_ID before generating a per-container execution ID.
export STRATUM_SCAN_ID="${STRATUM_RUN_ID:-}"
RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4().hex[:16])' 2>/dev/null || echo "run_$$")
export STRATUM_RUN_ID="$RUN_ID"
export STRATUM_REPO_ID="$REPO_URL"
export STRATUM_EVENTS_FILE="$EVENTS_FILE"
export STRATUM_VLLM_MODEL="${STRATUM_VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

log "  SCAN_ID=${STRATUM_SCAN_ID:-<standalone>}  RUN_ID=$RUN_ID  MODEL=$STRATUM_VLLM_MODEL"


# ============================================================================
# STEP 3: .env FILE HANDLING
# ============================================================================

log "Step 3: Handling .env files ..."

# Copy any .env.example / .env.sample / env.example files found in the repo
find /tmp/repo -name '.env.example' -o -name '.env.sample' -o -name 'env.example' | while read envfile; do
    target="$(dirname "$envfile")/.env"
    if [ ! -f "$target" ]; then
        log "  Creating $target from $envfile"
        cp "$envfile" "$target"
        sed -i 's/=$/=placeholder/g' "$target"
        sed -i 's/=""$/="placeholder"/g' "$target"
        sed -i "s/=''$/='placeholder'/g" "$target"
    fi
done

# Ensure a root-level .env always exists
if [ ! -f /tmp/repo/.env ]; then
    log "  Creating default /tmp/repo/.env"
    echo "OPENAI_API_KEY=sk-placeholder" > /tmp/repo/.env
    echo "OPENAI_BASE_URL=${VLLM_HOST}/v1" >> /tmp/repo/.env
fi


# ============================================================================
# STEP 4: DEPENDENCY INSTALLATION
# ============================================================================

log "Step 4: Installing dependencies ..."

# Protected packages — pre-installed in the Docker image, must not be overwritten
# by repo requirements (which may pin incompatible versions).
PROTECTED_RE="^(openai|langchain|langchain-core|langchain-openai|langgraph|crewai|pyautogen|autogen|litellm|anthropic)([=><!\[]|$)"

# Install all requirements*.txt files found within 3 levels
find /tmp/repo -maxdepth 3 -name 'requirements*.txt' | while read reqfile; do
    log "  Installing $reqfile"
    FILTERED=$(mktemp)
    grep -viE "$PROTECTED_RE" "$reqfile" > "$FILTERED" 2>/dev/null || true
    TOTAL_LINES=$(wc -l < "$reqfile" | tr -d ' ')
    FILTERED_LINES=$(wc -l < "$FILTERED" | tr -d ' ')
    SKIPPED=$(( TOTAL_LINES - FILTERED_LINES ))
    if [ "$SKIPPED" -gt 0 ]; then
        log "    Skipped $SKIPPED protected packages"
    fi
    pip install -r "$FILTERED" --quiet 2>&1 | tail -5
    rm -f "$FILTERED"
done

# Install the repo itself if it has pyproject.toml or setup.py
# --no-deps: deps already installed (filtered) from requirements.txt above;
# without --no-deps, editable install can override protected packages
# (e.g., pyproject.toml pins crewai==0.71.0 -> silently downgrades ours)
if [ -f /tmp/repo/pyproject.toml ] || [ -f /tmp/repo/setup.py ]; then
    log "  Installing repo package (editable, --no-deps)"
    pip install -e /tmp/repo --no-deps --quiet 2>&1 | tail -5 || true
fi


# ============================================================================
# STEP 5: ENTRY POINT DETECTION (SCORING)
# ============================================================================

log "Step 5: Detecting entry point ..."

# We use a temp file to accumulate candidates with scores.
# Format: SCORE<TAB>PATH (one per line, highest score wins)
CANDIDATES_FILE=$(mktemp)

# Helper: add a candidate with a given score
add_candidate() {
    local score="$1"
    local path="$2"
    echo -e "${score}\t${path}" >> "$CANDIDATES_FILE"
}

# Helper: count directory depth relative to /tmp/repo
dir_depth() {
    local relpath="${1#/tmp/repo/}"
    echo "$relpath" | tr '/' '\n' | wc -l
}

# Helper: count lines in a file
line_count() {
    wc -l < "$1" 2>/dev/null || echo "0"
}

# Scan all Python files (exclude hidden dirs, venv, node_modules, __pycache__)
ALL_PY_FILES=$(mktemp)
find /tmp/repo -maxdepth 5 -name '*.py' \
    -not -path '*/\.*' -not -path '*/node_modules/*' \
    -not -path '*/venv/*' -not -path '*/.venv/*' \
    -not -path '*/__pycache__/*' \
    -not -name 'setup.py' -not -name 'conftest.py' \
    2>/dev/null > "$ALL_PY_FILES" || true

while IFS= read -r pyfile; do
    [ -z "$pyfile" ] && continue
    [ -f "$pyfile" ] || continue

    score=0
    relpath="${pyfile#/tmp/repo/}"
    basename_file=$(basename "$pyfile")
    dirpath=$(dirname "$pyfile")
    reldirpath="${dirpath#/tmp/repo}"
    reldirpath="${reldirpath#/}"  # strip leading slash

    has_main_guard=false
    has_framework_call=false
    has_framework_import=false
    has_server_import=false
    has_cli_import=false
    has_only_classes=false
    has_test_import=false

    # Check file content (single pass with grep)
    if grep -q '__name__.*__main__\|__name__.*==.*"__main__"' "$pyfile" 2>/dev/null; then
        has_main_guard=true
    fi
    if grep -qE '\.kickoff\s*\(|\.invoke\s*\(|\.run\s*\(|initiate_chat\s*\(' "$pyfile" 2>/dev/null; then
        has_framework_call=true
    fi
    if grep -qE 'from\s+(crewai|langgraph|autogen|langchain)|import\s+(crewai|langgraph|autogen|langchain)' "$pyfile" 2>/dev/null; then
        has_framework_import=true
    fi
    if grep -qE 'import\s+(uvicorn|flask|fastapi|gunicorn|streamlit|gradio)|from\s+(uvicorn|flask|fastapi|gunicorn|streamlit|gradio)' "$pyfile" 2>/dev/null; then
        has_server_import=true
    fi
    if grep -qE 'import\s+(argparse|click|typer)|from\s+(argparse|click|typer)' "$pyfile" 2>/dev/null; then
        has_cli_import=true
    fi
    if grep -qE 'import\s+(unittest|pytest)|from\s+(unittest|pytest)' "$pyfile" 2>/dev/null; then
        has_test_import=true
    fi

    # ── POSITIVE SIGNALS ──

    # +15: has BOTH framework imports AND execution calls (strongest signal)
    if [ "$has_framework_import" = true ] && [ "$has_framework_call" = true ]; then
        score=$((score + 15))
    fi

    # +10: has __main__ guard with framework execution inside
    if [ "$has_main_guard" = true ] && [ "$has_framework_call" = true ]; then
        score=$((score + 10))
    # +8: has __main__ guard with framework import (likely entry point)
    elif [ "$has_main_guard" = true ] && [ "$has_framework_import" = true ]; then
        score=$((score + 8))
    # +7: has framework call (standalone file, no main guard)
    elif [ "$has_framework_call" = true ]; then
        score=$((score + 7))
    # +2: has __main__ guard (generic, no framework)
    elif [ "$has_main_guard" = true ]; then
        score=$((score + 2))
    # +1: has framework import (might be a definition-only file — Tier 1.5 can handle it)
    elif [ "$has_framework_import" = true ]; then
        score=$((score + 1))
    else
        # No main guard, no framework call, no framework import -- not a candidate
        continue
    fi

    # +8: has CLI framework (argparse/click/typer) — likely a runnable entry point
    if [ "$has_cli_import" = true ]; then
        score=$((score + 8))
    fi

    # +5: named main.py/app.py/run.py at repo root
    if [ -z "$reldirpath" ] || [ "$reldirpath" = "." ]; then
        case "$basename_file" in
            main.py|app.py|run.py) score=$((score + 5)) ;;
        esac
    fi

    # +5: named cli.py, pipeline.py, workflow.py, crew.py (common entry points)
    case "$basename_file" in
        cli.py|pipeline.py|workflow.py|crew.py) score=$((score + 5)) ;;
    esac

    # +3: named main.py/app.py/run.py in src/ or examples/
    case "$reldirpath" in
        src|src/*|examples|examples/*)
            case "$basename_file" in
                main.py|app.py|run.py) score=$((score + 3)) ;;
            esac
            ;;
    esac

    # ── NEGATIVE SIGNALS ──

    # -20: imports uvicorn/flask/fastapi/gunicorn/streamlit/gradio (server-based)
    if [ "$has_server_import" = true ]; then
        score=$((score - 20))
    fi

    # -15: is in test/tests directory
    case "$relpath" in
        test/*|tests/*|*/test/*|*/tests/*) score=$((score - 15)) ;;
    esac

    # -10: filename starts with test_ or ends with _test.py
    case "$basename_file" in
        test_*|*_test.py) score=$((score - 10)) ;;
    esac

    # -5: imports unittest or pytest
    if [ "$has_test_import" = true ]; then
        score=$((score - 5))
    fi

    # -5: deeply nested (>3 dirs deep from repo root)
    depth=$(dir_depth "$pyfile")
    if [ "$depth" -gt 3 ]; then
        score=$((score - 5))
    fi

    # -5: file size > 500 lines
    lines=$(line_count "$pyfile")
    lines=$(echo "$lines" | tr -d ' ')
    if [ "$lines" -gt 500 ]; then
        score=$((score - 5))
    fi

    add_candidate "$score" "$pyfile"

done < "$ALL_PY_FILES"

rm -f "$ALL_PY_FILES"

# Pick the best candidate (highest score)
# For monorepo tie-breaking: among tied candidates, prefer fewest imports
ENTRY=""
BEST_SCORE=-999
RUNNER_UP=""
RUNNER_UP_SCORE=-999
TOTAL_CANDIDATES=0

if [ -s "$CANDIDATES_FILE" ]; then
    # Sort by score descending, then for tie-breaking by import count ascending
    while IFS=$'\t' read -r cand_score cand_path; do
        [ -z "$cand_score" ] && continue
        TOTAL_CANDIDATES=$((TOTAL_CANDIDATES + 1))
        if [ "$cand_score" -gt "$BEST_SCORE" ]; then
            # Demote current best to runner-up
            RUNNER_UP_SCORE=$BEST_SCORE
            RUNNER_UP="$ENTRY"
            BEST_SCORE=$cand_score
            ENTRY="$cand_path"
        elif [ "$cand_score" -eq "$BEST_SCORE" ] && [ -n "$ENTRY" ]; then
            # Tie-breaking: prefer candidate with fewer import lines (simpler)
            imports_current=$(grep -cE '^(import |from .* import )' "$ENTRY" 2>/dev/null || echo 999)
            imports_new=$(grep -cE '^(import |from .* import )' "$cand_path" 2>/dev/null || echo 999)
            if [ "$imports_new" -lt "$imports_current" ]; then
                RUNNER_UP_SCORE=$BEST_SCORE
                RUNNER_UP="$ENTRY"
                ENTRY="$cand_path"
            elif [ -z "$RUNNER_UP" ]; then
                RUNNER_UP_SCORE=$cand_score
                RUNNER_UP="$cand_path"
            fi
        elif [ "$cand_score" -gt "$RUNNER_UP_SCORE" ]; then
            RUNNER_UP_SCORE=$cand_score
            RUNNER_UP="$cand_path"
        fi
    done < "$CANDIDATES_FILE"
fi

rm -f "$CANDIDATES_FILE"

# Log detection reasoning
log "  Detection reasoning: $TOTAL_CANDIDATES candidate(s) evaluated"
if [ -n "$ENTRY" ]; then
    log "  Winner: ${ENTRY#/tmp/repo/} (score=$BEST_SCORE)"
fi
if [ -n "$RUNNER_UP" ]; then
    log "  Runner-up: ${RUNNER_UP#/tmp/repo/} (score=$RUNNER_UP_SCORE)"
fi

# Report detection result
TIER1_SKIP=""
if [ -z "$ENTRY" ]; then
    log "  No entry point found -- will try Tier 1.5 / Tier 2"
    TIER1_SKIP="no_entry_point"
elif [ "$BEST_SCORE" -lt 0 ]; then
    log "  Top candidate has negative score ($BEST_SCORE): $ENTRY -- server-based, skipping"
    write_output "SERVER_BASED" 0 "$ENTRY" 0
    TIER1_SKIP="server_based"
else
    log "  Entry point: $ENTRY (score=$BEST_SCORE)"
fi


# ============================================================================
# STEP 6: PYTHONPATH CONFIGURATION
# ============================================================================

log "Step 6: Configuring PYTHONPATH ..."

export PYTHONPATH="/tmp/repo:/tmp/repo/src:${PYTHONPATH:-}"

# Walk up from entry point dir to find and install the nearest package root
if [ -n "$ENTRY" ] && [ -z "$TIER1_SKIP" ]; then
    ENTRY_DIR=$(dirname "$ENTRY")
    SEARCH_DIR="$ENTRY_DIR"
    while [ "$SEARCH_DIR" != "/tmp" ] && [ "$SEARCH_DIR" != "/" ]; do
        if [ -f "$SEARCH_DIR/pyproject.toml" ] || [ -f "$SEARCH_DIR/setup.py" ]; then
            log "  Installing package at $SEARCH_DIR"
            pip install -e "$SEARCH_DIR" --no-deps --quiet 2>&1 | tail -5 || true
            break
        fi
        SEARCH_DIR=$(dirname "$SEARCH_DIR")
    done
fi

log "  PYTHONPATH=$PYTHONPATH"


# ============================================================================
# STEP 6.5: INJECT __main__ BLOCK IF NEEDED
# ============================================================================

if [ -n "$ENTRY" ] && [ -z "$TIER1_SKIP" ] && [ -f /app/inject_main.py ]; then
    log "Step 6.5: Checking if entry point needs __main__ injection ..."
    INJECTED_ENTRY=$(python3 /app/inject_main.py "$ENTRY" /tmp/repo 2>/dev/null)
    INJECT_EXIT=$?
    if [ $INJECT_EXIT -eq 0 ] && [ -n "$INJECTED_ENTRY" ] && [ "$INJECTED_ENTRY" != "$ENTRY" ]; then
        log "  Injected wrapper: $INJECTED_ENTRY (original: ${ENTRY#/tmp/repo/})"
        ENTRY="$INJECTED_ENTRY"
    elif [ $INJECT_EXIT -ne 0 ]; then
        log "  No orchestration objects found for injection"
    fi
fi


# ============================================================================
# STEP 7: TIER 1 -- NATIVE EXECUTION WITH AUTO-RETRY
# ============================================================================

# Tier-specific timeouts — DERIVED from container $TIMEOUT
# Budget: clone/install/detection gets 25%, remaining 75% split across tiers
# Split: Tier1 gets 15%, Tier1.5 gets 25%, Tier2 gets 35%
TIER1_TIMEOUT=$(( TIMEOUT * 15 / 100 ))
TIER1_5_TIMEOUT=$(( TIMEOUT * 25 / 100 ))
TIER2_TIMEOUT=$(( TIMEOUT * 35 / 100 ))

# Minimum floors so tiers get enough time even with small $TIMEOUT
[ "$TIER1_TIMEOUT" -lt 30 ] && TIER1_TIMEOUT=30
[ "$TIER1_5_TIMEOUT" -lt 45 ] && TIER1_5_TIMEOUT=45
[ "$TIER2_TIMEOUT" -lt 60 ] && TIER2_TIMEOUT=60

log "  Timeout budget: T1=${TIER1_TIMEOUT}s T1.5=${TIER1_5_TIMEOUT}s T2=${TIER2_TIMEOUT}s (container=${TIMEOUT}s)"

MAX_RETRIES=5
ATTEMPT=0
ATTEMPTED_MODULES=""
EXIT_CODE=0
TIER1_STATUS=""
TIER=1

# Export tier metadata for event tagging
export STRATUM_TIER="1"

if [ -n "$TIER1_SKIP" ]; then
    # Already classified above (no entry point or server-based)
    if [ "$TIER1_SKIP" = "server_based" ]; then
        TIER1_STATUS="SERVER_BASED"
    else
        TIER1_STATUS="NO_ENTRY_POINT"
    fi
    TIER=0
else
    log "Step 7: Executing Tier 1 (native, timeout=${TIER1_TIMEOUT}s) ..."

    for (( ATTEMPT=1; ATTEMPT<=MAX_RETRIES; ATTEMPT++ )); do
        log "  Attempt $ATTEMPT/$MAX_RETRIES: python $(basename "$ENTRY")"

        # Run from the entry point's directory
        cd "$(dirname "$ENTRY")"
        timeout "$TIER1_TIMEOUT" python "$(basename "$ENTRY")" \
            > "$OUTPUT_DIR/stdout.log" \
            2> "$OUTPUT_DIR/stderr.log"
        EXIT_CODE=$?

        # Separate stderr capture per run
        cp "$OUTPUT_DIR/stderr.log" "$OUTPUT_DIR/stderr_run_${STRATUM_RUN_NUMBER}.log" 2>/dev/null || true

        # ── Check for events regardless of exit code ──
        HAS_EVENTS=false
        if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
            HAS_EVENTS=true
        fi

        # ── Classify the result ──

        # Exit 0: clean success
        if [ $EXIT_CODE -eq 0 ]; then
            if [ "$HAS_EVENTS" = true ]; then
                TIER1_STATUS="SUCCESS"
            else
                TIER1_STATUS="NO_EVENTS"
            fi
            break
        fi

        # Exit 124: timeout
        if [ $EXIT_CODE -eq 124 ]; then
            if [ "$HAS_EVENTS" = true ]; then
                TIER1_STATUS="PARTIAL_SUCCESS"
            else
                TIER1_STATUS="TIMEOUT_NO_EVENTS"
            fi
            break
        fi

        # ── Check for ModuleNotFoundError (retryable) ──
        MISSING_MODULE=""
        if [ -f "$OUTPUT_DIR/stderr.log" ]; then
            MISSING_MODULE=$(grep -oP "No module named '\\K[^'.]+" "$OUTPUT_DIR/stderr.log" 2>/dev/null | head -1 || true)
        fi

        # Not an import error -- stop retrying
        if [ -z "$MISSING_MODULE" ]; then
            if [ "$HAS_EVENTS" = true ]; then
                TIER1_STATUS="PARTIAL_SUCCESS"
            else
                TIER1_STATUS="RUNTIME_ERROR"
            fi
            break
        fi

        # Duplicate module detection (already tried installing this one)
        if echo " $ATTEMPTED_MODULES " | grep -q " $MISSING_MODULE "; then
            log "  Module '$MISSING_MODULE' already attempted -- unresolvable"
            TIER1_STATUS="UNRESOLVABLE_IMPORT"
            break
        fi
        ATTEMPTED_MODULES="$ATTEMPTED_MODULES $MISSING_MODULE"

        # Skip if module is a local directory in the repo
        if [ -d "/tmp/repo/$MISSING_MODULE" ] || [ -d "/tmp/repo/src/$MISSING_MODULE" ]; then
            log "  Module '$MISSING_MODULE' is a local package -- adding to PYTHONPATH"
            if [ -d "/tmp/repo/$MISSING_MODULE" ]; then
                export PYTHONPATH="/tmp/repo:$PYTHONPATH"
            else
                export PYTHONPATH="/tmp/repo/src:$PYTHONPATH"
            fi
            continue
        fi

        # Resolve pip package name and install
        PIP_NAME=$(resolve_pip_name "$MISSING_MODULE")
        log "  Auto-installing: $MISSING_MODULE -> pip install $PIP_NAME"

        pip install "$PIP_NAME" --quiet 2>/dev/null
        if [ $? -ne 0 ]; then
            # Fallback: try raw module name if mapped name failed
            if [ "$PIP_NAME" != "$MISSING_MODULE" ]; then
                pip install "$MISSING_MODULE" --quiet 2>/dev/null
                if [ $? -ne 0 ]; then
                    log "  Failed to install $MISSING_MODULE -- unresolvable"
                    TIER1_STATUS="UNRESOLVABLE_IMPORT"
                    break
                fi
            else
                log "  Failed to install $PIP_NAME -- unresolvable"
                TIER1_STATUS="UNRESOLVABLE_IMPORT"
                break
            fi
        fi

        log "  Installed $PIP_NAME -- retrying ..."
    done

    # If we exhausted all retries without setting a status
    if [ -z "$TIER1_STATUS" ]; then
        TIER1_STATUS="MAX_RETRIES_EXCEEDED"
    fi
fi

log "  Tier 1 result: $TIER1_STATUS (exit_code=$EXIT_CODE)"


# ============================================================================
# STEP 7.5: TIER 1.5 -- IMPORT-AND-CALL FALLBACK
# ============================================================================

FINAL_STATUS="$TIER1_STATUS"
FINAL_EXIT_CODE="$EXIT_CODE"

# Determine if Tier 1 was a success (no need for fallback tiers)
TIER1_SUCCESS=false
case "$TIER1_STATUS" in
    SUCCESS|PARTIAL_SUCCESS) TIER1_SUCCESS=true ;;
esac

if [ "$TIER1_SUCCESS" = false ] && [ -f /app/synthetic_harness.py ]; then
    log "Step 7.5: Tier 1 failed ($TIER1_STATUS) -- trying Tier 1.5 (import-and-call, timeout=${TIER1_5_TIMEOUT}s) ..."

    # Clear any partial events from failed Tier 1
    rm -f "$EVENTS_FILE"

    export STRATUM_TIER="1.5"
    export STRATUM_TIMEOUT_SECONDS="$TIER1_5_TIMEOUT"

    timeout "$TIER1_5_TIMEOUT" python /app/synthetic_harness.py --tier1.5 /tmp/repo "$VLLM_HOST" "$OUTPUT_DIR" \
        > "$OUTPUT_DIR/tier1_5_stdout.log" \
        2> "$OUTPUT_DIR/tier1_5_stderr.log"
    T15_EXIT=$?

    T15_HAS_EVENTS=false
    if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
        T15_HAS_EVENTS=true
    fi

    if [ "$T15_HAS_EVENTS" = true ]; then
        FINAL_STATUS="TIER1_5_SUCCESS"
        FINAL_EXIT_CODE=$T15_EXIT
        TIER=2  # write_output only accepts int; use tier_detail.txt for 1.5
        echo "1.5" > "$OUTPUT_DIR/tier_detail.txt"
        log "  Tier 1.5 succeeded! (events captured with real topology)"
    else
        log "  Tier 1.5 failed (no events, exit=$T15_EXIT)"
    fi
fi


# ============================================================================
# STEP 8: TIER 2 -- SYNTHETIC HARNESS FALLBACK
# ============================================================================

# Only run Tier 2 if both Tier 1 and Tier 1.5 failed
NEED_TIER2=false
case "$FINAL_STATUS" in
    SUCCESS|PARTIAL_SUCCESS|TIER1_5_SUCCESS) NEED_TIER2=false ;;
    *) NEED_TIER2=true ;;
esac

if [ "$NEED_TIER2" = true ] && [ -f /app/synthetic_harness.py ]; then
    log "Step 8: Tier 1 + 1.5 failed -- invoking Tier 2 synthetic harness (timeout=${TIER2_TIMEOUT}s) ..."

    # Clear any partial events from failed Tier 1.5
    rm -f "$EVENTS_FILE"

    export STRATUM_TIER="2"
    export STRATUM_TIMEOUT_SECONDS="$TIER2_TIMEOUT"

    timeout "$TIER2_TIMEOUT" python /app/synthetic_harness.py /tmp/repo "$VLLM_HOST" "$OUTPUT_DIR" \
        > "$OUTPUT_DIR/tier2_stdout.log" \
        2> "$OUTPUT_DIR/tier2_stderr.log"
    T2_EXIT=$?

    T2_HAS_EVENTS=false
    if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
        T2_HAS_EVENTS=true
    fi

    if [ "$T2_HAS_EVENTS" = true ]; then
        FINAL_STATUS="TIER2_SUCCESS"
        FINAL_EXIT_CODE=$T2_EXIT
        TIER=2
        log "  Tier 2 succeeded! (events captured)"
    else
        FINAL_STATUS="TIER2_FAILED"
        FINAL_EXIT_CODE=$T2_EXIT
        TIER=0
        log "  Tier 2 also failed (no events, exit=$T2_EXIT)"
    fi
elif [ "$NEED_TIER2" = true ]; then
    if [ ! -f /app/synthetic_harness.py ]; then
        log "Step 8: Skipping Tier 2 (synthetic_harness.py not found)"
        TIER=0
    fi
else
    log "Step 8: Skipping Tier 2 (earlier tier succeeded)"
fi


# ============================================================================
# STEP 9: FINAL STATUS & GRAPH BUILDING
# ============================================================================

log "Step 9: Writing final output ..."
log "  Final status: $FINAL_STATUS (tier=$TIER, exit_code=$FINAL_EXIT_CODE)"

write_output "$FINAL_STATUS" "$FINAL_EXIT_CODE" "${ENTRY:-}" "$TIER"

# NOTE: graph_builder.py removed -- build_behavioral_records.py handles this post-hoc now.
# if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ] && [ -f /app/graph_builder.py ]; then
#     log "  Building behavioral graph from $(wc -l < "$EVENTS_FILE" | tr -d ' ') events ..."
#     python3 /app/graph_builder.py "$EVENTS_FILE" "$OUTPUT_DIR/graph.json" 2>/dev/null || true
# fi

EVENT_COUNT=0
if [ -f "$EVENTS_FILE" ]; then
    EVENT_COUNT=$(wc -l < "$EVENTS_FILE" 2>/dev/null || echo "0")
    EVENT_COUNT=$(echo "$EVENT_COUNT" | tr -d ' ')
fi

END_EPOCH=$(date +%s)
DURATION=$(( END_EPOCH - START_EPOCH ))

# Write per-run metadata
REPO_NAME=$(echo "$REPO_URL" | sed 's|.*/||;s|\.git$||')
REPO_FULL_NAME=$(echo "$REPO_URL" | sed 's|https://github.com/||;s|\.git$||')
FRAMEWORK_DETECTED="unknown"
if [ -f /tmp/repo/requirements.txt ]; then
    grep -qi crewai /tmp/repo/requirements.txt && FRAMEWORK_DETECTED="crewai"
    grep -qi langgraph /tmp/repo/requirements.txt && FRAMEWORK_DETECTED="langgraph"
    grep -qi autogen /tmp/repo/requirements.txt && FRAMEWORK_DETECTED="autogen"
fi
# Also check pyproject.toml dependencies if requirements.txt didn't yield a match
if [ "$FRAMEWORK_DETECTED" = "unknown" ] && [ -f /tmp/repo/pyproject.toml ]; then
    grep -qi crewai /tmp/repo/pyproject.toml && FRAMEWORK_DETECTED="crewai"
    grep -qi langgraph /tmp/repo/pyproject.toml && FRAMEWORK_DETECTED="langgraph"
    grep -qi autogen /tmp/repo/pyproject.toml && FRAMEWORK_DETECTED="autogen"
fi
# Check Python imports across repo source files as a final fallback
if [ "$FRAMEWORK_DETECTED" = "unknown" ]; then
    if grep -rqi 'import crewai\|from crewai' /tmp/repo --include='*.py' 2>/dev/null; then
        FRAMEWORK_DETECTED="crewai"
    elif grep -rqi 'import langgraph\|from langgraph' /tmp/repo --include='*.py' 2>/dev/null; then
        FRAMEWORK_DETECTED="langgraph"
    elif grep -rqi 'import autogen\|from autogen' /tmp/repo --include='*.py' 2>/dev/null; then
        FRAMEWORK_DETECTED="autogen"
    fi
fi

python3 -c "
import json, sys, os
from datetime import datetime, timezone
d = {
    'repo_url': sys.argv[1],
    'repo_full_name': sys.argv[2],
    'run_number': int(os.environ.get('STRATUM_RUN_NUMBER', '1')),
    'run_id': os.environ.get('STRATUM_RUN_ID', ''),
    'scan_id': os.environ.get('STRATUM_SCAN_ID', ''),
    'exit_code': int(sys.argv[3]),
    'status': sys.argv[4],
    'tier': int(sys.argv[5]),
    'entry_point': sys.argv[6],
    'framework_detected': sys.argv[7],
    'events_file': 'events_run_' + os.environ.get('STRATUM_RUN_NUMBER', '1') + '.jsonl',
    'event_count': int(sys.argv[8]),
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'timeout_seconds': int(sys.argv[9]),
    'model_name': os.environ.get('STRATUM_VLLM_MODEL', ''),
}
tier_detail_path = os.path.join(sys.argv[10], 'tier_detail.txt')
try:
    with open(tier_detail_path) as f:
        d['tier_detail'] = float(f.read().strip())
except (OSError, ValueError):
    d['tier_detail'] = float(d['tier'])
json.dump(d, open(os.path.join(sys.argv[10], f'run_metadata_{d[\"run_number\"]}.json'), 'w'), indent=2)
" "$REPO_URL" "$REPO_FULL_NAME" "$FINAL_EXIT_CODE" "$FINAL_STATUS" "$TIER" "${ENTRY:-}" "$FRAMEWORK_DETECTED" "$EVENT_COUNT" "$TIMEOUT" "$OUTPUT_DIR"

log "============================================"
log "  Repo:        $REPO_URL"
log "  Status:      $FINAL_STATUS"
log "  Tier:        $TIER"
log "  Entry:       ${ENTRY:-<none>}"
log "  Events:      $EVENT_COUNT"
log "  Duration:    ${DURATION}s"
log "  Run ID:      $RUN_ID"
log "  Framework:   $FRAMEWORK_DETECTED"
log "  Run Number:  $STRATUM_RUN_NUMBER"
log "============================================"

exit 0
