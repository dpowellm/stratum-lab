#!/usr/bin/env bash
# ============================================================================
# run_repo.sh — Stratum Lab Docker ENTRYPOINT
#
# Three-tier execution harness for behavioral scanning of AI agent repos.
#   Tier 1: Native execution (clone → install → run entry point)
#   Tier 2: Synthetic harness (extract agent defs → generate script → run)
#   Tier 3: Unrunnable (classify why and move on)
#
# Usage:  run_repo.sh <github_url> [timeout_seconds] [vllm_base_url]
#
# CRITICAL: No set -e ANYWHERE. The retry loop MUST survive Python failures.
# ============================================================================

GITHUB_URL="${1:-}"
if [ -z "$GITHUB_URL" ]; then
    echo "Usage: run_repo.sh <github_url> [timeout_seconds] [vllm_base_url]" >&2
    exit 1
fi
TIMEOUT="${2:-${STRATUM_TIMEOUT_SECONDS:-300}}"
VLLM_BASE_URL="${3:-${OPENAI_BASE_URL:-http://host.docker.internal:8000/v1}}"

# ── Fixed paths ──────────────────────────────────────────────────────────
REPO_DIR="/tmp/repo"
OUTPUT_DIR="/app/output"
EVENTS_FILE="$OUTPUT_DIR/stratum_events.jsonl"
RUNNER="/app/runner.py"
SYNTHETIC="/app/synthetic_harness.py"

# ── Per-repo env vars (Bug #11: set run_id and repo_id) ─────────────────
RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:16])' 2>/dev/null || echo "run_$$")"
export STRATUM_RUN_ID="$RUN_ID"
export STRATUM_REPO_ID="$GITHUB_URL"
export STRATUM_EVENTS_FILE="$EVENTS_FILE"
export STRATUM_VLLM_MODEL="${STRATUM_VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
export OPENAI_BASE_URL="$VLLM_BASE_URL"
export OPENAI_API_BASE="$VLLM_BASE_URL"

mkdir -p "$OUTPUT_DIR"

START_EPOCH=$(date +%s)
TIER=1

log() { echo "[stratum] $(date +%H:%M:%S) $*" >&2; }

# ── Output writers ───────────────────────────────────────────────────────

write_output() {
    local status="$1"
    local exit_code="${2:-0}"
    local entry_point="${3:-}"
    local tier="${4:-1}"

    echo "$status"     > "$OUTPUT_DIR/status.txt"
    echo "$exit_code"  > "$OUTPUT_DIR/exit_code.txt"
    echo "$entry_point"> "$OUTPUT_DIR/entry_point.txt"
    echo "$tier"       > "$OUTPUT_DIR/tier.txt"

    # Also write a machine-readable status.json for aggregator
    local event_count=0
    if [ -f "$EVENTS_FILE" ]; then
        event_count=$(wc -l < "$EVENTS_FILE" 2>/dev/null || echo "0")
        event_count=$(echo "$event_count" | tr -d ' ')
    fi
    local end_epoch
    end_epoch=$(date +%s)
    local duration=$(( end_epoch - START_EPOCH ))

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
    'vllm_model': os.environ.get('STRATUM_VLLM_MODEL', ''),
    'error_log_tail': '',
}
# Read stderr tail
try:
    with open('/tmp/run_stderr.log') as f:
        lines = f.readlines()
        d['error_log_tail'] = ''.join(lines[-20:])[:2000]
except: pass
with open(sys.argv[8], 'w') as f:
    json.dump(d, f, indent=2)
" "$GITHUB_URL" "$status" "$exit_code" "$entry_point" "$tier" \
  "$event_count" "$duration" "$OUTPUT_DIR/status.json" 2>/dev/null \
  || echo "{\"repo\":\"$GITHUB_URL\",\"status\":\"$status\",\"tier\":$tier}" > "$OUTPUT_DIR/status.json"
}

# ── PIP_NAME_MAP ─────────────────────────────────────────────────────────
declare -A PIP_NAME_MAP=(
    [cv2]=opencv-python  [cv]=opencv-python  [yaml]=pyyaml
    [dotenv]=python-dotenv  [sklearn]=scikit-learn  [PIL]=Pillow
    [bs4]=beautifulsoup4  [attr]=attrs  [gi]=PyGObject
    [Crypto]=pycryptodome  [dateutil]=python-dateutil
    [jose]=python-jose  [magic]=python-magic  [wx]=wxPython
    [usb]=pyusb  [serial]=pyserial  [skimage]=scikit-image
    [Bio]=biopython  [docx]=python-docx  [pptx]=python-pptx
    [jwt]=PyJWT  [Levenshtein]=python-Levenshtein
    [faiss]=faiss-cpu  [chromadb]=chromadb
    [pinecone]=pinecone-client  [weaviate]=weaviate-client
    [qdrant_client]=qdrant-client
    [langchain_community]=langchain-community
    [langchain_openai]=langchain-openai
    [langchain_anthropic]=langchain-anthropic
    [langchain_experimental]=langchain-experimental
    [langchain_core]=langchain-core
    [langchain_text_splitters]=langchain-text-splitters
    [tavily]=tavily-python  [exa_py]=exa-py
    [duckduckgo_search]=duckduckgo-search
    [newspaper]=newspaper3k  [trafilatura]=trafilatura
    [unstructured]=unstructured  [litellm]=litellm
    [tiktoken]=tiktoken  [cohere]=cohere  [together]=together
    [xlrd]=xlrd  [openpyxl]=openpyxl  [lxml]=lxml
    [markdown]=markdown  [rich]=rich  [tqdm]=tqdm
    [click]=click  [typer]=typer  [fire]=fire
    [colorama]=colorama  [termcolor]=termcolor
    [pydantic_settings]=pydantic-settings
    [crewai_tools]=crewai-tools
)

resolve_pip_name() {
    local module="$1"
    # 1. Explicit map
    if [ -n "${PIP_NAME_MAP[$module]+x}" ]; then
        echo "${PIP_NAME_MAP[$module]}"; return
    fi
    # 2. Replace underscores with hyphens
    local hyphenated="${module//_/-}"
    if [ "$hyphenated" != "$module" ]; then
        echo "$hyphenated"; return
    fi
    # 3. As-is
    echo "$module"
}

# ── Phase 1: Clone ──────────────────────────────────────────────────────

log "Cloning $GITHUB_URL ..."
if ! git clone --depth 1 "$GITHUB_URL" "$REPO_DIR" 2>/tmp/clone.log; then
    log "FAIL: Clone failed"
    cp /tmp/clone.log "$OUTPUT_DIR/stderr.log" 2>/dev/null || true
    write_output "CLONE_FAILED" 1 "" 1
    exit 0
fi
cd "$REPO_DIR"

# ── Phase 2: .env handling (early — before any imports) ─────────────────

for tpl in .env.example .env.template .env.sample; do
    if [ -f "$tpl" ] && [ ! -f .env ]; then
        log "Creating .env from $tpl"
        cp "$tpl" .env
        sed -i 's/=$/=stratum-placeholder/' .env 2>/dev/null || true
        sed -i 's/=""$/="stratum-placeholder"/' .env 2>/dev/null || true
        sed -i "s/=''$/='stratum-placeholder'/" .env 2>/dev/null || true
        break
    fi
done
# Also check subdirectories for .env.example
find "$REPO_DIR" -maxdepth 2 -name '.env.example' -print0 2>/dev/null | while IFS= read -r -d '' envex; do
    target="$(dirname "$envex")/.env"
    if [ ! -f "$target" ]; then
        cp "$envex" "$target" 2>/dev/null || true
        sed -i 's/=$/=stratum-placeholder/' "$target" 2>/dev/null || true
    fi
done

# ── Phase 3: Entry point detection ──────────────────────────────────────

declare -A CANDIDATES  # path -> score

# Strategy A: __main__ + framework call (score 10 — BEST signal)
while IFS= read -r pyfile; do
    if grep -q '__name__.*__main__' "$pyfile" 2>/dev/null; then
        if grep -qE '\.kickoff\s*\(|\.invoke\s*\(|initiate_chat\s*\(|\.run\s*\(' "$pyfile" 2>/dev/null; then
            cur="${CANDIDATES[$pyfile]:-0}"; [ 10 -gt "$cur" ] && CANDIDATES["$pyfile"]=10
        elif grep -qE '(import|from)\s+(crewai|langchain|langgraph|autogen|openai)\b' "$pyfile" 2>/dev/null; then
            cur="${CANDIDATES[$pyfile]:-0}"; [ 8 -gt "$cur" ] && CANDIDATES["$pyfile"]=8
        fi
    fi
done < <(find . -maxdepth 4 -name '*.py' \
    -not -path '*/\.*' -not -path '*/node_modules/*' \
    -not -path '*/venv/*' -not -path '*/.venv/*' \
    -not -path '*/test*' -not -path '*/__pycache__/*' \
    -not -name 'setup.py' -not -name 'conftest.py' 2>/dev/null || true)

# Strategy B: README parsing (score 9)
for readme in README.md README.rst readme.md; do
    if [ -f "$readme" ]; then
        while IFS= read -r match; do
            [ -f "$match" ] || continue
            cur="${CANDIDATES[$match]:-0}"; [ 9 -gt "$cur" ] && CANDIDATES["$match"]=9
        done < <(grep -oP 'python3?\s+\K\S+\.py' "$readme" 2>/dev/null | sort -u || true)
        break
    fi
done

# Strategy C: pyproject.toml scripts (score 8)
if [ -f pyproject.toml ]; then
    while IFS= read -r script_ref; do
        mod_path=$(echo "$script_ref" | sed 's/:.*//' | tr '.' '/')
        for c in "${mod_path}.py" "${mod_path}/__main__.py"; do
            [ -f "$c" ] || continue
            cur="${CANDIDATES[$c]:-0}"; [ 8 -gt "$cur" ] && CANDIDATES["$c"]=8
        done
    done < <(grep -A5 '\[project\.scripts\]\|\[tool\.poetry\.scripts\]' pyproject.toml 2>/dev/null \
             | grep -oP '=\s*"\K[^"]+' 2>/dev/null || true)
fi

# Strategy D: Known filenames at root (score 6)
for name in main.py app.py run.py agent.py crew.py start.py cli.py; do
    if [ -f "$name" ]; then
        # SKIP server-based files
        if grep -qE 'uvicorn\.|flask\.|FastAPI\(|Starlette\(|gunicorn' "$name" 2>/dev/null; then
            continue
        fi
        cur="${CANDIDATES[$name]:-0}"; [ 6 -gt "$cur" ] && CANDIDATES["$name"]=6
    fi
done

# Strategy E: Known filenames in subdirs (score 4)
for subdir in src app agents crew; do
    [ -d "$subdir" ] || continue
    for name in main.py app.py run.py agent.py crew.py start.py __main__.py; do
        [ -f "$subdir/$name" ] || continue
        if grep -qE 'uvicorn\.|flask\.|FastAPI\(|Starlette\(' "$subdir/$name" 2>/dev/null; then
            continue
        fi
        cur="${CANDIDATES[$subdir/$name]:-0}"; [ 4 -gt "$cur" ] && CANDIDATES["$subdir/$name"]=4
    done
done

# Strategy F: Any __main__ guard (score 2 — last resort)
while IFS= read -r pyfile; do
    if grep -q '__name__.*__main__' "$pyfile" 2>/dev/null; then
        if grep -qE 'uvicorn\.|flask\.|FastAPI\(' "$pyfile" 2>/dev/null; then
            continue  # Skip server files
        fi
        cur="${CANDIDATES[$pyfile]:-0}"; [ 2 -gt "$cur" ] && CANDIDATES["$pyfile"]=2
    fi
done < <(find . -maxdepth 3 -name '*.py' \
    -not -path '*/\.*' -not -path '*/venv/*' -not -path '*/.venv/*' \
    -not -path '*/test*' -not -path '*/__pycache__/*' \
    -not -name 'setup.py' -not -name 'conftest.py' 2>/dev/null || true)

# Pick best candidate
ENTRY_POINT=""
BEST_SCORE=0
for path in "${!CANDIDATES[@]}"; do
    score=${CANDIDATES[$path]}
    if [ "$score" -gt "$BEST_SCORE" ]; then
        BEST_SCORE=$score
        ENTRY_POINT="$path"
    fi
done
ENTRY_POINT="${ENTRY_POINT#./}"

TIER1_FAILED=""
if [ -z "$ENTRY_POINT" ]; then
    log "No entry point found — will try Tier 2"
    TIER1_FAILED="no_entry_point"
else
    log "Entry point: $ENTRY_POINT (score=$BEST_SCORE)"
fi

# ── Phase 4: Check for server-based repo ─────────────────────────────────

IS_SERVER=false
if [ -n "$ENTRY_POINT" ] && [ -f "$ENTRY_POINT" ]; then
    if grep -qE '(uvicorn\.run|app\.run\(|flask\.Flask|FastAPI\(\)|Starlette\(|gunicorn)' "$ENTRY_POINT" 2>/dev/null; then
        IS_SERVER=true
        log "Server-based repo detected — skipping native execution"
        TIER1_FAILED="server"
    fi
fi

# ── Phase 5: Install dependencies ───────────────────────────────────────

install_deps() {
    # Root requirements.txt
    if [ -f requirements.txt ]; then
        log "Installing requirements.txt ..."
        pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5 | while read -r l; do log "  pip: $l"; done
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log "WARNING: requirements.txt install had errors, trying line-by-line"
            while IFS= read -r req; do
                req=$(echo "$req" | sed 's/#.*//' | xargs)
                [ -z "$req" ] && continue
                [[ "$req" == -* ]] && continue
                pip install --no-cache-dir "$req" 2>/dev/null || log "  SKIP: $req"
            done < requirements.txt
        fi
    fi

    # Nested requirements files
    while IFS= read -r reqfile; do
        [ "$reqfile" = "./requirements.txt" ] && continue
        log "Installing $reqfile ..."
        pip install --no-cache-dir -r "$reqfile" 2>/dev/null || true
    done < <(find . -maxdepth 3 -name 'requirements*.txt' \
        -not -path '*/venv/*' -not -path '*/.venv/*' 2>/dev/null || true)

    # pyproject.toml / setup.py fallback
    if [ ! -f requirements.txt ]; then
        if [ -f pyproject.toml ]; then
            log "Installing from pyproject.toml ..."
            pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir . 2>/dev/null || true
        elif [ -f setup.py ]; then
            log "Installing from setup.py ..."
            pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir . 2>/dev/null || true
        fi
    fi
}

if [ "$IS_SERVER" = false ] && [ -n "$ENTRY_POINT" ]; then
    install_deps
fi

# ── Phase 6: PYTHONPATH (Bug #4: NEVER add entry point directory) ────────

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
if [ -d "$REPO_DIR/src" ]; then
    export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"
fi

# ── Phase 7: Tier 1 — Native execution with retry loop ──────────────────

MAX_ATTEMPTS=5
ATTEMPT=0
ATTEMPTED_MODULES=""
RUN_EXIT_CODE=0
TIER1_STATUS=""

if [ -n "$TIER1_FAILED" ]; then
    if [ "$TIER1_FAILED" = "server" ]; then
        TIER1_STATUS="SERVER_BASED"
    else
        TIER1_STATUS="NO_ENTRY_POINT"
    fi
else
    for (( ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++ )); do
        log "Tier 1 attempt $ATTEMPT/$MAX_ATTEMPTS"

        # NO set -e. Capture exit code directly.
        timeout "$TIMEOUT" python "$RUNNER" "$ENTRY_POINT" \
            >"$OUTPUT_DIR/stdout.log" 2>/tmp/run_stderr.log
        RUN_EXIT_CODE=$?

        # Append stderr to output
        cp /tmp/run_stderr.log "$OUTPUT_DIR/stderr.log" 2>/dev/null || true

        # Check for ModuleNotFoundError
        MISSING_MODULE=""
        if [ $RUN_EXIT_CODE -ne 0 ] && [ -f /tmp/run_stderr.log ]; then
            MISSING_MODULE=$(grep -oP "No module named '\\K[^'.]+" /tmp/run_stderr.log 2>/dev/null | head -1 || true)
        fi

        # If success or not an import error, stop retrying
        if [ -z "$MISSING_MODULE" ]; then
            break
        fi

        # Duplicate detection
        if echo " $ATTEMPTED_MODULES " | grep -q " $MISSING_MODULE "; then
            log "Module '$MISSING_MODULE' already attempted — unresolvable"
            TIER1_STATUS="UNRESOLVABLE_IMPORT"
            TIER1_FAILED="import:$MISSING_MODULE"
            break
        fi
        ATTEMPTED_MODULES="$ATTEMPTED_MODULES $MISSING_MODULE"

        PIP_NAME=$(resolve_pip_name "$MISSING_MODULE")
        log "Auto-installing: $MISSING_MODULE → pip install $PIP_NAME"

        pip install --no-cache-dir "$PIP_NAME" 2>/dev/null
        if [ $? -ne 0 ]; then
            # Try raw module name if mapped name failed
            if [ "$PIP_NAME" != "$MISSING_MODULE" ]; then
                pip install --no-cache-dir "$MISSING_MODULE" 2>/dev/null
                if [ $? -ne 0 ]; then
                    TIER1_STATUS="UNRESOLVABLE_IMPORT"
                    TIER1_FAILED="import:$MISSING_MODULE"
                    break
                fi
            else
                TIER1_STATUS="UNRESOLVABLE_IMPORT"
                TIER1_FAILED="import:$MISSING_MODULE"
                break
            fi
        fi
    done
fi

# ── Phase 8: Classify Tier 1 ────────────────────────────────────────────

if [ -z "$TIER1_STATUS" ]; then
    if [ $RUN_EXIT_CODE -eq 0 ]; then
        if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
            TIER1_STATUS="SUCCESS"
        else
            TIER1_STATUS="NO_EVENTS"
            TIER1_FAILED="no_events"
        fi
    elif [ $RUN_EXIT_CODE -eq 124 ]; then
        if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
            TIER1_STATUS="PARTIAL_SUCCESS"
        else
            TIER1_STATUS="TIMEOUT_NO_EVENTS"
            TIER1_FAILED="timeout"
        fi
    else
        TIER1_STATUS="RUNTIME_ERROR"
        TIER1_FAILED="runtime:$RUN_EXIT_CODE"
    fi
fi

log "Tier 1 result: $TIER1_STATUS"

# ── Phase 9: Tier 2 — Synthetic harness fallback ────────────────────────

FINAL_STATUS="$TIER1_STATUS"
TIER=1

if [ -n "$TIER1_FAILED" ] && [ -f "$SYNTHETIC" ]; then
    log "Tier 1 failed ($TIER1_FAILED) — invoking Tier 2 synthetic harness"

    # Clear partial events from failed Tier 1
    rm -f "$EVENTS_FILE"

    timeout "$TIMEOUT" python "$SYNTHETIC" "$REPO_DIR" "$EVENTS_FILE" \
        >"$OUTPUT_DIR/tier2_stdout.log" 2>"$OUTPUT_DIR/tier2_stderr.log"
    T2_EXIT=$?

    if [ $T2_EXIT -eq 0 ] && [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
        FINAL_STATUS="TIER2_SUCCESS"
        TIER=2
        log "Tier 2 succeeded!"
    elif [ $T2_EXIT -eq 124 ] && [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
        FINAL_STATUS="TIER2_PARTIAL"
        TIER=2
        log "Tier 2 partial (timeout with events)"
    else
        log "Tier 2 also failed (exit=$T2_EXIT)"
        # Keep original Tier 1 status
    fi
fi

# ── Phase 10: Build graph (if we have events) ───────────────────────────

if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ] && [ -f /app/graph_builder.py ]; then
    python3 /app/graph_builder.py "$EVENTS_FILE" "$OUTPUT_DIR/graph.json" 2>/dev/null || true
fi

# ── Phase 11: Write final output ────────────────────────────────────────

log "Final: $FINAL_STATUS (tier=$TIER)"
write_output "$FINAL_STATUS" "$RUN_EXIT_CODE" "$ENTRY_POINT" "$TIER"
exit 0
