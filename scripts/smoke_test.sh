#!/bin/bash
# smoke_test.sh — validates entire pipeline before 1k scan
# Usage: ./smoke_test.sh [vllm_host]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCHER_ROOT="$SCRIPT_DIR/.."
VLLM_HOST=${1:-"http://localhost:8000"}
VLLM_MODEL="${STRATUM_VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PASS=0; FAIL=0

pass_test() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail_test() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

# Common env vars for test subprocesses
export_env() {
    local dir="$1" run_id="$2" repo_id="$3" framework="${4:-unknown}"
    export STRATUM_EVENTS_FILE="$dir/stratum_events.jsonl"
    export STRATUM_RUN_ID="$run_id" STRATUM_REPO_ID="$repo_id"
    export STRATUM_FRAMEWORK="$framework" STRATUM_VLLM_MODEL="$VLLM_MODEL"
    export OPENAI_BASE_URL="${VLLM_HOST}/v1" OPENAI_API_KEY="sk-stratum-local"
    export PYTHONPATH="$PATCHER_ROOT:${PYTHONPATH:-}"
}

# --- Test 0: vLLM Reachability -----------------------------------------------
echo "--- Test 0: vLLM Reachability ---"
if curl -s --max-time 5 "${VLLM_HOST}/v1/models" > /dev/null 2>&1; then
    pass_test "vLLM reachable"
else
    fail_test "vLLM unreachable at $VLLM_HOST"
    echo "Cannot continue without vLLM. Exiting."
    exit 1
fi

# --- Test 1: Raw OpenAI + Patcher --------------------------------------------
echo ""
echo "--- Test 1: Raw OpenAI + Patcher ---"
T1=$(mktemp -d); export_env "$T1" smoke-1 smoke-openai
cat > "$T1/run.py" <<'PYEOF'
import os, sys, stratum_patcher.openai_patch
from openai import OpenAI
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
r = client.chat.completions.create(
    model=os.environ["STRATUM_VLLM_MODEL"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}], max_tokens=50)
print("OK:", r.choices[0].message.content[:80])
PYEOF
python "$T1/run.py" > "$T1/out" 2> "$T1/err"
if [ $? -ne 0 ]; then
    fail_test "OpenAI script failed"; head -3 "$T1/err" 2>/dev/null
elif [ ! -f "$T1/stratum_events.jsonl" ]; then
    fail_test "No events file produced"
else
    CNT=$(wc -l < "$T1/stratum_events.jsonl" | tr -d ' ')
    SCHEMA=$(python -c "
import json,sys
with open(sys.argv[1]) as f:
    ok=all(all(k in json.loads(l) for k in ('event_id','timestamp_ns','source_node','event_type')) for l in f if l.strip())
print('ok' if ok else 'bad')
" "$T1/stratum_events.jsonl" 2>&1)
    if [ "$CNT" -ge 2 ] && [ "$SCHEMA" = "ok" ]; then
        pass_test "OpenAI + Patcher ($CNT events, schema valid)"
    else
        fail_test "OpenAI events: count=$CNT schema=$SCHEMA (need >=2, ok)"
    fi
fi
rm -rf "$T1"

# --- Test 2: CrewAI 2-Agent Crew ---------------------------------------------
echo ""
echo "--- Test 2: CrewAI 2-Agent Crew ---"
T2=$(mktemp -d); export_env "$T2" smoke-2 smoke-crewai crewai
cat > "$T2/run.py" <<'PYEOF'
import os, sys, stratum_patcher.openai_patch, stratum_patcher.crewai_patch
from crewai import Agent, Task, Crew, LLM
url = os.environ["OPENAI_BASE_URL"]
model = os.environ["STRATUM_VLLM_MODEL"]
key = os.environ["OPENAI_API_KEY"]
llm = LLM(model=f"openai/{model}", base_url=url, api_key=key)
researcher = Agent(role="Researcher", goal="Find key facts about a topic",
    backstory="An experienced researcher", llm=llm, verbose=False, allow_delegation=False)
writer = Agent(role="Writer", goal="Write a clear summary from research findings",
    backstory="A skilled technical writer", llm=llm, verbose=False, allow_delegation=False)
t1 = Task(description="Research observability basics. Keep it brief.",
    expected_output="A short list of key facts", agent=researcher)
t2 = Task(description="Write a one-paragraph summary of the research.",
    expected_output="A concise paragraph", agent=writer)
crew = Crew(agents=[researcher, writer], tasks=[t1, t2], verbose=False)
crew.kickoff()
PYEOF
python "$T2/run.py" > "$T2/out" 2> "$T2/err"
if [ $? -ne 0 ]; then
    fail_test "CrewAI script failed"; head -3 "$T2/err" 2>/dev/null
elif [ ! -f "$T2/stratum_events.jsonl" ]; then
    fail_test "No CrewAI events file produced"
else
    T2_CHK=$(python -c "
import json,sys
types=set(); has_hash=False
for l in open(sys.argv[1]):
    ev=json.loads(l.strip()); types.add(ev.get('event_type',''))
    if ev.get('event_type')=='agent.task_end' and (ev.get('payload') or {}).get('output_hash'):
        has_hash=True
missing={'agent.task_start','agent.task_end','execution.start','execution.end'}-types
if missing: print(f'missing:{missing}')
elif not has_hash: print('no_output_hash')
else: print('ok')
" "$T2/stratum_events.jsonl" 2>&1)
    if [ "$T2_CHK" = "ok" ]; then
        CNT=$(wc -l < "$T2/stratum_events.jsonl" | tr -d ' ')
        pass_test "CrewAI 2-agent crew ($CNT events, all event types present)"
    else
        fail_test "CrewAI events: $T2_CHK"
    fi
fi
# Preserve events for Tests 3-4
T2_SAVED=""; if [ -f "$T2/stratum_events.jsonl" ]; then T2_SAVED=$(mktemp); cp "$T2/stratum_events.jsonl" "$T2_SAVED"; fi
rm -rf "$T2"

# --- Test 3: Graph Builder ---------------------------------------------------
echo ""
echo "--- Test 3: Graph Builder ---"
T3=$(mktemp -d)
if [ -n "$T2_SAVED" ] && [ -f "$T2_SAVED" ]; then
    python "$SCRIPT_DIR/graph_builder.py" "$T2_SAVED" "$T3/graph.json" > "$T3/out" 2> "$T3/err"
    if [ $? -ne 0 ]; then
        fail_test "graph_builder.py failed (exit $?)"; head -3 "$T3/err" 2>/dev/null
    elif [ ! -f "$T3/graph.json" ]; then
        fail_test "graph_builder.py produced no output"
    else
        T3_CHK=$(python -c "
import json,sys; g=json.load(open(sys.argv[1]))
missing=[k for k in ('nodes','edges','content_flow','risk_indicators','topology_type') if k not in g]
print(f'missing:{missing}' if missing else f'ok:nodes={len(g[\"nodes\"])},topo={g[\"topology_type\"]}')
" "$T3/graph.json" 2>&1)
        if echo "$T3_CHK" | grep -q "^ok:"; then pass_test "Graph builder ($T3_CHK)"
        else fail_test "Graph builder: $T3_CHK"; fi
    fi
else
    fail_test "Graph builder skipped (no events from Test 2)"
fi
T3_SAVED=""; if [ -f "$T3/graph.json" ]; then T3_SAVED=$(mktemp); cp "$T3/graph.json" "$T3_SAVED"; fi
rm -rf "$T3"

# --- Test 4: Risk Analyzer ---------------------------------------------------
echo ""
echo "--- Test 4: Risk Analyzer ---"
T4=$(mktemp -d)
if [ -f "$SCRIPT_DIR/risk_analyzer.py" ]; then
    if [ -n "$T3_SAVED" ] && [ -f "$T3_SAVED" ]; then
        python "$SCRIPT_DIR/risk_analyzer.py" "$T3_SAVED" "$T4/risk.json" > "$T4/out" 2> "$T4/err"
        if [ $? -ne 0 ]; then
            fail_test "risk_analyzer.py failed"; head -3 "$T4/err" 2>/dev/null
        elif [ ! -f "$T4/risk.json" ]; then
            fail_test "risk_analyzer.py produced no output"
        else
            T4_CHK=$(python -c "
import json,sys; r=json.load(open(sys.argv[1]))
missing=[k for k in ('risk_scores','topology_type','findings') if k not in r]
print(f'missing:{missing}' if missing else f'ok:findings={len(r[\"findings\"])},topo={r[\"topology_type\"]}')
" "$T4/risk.json" 2>&1)
            if echo "$T4_CHK" | grep -q "^ok:"; then pass_test "Risk analyzer ($T4_CHK)"
            else fail_test "Risk analyzer: $T4_CHK"; fi
        fi
    else
        fail_test "Risk analyzer skipped (no graph from Test 3)"
    fi
else
    fail_test "risk_analyzer.py not found (not yet implemented)"
fi
rm -rf "$T4"
[ -n "$T2_SAVED" ] && rm -f "$T2_SAVED"
[ -n "$T3_SAVED" ] && rm -f "$T3_SAVED"

# --- Test 5: Tier 2 Synthetic Harness ----------------------------------------
echo ""
echo "--- Test 5: Tier 2 Synthetic Harness ---"
T5=$(mktemp -d); mkdir -p "$T5/repo" "$T5/output"
export_env "$T5/output" smoke-5 smoke-synthetic synthetic
cat > "$T5/repo/main.py" <<'PYEOF'
from crewai import Agent, Task, Crew
researcher = Agent(role="Researcher", goal="Research a given topic thoroughly",
    backstory="An experienced research analyst")
task = Task(description="Summarize recent trends in AI agents",
    expected_output="A brief summary of AI agent trends", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
PYEOF
python "$SCRIPT_DIR/synthetic_harness.py" "$T5/repo" "$VLLM_HOST" "$T5/output" > "$T5/out" 2> "$T5/err"
if [ $? -ne 0 ]; then
    fail_test "Synthetic harness failed"; head -3 "$T5/err" 2>/dev/null
elif [ -f "$T5/output/stratum_events.jsonl" ]; then
    CNT=$(wc -l < "$T5/output/stratum_events.jsonl" | tr -d ' ')
    if [ "$CNT" -ge 1 ]; then pass_test "Synthetic harness ($CNT events)"
    else fail_test "Synthetic harness: empty events file"; fi
else
    fail_test "Synthetic harness: no events file"
fi
rm -rf "$T5"

# --- Test 6: Output Classifier -----------------------------------------------
echo ""
echo "--- Test 6: Output Classifier ---"
T6_CHK=$(python -c "
import subprocess, json, sys, os
script = os.path.join('$SCRIPT_DIR', 'output_classifier.py')
tests = [
    ('The GDP of France was 2.78 trillion USD in 2023. Paris is the capital.', 'str', '100', None),
    ('I cannot provide that information as it is outside my capabilities.', 'str', '80', 'refusal'),
    ('You should consider using Python. I recommend starting with Flask.', 'str', '80', 'recommendation'),
    ('x', 'str', '1', 'error_empty'),
]
ok = True; results = []
valid = {'factual','recommendation','speculative','refusal','structured','generative','delegative','error_empty'}
for text, otype, size, expected in tests:
    r = subprocess.run([sys.executable, script, '--text', text, '--type', otype, '--size', size],
                       capture_output=True, text=True, timeout=10)
    cls = json.loads(r.stdout).get('primary', '')
    if cls not in valid: results.append(f'INVALID:{cls}'); ok = False
    elif expected and cls != expected: results.append(f'WRONG:{cls}!={expected}'); ok = False
    else: results.append(cls)
print(('ok:' if ok else 'fail:') + ';'.join(results))
" 2>&1)
if echo "$T6_CHK" | grep -q "^ok:"; then pass_test "Output classifier ($T6_CHK)"
else fail_test "Output classifier: $T6_CHK"; fi

# --- Final Summary -----------------------------------------------------------
echo ""
echo "==========================="
echo "PASS: $PASS  FAIL: $FAIL"
if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED — ready for 1k scan"
else
    echo "FAILURES DETECTED — fix before scanning"
    exit 1
fi
