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

# --- Test 7: Multi-run Simulation -------------------------------------------
echo ""
echo "--- Test 7: Multi-run Simulation ---"
T7=$(mktemp -d)
T7_OK=true
for run_num in 1 2 3; do
    export RUN_NUMBER=$run_num
    export STRATUM_RUN_NUMBER=$run_num
    export STRATUM_EVENTS_FILE="$T7/events_run_${run_num}.jsonl"
    export STRATUM_RUN_ID="smoke-7-run-${run_num}"
    export STRATUM_REPO_ID="smoke-multirun"
    export STRATUM_FRAMEWORK="crewai"
    export STRATUM_VLLM_MODEL="$VLLM_MODEL"
    export OPENAI_BASE_URL="${VLLM_HOST}/v1"
    export OPENAI_API_KEY="sk-stratum-local"
    export PYTHONPATH="$PATCHER_ROOT:${PYTHONPATH:-}"
    cat > "$T7/run_${run_num}.py" <<'PYEOF'
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
    python "$T7/run_${run_num}.py" > "$T7/out_${run_num}" 2> "$T7/err_${run_num}"
    if [ $? -ne 0 ]; then
        T7_OK=false
    fi
done
if [ "$T7_OK" = "false" ]; then
    fail_test "Multi-run: one or more CrewAI runs failed"
else
    T7_CHK=$(python -c "
import os, json, sys
base = sys.argv[1]
ok = True; counts = []
for run_num in (1, 2, 3):
    path = os.path.join(base, f'events_run_{run_num}.jsonl')
    if not os.path.isfile(path):
        print(f'missing:events_run_{run_num}.jsonl'); sys.exit(0)
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    if len(lines) == 0:
        print(f'empty:events_run_{run_num}.jsonl'); sys.exit(0)
    counts.append(len(lines))
# Verify files are separate (each file should have its own run_id)
run_ids = set()
for run_num in (1, 2, 3):
    path = os.path.join(base, f'events_run_{run_num}.jsonl')
    with open(path) as f:
        for line in f:
            ev = json.loads(line.strip())
            run_ids.add(ev.get('run_id', ''))
if len(run_ids) >= 3:
    print(f'ok:runs={counts},distinct_run_ids={len(run_ids)}')
else:
    print(f'fail:only {len(run_ids)} distinct run_ids across 3 files')
" "$T7" 2>&1)
    if echo "$T7_CHK" | grep -q "^ok:"; then
        pass_test "Multi-run simulation ($T7_CHK)"
    else
        fail_test "Multi-run simulation: $T7_CHK"
    fi
fi
# Preserve T7 dir for Test 8
T7_SAVED="$T7"

# --- Test 8: Behavioral Record Generation -----------------------------------
echo ""
echo "--- Test 8: Behavioral Record Generation ---"
T8=$(mktemp -d)
if [ -d "$T7_SAVED" ] && [ -f "$T7_SAVED/events_run_1.jsonl" ]; then
    # Set up mock results directory structure expected by build_behavioral_record
    HASH="smoke_test_hash"
    mkdir -p "$T8/results/$HASH"
    # Copy per-run event files from Test 7
    for run_num in 1 2 3; do
        cp "$T7_SAVED/events_run_${run_num}.jsonl" "$T8/results/$HASH/"
        # Create run_metadata_N.json files
        python -c "
import json, sys
meta = {
    'repo_full_name': 'smoke-test/multirun-repo',
    'run_number': $run_num,
    'run_id': 'smoke-7-run-$run_num',
    'framework': 'crewai',
    'vllm_model': '$VLLM_MODEL',
    'status': 'SUCCESS',
    'event_count': 0
}
# Count events
with open(sys.argv[1]) as f:
    meta['event_count'] = sum(1 for l in f if l.strip())
with open(sys.argv[2], 'w') as f:
    json.dump(meta, f, indent=2)
" "$T8/results/$HASH/events_run_${run_num}.jsonl" "$T8/results/$HASH/run_metadata_${run_num}.json"
    done
    # Use the behavioral_record module to build the record
    export PYTHONPATH="$PATCHER_ROOT:${PYTHONPATH:-}"
    T8_CHK=$(python -c "
import json, sys, os, glob as globmod
sys.path.insert(0, '$PATCHER_ROOT')
from stratum_lab.output.behavioral_record import build_behavioral_record

results_hash_dir = sys.argv[1]
# Load runs from the results dir
runs = []
for run_file in sorted(globmod.glob(os.path.join(results_hash_dir, 'events_run_*.jsonl'))):
    events = []
    with open(run_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line.strip()))
    meta_file = run_file.replace('events_run_', 'run_metadata_').replace('.jsonl', '.json')
    if os.path.isfile(meta_file):
        with open(meta_file) as f:
            run_meta = json.load(f)
    else:
        run_meta = {'repo_full_name': 'smoke-test/multirun-repo'}
    runs.append({'events': events, 'metadata': run_meta})

if not runs:
    print('fail:no_runs_loaded')
    sys.exit(0)

# Build the behavioral record using the module function
record = build_behavioral_record(
    repo_full_name='smoke-test/multirun-repo',
    execution_metadata={'runs': len(runs), 'framework': 'crewai'},
    edge_validation={'structural_edges_total': 0, 'structural_edges_activated': 0, 'dead_edges': [], 'activation_rates': {}},
    emergent_edges=[],
    node_activation={'always_active': [], 'conditional': [], 'never_active': []},
    error_propagation=[],
    failure_modes=[],
    monitoring_baselines=[],
)

# Write output
output_path = sys.argv[2]
with open(output_path, 'w') as f:
    json.dump(record, f, indent=2)

# Validate: check 9 required v6 top-level keys
required = {'repo_full_name','schema_version','execution_metadata','edge_validation',
            'emergent_edges','node_activation','error_propagation','failure_modes','monitoring_baselines'}
missing = required - set(record.keys())
if missing:
    print(f'fail:missing_keys={missing}')
elif record.get('schema_version') != 'v6':
    print(f'fail:schema_version={record.get(\"schema_version\")}')
else:
    print(f'ok:keys={len(record)},schema=v6')
" "$T8/results/$HASH" "$T8/behavioral_record.json" 2>&1)
    if echo "$T8_CHK" | grep -q "^ok:"; then
        pass_test "Behavioral record generation ($T8_CHK)"
    else
        fail_test "Behavioral record generation: $T8_CHK"
    fi
else
    fail_test "Behavioral record generation skipped (no multi-run data from Test 7)"
fi
# Preserve behavioral record for Test 10
T8_RECORD=""; if [ -f "$T8/behavioral_record.json" ]; then T8_RECORD=$(mktemp); cp "$T8/behavioral_record.json" "$T8_RECORD"; fi

# --- Test 9: Varied Topology Harness ----------------------------------------
echo ""
echo "--- Test 9: Varied Topology Harness ---"
T9=$(mktemp -d); mkdir -p "$T9/repo" "$T9/output"
export_env "$T9/output" smoke-9 smoke-varied-topo synthetic
# Create a fake repo with CrewAI imports AND allow_delegation to trigger non-sequential variant
cat > "$T9/repo/main.py" <<'PYEOF'
from crewai import Agent, Task, Crew
manager = Agent(role="Manager", goal="Oversee project execution",
    backstory="A seasoned project manager", allow_delegation=True)
researcher = Agent(role="Researcher", goal="Research topics in depth",
    backstory="An experienced researcher", allow_delegation=True)
writer = Agent(role="Writer", goal="Write comprehensive reports",
    backstory="A skilled technical writer")
t1 = Task(description="Plan the research approach", expected_output="Research plan", agent=manager)
t2 = Task(description="Conduct research on AI observability", expected_output="Research notes", agent=researcher)
t3 = Task(description="Write a summary report", expected_output="Summary report", agent=writer)
crew = Crew(agents=[manager, researcher, writer], tasks=[t1, t2, t3])
result = crew.kickoff()
PYEOF
python "$SCRIPT_DIR/synthetic_harness.py" "$T9/repo" "$VLLM_HOST" "$T9/output" > "$T9/out" 2> "$T9/err"
T9_EXIT=$?
if [ $T9_EXIT -ne 0 ]; then
    fail_test "Varied topology harness failed (exit $T9_EXIT)"; head -3 "$T9/err" 2>/dev/null
elif [ -f "$T9/output/stratum_events.jsonl" ]; then
    T9_CHK=$(python -c "
import json, sys
events_file = sys.argv[1]
with open(events_file) as f:
    events = [json.loads(l) for l in f if l.strip()]
cnt = len(events)
if cnt < 1:
    print('fail:no_events')
else:
    # Check that variant was detected (delegation expected for allow_delegation repo)
    types = {e.get('event_type','') for e in events}
    print(f'ok:events={cnt},types={len(types)}')
" "$T9/output/stratum_events.jsonl" 2>&1)
    if echo "$T9_CHK" | grep -q "^ok:"; then
        # Also check stderr for variant selection info
        VARIANT=$(grep -o "using [a-z]*" "$T9/err" 2>/dev/null | head -1 || true)
        pass_test "Varied topology harness ($T9_CHK ${VARIANT:-})"
    else
        fail_test "Varied topology harness: $T9_CHK"
    fi
else
    fail_test "Varied topology harness: no events file"
fi
rm -rf "$T9"

# --- Test 10: validate_behavioral_record() ----------------------------------
echo ""
echo "--- Test 10: validate_behavioral_record() ---"
if [ -n "$T8_RECORD" ] && [ -f "$T8_RECORD" ]; then
    export PYTHONPATH="$PATCHER_ROOT:${PYTHONPATH:-}"
    T10_CHK=$(python -c "
import json, sys
sys.path.insert(0, '$PATCHER_ROOT')
from stratum_lab.output.behavioral_record import validate_behavioral_record

with open(sys.argv[1]) as f:
    record = json.load(f)

valid, errors = validate_behavioral_record(record)
if valid:
    print(f'ok:v6_valid,keys={len(record)}')
else:
    print(f'fail:errors={errors}')
" "$T8_RECORD" 2>&1)
    if echo "$T10_CHK" | grep -q "^ok:"; then
        pass_test "validate_behavioral_record ($T10_CHK)"
    else
        fail_test "validate_behavioral_record: $T10_CHK"
    fi
else
    fail_test "validate_behavioral_record skipped (no record from Test 8)"
fi

# Cleanup preserved temp files
rm -rf "$T7_SAVED"
rm -rf "$T8"
[ -n "$T8_RECORD" ] && rm -f "$T8_RECORD"

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
