#!/bin/bash

VLLM="https://aishvbx8prhm6k-8000.proxy.runpod.net"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE=~/stratum-lab/test_results

declare -A REPOS
REPOS[final_crewai]="https://github.com/binbakhsh/QBO-CrewAI"
REPOS[final_langgraph]="https://github.com/DevJadhav/agentic-doc-extraction-system"
REPOS[final_langchain]="https://github.com/itay601/langGraph"
REPOS[final_autogen]="https://github.com/Sonlux/ESCAI"

for name in final_crewai final_langgraph final_langchain final_autogen; do
    url="${REPOS[$name]}"
    outdir="$BASE/$name"
    rm -rf "$outdir"
    mkdir -p "$outdir"
    
    echo ""
    echo "================================================================"
    echo "  TESTING: $name -> $url"
    echo "================================================================"
    
    docker run --rm --network=host \
      -v /tmp/pip-cache:/root/.cache/pip \
      --entrypoint bash \
      -v ~/stratum-lab/run_repo.sh:/app/run_repo.sh:ro \
      -v ~/stratum-lab/scripts:/app/scripts:ro \
      -v ~/stratum-lab/scripts/synthetic_harness.py:/app/synthetic_harness.py:ro \
      -v ~/stratum-lab/test_results/$name:/app/output \
      -e STRATUM_VLLM_MODEL="$MODEL" \
      -e STRATUM_CAPTURE_PROMPTS=1 \
      stratum-lab-base \
      /app/run_repo.sh "$url" "$VLLM" /app/output 600 \
      2>&1 | tee "$outdir/full.log"
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  DETAILED ANALYSIS: $name"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Find events file
    EFILE="$outdir/events_run_1.jsonl"
    if [ ! -f "$EFILE" ]; then
        EFILE="$outdir/stratum_events.jsonl"
    fi

    # --- 1. STATUS & TIER ---
    echo ""
    echo "--- 1. STATUS ---"
    grep "Status:\|Tier:\|Duration:\|Entry:" "$outdir/full.log" | tail -4

    # --- 2. PIP HANDLING ---
    echo ""
    echo "--- 2. PIP HANDLING ---"
    echo "  Protected packages skipped:"
    grep "skipped.*protected" "$outdir/full.log" || echo "    (none logged)"
    echo "  Local package detection:"
    grep "local package" "$outdir/full.log" || echo "    (none logged)"
    echo "  Auto-install attempts:"
    grep "Auto-installing" "$outdir/full.log" || echo "    (none)"
    echo "  Pip failures:"
    grep "Failed to install\|unresolvable" "$outdir/full.log" || echo "    (none)"

    # --- 3. ENTRY POINT ---
    echo ""
    echo "--- 3. ENTRY POINT DETECTION ---"
    grep "Winner:\|Runner-up:\|Entry point:" "$outdir/full.log" || echo "    (not found)"
    echo "  Test file selected as entry?"
    ENTRY=$(grep "Entry point:" "$outdir/full.log" | head -1)
    if echo "$ENTRY" | grep -qE "test_|_test\.py|conftest"; then
        echo "    *** WARNING: Test file selected as entry point ***"
    else
        echo "    OK (no test file)"
    fi

    # --- 4. EVENTS ---
    if [ -f "$EFILE" ]; then
        echo ""
        echo "--- 4. EVENT COUNTS ---"
        python3 -c "
import sys, json, collections
counts = collections.Counter()
for l in open('$EFILE'):
    e = json.loads(l.strip())
    counts[e['event_type']] += 1
for et, c in sorted(counts.items()):
    marker = '  ✅' if et in ('llm.call_start','llm.call_end','agent.task_start','agent.task_end','execution.start','execution.end','edge.traversed') else ''
    print(f'    {et}: {c}{marker}')
BEHAVIORAL = {'llm.call_start','llm.call_end','agent.task_start','agent.task_end','execution.start','execution.end'}
beh = sum(c for et, c in counts.items() if et in BEHAVIORAL)
print(f'  Total behavioral: {beh}')
print(f'  BEHAVIORAL PASS: {\"YES ✅\" if beh > 0 else \"NO ❌\"}')
"

        # --- 5. LLM I/O CAPTURE ---
        echo ""
        echo "--- 5. LLM I/O CAPTURE ---"
        python3 -c "
import sys, json
starts = []
ends = []
for l in open('$EFILE'):
    e = json.loads(l.strip())
    p = e.get('payload', {})
    if e['event_type'] == 'llm.call_start':
        starts.append(p)
    elif e['event_type'] == 'llm.call_end':
        ends.append(p)

if not starts:
    print('  NO llm.call_start events ❌')
else:
    s = starts[0]
    print(f'  llm.call_start (first of {len(starts)}):')
    print(f'    system_prompt_preview: {\"✅ \" + str(s.get(\"system_prompt_preview\",\"\"))[:100] if s.get(\"system_prompt_preview\") else \"❌ MISSING\"}')
    print(f'    last_user_message_preview: {\"✅ \" + str(s.get(\"last_user_message_preview\",\"\"))[:100] if s.get(\"last_user_message_preview\") else \"❌ MISSING\"}')
    print(f'    system_prompt_hash: {\"✅\" if s.get(\"system_prompt_hash\") else \"❌ MISSING\"}')
    print(f'    last_user_message_hash: {\"✅\" if s.get(\"last_user_message_hash\") else \"❌ MISSING\"}')
    print(f'    has_tools: {\"✅\" if \"has_tools\" in s else \"❌ MISSING\"}')
    print(f'    message_count: {s.get(\"message_count\", \"❌ MISSING\")}')
    print(f'    model_actual: {s.get(\"model_actual\", \"❌ MISSING\")}')

if not ends:
    print('  NO llm.call_end events ❌')
else:
    e = ends[0]
    print(f'  llm.call_end (first of {len(ends)}):')
    print(f'    output_preview: {\"✅ \" + str(e.get(\"output_preview\",\"\"))[:100] if e.get(\"output_preview\") else \"❌ MISSING\"}')
    print(f'    output_hash: {\"✅\" if e.get(\"output_hash\") else \"❌ MISSING\"}')
    print(f'    output_type: {e.get(\"output_type\", \"❌ MISSING\")}')
    print(f'    output_size_bytes: {e.get(\"output_size_bytes\", \"❌ MISSING\")}')
    print(f'    input_tokens: {e.get(\"input_tokens\", \"❌ MISSING\")}')
    print(f'    output_tokens: {e.get(\"output_tokens\", \"❌ MISSING\")}')
    print(f'    finish_reason: {e.get(\"finish_reason\", \"❌ MISSING\")}')
    print(f'    latency_ms: {e.get(\"latency_ms\", \"❌ MISSING\")}')
"

        # --- 6. AGENT EVENTS ---
        echo ""
        echo "--- 6. AGENT EVENTS ---"
        python3 -c "
import sys, json
for l in open('$EFILE'):
    e = json.loads(l.strip())
    p = e.get('payload', {})
    if e['event_type'] == 'agent.task_start':
        print(f'  agent.task_start:')
        print(f'    agent_role: {p.get(\"agent_role\", p.get(\"agent_name\", \"❌\"))}')
        print(f'    agent_goal: {str(p.get(\"agent_goal\", p.get(\"agent_type\", \"❌\")))[:100]}')
        print(f'    task_description: {str(p.get(\"task_description\", \"❌\"))[:100]}')
        print(f'    tools_available: {p.get(\"tools_available\", \"❌\")}')
        print(f'    input_source: {p.get(\"input_source\", \"n/a\")}')
        break
for l in open('$EFILE'):
    e = json.loads(l.strip())
    p = e.get('payload', {})
    if e['event_type'] == 'agent.task_end':
        print(f'  agent.task_end:')
        print(f'    output_preview: {\"✅ \" + str(p.get(\"output_preview\",\"\"))[:100] if p.get(\"output_preview\") else \"❌ MISSING\"}')
        print(f'    output_hash: {\"✅\" if p.get(\"output_hash\") else \"❌ MISSING\"}')
        print(f'    latency_ms: {p.get(\"latency_ms\", \"❌\")}')
        print(f'    status: {p.get(\"status\", \"❌\")}')
        break
"

        # --- 7. EXECUTION EVENTS ---
        echo ""
        echo "--- 7. EXECUTION EVENTS ---"
        python3 -c "
import sys, json
for l in open('$EFILE'):
    e = json.loads(l.strip())
    p = e.get('payload', {})
    if e['event_type'] == 'execution.start':
        print(f'  execution.start:')
        for k,v in sorted(p.items()):
            print(f'    {k}: {str(v)[:150]}')
        break
for l in open('$EFILE'):
    e = json.loads(l.strip())
    p = e.get('payload', {})
    if e['event_type'] == 'execution.end':
        print(f'  execution.end:')
        for k,v in sorted(p.items()):
            print(f'    {k}: {str(v)[:150]}')
        break
"

        # --- 8. SOURCE NODES (graph construction) ---
        echo ""
        echo "--- 8. SOURCE NODES (for graph construction) ---"
        python3 -c "
import sys, json
nodes = set()
for l in open('$EFILE'):
    e = json.loads(l.strip())
    sn = e.get('source_node', {})
    if sn and e['event_type'] not in ('patcher.status','file.read','file.write'):
        nodes.add((sn.get('node_type','?'), sn.get('node_id','?')[:80], sn.get('node_name','?')))
for ntype, nid, nname in sorted(nodes):
    print(f'    [{ntype}] {nname} -> {nid}')
"
    else
        echo ""
        echo "  ❌ NO EVENTS FILE FOUND"
    fi
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FINAL SCORECARD                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for name in final_crewai final_langgraph final_langchain final_autogen; do
    status=$(grep "Status:" "$BASE/$name/full.log" | tail -1 | awk '{print $NF}')
    events=$(grep "Events:" "$BASE/$name/full.log" | tail -1 | awk '{print $NF}')
    tier=$(grep "Tier:" "$BASE/$name/full.log" | tail -1 | awk '{print $NF}')
    
    EFILE="$BASE/$name/events_run_1.jsonl"
    [ ! -f "$EFILE" ] && EFILE="$BASE/$name/stratum_events.jsonl"
    
    if [ -f "$EFILE" ]; then
        io=$(python3 -c "
import json
has = {'sp':0,'um':0,'op':0,'tk':0}
for l in open('$EFILE'):
    e=json.loads(l)
    p=e.get('payload',{})
    if p.get('system_prompt_preview'): has['sp']=1
    if p.get('last_user_message_preview'): has['um']=1
    if p.get('output_preview'): has['op']=1
    if p.get('input_tokens'): has['tk']=1
parts=[]
if has['sp']: parts.append('sys_prompt')
if has['um']: parts.append('user_msg')
if has['op']: parts.append('output')
if has['tk']: parts.append('tokens')
print(','.join(parts) if parts else 'none')
")
    else
        io="no_events"
    fi
    
    pass="✅" 
    [ "$status" = "TIER2_FAILED" ] && pass="❌"
    [ "$status" = "" ] && pass="❌"
    
    printf "  %-20s %s  status=%-15s tier=%s events=%-4s io=[%s]\n" "$name" "$pass" "$status" "$tier" "$events" "$io"
done
