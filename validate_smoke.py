import json, sys
from collections import Counter
from pathlib import Path

def validate(results_dir, expected_framework="unknown"):
    p = Path(results_dir)
    events_file = p / "stratum_events.jsonl"
    
    print(f"\n{'='*60}")
    print(f"Validating: {results_dir} (expected: {expected_framework})")
    
    if not events_file.exists():
        print("FAIL: No stratum_events.jsonl")
        stderr = p / "stderr.log"
        if stderr.exists():
            print(f"stderr tail:\n{stderr.read_text()[-500:]}")
        stdout = p / "stdout.log"
        if stdout.exists():
            print(f"stdout tail:\n{stdout.read_text()[-500:]}")
        return False
    
    events = []
    for line in events_file.read_text().strip().split("\n"):
        if line:
            events.append(json.loads(line))
    
    print(f"Total events: {len(events)}")
    
    if len(events) == 0:
        print("FAIL: Zero events")
        return False
    
    type_counts = Counter(e.get("event_type") for e in events)
    print(f"\nEvent types:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")
    
    llm_events = type_counts.get("llm.call_end", 0)
    if llm_events > 0:
        print(f"\nPASS: LLM patcher fired ({llm_events} completions)")
    else:
        print(f"\nFAIL: No llm.call_end events")
    
    node_events = [e for e in events if e.get("source_node_id")]
    unique_nodes = set(e["source_node_id"] for e in node_events)
    print(f"Nodes: {len(node_events)} attributed, {len(unique_nodes)} unique: {unique_nodes}")
    
    semantic = [e for e in events if e.get("data", {}).get("output_hash")]
    print(f"Semantic capture: {len(semantic)}/{len(events)} have output_hash")
    
    delegations = [e for e in events if "delegation" in e.get("event_type", "")]
    print(f"Delegations: {len(delegations)}")
    
    exit_file = p / "exit_code.txt"
    if exit_file.exists():
        code = exit_file.read_text().strip()
        print(f"Exit code: {code}")
    
    print(f"{'='*60}")
    return llm_events > 0

if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown")
