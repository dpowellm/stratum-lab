"""Compare node IDs produced by stratum-cli structural scan vs. stratum-patcher instrumentation.

Usage: stratum-lab validate-ids <repo_path> [--graph-json <path>] [--events-dir <path>]

Flow:
1. If --graph-json provided, load structural node IDs from it
2. If --events-dir provided, load event node IDs from JSONL files
3. Compare the two sets:
   - matched: IDs appearing in both sets
   - structural_only: IDs in graph but not in events (dead nodes or uninstrumented)
   - behavioral_only: IDs in events but not in graph (unmapped events)
4. Report match rate and specific mismatches
"""

import json
from pathlib import Path
from typing import Dict, Set


def load_structural_ids(graph_path: Path) -> Set[str]:
    """Extract node IDs from a stratum-cli structural graph JSON."""
    with open(graph_path) as f:
        graph = json.load(f)
    nodes = graph.get("nodes", {})
    return set(nodes.keys())


def load_event_ids(events_dir: Path) -> Set[str]:
    """Extract unique source_node_id values from JSONL event files."""
    ids: Set[str] = set()
    for jsonl_file in events_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    src = event.get("source_node_id")
                    if src:
                        ids.add(src)
    return ids


def compare_ids(structural: Set[str], behavioral: Set[str]) -> Dict:
    """Compare two sets of node IDs and report compatibility."""
    matched = structural & behavioral
    structural_only = structural - behavioral
    behavioral_only = behavioral - structural
    total = len(structural | behavioral)
    match_rate = len(matched) / total if total > 0 else 0.0

    return {
        "match_rate": match_rate,
        "matched_count": len(matched),
        "structural_only_count": len(structural_only),
        "behavioral_only_count": len(behavioral_only),
        "matched": sorted(matched),
        "structural_only": sorted(structural_only),
        "behavioral_only": sorted(behavioral_only),
        "compatible": match_rate >= 0.5 and len(behavioral_only) == 0,
    }
