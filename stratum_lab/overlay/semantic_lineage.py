"""Reconstruct semantic data flow across the agent graph.

From the event stream, build a lineage graph showing:
- Which node's output becomes which node's input (content hash matching)
- Where classification/routing decisions enter the chain
- Which handoffs are validated (guardrail between) vs. unvalidated
- The semantic blast radius of each node
"""

from collections import defaultdict
from typing import Any, Dict, List, Set


def reconstruct_lineage(events: List[Dict], structural_graph: Dict) -> Dict:
    """Build the semantic lineage graph from events."""

    # Build output map: node_id -> [(run_id, output_hash, output_type, classification_fields)]
    node_outputs = defaultdict(list)
    for event in events:
        if event.get("event_type") in ("agent.task_end", "llm.call_end"):
            payload = event.get("payload", {})
            source = event.get("source_node", {})
            if payload.get("output_hash"):
                node_outputs[source.get("node_id", "")].append({
                    "run_id": event.get("run_id", ""),
                    "output_hash": payload["output_hash"],
                    "output_type": payload.get("output_type", "unknown"),
                    "classification_fields": payload.get("classification_fields"),
                })

    # Build input map: target_node -> [(run_id, context_hash, source_node)]
    node_inputs = defaultdict(list)
    for event in events:
        if event.get("event_type") == "delegation.initiated":
            payload = event.get("payload", {})
            target = event.get("target_node", {})
            if payload.get("context_hash"):
                node_inputs[target.get("node_id", "")].append({
                    "run_id": event.get("run_id", ""),
                    "context_hash": payload["context_hash"],
                    "context_source_node": payload.get("context_source_node", ""),
                    "has_classification_dependency": payload.get("has_classification_dependency", False),
                })

    # Match outputs to inputs via hash -> build handoff edges
    handoffs = []
    for target_node, inputs in node_inputs.items():
        for inp in inputs:
            source_node = inp["context_source_node"]
            if not source_node:
                for cand_node, outputs in node_outputs.items():
                    for out in outputs:
                        if out["output_hash"] == inp["context_hash"] and out["run_id"] == inp["run_id"]:
                            source_node = cand_node
                            break

            if source_node:
                handoffs.append({
                    "source_node": source_node,
                    "target_node": target_node,
                    "content_hash": inp["context_hash"],
                    "run_id": inp["run_id"],
                    "has_classification_dependency": inp["has_classification_dependency"],
                    "validated": _is_validated_handoff(source_node, target_node, structural_graph),
                })

    unvalidated = [h for h in handoffs if not h["validated"]]

    # Semantic blast radius: per node, transitive downstream consumers
    adjacency = defaultdict(set)
    for h in handoffs:
        adjacency[h["source_node"]].add(h["target_node"])

    blast_radius = {}
    for node_id in adjacency:
        visited: Set[str] = set()
        _dfs(node_id, adjacency, visited)
        visited.discard(node_id)
        blast_radius[node_id] = len(visited)

    # Classification injection points
    classification_nodes = {
        node_id for node_id, outputs in node_outputs.items()
        if any(o.get("classification_fields") for o in outputs)
    }

    # Semantic determinism
    semantic_determinism = _compute_semantic_determinism(node_outputs, events)

    return {
        "handoffs": handoffs,
        "total_handoffs": len(handoffs),
        "unvalidated_handoffs": unvalidated,
        "unvalidated_count": len(unvalidated),
        "unvalidated_fraction": len(unvalidated) / max(len(handoffs), 1),
        "semantic_blast_radius": blast_radius,
        "max_blast_radius": max(blast_radius.values(), default=0),
        "classification_injection_points": list(classification_nodes),
        "classification_injection_count": len(classification_nodes),
        "semantic_determinism": semantic_determinism,
        "semantic_chain_depth": _max_chain_depth(adjacency),
    }


def _is_validated_handoff(source: str, target: str, graph: Dict) -> bool:
    """Check if a guardrail node sits between source and target."""
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    guardrails = set()
    for nid, n in nodes.items():
        struct = n.get("structural", n)
        if struct.get("node_type") == "guardrail":
            guardrails.add(nid)

    for gid in guardrails:
        incoming = any(
            e.get("structural", e).get("source") == source
            and e.get("structural", e).get("target") == gid
            for e in edges.values()
        )
        outgoing = any(
            e.get("structural", e).get("source") == gid
            and e.get("structural", e).get("target") == target
            for e in edges.values()
        )
        if incoming and outgoing:
            return True
    return False


def _dfs(node: str, adjacency: Dict[str, Set[str]], visited: Set[str]) -> None:
    if node in visited:
        return
    visited.add(node)
    for neighbor in adjacency.get(node, set()):
        _dfs(neighbor, adjacency, visited)


def _max_chain_depth(adjacency: Dict[str, Set[str]]) -> int:
    def depth(node: str, visited: set) -> int:
        if node in visited:
            return 0
        visited.add(node)
        return max(
            (1 + depth(n, visited.copy()) for n in adjacency.get(node, set())),
            default=0,
        )
    return max((depth(n, set()) for n in adjacency), default=0)


def _compute_semantic_determinism(
    node_outputs: Dict[str, list], events: List[Dict]
) -> Dict:
    """For same-input runs, check if each node produces the same output hash."""
    run_inputs: Dict[str, str] = {}
    for e in events:
        if e.get("event_type") == "execution.start":
            payload = e.get("payload", {})
            run_inputs[e.get("run_id", "")] = payload.get("input_hash", "")

    groups: Dict[tuple, list] = defaultdict(list)
    for node_id, outputs in node_outputs.items():
        for out in outputs:
            ih = run_inputs.get(out["run_id"], "")
            if ih:
                groups[(node_id, ih)].append(out["output_hash"])

    results = {}
    for (node_id, _), hashes in groups.items():
        if len(hashes) >= 2:
            unique = len(set(hashes))
            results[node_id] = {
                "same_input_runs": len(hashes),
                "unique_outputs": unique,
                "semantically_deterministic": unique == 1,
                "semantic_consistency": 1.0 - (unique - 1) / max(len(hashes) - 1, 1),
            }
    return results
