#!/usr/bin/env python3
"""Convert stratum_events.jsonl into a behavioral graph JSON file.

Major upgrade: event-type-specific node enrichment, payload-driven edge
extraction, content-flow tracking with transformation analysis, structural
risk indicators, topology classification, and topology hashing.

Usage:  python graph_builder.py <events_file> <output_file> [--tier N]
Exit codes: 0 = success, 1 = error or no events
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(path):
    """Load JSONL events, skipping malformed lines."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed line {lineno}", file=sys.stderr)
    return events


# ---------------------------------------------------------------------------
# Node extraction with event-type-specific enrichment
# ---------------------------------------------------------------------------

def extract_nodes(events):
    """Build enriched node map from source_node fields and event payloads."""
    nodes = {}

    for event in events:
        sn = event.get("source_node")
        if not sn or not sn.get("node_id"):
            continue

        node_id = sn["node_id"]
        ts = event.get("timestamp_ns", 0)
        payload = event.get("payload") or {}

        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "type": sn.get("node_type", "unknown"),
                "name": sn.get("node_name", ""),
                "metadata": {},
                "event_count": 0,
                "first_seen_ns": ts,
                "last_seen_ns": ts,
            }

        node = nodes[node_id]
        node["event_count"] += 1
        if ts and (node["first_seen_ns"] == 0 or ts < node["first_seen_ns"]):
            node["first_seen_ns"] = ts
        if ts and ts > node["last_seen_ns"]:
            node["last_seen_ns"] = ts

        # Event-type-specific enrichment
        event_type = event.get("event_type", "")

        if event_type == "execution.start":
            node["type"] = "orchestrator"
            node["metadata"]["agent_count"] = payload.get("agent_count")
            node["metadata"]["agent_roles"] = payload.get("agent_roles")

        if event_type == "agent.task_start":
            node["metadata"]["role"] = payload.get("agent_role")
            node["metadata"]["goal_hash"] = payload.get("agent_goal_hash")
            node["metadata"]["tools"] = payload.get("tools_available")

    # Also register nodes seen only in target_node
    for event in events:
        tn = event.get("target_node")
        if not tn or not tn.get("node_id"):
            continue
        node_id = tn["node_id"]
        ts = event.get("timestamp_ns", 0)
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "type": tn.get("node_type", "unknown"),
                "name": tn.get("node_name", ""),
                "metadata": {},
                "event_count": 0,
                "first_seen_ns": ts,
                "last_seen_ns": ts,
            }
        else:
            node = nodes[node_id]
            if ts and (node["first_seen_ns"] == 0 or ts < node["first_seen_ns"]):
                node["first_seen_ns"] = ts
            if ts and ts > node["last_seen_ns"]:
                node["last_seen_ns"] = ts

    return nodes


# ---------------------------------------------------------------------------
# Edge extraction from payload fields
# ---------------------------------------------------------------------------

def extract_edges(events):
    """Build deduplicated edges from event payload relationships."""
    edge_map = {}

    for event in events:
        sn = event.get("source_node")
        if not sn or not sn.get("node_id"):
            continue

        event_type = event.get("event_type", "")
        payload = event.get("payload") or {}
        src_id = sn["node_id"]

        if event_type == "agent.task_start":
            parent = payload.get("parent_node_id")
            if parent:
                _add_edge(edge_map, parent, src_id, "orchestrates")
            input_src = payload.get("input_source")
            if input_src:
                _add_edge(edge_map, input_src, src_id, "output_flows_to")

        if event_type == "llm.call_end":
            stack = payload.get("active_node_stack", [])
            if stack:
                _add_edge(edge_map, stack[-1], src_id, "calls_llm")

        if "delegation" in event_type:
            tn = event.get("target_node")
            if tn and tn.get("node_id"):
                _add_edge(edge_map, src_id, tn["node_id"], "delegates_to")

    return list(edge_map.values())


def _add_edge(edge_map, source, target, edge_type):
    """Add or increment an edge in the dedup map."""
    key = (source, target, edge_type)
    if key not in edge_map:
        edge_map[key] = {
            "source": source,
            "target": target,
            "type": edge_type,
            "count": 0,
        }
    edge_map[key]["count"] += 1


# ---------------------------------------------------------------------------
# Content flow tracking
# ---------------------------------------------------------------------------

def extract_content_flow(events, output_hashes):
    """Track content flowing between nodes via output_hash and input_source."""
    content_flow = []

    for event in events:
        if event.get("event_type") != "agent.task_start":
            continue
        payload = event.get("payload") or {}
        input_src = payload.get("input_source")
        sn = event.get("source_node")
        if not input_src or not sn or not sn.get("node_id"):
            continue
        if input_src in output_hashes:
            content_flow.append({
                "from_node": input_src,
                "to_node": sn["node_id"],
                "output_hash": output_hashes[input_src]["hash"],
                "output_type": output_hashes[input_src]["type"],
                "output_size": output_hashes[input_src]["size"],
            })

    return content_flow


def build_output_hashes(events):
    """Map node_id -> output metadata from agent.task_end events."""
    output_hashes = {}

    for event in events:
        if event.get("event_type") != "agent.task_end":
            continue
        payload = event.get("payload") or {}
        oh = payload.get("output_hash")
        if not oh:
            continue
        sn = event.get("source_node")
        if not sn or not sn.get("node_id"):
            continue
        node_id = sn["node_id"]
        preview = payload.get("output_preview", "") or ""
        output_hashes[node_id] = {
            "hash": oh,
            "type": payload.get("output_type"),
            "size": payload.get("output_size_bytes"),
            "preview": preview[:100],
        }

    return output_hashes


# ---------------------------------------------------------------------------
# Transformation analysis on content flow
# ---------------------------------------------------------------------------

def analyze_transformations(content_flow, output_hashes):
    """Annotate content flow entries with transformation ratios and types."""
    for flow in content_flow:
        from_size = (output_hashes.get(flow["from_node"]) or {}).get("size", 0) or 0
        to_size = (output_hashes.get(flow["to_node"]) or {}).get("size", 0) or 0
        if from_size > 0 and to_size > 0:
            ratio = to_size / from_size
            flow["transformation_ratio"] = round(ratio, 4)
            if ratio > 3.0:
                flow["transformation_type"] = "elaboration"
            elif ratio < 0.3:
                flow["transformation_type"] = "compression"
            elif 0.8 < ratio < 1.2:
                flow["transformation_type"] = "pass_through"
            else:
                flow["transformation_type"] = "transformation"


# ---------------------------------------------------------------------------
# Structural risk indicators
# ---------------------------------------------------------------------------

def compute_longest_path(nodes, edges):
    """Compute the longest path length in the directed graph (edge count)."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["source"]].append(e["target"])

    node_ids = set(nodes.keys())
    memo = {}

    def dfs(node, visited):
        if node in memo:
            return memo[node]
        if node in visited:
            return 0  # cycle detected, stop
        visited.add(node)
        max_depth = 0
        for neighbor in adj.get(node, []):
            if neighbor in node_ids:
                max_depth = max(max_depth, 1 + dfs(neighbor, visited))
        visited.discard(node)
        memo[node] = max_depth
        return max_depth

    longest = 0
    for nid in node_ids:
        longest = max(longest, dfs(nid, set()))
    return longest


def count_outgoing(node_id, edges):
    """Count outgoing edges from a node."""
    return sum(1 for e in edges if e["source"] == node_id)


def extract_risk_indicators(events, nodes, edges, content_flow):
    """Extract structural risk indicators from the event stream and graph."""
    unique_models = set()
    for e in events:
        if e.get("event_type") == "llm.call_end":
            model = (e.get("payload") or {}).get("model_actual")
            if model:
                unique_models.add(model)
        if e.get("event_type") == "llm.call_start":
            model = (e.get("payload") or {}).get("model_requested")
            if model:
                unique_models.add(model)

    max_fan_out = 0
    if nodes:
        max_fan_out = max(count_outgoing(nid, edges) for nid in nodes)

    model_degradation_detected = any(
        (e.get("payload") or {}).get("model_requested") != (e.get("payload") or {}).get("model_actual")
        for e in events
        if e.get("event_type") == "llm.call_end"
    )

    llm_start_events = [e for e in events if e.get("event_type") == "llm.call_start"]
    all_calls_model_remapped = (
        all(
            (e.get("payload") or {}).get("model_mapped", False)
            for e in llm_start_events
        )
        if llm_start_events
        else False
    )

    return {
        "single_model_dependency": len(unique_models) <= 1,
        "unvalidated_pass_through": any(
            f.get("transformation_type") == "pass_through"
            for f in content_flow
            if "transformation_type" in f
        ),
        "max_chain_depth": compute_longest_path(nodes, edges),
        "has_error_boundaries": any(
            e.get("event_type") == "error.occurred" for e in events
        ),
        "fan_out_degree": max_fan_out,
        "model_degradation_detected": model_degradation_detected,
        "all_calls_model_remapped": all_calls_model_remapped,
        "no_agent_detected_degradation": True,
    }


# ---------------------------------------------------------------------------
# Topology classification
# ---------------------------------------------------------------------------

def classify_topology(nodes, edges):
    """Classify the agent topology based on node types and edge patterns."""
    agent_nodes = [
        n for n in nodes.values()
        if n["type"] in ("agent", "orchestrator")
    ]

    if len(agent_nodes) <= 1:
        return "single_agent"

    max_fan_out = max(
        count_outgoing(n["id"], edges) for n in agent_nodes
    ) if agent_nodes else 0

    has_delegation = any(e["type"] == "delegates_to" for e in edges)

    if has_delegation:
        return "hierarchical_delegation"
    if max_fan_out > 2:
        return "hub_and_spoke"
    return f"sequential_{len(agent_nodes)}_agent"


# ---------------------------------------------------------------------------
# Topology hash
# ---------------------------------------------------------------------------

def compute_topology_hash(edges):
    """SHA-256 of sorted edge tuples (source, target, type) for dedup."""
    edge_tuples = sorted((e["source"], e["target"], e["type"]) for e in edges)
    raw = json.dumps(edge_tuples, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(events, nodes, edges):
    """Build summary statistics for the graph."""
    type_dist = defaultdict(int)
    min_ts = None
    max_ts = None

    for ev in events:
        type_dist[ev.get("event_type", "unknown")] += 1
        ts = ev.get("timestamp_ns")
        if ts:
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts

    duration_ms = (max_ts - min_ts) // 1_000_000 if min_ts and max_ts else 0

    return {
        "total_events": len(events),
        "unique_nodes": len(nodes),
        "unique_edges": len(edges),
        "event_type_distribution": dict(type_dist),
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------

def resolve_tier(events_file, cli_tier):
    """Read tier from status.json in the same directory, else use CLI flag."""
    status_path = os.path.join(os.path.dirname(events_file), "status.json")
    if os.path.isfile(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            tier = status.get("tier")
            if tier is not None:
                return int(tier)
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    return cli_tier


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_graph(events, tier):
    """Build the full behavioral graph structure from a list of events."""
    if not events:
        return None

    first = events[0]

    # Nodes
    nodes = extract_nodes(events)

    # Edges
    edges = extract_edges(events)

    # Content flow
    output_hashes = build_output_hashes(events)
    content_flow = extract_content_flow(events, output_hashes)
    analyze_transformations(content_flow, output_hashes)

    # Risk indicators
    risk = extract_risk_indicators(events, nodes, edges, content_flow)

    # Topology
    topology_type = classify_topology(nodes, edges)
    topology_hash = compute_topology_hash(edges)

    # Summary
    summary = build_summary(events, nodes, edges)

    return {
        "repo_id": first.get("repo_id", ""),
        "framework": first.get("framework", ""),
        "tier": tier,
        "nodes": list(nodes.values()),
        "edges": edges,
        "content_flow": content_flow,
        "risk_indicators": risk,
        "topology_type": topology_type,
        "topology_hash": topology_hash,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert stratum_events.jsonl into a behavioral graph JSON file."
    )
    parser.add_argument("events_file", help="Path to stratum_events.jsonl")
    parser.add_argument("output_file", help="Path for output graph JSON")
    parser.add_argument(
        "--tier", type=int, default=1,
        help="Tier number (default 1; overridden by status.json if present)"
    )
    args = parser.parse_args()

    tier = resolve_tier(args.events_file, args.tier)

    try:
        events = load_events(args.events_file)
    except FileNotFoundError:
        print(f"ERROR: events file not found: {args.events_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: failed to read events: {exc}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print("ERROR: no events found in input file", file=sys.stderr)
        sys.exit(1)

    graph = build_graph(events, tier)

    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
    except Exception as exc:
        print(f"ERROR: failed to write output: {exc}", file=sys.stderr)
        sys.exit(1)

    s = graph["summary"]
    topo = graph["topology_type"]
    print(
        f"Graph built: {s['unique_nodes']} nodes, {s['unique_edges']} edges, "
        f"{s['total_events']} events, topology: {topo}"
    )


if __name__ == "__main__":
    main()
