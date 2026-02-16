#!/usr/bin/env python3
"""Convert stratum_events.jsonl into a behavioral graph JSON file.

Runs inside the Docker container after execution completes.

Usage:  python graph_builder.py <events_file> <output_file>
Exit codes: 0 = success, 1 = error or no events
"""

import json
import sys
from collections import defaultdict


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


def extract_nodes(events):
    """Build node map from source_node and target_node fields."""
    nodes = {}

    for ev in events:
        ts = ev.get("timestamp_ns", 0)
        for role in ("source_node", "target_node"):
            info = ev.get(role)
            if not info or not info.get("node_id"):
                continue
            nid = info["node_id"]
            if nid not in nodes:
                nodes[nid] = {
                    "node_id": nid,
                    "node_type": info.get("node_type", "unknown"),
                    "node_name": info.get("node_name", ""),
                    "event_count": 0,
                    "first_seen_ns": ts,
                    "last_seen_ns": ts,
                }
            node = nodes[nid]
            if role == "source_node":
                node["event_count"] += 1
            if ts and ts < node["first_seen_ns"]:
                node["first_seen_ns"] = ts
            if ts and ts > node["last_seen_ns"]:
                node["last_seen_ns"] = ts

    return nodes


def extract_edges(events):
    """Build deduplicated edges from events with both source and target."""
    edge_map = {}

    for ev in events:
        src = ev.get("source_node")
        tgt = ev.get("target_node")
        if not src or not tgt:
            continue
        src_id, tgt_id = src.get("node_id"), tgt.get("node_id")
        if not src_id or not tgt_id:
            continue

        etype = ev.get("edge_type", "unknown")
        ts = ev.get("timestamp_ns", 0)
        key = (src_id, tgt_id, etype)

        if key not in edge_map:
            edge_map[key] = {
                "source": src_id, "target": tgt_id,
                "edge_type": etype, "event_count": 0, "first_seen_ns": ts,
            }
        edge_map[key]["event_count"] += 1
        if ts and ts < edge_map[key]["first_seen_ns"]:
            edge_map[key]["first_seen_ns"] = ts

    return list(edge_map.values())


def extract_content_flow(events):
    """Track output hashes flowing between nodes via payload.output_hash."""
    hash_map = {}

    for ev in events:
        payload = ev.get("payload") or {}
        h = payload.get("output_hash")
        if not h:
            continue

        src_id = (ev.get("source_node") or {}).get("node_id")
        tgt_id = (ev.get("target_node") or {}).get("node_id")

        if h not in hash_map:
            hash_map[h] = {
                "hash": h, "producer_node": src_id,
                "consumer_nodes": [], "event_type": ev.get("event_type", ""),
            }
        entry = hash_map[h]
        if entry["producer_node"] is None and src_id:
            entry["producer_node"] = src_id
        if tgt_id and tgt_id != entry["producer_node"]:
            if tgt_id not in entry["consumer_nodes"]:
                entry["consumer_nodes"].append(tgt_id)

    return list(hash_map.values())


def extract_risk_indicators(events):
    """Extract risk-related metrics from the event stream."""
    max_depth = 0
    error_count = 0
    error_types = set()
    tool_failure_count = 0
    llm_call_count = 0
    error_etypes = {"error.occurred", "error.propagated"}

    for ev in events:
        etype = ev.get("event_type", "")
        depth = ev.get("stack_depth", 0)
        if isinstance(depth, int) and depth > max_depth:
            max_depth = depth
        if etype in error_etypes:
            error_count += 1
            err_type = (ev.get("payload") or {}).get("error_type")
            if err_type:
                error_types.add(err_type)
        if etype == "tool.call_failure":
            tool_failure_count += 1
        if etype == "llm.call_start":
            llm_call_count += 1

    return {
        "max_delegation_depth": max_depth,
        "error_count": error_count,
        "tool_failure_count": tool_failure_count,
        "llm_call_count": llm_call_count,
        "unique_error_types": sorted(error_types),
    }


def build_summary(events, nodes, edges):
    """Build summary statistics for the graph."""
    type_dist = defaultdict(int)
    min_ts = max_ts = None

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


def build_graph(events):
    """Build the full graph structure from a list of events."""
    if not events:
        return None

    first = events[0]
    nodes = extract_nodes(events)
    edges = extract_edges(events)
    content_flow = extract_content_flow(events)
    risk = extract_risk_indicators(events)
    summary = build_summary(events, nodes, edges)

    return {
        "repo_id": first.get("repo_id", ""),
        "run_id": first.get("run_id", ""),
        "framework": first.get("framework", ""),
        "nodes": list(nodes.values()),
        "edges": edges,
        "content_flow": content_flow,
        "risk_indicators": risk,
        "summary": summary,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python graph_builder.py <events_file> <output_file>",
              file=sys.stderr)
        sys.exit(1)

    events_file, output_file = sys.argv[1], sys.argv[2]

    try:
        events = load_events(events_file)
    except FileNotFoundError:
        print(f"ERROR: events file not found: {events_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: failed to read events: {exc}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print("ERROR: no events found in input file", file=sys.stderr)
        sys.exit(1)

    graph = build_graph(events)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
    except Exception as exc:
        print(f"ERROR: failed to write output: {exc}", file=sys.stderr)
        sys.exit(1)

    s = graph["summary"]
    print(f"Graph built: {s['unique_nodes']} nodes, "
          f"{s['unique_edges']} edges, {s['total_events']} events")


if __name__ == "__main__":
    main()
