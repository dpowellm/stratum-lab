"""Export behavioral traces from raw event JSONL for stratum-graph ingestion.

Reads event files captured by the patcher during sandboxed execution and
produces a clean denormalized summary: one JSON line per repo in
behavioral_traces.jsonl.

Usage:
    python -m stratum_lab.export --results-dir results/full_scan --output behavioral_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS"}


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(events_path: Path) -> list[dict[str, Any]]:
    """Read JSONL, skip malformed lines, sort by timestamp_ns."""
    events: list[dict[str, Any]] = []
    if not events_path.exists():
        return events
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda e: (e.get("timestamp_ns", 0), e.get("timestamp", "")))
    return events


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _payload(evt: dict) -> dict:
    p = evt.get("payload")
    return p if isinstance(p, dict) else {}


def _node_id(evt: dict) -> str:
    sn = evt.get("source_node")
    if isinstance(sn, dict):
        return sn.get("node_id", "")
    return _payload(evt).get("node_id", "")


def _node_name(evt: dict) -> str:
    sn = evt.get("source_node")
    if isinstance(sn, dict):
        return sn.get("node_name", "")
    return ""


def _agent_name(evt: dict) -> str:
    """Extract agent name: payload.agent_role → source_node.node_name."""
    p = _payload(evt)
    role = p.get("agent_role", "")
    if role:
        return role
    return _node_name(evt)


def _timestamp_ms(evt: dict) -> float:
    ns = evt.get("timestamp_ns")
    if ns is not None:
        try:
            return float(ns) / 1e6
        except (TypeError, ValueError):
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Single-run reconstruction
# ---------------------------------------------------------------------------

def _build_run(events: list[dict], run_number: int) -> dict[str, Any]:
    """Reconstruct a single run from its events."""
    # Agent spans keyed by node_id (or agent_name as fallback)
    agent_spans: dict[str, dict] = {}
    # Ordering: preserve insertion order for delegation inference
    agent_order: list[str] = []
    delegation_chains: list[dict] = []
    errors: list[str] = []
    exec_start_ms = 0.0
    exec_end_ms = 0.0

    for evt in events:
        etype = evt.get("event_type", "")
        p = _payload(evt)
        nid = _node_id(evt)

        if etype == "execution.start":
            exec_start_ms = _timestamp_ms(evt)

        elif etype == "execution.end":
            exec_end_ms = _timestamp_ms(evt)

        elif etype == "agent.task_start":
            key = nid or _agent_name(evt) or f"agent_{len(agent_spans)}"
            name = _agent_name(evt)
            if not name:
                name = key
            span = agent_spans.get(key)
            if span is None:
                span = {
                    "agent_name": name,
                    "agent_role": p.get("agent_role", name),
                    "agent_goal": p.get("agent_goal", ""),
                    "agent_goal_hash": p.get("agent_goal_hash", ""),
                    "node_id": nid,
                    "tools_available": p.get("tools_available", []),
                    "tasks": [],
                }
                agent_spans[key] = span
                agent_order.append(key)
            # Start a new task within this agent
            task = {
                "task_description": p.get("task_description", ""),
                "task_description_hash": p.get("task_description_hash", ""),
                "status": "",
                "start_ms": _timestamp_ms(evt),
                "end_ms": 0.0,
                "output_text": "",
                "output_hash": "",
                "output_type": "",
                "output_size_bytes": 0,
                "llm_calls": [],
                "_input_source": p.get("input_source", ""),
                "_parent_node_id": p.get("parent_node_id", ""),
            }
            span["tasks"].append(task)

        elif etype == "agent.task_end":
            key = nid or _agent_name(evt)
            span = agent_spans.get(key)
            if span and span["tasks"]:
                task = span["tasks"][-1]
                task["end_ms"] = _timestamp_ms(evt)
                task["status"] = p.get("status", "success")
                task["output_text"] = p.get("output_preview", "")
                task["output_hash"] = p.get("output_hash", "")
                task["output_type"] = p.get("output_type", "")
                task["output_size_bytes"] = p.get("output_size_bytes", 0) or 0

        elif etype == "llm.call_end":
            key = nid or _agent_name(evt)
            span = agent_spans.get(key)
            llm_rec = {
                "model_requested": p.get("model_requested", ""),
                "model_actual": p.get("model_actual", ""),
                "input_tokens": p.get("input_tokens", 0) or 0,
                "output_tokens": p.get("output_tokens", 0) or 0,
                "latency_ms": p.get("latency_ms", 0) or 0,
                "finish_reason": p.get("finish_reason", ""),
                "output_hash": p.get("output_hash", ""),
                "output_preview": p.get("output_preview", ""),
            }
            if span and span["tasks"]:
                span["tasks"][-1]["llm_calls"].append(llm_rec)

        elif etype == "delegation.initiated":
            src_name = _agent_name(evt)
            src_nid = nid
            tn = evt.get("target_node")
            if isinstance(tn, dict):
                tgt_name = tn.get("node_name", "")
                tgt_nid = tn.get("node_id", "")
            else:
                tgt_name = p.get("delegate", "")
                tgt_nid = ""
            # Find upstream output hash
            upstream_hash = ""
            src_span = agent_spans.get(src_nid) or agent_spans.get(src_name)
            if src_span and src_span["tasks"]:
                upstream_hash = src_span["tasks"][-1].get("output_hash", "")
            if src_name and tgt_name:
                delegation_chains.append({
                    "upstream_agent": src_name,
                    "upstream_node_id": src_nid,
                    "downstream_agent": tgt_name,
                    "downstream_node_id": tgt_nid,
                    "upstream_output_hash": upstream_hash,
                })

        elif etype.startswith("error."):
            errors.append(p.get("message", etype))

    # Infer delegation from input_source == 'delegation' if no explicit events
    if not delegation_chains and len(agent_order) >= 2:
        for i, key in enumerate(agent_order):
            span = agent_spans[key]
            for task in span["tasks"]:
                if task.get("_input_source") == "delegation" and i > 0:
                    prev_key = agent_order[i - 1]
                    prev_span = agent_spans[prev_key]
                    upstream_hash = ""
                    if prev_span["tasks"]:
                        upstream_hash = prev_span["tasks"][-1].get("output_hash", "")
                    delegation_chains.append({
                        "upstream_agent": prev_span["agent_name"],
                        "upstream_node_id": prev_span["node_id"],
                        "downstream_agent": span["agent_name"],
                        "downstream_node_id": span["node_id"],
                        "upstream_output_hash": upstream_hash,
                    })
                    break  # one delegation link per agent

    # Build final agents list and compute task latencies
    agents_out: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    total_tasks = 0

    for key in agent_order:
        span = agent_spans[key]
        tasks_out: list[dict] = []
        for task in span["tasks"]:
            latency = 0.0
            if task["end_ms"] > task["start_ms"]:
                latency = task["end_ms"] - task["start_ms"]
            t_input = sum(c["input_tokens"] for c in task["llm_calls"])
            t_output = sum(c["output_tokens"] for c in task["llm_calls"])
            total_input_tokens += t_input
            total_output_tokens += t_output
            total_llm_calls += len(task["llm_calls"])
            total_tasks += 1
            tasks_out.append({
                "task_description": task["task_description"],
                "task_description_hash": task["task_description_hash"],
                "status": task["status"] or ("incomplete" if task["end_ms"] == 0 else "success"),
                "latency_ms": latency,
                "output_text": task["output_text"],
                "output_hash": task["output_hash"],
                "output_type": task["output_type"],
                "output_size_bytes": task["output_size_bytes"],
                "llm_calls": task["llm_calls"],
            })
        agents_out.append({
            "agent_name": span["agent_name"],
            "agent_role": span["agent_role"],
            "agent_goal": span["agent_goal"],
            "agent_goal_hash": span["agent_goal_hash"],
            "node_id": span["node_id"],
            "tools_available": span["tools_available"],
            "tasks": tasks_out,
        })

    # Compute delegation depth
    delegation_depth = 0
    if delegation_chains:
        # Build adjacency and find longest path
        adj: dict[str, list[str]] = {}
        for dc in delegation_chains:
            adj.setdefault(dc["upstream_agent"], []).append(dc["downstream_agent"])
        for root in adj:
            stack = [(root, 0)]
            visited: set[str] = set()
            while stack:
                node, depth = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                delegation_depth = max(delegation_depth, depth)
                for child in adj.get(node, []):
                    stack.append((child, depth + 1))

    total_duration_ms = 0.0
    if exec_end_ms > exec_start_ms:
        total_duration_ms = exec_end_ms - exec_start_ms

    return {
        "run_number": run_number,
        "agents": agents_out,
        "delegation_chains": delegation_chains,
        "execution_summary": {
            "total_agents": len(agents_out),
            "total_tasks": total_tasks,
            "total_llm_calls": total_llm_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration_ms": total_duration_ms,
            "has_delegation": len(delegation_chains) > 0,
            "delegation_depth": delegation_depth,
            "errors": errors,
        },
    }


# ---------------------------------------------------------------------------
# Per-repo summary
# ---------------------------------------------------------------------------

def build_repo_summary(repo_dir: Path) -> dict[str, Any] | None:
    """Build a full summary for one repo directory.

    Returns None if no usable runs (all failed or missing events).
    """
    repo_dir = Path(repo_dir)
    runs: list[dict] = []

    # Discover run metadata files
    meta_files = sorted(repo_dir.glob("run_metadata_*.json"))
    if not meta_files:
        return None

    repo_id = ""
    framework = ""
    repo_hash = repo_dir.name

    for meta_path in meta_files:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        status = meta.get("status", meta.get("execution_status", ""))
        if status not in VALID_STATUSES:
            continue

        # Extract run number
        stem = meta_path.stem  # run_metadata_1
        run_num_str = stem.rsplit("_", 1)[-1]
        try:
            run_number = int(run_num_str)
        except ValueError:
            run_number = 1

        events_path = repo_dir / f"events_run_{run_number}.jsonl"
        if not events_path.exists():
            continue

        events = load_events(events_path)
        if not events:
            continue

        # Extract repo_id and framework from events (top-level fields)
        if not repo_id:
            for evt in events:
                if not repo_id:
                    repo_id = evt.get("repo_id", "")
                if not framework:
                    framework = evt.get("framework", "")
                if repo_id and framework:
                    break
        # Fallback from metadata
        if not repo_id:
            repo_id = meta.get("repo_url", meta.get("repo_id", ""))
        if not framework:
            framework = meta.get("framework", "")

        run = _build_run(events, run_number)
        runs.append(run)

    if not runs:
        return None

    scan_status = "SUCCESS"  # At least one usable run exists

    return {
        "repo_id": repo_id,
        "repo_hash": repo_hash,
        "framework": framework,
        "scan_status": scan_status,
        "run_count": len(runs),
        "runs": runs,
        "raw_events_path": str(repo_dir),
    }


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def export_behavioral_traces(
    results_dir: Path,
    output_path: Path,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Export all repos to behavioral_traces.jsonl.

    Returns export statistics dict.
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_repos_scanned": 0,
        "total_repos_exported": 0,
        "success_count": 0,
        "partial_count": 0,
        "skipped_count": 0,
        "total_agents": 0,
        "total_llm_calls": 0,
        "frameworks": {},
    }

    seen_repo_ids: set[str] = set()
    raw_dir = output_path.parent / "raw_events" if include_raw else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        if not results_dir.is_dir():
            logger.warning("Results directory does not exist: %s", results_dir)
            return stats

        for repo_dir in sorted(results_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            stats["total_repos_scanned"] += 1

            summary = build_repo_summary(repo_dir)
            if summary is None:
                stats["skipped_count"] += 1
                continue

            # Skip duplicates
            rid = summary["repo_id"]
            if rid in seen_repo_ids:
                continue
            seen_repo_ids.add(rid)

            out.write(json.dumps(summary, default=str) + "\n")
            stats["total_repos_exported"] += 1

            # Count status
            if summary["scan_status"] == "SUCCESS":
                stats["success_count"] += 1
            else:
                stats["partial_count"] += 1

            # Aggregate stats
            fw = summary.get("framework", "unknown")
            stats["frameworks"][fw] = stats["frameworks"].get(fw, 0) + 1
            for run in summary["runs"]:
                es = run["execution_summary"]
                stats["total_agents"] += es["total_agents"]
                stats["total_llm_calls"] += es["total_llm_calls"]

            # Copy raw events if requested
            if raw_dir:
                dest = raw_dir / repo_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for evf in repo_dir.glob("events_run_*.jsonl"):
                    shutil.copy2(evf, dest / evf.name)

    logger.info(
        "Exported %d repos to %s", stats["total_repos_exported"], output_path,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="stratum-lab-export",
        description="Export behavioral traces for stratum-graph ingestion",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to scan results directory (e.g. results/full_scan/results)",
    )
    parser.add_argument(
        "--output",
        default="behavioral_traces.jsonl",
        help="Output JSONL file path (default: behavioral_traces.jsonl)",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Copy raw event JSONL files to raw_events/ directory",
    )
    args = parser.parse_args()

    stats = export_behavioral_traces(
        Path(args.results_dir),
        Path(args.output),
        include_raw=args.include_raw,
    )

    print(f"\nExport complete:")
    print(f"  Repos scanned:  {stats['total_repos_scanned']}")
    print(f"  Repos exported: {stats['total_repos_exported']}")
    print(f"  Skipped:        {stats['skipped_count']}")
    print(f"  Agents:         {stats['total_agents']}")
    print(f"  LLM calls:      {stats['total_llm_calls']}")
    print(f"  Frameworks:     {stats['frameworks']}")


if __name__ == "__main__":
    main()
