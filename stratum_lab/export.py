"""Export behavioral traces from raw event JSONL for stratum-graph ingestion.

Reads event files captured by the patcher during sandboxed execution and
produces a clean denormalized summary: one JSON line per repo in
behavioral_traces.jsonl.

Usage:
    python -m stratum_lab.export --scan-dirs results/full_scan/results --output behavioral_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(events_path: Path) -> list[dict[str, Any]]:
    """Read JSONL, skip malformed lines, sort by timestamp_ns."""
    events: list[dict[str, Any]] = []
    if not events_path.exists():
        return events
    with open(events_path, encoding="utf-8", errors="replace") as f:
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
    """Extract agent name: payload.agent_role -> source_node.node_name."""
    p = _payload(evt)
    role = p.get("agent_role", "")
    if role:
        return role
    return _node_name(evt)


def _repo_full_name(repo_url: str) -> str:
    """Extract 'org/repo' from a GitHub URL."""
    return "/".join(repo_url.rstrip("/").split("/")[-2:])


# ---------------------------------------------------------------------------
# Per-repo summary builder
# ---------------------------------------------------------------------------

def build_repo_summary(repo_dir: Path) -> dict[str, Any] | None:
    """Build a denormalized summary for one repo result directory.

    Reads:
    - status.json  -- scan metadata (repo URL, status, entry_point, etc.)
    - events_run_1.jsonl -- raw behavioral events

    Returns None if status.json is missing or no events found.
    """
    repo_dir = Path(repo_dir)
    status_path = repo_dir / "status.json"
    if not status_path.exists():
        return None

    try:
        with open(status_path, encoding="utf-8", errors="replace") as f:
            status = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    repo_url = status.get("repo", "")
    scan_status = status.get("status", "")
    entry_point = status.get("entry_point", "")
    event_count = status.get("event_count", 0)
    duration_seconds = status.get("duration_seconds", 0)
    vllm_model = status.get("vllm_model", "")
    status_run_id = status.get("run_id", "")

    repo_full = _repo_full_name(repo_url) if repo_url else repo_dir.name

    # Load events
    events_path = repo_dir / "events_run_1.jsonl"
    events = load_events(events_path)
    if not events:
        return None

    # Extract framework from events
    framework = ""
    for evt in events:
        framework = evt.get("framework", "")
        if framework:
            break

    # ------------------------------------------------------------------
    # Parse events into agents, llm_calls, delegation_chains
    # ------------------------------------------------------------------
    agents: dict[str, dict] = {}          # keyed by node_id
    agent_order: list[str] = []
    agent_names: dict[str, str] = {}      # node_id -> display name
    llm_calls: list[dict] = []
    delegation_chains: list[dict] = []
    current_agent_key: str = ""           # track active agent for LLM attribution

    for evt in events:
        etype = evt.get("event_type", "")
        p = _payload(evt)
        nid = _node_id(evt)

        if etype == "agent.task_start":
            name = _agent_name(evt)
            key = nid or name or f"agent_{len(agents)}"
            if key not in agents:
                agents[key] = {
                    "agent_name": name or key,
                    "node_id": nid,
                    "tasks": [],
                }
                agent_order.append(key)
            if nid:
                agent_names[nid] = name or key
            current_agent_key = key

            agents[key]["tasks"].append({
                "task_description": p.get("task_description", ""),
                "agent_goal": p.get("agent_goal", ""),
                "input_source": p.get("input_source", ""),
                "status": "",
                "output_text": "",
                "output_hash": "",
                "run_id": evt.get("run_id", status_run_id),
            })

        elif etype == "agent.task_end":
            name = _agent_name(evt)
            key = nid or name
            span = agents.get(key)
            if span and span["tasks"]:
                task = span["tasks"][-1]
                task["status"] = p.get("status", "success")
                task["output_text"] = (
                    p.get("output_preview") or p.get("output_text") or ""
                )
                task["output_hash"] = p.get("output_hash", "")
            current_agent_key = ""

        elif etype == "llm.call_end":
            # Attribute to parent agent via active_node_stack,
            # fallback to currently-open agent span.
            stack = p.get("active_node_stack", [])
            parent_nid = ""
            parent_name = ""
            for stack_nid in stack:
                if stack_nid in agent_names:
                    parent_nid = stack_nid
                    parent_name = agent_names[stack_nid]
                    break
            if not parent_nid and current_agent_key:
                span = agents.get(current_agent_key)
                if span:
                    parent_nid = span["node_id"]
                    parent_name = span["agent_name"]

            llm_calls.append({
                "agent_name": parent_name,
                "node_id": parent_nid,
                "model": p.get("model_actual") or p.get("model_requested") or "",
                "output_text": (
                    p.get("output_preview") or p.get("output_text") or ""
                ),
                "output_hash": p.get("output_hash", ""),
                "input_tokens": p.get("input_tokens", 0) or 0,
                "output_tokens": p.get("output_tokens", 0) or 0,
                "latency_ms": p.get("latency_ms", 0) or 0,
                "status": "error" if p.get("error_type") else "success",
            })

        elif etype == "edge.traversed":
            src = p.get("src_node_id", "") or _node_id(evt)
            tn = evt.get("target_node")
            dst = p.get("dst_node_id", "")
            if not dst and isinstance(tn, dict):
                dst = tn.get("node_id", "")
            delegation_chains.append({
                "src_node": src,
                "dst_node": dst,
                "edge_type": (
                    p.get("edge_type") or evt.get("edge_type") or "delegation"
                ),
            })

    # Infer delegation from input_source when no explicit edge events
    if not delegation_chains and len(agent_order) >= 2:
        for i, key in enumerate(agent_order):
            if i == 0:
                continue
            span = agents[key]
            for task in span["tasks"]:
                if task.get("input_source") == "delegation":
                    prev_key = agent_order[i - 1]
                    delegation_chains.append({
                        "src_node": agents[prev_key]["node_id"],
                        "dst_node": span["node_id"],
                        "edge_type": "delegation",
                    })
                    break

    agents_list = [agents[k] for k in agent_order]

    return {
        "repo_full_name": repo_full,
        "repo_url": repo_url,
        "scan_status": scan_status,
        "framework": framework,
        "entry_point": entry_point,
        "event_count": event_count,
        "duration_seconds": duration_seconds,
        "vllm_model": vllm_model,
        "agents": agents_list,
        "llm_calls": llm_calls,
        "delegation_chains": delegation_chains,
        "execution_summary": {
            "total_agents": len(agents_list),
            "total_llm_calls": len(llm_calls),
            "total_events": len(events),
            "has_delegation": len(delegation_chains) > 0,
        },
    }


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def export_behavioral_traces(
    scan_dirs: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Export all repos across scan directories to behavioral_traces.jsonl.

    Iterates all repo result directories across *scan_dirs*, calls
    build_repo_summary for each, and writes one JSON line per repo.

    Returns export statistics dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "total_repos_scanned": 0,
        "total_repos_exported": 0,
        "skipped_count": 0,
        "total_agents": 0,
        "total_llm_calls": 0,
        "frameworks": {},
        "scan_statuses": {},
    }

    seen: set[str] = set()

    with open(output_path, "w", encoding="utf-8") as out:
        for scan_dir in scan_dirs:
            scan_dir = Path(scan_dir)
            if not scan_dir.is_dir():
                logger.warning("Scan directory does not exist: %s", scan_dir)
                continue

            for repo_dir in sorted(scan_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue
                stats["total_repos_scanned"] += 1

                summary = build_repo_summary(repo_dir)
                if summary is None:
                    stats["skipped_count"] += 1
                    continue

                # Deduplicate by repo_full_name
                key = summary["repo_full_name"]
                if key in seen:
                    continue
                seen.add(key)

                out.write(json.dumps(summary, default=str) + "\n")
                stats["total_repos_exported"] += 1

                fw = summary.get("framework") or "unknown"
                stats["frameworks"][fw] = stats["frameworks"].get(fw, 0) + 1
                st = summary.get("scan_status") or "unknown"
                stats["scan_statuses"][st] = stats["scan_statuses"].get(st, 0) + 1
                stats["total_agents"] += summary["execution_summary"]["total_agents"]
                stats["total_llm_calls"] += summary["execution_summary"]["total_llm_calls"]

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
        "--scan-dirs",
        nargs="+",
        required=True,
        help="One or more scan result directories",
    )
    parser.add_argument(
        "--output",
        default="behavioral_traces.jsonl",
        help="Output JSONL file path (default: behavioral_traces.jsonl)",
    )
    args = parser.parse_args()

    stats = export_behavioral_traces(
        [Path(d) for d in args.scan_dirs],
        Path(args.output),
    )

    print(f"\nExport complete:")
    print(f"  Repos scanned:  {stats['total_repos_scanned']}")
    print(f"  Repos exported: {stats['total_repos_exported']}")
    print(f"  Skipped:        {stats['skipped_count']}")
    print(f"  Agents:         {stats['total_agents']}")
    print(f"  LLM calls:      {stats['total_llm_calls']}")
    print(f"  Frameworks:     {stats['frameworks']}")
    print(f"  Statuses:       {stats['scan_statuses']}")


if __name__ == "__main__":
    main()
