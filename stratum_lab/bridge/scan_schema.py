"""Transform stratum-cli scan output into stratum-lab scorer format.

stratum-cli produces ScanResult/TelemetryProfile/ScanProfile JSON.
stratum-lab's selection/scorer.py expects a dict with specific fields:
  - agent_definitions, graph_edges, taxonomy_preconditions, risk_surface
  - archetype_id, detected_entry_point, detected_requirements, etc.

TelemetryProfile is the anonymized telemetry schema — it has NO
agent_definitions, NO graph.nodes, NO repo_url, NO files list.  Instead
it carries: crew_size_distribution, graph_topology_metrics,
total_capabilities, archetype_class, selection_stratum,
deployment_signals, framework_versions, llm_providers, repo_metadata.

This module bridges the gap by extracting and reshaping fields from
whichever schema variant is present.
"""
from __future__ import annotations

from typing import Any

from stratum_lab.config import PRECONDITION_TO_FINDING

# Map archetype class strings → integer IDs used by scorer.
# These match the ARCHETYPES dict in selection/scorer.py.
_ARCHETYPE_NAME_TO_ID: dict[str, int] = {
    "single_agent_tool_use": 1,
    "sequential_pipeline": 2,
    "parallel_fan_out": 3,
    "supervisor_worker": 4,
    "debate_consensus": 5,
    "reflection_loop": 6,
    "hub_and_spoke_shared_state": 7,
    "hierarchical_delegation": 8,
    "market_auction": 9,
    "blackboard_architecture": 10,
    "human_in_the_loop": 11,
    "guardrail_gated_pipeline": 12,
}

# Reverse lookup: finding_id → precondition name
_FINDING_TO_PRECONDITION = {v: k for k, v in PRECONDITION_TO_FINDING.items()}


def _extract_graph_edges(scan: dict[str, Any]) -> list[dict[str, str]]:
    """Extract graph edges from scan result.

    Tries multiple sources in order:
    1. graph.edges (dict or list from stratum-cli graph output)
    2. agent_relationships (list of relationship dicts)

    Returns empty list for TelemetryProfile records (no graph data).
    """
    edges: list[dict[str, str]] = []

    # Source 1: graph.edges
    graph = scan.get("graph", {})
    if isinstance(graph, dict):
        raw_edges = graph.get("edges", {})
        if isinstance(raw_edges, dict):
            for edge_id, edge_data in raw_edges.items():
                if isinstance(edge_data, dict):
                    edges.append({
                        "edge_id": edge_id,
                        "source": edge_data.get("source", ""),
                        "target": edge_data.get("target", ""),
                        "edge_type": edge_data.get("edge_type", ""),
                    })
        elif isinstance(raw_edges, list):
            for edge in raw_edges:
                if isinstance(edge, dict):
                    edges.append({
                        "edge_id": edge.get("edge_id", ""),
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "edge_type": edge.get("edge_type", ""),
                    })

    # Source 2: agent_relationships (fallback)
    if not edges:
        for rel in scan.get("agent_relationships", []):
            if isinstance(rel, dict):
                edges.append({
                    "edge_id": f"{rel.get('source', '')}->{rel.get('target', '')}",
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "edge_type": rel.get("relationship_type", rel.get("edge_type", "")),
                })

    return edges


def _extract_taxonomy_preconditions(scan: dict[str, Any]) -> list[str]:
    """Derive taxonomy preconditions from findings.

    Maps finding IDs (e.g. STRAT-DC-001) back to precondition names
    (e.g. unbounded_delegation_depth).
    """
    preconditions: list[str] = []

    # Try findings list
    for finding in scan.get("findings", []):
        if isinstance(finding, dict):
            fid = finding.get("finding_id", "")
        elif isinstance(finding, str):
            fid = finding
        else:
            continue
        if fid in _FINDING_TO_PRECONDITION:
            preconditions.append(_FINDING_TO_PRECONDITION[fid])

    # Try finding_ids list (ScanProfile format)
    for fid in scan.get("finding_ids", []):
        if isinstance(fid, str) and fid in _FINDING_TO_PRECONDITION:
            name = _FINDING_TO_PRECONDITION[fid]
            if name not in preconditions:
                preconditions.append(name)

    return preconditions


def _compute_risk_surface(scan: dict[str, Any], edges: list[dict]) -> dict[str, int]:
    """Compute risk_surface metrics from graph structure.

    When full graph edges are available, analyzes them to compute
    delegation depth, shared state conflicts, feedback loops, and trust
    boundary crossings.

    When edges are absent (TelemetryProfile), falls back to
    graph_topology_metrics: max_degree → delegation depth proxy,
    clustering_coefficient → feedback loop proxy.
    """
    # ── TelemetryProfile fallback: use graph_topology_metrics ──
    if not edges:
        topo = scan.get("graph_topology_metrics", {})
        if topo:
            # clustering_coefficient > 0.3 suggests cycles / feedback
            cc = topo.get("clustering_coefficient", 0) or 0
            return {
                "max_delegation_depth": topo.get("max_degree", 0),
                "shared_state_conflict_count": 0,
                "feedback_loop_count": 1 if cc > 0.3 else 0,
                "trust_boundary_crossing_count": 0,
            }

    # ── Full graph path: analyze edges ──
    delegation_adj: dict[str, list[str]] = {}
    shared_state_nodes: set[str] = set()
    trust_crossings = 0

    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        etype = edge.get("edge_type", "")

        if etype in ("delegates_to", "delegation"):
            delegation_adj.setdefault(src, []).append(tgt)
        elif etype in ("shares_with", "reads_from", "writes_to"):
            shared_state_nodes.add(src)
            shared_state_nodes.add(tgt)
        if etype in ("sends_to",) and src.startswith("ext_"):
            trust_crossings += 1
        elif etype in ("sends_to",) and tgt.startswith("ext_"):
            trust_crossings += 1

    # Compute max delegation depth via DFS
    max_depth = 0
    for root in delegation_adj:
        stack = [(root, 0)]
        visited: set[str] = set()
        while stack:
            node, depth = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            max_depth = max(max_depth, depth)
            for child in delegation_adj.get(node, []):
                stack.append((child, depth + 1))

    # Detect feedback loops (cycles in delegation graph)
    feedback_loops = 0
    for root in delegation_adj:
        stack = [(root, {root})]
        while stack:
            node, path = stack.pop()
            for child in delegation_adj.get(node, []):
                if child in path:
                    feedback_loops += 1
                else:
                    stack.append((child, path | {child}))

    # Shared state conflicts: nodes that both read and write
    graph = scan.get("graph", {})
    graph_edges = graph.get("edges", {})
    writers: set[str] = set()
    readers: set[str] = set()
    if isinstance(graph_edges, dict):
        for edge_data in graph_edges.values():
            if isinstance(edge_data, dict):
                if edge_data.get("edge_type") == "writes_to":
                    writers.add(edge_data.get("target", ""))
                elif edge_data.get("edge_type") == "reads_from":
                    readers.add(edge_data.get("source", ""))
    shared_state_conflicts = len(writers & readers & shared_state_nodes)

    return {
        "max_delegation_depth": max_depth,
        "shared_state_conflict_count": shared_state_conflicts,
        "feedback_loop_count": feedback_loops,
        "trust_boundary_crossing_count": trust_crossings,
    }


def _resolve_archetype_id(scan: dict[str, Any]) -> int:
    """Map archetype string to integer ID.

    For TelemetryProfile, reads archetype_class directly.
    Falls back to graph heuristics or crew_size_distribution.
    """
    # Try archetype_id directly
    aid = scan.get("archetype_id")
    if isinstance(aid, int) and aid > 0:
        return aid

    # Try archetype string fields (archetype_class from TelemetryProfile)
    for key in ("archetype", "archetype_class"):
        name = scan.get(key, "")
        if isinstance(name, str) and name in _ARCHETYPE_NAME_TO_ID:
            return _ARCHETYPE_NAME_TO_ID[name]

    # Heuristic from graph structure
    graph = scan.get("graph", {})
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
    agents = [n for n in (nodes.values() if isinstance(nodes, dict) else nodes)
              if isinstance(n, dict) and n.get("node_type") in ("agent", "Agent")]

    if agents:
        if len(agents) <= 1:
            return 1  # single_agent_tool_use
        return 2  # sequential_pipeline (safe default for multi-agent)

    # Heuristic from crew_size_distribution (TelemetryProfile)
    crew_sizes = scan.get("crew_size_distribution", [])
    if crew_sizes:
        total_agents = sum(crew_sizes)
        if total_agents <= 1:
            return 1  # single_agent_tool_use
        return 2  # sequential_pipeline

    return 1  # single_agent_tool_use (default)


def adapt_scan_result(scan: dict[str, Any]) -> dict[str, Any]:
    """Transform a stratum-cli scan result dict into scorer-expected format.

    Handles both full ScanResult (with graph, agent_definitions, files) and
    TelemetryProfile (anonymized: crew_size_distribution,
    graph_topology_metrics, deployment_signals, framework_versions, etc.).

    The returned dict can be passed directly to selection/scorer.py functions.
    """
    edges = _extract_graph_edges(scan)
    preconditions = _extract_taxonomy_preconditions(scan)
    risk_surface = _compute_risk_surface(scan, edges)
    archetype_id = _resolve_archetype_id(scan)

    # ── Agent definitions ──
    # Priority: agent_definitions → graph.nodes → crew_size_distribution
    agent_defs = scan.get("agent_definitions", [])
    total_capabilities = scan.get("total_capabilities", 0)

    if not agent_defs:
        graph = scan.get("graph", {})
        nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
        for nid, ndata in (nodes.items() if isinstance(nodes, dict) else []):
            if isinstance(ndata, dict) and ndata.get("node_type") in ("agent", "Agent"):
                agent_defs.append({
                    "name": ndata.get("node_name", nid),
                    "tool_names": ndata.get("tool_names", []),
                })

    if not agent_defs:
        # TelemetryProfile: crew_size_distribution → synthetic agent defs
        crew_sizes = scan.get("crew_size_distribution", [])
        if crew_sizes:
            total_agents = sum(crew_sizes)
            # Distribute total_capabilities across agents as synthetic tools
            caps_per_agent = (
                total_capabilities // total_agents if total_agents > 0 else 0
            )
            remainder = (
                total_capabilities % total_agents if total_agents > 0 else 0
            )
            idx = 0
            for crew_idx, crew_size in enumerate(crew_sizes):
                for agent_i in range(crew_size):
                    n_tools = caps_per_agent + (1 if idx < remainder else 0)
                    tool_names = [f"cap_{j}" for j in range(idx, idx + n_tools)]
                    agent_defs.append({
                        "name": f"agent_{idx}",
                        "tool_names": tool_names,
                    })
                    idx += 1

    # ── Frameworks ──
    # Priority: detected_frameworks → frameworks → framework_versions keys
    #         → repo_metadata.primary_framework → primary_framework
    detected_frameworks = scan.get("detected_frameworks", [])
    if not detected_frameworks:
        detected_frameworks = scan.get("frameworks", [])
    if not detected_frameworks:
        fw_versions = scan.get("framework_versions", {})
        if isinstance(fw_versions, dict) and fw_versions:
            detected_frameworks = list(fw_versions.keys())
    if not detected_frameworks:
        repo_meta = scan.get("repo_metadata", {})
        if isinstance(repo_meta, dict):
            primary = repo_meta.get("primary_framework", "")
            if primary:
                detected_frameworks = [primary]
    if not detected_frameworks:
        primary = scan.get("primary_framework", "")
        if primary:
            detected_frameworks = [primary]

    # ── Entry point detection ──
    metadata = scan.get("metadata", {})
    files = scan.get("files", [])
    deploy = scan.get("deployment_signals", {}) or {}

    detected_entry_point = scan.get("detected_entry_point", "")
    if not detected_entry_point and metadata.get("has_entry_point"):
        detected_entry_point = "detected"
    if not detected_entry_point:
        for f in files:
            if isinstance(f, str) and f.endswith(("main.py", "app.py", "__main__.py")):
                detected_entry_point = f
                break
    # TelemetryProfile: deployment_signals.has_lockfile as runnability proxy
    if not detected_entry_point and deploy.get("has_lockfile"):
        detected_entry_point = "inferred"

    # ── Requirements detection ──
    detected_requirements = scan.get("detected_requirements", "")
    if not detected_requirements and metadata.get("has_requirements"):
        detected_requirements = "requirements.txt"
    if not detected_requirements:
        for f in files:
            if isinstance(f, str) and f.endswith(("requirements.txt", "pyproject.toml")):
                detected_requirements = f
                break
    # TelemetryProfile: deployment_signals.has_lockfile
    if not detected_requirements and deploy.get("has_lockfile"):
        detected_requirements = "lockfile"

    # ── repo_url: construct from repo_full_name if missing ──
    repo_url = scan.get("repo_url", scan.get("url", ""))
    if not repo_url:
        rfn = scan.get("repo_full_name", "")
        if rfn:
            repo_url = f"https://github.com/{rfn}"

    # ── Runnability signals from deployment_signals ──
    requires_docker = scan.get("requires_docker", False)
    if not requires_docker and deploy.get("has_dockerfile"):
        requires_docker = True

    return {
        # Pass-through fields
        "repo_id": scan.get("repo_id", scan.get("repo_full_name", "")),
        "repo_url": repo_url,

        # Reshaped fields
        "agent_definitions": agent_defs,
        "graph_edges": edges,
        "taxonomy_preconditions": preconditions,
        "risk_surface": risk_surface,
        "archetype_id": archetype_id,
        "detected_frameworks": detected_frameworks,
        "detected_entry_point": detected_entry_point,
        "detected_requirements": detected_requirements,

        # Runnability signals — pass through or derive
        "has_readme_with_usage": scan.get("has_readme_with_usage", False),
        "requires_docker": requires_docker,
        "requires_database": scan.get("requires_database", False),
        "recent_commit": scan.get("recent_commit", False),

        # Schema-required fields (for selection/schema.py validation)
        "repo_full_name": scan.get("repo_full_name", scan.get("repo_id", "")),
        "graph": scan.get("graph", {}),
        "findings": scan.get("findings", []),
        "control_inventory": scan.get("control_inventory", {
            "present_controls": [],
            "absent_controls": [],
        }),
        "framework": (detected_frameworks[0].lower() if detected_frameworks
                      else scan.get("primary_framework", "custom")),
        "agent_count": len(agent_defs),
        "edge_count": len(edges),

        # TelemetryProfile pass-through fields
        "total_capabilities": total_capabilities,
        "selection_stratum": scan.get("selection_stratum", ""),
    }
