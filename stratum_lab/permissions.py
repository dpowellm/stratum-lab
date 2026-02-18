"""Permission blast radius computation for stratum-lab.

Transitive closure of effective permissions through delegation graph.
Research basis: MAC for Agent Systems, AgenTRIM, AgentBound.
"""
from __future__ import annotations

PERMISSION_CATEGORIES = [
    "read_data",
    "write_data",
    "execute_code",
    "network_access",
    "file_system",
    "database",
    "api_external",
    "user_communication",
    "system_admin",
]

TOOL_PERMISSION_MAP = {
    "read_data": ["search", "query", "fetch", "get", "read", "list", "lookup", "retrieve", "find"],
    "write_data": ["write", "update", "create", "insert", "put", "save", "store", "modify", "edit", "set"],
    "execute_code": ["execute", "run", "eval", "exec", "shell", "command", "script", "python", "code"],
    "network_access": ["http", "request", "api", "url", "web", "browse", "scrape", "download", "curl"],
    "file_system": ["file", "directory", "path", "folder", "move", "copy", "delete", "rename", "open"],
    "database": ["sql", "db", "database", "table", "query", "select", "insert", "mongo", "redis", "postgres"],
    "api_external": ["api", "endpoint", "service", "webhook", "oauth", "integration", "third_party"],
    "user_communication": ["email", "send", "notify", "message", "slack", "sms", "chat", "alert"],
    "system_admin": ["admin", "sudo", "root", "deploy", "restart", "config", "permission", "role", "grant"],
}


def classify_tool_permissions(tool_name: str) -> list[str]:
    """Classify a tool name into permission categories."""
    lower = tool_name.lower()
    perms: set[str] = set()
    for category, keywords in TOOL_PERMISSION_MAP.items():
        if any(kw in lower for kw in keywords):
            perms.add(category)
    return sorted(perms) if perms else ["read_data"]


def build_direct_permissions(tool_registrations: dict[str, list[str]]) -> dict[str, list[str]]:
    """For each node, classify all its registered tools and union permissions."""
    result: dict[str, list[str]] = {}
    for node_id, tools in tool_registrations.items():
        perms: set[str] = set()
        for tool in tools:
            perms.update(classify_tool_permissions(tool))
        result[node_id] = sorted(perms)
    return result


def compute_transitive_permissions(direct_permissions: dict[str, list[str]],
                                    edges: list[dict]) -> dict[str, dict]:
    """Compute effective permissions via BFS transitive closure."""
    # Build adjacency (outgoing delegation edges)
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(tgt)

    result: dict[str, dict] = {}
    for node_id in direct_permissions:
        direct = set(direct_permissions.get(node_id, []))

        # BFS to find all reachable nodes
        visited: set[str] = set()
        queue = list(adjacency.get(node_id, []))
        paths: dict[str, list[str]] = {}  # perm -> path

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Union permissions from all reachable nodes
        effective = set(direct)
        for reachable in visited:
            reachable_perms = set(direct_permissions.get(reachable, []))
            for perm in reachable_perms:
                if perm not in direct:
                    if perm not in paths:
                        paths[perm] = [node_id, reachable]
                effective.update(reachable_perms)

        escalated = sorted(effective - direct)
        escalation_path = {p: paths.get(p, []) for p in escalated}

        result[node_id] = {
            "direct_permissions": sorted(direct),
            "effective_permissions": sorted(effective),
            "escalated_permissions": escalated,
            "escalation_path": escalation_path,
            "blast_radius": len(effective),
            "escalation_count": len(escalated),
        }

    return result


def find_permission_asymmetries(transitive_perms: dict[str, dict],
                                 edges: list[dict]) -> list[dict]:
    """Find edges where delegation introduces permission escalation."""
    asymmetries: list[dict] = []
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        src_direct = set(transitive_perms.get(src, {}).get("direct_permissions", []))
        tgt_direct = set(transitive_perms.get(tgt, {}).get("direct_permissions", []))
        escalated = sorted(tgt_direct - src_direct)

        if not escalated:
            continue

        if "system_admin" in escalated or "execute_code" in escalated:
            risk = "critical"
        elif "write_data" in escalated or "database" in escalated:
            risk = "high"
        else:
            risk = "medium"

        asymmetries.append({
            "edge_id": edge.get("edge_id", f"{src}->{tgt}"),
            "source_node": src,
            "target_node": tgt,
            "source_direct": sorted(src_direct),
            "target_direct": sorted(tgt_direct),
            "escalated_via_delegation": escalated,
            "risk_level": risk,
        })

    return asymmetries


def compute_unused_permissions(direct_permissions: dict[str, list[str]],
                                runtime_tool_calls: dict[str, list[str]]) -> dict[str, dict]:
    """Detect over-provisioned agents."""
    # We need the original tool registrations to compare, but we only have
    # permissions. Use runtime_tool_calls to compare against registered tools.
    # Since we don't have the original tool list, we'll work with what we have.
    result: dict[str, dict] = {}
    for node_id in direct_permissions:
        registered = set(direct_permissions.get(node_id, []))
        used_tools = runtime_tool_calls.get(node_id, [])
        used_perms: set[str] = set()
        for tool in used_tools:
            used_perms.update(classify_tool_permissions(tool))

        unused_perms = sorted(registered - used_perms)
        utilization = len(used_perms & registered) / max(len(registered), 1)

        result[node_id] = {
            "registered_tools": sorted(registered),
            "used_tools": used_tools,
            "unused_tools": sorted(set(used_tools) - set(used_tools)),  # placeholder
            "utilization_rate": round(utilization, 3),
            "unused_permission_categories": unused_perms,
        }

    return result


def compute_permission_blast_radius(nodes: list[dict], edges: list[dict],
                                     tool_registrations: dict, runtime_tool_calls: dict) -> dict:
    """Top-level permission blast radius computation."""
    direct = build_direct_permissions(tool_registrations)
    transitive = compute_transitive_permissions(direct, edges)
    asymmetries = find_permission_asymmetries(transitive, edges)
    unused = compute_unused_permissions(direct, runtime_tool_calls)

    max_blast = max((t["blast_radius"] for t in transitive.values()), default=0)
    total_esc = sum(t["escalation_count"] for t in transitive.values())
    critical = sum(1 for a in asymmetries if a["risk_level"] == "critical")
    util_rates = [u["utilization_rate"] for u in unused.values()]
    mean_util = round(sum(util_rates) / max(len(util_rates), 1), 3)
    risk_score = min(1.0, critical * 0.3 + total_esc * 0.05)

    return {
        "direct_permissions": {n: direct.get(n, []) for n in direct},
        "transitive_permissions": transitive,
        "asymmetries": asymmetries,
        "unused_permissions": unused,
        "max_blast_radius": max_blast,
        "total_escalation_count": total_esc,
        "critical_asymmetries": critical,
        "mean_utilization_rate": mean_util,
        "permission_risk_score": round(risk_score, 3),
        "finding_triggered": risk_score > 0.3,
    }
