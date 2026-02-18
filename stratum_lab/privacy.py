"""Privacy topology analysis for stratum-lab.

Detects cross-domain data convergence, information fan-in, and compositional
privacy exposure at each node. Research basis: TOP-R, Compositional Privacy
Leakage, AgentLeak.
"""
from __future__ import annotations

DATA_DOMAINS = [
    "personal_identifiable",
    "financial",
    "health_medical",
    "credentials_secrets",
    "behavioral_preference",
    "organizational_internal",
    "communication_content",
    "location_temporal",
    "generic",
]

DOMAIN_KEYWORDS = {
    "personal_identifiable": ["user", "name", "email", "address", "phone", "ssn", "profile", "identity", "person", "contact", "customer"],
    "financial": ["payment", "transaction", "account", "balance", "credit", "invoice", "price", "billing", "finance", "bank", "salary", "compensation"],
    "health_medical": ["health", "medical", "diagnosis", "patient", "prescription", "clinical", "symptom", "vitals", "hipaa", "phi"],
    "credentials_secrets": ["api_key", "password", "token", "secret", "credential", "certificate", "auth", "key", "oauth"],
    "behavioral_preference": ["history", "preference", "browsing", "purchase", "behavior", "analytics", "tracking", "cookie"],
    "organizational_internal": ["employee", "hr", "strategy", "internal", "confidential", "proprietary", "memo", "policy", "org"],
    "communication_content": ["email", "message", "chat", "slack", "notification", "inbox", "mail", "correspondence"],
    "location_temporal": ["location", "gps", "calendar", "schedule", "timezone", "meeting", "appointment", "geo"],
}


def classify_data_domain(text: str) -> list[str]:
    """Classify a text string into data domains via keyword substring matching."""
    lower = text.lower()
    matched: list[str] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            matched.append(domain)
    return sorted(set(matched)) if matched else ["generic"]


def compute_node_data_domains(node_id: str, tool_names: list[str], state_keys: list[str]) -> list[str]:
    """Union of data domains from all tools and state keys for a node."""
    domains: set[str] = set()
    for tool in tool_names:
        domains.update(classify_data_domain(tool))
    for key in state_keys:
        domains.update(classify_data_domain(key))
    result = sorted(domains)
    return result if result else ["generic"]


def compute_information_fan_in(node_id: str, edges: list[dict], node_domains: dict[str, list[str]]) -> dict:
    """Compute information convergence at a node."""
    incoming = [e for e in edges if e.get("target") == node_id]
    source_nodes = [e.get("source", "") for e in incoming]
    converging: set[str] = set()
    for src in source_nodes:
        converging.update(node_domains.get(src, ["generic"]))

    converging_list = sorted(converging)
    domain_count = len(converging_list)
    heterogeneous = domain_count >= 2 and converging_list != ["generic"]

    if domain_count >= 3:
        risk = "high"
    elif domain_count == 2:
        risk = "medium"
    else:
        risk = "low"

    return {
        "node_id": node_id,
        "incoming_edge_count": len(incoming),
        "source_nodes": source_nodes,
        "converging_domains": converging_list,
        "domain_count": domain_count,
        "heterogeneous": heterogeneous,
        "fan_in_risk": risk,
    }


def detect_cross_domain_flows(edges: list[dict], node_domains: dict[str, list[str]]) -> list[dict]:
    """Detect edges that cross data domain boundaries."""
    flows: list[dict] = []
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        src_domains = set(node_domains.get(src, ["generic"]))
        tgt_domains = set(node_domains.get(tgt, ["generic"]))
        new_at_target = sorted(tgt_domains - src_domains)

        if new_at_target:
            flows.append({
                "edge_id": edge.get("edge_id", f"{src}->{tgt}"),
                "source_node": src,
                "target_node": tgt,
                "source_domains": sorted(src_domains),
                "target_domains": sorted(tgt_domains),
                "new_domains_at_target": new_at_target,
                "cross_domain": True,
                "risk_note": f"Data flows from {sorted(src_domains)} to node handling {sorted(tgt_domains)}",
            })
    return flows


def compute_privacy_topology(nodes: list[dict], edges: list[dict],
                              tool_registrations: dict, state_access_events: list[dict]) -> dict:
    """Top-level privacy topology computation."""
    # Step 1: collect per-node data
    node_tool_names: dict[str, list[str]] = {}
    node_state_keys: dict[str, set[str]] = {}

    for node in nodes:
        nid = node.get("node_id", "")
        node_tool_names[nid] = tool_registrations.get(nid, [])
        node_state_keys[nid] = set()

    for evt in state_access_events:
        accessor = evt.get("accessor_node", "")
        key = evt.get("state_key", "")
        if accessor and key:
            if accessor not in node_state_keys:
                node_state_keys[accessor] = set()
            node_state_keys[accessor].add(key)

    # Step 2: compute domains per node
    node_domains: dict[str, list[str]] = {}
    for node in nodes:
        nid = node.get("node_id", "")
        tools = node_tool_names.get(nid, [])
        keys = sorted(node_state_keys.get(nid, set()))
        node_domains[nid] = compute_node_data_domains(nid, tools, keys)

    # Step 3: fan-in analysis
    fan_in_results = []
    for node in nodes:
        nid = node.get("node_id", "")
        fi = compute_information_fan_in(nid, edges, node_domains)
        if fi["fan_in_risk"] != "low":
            fan_in_results.append(fi)

    # Step 4: cross-domain flows
    cross_flows = detect_cross_domain_flows(edges, node_domains)

    # Step 5: scoring
    high_fan_in = sum(1 for fi in fan_in_results if fi["fan_in_risk"] == "high")
    cross_count = len(cross_flows)
    all_fan_in = [compute_information_fan_in(n.get("node_id", ""), edges, node_domains) for n in nodes]
    max_convergence = max((fi["domain_count"] for fi in all_fan_in), default=0)
    exposure = min(1.0, high_fan_in * 0.2 + cross_count * 0.1)

    return {
        "node_domains": node_domains,
        "fan_in_analysis": fan_in_results,
        "cross_domain_flows": cross_flows,
        "privacy_exposure_score": round(exposure, 3),
        "max_domain_convergence": max_convergence,
        "nodes_with_high_fan_in": high_fan_in,
        "cross_domain_edge_count": cross_count,
        "finding_triggered": exposure > 0.3,
    }
