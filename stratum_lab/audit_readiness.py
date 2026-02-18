"""Audit and compliance readiness scoring for stratum-lab.

Measures traceability, separation of duties, event completeness.
Research basis: ISACA Auditing Agentic AI, EU AI Act, NIST AI RMF.
"""
from __future__ import annotations

from stratum_lab.permissions import classify_tool_permissions

REGULATORY_FRAMEWORKS = {
    "sox": {
        "name": "Sarbanes-Oxley (SOX)",
        "requirements": ["financial_decision_traceability", "separation_of_duties", "audit_trail_completeness"],
    },
    "hipaa": {
        "name": "HIPAA",
        "requirements": ["phi_access_logging", "minimum_necessary_enforcement", "audit_trail_retention"],
    },
    "gdpr": {
        "name": "GDPR",
        "requirements": ["data_protection_by_design", "purpose_limitation", "data_subject_rights"],
    },
    "eu_ai_act": {
        "name": "EU AI Act",
        "requirements": ["risk_classification", "transparency", "human_oversight", "technical_documentation"],
    },
    "nist_ai_rmf": {
        "name": "NIST AI RMF",
        "requirements": ["governance", "risk_mapping", "measurement", "management"],
    },
}

# Finding → regulatory framework mapping
_FINDING_FRAMEWORK_MAP = {
    "STRAT-DC-001": ["sox", "eu_ai_act"],
    "STRAT-SI-001": ["sox", "nist_ai_rmf"],
    "STRAT-HC-001": ["eu_ai_act", "gdpr"],
    "STRAT-SD-001": ["eu_ai_act"],
    "STRAT-CE-001": ["sox"],
    "STRAT-SC-001": ["eu_ai_act"],
    "STRAT-TV-001": ["nist_ai_rmf"],
    "STRAT-PL-001": ["gdpr", "hipaa"],
    "STRAT-PE-001": ["sox", "nist_ai_rmf"],
    "STRAT-CR-001": [],
    "STRAT-AU-001": ["sox", "eu_ai_act", "nist_ai_rmf"],
}


def compute_event_completeness(events_by_run: dict) -> dict:
    """Measure completeness of event logging across runs."""
    task_rates: list[float] = []
    deleg_rates: list[float] = []
    llm_rates: list[float] = []
    total_orphaned_starts = 0
    total_orphaned_ends = 0

    for run_name, events in events_by_run.items():
        task_starts: dict[str, int] = {}
        task_ends: dict[str, int] = {}
        deleg_starts: dict[str, int] = {}
        deleg_ends: dict[str, int] = {}
        llm_starts: dict[str, int] = {}
        llm_ends: dict[str, int] = {}

        for evt in events:
            etype = evt.get("event_type", "")
            nid = evt.get("node_id", "")
            did = evt.get("delegation_id", "")

            if etype == "agent.task_start":
                task_starts[nid] = task_starts.get(nid, 0) + 1
            elif etype == "agent.task_end":
                task_ends[nid] = task_ends.get(nid, 0) + 1
            elif etype == "delegation.initiated":
                deleg_starts[did] = deleg_starts.get(did, 0) + 1
            elif etype == "delegation.completed":
                deleg_ends[did] = deleg_ends.get(did, 0) + 1
            elif etype == "llm.call_start":
                llm_starts[nid] = llm_starts.get(nid, 0) + 1
            elif etype == "llm.call_end":
                llm_ends[nid] = llm_ends.get(nid, 0) + 1

        # Task completion: count nodes that have both start and end
        matched_tasks = sum(1 for n in task_starts if n in task_ends)
        total_task_starts = len(task_starts)
        task_rates.append(matched_tasks / max(total_task_starts, 1))

        matched_deleg = sum(1 for d in deleg_starts if d in deleg_ends)
        total_deleg_starts = len(deleg_starts)
        deleg_rates.append(matched_deleg / max(total_deleg_starts, 1))

        matched_llm = sum(1 for n in llm_starts if n in llm_ends)
        total_llm_starts = len(llm_starts)
        llm_rates.append(matched_llm / max(total_llm_starts, 1))

        # Orphans
        total_orphaned_starts += sum(1 for n in task_starts if n not in task_ends)
        total_orphaned_starts += sum(1 for d in deleg_starts if d not in deleg_ends)
        total_orphaned_starts += sum(1 for n in llm_starts if n not in llm_ends)
        total_orphaned_ends += sum(1 for n in task_ends if n not in task_starts)
        total_orphaned_ends += sum(1 for d in deleg_ends if d not in deleg_starts)
        total_orphaned_ends += sum(1 for n in llm_ends if n not in llm_starts)

    task_rate = sum(task_rates) / max(len(task_rates), 1)
    deleg_rate = sum(deleg_rates) / max(len(deleg_rates), 1)
    llm_rate = sum(llm_rates) / max(len(llm_rates), 1)

    return {
        "task_completion_rate": round(task_rate, 3),
        "delegation_completion_rate": round(deleg_rate, 3),
        "llm_completion_rate": round(llm_rate, 3),
        "overall_completeness": round((task_rate + deleg_rate + llm_rate) / 3.0, 3),
        "orphaned_starts": total_orphaned_starts,
        "orphaned_ends": total_orphaned_ends,
    }


def assess_separation_of_duties(nodes: list[dict], edges: list[dict],
                                  tool_registrations: dict) -> dict:
    """Assess separation of duties."""
    self_review_nodes: list[str] = []
    unreviewed_write_nodes: list[str] = []
    single_points: list[str] = []

    review_keywords = ["review", "validate", "check", "verify", "audit", "approve"]

    # Build fan-in/fan-out
    fan_in: dict[str, int] = {}
    fan_out: dict[str, int] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        fan_out[src] = fan_out.get(src, 0) + 1
        fan_in[tgt] = fan_in.get(tgt, 0) + 1

    targets = set(e.get("target", "") for e in edges)

    for node in nodes:
        nid = node.get("node_id", "")
        nid_lower = nid.lower()

        # Self-review check
        has_review_keyword = any(kw in nid_lower for kw in review_keywords)
        has_output = fan_out.get(nid, 0) > 0 or fan_in.get(nid, 0) > 0
        if has_review_keyword and has_output:
            self_review_nodes.append(nid)

        # Write without approval
        tools = tool_registrations.get(nid, [])
        perms: set[str] = set()
        for tool in tools:
            perms.update(classify_tool_permissions(tool))

        has_write = "write_data" in perms or "database" in perms

        if has_write and nid in targets:
            # Check if there's a review node between source and this node
            has_reviewer = False
            for edge in edges:
                if edge.get("target") == nid:
                    src = edge.get("source", "")
                    if any(kw in src.lower() for kw in review_keywords):
                        has_reviewer = True
                        break
            if not has_reviewer:
                unreviewed_write_nodes.append(nid)

        # Single point of control
        if fan_in.get(nid, 0) == 0 and fan_out.get(nid, 0) > 0 and has_write:
            single_points.append(nid)

    score = 1.0
    if self_review_nodes:
        score -= 0.3
    score -= min(0.4, len(unreviewed_write_nodes) * 0.1)
    score -= min(0.4, len(single_points) * 0.2)
    score = max(0.0, score)

    return {
        "self_review_detected": len(self_review_nodes) > 0,
        "self_review_nodes": self_review_nodes,
        "write_without_approval": len(unreviewed_write_nodes) > 0,
        "unreviewed_write_nodes": unreviewed_write_nodes,
        "single_points_of_control": single_points,
        "separation_score": round(score, 3),
    }


def compute_decision_traceability(events_by_run: dict) -> dict:
    """Measure whether agent decisions can be reconstructed from event log."""
    routing_total = 0
    outputs_with_preview = 0
    total_decision_points = 0
    has_routing = False
    has_state_access = False

    for events in events_by_run.values():
        for evt in events:
            etype = evt.get("event_type", "")
            if etype == "routing.decision":
                routing_total += 1
                has_routing = True
                total_decision_points += 1
            elif etype == "state.access":
                has_state_access = True
            elif etype in ("agent.task_end", "llm.call_end"):
                total_decision_points += 1
                preview = evt.get("output_preview", "")
                if preview and len(preview) > 0:
                    outputs_with_preview += 1

    frac = outputs_with_preview / max(total_decision_points, 1)

    return {
        "routing_decisions_logged": routing_total,
        "outputs_with_preview": outputs_with_preview,
        "total_decision_points": total_decision_points,
        "traceability_fraction": round(frac, 3),
        "has_routing_decisions": has_routing,
        "has_state_access_logging": has_state_access,
    }


def map_findings_to_regulations(findings: list[dict]) -> list[dict]:
    """Map STRAT- findings to regulatory frameworks."""
    result: list[dict] = []
    for finding in findings:
        fid = finding.get("finding_id", "")
        severity = finding.get("severity", "medium")
        frameworks = _FINDING_FRAMEWORK_MAP.get(fid, [])

        reqs: list[str] = []
        for fw_key in frameworks:
            fw = REGULATORY_FRAMEWORKS.get(fw_key, {})
            reqs.extend(fw.get("requirements", []))

        result.append({
            "finding_id": fid,
            "severity": severity,
            "frameworks_implicated": frameworks,
            "specific_requirements": sorted(set(reqs)),
        })
    return result


def compute_audit_readiness(events_by_run: dict, nodes: list[dict], edges: list[dict],
                             tool_registrations: dict, findings: list[dict]) -> dict:
    """Top-level audit readiness computation."""
    completeness = compute_event_completeness(events_by_run)
    separation = assess_separation_of_duties(nodes, edges, tool_registrations)
    traceability = compute_decision_traceability(events_by_run)
    reg_mapping = map_findings_to_regulations(findings)

    unique_frameworks: set[str] = set()
    for m in reg_mapping:
        unique_frameworks.update(m.get("frameworks_implicated", []))

    framework_fraction = len(unique_frameworks) / max(len(REGULATORY_FRAMEWORKS), 1)

    score = (
        completeness["overall_completeness"] * 0.3
        + separation["separation_score"] * 0.3
        + traceability["traceability_fraction"] * 0.25
        + (1.0 - framework_fraction) * 0.15
    )
    score = max(0.0, min(1.0, round(score, 3)))

    return {
        "event_completeness": completeness,
        "separation_of_duties": separation,
        "decision_traceability": traceability,
        "regulatory_mapping": reg_mapping,
        "audit_readiness_score": score,
        "frameworks_at_risk": sorted(unique_frameworks),
        "finding_triggered": score < 0.5,
    }
