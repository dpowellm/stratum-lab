"""Risk predictor — produces risk predictions from matched patterns.

Combines structural graph analysis with behavioral pattern data from the
knowledge base to predict which taxonomy preconditions are likely to
manifest as real failures, how severe they would be, and what remediation
steps are recommended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stratum_lab.query.matcher import Match


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PredictedRisk:
    """A single predicted risk tied to a taxonomy precondition."""

    precondition_id: str
    precondition_name: str
    structural_evidence: list[str]
    manifestation_probability: float
    confidence_interval: tuple[float, float]
    sample_size: int
    severity_when_manifested: str
    behavioral_description: str
    similar_repo_outcomes: list[dict] = field(default_factory=list)
    fragility_flag: bool = False
    remediation: str = ""


@dataclass
class RiskPrediction:
    """Aggregate risk prediction for a structural graph."""

    graph_fingerprint: dict
    archetype: str
    archetype_prevalence: float
    overall_risk_score: float          # 0-100
    predicted_risks: list[PredictedRisk]
    structural_only_risks: list[dict] = field(default_factory=list)
    positive_signals: list[str] = field(default_factory=list)
    framework_comparison: dict = field(default_factory=dict)
    dataset_coverage: dict = field(default_factory=dict)
    semantic_analysis: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity weights (used for overall_risk_score)
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT: dict[str, int] = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0,
}


# ---------------------------------------------------------------------------
# Remediation suggestions keyed by taxonomy precondition ID
# ---------------------------------------------------------------------------

REMEDIATIONS: dict[str, str] = {
    "shared_state_no_arbitration": (
        "Add write arbitration (mutex, queue, or optimistic locking) to every "
        "shared data store that receives writes from multiple agents. Consider "
        "an event-sourced architecture where each agent appends events rather "
        "than mutating state directly."
    ),
    "unbounded_delegation_depth": (
        "Add depth limits or human checkpoints at delegation boundaries. "
        "Enforce a maximum chain length (e.g., 3-4 hops) and require "
        "explicit approval before delegating beyond the limit."
    ),
    "no_timeout_on_delegation": (
        "Configure explicit timeouts on every delegation edge. Use a "
        "circuit-breaker pattern so that a stalled downstream agent does not "
        "block the entire pipeline indefinitely."
    ),
    "no_output_validation": (
        "Add output schema validation or a guardrail node between the "
        "producing agent and any downstream consumer. Validate structure, "
        "type, and semantic constraints before propagation."
    ),
    "single_point_of_failure": (
        "Introduce redundancy for the hub agent (active-passive or "
        "load-balanced). Add health-check probes and automatic failover. "
        "Consider distributing responsibilities across multiple coordinators."
    ),
    "implicit_ordering_dependency": (
        "Make ordering dependencies explicit with sequencing edges or "
        "message queues that enforce FIFO delivery. Add idempotency tokens "
        "so that out-of-order processing is detected and corrected."
    ),
    "unhandled_tool_failure": (
        "Wrap every tool call in a try/catch with a defined fallback "
        "strategy (retry with backoff, degrade gracefully, or escalate "
        "to a human). Log tool failures for observability."
    ),
    "shared_tool_no_concurrency_control": (
        "Add concurrency control (semaphore, token bucket, or connection "
        "pooling) to shared tool endpoints. Ensure agents cannot exhaust "
        "shared resources under load."
    ),
    "no_fallback_for_external": (
        "Implement a fallback path for every external service dependency. "
        "Use caching, default responses, or alternative providers so that "
        "external outages do not cascade into agent failures."
    ),
    "circular_delegation": (
        "Break delegation cycles by introducing a termination condition or "
        "a maximum iteration count. Add cycle detection in the orchestrator "
        "so that circular delegation is surfaced as an error."
    ),
    "no_rate_limiting": (
        "Add rate limiting on agent-to-agent and agent-to-tool edges. "
        "Use token-bucket or leaky-bucket algorithms to prevent runaway "
        "loops from overwhelming downstream services."
    ),
    "data_store_no_schema_enforcement": (
        "Enforce a schema (JSON Schema, Pydantic model, or equivalent) on "
        "every data store. Reject writes that do not conform and surface "
        "schema violations as structured errors."
    ),
    "trust_boundary_no_sanitization": (
        "Add input sanitization and output encoding at every trust boundary "
        "crossing. Validate data from external services before passing it "
        "to internal agents."
    ),
    "no_error_propagation_strategy": (
        "Define an explicit error propagation strategy for every edge. "
        "Decide per-edge whether errors should propagate, be swallowed with "
        "a default, or trigger a retry. Document the strategy in the graph."
    ),
    "capability_overlap_no_priority": (
        "Assign explicit priority or routing rules when multiple agents "
        "share overlapping capabilities. Use a capability registry with "
        "priority scores to resolve conflicts deterministically."
    ),
    "no_guardrail_on_output": (
        "Add a guardrail node that inspects agent outputs before they "
        "reach end-users or irreversible actions. Filter for harmful, "
        "off-topic, or structurally invalid outputs."
    ),
    "unbounded_iteration_loop": (
        "Set a hard upper bound on iteration count for any looping agent. "
        "Add a convergence check or progress metric so that the loop "
        "terminates when no further progress is made."
    ),
    "no_idempotency_guarantee": (
        "Make tool calls and state mutations idempotent by using unique "
        "request IDs and deduplication logic. This prevents duplicate "
        "side-effects when retries or replays occur."
    ),
    "missing_input_validation": (
        "Validate all inputs at the agent boundary using schema checks, "
        "type assertions, and range constraints. Reject malformed inputs "
        "early rather than propagating garbage downstream."
    ),
    "no_observability_hooks": (
        "Add structured logging, tracing (OpenTelemetry spans), and "
        "metrics emission at every agent and edge. Without observability, "
        "failures are invisible in production."
    ),
    "hardcoded_model_dependency": (
        "Abstract the model behind a provider interface so that the model "
        "can be swapped, versioned, or rate-limited independently. Add "
        "fallback models for degraded-quality operation."
    ),
    "no_graceful_degradation": (
        "Design each agent to operate in a degraded mode when dependencies "
        "are unavailable. Return partial results with quality indicators "
        "rather than failing completely."
    ),
    "unprotected_state_mutation": (
        "Wrap state mutations in transactions or use compare-and-swap "
        "semantics. Ensure that concurrent mutations cannot leave the "
        "data store in an inconsistent state."
    ),
    "no_consensus_mechanism": (
        "Add a consensus protocol (voting, quorum, or leader-election) "
        "when multiple agents must agree on a shared decision. Without "
        "consensus, conflicting decisions can propagate simultaneously."
    ),
    "missing_capability_for_task": (
        "Ensure every delegated task has at least one agent with the "
        "required capability. Add capability checks before delegation "
        "and surface missing capabilities as configuration errors."
    ),
    "over_permissioned_agent": (
        "Apply the principle of least privilege: restrict each agent's "
        "tool access and data store permissions to only what is needed "
        "for its specific role."
    ),
    "no_audit_trail": (
        "Add an append-only audit log that records every agent action, "
        "decision, and state mutation with timestamps and actor IDs. "
        "This is critical for debugging and compliance."
    ),
    "unencrypted_data_in_transit": (
        "Encrypt all inter-agent and agent-to-service communication "
        "using TLS. Ensure that sensitive payloads are never sent in "
        "plaintext, especially across trust boundaries."
    ),
    "no_resource_limits": (
        "Set resource limits (CPU, memory, token budget, time budget) "
        "on every agent. Use cgroups, container limits, or application-"
        "level budgets to prevent runaway resource consumption."
    ),
    "missing_retry_strategy": (
        "Add a structured retry strategy (exponential backoff with jitter) "
        "for transient failures on tool calls and external service "
        "interactions. Cap the maximum number of retries."
    ),
    "no_dead_letter_queue": (
        "Route permanently failed messages to a dead-letter queue for "
        "later inspection and manual remediation. This prevents message "
        "loss and enables post-mortem analysis."
    ),
    "no_circuit_breaker": (
        "Add a circuit breaker on external service calls and inter-agent "
        "delegation edges. Open the circuit after consecutive failures "
        "and probe periodically to check recovery."
    ),
}


# ---------------------------------------------------------------------------
# Positive-signal detection
# ---------------------------------------------------------------------------


def _delegation_depth(graph: dict) -> int:
    """Compute max delegation depth from graph edges (BFS)."""
    edges = graph.get("edges", {})
    adj: dict[str, list[str]] = {}
    targets: set[str] = set()
    for e in edges.values():
        s = e.get("structural", e)
        if s.get("edge_type") == "delegates_to":
            src = s.get("source", "")
            tgt = s.get("target", "")
            if src and tgt:
                adj.setdefault(src, []).append(tgt)
                targets.add(tgt)
    if not adj:
        return 0
    roots = set(adj.keys()) - targets
    if not roots:
        roots = set(adj.keys())
    max_depth = 0
    for root in roots:
        visited = {root}
        queue = [(root, 1)]
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for t in adj.get(node, []):
                if t not in visited:
                    visited.add(t)
                    queue.append((t, depth + 1))
    return max_depth


def _shared_state_writer_count(graph: dict) -> int:
    """Count agents writing to data stores with 2+ writers."""
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    data_stores = {
        nid for nid, n in nodes.items()
        if n.get("structural", n).get("node_type") == "data_store"
    }
    writers_per_store: dict[str, set[str]] = {}
    for e in edges.values():
        s = e.get("structural", e)
        if s.get("edge_type") == "writes_to":
            tgt = s.get("target", "")
            src = s.get("source", "")
            if tgt in data_stores and src:
                writers_per_store.setdefault(tgt, set()).add(src)
    total = 0
    for writers in writers_per_store.values():
        if len(writers) >= 2:
            total += len(writers)
    return total


def _all_externals_have_error_boundary(graph: dict) -> bool:
    """Check if every external_service node has an upstream error boundary."""
    ext_nodes = [
        nid for nid, n in graph.get("nodes", {}).items()
        if n.get("structural", n).get("node_type") == "external"
    ]
    if not ext_nodes:
        return False  # no externals = nothing to report
    edges = graph.get("edges", {})
    for ext_id in ext_nodes:
        incoming = [
            e for e in edges.values()
            if (e.get("structural", e).get("target") == ext_id)
        ]
        if not any(
            graph["nodes"].get(e.get("structural", e).get("source", ""), {})
            .get("structural", graph["nodes"].get(e.get("structural", e).get("source", ""), {}))
            .get("error_handling")
            for e in incoming
        ):
            return False
    return True


# Structural features that indicate good engineering practices.
# Each signal has a ``detect`` lambda that inspects the actual structural
# graph and returns ``True`` only when the structural evidence is present.
_POSITIVE_INDICATORS: list[dict[str, Any]] = [
    {
        "signal": "Guardrails present — output filtering reduces risk of harmful or malformed agent outputs",
        "detect": lambda g: any(
            n.get("structural", n).get("node_type") == "guardrail"
            for n in g.get("nodes", {}).values()
        ),
    },
    {
        "signal": "Human checkpoint configured — human-in-the-loop prevents fully autonomous irreversible actions",
        "detect": lambda g: any(
            n.get("structural", n).get("node_type") in ("human_checkpoint", "approval_gate")
            or "human" in n.get("structural", n).get("name", "").lower()
            for n in g.get("nodes", {}).values()
        ) or g.get("structural_metrics", {}).get("has_human_checkpoint", 0) == 1,
    },
    {
        "signal": "Observability sinks present — agent behavior is being logged for monitoring",
        "detect": lambda g: (
            any(
                n.get("structural", n).get("node_type") in ("observability", "logger", "tracer")
                for n in g.get("nodes", {}).values()
            )
            or any(
                e.get("structural", e).get("edge_type") in ("logs_to", "traces_to", "monitors", "feeds_into")
                for e in g.get("edges", {}).values()
            )
        ),
    },
    {
        "signal": "Shallow delegation chain — delegation depth ≤2 limits cascading failure risk",
        "detect": lambda g: _delegation_depth(g) <= 2,
    },
    {
        "signal": "No shared mutable state — agents use isolated data stores, eliminating write contention",
        "detect": lambda g: _shared_state_writer_count(g) == 0,
    },
    {
        "signal": "Single-framework architecture — consistent error handling and lifecycle management",
        "detect": lambda g: len(set(
            n.get("structural", n).get("framework", "")
            for n in g.get("nodes", {}).values()
            if n.get("structural", n).get("node_type") == "agent"
        ) - {""}) <= 1,
    },
    {
        "signal": "All external calls guarded — external service dependencies have error boundaries",
        "detect": _all_externals_have_error_boundary,
    },
    {
        "signal": "Low topology complexity — simple graph structure reduces emergent interaction risk",
        "detect": lambda g: len(g.get("edges", {})) <= 2 * len(g.get("nodes", {})),
    },
]


# ---------------------------------------------------------------------------
# Semantic risk scoring (Issue 16)
# ---------------------------------------------------------------------------

def _compute_semantic_risk(structural_graph: dict) -> dict[str, Any]:
    """Compute semantic cascade risk from structural graph and semantic lineage."""
    semantic = structural_graph.get("semantic_lineage", {})

    unvalidated = semantic.get("unvalidated_fraction", 1.0)
    chain_depth = semantic.get("semantic_chain_depth", 0)
    max_blast = semantic.get("max_blast_radius", 0)
    class_count = semantic.get("classification_injection_count", 0)

    score = (
        unvalidated * 30
        + min(chain_depth / 5.0, 1.0) * 25
        + min(max_blast / 4.0, 1.0) * 20
        + min(class_count / 3.0, 1.0) * 25
    )

    det = semantic.get("semantic_determinism", {})
    nondet = [
        n for n, m in det.items()
        if not m.get("semantically_deterministic", True)
    ]

    return {
        "semantic_risk_score": round(score, 1),
        "unvalidated_handoff_fraction": unvalidated,
        "semantic_chain_depth": chain_depth,
        "max_blast_radius": max_blast,
        "classification_injection_points": class_count,
        "nondeterministic_nodes": nondet,
        "verdict": _semantic_verdict(score, chain_depth, class_count),
    }


def _semantic_verdict(score: float, depth: int, class_count: int) -> str:
    """Generate a human-readable verdict for semantic risk."""
    if score >= 60 and class_count > 0:
        return (
            f"HIGH SEMANTIC RISK: Classification decisions propagate through "
            f"{depth}-deep agent chain with no validation. A single "
            f"misclassification cascades unchecked to the final output."
        )
    elif score >= 40:
        return (
            "MODERATE SEMANTIC RISK: Unvalidated handoffs where upstream "
            "output becomes downstream context without checking."
        )
    elif score >= 20:
        return "LOW SEMANTIC RISK: Some unvalidated handoffs but limited chain depth."
    else:
        return "MINIMAL SEMANTIC RISK: Handoffs are validated or chain is shallow."


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_positive_signals(structural_graph: dict) -> list[str]:
    """Scan the structural graph for indicators of good practices."""
    signals: list[str] = []
    for indicator in _POSITIVE_INDICATORS:
        try:
            if indicator["detect"](structural_graph):
                signals.append(indicator["signal"])
        except Exception:
            continue
    return signals


def _classify_archetype(fingerprint: dict) -> tuple[str, float]:
    """Classify the fingerprint archetype and compute prevalence.

    Returns ``(archetype_name, prevalence_placeholder)``.  True prevalence
    is computed after checking the knowledge base.
    """
    motifs = fingerprint.get("motifs", [])
    motif_names: set[str] = set()
    if isinstance(motifs, list):
        for m in motifs:
            if isinstance(m, str):
                motif_names.add(m)
            elif isinstance(m, dict):
                motif_names.add(m.get("motif_name", ""))

    # Combined motif rules (checked first)
    if "linear_delegation_chain" in motif_names and "shared_state_without_arbitration" in motif_names:
        return ("hierarchical_delegation", 0.0)
    if "hub_and_spoke" in motif_names and "shared_state_without_arbitration" in motif_names:
        return ("hub_and_spoke_shared_state", 0.0)

    # Single motif rules
    if "hub_and_spoke" in motif_names:
        return ("hub_and_spoke", 0.0)
    if "linear_delegation_chain" in motif_names:
        return ("linear_pipeline", 0.0)
    if "feedback_loop" in motif_names:
        return ("feedback_loop_system", 0.0)
    if "shared_state_without_arbitration" in motif_names:
        return ("shared_state_system", 0.0)
    if "trust_boundary_crossing" in motif_names:
        return ("trust_boundary_system", 0.0)

    # Count agents
    agent_count = sum(
        1 for n in fingerprint.get("nodes", {}).values()
        if n.get("structural", n).get("node_type") == "agent"
    )
    if agent_count <= 1:
        return ("simple_agent", 0.0)

    return ("generic", 0.0)


def _build_behavioral_description(
    precondition_id: str,
    probability: float,
    sample_size: int,
    severity_label: str,
    similar_outcomes: list[dict],
) -> str:
    """Generate a natural-language behavioral description."""
    pct = round(probability * 100, 1)

    base = f"In {pct}% of repos with this structural pattern"
    if sample_size > 0:
        base += f" (n={sample_size})"

    if severity_label == "high":
        base += (
            ", the precondition manifested as a high-severity failure "
            "with cascading downstream impact."
        )
    elif severity_label == "medium":
        base += (
            ", the precondition manifested as a medium-severity failure, "
            "typically contained within the affected subsystem."
        )
    elif severity_label == "low":
        base += (
            ", the precondition manifested as a low-severity issue, "
            "often caught by existing error handling."
        )
    else:
        base += ", insufficient behavioral data was available to assess severity."

    if similar_outcomes:
        outcome_count = len(similar_outcomes)
        base += (
            f" {outcome_count} similar repo(s) in the dataset showed "
            "comparable structural characteristics."
        )

    return base


def _check_fragility(
    precondition_id: str,
    fragility_map: list[dict[str, Any]],
    structural_graph: dict,
) -> bool:
    """Check whether this precondition sits at a model-sensitive position."""
    # Map preconditions to structural roles where fragility matters most
    precondition_role_map: dict[str, list[str]] = {
        "shared_state_no_arbitration": ["connector_node", "hub_node"],
        "unbounded_delegation_depth": ["chain_node"],
        "single_point_of_failure": ["hub_node"],
        "no_timeout_on_delegation": ["chain_node", "leaf_node"],
        "unhandled_tool_failure": ["leaf_node", "chain_node"],
        "circular_delegation": ["chain_node", "connector_node"],
        "hardcoded_model_dependency": ["hub_node", "chain_node", "leaf_node"],
        "no_fallback_for_external": ["leaf_node"],
    }

    relevant_roles = precondition_role_map.get(precondition_id, [])
    if not relevant_roles:
        return False

    for entry in fragility_map:
        role = entry.get("structural_position", "")
        if role in relevant_roles:
            sensitivity = entry.get("sensitivity_score", 0.0)
            if sensitivity > 0.3:
                return True

    return False


def _find_similar_repo_outcomes(
    precondition_id: str,
    patterns: list[dict[str, Any]],
    kb_fingerprints: list[dict[str, Any]],
    taxonomy_probs: dict[str, Any],
) -> list[dict]:
    """Find repos from the dataset that had this precondition and report outcomes."""
    outcomes: list[dict] = []

    # Check patterns for repos where this precondition's associated motif appeared
    # Map precondition to its related motif(s)
    precondition_motif_map: dict[str, list[str]] = {
        "shared_state_no_arbitration": ["shared_state_without_arbitration"],
        "unbounded_delegation_depth": ["linear_delegation_chain"],
        "single_point_of_failure": ["hub_and_spoke"],
        "circular_delegation": ["feedback_loop"],
        "trust_boundary_no_sanitization": ["trust_boundary_crossing"],
        "no_fallback_for_external": ["trust_boundary_crossing"],
        "no_timeout_on_delegation": ["linear_delegation_chain"],
    }

    related_motifs = precondition_motif_map.get(precondition_id, [])

    for pat in patterns:
        if pat.get("pattern_name") in related_motifs:
            prevalence = pat.get("prevalence", {})
            behavioral = pat.get("behavioral_distribution", {})
            for repo_id in prevalence.get("repo_ids", [])[:5]:
                outcomes.append({
                    "repo_id": repo_id,
                    "pattern_name": pat.get("pattern_name", ""),
                    "failure_rate": behavioral.get("failure_rate", 0.0),
                    "sample_size": behavioral.get("sample_size", 0),
                })

    return outcomes[:10]  # Cap at 10 similar repo outcomes


def _compute_archetype_prevalence(
    archetype: str,
    kb_fingerprints: list[dict[str, Any]],
) -> float:
    """Compute what fraction of KB fingerprints share the given archetype."""
    if not kb_fingerprints:
        return 0.0

    count = 0
    for fp in kb_fingerprints:
        fp_archetype, _ = _classify_archetype(fp)
        if fp_archetype == archetype:
            count += 1

    return round(count / len(kb_fingerprints), 4)


def _compute_framework_comparison(
    matches: list[Match],
    framework_comparisons: list[dict[str, Any]] | None,
) -> dict:
    """Extract relevant framework comparison data from matches."""
    if not framework_comparisons:
        return {}

    # Collect all motif names from exact matches
    motif_names = {m.pattern_name for m in matches if m.match_type == "exact_motif"}

    comparison: dict[str, Any] = {}
    for cmp in framework_comparisons:
        motif = cmp.get("motif_name", "")
        if motif in motif_names:
            comparison[motif] = {
                "frameworks_compared": cmp.get("frameworks_compared", []),
                "per_framework": {
                    fw: {
                        "repos_count": data.get("repos_count", 0),
                        "failure_rate": data.get(
                            "behavioral_distribution", {}
                        ).get("failure_rate", 0.0),
                    }
                    for fw, data in cmp.get("per_framework", {}).items()
                },
            }

    return comparison


def _compute_dataset_coverage(
    taxonomy_probs: dict[str, Any],
    patterns: list[dict[str, Any]],
    precondition_ids: list[str],
) -> dict:
    """Report how many preconditions have KB data and total sample sizes."""
    preconditions_with_data = 0
    total_sample_size = 0
    preconditions_without_data: list[str] = []

    for pc_id in precondition_ids:
        prob_data = taxonomy_probs.get(pc_id, {})
        if isinstance(prob_data, dict) and prob_data.get("sample_size", 0) > 0:
            preconditions_with_data += 1
            total_sample_size += prob_data.get("sample_size", 0)
        else:
            preconditions_without_data.append(pc_id)

    return {
        "preconditions_queried": len(precondition_ids),
        "preconditions_with_data": preconditions_with_data,
        "preconditions_without_data": preconditions_without_data,
        "total_sample_size": total_sample_size,
        "patterns_in_kb": len(patterns),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_risks(
    structural_graph: dict,
    matches: list[Match],
    taxonomy_preconditions: list[str],
    knowledge_base_path: Path,
) -> RiskPrediction:
    """Combine structural analysis with behavioral predictions.

    For each taxonomy precondition present in the graph:

    1. Look up manifestation probability from ``taxonomy_probabilities.json``.
    2. Find similar repos with this precondition and check behavioral outcomes.
    3. Compute severity from behavioral data.
    4. Flag if the precondition sits at a model-sensitive position
       (fragility map).

    The overall risk score is computed as::

        overall_risk_score = sum(
            probability_i * severity_weight_i * position_importance_i
        )

    scaled to the 0-100 range.

    Parameters
    ----------
    structural_graph:
        The customer's structural graph dict (with ``nodes``, ``edges``,
        ``motifs``, ``taxonomy_preconditions``).
    matches:
        List of :class:`Match` objects from :func:`matcher.match_against_dataset`.
    taxonomy_preconditions:
        List of taxonomy precondition IDs flagged for this graph.
    knowledge_base_path:
        Path to the directory containing knowledge-base JSON files.

    Returns
    -------
    A :class:`RiskPrediction` with detailed per-precondition risks,
    positive signals, framework comparisons, and dataset coverage.
    """
    kb = Path(knowledge_base_path)

    # Load knowledge-base assets
    taxonomy_probs: dict[str, Any] = _load_json(kb / "taxonomy_probabilities.json") or {}
    patterns: list[dict[str, Any]] = _load_json(kb / "patterns.json") or []
    fragility_map: list[dict[str, Any]] = _load_json(kb / "fragility_map.json") or []
    framework_comparisons: list[dict[str, Any]] | None = _load_json(
        kb / "framework_comparisons.json"
    )
    _raw_fps = _load_json(kb / "fingerprints.json") or {}
    if isinstance(_raw_fps, dict):
        kb_fingerprints: list[dict[str, Any]] = list(_raw_fps.values())
    else:
        kb_fingerprints: list[dict[str, Any]] = _raw_fps

    # Compute fingerprint for archetype classification
    from stratum_lab.query.fingerprint import compute_graph_fingerprint
    customer_fp = compute_graph_fingerprint(structural_graph)

    # Classify archetype using the fingerprint (which has motifs)
    archetype, _ = _classify_archetype(customer_fp)
    archetype_prevalence = _compute_archetype_prevalence(archetype, kb_fingerprints)

    # Detect positive signals
    positive_signals = _detect_positive_signals(structural_graph)

    # Build predicted risks for each taxonomy precondition
    predicted_risks: list[PredictedRisk] = []
    structural_only_risks: list[dict] = []

    # Accumulator for overall risk score
    weighted_risk_sum = 0.0
    max_possible_weight = 0.0

    for pc_id in taxonomy_preconditions:
        prob_data = taxonomy_probs.get(pc_id, {})

        # Determine if we have behavioral data for this precondition
        has_behavioral_data = (
            isinstance(prob_data, dict) and prob_data.get("sample_size", 0) > 0
        )

        if not has_behavioral_data:
            # Structural-only risk: no behavioral data available
            structural_only_risks.append({
                "precondition_id": pc_id,
                "reason": "No behavioral data in the knowledge base for this precondition.",
                "remediation": REMEDIATIONS.get(pc_id, ""),
            })
            continue

        # Extract probability and severity
        probability = prob_data.get("probability", 0.0) or 0.0
        ci = prob_data.get("confidence_interval", [0.0, 1.0])
        confidence_interval = (
            ci[0] if ci[0] is not None else 0.0,
            ci[1] if ci[1] is not None else 1.0,
        )
        sample_size = prob_data.get("sample_size", 0)

        severity_data = prob_data.get("severity_when_manifested", {})
        if isinstance(severity_data, dict):
            severity_label = severity_data.get("severity_label", "low")
        elif isinstance(severity_data, str):
            severity_label = severity_data
        else:
            severity_label = "low"

        # Build structural evidence: list of structural reasons this is flagged
        structural_evidence: list[str] = []
        for m in matches:
            if m.match_type == "exact_motif":
                structural_evidence.append(
                    f"Exact motif '{m.pattern_name}' found in {m.matched_repos} repo(s)"
                )
            elif m.match_type == "structural_similarity" and m.similarity_score > 0.7:
                structural_evidence.append(
                    f"High structural similarity ({m.similarity_score:.2f}) "
                    f"to known pattern '{m.pattern_name}'"
                )
            elif m.match_type == "archetype":
                structural_evidence.append(
                    f"Archetype '{m.pattern_name}' matches {m.matched_repos} repo(s)"
                )

        if not structural_evidence:
            structural_evidence.append(
                f"Precondition '{pc_id}' flagged by structural scan"
            )

        # Find similar repo outcomes
        similar_outcomes = _find_similar_repo_outcomes(
            pc_id, patterns, kb_fingerprints, taxonomy_probs,
        )

        # Check fragility
        fragility_flag = _check_fragility(pc_id, fragility_map, structural_graph)

        # Build behavioral description
        behavioral_description = _build_behavioral_description(
            pc_id, probability, sample_size, severity_label, similar_outcomes,
        )

        # Get remediation
        remediation = REMEDIATIONS.get(pc_id, "")

        predicted_risks.append(PredictedRisk(
            precondition_id=pc_id,
            precondition_name=pc_id.replace("_", " ").title(),
            structural_evidence=structural_evidence,
            manifestation_probability=round(probability, 4),
            confidence_interval=confidence_interval,
            sample_size=sample_size,
            severity_when_manifested=severity_label,
            behavioral_description=behavioral_description,
            similar_repo_outcomes=similar_outcomes,
            fragility_flag=fragility_flag,
            remediation=remediation,
        ))

        # Accumulate for overall risk score
        severity_weight = _SEVERITY_WEIGHT.get(severity_label, 1)
        position_importance = 1.5 if fragility_flag else 1.0
        weighted_risk_sum += probability * severity_weight * position_importance
        max_possible_weight += 3.0 * 1.5  # max severity * max position importance

    # Compute overall risk score (0-100)
    if max_possible_weight > 0:
        overall_risk_score = round(
            (weighted_risk_sum / max_possible_weight) * 100.0, 2
        )
    else:
        overall_risk_score = 0.0

    # Clamp to [0, 100]
    overall_risk_score = max(0.0, min(100.0, overall_risk_score))

    # Framework comparison
    framework_comparison = _compute_framework_comparison(
        matches, framework_comparisons,
    )

    # Dataset coverage
    dataset_coverage = _compute_dataset_coverage(
        taxonomy_probs, patterns, taxonomy_preconditions,
    )

    # Semantic risk analysis
    semantic_analysis = _compute_semantic_risk(structural_graph)

    # Sort predicted risks by manifestation probability descending
    predicted_risks.sort(key=lambda r: r.manifestation_probability, reverse=True)

    return RiskPrediction(
        graph_fingerprint=customer_fp,
        archetype=archetype,
        archetype_prevalence=archetype_prevalence,
        overall_risk_score=overall_risk_score,
        predicted_risks=predicted_risks,
        structural_only_risks=structural_only_risks,
        positive_signals=positive_signals,
        framework_comparison=framework_comparison,
        dataset_coverage=dataset_coverage,
        semantic_analysis=semantic_analysis,
    )
