"""Shared data models for stratum-lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    AGENT = "agent"
    CAPABILITY = "capability"
    DATA_STORE = "data_store"
    EXTERNAL_SERVICE = "external"
    MCP_SERVER = "mcp_server"
    GUARDRAIL = "guardrail"


class EdgeType(str, Enum):
    READS_FROM = "reads_from"
    WRITES_TO = "writes_to"
    SENDS_TO = "sends_to"
    CALLS = "calls"
    SHARES_WITH = "shares_with"
    FILTERED_BY = "filtered_by"
    GATED_BY = "gated_by"
    TOOL_OF = "tool_of"
    DELEGATES_TO = "delegates_to"
    FEEDS_INTO = "feeds_into"
    SHARES_TOOL = "shares_tool"


@dataclass
class RepoSelection:
    """A single repo selected for behavioral scanning."""
    repo_id: str
    repo_url: str
    framework: str
    selection_score: float
    structural_value: float
    archetype_id: int
    archetype_name: str
    runnability_score: float
    agent_count: int
    taxonomy_preconditions: list[str]
    detected_entry_point: str
    detected_requirements: str
    estimated_complexity: str  # "low", "medium", "high"


@dataclass
class SelectionSummary:
    """Aggregate summary of the selection."""
    total_selected: int
    by_framework: dict[str, int]
    by_archetype: dict[str, int]
    avg_structural_value: float
    avg_agent_count: float
    total_taxonomy_preconditions_covered: int


@dataclass
class RunMetadata:
    """Metadata for a single execution run."""
    repo_id: str
    run_id: str
    run_number: int
    input_hash: str
    framework: str
    timestamp: str
    status: str
    duration_ms: int
    events_count: int
    peak_memory_mb: float
    exit_code: int | None
    error_message: str | None = None


@dataclass
class EventRecord:
    """A single event from the JSONL event stream."""
    event_id: str
    timestamp_ns: int
    run_id: str
    repo_id: str
    framework: str
    event_type: str
    source_node: dict[str, str] | None = None
    target_node: dict[str, str] | None = None
    edge_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    stack_depth: int = 0
    parent_event_id: str | None = None


@dataclass
class NodeBehavior:
    """Behavioral overlay data for a single node."""
    activation_count: int = 0
    activation_rate: float = 0.0
    throughput: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    error_behavior: dict[str, Any] = field(default_factory=dict)
    decision_behavior: dict[str, Any] | None = None
    model_sensitivity: dict[str, Any] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeBehavior:
    """Behavioral overlay data for a single edge."""
    traversal_count: int = 0
    activation_rate: float = 0.0
    never_activated: bool = True
    data_flow: dict[str, Any] = field(default_factory=dict)
    error_crossings: dict[str, Any] = field(default_factory=dict)
    latency_contribution_ms: float = 0.0
    conditional_behavior: dict[str, float] | None = None


@dataclass
class EmergentEdge:
    """A runtime-only edge not present in the structural graph."""
    edge_id: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    runtime_only: bool = True
    behavioral: dict[str, Any] = field(default_factory=dict)
    significance: str = "medium"


@dataclass
class DeadEdge:
    """A structural edge that never activated at runtime."""
    edge_id: str
    dead: bool = True
    runs_observed: int = 0
    possible_reasons: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A structural pattern with behavioral statistics."""
    pattern_id: str
    pattern_name: str
    structural_signature: dict[str, Any]
    prevalence: dict[str, Any]
    behavioral_distribution: dict[str, Any]
    fragility_data: dict[str, Any]
    risk_assessment: dict[str, Any]


@dataclass
class EnrichedGraph:
    """A structural graph enriched with behavioral data."""
    repo_id: str
    framework: str
    nodes: dict[str, dict[str, Any]]  # node_id -> {structural, behavioral}
    edges: dict[str, dict[str, Any]]  # edge_id -> {structural, behavioral}
    emergent_edges: list[EmergentEdge] = field(default_factory=list)
    dead_edges: list[DeadEdge] = field(default_factory=list)
    run_metadata: list[RunMetadata] = field(default_factory=list)
