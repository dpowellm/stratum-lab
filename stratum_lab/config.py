"""Global configuration constants for stratum-lab."""

from pathlib import Path

# Pipeline defaults
DEFAULT_SELECTION_TARGET = 1500
DEFAULT_MAX_PER_ARCHETYPE = 200
DEFAULT_MIN_PER_ARCHETYPE = 30
DEFAULT_MIN_RUNNABILITY = 15
DEFAULT_CONCURRENT_CONTAINERS = 5
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 600
DEFAULT_RUNS_PER_REPO = 5  # 3 diverse inputs + 2 repeat
MIN_EVENT_THRESHOLD = 10  # Minimum events for SUCCESS (vs PARTIAL_SUCCESS)

# vLLM defaults
VLLM_BASE_URL = "http://host.docker.internal:8000/v1"
VLLM_API_KEY = "sk-stratum-local"
VLLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Container config
DOCKER_IMAGE_NAME = "stratum-lab-runner"
DOCKER_IMAGE_TAG = "latest"
PATCHER_CONTAINER_PATH = "/opt/stratum"
EVENTS_FILE_PATH = "/app/stratum_events.jsonl"
CONTAINER_WORKDIR = "/app"

# Output directories
DATA_DIR = Path("data")
ENRICHED_GRAPHS_DIR = DATA_DIR / "enriched_graphs"
PATTERN_KB_DIR = DATA_DIR / "pattern_knowledge_base"
RAW_EVENTS_DIR = DATA_DIR / "raw_events"
EXECUTION_META_DIR = DATA_DIR / "execution_metadata"
BENCHMARK_DIR = DATA_DIR / "benchmark"

# Execution status codes
class ExecutionStatus:
    # Tier 1: Native execution
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_EVENTS = "NO_EVENTS"
    TIMEOUT_NO_EVENTS = "TIMEOUT_NO_EVENTS"
    UNRESOLVABLE_IMPORT = "UNRESOLVABLE_IMPORT"
    NO_ENTRY_POINT = "NO_ENTRY_POINT"
    CLONE_FAILED = "CLONE_FAILED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    SERVER_BASED = "SERVER_BASED"
    # Tier 2: Synthetic harness
    TIER2_SUCCESS = "TIER2_SUCCESS"
    TIER2_PARTIAL = "TIER2_PARTIAL"
    # Tier 3: Unrunnable
    UNRUNNABLE = "UNRUNNABLE"
    # Legacy status codes (kept for backward compatibility)
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    ENTRY_POINT_FAILURE = "ENTRY_POINT_FAILURE"
    MODEL_FAILURE = "MODEL_FAILURE"
    TIMEOUT = "TIMEOUT"
    CRASH = "CRASH"
    INSTRUMENTATION_FAILURE = "INSTRUMENTATION_FAILURE"

# Supported frameworks
SUPPORTED_FRAMEWORKS = ["crewai", "langgraph", "autogen", "langchain", "custom"]

# Known entry point filenames
ENTRY_POINT_NAMES = [
    "main.py", "app.py", "crew.py", "run.py", "start.py",
    "__main__.py", "cli.py", "agent.py", "agents.py",
]

# Finding ID mapping: precondition name -> STRAT-XX-NNN
PRECONDITION_TO_FINDING = {
    "no_error_propagation_strategy": "STRAT-SI-001",
    "unbounded_delegation_depth": "STRAT-DC-001",
    "no_timeout_on_delegation": "STRAT-DC-002",
    "no_output_validation": "STRAT-SI-004",
    "unhandled_tool_failure": "STRAT-EA-001",
    "shared_state_no_arbitration": "STRAT-OC-002",
    "no_rate_limiting": "STRAT-AB-001",
    "shared_tool_no_concurrency_control": "STRAT-OC-001",
    "single_point_of_failure": "STRAT-EA-002",
    "no_fallback_for_external": "STRAT-EA-003",
    "circular_delegation": "STRAT-DC-003",
    "implicit_ordering_dependency": "STRAT-SI-002",
    "unvalidated_semantic_chain": "STRAT-SI-003",
    "trust_boundary_no_sanitization": "STRAT-SI-005",
    "capability_overlap_no_priority": "STRAT-OC-003",
    "classification_without_validation": "STRAT-SI-006",
    "data_store_no_schema_enforcement": "STRAT-SI-007",
    "missing_guardrail": "STRAT-DC-004",
    "unbounded_retry_loop": "STRAT-AB-002",
}

# Reverse lookup
FINDING_TO_PRECONDITION = {v: k for k, v in PRECONDITION_TO_FINDING.items()}

# Canonical mapping: which metric monitors which finding
METRIC_TO_FINDING = {
    "max_delegation_depth": "STRAT-DC-001",
    "error_swallow_rate": "STRAT-SI-001",
    "total_llm_calls_per_run": "STRAT-AB-001",
    "concurrent_state_write_rate": "STRAT-OC-002",
    "schema_mismatch_rate": "STRAT-SI-004",
    "tool_call_failure_rate": "STRAT-EA-001",
    "delegation_latency_p95_ms": "STRAT-DC-002",
}

# Scanner-aligned metric names (what the reliability scanner expects)
SCANNER_METRIC_NAMES = {
    "max_delegation_depth": "delegation_chain_depth",
    "error_swallow_rate": "error_handling_silent_rate",
    "total_llm_calls_per_run": "llm_invocation_count",
    "concurrent_state_write_rate": "state_write_conflict_rate",
    "schema_mismatch_rate": "output_validation_failure_rate",
    "tool_call_failure_rate": "tool_reliability_rate",
    "delegation_latency_p95_ms": "delegation_latency_p95",
}

# Human-readable finding names for catalog display.
# Each name MUST match its taxonomy category:
#   DC = decision chains, delegation, human oversight, checkpoints, reversibility
#   OC = objectives, incentives, conflicts, shared state, competing agents
#   SI = signal integrity, error propagation, error laundering, schema validation
#   EA = authority, scope, delegation escalation, permissions, role boundaries
#   AB = aggregate behavior, volume, monitoring, population-level risk
FINDING_NAMES = {
    # Decision Chain Risk (DC)
    "STRAT-DC-001": "Unsupervised delegation chain",
    "STRAT-DC-002": "Irreversible action without checkpoint",
    "STRAT-DC-003": "Circular delegation",
    "STRAT-DC-004": "Missing human oversight checkpoint",
    # Objective & Incentive Conflict (OC)
    "STRAT-OC-001": "Competing resource access without coordination",
    "STRAT-OC-002": "Shared state contention",
    "STRAT-OC-003": "Capability overlap without priority",
    # Signal Integrity & Error Propagation (SI)
    "STRAT-SI-001": "Error laundering",
    "STRAT-SI-002": "Implicit ordering dependency",
    "STRAT-SI-003": "Unvalidated semantic chain",
    "STRAT-SI-004": "Missing output validation",
    "STRAT-SI-005": "Trust boundary without sanitization",
    "STRAT-SI-006": "Classification without validation",
    "STRAT-SI-007": "Missing data schema enforcement",
    # Emergent Authority & Scope Creep (EA)
    "STRAT-EA-001": "Transitive authority escalation",
    "STRAT-EA-002": "Authority concentration at bottleneck",
    "STRAT-EA-003": "Unbounded external service scope",
    # Aggregate Behavioral Exposure (AB)
    "STRAT-AB-001": "Unbounded aggregate volume",
    "STRAT-AB-002": "Aggregate retry amplification",
    # Cross-cutting
    "STRAT-XCOMP-001": "Cross-domain: security x reliability",
    "STRAT-XCOMP-006": "Cross-domain: availability x integrity",
}
