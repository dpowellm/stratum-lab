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
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
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
