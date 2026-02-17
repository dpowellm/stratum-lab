"""Judge pipeline configuration — model, API key, batch settings, cost tracking."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024
TEMPERATURE = 0.0  # Low temp for evaluation tasks (research consensus)

# ---------------------------------------------------------------------------
# API key — override via --api-key CLI flag or ANTHROPIC_API_KEY env var
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY", "sk-ant-REPLACE-WITH-YOUR-KEY"
)

# ---------------------------------------------------------------------------
# Batch settings
# ---------------------------------------------------------------------------
BATCH_SIZE = 10_000        # Anthropic batch limit
POLL_INTERVAL_SECONDS = 60
MAX_WAIT_SECONDS = 86_400  # 24h

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
# Batch API pricing (50% discount over real-time)
INPUT_COST_PER_M = 1.50   # USD per 1M input tokens
OUTPUT_COST_PER_M = 7.50   # USD per 1M output tokens

COST_ABORT_THRESHOLD_USD = 50.0  # Abort if projected cost exceeds this

# Filter settings
VALID_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS"}


# ---------------------------------------------------------------------------
# Cost utilities
# ---------------------------------------------------------------------------
class CostTracker:
    """Running total of token usage and estimated cost."""

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    @property
    def cost_usd(self) -> float:
        return (
            (self.total_input_tokens / 1_000_000) * INPUT_COST_PER_M
            + (self.total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
        )

    def estimate_cost(self, n_calls: int, avg_input: int = 2000,
                      avg_output: int = 300) -> float:
        """Estimate total cost for *n_calls* judge calls."""
        return (
            (n_calls * avg_input / 1_000_000) * INPUT_COST_PER_M
            + (n_calls * avg_output / 1_000_000) * OUTPUT_COST_PER_M
        )

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.cost_usd, 2),
        }
