"""LLM-as-judge post-processing pipeline for stratum-lab behavioral scans.

Evaluates agent outputs from behavioral scan results using Claude Sonnet 4.5
via the Anthropic Batch API.  Six independent criteria per G-Eval decomposition:
task adherence, hallucination, instruction leakage, output quality,
delegation fidelity, and error propagation.
"""
from stratum_lab.judge.event_loader import (
    AgentExecution,
    ExecutionContext,
    LLMCall,
    load_execution_context,
)
from stratum_lab.judge.config import CostTracker
