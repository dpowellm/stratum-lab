"""Batch API submission, polling, and result parsing for the judge pipeline."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from stratum_lab.judge.config import (
    JUDGE_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    BATCH_SIZE,
    POLL_INTERVAL_SECONDS,
    MAX_WAIT_SECONDS,
    CostTracker,
)
from stratum_lab.judge.criteria import PER_AGENT_CRITERIA, CHAIN_CRITERIA
from stratum_lab.judge.event_loader import AgentExecution, ExecutionContext

logger = logging.getLogger(__name__)

# JSON schema reminder for retry on malformed output
_RETRY_REMINDER = (
    "Your previous response was not valid JSON. "
    "Please respond with ONLY a JSON object matching the requested schema. "
    "Do not include any text before or after the JSON."
)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_batch_requests(
    repo_contexts: list[ExecutionContext],
) -> list[dict[str, Any]]:
    """Build all judge requests for all repos.

    Returns a list of Anthropic Batch API request dicts, each with a
    ``custom_id`` encoding repo|agent|criterion for result routing.
    """
    requests: list[dict[str, Any]] = []

    for ctx in repo_contexts:
        repo_id = ctx.repo_url or "unknown"

        # Per-agent criteria (1-4)
        for agent in ctx.agents:
            for criterion_name, prompt_fn in PER_AGENT_CRITERIA.items():
                prompt = prompt_fn(agent, ctx)
                if not prompt:
                    continue  # Skip empty/short outputs
                requests.append(_make_request(
                    custom_id=f"{repo_id}|{agent.agent_name}|{criterion_name}",
                    prompt=prompt,
                ))

        # Chain-level criteria (5-6) — only if 2+ agents
        if len(ctx.agents) >= 2:
            for i in range(1, len(ctx.agents)):
                upstream = ctx.agents[i - 1]
                downstream = ctx.agents[i]

                # Delegation fidelity
                prompt = CHAIN_CRITERIA["delegation_fidelity"](
                    upstream, downstream, ctx,
                )
                if prompt is not None:
                    requests.append(_make_request(
                        custom_id=(
                            f"{repo_id}|"
                            f"{upstream.agent_name}->{downstream.agent_name}|"
                            f"delegation_fidelity"
                        ),
                        prompt=prompt,
                    ))

                # Error propagation — only if upstream has known issues
                # (placeholder: upstream_issues populated during aggregation)

    return requests


def _make_request(custom_id: str, prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "params": {
            "model": JUDGE_MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


# ---------------------------------------------------------------------------
# Batch submission and polling
# ---------------------------------------------------------------------------

def submit_batch(
    requests: list[dict[str, Any]],
    client: Any,
) -> str:
    """Submit a batch and return the batch_id.

    Parameters
    ----------
    requests : list of request dicts (custom_id + params)
    client : anthropic.Anthropic instance
    """
    batch = client.messages.batches.create(requests=requests)
    logger.info("Submitted batch %s with %d requests", batch.id, len(requests))
    return batch.id


def poll_batch(
    batch_id: str,
    client: Any,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    max_wait: int = MAX_WAIT_SECONDS,
) -> list[Any]:
    """Poll until batch completes. Returns list of result objects."""
    elapsed = 0
    while elapsed < max_wait:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            "Batch %s: %s (succeeded=%d processing=%d errored=%d)",
            batch_id,
            batch.processing_status,
            counts.succeeded,
            counts.processing,
            counts.errored,
        )

        if batch.processing_status == "ended":
            results = list(client.messages.batches.results(batch_id))
            logger.info("Batch %s completed: %d results", batch_id, len(results))
            return results

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait}s")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_judge_result(result: Any, cost_tracker: CostTracker | None = None) -> dict[str, Any] | None:
    """Parse a single batch result into a judge record dict.

    Handles malformed JSON by returning a judge_error record.
    """
    custom_id = result.custom_id
    parts = custom_id.split("|", 2)
    repo_url = parts[0] if len(parts) > 0 else ""
    agent_name = parts[1] if len(parts) > 1 else ""
    criterion = parts[2] if len(parts) > 2 else ""

    # Check for API-level error
    if result.result.type == "error":
        return {
            "repo_url": repo_url,
            "agent_name": agent_name,
            "criterion": criterion,
            "judge_error": str(result.result.error),
        }

    # Extract text content
    message = result.result.message
    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text += block.text

    # Track tokens
    if cost_tracker and hasattr(message, "usage"):
        cost_tracker.record(
            message.usage.input_tokens,
            message.usage.output_tokens,
        )

    # Parse JSON
    parsed = _try_parse_json(text)
    if parsed is None:
        return {
            "repo_url": repo_url,
            "agent_name": agent_name,
            "criterion": criterion,
            "judge_error": "malformed_json",
            "raw_response": text[:500],
        }

    return {
        "repo_url": repo_url,
        "agent_name": agent_name,
        "criterion": criterion,
        **parsed,
        "judge_model": JUDGE_MODEL,
    }


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from judge response, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def build_retry_request(original_request: dict[str, Any]) -> dict[str, Any]:
    """Build a retry request with a reminder to return valid JSON."""
    req = {
        "custom_id": original_request["custom_id"],
        "params": {
            **original_request["params"],
            "messages": [
                *original_request["params"]["messages"],
                {"role": "assistant", "content": "I'll provide the evaluation."},
                {"role": "user", "content": _RETRY_REMINDER},
            ],
        },
    }
    return req
