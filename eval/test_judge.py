"""Tests for the LLM-as-judge pipeline.

Covers:
  - ExecutionContext reconstruction from synthetic events
  - All 6 criteria prompt generation and validation
  - Batch request building
  - Result parsing (valid JSON, malformed JSON, errors)
  - Aggregator summary building
  - Cost tracking
  - CLI discovery logic
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from stratum_lab.judge.config import CostTracker
from stratum_lab.judge.event_loader import (
    AgentExecution,
    ExecutionContext,
    LLMCall,
    load_events,
    reconstruct_context,
    load_execution_context,
)
from stratum_lab.judge.criteria import (
    format_task_adherence,
    format_hallucination,
    format_instruction_leakage,
    format_output_quality,
    format_delegation_fidelity,
    format_error_propagation,
    PER_AGENT_CRITERIA,
    CHAIN_CRITERIA,
)
from stratum_lab.judge.runner import (
    build_batch_requests,
    _try_parse_json,
    build_retry_request,
)
from stratum_lab.judge.aggregator import build_summary, _generate_headlines


# ===========================================================================
# Synthetic event fixtures
# ===========================================================================

SYNTHETIC_EVENTS = [
    {
        "event_id": "evt-000",
        "event_type": "execution.start",
        "timestamp_ns": 1000000000,
        "timestamp": "2026-02-17T10:00:00.000Z",
        "payload": {
            "repo_url": "https://github.com/test-org/crew-demo",
            "framework": "crewai",
        },
    },
    {
        "event_id": "evt-001",
        "event_type": "patcher.status",
        "timestamp_ns": 1000100000,
        "timestamp": "2026-02-17T10:00:00.100Z",
        "payload": {"patches_ok": ["crewai", "openai"], "patches_skipped": []},
    },
    {
        "event_id": "evt-010",
        "event_type": "agent.task_start",
        "timestamp_ns": 2000000000,
        "timestamp": "2026-02-17T10:00:01.000Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {"task": "Find the top 3 most popular Python web frameworks in 2025."},
    },
    {
        "event_id": "evt-011",
        "event_type": "llm.call_start",
        "timestamp_ns": 2100000000,
        "timestamp": "2026-02-17T10:00:01.100Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {"model_requested": "gpt-4"},
    },
    {
        "event_id": "evt-012",
        "event_type": "llm.call_end",
        "timestamp_ns": 3500000000,
        "timestamp": "2026-02-17T10:00:02.500Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
            "input_tokens": 800,
            "output_tokens": 400,
            "latency_ms": 1400.0,
            "output_preview": "Based on my research, the top 3 Python web frameworks in 2025 are Django, FastAPI, and Flask.",
        },
    },
    {
        "event_id": "evt-013",
        "event_type": "agent.task_end",
        "timestamp_ns": 4000000000,
        "timestamp": "2026-02-17T10:00:03.000Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {
            "output_preview": "The top 3 Python web frameworks in 2025 are:\n1. Django - mature, batteries-included\n2. FastAPI - async-first, high performance\n3. Flask - lightweight, flexible",
            "output_hash": "abc123",
            "output_type": "research_report",
        },
    },
    {
        "event_id": "evt-020",
        "event_type": "delegation.initiated",
        "timestamp_ns": 4100000000,
        "timestamp": "2026-02-17T10:00:03.100Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "target_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "payload": {"delegator": "Researcher", "delegate": "Writer"},
    },
    {
        "event_id": "evt-030",
        "event_type": "agent.task_start",
        "timestamp_ns": 5000000000,
        "timestamp": "2026-02-17T10:00:04.000Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "payload": {"task": "Write a 200-word summary comparing the top 3 frameworks."},
    },
    {
        "event_id": "evt-031",
        "event_type": "llm.call_end",
        "timestamp_ns": 7000000000,
        "timestamp": "2026-02-17T10:00:06.000Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "payload": {
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
            "input_tokens": 1200,
            "output_tokens": 600,
            "latency_ms": 2000.0,
            "output_preview": "A comparison of Django, FastAPI, and Flask...",
        },
    },
    {
        "event_id": "evt-032",
        "event_type": "agent.task_end",
        "timestamp_ns": 8000000000,
        "timestamp": "2026-02-17T10:00:07.000Z",
        "source_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "payload": {
            "output_preview": "Django remains the go-to for large applications with its ORM and admin panel. FastAPI has surged in popularity for API development, offering async support and automatic OpenAPI docs. Flask continues to thrive as a lightweight option for developers who want flexibility without opinions. For new projects, FastAPI is the rising star, while Django dominates enterprise use cases.",
            "output_hash": "def456",
            "output_type": "summary",
        },
    },
    {
        "event_id": "evt-099",
        "event_type": "execution.end",
        "timestamp_ns": 9000000000,
        "timestamp": "2026-02-17T10:00:08.000Z",
        "payload": {},
    },
]


# ===========================================================================
# EVENT LOADER TESTS
# ===========================================================================

class TestEventLoader:
    def test_load_events_from_file(self, tmp_path):
        """load_events reads JSONL and returns sorted events."""
        events_file = tmp_path / "events.jsonl"
        with open(events_file, "w") as f:
            for evt in SYNTHETIC_EVENTS:
                f.write(json.dumps(evt) + "\n")
        events = load_events(events_file)
        assert len(events) == len(SYNTHETIC_EVENTS)
        # Should be sorted by timestamp_ns
        ts = [e.get("timestamp_ns", 0) for e in events]
        assert ts == sorted(ts)

    def test_load_events_missing_file(self):
        """load_events returns empty list for nonexistent file."""
        events = load_events(Path("/nonexistent/path.jsonl"))
        assert events == []

    def test_load_events_malformed_lines(self, tmp_path):
        """load_events skips malformed JSON lines."""
        events_file = tmp_path / "events.jsonl"
        with open(events_file, "w") as f:
            f.write('{"event_type": "ok", "timestamp_ns": 1}\n')
            f.write("this is not json\n")
            f.write('{"event_type": "also_ok", "timestamp_ns": 2}\n')
        events = load_events(events_file)
        assert len(events) == 2

    def test_reconstruct_context_agents(self):
        """reconstruct_context extracts correct agent names and count."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS, status="SUCCESS")
        assert len(ctx.agents) == 2
        names = [a.agent_name for a in ctx.agents]
        assert "Researcher" in names
        assert "Writer" in names

    def test_reconstruct_context_repo_url(self):
        """reconstruct_context extracts repo_url from execution.start."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        assert ctx.repo_url == "https://github.com/test-org/crew-demo"

    def test_reconstruct_context_framework(self):
        """reconstruct_context extracts framework."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        assert ctx.framework == "crewai"

    def test_reconstruct_context_delegation_chain(self):
        """reconstruct_context builds delegation chain."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        assert len(ctx.delegation_chain) == 1
        assert ctx.delegation_chain[0] == ("Researcher", "Writer")

    def test_agent_task_description(self):
        """Agent task_description extracted from agent.task_start."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert "top 3" in researcher.task_description
        assert "Python web frameworks" in researcher.task_description

    def test_agent_output_text(self):
        """Agent output_text extracted from agent.task_end preview."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert "Django" in researcher.output_text
        assert "FastAPI" in researcher.output_text
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        assert "Django remains" in writer.output_text

    def test_agent_output_hash(self):
        """output_hash extracted from agent.task_end."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert researcher.output_hash == "abc123"

    def test_agent_llm_calls(self):
        """LLM calls attached to correct agents."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert len(researcher.llm_calls) == 1
        assert researcher.llm_calls[0].input_tokens == 800
        assert researcher.llm_calls[0].output_tokens == 400
        assert researcher.llm_calls[0].latency_ms == 1400.0

    def test_agent_token_totals(self):
        """Token totals summed across LLM calls."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert researcher.token_count_in == 800
        assert researcher.token_count_out == 400

    def test_agent_duration(self):
        """Agent duration computed from task_start to task_end."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert researcher.duration_seconds == pytest.approx(2.0, abs=0.1)

    def test_total_events(self):
        """total_events matches input count."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        assert ctx.total_events == len(SYNTHETIC_EVENTS)

    def test_total_duration(self):
        """total_duration_seconds from execution.start to execution.end."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS)
        assert ctx.total_duration_seconds == pytest.approx(8.0, abs=0.1)

    def test_incomplete_agent(self):
        """Missing task_end marks agent as incomplete, uses LLM output."""
        # Remove the task_end for Writer
        events = [e for e in SYNTHETIC_EVENTS if not (
            e["event_type"] == "agent.task_end"
            and e.get("source_node", {}).get("node_name") == "Writer"
        )]
        ctx = reconstruct_context(events)
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        assert writer.incomplete is True
        # Should use last LLM call output as fallback
        assert "comparison" in writer.output_text.lower() or "Django" in writer.output_text

    def test_load_execution_context_from_file(self, tmp_path):
        """Full load_execution_context reads file and reconstructs."""
        events_file = tmp_path / "events_run_1.jsonl"
        with open(events_file, "w") as f:
            for evt in SYNTHETIC_EVENTS:
                f.write(json.dumps(evt) + "\n")
        ctx = load_execution_context(events_file, status="SUCCESS")
        assert ctx.status == "SUCCESS"
        assert len(ctx.agents) == 2

    def test_empty_events(self):
        """Empty event list produces empty context."""
        ctx = reconstruct_context([])
        assert ctx.agents == []
        assert ctx.delegation_chain == []
        assert ctx.total_events == 0


# ===========================================================================
# CRITERIA PROMPT TESTS
# ===========================================================================

class TestCriteriaPrompts:
    @pytest.fixture
    def ctx(self):
        return reconstruct_context(SYNTHETIC_EVENTS, status="SUCCESS")

    def test_task_adherence_prompt_valid(self, ctx):
        """Task adherence prompt contains task and output."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        prompt = format_task_adherence(researcher, ctx)
        assert "TASK DESCRIPTION:" in prompt
        assert "AGENT OUTPUT:" in prompt
        assert "top 3" in prompt
        assert "Django" in prompt
        assert '"score": 1|2|3' in prompt

    def test_task_adherence_empty_output_skipped(self, ctx):
        """Task adherence returns empty string for short output."""
        agent = AgentExecution(agent_name="Empty", output_text="hi")
        prompt = format_task_adherence(agent, ctx)
        assert prompt == ""

    def test_hallucination_prompt_valid(self, ctx):
        """Hallucination prompt contains task and output."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        prompt = format_hallucination(researcher, ctx)
        assert "hallucinated content" in prompt.lower() or "Hallucination" in prompt
        assert "AGENT OUTPUT:" in prompt

    def test_hallucination_with_upstream_context(self, ctx):
        """Hallucination prompt includes upstream context for delegated agents."""
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        prompt = format_hallucination(writer, ctx)
        assert "UPSTREAM CONTEXT" in prompt
        assert "Researcher" in prompt

    def test_instruction_leakage_prompt_valid(self, ctx):
        """Instruction leakage prompt contains output and backstory."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        prompt = format_instruction_leakage(researcher, ctx)
        assert "AGENT OUTPUT:" in prompt
        assert "AGENT ROLE/BACKSTORY" in prompt
        assert '"leakage_detected": true|false' in prompt

    def test_output_quality_prompt_valid(self, ctx):
        """Output quality prompt contains task and output."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        prompt = format_output_quality(researcher, ctx)
        assert "TASK DESCRIPTION:" in prompt
        assert "AGENT OUTPUT:" in prompt
        assert "Mistral-7B" in prompt

    def test_output_quality_empty_output_skipped(self, ctx):
        """Output quality returns empty string for short output."""
        agent = AgentExecution(agent_name="Short", output_text="ok")
        prompt = format_output_quality(agent, ctx)
        assert prompt == ""

    def test_delegation_fidelity_prompt_valid(self, ctx):
        """Delegation fidelity prompt has upstream and downstream."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        prompt = format_delegation_fidelity(researcher, writer, ctx)
        assert prompt is not None
        assert "UPSTREAM AGENT: Researcher" in prompt
        assert "DOWNSTREAM AGENT: Writer" in prompt
        assert "UPSTREAM OUTPUT:" in prompt
        assert "DOWNSTREAM OUTPUT:" in prompt

    def test_delegation_fidelity_missing_output(self, ctx):
        """Delegation fidelity returns None when output is missing."""
        empty = AgentExecution(agent_name="Empty", output_text="")
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        assert format_delegation_fidelity(empty, researcher, ctx) is None

    def test_error_propagation_prompt_valid(self, ctx):
        """Error propagation prompt includes issues."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        prompt = format_error_propagation(
            researcher, writer, ctx,
            upstream_issues="Hallucinated citation: 'Smith et al. 2025'",
        )
        assert prompt is not None
        assert "KNOWN UPSTREAM ISSUES:" in prompt
        assert "Smith et al." in prompt

    def test_error_propagation_no_issues_skipped(self, ctx):
        """Error propagation returns None when no upstream issues."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        writer = next(a for a in ctx.agents if a.agent_name == "Writer")
        assert format_error_propagation(researcher, writer, ctx) is None

    def test_all_per_agent_criteria_registered(self):
        """All 4 per-agent criteria are in the registry."""
        assert "task_adherence" in PER_AGENT_CRITERIA
        assert "hallucination" in PER_AGENT_CRITERIA
        assert "instruction_leakage" in PER_AGENT_CRITERIA
        assert "output_quality" in PER_AGENT_CRITERIA

    def test_all_chain_criteria_registered(self):
        """Both chain-level criteria are in the registry."""
        assert "delegation_fidelity" in CHAIN_CRITERIA
        assert "error_propagation" in CHAIN_CRITERIA

    def test_prompts_request_json_only(self, ctx):
        """All prompts end with a JSON-only instruction."""
        researcher = next(a for a in ctx.agents if a.agent_name == "Researcher")
        for name, fn in PER_AGENT_CRITERIA.items():
            prompt = fn(researcher, ctx)
            if prompt:
                assert "json" in prompt.lower(), f"{name} doesn't mention JSON"


# ===========================================================================
# RUNNER TESTS
# ===========================================================================

class TestRunner:
    def test_build_batch_requests(self):
        """build_batch_requests generates correct request structure."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS, status="SUCCESS")
        requests = build_batch_requests([ctx])
        assert len(requests) > 0
        for req in requests:
            assert "custom_id" in req
            assert "params" in req
            assert "model" in req["params"]
            assert "messages" in req["params"]
            assert len(req["params"]["messages"]) == 1
            assert req["params"]["messages"][0]["role"] == "user"

    def test_batch_request_count(self):
        """Correct number of requests: 4 per agent + chain criteria."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS, status="SUCCESS")
        requests = build_batch_requests([ctx])
        # 2 agents × 4 per-agent criteria = 8
        # + 1 delegation_fidelity (2 agents, 1 pair) = 1
        # = 9 total
        assert len(requests) == 9

    def test_batch_request_custom_id_format(self):
        """custom_id encodes repo|agent|criterion."""
        ctx = reconstruct_context(SYNTHETIC_EVENTS, status="SUCCESS")
        requests = build_batch_requests([ctx])
        ids = [r["custom_id"] for r in requests]
        # Should have pipe-separated parts
        for cid in ids:
            parts = cid.split("|")
            assert len(parts) == 3, f"Bad custom_id: {cid}"

    def test_try_parse_json_valid(self):
        """_try_parse_json parses valid JSON."""
        result = _try_parse_json('{"score": 3, "reasoning": "good"}')
        assert result == {"score": 3, "reasoning": "good"}

    def test_try_parse_json_with_fences(self):
        """_try_parse_json strips markdown code fences."""
        result = _try_parse_json('```json\n{"score": 2}\n```')
        assert result == {"score": 2}

    def test_try_parse_json_with_text(self):
        """_try_parse_json finds JSON in surrounding text."""
        result = _try_parse_json('Here is my evaluation:\n{"score": 1}\nDone.')
        assert result == {"score": 1}

    def test_try_parse_json_invalid(self):
        """_try_parse_json returns None for unparseable text."""
        assert _try_parse_json("This is not JSON at all") is None

    def test_build_retry_request(self):
        """build_retry_request adds reminder message."""
        orig = {
            "custom_id": "test|agent|criterion",
            "params": {
                "model": "test-model",
                "max_tokens": 100,
                "temperature": 0,
                "messages": [{"role": "user", "content": "evaluate this"}],
            },
        }
        retry = build_retry_request(orig)
        assert retry["custom_id"] == orig["custom_id"]
        assert len(retry["params"]["messages"]) == 3
        assert "not valid JSON" in retry["params"]["messages"][2]["content"]

    def test_single_agent_no_chain_criteria(self):
        """Single-agent context doesn't generate chain criteria."""
        # Only include Researcher events
        events = [e for e in SYNTHETIC_EVENTS if
                  e["event_type"] in ("execution.start", "execution.end", "patcher.status")
                  or e.get("source_node", {}).get("node_name") == "Researcher"]
        ctx = reconstruct_context(events, status="SUCCESS")
        assert len(ctx.agents) == 1
        requests = build_batch_requests([ctx])
        for req in requests:
            assert "delegation_fidelity" not in req["custom_id"]


# ===========================================================================
# AGGREGATOR TESTS
# ===========================================================================

class TestAggregator:
    def test_build_summary_empty(self):
        """build_summary handles empty results."""
        summary = build_summary([])
        assert summary["meta"]["total_repos_evaluated"] == 0
        assert summary["meta"]["total_judge_calls"] == 0

    def test_build_summary_task_adherence(self):
        """Summary computes task adherence statistics."""
        results = [
            {"repo_url": "r1", "criterion": "task_adherence", "score": 3, "framework": "crewai"},
            {"repo_url": "r1", "criterion": "task_adherence", "score": 2, "framework": "crewai"},
            {"repo_url": "r2", "criterion": "task_adherence", "score": 1, "framework": "langgraph"},
        ]
        summary = build_summary(results)
        ta = summary["per_criterion_summary"]["task_adherence"]
        assert ta["mean_score"] == 2.0
        assert ta["distribution"]["3"] == 1
        assert ta["distribution"]["2"] == 1
        assert ta["distribution"]["1"] == 1

    def test_build_summary_hallucination(self):
        """Summary computes hallucination detection rate."""
        results = [
            {"repo_url": "r1", "criterion": "hallucination", "hallucination_detected": True, "confidence": "high", "framework": "crewai"},
            {"repo_url": "r1", "criterion": "hallucination", "hallucination_detected": False, "confidence": "low", "framework": "crewai"},
            {"repo_url": "r2", "criterion": "hallucination", "hallucination_detected": True, "confidence": "medium", "framework": "langgraph"},
        ]
        summary = build_summary(results)
        hal = summary["per_criterion_summary"]["hallucination"]
        assert hal["detection_rate"] == 0.67
        assert hal["high_confidence_rate"] == 0.33

    def test_build_summary_framework_breakdown(self):
        """Summary has per-framework breakdown."""
        results = [
            {"repo_url": "r1", "criterion": "task_adherence", "score": 3, "framework": "crewai"},
            {"repo_url": "r2", "criterion": "task_adherence", "score": 1, "framework": "langgraph"},
        ]
        summary = build_summary(results)
        by_fw = summary["per_criterion_summary"]["task_adherence"]["by_framework"]
        assert "crewai" in by_fw
        assert "langgraph" in by_fw
        assert by_fw["crewai"]["mean"] == 3.0
        assert by_fw["langgraph"]["mean"] == 1.0

    def test_build_summary_delegation_fidelity(self):
        """Summary computes delegation fidelity stats."""
        results = [
            {"repo_url": "r1", "criterion": "delegation_fidelity", "score": 3},
            {"repo_url": "r1", "criterion": "delegation_fidelity", "score": 1},
            {"repo_url": "r2", "criterion": "delegation_fidelity", "score": 2},
        ]
        summary = build_summary(results)
        df = summary["per_criterion_summary"]["delegation_fidelity"]
        assert df["n_evaluated"] == 3
        assert df["mean_score"] == 2.0
        assert df["ignored_rate"] == 0.33
        assert df["full_use_rate"] == 0.33

    def test_build_summary_error_propagation(self):
        """Summary computes error propagation distribution."""
        results = [
            {"repo_url": "r1", "criterion": "error_propagation", "propagation_type": "amplified"},
            {"repo_url": "r1", "criterion": "error_propagation", "propagation_type": "corrected"},
            {"repo_url": "r2", "criterion": "error_propagation", "propagation_type": "amplified"},
        ]
        summary = build_summary(results)
        ep = summary["per_criterion_summary"]["error_propagation"]
        assert ep["n_evaluated"] == 3
        assert ep["distribution"]["amplified"] == 2
        assert ep["amplification_rate"] == 0.67

    def test_generate_headlines(self):
        """Headline generator produces human-readable strings."""
        summary = {
            "per_criterion_summary": {
                "hallucination": {
                    "detection_rate": 0.43,
                    "by_framework": {
                        "crewai": {"rate": 0.38, "n": 100},
                        "langgraph": {"rate": 0.51, "n": 80},
                    },
                },
                "instruction_leakage": {"detection_rate": 0.22},
                "output_quality": {"garbage_rate": 0.23},
                "error_propagation": {
                    "amplification_rate": 0.26,
                    "n_evaluated": 50,
                },
            }
        }
        headlines = _generate_headlines(summary)
        assert any("43%" in h for h in headlines)
        assert any("hallucin" in h.lower() for h in headlines)
        assert any("26%" in h for h in headlines)

    def test_judge_errors_excluded_from_summary(self):
        """Records with judge_error are excluded from aggregation."""
        results = [
            {"repo_url": "r1", "criterion": "task_adherence", "score": 3},
            {"repo_url": "r1", "criterion": "task_adherence", "judge_error": "malformed_json"},
        ]
        summary = build_summary(results)
        ta = summary["per_criterion_summary"]["task_adherence"]
        assert ta["distribution"]["3"] == 1
        # Only 1 valid result
        assert sum(int(v) for v in ta["distribution"].values()) == 1


# ===========================================================================
# COST TRACKER TESTS
# ===========================================================================

class TestCostTracker:
    def test_initial_state(self):
        """New tracker starts at zero."""
        ct = CostTracker()
        assert ct.total_calls == 0
        assert ct.cost_usd == 0.0

    def test_record_updates_totals(self):
        """record() accumulates tokens and calls."""
        ct = CostTracker()
        ct.record(1000, 200)
        ct.record(2000, 300)
        assert ct.total_calls == 2
        assert ct.total_input_tokens == 3000
        assert ct.total_output_tokens == 500

    def test_cost_calculation(self):
        """Cost calculated from batch API pricing."""
        ct = CostTracker()
        ct.record(1_000_000, 0)  # 1M input tokens
        assert ct.cost_usd == pytest.approx(1.50)  # $1.50/M input
        ct.record(0, 1_000_000)  # 1M output tokens
        assert ct.cost_usd == pytest.approx(1.50 + 7.50)

    def test_estimate_cost(self):
        """estimate_cost projects cost for N calls."""
        ct = CostTracker()
        # 3000 calls × 2000 input + 300 output each
        est = ct.estimate_cost(3000, avg_input=2000, avg_output=300)
        # (3000 × 2000 / 1M) × 1.50 + (3000 × 300 / 1M) × 7.50
        # = 6M/1M × 1.50 + 0.9M/1M × 7.50 = 9.00 + 6.75 = 15.75
        assert est == pytest.approx(15.75)

    def test_summary_dict(self):
        """summary() returns expected dict shape."""
        ct = CostTracker()
        ct.record(100, 50)
        s = ct.summary()
        assert "total_calls" in s
        assert "total_input_tokens" in s
        assert "total_output_tokens" in s
        assert "total_cost_usd" in s


# ===========================================================================
# CLI DISCOVERY TESTS
# ===========================================================================

class TestDiscovery:
    def test_discover_runs_with_success(self, tmp_path):
        """discover_runs finds SUCCESS repos with events."""
        from stratum_lab.judge.cli import discover_runs

        repo_dir = tmp_path / "results" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "run_metadata_1.json").write_text(json.dumps({
            "status": "SUCCESS",
            "repo_url": "https://github.com/org/repo",
        }))
        (repo_dir / "events_run_1.jsonl").write_text(
            json.dumps({"event_type": "test"}) + "\n"
        )

        runs = discover_runs(tmp_path / "results")
        assert len(runs) == 1
        assert runs[0]["status"] == "SUCCESS"

    def test_discover_runs_filters_status(self, tmp_path):
        """discover_runs respects filter_status."""
        from stratum_lab.judge.cli import discover_runs

        repo_dir = tmp_path / "results" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "run_metadata_1.json").write_text(json.dumps({
            "status": "NO_EVENTS",
        }))
        (repo_dir / "events_run_1.jsonl").write_text("")

        runs = discover_runs(tmp_path / "results", filter_status={"SUCCESS"})
        assert len(runs) == 0

    def test_discover_runs_resume_skips_existing(self, tmp_path):
        """discover_runs with resume=True skips repos with judge results."""
        from stratum_lab.judge.cli import discover_runs

        repo_dir = tmp_path / "results" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "run_metadata_1.json").write_text(json.dumps({
            "status": "SUCCESS",
        }))
        (repo_dir / "events_run_1.jsonl").write_text("")
        (repo_dir / "judge_results_1.jsonl").write_text("")  # Already judged

        runs = discover_runs(tmp_path / "results", resume=True)
        assert len(runs) == 0

    def test_discover_runs_max_repos(self, tmp_path):
        """discover_runs respects max_repos cap."""
        from stratum_lab.judge.cli import discover_runs

        for i in range(5):
            repo_dir = tmp_path / "results" / f"repo_{i}"
            repo_dir.mkdir(parents=True)
            (repo_dir / "run_metadata_1.json").write_text(json.dumps({
                "status": "SUCCESS",
            }))
            (repo_dir / "events_run_1.jsonl").write_text("")

        runs = discover_runs(tmp_path / "results", max_repos=2)
        assert len(runs) == 2

    def test_discover_runs_missing_events(self, tmp_path):
        """discover_runs skips repos without events file."""
        from stratum_lab.judge.cli import discover_runs

        repo_dir = tmp_path / "results" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "run_metadata_1.json").write_text(json.dumps({
            "status": "SUCCESS",
        }))
        # No events_run_1.jsonl

        runs = discover_runs(tmp_path / "results")
        assert len(runs) == 0
