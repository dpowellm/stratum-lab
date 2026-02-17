"""Tests for the behavioral trace export pipeline.

All tests use synthetic event fixtures matching real patcher field paths.
No network calls or API usage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from stratum_lab.export import load_events, build_repo_summary, export_behavioral_traces
from stratum_lab.validate_export import validate_export


# ===========================================================================
# Fixture helpers
# ===========================================================================

_TS_COUNTER = 1_000_000_000


def _next_ts():
    global _TS_COUNTER
    _TS_COUNTER += 500_000_000  # 0.5s increments
    return _TS_COUNTER


def make_event(event_type: str, **overrides) -> dict:
    """Build a realistic event dict with proper structure."""
    ts = _next_ts()
    evt = {
        "event_id": f"evt-{ts}",
        "timestamp_ns": ts,
        "run_id": overrides.pop("run_id", "run-1"),
        "repo_id": overrides.pop("repo_id", "https://github.com/test-org/demo-crew"),
        "framework": overrides.pop("framework", "crewai"),
        "event_type": event_type,
        "source_node": overrides.pop("source_node", {
            "node_type": "agent",
            "node_id": "crewai:Agent:agents.py:0",
            "node_name": "Agent",
        }),
        "payload": overrides.pop("payload", {}),
        "stack_depth": overrides.pop("stack_depth", 0),
    }
    if overrides.get("parent_event_id"):
        evt["parent_event_id"] = overrides.pop("parent_event_id")
    evt.update(overrides)
    return evt


def make_agent_events(
    agent_name: str,
    node_id: str,
    task_desc: str = "Do a task",
    output_text: str = "Task completed successfully.",
    agent_goal: str = "Accomplish the mission",
    tools: list | None = None,
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    input_tokens: int = 200,
    output_tokens: int = 500,
    latency_ms: float = 1500.0,
    input_source: str = "",
    parent_node_id: str = "",
    task_desc_hash: str = "hash_td_123",
    agent_goal_hash: str = "hash_ag_456",
    output_hash: str = "hash_out_789",
    system_prompt_preview: str = "You are a helpful assistant.",
    system_prompt_hash: str = "sp_hash_abc",
    last_user_message_preview: str = "Please complete the task.",
    last_user_message_hash: str = "um_hash_def",
    message_count: int = 3,
    has_tools: bool = False,
) -> list[dict]:
    """Generate a full agent span: task_start, llm_start, llm_end, task_end."""
    sn = {"node_type": "agent", "node_id": node_id, "node_name": agent_name}
    events = []

    start_payload = {
        "agent_role": agent_name,
        "agent_goal": agent_goal,
        "agent_goal_hash": agent_goal_hash,
        "task_description": task_desc,
        "task_description_hash": task_desc_hash,
        "tools_available": tools or [],
    }
    if input_source:
        start_payload["input_source"] = input_source
    if parent_node_id:
        start_payload["parent_node_id"] = parent_node_id

    events.append(make_event(
        "agent.task_start",
        source_node=sn,
        payload=start_payload,
    ))

    events.append(make_event(
        "llm.call_start",
        source_node=sn,
        payload={
            "model_requested": model,
            "system_prompt_preview": system_prompt_preview,
            "system_prompt_hash": system_prompt_hash,
            "last_user_message_preview": last_user_message_preview,
            "last_user_message_hash": last_user_message_hash,
            "message_count": message_count,
            "has_tools": has_tools,
        },
    ))

    events.append(make_event(
        "llm.call_end",
        source_node=sn,
        payload={
            "model_requested": model,
            "model_actual": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "finish_reason": "stop",
            "output_hash": output_hash,
            "output_preview": output_text[:500],
        },
    ))

    events.append(make_event(
        "agent.task_end",
        source_node=sn,
        payload={
            "status": "success",
            "output_preview": output_text,
            "output_hash": output_hash,
            "output_type": "long_text",
            "output_size_bytes": len(output_text.encode()),
        },
    ))

    return events


def make_two_agent_crew() -> list[dict]:
    """Generate a realistic 2-agent crew with delegation."""
    global _TS_COUNTER
    _TS_COUNTER = 1_000_000_000

    events = [
        make_event("execution.start", source_node={}, payload={
            "repo_url": "https://github.com/test-org/demo-crew",
            "framework": "crewai",
        }),
    ]

    events += make_agent_events(
        agent_name="Content Planner",
        node_id="crewai:Content Planner:agents.py:0",
        task_desc="Plan content about AI trends",
        output_text="Here is the content plan:\n1. Introduction to AI trends\n2. Key developments in 2025\n3. Future outlook",
        agent_goal="Plan engaging content on given topics",
        tools=["search_tool", "web_scraper"],
        input_tokens=209,
        output_tokens=810,
        latency_ms=22150.33,
        task_desc_hash="td_hash_planner",
        agent_goal_hash="ag_hash_planner",
        output_hash="e2e97aa_planner",
    )

    sn_planner = {"node_type": "agent", "node_id": "crewai:Content Planner:agents.py:0", "node_name": "Content Planner"}
    sn_writer = {"node_type": "agent", "node_id": "crewai:Content Writer:agents.py:1", "node_name": "Content Writer"}

    events.append(make_event(
        "delegation.initiated",
        source_node=sn_planner,
        target_node=sn_writer,
        payload={"delegator": "Content Planner", "delegate": "Content Writer"},
    ))

    events += make_agent_events(
        agent_name="Content Writer",
        node_id="crewai:Content Writer:agents.py:1",
        task_desc="Write article based on content plan",
        output_text="AI is transforming industries at an unprecedented pace. In 2025, we see breakthroughs in multi-agent systems...",
        agent_goal="Write compelling articles based on research",
        tools=["write_tool"],
        input_tokens=1500,
        output_tokens=2000,
        latency_ms=35000.0,
        input_source="delegation",
        parent_node_id="crewai:Content Planner:agents.py:0",
        task_desc_hash="td_hash_writer",
        agent_goal_hash="ag_hash_writer",
        output_hash="f3a8bcd_writer",
    )

    events.append(make_event("execution.end", source_node={}, payload={}))
    return events


def write_repo(tmp_path, events, status="SUCCESS", run_number=1, repo_hash="abc123"):
    """Write a repo directory with metadata and events."""
    repo_dir = tmp_path / "results" / repo_hash
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / f"run_metadata_{run_number}.json").write_text(json.dumps({
        "status": status,
        "repo_url": "https://github.com/test-org/demo-crew",
    }))
    with open(repo_dir / f"events_run_{run_number}.jsonl", "w") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return repo_dir


# ===========================================================================
# TestLoadEvents
# ===========================================================================

class TestLoadEvents:
    def test_load_valid_jsonl(self, tmp_path):
        """Loads events file, returns list sorted by timestamp."""
        events = make_two_agent_crew()
        path = tmp_path / "events.jsonl"
        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        loaded = load_events(path)
        assert len(loaded) == len(events)
        ts_list = [e.get("timestamp_ns", 0) for e in loaded]
        assert ts_list == sorted(ts_list)

    def test_load_skips_malformed(self, tmp_path):
        """Malformed lines skipped, valid lines loaded."""
        path = tmp_path / "events.jsonl"
        with open(path, "w") as f:
            f.write('{"event_type": "ok", "timestamp_ns": 1}\n')
            f.write("NOT JSON\n")
            f.write('{"event_type": "also_ok", "timestamp_ns": 2}\n')
        loaded = load_events(path)
        assert len(loaded) == 2

    def test_load_missing_file(self):
        """Returns empty list for missing file."""
        assert load_events(Path("/nonexistent/path.jsonl")) == []

    def test_load_empty_file(self, tmp_path):
        """Returns empty list for empty file."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_events(path) == []


# ===========================================================================
# TestBuildRepoSummary
# ===========================================================================

class TestBuildRepoSummary:
    def test_basic_two_agent_crew(self, tmp_path):
        """2 agents, sequential tasks, LLM calls for each."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        assert summary is not None
        assert summary["repo_id"] == "https://github.com/test-org/demo-crew"
        assert summary["framework"] == "crewai"
        assert summary["run_count"] == 1
        run = summary["runs"][0]
        assert len(run["agents"]) == 2
        assert run["agents"][0]["agent_name"] == "Content Planner"
        assert run["agents"][1]["agent_name"] == "Content Writer"

    def test_agent_name_from_agent_role(self, tmp_path):
        """payload.agent_role used as agent_name."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        # agent_role is set in fixture, should be agent_name
        assert summary["runs"][0]["agents"][0]["agent_role"] == "Content Planner"

    def test_agent_name_fallback_to_node_name(self, tmp_path):
        """When agent_role missing, source_node.node_name used."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:Fallback:f.py:0", "node_name": "FallbackAgent"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "task_description": "do stuff",
                "task_description_hash": "h1",
            }),
            make_event("agent.task_end", source_node=sn, payload={
                "output_preview": "done",
                "output_hash": "oh1",
                "status": "success",
            }),
            make_event("execution.end", source_node={}, payload={}),
        ]
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        assert summary["runs"][0]["agents"][0]["agent_name"] == "FallbackAgent"

    def test_output_text_from_output_preview(self, tmp_path):
        """payload.output_preview extracted (NOT payload.preview)."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert "content plan" in task["output_text"]

    def test_task_description_extracted(self, tmp_path):
        """payload.task_description from agent.task_start."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert task["task_description"] == "Plan content about AI trends"
        assert task["task_description_hash"] == "td_hash_planner"

    def test_task_description_missing_graceful(self, tmp_path):
        """When task_description absent (first 35 repos), empty string used."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:A:a.py:0", "node_name": "OldAgent"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "agent_role": "OldAgent",
                "task_description_hash": "legacy_hash",
                # no task_description field
            }),
            make_event("agent.task_end", source_node=sn, payload={
                "output_preview": "output",
                "output_hash": "oh",
                "status": "success",
            }),
            make_event("execution.end", source_node={}, payload={}),
        ]
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert task["task_description"] == ""
        assert task["task_description_hash"] == "legacy_hash"

    def test_agent_goal_extracted(self, tmp_path):
        """payload.agent_goal from agent.task_start."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        agent = summary["runs"][0]["agents"][0]
        assert agent["agent_goal"] == "Plan engaging content on given topics"
        assert agent["agent_goal_hash"] == "ag_hash_planner"

    def test_tools_available_extracted(self, tmp_path):
        """payload.tools_available from agent.task_start."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        agent = summary["runs"][0]["agents"][0]
        assert agent["tools_available"] == ["search_tool", "web_scraper"]

    def test_llm_calls_nested_in_agent(self, tmp_path):
        """LLM calls correctly associated with parent agent."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        planner = summary["runs"][0]["agents"][0]
        assert len(planner["tasks"][0]["llm_calls"]) == 1
        writer = summary["runs"][0]["agents"][1]
        assert len(writer["tasks"][0]["llm_calls"]) == 1

    def test_llm_call_fields(self, tmp_path):
        """All LLM call fields extracted correctly."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["model_requested"] == "mistralai/Mistral-7B-Instruct-v0.3"
        assert llm["model_actual"] == "mistralai/Mistral-7B-Instruct-v0.3"
        assert llm["input_tokens"] == 209
        assert llm["output_tokens"] == 810
        assert llm["latency_ms"] == 22150.33
        assert llm["finish_reason"] == "stop"

    def test_delegation_chain_from_events(self, tmp_path):
        """delegation.initiated events produce correct chains."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        chains = summary["runs"][0]["delegation_chains"]
        assert len(chains) == 1
        assert chains[0]["upstream_agent"] == "Content Planner"
        assert chains[0]["downstream_agent"] == "Content Writer"

    def test_delegation_chain_inferred(self, tmp_path):
        """When no delegation events, infer from input_source=='delegation'."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        # Agent A (no delegation events)
        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="AgentA",
            node_id="crewai:AgentA:a.py:0",
            task_desc="first task",
            output_text="done first",
            output_hash="hash_a",
        )
        # Agent B with input_source=delegation (but no delegation.initiated event)
        events += make_agent_events(
            agent_name="AgentB",
            node_id="crewai:AgentB:a.py:1",
            task_desc="second task",
            output_text="done second",
            input_source="delegation",
            parent_node_id="crewai:AgentA:a.py:0",
            output_hash="hash_b",
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        chains = summary["runs"][0]["delegation_chains"]
        assert len(chains) == 1
        assert chains[0]["upstream_agent"] == "AgentA"
        assert chains[0]["downstream_agent"] == "AgentB"

    def test_execution_summary_totals(self, tmp_path):
        """Totals computed correctly."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        es = summary["runs"][0]["execution_summary"]
        assert es["total_agents"] == 2
        assert es["total_tasks"] == 2
        assert es["total_llm_calls"] == 2
        assert es["total_input_tokens"] == 209 + 1500
        assert es["total_output_tokens"] == 810 + 2000

    def test_execution_summary_delegation_depth(self, tmp_path):
        """delegation_depth computed from chain length."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        assert summary["runs"][0]["execution_summary"]["delegation_depth"] == 1

    def test_incomplete_agent(self, tmp_path):
        """agent.task_start without matching agent.task_end handled gracefully."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:Crash:c.py:0", "node_name": "CrashAgent"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "agent_role": "CrashAgent",
                "task_description": "will crash",
                "task_description_hash": "h1",
            }),
            # No agent.task_end — crashed
            make_event("execution.end", source_node={}, payload={}),
        ]
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert task["status"] == "incomplete"
        assert task["output_text"] == ""

    def test_no_llm_calls(self, tmp_path):
        """Agent with no LLM calls (e.g. routing agent)."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:Router:r.py:0", "node_name": "Router"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "agent_role": "Router",
                "task_description": "route request",
                "task_description_hash": "h_route",
            }),
            make_event("agent.task_end", source_node=sn, payload={
                "output_preview": "routed to agent B",
                "output_hash": "oh_route",
                "status": "success",
            }),
            make_event("execution.end", source_node={}, payload={}),
        ]
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert task["llm_calls"] == []
        assert task["output_text"] == "routed to agent B"

    def test_empty_output(self, tmp_path):
        """Empty output_preview handled (empty string, not None)."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:Empty:e.py:0", "node_name": "EmptyAgent"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "agent_role": "EmptyAgent",
                "task_description": "produce nothing",
                "task_description_hash": "h_empty",
            }),
            make_event("agent.task_end", source_node=sn, payload={
                "output_preview": "",
                "output_hash": "",
                "status": "success",
            }),
            make_event("execution.end", source_node={}, payload={}),
        ]
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        task = summary["runs"][0]["agents"][0]["tasks"][0]
        assert task["output_text"] == ""
        assert isinstance(task["output_text"], str)

    def test_skip_failed_run(self, tmp_path):
        """run_metadata with status FAILED returns None."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events, status="RUNTIME_ERROR")
        summary = build_repo_summary(repo_dir)
        assert summary is None

    def test_multiple_runs(self, tmp_path):
        """Directory with events_run_1 and events_run_2 produces 2 runs."""
        global _TS_COUNTER
        events1 = make_two_agent_crew()
        _TS_COUNTER = 1_000_000_000
        events2 = make_two_agent_crew()

        repo_dir = tmp_path / "results" / "multi_run"
        repo_dir.mkdir(parents=True)
        for run_num, evts in [(1, events1), (2, events2)]:
            (repo_dir / f"run_metadata_{run_num}.json").write_text(json.dumps({
                "status": "SUCCESS",
                "repo_url": "https://github.com/test-org/demo-crew",
            }))
            with open(repo_dir / f"events_run_{run_num}.jsonl", "w") as f:
                for e in evts:
                    f.write(json.dumps(e) + "\n")

        summary = build_repo_summary(repo_dir)
        assert summary is not None
        assert summary["run_count"] == 2
        assert len(summary["runs"]) == 2
        assert summary["runs"][0]["run_number"] == 1
        assert summary["runs"][1]["run_number"] == 2

    def test_single_agent(self, tmp_path):
        """Single agent repo (no delegation chains)."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="Solo",
            node_id="crewai:Solo:s.py:0",
            task_desc="work alone",
            output_text="did it solo",
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        assert len(summary["runs"][0]["agents"]) == 1
        assert summary["runs"][0]["delegation_chains"] == []
        assert summary["runs"][0]["execution_summary"]["has_delegation"] is False

    def test_three_agent_chain(self, tmp_path):
        """3 agents in sequence with delegation."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn_a = {"node_type": "agent", "node_id": "crewai:A:a.py:0", "node_name": "A"}
        sn_b = {"node_type": "agent", "node_id": "crewai:B:a.py:1", "node_name": "B"}
        sn_c = {"node_type": "agent", "node_id": "crewai:C:a.py:2", "node_name": "C"}

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events("A", "crewai:A:a.py:0", output_hash="h_a")
        events.append(make_event("delegation.initiated", source_node=sn_a, target_node=sn_b, payload={}))
        events += make_agent_events("B", "crewai:B:a.py:1", input_source="delegation", output_hash="h_b")
        events.append(make_event("delegation.initiated", source_node=sn_b, target_node=sn_c, payload={}))
        events += make_agent_events("C", "crewai:C:a.py:2", input_source="delegation", output_hash="h_c")
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        assert len(summary["runs"][0]["agents"]) == 3
        chains = summary["runs"][0]["delegation_chains"]
        assert len(chains) == 2
        assert chains[0]["upstream_agent"] == "A"
        assert chains[0]["downstream_agent"] == "B"
        assert chains[1]["upstream_agent"] == "B"
        assert chains[1]["downstream_agent"] == "C"
        assert summary["runs"][0]["execution_summary"]["delegation_depth"] == 2


# ===========================================================================
# TestExportBehavioralTraces
# ===========================================================================

class TestExportBehavioralTraces:
    def test_export_writes_jsonl(self, tmp_path):
        """Creates output file with one JSON line per repo."""
        events = make_two_agent_crew()
        write_repo(tmp_path, events, repo_hash="repo_1")
        output = tmp_path / "traces.jsonl"
        export_behavioral_traces(tmp_path / "results", output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["repo_id"] == "https://github.com/test-org/demo-crew"

    def test_export_skips_failed(self, tmp_path):
        """Repos with FAILED status not in output."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000
        events = make_two_agent_crew()
        write_repo(tmp_path, events, status="RUNTIME_ERROR", repo_hash="fail_1")
        output = tmp_path / "traces.jsonl"
        stats = export_behavioral_traces(tmp_path / "results", output)
        assert stats["total_repos_exported"] == 0
        assert stats["skipped_count"] == 1

    def test_export_includes_success_and_partial(self, tmp_path):
        """Both SUCCESS and PARTIAL_SUCCESS included."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000
        events1 = make_two_agent_crew()
        write_repo(tmp_path, events1, status="SUCCESS", repo_hash="s1")

        _TS_COUNTER = 1_000_000_000
        events2 = make_two_agent_crew()
        # Change repo_id for second repo to avoid duplicate
        for e in events2:
            e["repo_id"] = "https://github.com/test-org/demo-crew-2"
        repo_dir2 = tmp_path / "results" / "s2"
        repo_dir2.mkdir(parents=True)
        (repo_dir2 / "run_metadata_1.json").write_text(json.dumps({
            "status": "PARTIAL_SUCCESS",
            "repo_url": "https://github.com/test-org/demo-crew-2",
        }))
        with open(repo_dir2 / "events_run_1.jsonl", "w") as f:
            for e in events2:
                f.write(json.dumps(e) + "\n")

        output = tmp_path / "traces.jsonl"
        stats = export_behavioral_traces(tmp_path / "results", output)
        assert stats["total_repos_exported"] == 2

    def test_export_returns_statistics(self, tmp_path):
        """Returns dict with total_repos, success_count, etc."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000
        events = make_two_agent_crew()
        write_repo(tmp_path, events, repo_hash="stat_1")
        output = tmp_path / "traces.jsonl"
        stats = export_behavioral_traces(tmp_path / "results", output)
        assert "total_repos_scanned" in stats
        assert "total_repos_exported" in stats
        assert "total_agents" in stats
        assert "total_llm_calls" in stats
        assert "frameworks" in stats
        assert stats["total_repos_exported"] == 1

    def test_export_include_raw_copies_events(self, tmp_path):
        """With include_raw=True, raw events copied to raw_events/ dir."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000
        events = make_two_agent_crew()
        write_repo(tmp_path, events, repo_hash="raw_1")
        output = tmp_path / "traces.jsonl"
        export_behavioral_traces(tmp_path / "results", output, include_raw=True)
        raw_dir = tmp_path / "raw_events" / "raw_1"
        assert raw_dir.exists()
        assert (raw_dir / "events_run_1.jsonl").exists()

    def test_export_no_duplicates(self, tmp_path):
        """Same repo not exported twice."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000
        events = make_two_agent_crew()
        # Two directories with same repo_id
        write_repo(tmp_path, events, repo_hash="dup_1")
        write_repo(tmp_path, events, repo_hash="dup_2")
        output = tmp_path / "traces.jsonl"
        stats = export_behavioral_traces(tmp_path / "results", output)
        assert stats["total_repos_exported"] == 1
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1


# ===========================================================================
# TestValidateExport
# ===========================================================================

class TestValidateExport:
    def _write_valid_export(self, tmp_path) -> Path:
        """Helper: write a valid behavioral_traces.jsonl."""
        path = tmp_path / "traces.jsonl"
        rec = {
            "repo_id": "https://github.com/org/repo",
            "repo_hash": "abc123",
            "framework": "crewai",
            "scan_status": "SUCCESS",
            "run_count": 1,
            "runs": [{
                "run_number": 1,
                "agents": [{
                    "agent_name": "Agent",
                    "agent_role": "Agent",
                    "agent_goal": "do stuff",
                    "agent_goal_hash": "h1",
                    "node_id": "crewai:Agent:a.py:0",
                    "tools_available": [],
                    "tasks": [{
                        "task_description": "a task",
                        "task_description_hash": "th1",
                        "status": "success",
                        "latency_ms": 1000.0,
                        "output_text": "output here",
                        "output_hash": "oh1",
                        "output_type": "long_text",
                        "output_size_bytes": 11,
                        "llm_calls": [],
                    }],
                }],
                "delegation_chains": [],
                "execution_summary": {
                    "total_agents": 1,
                    "total_tasks": 1,
                    "total_llm_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_duration_ms": 5000.0,
                    "has_delegation": False,
                    "delegation_depth": 0,
                    "errors": [],
                },
            }],
        }
        path.write_text(json.dumps(rec) + "\n")
        return path

    def test_valid_export_passes(self, tmp_path):
        """Well-formed export passes all checks."""
        path = self._write_valid_export(tmp_path)
        result = validate_export(path)
        assert result.valid is True
        assert result.total_repos == 1
        assert len(result.errors) == 0

    def test_invalid_json_detected(self, tmp_path):
        """Line with bad JSON flagged."""
        path = tmp_path / "traces.jsonl"
        path.write_text("NOT JSON\n")
        result = validate_export(path)
        assert result.valid is False
        assert any("invalid JSON" in e for e in result.errors)

    def test_missing_output_text_flagged(self, tmp_path):
        """Repo with all empty outputs flagged as warning."""
        path = tmp_path / "traces.jsonl"
        rec = {
            "repo_id": "https://github.com/org/empty",
            "repo_hash": "e1",
            "framework": "crewai",
            "scan_status": "SUCCESS",
            "runs": [{"agents": [{"agent_name": "A", "tasks": [
                {"output_text": "", "llm_calls": []}
            ]}], "delegation_chains": [], "run_number": 1}],
        }
        path.write_text(json.dumps(rec) + "\n")
        result = validate_export(path)
        assert any("no non-empty output_text" in w for w in result.warnings)

    def test_invalid_repo_id_flagged(self, tmp_path):
        """Non-GitHub URL flagged."""
        path = tmp_path / "traces.jsonl"
        rec = {
            "repo_id": "not-a-url",
            "repo_hash": "x",
            "framework": "crewai",
            "scan_status": "SUCCESS",
            "runs": [],
        }
        path.write_text(json.dumps(rec) + "\n")
        result = validate_export(path)
        assert result.valid is False
        assert any("invalid repo_id" in e for e in result.errors)

    def test_duplicate_repo_id_flagged(self, tmp_path):
        """Duplicate repo_id detected."""
        path = tmp_path / "traces.jsonl"
        rec = {
            "repo_id": "https://github.com/org/dup",
            "repo_hash": "d1",
            "framework": "crewai",
            "scan_status": "SUCCESS",
            "runs": [{"agents": [], "delegation_chains": [], "run_number": 1}],
        }
        path.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
        result = validate_export(path)
        assert result.valid is False
        assert any("duplicate" in e for e in result.errors)

    def test_dangling_delegation_reference(self, tmp_path):
        """Delegation chain referencing non-existent agent flagged."""
        path = tmp_path / "traces.jsonl"
        rec = {
            "repo_id": "https://github.com/org/dangling",
            "repo_hash": "dg",
            "framework": "crewai",
            "scan_status": "SUCCESS",
            "runs": [{
                "run_number": 1,
                "agents": [{"agent_name": "A", "tasks": [
                    {"output_text": "ok", "llm_calls": []}
                ]}],
                "delegation_chains": [{
                    "upstream_agent": "A",
                    "downstream_agent": "NonExistent",
                }],
            }],
        }
        path.write_text(json.dumps(rec) + "\n")
        result = validate_export(path)
        assert result.valid is False
        assert any("non-existent" in e for e in result.errors)

    def test_missing_required_fields(self, tmp_path):
        """Missing repo_id or runs flagged."""
        path = tmp_path / "traces.jsonl"
        path.write_text(json.dumps({"framework": "crewai"}) + "\n")
        result = validate_export(path)
        assert result.valid is False
        assert any("missing required" in e for e in result.errors)

    def test_statistics_computed(self, tmp_path):
        """Validation returns correct summary statistics."""
        path = self._write_valid_export(tmp_path)
        result = validate_export(path)
        assert result.total_repos == 1
        assert result.success_count == 1
        assert len(result.agents_per_repo) == 1
        assert result.agents_per_repo[0] == 1
        assert len(result.output_lengths) == 1
        assert result.frameworks["crewai"] == 1


# ===========================================================================
# TestLLMInputFields
# ===========================================================================

class TestLLMInputFields:
    """Tests for input-side fields from llm.call_start paired with llm.call_end."""

    def test_input_fields_present(self, tmp_path):
        """LLM call records include input-side fields from llm.call_start."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert "system_prompt_preview" in llm
        assert "system_prompt_hash" in llm
        assert "last_user_message_preview" in llm
        assert "last_user_message_hash" in llm
        assert "message_count" in llm
        assert "has_tools" in llm

    def test_input_fields_values(self, tmp_path):
        """Input-side fields carry correct values from llm.call_start payload."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="TestAgent",
            node_id="crewai:TestAgent:t.py:0",
            system_prompt_preview="You are an expert researcher.",
            system_prompt_hash="sp_hash_research",
            last_user_message_preview="Find the latest AI papers.",
            last_user_message_hash="um_hash_papers",
            message_count=5,
            has_tools=True,
            tools=["search_tool"],
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["system_prompt_preview"] == "You are an expert researcher."
        assert llm["system_prompt_hash"] == "sp_hash_research"
        assert llm["last_user_message_preview"] == "Find the latest AI papers."
        assert llm["last_user_message_hash"] == "um_hash_papers"
        assert llm["message_count"] == 5
        assert llm["has_tools"] is True

    def test_input_fields_defaults_without_start(self, tmp_path):
        """LLM call without matching llm.call_start gets empty defaults."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        sn = {"node_type": "agent", "node_id": "crewai:NoStart:n.py:0", "node_name": "NoStart"}
        events = [
            make_event("execution.start", source_node={}, payload={}),
            make_event("agent.task_start", source_node=sn, payload={
                "agent_role": "NoStart",
                "task_description": "task without llm start",
                "task_description_hash": "h_ns",
            }),
            # No llm.call_start — only llm.call_end
            make_event("llm.call_end", source_node=sn, payload={
                "model_requested": "gpt-4",
                "model_actual": "gpt-4",
                "input_tokens": 100,
                "output_tokens": 200,
                "latency_ms": 500.0,
                "finish_reason": "stop",
                "output_hash": "oh_ns",
                "output_preview": "response",
            }),
            make_event("agent.task_end", source_node=sn, payload={
                "status": "success",
                "output_preview": "response",
                "output_hash": "oh_ns",
            }),
            make_event("execution.end", source_node={}, payload={}),
        ]

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["system_prompt_preview"] == ""
        assert llm["system_prompt_hash"] == ""
        assert llm["last_user_message_preview"] == ""
        assert llm["last_user_message_hash"] == ""
        assert llm["message_count"] == 0
        assert llm["has_tools"] is False

    def test_input_fields_per_agent_pairing(self, tmp_path):
        """Each agent's llm.call_start pairs with its own llm.call_end."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)

        planner_llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        writer_llm = summary["runs"][0]["agents"][1]["tasks"][0]["llm_calls"][0]

        # Both agents should have their own input fields (defaults from make_agent_events)
        assert planner_llm["system_prompt_preview"] == "You are a helpful assistant."
        assert writer_llm["system_prompt_preview"] == "You are a helpful assistant."
        assert planner_llm["message_count"] == 3
        assert writer_llm["message_count"] == 3

    def test_has_tools_false_by_default(self, tmp_path):
        """has_tools defaults to False when not set in llm.call_start."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="NoTools",
            node_id="crewai:NoTools:nt.py:0",
            has_tools=False,
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["has_tools"] is False

    def test_has_tools_true(self, tmp_path):
        """has_tools=True when tools are present."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="WithTools",
            node_id="crewai:WithTools:wt.py:0",
            has_tools=True,
            tools=["calculator", "web_search"],
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["has_tools"] is True

    def test_empty_system_prompt(self, tmp_path):
        """Empty system prompt captured as empty string."""
        global _TS_COUNTER
        _TS_COUNTER = 1_000_000_000

        events = [make_event("execution.start", source_node={}, payload={})]
        events += make_agent_events(
            agent_name="EmptySP",
            node_id="crewai:EmptySP:esp.py:0",
            system_prompt_preview="",
            system_prompt_hash="",
        )
        events.append(make_event("execution.end", source_node={}, payload={}))

        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        assert llm["system_prompt_preview"] == ""
        assert llm["system_prompt_hash"] == ""

    def test_output_fields_preserved(self, tmp_path):
        """Existing output-side fields still correct after adding input fields."""
        events = make_two_agent_crew()
        repo_dir = write_repo(tmp_path, events)
        summary = build_repo_summary(repo_dir)
        llm = summary["runs"][0]["agents"][0]["tasks"][0]["llm_calls"][0]
        # Output-side fields unchanged
        assert llm["model_requested"] == "mistralai/Mistral-7B-Instruct-v0.3"
        assert llm["model_actual"] == "mistralai/Mistral-7B-Instruct-v0.3"
        assert llm["input_tokens"] == 209
        assert llm["output_tokens"] == 810
        assert llm["latency_ms"] == 22150.33
        assert llm["finish_reason"] == "stop"
        assert llm["output_hash"] == "e2e97aa_planner"
