import pytest


@pytest.fixture
def sample_structural_graph():
    """A 4-agent crewAI system with tools, data stores, external services, and guardrails."""
    return {
        "repo_id": "test_repo",
        "framework": "crewai",
        "nodes": {
            "agent_researcher": {"node_type": "agent", "name": "Researcher", "source_file": "agents.py", "line_number": 10, "error_handling": {"strategy": "fail_silent"}, "timeout_config": "none"},
            "agent_writer": {"node_type": "agent", "name": "Writer", "source_file": "agents.py", "line_number": 30},
            "agent_reviewer": {"node_type": "agent", "name": "Reviewer", "source_file": "agents.py", "line_number": 50},
            "agent_manager": {"node_type": "agent", "name": "Manager", "source_file": "crew.py", "line_number": 1},
            "cap_web_search": {"node_type": "capability", "name": "WebSearch", "class_name": "WebSearch", "kind": "tool", "source_file": "tools.py", "line_number": 5},
            "cap_file_writer": {"node_type": "capability", "name": "FileWriter", "class_name": "FileWriter", "kind": "tool", "source_file": "tools.py", "line_number": 25},
            "cap_llm_call": {"node_type": "capability", "name": "LLMCall", "class_name": "LLMCall", "kind": "llm"},
            "ds_shared_memory": {"node_type": "data_store", "name": "SharedMemory", "source_file": "memory.py", "line_number": 1},
            "ext_web_api": {"node_type": "external", "name": "WebAPI", "service_domain": "api.search.com"},
            "guard_quality": {"node_type": "guardrail", "name": "QualityGuard", "kind": "output_validation"},
        },
        "edges": {
            "e1": {"edge_type": "delegates_to", "source": "agent_manager", "target": "agent_researcher"},
            "e2": {"edge_type": "delegates_to", "source": "agent_manager", "target": "agent_writer"},
            "e3": {"edge_type": "delegates_to", "source": "agent_manager", "target": "agent_reviewer"},
            "e4": {"edge_type": "uses", "source": "agent_researcher", "target": "cap_web_search"},
            "e5": {"edge_type": "uses", "source": "agent_writer", "target": "cap_file_writer"},
            "e6": {"edge_type": "reads_from", "source": "agent_researcher", "target": "ds_shared_memory"},
            "e7": {"edge_type": "writes_to", "source": "agent_writer", "target": "ds_shared_memory"},
            "e8": {"edge_type": "writes_to", "source": "agent_researcher", "target": "ds_shared_memory"},
            "e9": {"edge_type": "calls", "source": "cap_web_search", "target": "ext_web_api"},
            "e10": {"edge_type": "filtered_by", "source": "agent_reviewer", "target": "guard_quality"},
        },
        "taxonomy_preconditions": [
            "shared_state_no_arbitration",
            "no_timeout_on_delegation",
            "unhandled_tool_failure",
        ],
    }


@pytest.fixture
def sample_events():
    """Generate events for 3 runs covering all event types."""
    events = []
    for run_idx in range(3):
        run_id = f"run_{run_idx:03d}"
        base_ts = 1000000000000 + run_idx * 100000000
        ts = base_ts

        # Agent events
        for agent_name in ["Researcher", "Writer", "Reviewer"]:
            start_id = f"evt_start_{agent_name}_{run_idx}"
            events.append({
                "event_id": start_id,
                "timestamp_ns": ts,
                "run_id": run_id,
                "repo_id": "test_repo",
                "event_type": "agent.task_start",
                "source_node": {"node_type": "agent", "node_id": f"agent_{agent_name.lower()}", "node_name": agent_name},
            })
            ts += 1000000
            events.append({
                "event_id": f"evt_end_{agent_name}_{run_idx}",
                "timestamp_ns": ts,
                "run_id": run_id,
                "repo_id": "test_repo",
                "event_type": "agent.task_end",
                "source_node": {"node_type": "agent", "node_id": f"agent_{agent_name.lower()}", "node_name": agent_name},
                "parent_event_id": start_id,
            })
            ts += 1000000

        # Tool events
        tool_start_id = f"evt_tool_start_{run_idx}"
        events.append({
            "event_id": tool_start_id,
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "tool.invoked",
            "source_node": {"node_type": "capability", "node_id": "cap_web_search", "node_name": "WebSearch"},
            "payload": {"agent_id": "Researcher", "tool_name": "WebSearch"},
        })
        ts += 500000
        events.append({
            "event_id": f"evt_tool_end_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "tool.completed",
            "source_node": {"node_type": "capability", "node_id": "cap_web_search", "node_name": "WebSearch"},
            "parent_event_id": tool_start_id,
            "payload": {"agent_id": "Researcher", "tool_name": "WebSearch", "status": "success"},
        })
        ts += 500000

        # Data events
        events.append({
            "event_id": f"evt_data_read_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "data.read",
            "source_node": {"node_type": "agent", "node_id": "agent_researcher", "node_name": "Researcher"},
            "target_node": {"node_type": "data_store", "node_id": "ds_shared_memory", "node_name": "SharedMemory"},
        })
        ts += 500000
        events.append({
            "event_id": f"evt_data_write_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "data.write",
            "source_node": {"node_type": "agent", "node_id": "agent_writer", "node_name": "Writer"},
            "target_node": {"node_type": "data_store", "node_id": "ds_shared_memory", "node_name": "SharedMemory"},
        })
        ts += 500000

        # LLM events
        llm_start_id = f"evt_llm_start_{run_idx}"
        events.append({
            "event_id": llm_start_id,
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "llm.call_start",
            "source_node": {"node_type": "capability", "node_id": "cap_llm_call", "node_name": "LLMCall"},
            "payload": {"agent_id": "Researcher"},
        })
        ts += 2000000
        events.append({
            "event_id": f"evt_llm_end_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "llm.call_end",
            "source_node": {"node_type": "capability", "node_id": "cap_llm_call", "node_name": "LLMCall"},
            "parent_event_id": llm_start_id,
            "payload": {"agent_id": "Researcher", "input_tokens": 100, "output_tokens": 50},
        })
        ts += 500000

        # External call
        events.append({
            "event_id": f"evt_external_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "external.call",
            "source_node": {"node_type": "capability", "node_id": "cap_web_search", "node_name": "WebSearch"},
            "target_node": {"node_type": "external", "node_id": "ext_web_api", "node_name": "WebAPI"},
        })
        ts += 500000

        # Delegation events
        for target_agent in ["Researcher", "Writer", "Reviewer"]:
            events.append({
                "event_id": f"evt_deleg_{target_agent}_{run_idx}",
                "timestamp_ns": ts,
                "run_id": run_id,
                "repo_id": "test_repo",
                "event_type": "delegation.initiated",
                "source_node": {"node_type": "agent", "node_id": "agent_manager", "node_name": "Manager"},
                "target_node": {"node_type": "agent", "node_id": f"agent_{target_agent.lower()}", "node_name": target_agent},
            })
            ts += 200000

        # Decision event
        events.append({
            "event_id": f"evt_decision_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "decision.made",
            "source_node": {"node_type": "agent", "node_id": "agent_manager", "node_name": "Manager"},
            "payload": {
                "selected_option_hash": "abc123" if run_idx < 2 else "def456",
                "confidence": 0.85,
                "input_hash": "input_001" if run_idx < 2 else "input_002",
            },
        })
        ts += 500000

        # Guardrail event
        events.append({
            "event_id": f"evt_guardrail_{run_idx}",
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": "test_repo",
            "event_type": "guardrail.triggered",
            "source_node": {"node_type": "guardrail", "node_id": "guard_quality", "node_name": "QualityGuard"},
            "payload": {"action_prevented": run_idx == 0, "bypassed": run_idx == 1, "retry_triggered": run_idx == 2, "latency_ms": 5.0},
        })
        ts += 500000

        # Error event (only on first run)
        if run_idx == 0:
            events.append({
                "event_id": f"evt_error_{run_idx}",
                "timestamp_ns": ts,
                "run_id": run_id,
                "repo_id": "test_repo",
                "event_type": "error.occurred",
                "source_node": {"node_type": "agent", "node_id": "agent_researcher", "node_name": "Researcher"},
                "payload": {"error_type": "timeout", "error_handling": "fail_silent"},
            })
            ts += 200000
            events.append({
                "event_id": f"evt_error_prop_{run_idx}",
                "timestamp_ns": ts,
                "run_id": run_id,
                "repo_id": "test_repo",
                "event_type": "error.propagated",
                "source_node": {"node_type": "agent", "node_id": "agent_researcher", "node_name": "Researcher"},
                "payload": {
                    "error_type": "timeout",
                    "propagation_path": ["agent_researcher", "ds_shared_memory"],
                    "downstream_impact": True,
                },
            })

    return events


@pytest.fixture
def sample_run_records(sample_events):
    """Build run records from sample events, with events attached."""
    from collections import defaultdict
    runs = defaultdict(list)
    for ev in sample_events:
        runs[ev["run_id"]].append(ev)

    records = []
    for run_id, evts in sorted(runs.items()):
        records.append({
            "run_id": run_id,
            "repo_id": "test_repo",
            "framework": "crewai",
            "events": evts,
            "metadata": {"input_hash": "input_001" if "000" in run_id or "001" in run_id else "input_002"},
        })
    return records
