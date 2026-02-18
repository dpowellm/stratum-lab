#!/usr/bin/env python3
"""Test that all three framework patchers emit agent.task_start/end events.

Run inside the Docker container against the live vLLM endpoint:

    docker run --rm -it \
        -e STRATUM_EVENTS_FILE=/tmp/test_events.jsonl \
        -e VLLM_HOST=http://host.docker.internal:8000 \
        -e STRATUM_VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
        stratum-lab-base \
        python /opt/stratum_patcher/test_patcher_frameworks.py

Or run locally for patcher-only testing (no LLM calls):

    STRATUM_EVENTS_FILE=/tmp/test_events.jsonl python test_patcher_frameworks.py --no-llm
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


def read_events(path: str) -> list[dict]:
    """Read JSONL events file."""
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def clear_events(path: str) -> None:
    """Clear the events file."""
    open(path, "w").close()


def check_events(events: list[dict], expected_types: list[str], label: str) -> bool:
    """Check that all expected event types are present."""
    found_types = {e["event_type"] for e in events}
    missing = set(expected_types) - found_types
    counts = Counter(e["event_type"] for e in events)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total events: {len(events)}")
    for et, count in sorted(counts.items()):
        marker = "OK" if et in expected_types else "  "
        print(f"  [{marker}] {et}: {count}")

    if missing:
        print(f"\n  MISSING: {missing}")
        return False
    else:
        print(f"\n  ALL EXPECTED EVENT TYPES PRESENT")
        return True


def test_crewai(events_file: str, use_llm: bool) -> bool:
    """Test CrewAI patcher."""
    clear_events(events_file)

    try:
        from crewai import Agent, Task, Crew
    except ImportError:
        print("\n[SKIP] CrewAI not installed")
        return True

    print("\n[TEST] CrewAI: Creating 2-agent crew...")

    if use_llm:
        agent1 = Agent(
            role="Researcher",
            goal="Research the topic",
            backstory="You are a research specialist.",
        )
        agent2 = Agent(
            role="Writer",
            goal="Write a summary",
            backstory="You are a writing specialist.",
        )
        task1 = Task(
            description="Research the history of Python programming language",
            agent=agent1,
            expected_output="A brief research summary",
        )
        task2 = Task(
            description="Summarize the research findings",
            agent=agent2,
            expected_output="A concise summary",
        )
        crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
        try:
            crew.kickoff()
        except Exception as e:
            print(f"  CrewAI execution error (expected in test): {e}")
    else:
        # Trigger patch loading without execution
        print("  [no-llm mode] Patches applied, skipping execution")

    events = read_events(events_file)
    expected = ["execution.start", "agent.task_start", "agent.task_end", "execution.end"]
    if not use_llm:
        expected = ["patcher.status"]  # Minimum when not executing
    return check_events(events, expected, "CrewAI Results")


def test_langgraph(events_file: str, use_llm: bool) -> bool:
    """Test LangGraph patcher."""
    clear_events(events_file)

    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("\n[SKIP] LangGraph not installed")
        return True

    print("\n[TEST] LangGraph: Creating 2-node graph...")

    from typing import TypedDict

    class TestState(TypedDict):
        messages: list[str]
        result: str

    def node_a(state: TestState) -> dict:
        """First processing node."""
        return {"messages": state["messages"] + ["processed by node_a"], "result": ""}

    def node_b(state: TestState) -> dict:
        """Second processing node."""
        return {"result": "final output from node_b"}

    graph = StateGraph(TestState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)

    compiled = graph.compile()

    try:
        result = compiled.invoke({"messages": ["hello"], "result": ""})
        print(f"  Graph result: {result}")
    except Exception as e:
        print(f"  LangGraph execution error: {e}")

    events = read_events(events_file)
    expected = ["execution.start", "agent.task_start", "agent.task_end", "execution.end"]
    return check_events(events, expected, "LangGraph Results")


def test_autogen(events_file: str, use_llm: bool) -> bool:
    """Test AutoGen patcher."""
    clear_events(events_file)

    ConversableAgent = None
    try:
        from autogen import ConversableAgent
    except ImportError:
        try:
            from pyautogen import ConversableAgent
        except ImportError:
            print("\n[SKIP] AutoGen not installed")
            return True

    print("\n[TEST] AutoGen: Creating 2-agent chat...")

    if use_llm:
        llm_config = {"config_list": [{"model": "gpt-4", "api_key": "sk-test"}]}
        agent1 = ConversableAgent(
            name="alice",
            system_message="You are Alice, a helpful assistant.",
            llm_config=llm_config,
        )
        agent2 = ConversableAgent(
            name="bob",
            system_message="You are Bob, another assistant.",
            llm_config=llm_config,
        )
        try:
            agent1.initiate_chat(agent2, message="Hello Bob!", max_turns=2)
        except Exception as e:
            print(f"  AutoGen execution error (expected in test): {e}")
    else:
        agent1 = ConversableAgent(
            name="alice",
            system_message="You are Alice.",
            llm_config=False,
            human_input_mode="NEVER",
        )
        agent2 = ConversableAgent(
            name="bob",
            system_message="You are Bob.",
            llm_config=False,
            human_input_mode="NEVER",
            default_auto_reply="Got it!",
        )
        try:
            agent1.initiate_chat(agent2, message="Hello Bob!", max_turns=2)
        except Exception as e:
            print(f"  AutoGen execution error: {e}")

    events = read_events(events_file)
    expected = ["execution.start", "agent.task_start", "agent.task_end", "execution.end"]
    return check_events(events, expected, "AutoGen Results")


def main() -> None:
    use_llm = "--no-llm" not in sys.argv

    # Set up events file
    events_file = os.environ.get("STRATUM_EVENTS_FILE", "")
    if not events_file:
        events_file = os.path.join(tempfile.mkdtemp(), "test_events.jsonl")
        os.environ["STRATUM_EVENTS_FILE"] = events_file
    print(f"Events file: {events_file}")
    print(f"LLM mode: {'ON' if use_llm else 'OFF (patcher-only testing)'}")

    # Import patcher (triggers all patches)
    import stratum_patcher  # noqa: F401

    results = {}
    results["langgraph"] = test_langgraph(events_file, use_llm)
    results["autogen"] = test_autogen(events_file, use_llm)
    results["crewai"] = test_crewai(events_file, use_llm)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for fw, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {fw:15s}: {status}")

    if all_pass:
        print(f"\n  ALL FRAMEWORKS PASSED")
        sys.exit(0)
    else:
        print(f"\n  SOME FRAMEWORKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
