"""Evaluation script for the input generator.

Tests generate_inputs() with a mocked vLLM endpoint, plan_runs(), and
find_example_inputs() against a temporary repo directory.

Prints:
  - The generated inputs (5)
  - Diversity check (all different)
  - Parseability check
  - Run plan from plan_runs()
  - Example inputs found by find_example_inputs()
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.harness.input_generator import (
    generate_inputs,
    find_example_inputs,
    input_hash,
    plan_runs,
)

# =========================================================================
# 1.  Synthetic README and entry-point code
# =========================================================================

SYNTHETIC_README = """\
# Multi-Agent Research Assistant

A CrewAI application that coordinates a team of agents to research topics,
write summaries, and review the output.

## Usage

```bash
python main.py --topic "artificial intelligence trends"
```

You can also pass a JSON config:

```json
{"topic": "machine learning", "depth": "detailed", "max_sources": 5}
```

### Example prompts

prompt = "Summarize recent advances in quantum computing"
query = "Compare Python and Rust for systems programming"
"""

SYNTHETIC_ENTRY_POINT = """\
import os
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find and analyze information on a given topic",
    tools=[search_tool, scrape_tool],
)

writer = Agent(
    role="Content Writer",
    goal="Write compelling summaries based on research",
)

def main(user_input: str):
    task = Task(description=user_input, agent=researcher)
    crew = Crew(agents=[researcher, writer], tasks=[task])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI trends"
    main(topic)
"""

# =========================================================================
# 2.  Mock the OpenAI client
# =========================================================================

# The canned response the mock vLLM will return
CANNED_INPUTS = [
    "Research the latest breakthroughs in renewable energy storage and summarize the top 3 technologies.",
    '{"topic": "autonomous vehicles", "depth": "comprehensive", "max_sources": 10}',
    "Compare and contrast large language models: GPT-4, Claude, and Gemini across performance benchmarks.",
    "Investigate the current state of CRISPR gene therapy clinical trials and write a patient-friendly summary.",
    "Analyze the economic impact of AI automation on the manufacturing sector in 2025.",
]


def _build_mock_openai_client() -> MagicMock:
    """Create a MagicMock that mimics OpenAI().chat.completions.create()."""
    mock_client = MagicMock()

    # Build the response object
    mock_message = MagicMock()
    mock_message.content = json.dumps(CANNED_INPUTS)

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# =========================================================================
# 3.  Test generate_inputs()
# =========================================================================

print("=" * 80)
print("TEST: generate_inputs() with mocked vLLM")
print("=" * 80)

mock_client = _build_mock_openai_client()

with patch("stratum_lab.harness.input_generator.OpenAI", return_value=mock_client):
    generated = generate_inputs(
        repo_url="https://github.com/example/research-crew",
        readme_content=SYNTHETIC_README,
        entry_point_code=SYNTHETIC_ENTRY_POINT,
        detected_input_type="user prompt string",
        vllm_url="http://localhost:8000/v1",
        framework="crewai",
        count=5,
    )

print(f"\n  Generated {len(generated)} inputs:\n")
for i, inp in enumerate(generated, 1):
    print(f"  [{i}] {inp[:120]}{'...' if len(inp) > 120 else ''}")

# ---- Diversity check ----
unique_inputs = set(generated)
all_different = len(unique_inputs) == len(generated)
print(f"\n  Diversity check: {'PASS' if all_different else 'FAIL'} "
      f"({len(unique_inputs)} unique out of {len(generated)})")

# ---- Parseability check ----
all_parseable = all(isinstance(inp, str) and len(inp) > 0 for inp in generated)
print(f"  Parseability check: {'PASS' if all_parseable else 'FAIL'}")

# ---- Verify the mock was called correctly ----
mock_client.chat.completions.create.assert_called_once()
call_kwargs = mock_client.chat.completions.create.call_args
print(f"  Mock called with model: {call_kwargs.kwargs.get('model', call_kwargs[1].get('model', '?'))}")
print(f"  Mock called with temperature: {call_kwargs.kwargs.get('temperature', call_kwargs[1].get('temperature', '?'))}")


# =========================================================================
# 4.  Test input_hash()
# =========================================================================

print("\n" + "=" * 80)
print("TEST: input_hash()")
print("=" * 80)

hashes = [input_hash(inp) for inp in generated]
print(f"\n  Hashes: {hashes}")
unique_hashes = set(hashes)
print(f"  All unique: {'PASS' if len(unique_hashes) == len(hashes) else 'FAIL'}")
print(f"  Length (12 hex chars): {'PASS' if all(len(h) == 12 for h in hashes) else 'FAIL'}")


# =========================================================================
# 5.  Test plan_runs()
# =========================================================================

print("\n" + "=" * 80)
print("TEST: plan_runs()")
print("=" * 80)

run_plan = plan_runs(generated, total_runs=5)
print(f"\n  Run plan ({len(run_plan)} runs):\n")
for inp, run_num in run_plan:
    print(f"    Run {run_num}: {inp[:80]}{'...' if len(inp) > 80 else ''}")

# Verify structure: 3 diverse + 2 repeats of first input
diverse_inputs = [inp for inp, rn in run_plan[:3]]
repeat_inputs = [inp for inp, rn in run_plan[3:]]
print(f"\n  First 3 (diverse): {len(set(diverse_inputs))} unique")
print(f"  Last 2 (repeats):  all equal to first? "
      f"{'PASS' if all(r == generated[0] for r in repeat_inputs) else 'FAIL'}")

# Verify run numbers are 1-indexed and sequential
run_numbers = [rn for _, rn in run_plan]
expected_numbers = list(range(1, 6))
print(f"  Run numbers sequential [1..5]: "
      f"{'PASS' if run_numbers == expected_numbers else 'FAIL'}")

# Test with fewer inputs than runs
small_plan = plan_runs(["single_input"], total_runs=5)
print(f"\n  plan_runs with 1 input, 5 runs: {len(small_plan)} entries "
      f"({'PASS' if len(small_plan) == 5 else 'FAIL'})")

# Test with empty input list (should use generic fallback)
empty_plan = plan_runs([], total_runs=5)
print(f"  plan_runs with 0 inputs (fallback): {len(empty_plan)} entries "
      f"({'PASS' if len(empty_plan) == 5 else 'FAIL'})")


# =========================================================================
# 6.  Test find_example_inputs()
# =========================================================================

print("\n" + "=" * 80)
print("TEST: find_example_inputs()")
print("=" * 80)

# Create a temporary directory mimicking a repo with README, examples, tests
tmp_repo = Path(tempfile.mkdtemp(prefix="stratum_eval_repo_"))

# Write a README with example prompts
readme_path = tmp_repo / "README.md"
readme_path.write_text(SYNTHETIC_README, encoding="utf-8")

# Write an examples directory with a Python file
examples_dir = tmp_repo / "examples"
examples_dir.mkdir()
(examples_dir / "demo.py").write_text(
    '''\
"""Example usage of the research agent."""

prompt = "Analyze the impact of social media on mental health in teenagers"

task = "Write a comprehensive report on the future of electric vehicles"

long_input = """
Investigate the following questions about climate change:
1. What are the latest IPCC projections?
2. How are major economies responding?
3. What technological solutions are most promising?
"""
''',
    encoding="utf-8",
)

# Write a test directory
tests_dir = tmp_repo / "tests"
tests_dir.mkdir()
(tests_dir / "test_agent.py").write_text(
    '''\
"""Test suite for research agent."""

query = "Summarize the history of the Internet from ARPANET to today"

input = "Compare renewable energy adoption rates across G7 nations"
''',
    encoding="utf-8",
)

# Write an example JSON input file
(tmp_repo / "input.example.json").write_text(
    json.dumps({"topic": "space exploration", "depth": "summary", "sources": 3}),
    encoding="utf-8",
)

found = find_example_inputs(tmp_repo)
print(f"\n  Found {len(found)} example inputs:\n")
for i, inp in enumerate(found, 1):
    display = inp.replace("\n", " ")[:100]
    print(f"  [{i}] {display}{'...' if len(inp) > 100 else ''}")

print(f"\n  At least some inputs found: {'PASS' if len(found) > 0 else 'FAIL'}")

# ---- Clean up ----
import shutil

try:
    shutil.rmtree(tmp_repo)
except Exception:
    pass


# =========================================================================
# 7.  Test generate_inputs() fallback on failure
# =========================================================================

print("\n" + "=" * 80)
print("TEST: generate_inputs() fallback on API error")
print("=" * 80)

mock_client_bad = MagicMock()
mock_client_bad.chat.completions.create.side_effect = ConnectionError("vLLM is down")

with patch("stratum_lab.harness.input_generator.OpenAI", return_value=mock_client_bad):
    fallback = generate_inputs(
        repo_url="https://github.com/example/broken",
        readme_content="",
        entry_point_code="",
        detected_input_type="user prompt",
        vllm_url="http://localhost:9999/v1",
        framework="unknown",
        count=5,
    )

print(f"\n  Fallback generated {len(fallback)} inputs (expected 5): "
      f"{'PASS' if len(fallback) == 5 else 'FAIL'}")
print(f"  All are non-empty strings: "
      f"{'PASS' if all(isinstance(s, str) and len(s) > 0 for s in fallback) else 'FAIL'}")
for i, inp in enumerate(fallback, 1):
    print(f"    [{i}] {inp[:100]}")


print("\nDone.")
