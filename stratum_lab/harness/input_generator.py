"""Synthetic input generation for repo execution.

Uses vLLM (via the OpenAI-compatible API) to generate diverse test inputs
tailored to each repo's entry point, README, and detected input type.
Falls back to generic inputs when generation fails.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console

from stratum_lab.config import VLLM_API_KEY, VLLM_MODEL

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERIC_INPUTS: list[str] = [
    '{"task": "Summarize the latest news about artificial intelligence."}',
    '{"task": "Write a short report on climate change impacts in 2024."}',
    '{"task": "Research the top 3 programming languages and compare them."}',
    '{"task": "Plan a weekend trip to San Francisco and list activities."}',
    '{"task": "Analyze the pros and cons of remote work for software teams."}',
]

_INPUT_GENERATION_PROMPT = """\
You are generating synthetic test inputs for an AI agent application.

The repo uses the **{framework}** framework.

## README excerpt
{readme_excerpt}

## Entry point code excerpt
{entry_point_excerpt}

## Detected input type
{detected_input_type}

---

Generate exactly {count} diverse, realistic test inputs that this application \
would accept.  Each input should exercise a different code path or capability \
of the agent system.  Vary the topic, complexity, and structure of each input.

Return a JSON array of strings, where each string is the input the program \
would receive (e.g. a user prompt, a JSON payload, a CLI argument string).

Important:
- Each input should be a single string.
- Inputs must be realistic and non-trivial.
- Do NOT wrap the array in markdown code fences.
- Return ONLY the JSON array, nothing else.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_inputs(
    repo_url: str,
    readme_content: str,
    entry_point_code: str,
    detected_input_type: str,
    vllm_url: str,
    framework: str = "unknown",
    count: int = 5,
) -> list[str]:
    """Use vLLM to generate *count* diverse synthetic test inputs.

    Parameters
    ----------
    repo_url:
        Git URL (used only for logging/context).
    readme_content:
        Raw text of the repo's README (or excerpt).
    entry_point_code:
        Source code of the detected entry point file (or excerpt).
    detected_input_type:
        Short description of expected input (e.g. ``"user prompt"``,
        ``"JSON config"``).
    vllm_url:
        OpenAI-compatible vLLM base URL.
    framework:
        Framework name for prompt context.
    count:
        Number of inputs to generate.

    Returns
    -------
    list[str]
        Generated inputs.  Falls back to generic inputs on failure.
    """
    # Truncate long content so we don't blow up the context window
    readme_excerpt = _truncate(readme_content, max_chars=3000)
    entry_point_excerpt = _truncate(entry_point_code, max_chars=4000)

    prompt = _INPUT_GENERATION_PROMPT.format(
        framework=framework,
        readme_excerpt=readme_excerpt or "(not available)",
        entry_point_excerpt=entry_point_excerpt or "(not available)",
        detected_input_type=detected_input_type or "user prompt string",
        count=count,
    )

    try:
        client = OpenAI(base_url=vllm_url, api_key=VLLM_API_KEY)
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": "You generate test inputs for AI agent applications. Respond only with a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or ""
        inputs = _parse_json_array(raw)

        if inputs and len(inputs) >= count:
            console.print(
                f"  [green]Generated {len(inputs)} synthetic inputs[/green] for {repo_url}"
            )
            return inputs[:count]

        console.print(
            f"  [yellow]vLLM returned {len(inputs)} inputs (expected {count}), "
            f"padding with generic inputs[/yellow]"
        )
        return _pad_inputs(inputs, count)

    except Exception as exc:
        console.print(
            f"  [yellow]Input generation failed for {repo_url}: {exc}. "
            f"Using generic inputs.[/yellow]"
        )
        return _generic_fallback(count)


def find_example_inputs(repo_path: str | Path) -> list[str]:
    """Search a cloned repo for existing example inputs.

    Looks in README files, test directories, and example directories for
    strings that look like inputs (prompts, JSON payloads, etc.).

    Parameters
    ----------
    repo_path:
        Local path to the cloned repo.

    Returns
    -------
    list[str]
        Discovered example inputs (may be empty).
    """
    repo_path = Path(repo_path)
    found: list[str] = []

    # 1. Search README for code blocks that look like inputs
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme_file = repo_path / readme_name
        if readme_file.is_file():
            try:
                text = readme_file.read_text(encoding="utf-8", errors="replace")
                found.extend(_extract_inputs_from_readme(text))
            except Exception:
                pass

    # 2. Search example directories
    for example_dir_name in ("examples", "example", "demo", "demos", "samples"):
        example_dir = repo_path / example_dir_name
        if example_dir.is_dir():
            for py_file in example_dir.rglob("*.py"):
                try:
                    code = py_file.read_text(encoding="utf-8", errors="replace")
                    found.extend(_extract_string_literals(code))
                except Exception:
                    pass

    # 3. Search test directories
    for test_dir_name in ("tests", "test"):
        test_dir = repo_path / test_dir_name
        if test_dir.is_dir():
            for py_file in test_dir.rglob("*.py"):
                try:
                    code = py_file.read_text(encoding="utf-8", errors="replace")
                    found.extend(_extract_string_literals(code))
                except Exception:
                    pass

    # 4. Look for JSON/YAML input files
    for pattern in ("*.input.json", "input*.json", "*.example.json"):
        for json_file in repo_path.rglob(pattern):
            try:
                text = json_file.read_text(encoding="utf-8", errors="replace")
                found.append(text.strip())
            except Exception:
                pass

    # Deduplicate and filter
    seen: set[str] = set()
    unique: list[str] = []
    for inp in found:
        inp = inp.strip()
        if inp and len(inp) > 10 and inp not in seen:
            seen.add(inp)
            unique.append(inp)

    return unique[:20]  # Cap at 20


def plan_runs(
    inputs: list[str],
    total_runs: int = 5,
) -> list[tuple[str, int]]:
    """Plan the run schedule: 3 diverse inputs + 2 repeats of the first.

    Returns a list of ``(input_data, run_number)`` pairs.  If fewer than 3
    diverse inputs are available, the remaining diverse slots are filled
    with available inputs in round-robin.

    Parameters
    ----------
    inputs:
        Available test inputs.
    total_runs:
        Total number of runs to schedule.

    Returns
    -------
    list[tuple[str, int]]
        ``(input_data, run_number)`` pairs, 1-indexed.
    """
    if not inputs:
        inputs = _generic_fallback(3)

    diverse_count = min(3, total_runs)
    repeat_count = total_runs - diverse_count

    runs: list[tuple[str, int]] = []
    run_number = 1

    # Diverse runs — use first N unique inputs
    for i in range(diverse_count):
        inp = inputs[i % len(inputs)]
        runs.append((inp, run_number))
        run_number += 1

    # Repeat runs — always repeat the first input for reproducibility checks
    first_input = inputs[0]
    for _ in range(repeat_count):
        runs.append((first_input, run_number))
        run_number += 1

    return runs


def input_hash(input_data: str) -> str:
    """Return a short hex hash of an input string for cache keying."""
    return hashlib.sha256(input_data.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate text to *max_chars*, adding an ellipsis if truncated."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _parse_json_array(raw: str) -> list[str]:
    """Try to parse a JSON array of strings from possibly noisy LLM output."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array substring
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass

    return []


def _generic_fallback(count: int) -> list[str]:
    """Return up to *count* generic inputs."""
    result: list[str] = []
    for i in range(count):
        result.append(_GENERIC_INPUTS[i % len(_GENERIC_INPUTS)])
    return result


def _pad_inputs(partial: list[str], target: int) -> list[str]:
    """Pad a partial input list to *target* length using generic fallbacks."""
    result = list(partial)
    generic = _generic_fallback(target)
    while len(result) < target:
        result.append(generic[len(result) % len(generic)])
    return result[:target]


def _extract_inputs_from_readme(text: str) -> list[str]:
    """Extract plausible inputs from README code blocks and usage examples."""
    inputs: list[str] = []

    # Fenced code blocks
    for match in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        block = match.group(1).strip()
        # Heuristic: blocks that look like prompts (short, no import statements)
        if (
            10 < len(block) < 500
            and "import " not in block
            and "def " not in block
            and "class " not in block
        ):
            inputs.append(block)

    # Quoted strings after "input", "prompt", "query", "task" keywords
    for match in re.finditer(
        r'(?:input|prompt|query|task|question)\s*[:=]\s*["\'](.+?)["\']',
        text,
        re.IGNORECASE,
    ):
        candidate = match.group(1).strip()
        if len(candidate) > 10:
            inputs.append(candidate)

    return inputs


def _extract_string_literals(code: str) -> list[str]:
    """Extract long string literals from Python source that might be inputs."""
    inputs: list[str] = []

    # Triple-quoted strings
    for match in re.finditer(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', code, re.DOTALL):
        candidate = (match.group(1) or match.group(2) or "").strip()
        if 20 < len(candidate) < 1000:
            inputs.append(candidate)

    # Single/double-quoted strings near "input"/"prompt"/"query" variables
    for match in re.finditer(
        r'(?:input|prompt|query|task|question)\s*=\s*["\'](.+?)["\']',
        code,
        re.IGNORECASE,
    ):
        candidate = match.group(1).strip()
        if len(candidate) > 10:
            inputs.append(candidate)

    return inputs
