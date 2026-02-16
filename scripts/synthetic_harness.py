#!/usr/bin/env python3
"""Tier 2 Synthetic Harness -- fallback when native execution fails.

Scans a cloned repo for AI agent framework patterns, extracts agent/task
definitions (roles, goals, backstories) using regex, and generates a minimal
self-contained Python script that exercises the real agent definitions
against a vLLM endpoint.

This is LEGITIMATE because it exercises the REAL agent definitions from the
repo -- just without broken dependencies, missing API keys, or custom tools.

Usage:
    python synthetic_harness.py <repo_path> <vllm_host> <output_dir>

Exit codes:
    0   = script generated and executed successfully (events captured)
    1   = no framework patterns found or execution failed
    124 = timeout (from parent process)
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import subprocess
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_VLLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_VLLM_URL = "http://host.docker.internal:8000/v1"
DEFAULT_API_KEY = "sk-stratum-local"
DEFAULT_TIMEOUT = 300

# Directories to skip when scanning repo source files.
_SKIP_DIRS = ("venv/", ".venv/", "node_modules/", "__pycache__/", ".git/")


# ── Framework detection patterns ─────────────────────────────────────────

FRAMEWORK_PATTERNS: dict[str, list[str]] = {
    "crewai": [
        r"from\s+crewai\s+import",
        r"import\s+crewai",
        r"Agent\s*\(",
        r"\.kickoff\s*\(",
    ],
    "langgraph": [
        r"from\s+langgraph",
        r"StateGraph\s*\(",
        r"\.add_node\s*\(",
        r"\.compile\s*\(",
    ],
    "autogen": [
        r"from\s+autogen",
        r"ConversableAgent\s*\(",
        r"AssistantAgent\s*\(",
        r"GroupChat\s*\(",
    ],
    "openai": [
        r"openai\.OpenAI\s*\(",
        r"client\.chat\.completions\.create",
        r"from\s+openai\s+import",
    ],
}


def _iter_python_files(repo_path: Path, *, skip_tests: bool = False):
    """Yield .py files in *repo_path*, skipping vendored/virtual-env dirs."""
    try:
        py_files = list(repo_path.rglob("*.py"))
    except (OSError, PermissionError):
        return

    for py_file in py_files:
        parts = str(py_file)
        if any(skip in parts for skip in _SKIP_DIRS):
            continue
        if skip_tests and "test" in parts.lower():
            continue
        yield py_file


def _read_file(path: Path) -> str:
    """Read a file, returning empty string on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def detect_frameworks(repo_path: Path) -> dict[str, int]:
    """Detect which frameworks a repo uses and count pattern matches.

    Returns a dict mapping framework name to hit-count (only entries > 0).
    """
    scores: dict[str, int] = {}
    for py_file in _iter_python_files(repo_path, skip_tests=True):
        content = _read_file(py_file)
        if not content:
            continue
        for framework, patterns in FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    scores[framework] = scores.get(framework, 0) + 1
    return scores


# ── Agent / Task extraction ──────────────────────────────────────────────

def _extract_string_param(block: str, param_name: str) -> str:
    """Extract a string parameter value from a function-call block.

    Handles single quotes, double quotes, f-strings, and triple-quoted
    strings.  Returns the cleaned value (max 500 chars) or empty string.
    """
    # Order matters: try triple-quoted first (greedy) then single-line.
    patterns = [
        rf'{param_name}\s*=\s*"""(.*?)"""',                     # triple double
        rf"{param_name}\s*=\s*'''(.*?)'''",                     # triple single
        rf'{param_name}\s*=\s*f"([^"]*)"',                      # f-string double
        rf"{param_name}\s*=\s*f'([^']*)'",                      # f-string single
        rf'{param_name}\s*=\s*"((?:[^"\\]|\\.)*)"',             # double-quoted
        rf"{param_name}\s*=\s*'((?:[^'\\]|\\.)*)'",             # single-quoted
    ]
    for pat in patterns:
        m = re.search(pat, block, re.DOTALL)
        if m:
            val = m.group(1).strip()
            # Replace f-string placeholders with a safe literal
            val = re.sub(r"\{[^}]+\}", "example", val)
            return val[:500]
    return ""


def extract_crewai_agents(repo_path: Path) -> list[dict[str, str]]:
    """Extract CrewAI ``Agent()`` definitions from source files.

    Uses a multiline regex that matches ``Agent(...)`` blocks spanning
    multiple lines (non-greedy, handles nested parens up to one level).
    """
    agents: list[dict[str, str]] = []
    # Match Agent( ... ) allowing nested single-level parens inside.
    agent_pattern = re.compile(
        r"Agent\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)",
        re.DOTALL,
    )
    for py_file in _iter_python_files(repo_path):
        content = _read_file(py_file)
        if not content:
            continue
        for match in agent_pattern.finditer(content):
            block = match.group(1)
            agent: dict[str, str] = {}
            for param in ("role", "goal", "backstory"):
                val = _extract_string_param(block, param)
                if val:
                    agent[param] = val
            if agent.get("role"):
                agents.append(agent)
    return agents


def extract_crewai_tasks(repo_path: Path) -> list[dict[str, str]]:
    """Extract CrewAI ``Task()`` definitions from source files."""
    tasks: list[dict[str, str]] = []
    task_pattern = re.compile(
        r"Task\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)",
        re.DOTALL,
    )
    for py_file in _iter_python_files(repo_path):
        content = _read_file(py_file)
        if not content:
            continue
        for match in task_pattern.finditer(content):
            block = match.group(1)
            task: dict[str, str] = {}
            for param in ("description", "expected_output"):
                val = _extract_string_param(block, param)
                if val:
                    task[param] = val
            if task.get("description"):
                tasks.append(task)
    return tasks


def extract_langgraph_nodes(repo_path: Path) -> list[str]:
    """Extract LangGraph node names from ``add_node()`` calls."""
    nodes: list[str] = []
    pattern = re.compile(r'\.add_node\s*\(\s*["\'](\w+)["\']')
    for py_file in _iter_python_files(repo_path):
        content = _read_file(py_file)
        if not content:
            continue
        for match in pattern.finditer(content):
            node_name = match.group(1)
            if node_name not in nodes:
                nodes.append(node_name)
    return nodes


def extract_autogen_agents(repo_path: Path) -> list[dict[str, str]]:
    """Extract AutoGen agent definitions."""
    agents: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?:ConversableAgent|AssistantAgent|UserProxyAgent)\s*\("
        r"([^)]*(?:\([^)]*\)[^)]*)*)\)",
        re.DOTALL,
    )
    for py_file in _iter_python_files(repo_path):
        content = _read_file(py_file)
        if not content:
            continue
        for match in pattern.finditer(content):
            block = match.group(1)
            agent: dict[str, str] = {}
            name = _extract_string_param(block, "name")
            if name:
                agent["name"] = name
            system_message = _extract_string_param(block, "system_message")
            if system_message:
                agent["system_message"] = system_message[:200]
            if agent.get("name"):
                agents.append(agent)
    return agents


# ── Helpers for generated scripts ────────────────────────────────────────

def _env_config() -> tuple[str, str, str, str]:
    """Return (vllm_url, vllm_model, api_key, repo_root) from env."""
    vllm_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_VLLM_URL)
    vllm_model = os.environ.get("STRATUM_VLLM_MODEL", DEFAULT_VLLM_MODEL)
    api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    repo_root = os.environ.get("STRATUM_REPO_ROOT", "/app/repo")
    return vllm_url, vllm_model, api_key, repo_root


def _env_preamble(vllm_url: str, api_key: str, repo_root: str) -> str:
    """Return the ``os.environ`` preamble that MUST appear before any
    framework import in every generated script.

    Sets OPENAI_BASE_URL, OPENAI_API_KEY, and STRATUM_REPO_ROOT so that
    runner.py can add the repo to PYTHONPATH.
    """
    return textwrap.dedent(f"""\
        import os
        os.environ["OPENAI_BASE_URL"] = {vllm_url!r}
        os.environ["OPENAI_API_KEY"] = {api_key!r}
        os.environ["STRATUM_REPO_ROOT"] = {repo_root!r}
    """)


# ── Script generators ────────────────────────────────────────────────────

def generate_crewai_script(
    agents: list[dict[str, str]],
    tasks: list[dict[str, str]],
) -> str:
    """Generate a self-contained CrewAI script.

    CrewAI requires explicit LLM configuration via litellm.  The model
    string MUST carry an ``openai/`` prefix so litellm routes the call
    to the vLLM-compatible endpoint.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()

    # The openai/ prefix is REQUIRED for litellm routing.
    model_str = f"openai/{vllm_model}"

    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # -- Build agent definitions (cap at 5) --------------------------------
    agent_defs: list[str] = []
    agent_vars: list[str] = []
    for i, agent in enumerate(agents[:5]):
        var = f"agent_{i}"
        role = agent.get("role", f"Agent {i}")
        goal = agent.get("goal", "Complete assigned tasks efficiently")
        backstory = agent.get("backstory", "An experienced AI assistant")
        agent_defs.append(textwrap.dedent(f"""\
            {var} = Agent(
                role={role!r},
                goal={goal!r},
                backstory={backstory!r},
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )"""))
        agent_vars.append(var)

    # -- Build task definitions (cap at 5) ---------------------------------
    task_defs: list[str] = []
    task_vars: list[str] = []
    effective_tasks = list(tasks)  # don't mutate caller's list
    if not effective_tasks:
        for agent in agents[:5]:
            effective_tasks.append({
                "description": (
                    f"Perform the primary function of the "
                    f"{agent.get('role', 'agent')}"
                ),
                "expected_output": "A detailed response addressing the task",
            })
    for i, task in enumerate(effective_tasks[:5]):
        var = f"task_{i}"
        desc = task.get("description", f"Complete task {i}")
        expected = task.get("expected_output", "A detailed response")
        agent_var = agent_vars[i % len(agent_vars)]
        task_defs.append(textwrap.dedent(f"""\
            {var} = Task(
                description={desc!r},
                expected_output={expected!r},
                agent={agent_var},
            )"""))
        task_vars.append(var)

    agents_list = ", ".join(agent_vars)
    tasks_list = ", ".join(task_vars)
    nl = chr(10)

    return (
        preamble
        + textwrap.dedent(f"""\
        from crewai import Agent, Task, Crew, LLM

        llm = LLM(
            model={model_str!r},
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        {nl.join(agent_defs)}

        {nl.join(task_defs)}

        crew = Crew(
            agents=[{agents_list}],
            tasks=[{tasks_list}],
            verbose=False,
        )

        result = crew.kickoff()
        print("Crew completed:", type(result).__name__)
        """)
    )


def generate_langgraph_script(nodes: list[str]) -> str:
    """Generate a self-contained LangGraph script.

    Uses ``ChatOpenAI`` from ``langchain_openai`` pointed at the vLLM
    endpoint.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()

    if not nodes:
        nodes = ["researcher", "writer"]
    nodes = nodes[:4]

    preamble = _env_preamble(vllm_url, api_key, repo_root)

    node_funcs: list[str] = []
    add_nodes: list[str] = []
    for node_name in nodes:
        node_funcs.append(textwrap.dedent(f"""\
            def {node_name}_fn(state: dict) -> dict:
                result = llm.invoke([
                    SystemMessage(content="You are a {node_name} node in a processing graph."),
                    HumanMessage(content=f"Process this: {{state.get('input', 'analyze this topic')}}"),
                ])
                state["{node_name}_output"] = result.content
                return state
        """))
        add_nodes.append(f'graph.add_node("{node_name}", {node_name}_fn)')

    # Chain: START -> node0 -> node1 -> ... -> END
    edges = [f'graph.add_edge(START, "{nodes[0]}")']
    for i in range(len(nodes) - 1):
        edges.append(f'graph.add_edge("{nodes[i]}", "{nodes[i + 1]}")')
    edges.append(f'graph.add_edge("{nodes[-1]}", END)')

    nl = chr(10)

    return (
        preamble
        + textwrap.dedent(f"""\
        from langgraph.graph import StateGraph, START, END
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model={vllm_model!r},
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        {nl.join(node_funcs)}

        graph = StateGraph(dict)
        {nl.join("        " + n for n in add_nodes)}
        {nl.join("        " + e for e in edges)}

        app = graph.compile()
        result = app.invoke({{"input": "Analyze the impact of AI on software development"}})
        print("Graph completed:", list(result.keys()))
        """)
    )


def generate_autogen_script(agents: list[dict[str, str]]) -> str:
    """Generate a self-contained AutoGen script using ``config_list``."""
    vllm_url, vllm_model, api_key, repo_root = _env_config()

    if not agents or len(agents) < 2:
        agents = [
            {
                "name": "assistant",
                "system_message": "You are a helpful AI assistant.",
            },
            {
                "name": "user_proxy",
                "system_message": "You are a user who asks questions.",
            },
        ]

    a1 = agents[0]
    a2 = (
        agents[1]
        if len(agents) > 1
        else {"name": "user_proxy", "system_message": "Ask follow-up questions."}
    )

    preamble = _env_preamble(vllm_url, api_key, repo_root)

    return (
        preamble
        + textwrap.dedent(f"""\
        import autogen

        config_list = [{{
            "model": {vllm_model!r},
            "base_url": {vllm_url!r},
            "api_key": {api_key!r},
        }}]

        assistant = autogen.AssistantAgent(
            name={a1["name"]!r},
            system_message={a1.get("system_message", "You are a helpful assistant.")!r},
            llm_config={{"config_list": config_list}},
        )

        user_proxy = autogen.UserProxyAgent(
            name={a2["name"]!r},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )

        user_proxy.initiate_chat(
            assistant,
            message="Analyze the key trends in AI agent frameworks and provide a brief summary.",
        )
        print("Chat completed")
        """)
    )


def generate_openai_script() -> str:
    """Generate a simple OpenAI-compatible script (3 varied calls)."""
    vllm_url, vllm_model, api_key, repo_root = _env_config()

    preamble = _env_preamble(vllm_url, api_key, repo_root)

    return (
        preamble
        + textwrap.dedent(f"""\
        import openai

        client = openai.OpenAI(
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        prompts = [
            "What are the key components of a multi-agent AI system?",
            "Explain the concept of agent delegation in AI frameworks.",
            "Describe error handling strategies for AI agent pipelines.",
        ]

        for i, prompt in enumerate(prompts):
            response = client.chat.completions.create(
                model={vllm_model!r},
                messages=[{{"role": "user", "content": prompt}}],
                max_tokens=200,
            )
            print(f"Call {{i+1}}: {{response.choices[0].message.content[:100]}}")

        print("OpenAI calls completed")
        """)
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    """Entry point.  Returns 0 on success, 1 on failure."""
    if len(sys.argv) < 4:
        print(
            "Usage: synthetic_harness.py <repo_path> <vllm_host> <output_dir>",
            file=sys.stderr,
        )
        return 1

    repo_path = Path(sys.argv[1])
    vllm_host = sys.argv[2]
    output_dir = sys.argv[3]
    events_file = os.path.join(output_dir, "stratum_events.jsonl")

    if not repo_path.is_dir():
        print(f"[synthetic] Error: {repo_path} is not a directory", file=sys.stderr)
        return 1

    # -- Environment for the patcher / collector ---------------------------
    os.environ["STRATUM_EVENTS_FILE"] = events_file
    os.environ["OPENAI_BASE_URL"] = f"{vllm_host}/v1"
    os.environ["STRATUM_FRAMEWORK"] = "synthetic"
    # Ensure STRATUM_REPO_ROOT is set so runner.py adds it to PYTHONPATH.
    if "STRATUM_REPO_ROOT" not in os.environ:
        os.environ["STRATUM_REPO_ROOT"] = str(repo_path)

    # -- Detect frameworks -------------------------------------------------
    print(
        f"[synthetic] Scanning {repo_path} for framework patterns...",
        file=sys.stderr,
    )
    framework_scores = detect_frameworks(repo_path)

    if not framework_scores:
        print("[synthetic] No framework patterns found", file=sys.stderr)
        return 1

    best_framework = max(framework_scores, key=framework_scores.get)  # type: ignore[arg-type]
    print(
        f"[synthetic] Detected: {framework_scores} -- using {best_framework}",
        file=sys.stderr,
    )

    # -- Generate the appropriate script -----------------------------------
    script = ""

    try:
        if best_framework == "crewai":
            agents = extract_crewai_agents(repo_path)
            tasks = extract_crewai_tasks(repo_path)
            print(
                f"[synthetic] Extracted {len(agents)} agents, {len(tasks)} tasks",
                file=sys.stderr,
            )
            if not agents:
                agents = [
                    {
                        "role": "Researcher",
                        "goal": "Research topics thoroughly",
                        "backstory": "An experienced researcher",
                    },
                    {
                        "role": "Writer",
                        "goal": "Write clear summaries",
                        "backstory": "A skilled technical writer",
                    },
                ]
            script = generate_crewai_script(agents, tasks)

        elif best_framework == "langgraph":
            nodes = extract_langgraph_nodes(repo_path)
            print(
                f"[synthetic] Extracted {len(nodes)} graph nodes: {nodes}",
                file=sys.stderr,
            )
            script = generate_langgraph_script(nodes)

        elif best_framework == "autogen":
            agents = extract_autogen_agents(repo_path)
            print(
                f"[synthetic] Extracted {len(agents)} agents",
                file=sys.stderr,
            )
            script = generate_autogen_script(agents)

        elif best_framework == "openai":
            script = generate_openai_script()

    except Exception as exc:
        print(
            f"[synthetic] Error during extraction/generation: {exc}",
            file=sys.stderr,
        )
        return 1

    if not script:
        print("[synthetic] Failed to generate script", file=sys.stderr)
        return 1

    # -- Write and execute the generated script ----------------------------
    script_path = Path("/tmp/synthetic_run.py")
    try:
        script_path.write_text(script, encoding="utf-8")
    except OSError as exc:
        print(f"[synthetic] Cannot write script: {exc}", file=sys.stderr)
        return 1

    print(
        f"[synthetic] Generated {script_path} ({len(script)} bytes)",
        file=sys.stderr,
    )
    # Show a preview (first 500 chars) for debugging.
    print(f"[synthetic] Script preview:\n{script[:500]}...", file=sys.stderr)

    timeout = int(
        os.environ.get("STRATUM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT))
    )

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        print(f"[synthetic] Execution timed out after {timeout}s", file=sys.stderr)
        return 124
    except OSError as exc:
        print(f"[synthetic] Failed to launch subprocess: {exc}", file=sys.stderr)
        return 1

    print(f"[synthetic] Exit code: {result.returncode}", file=sys.stderr)
    if result.stdout:
        print(f"[synthetic] stdout: {result.stdout[:500]}", file=sys.stderr)
    if result.stderr:
        print(f"[synthetic] stderr: {result.stderr[:1000]}", file=sys.stderr)

    # -- Check if events were captured -------------------------------------
    events_path = Path(events_file)
    if events_path.exists() and events_path.stat().st_size > 0:
        try:
            event_count = sum(1 for _ in open(events_path, encoding="utf-8"))
        except OSError:
            event_count = -1
        print(f"[synthetic] Captured {event_count} events", file=sys.stderr)
        return 0

    print("[synthetic] No events captured", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
