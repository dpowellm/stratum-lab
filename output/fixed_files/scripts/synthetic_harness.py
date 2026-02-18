#!/usr/bin/env python3
"""Tier 1.5 + Tier 2 Synthetic Harness.

Tier 1.5 (import-and-call): Uses AST to find orchestration objects in the
repo's actual code, then generates a minimal trigger script that imports
and calls the real orchestration.  This preserves the repo's real topology,
routing logic, and delegation chains.

Tier 2 (full synthetic): Extracts agent/task definitions via regex and
generates a self-contained script exercising them against vLLM.  This loses
real topology but still captures agent definitions and LLM call patterns.

Usage:
    python synthetic_harness.py <repo_path> <vllm_host> <output_dir>
    python synthetic_harness.py --tier1.5 <repo_path> <vllm_host> <output_dir>

Exit codes:
    0   = script generated and executed successfully (events captured)
    1   = no framework patterns found or execution failed
    124 = timeout (from parent process)
"""

from __future__ import annotations

import ast
import os
import re
import sys
import textwrap
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_VLLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_VLLM_URL = "http://host.docker.internal:8000/v1"
DEFAULT_API_KEY = "sk-stratum-local"
DEFAULT_TIMEOUT = 300

# Directories to skip when scanning repo source files.
_SKIP_DIRS = ("venv/", ".venv/", "node_modules/", "__pycache__/", ".git/")


# ── Tier 1.5: AST-based import-and-call ──────────────────────────────────

class OrchTarget(NamedTuple):
    """An orchestration object found via AST analysis."""
    framework: str
    variable: str       # variable name or "func_name()" for function calls
    call_method: str    # kickoff, invoke, initiate_chat
    file_path: str      # absolute path to the source file
    module_path: str    # importable dotted module path


def _ast_has_main_guard(tree: ast.AST) -> bool:
    """Check if AST has ``if __name__ == "__main__"``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                for comp in node.test.comparators:
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        return True
    return False


def _compute_module_path(file_path: str, repo_root: str) -> str:
    """Convert a file path to a dotted importable module path."""
    rel = os.path.relpath(file_path, repo_root)
    mod = rel.replace(os.sep, ".").replace("/", ".")
    if mod.endswith(".py"):
        mod = mod[:-3]
    if mod.endswith(".__init__"):
        mod = mod[:-9]
    return mod


def _find_orch_targets_in_file(
    file_path: str, repo_root: str
) -> tuple[bool, list[OrchTarget]]:
    """Parse one file and return (has_main, targets)."""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError, OSError):
        return False, []

    has_main = _ast_has_main_guard(tree)
    module_path = _compute_module_path(file_path, repo_root)
    targets: list[OrchTarget] = []

    # Track imports
    crewai_names: set[str] = set()
    langgraph_names: set[str] = set()
    autogen_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in (node.names or []):
                name = alias.asname or alias.name
                if "crewai" in node.module:
                    crewai_names.add(name)
                elif "langgraph" in node.module:
                    langgraph_names.add(name)
                elif "autogen" in node.module:
                    autogen_names.add(name)

    # Track variable assignments: var -> class_name
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        assignments[tgt.id] = func.id
                    elif isinstance(func, ast.Attribute):
                        assignments[tgt.id] = func.attr

    # CrewAI: Crew() variable or function returning Crew()
    for var, cls in assignments.items():
        if cls == "Crew":
            targets.append(OrchTarget("crewai", var, "kickoff", file_path, module_path))

    # CrewAI: functions that return Crew()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                    func = child.value.func
                    fname = func.id if isinstance(func, ast.Name) else (
                        func.attr if isinstance(func, ast.Attribute) else None)
                    if fname == "Crew":
                        targets.append(OrchTarget(
                            "crewai", f"{node.name}()", "kickoff",
                            file_path, module_path))
                        break

    # LangGraph: compiled graphs
    for var, cls in assignments.items():
        if cls == "compile":
            targets.append(OrchTarget("langgraph", var, "invoke", file_path, module_path))

    # LangGraph: functions that build+compile a StateGraph
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_sg = has_comp = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    f = child.func
                    if isinstance(f, ast.Name) and f.id == "StateGraph":
                        has_sg = True
                    if isinstance(f, ast.Attribute) and f.attr == "compile":
                        has_comp = True
            if has_sg and has_comp:
                targets.append(OrchTarget(
                    "langgraph", f"{node.name}()", "invoke",
                    file_path, module_path))

    # AutoGen: GroupChat or initiate_chat patterns
    for var, cls in assignments.items():
        if cls in ("GroupChat", "GroupChatManager"):
            targets.append(OrchTarget("autogen", var, "initiate_chat", file_path, module_path))
        if cls in ("UserProxyAgent", "ConversableAgent"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (node.func.attr == "initiate_chat"
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == var):
                        targets.append(OrchTarget(
                            "autogen", var, "initiate_chat", file_path, module_path))
                        break

    return has_main, targets


def scan_repo_for_orch_targets(repo_path: Path) -> list[OrchTarget]:
    """Scan all .py files for orchestration targets that LACK a __main__ block.

    These are candidates for Tier 1.5 import-and-call.
    """
    skip = {".git", "venv", ".venv", "node_modules", "__pycache__",
            "test", "tests", ".tox", ".mypy_cache"}
    results: list[OrchTarget] = []
    repo_str = str(repo_path)

    for root, dirs, files in os.walk(repo_str):
        dirs[:] = [d for d in dirs if d not in skip]
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            has_main, targets = _find_orch_targets_in_file(fpath, repo_str)
            if targets and not has_main:
                results.extend(targets)

    return results


def generate_import_and_call_script(
    target: OrchTarget,
    repo_root: str,
    vllm_url: str,
    vllm_model: str,
    api_key: str,
) -> str:
    """Generate a minimal trigger script for Tier 1.5.

    Uses line-by-line construction to avoid multi-line f-string indentation
    bugs when call_block spans multiple lines.

    The script:
      1. Sets env vars (OPENAI_BASE_URL, API keys, events file)
      2. Mocks unavailable service dependencies (Pinecone, Tavily, etc.)
      3. Adds repo to PYTHONPATH
      4. Imports the real module
      5. Calls the execution method with minimal inputs
    """
    var = target.variable
    is_func = var.endswith("()")
    func_name = var[:-2] if is_func else None

    # Build import + call lines (each element is one line of code)
    import_lines: list[str] = []
    call_lines: list[str] = []

    if is_func and func_name:
        import_lines.append(f"from {target.module_path} import {func_name}")
        if target.framework == "crewai":
            call_lines.append(f"result = {func_name}().kickoff()")
        elif target.framework == "langgraph":
            call_lines.append(f"graph = {func_name}()")
            call_lines.append(f'result = graph.invoke({{"input": "Analyze AI agent frameworks and their capabilities"}})')
        else:
            call_lines.append(f"result = {func_name}()")
    else:
        import_lines.append("import importlib")
        import_lines.append(f"_mod = importlib.import_module({target.module_path!r})")
        if target.framework == "crewai":
            call_lines.append(f"result = _mod.{var}.kickoff()")
        elif target.framework == "langgraph":
            call_lines.append(
                f'result = _mod.{var}.invoke('
                f'{{"input": "Analyze AI agent frameworks and their capabilities"}})')
        elif target.framework == "autogen":
            call_lines.append("# Find a target agent for initiate_chat")
            call_lines.append("_agents = [")
            call_lines.append("    v for k, v in vars(_mod).items()")
            call_lines.append(f"    if hasattr(v, 'generate_reply') and k != {var!r}")
            call_lines.append("]")
            call_lines.append("if _agents:")
            call_lines.append(f"    _mod.{var}.initiate_chat(")
            call_lines.append(f'        _agents[0], message="Analyze AI agent frameworks"')
            call_lines.append("    )")
        else:
            call_lines.append(f"result = _mod.{var}()")

    # Build the script line by line
    lines: list[str] = []
    lines.append("#!/usr/bin/env python3")
    lines.append(f'"""Tier 1.5: Import-and-call wrapper for {target.framework}.')
    lines.append(f"Source: {target.file_path}")
    lines.append(f"Module: {target.module_path}")
    lines.append(f"Target: {target.variable}.{target.call_method}()")
    lines.append('"""')
    lines.append("import os")
    lines.append("import sys")
    lines.append("import signal")
    lines.append("import importlib")
    lines.append("")
    lines.append("# Timeout protection")
    lines.append("def _timeout_handler(signum, frame):")
    lines.append('    raise SystemExit("STRATUM TIMEOUT in Tier 1.5 wrapper")')
    lines.append("")
    lines.append('if hasattr(signal, "SIGALRM"):')
    lines.append("    signal.signal(signal.SIGALRM, _timeout_handler)")
    lines.append('    signal.alarm(int(os.environ.get("STRATUM_TIMEOUT_SECONDS", "180")))')
    lines.append("")
    lines.append("# Environment setup")
    lines.append(f'os.environ.setdefault("OPENAI_BASE_URL", {vllm_url!r})')
    lines.append(f'os.environ.setdefault("OPENAI_API_KEY", {api_key!r})')
    lines.append(f'os.environ.setdefault("STRATUM_VLLM_MODEL", {vllm_model!r})')
    lines.append('os.environ.setdefault("STRATUM_EVENTS_FILE", "/app/stratum_events.jsonl")')
    lines.append('os.environ.setdefault("STRATUM_CAPTURE_PROMPTS", "1")')
    lines.append('os.environ.setdefault("STRATUM_TIER", "1.5")')
    lines.append("")
    lines.append("# Mock unavailable service dependencies so imports don't crash")
    lines.append("from unittest.mock import MagicMock")
    lines.append("_MOCK_MODULES = [")
    lines.append("    'pinecone', 'tavily', 'chromadb', 'weaviate', 'qdrant_client',")
    lines.append("    'firebase_admin', 'supabase', 'notion_client',")
    lines.append("    'google', 'google.cloud', 'google.auth', 'google.oauth2',")
    lines.append("    'firecrawl', 'browserbase', 'e2b', 'composio',")
    lines.append("]")
    lines.append("for _mod_name in _MOCK_MODULES:")
    lines.append("    if _mod_name not in sys.modules:")
    lines.append("        try:")
    lines.append("            importlib.import_module(_mod_name)")
    lines.append("        except ImportError:")
    lines.append("            _m = MagicMock()")
    lines.append("            _m.__path__ = []")
    lines.append("            _m.__file__ = f'mock_{_mod_name}'")
    lines.append("            _m.__spec__ = None")
    lines.append("            sys.modules[_mod_name] = _m")
    lines.append("")
    lines.append("# Add repo to PYTHONPATH")
    lines.append(f"repo_root = {repo_root!r}")
    lines.append("sys.path.insert(0, repo_root)")
    lines.append("src_dir = os.path.join(repo_root, 'src')")
    lines.append("if os.path.isdir(src_dir):")
    lines.append("    sys.path.insert(0, src_dir)")
    lines.append("")
    lines.append("try:")
    for il in import_lines:
        lines.append(f"    {il}")
    for cl in call_lines:
        lines.append(f"    {cl}")
    lines.append('    print("Tier 1.5 execution completed successfully")')
    lines.append("except ImportError as e:")
    lines.append('    print(f"Tier 1.5 import failed: {e}", file=sys.stderr)')
    lines.append("    sys.exit(1)")
    lines.append("except Exception as e:")
    lines.append('    print(f"Tier 1.5 execution failed: {e}", file=sys.stderr)')
    lines.append("    sys.exit(1)")
    lines.append("")

    script = "\n".join(lines)

    # Compile check — catch indentation bugs before writing
    try:
        compile(script, "<tier1_5_wrapper>", "exec")
    except SyntaxError as exc:
        _fallback = (
            "import sys\n"
            f'print("SyntaxError in Tier 1.5 wrapper: {exc}", file=sys.stderr)\n'
            "sys.exit(1)\n"
        )
        return _fallback

    return script


def try_tier_1_5(
    repo_path: Path,
    vllm_host: str,
    output_dir: str,
    timeout: int = 180,
) -> int:
    """Attempt Tier 1.5 import-and-call execution.

    Returns 0 if events were captured, 1 otherwise.
    """
    vllm_url = f"{vllm_host}/v1"
    vllm_model = os.environ.get("STRATUM_VLLM_MODEL", DEFAULT_VLLM_MODEL)
    api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    events_file = os.environ.get("STRATUM_EVENTS_FILE",
                                  os.path.join(output_dir, "stratum_events.jsonl"))

    print(f"[tier1.5] Scanning {repo_path} for orchestration targets...", file=sys.stderr)
    targets = scan_repo_for_orch_targets(repo_path)

    if not targets:
        print("[tier1.5] No import-and-call targets found", file=sys.stderr)
        return 1

    print(f"[tier1.5] Found {len(targets)} targets:", file=sys.stderr)
    for t in targets[:5]:
        print(f"  {t.framework}: {t.module_path}.{t.variable}.{t.call_method}()", file=sys.stderr)

    # Try each target until one produces events (max 3 attempts)
    for i, target in enumerate(targets[:3]):
        print(f"[tier1.5] Attempt {i+1}: {target.module_path}.{target.variable}", file=sys.stderr)

        script = generate_import_and_call_script(
            target, str(repo_path), vllm_url, vllm_model, api_key)

        script_path = Path(f"/tmp/tier1_5_run_{i}.py")
        try:
            script_path.write_text(script, encoding="utf-8")
        except OSError as e:
            print(f"[tier1.5] Cannot write script: {e}", file=sys.stderr)
            continue

        print(f"[tier1.5] Generated {script_path} ({len(script)} bytes)", file=sys.stderr)

        # Clear events from previous attempts
        try:
            Path(events_file).unlink(missing_ok=True)
        except OSError:
            pass

        # Execute
        env = os.environ.copy()
        env["STRATUM_EVENTS_FILE"] = events_file
        env["STRATUM_TIER"] = "1.5"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[tier1.5] Timed out after {timeout}s", file=sys.stderr)
            continue
        except OSError as e:
            print(f"[tier1.5] Launch failed: {e}", file=sys.stderr)
            continue

        print(f"[tier1.5] Exit code: {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"[tier1.5] stderr: {result.stderr[:500]}", file=sys.stderr)

        # Check for events
        events_path = Path(events_file)
        if events_path.exists() and events_path.stat().st_size > 0:
            try:
                event_count = sum(1 for _ in open(events_path, encoding="utf-8"))
            except OSError:
                event_count = 0
            if event_count > 0:
                print(f"[tier1.5] SUCCESS: {event_count} events captured", file=sys.stderr)
                return 0

        print(f"[tier1.5] No events from attempt {i+1}", file=sys.stderr)

    print("[tier1.5] All attempts failed", file=sys.stderr)
    return 1


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


# ── Variant selection ────────────────────────────────────────────────────

def read_all_py_files(repo_path: Path) -> str:
    """Concatenate all Python source files (excluding vendored dirs) into
    a single string for pattern matching.  Result is capped at 2 MB to
    avoid excessive memory use."""
    parts: list[str] = []
    total = 0
    cap = 2 * 1024 * 1024  # 2 MB
    for py_file in _iter_python_files(repo_path, skip_tests=True):
        content = _read_file(py_file)
        if content:
            parts.append(content)
            total += len(content)
            if total >= cap:
                break
    return "\n".join(parts)


def find_agents(source_code: str) -> list[str]:
    """Return a list of agent role names found in *source_code* via
    ``Agent(role=...)`` patterns.  Used by variant selection to gauge
    the number of distinct agents."""
    roles: list[str] = []
    for m in re.finditer(
        r"Agent\s*\([^)]*role\s*=\s*[\"']([^\"']+)[\"']", source_code, re.DOTALL
    ):
        role = m.group(1).strip()
        if role and role not in roles:
            roles.append(role)
    return roles


def select_variant(repo_path: Path, framework: str) -> str:
    """Choose a topology variant based on what was found in repo source.

    Returns a variant name string used to dispatch to the appropriate
    script-generation template.
    """
    source_code = read_all_py_files(repo_path)

    if framework == "crewai":
        if "allow_delegation" in source_code or "delegation" in source_code.lower():
            return "delegation"
        if "tool" in source_code.lower() and (
            "@tool" in source_code or "BaseTool" in source_code
        ):
            return "tooled"
        if len(find_agents(source_code)) >= 3:
            return "reviewed"
        return "delegation"  # default: at least exercise delegation path

    if framework == "langgraph":
        if (
            "conditional" in source_code.lower()
            or "add_conditional_edges" in source_code
        ):
            return "conditional"
        return "branching"

    if framework == "autogen":
        if "GroupChat" in source_code:
            return "groupchat"
        return "twoway"

    return "delegation"  # fallback


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
    variant: str = "sequential",
) -> str:
    """Generate a self-contained CrewAI script with topology variety.

    Variants:
      - ``"sequential"``  : basic sequential crew (original behaviour)
      - ``"delegation"``  : sequential with ``allow_delegation=True``
      - ``"reviewed"``    : 3 agents (researcher, writer, reviewer)
      - ``"tooled"``      : adds a simple tool to one agent

    CrewAI requires explicit LLM configuration via litellm.  The model
    string MUST carry an ``openai/`` prefix so litellm routes the call
    to the vLLM-compatible endpoint.
    """
    if variant == "delegation":
        return _generate_crewai_delegation(agents, tasks)
    if variant == "reviewed":
        return _generate_crewai_reviewed(agents)
    if variant == "tooled":
        return _generate_crewai_tooled(agents, tasks)
    # Default / "sequential" — original behaviour
    return _generate_crewai_sequential(agents, tasks)


# -- CrewAI variant: sequential (original) ---------------------------------

def _generate_crewai_sequential(
    agents: list[dict[str, str]],
    tasks: list[dict[str, str]],
) -> str:
    """Original sequential CrewAI script generator (unchanged)."""
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

    # Build body without textwrap.dedent to avoid indent mismatch with
    # embedded nl.join() content (which is already at 0-indent).
    body = (
        f"from crewai import Agent, Task, Crew, LLM\n"
        f"\n"
        f"llm = LLM(\n"
        f"    model={model_str!r},\n"
        f"    base_url={vllm_url!r},\n"
        f"    api_key={api_key!r},\n"
        f")\n"
        f"\n"
        + nl.join(agent_defs) + "\n"
        f"\n"
        + nl.join(task_defs) + "\n"
        f"\n"
        f"crew = Crew(\n"
        f"    agents=[{agents_list}],\n"
        f"    tasks=[{tasks_list}],\n"
        f"    verbose=False,\n"
        f")\n"
        f"\n"
        f"result = crew.kickoff()\n"
        f'print("Crew completed:", type(result).__name__)\n'
    )

    return preamble + body


# -- CrewAI variant: delegation (exercises delegation.initiated events) ----

def _generate_crewai_delegation(
    agents: list[dict[str, str]],
    tasks: list[dict[str, str]],
) -> str:
    """Sequential crew with ``allow_delegation=True`` on the manager agent.

    Task description includes "Delegate sub-tasks as needed" to trigger
    delegation events.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    model_str = f"openai/{vllm_model}"
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # Use the first extracted agent as manager, second as worker (fallback
    # to sensible defaults).
    mgr = agents[0] if agents else {}
    wrk = agents[1] if len(agents) > 1 else {}

    mgr_role = mgr.get("role", "Project Manager")
    mgr_goal = mgr.get("goal", "Oversee the project and delegate effectively")
    mgr_backstory = mgr.get("backstory", "An experienced project leader")

    wrk_role = wrk.get("role", "Research Analyst")
    wrk_goal = wrk.get("goal", "Execute delegated tasks thoroughly")
    wrk_backstory = wrk.get("backstory", "A diligent researcher and analyst")

    return (
        preamble
        + textwrap.dedent(f"""\
        from crewai import Agent, Task, Crew, LLM

        llm = LLM(
            model={model_str!r},
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        manager = Agent(
            role={mgr_role!r},
            goal={mgr_goal!r},
            backstory={mgr_backstory!r},
            llm=llm,
            allow_delegation=True,
            verbose=True,
        )

        worker = Agent(
            role={wrk_role!r},
            goal={wrk_goal!r},
            backstory={wrk_backstory!r},
            llm=llm,
            verbose=True,
        )

        task = Task(
            description="Complete a thorough analysis. Delegate sub-tasks as needed.",
            expected_output="A comprehensive analysis report.",
            agent=manager,
        )

        crew = Crew(
            agents=[manager, worker],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("Crew completed (delegation variant):", type(result).__name__)
        """)
    )


# -- CrewAI variant: reviewed (exercises trust boundary crossing) ----------

def _generate_crewai_reviewed(agents: list[dict[str, str]]) -> str:
    """3 agents: researcher, writer, reviewer.  The reviewer checks
    outputs, exercising trust-boundary-crossing patterns.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    model_str = f"openai/{vllm_model}"
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # Pull roles from extracted agents when available.
    r = agents[0] if agents else {}
    w = agents[1] if len(agents) > 1 else {}
    # The reviewer is always synthetic to guarantee the review step.

    r_role = r.get("role", "Researcher")
    r_goal = r.get("goal", "Research the topic thoroughly")
    r_backstory = r.get("backstory", "An experienced researcher with a keen eye for detail")

    w_role = w.get("role", "Technical Writer")
    w_goal = w.get("goal", "Write clear and accurate reports")
    w_backstory = w.get("backstory", "A skilled writer who turns research into readable content")

    return (
        preamble
        + textwrap.dedent(f"""\
        from crewai import Agent, Task, Crew, LLM

        llm = LLM(
            model={model_str!r},
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        researcher = Agent(
            role={r_role!r},
            goal={r_goal!r},
            backstory={r_backstory!r},
            llm=llm,
            verbose=True,
        )

        writer = Agent(
            role={w_role!r},
            goal={w_goal!r},
            backstory={w_backstory!r},
            llm=llm,
            verbose=True,
        )

        reviewer = Agent(
            role="Quality Reviewer",
            goal="Review and validate outputs for accuracy",
            backstory="You are a careful reviewer who checks for errors and inconsistencies.",
            llm=llm,
            verbose=True,
        )

        task1 = Task(
            description="Research the topic thoroughly.",
            expected_output="Research findings.",
            agent=researcher,
        )

        task2 = Task(
            description="Write a report based on the research.",
            expected_output="A written report.",
            agent=writer,
        )

        task3 = Task(
            description="Review the report for accuracy and completeness.",
            expected_output="Review feedback.",
            agent=reviewer,
        )

        crew = Crew(
            agents=[researcher, writer, reviewer],
            tasks=[task1, task2, task3],
            verbose=True,
        )

        result = crew.kickoff()
        print("Crew completed (reviewed variant):", type(result).__name__)
        """)
    )


# -- CrewAI variant: tooled (exercises tool.invoked events) ----------------

def _generate_crewai_tooled(
    agents: list[dict[str, str]],
    tasks: list[dict[str, str]],
) -> str:
    """Adds a simple tool (text word counter) to one agent to exercise
    ``tool.invoked`` events.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    model_str = f"openai/{vllm_model}"
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    a = agents[0] if agents else {}
    a_role = a.get("role", "Analyst")
    a_goal = a.get("goal", "Analyse text using available tools")
    a_backstory = a.get("backstory", "A meticulous analyst who uses tools to get precise results")

    return (
        preamble
        + textwrap.dedent(f"""\
        from crewai import Agent, Task, Crew, LLM
        from crewai.tools import tool

        @tool("word_counter")
        def word_counter(text: str) -> str:
            \"\"\"Count the number of words in the provided text.\"\"\"
            count = len(text.split())
            return f"The text contains {{count}} words."

        llm = LLM(
            model={model_str!r},
            base_url={vllm_url!r},
            api_key={api_key!r},
        )

        analyst = Agent(
            role={a_role!r},
            goal={a_goal!r},
            backstory={a_backstory!r},
            llm=llm,
            tools=[word_counter],
            verbose=True,
        )

        task = Task(
            description=(
                "Analyse the following text using the word_counter tool, "
                "then summarise your findings: "
                "'Artificial intelligence is transforming software development "
                "by enabling automated code review, intelligent testing, and "
                "predictive maintenance of complex systems.'"
            ),
            expected_output="Word count result and a brief analysis.",
            agent=analyst,
        )

        crew = Crew(
            agents=[analyst],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("Crew completed (tooled variant):", type(result).__name__)
        """)
    )


def generate_langgraph_script(
    nodes: list[str],
    variant: str = "sequential",
) -> str:
    """Generate a self-contained LangGraph script with topology variety.

    Variants:
      - ``"sequential"``   : basic linear chain (original behaviour)
      - ``"conditional"``  : linear with conditional edge (routing.decision)
      - ``"branching"``    : fan-out / fan-in topology

    Uses ``ChatOpenAI`` from ``langchain_openai`` pointed at the vLLM
    endpoint.
    """
    if variant == "conditional":
        return _generate_langgraph_conditional(nodes)
    if variant == "branching":
        return _generate_langgraph_branching(nodes)
    # Default / "sequential" — original behaviour
    return _generate_langgraph_sequential(nodes)


# -- LangGraph variant: sequential (original) ------------------------------

def _generate_langgraph_sequential(nodes: list[str]) -> str:
    """Original sequential LangGraph script generator (unchanged)."""
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

    # Build body without textwrap.dedent to avoid indent mismatch with
    # embedded nl.join() content (which is already at 0-indent).
    body = (
        "from langgraph.graph import StateGraph, START, END\n"
        "from langchain_openai import ChatOpenAI\n"
        "from langchain_core.messages import HumanMessage, SystemMessage\n"
        "\n"
        f"llm = ChatOpenAI(\n"
        f"    model={vllm_model!r},\n"
        f"    base_url={vllm_url!r},\n"
        f"    api_key={api_key!r},\n"
        f")\n"
        "\n"
        + nl.join(node_funcs) + "\n"
        "\n"
        "graph = StateGraph(dict)\n"
        + nl.join(add_nodes) + "\n"
        + nl.join(edges) + "\n"
        "\n"
        "app = graph.compile()\n"
        'result = app.invoke({"input": "Analyze the impact of AI on software development"})\n'
        'print("Graph completed:", list(result.keys()))\n'
    )

    return preamble + body


# -- LangGraph variant: conditional (exercises routing.decision events) ----

def _generate_langgraph_conditional(nodes: list[str]) -> str:
    """Linear graph with a conditional edge based on state, using
    ``add_conditional_edges`` with a routing function.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # We need at least two real nodes plus the routing targets.
    if not nodes:
        nodes = ["analyser", "writer"]
    nodes = nodes[:4]

    # The first node will have a conditional edge that routes to either
    # the second node or directly to END.
    first = nodes[0]
    second = nodes[1] if len(nodes) > 1 else "writer"

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

        def {first}_fn(state: dict) -> dict:
            result = llm.invoke([
                SystemMessage(content="You are a {first} node. Decide if more work is needed."),
                HumanMessage(content=f"Analyse: {{state.get('input', 'topic')}}"),
            ])
            state["{first}_output"] = result.content
            # Set a flag the router will inspect.
            state["needs_more"] = len(result.content) > 20
            return state

        def {second}_fn(state: dict) -> dict:
            result = llm.invoke([
                SystemMessage(content="You are a {second} node."),
                HumanMessage(content=f"Expand on: {{state.get('{first}_output', '')}}"),
            ])
            state["{second}_output"] = result.content
            return state

        def route_decision(state: dict) -> str:
            \"\"\"Route to the next node or finish based on state.\"\"\"
            if state.get("needs_more", False):
                return "{second}"
            return "end"

        graph = StateGraph(dict)
        graph.add_node("{first}", {first}_fn)
        graph.add_node("{second}", {second}_fn)

        graph.add_edge(START, "{first}")
        graph.add_conditional_edges(
            "{first}",
            route_decision,
            {{"{second}": "{second}", "end": END}},
        )
        graph.add_edge("{second}", END)

        app = graph.compile()
        result = app.invoke({{"input": "Analyze the impact of AI on software development"}})
        print("Graph completed (conditional variant):", list(result.keys()))
        """)
    )


# -- LangGraph variant: branching (fan-out topology) -----------------------

def _generate_langgraph_branching(nodes: list[str]) -> str:
    """Graph with multiple parallel paths that merge at a collector node."""
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # Ensure we have at least two branch nodes.
    if not nodes or len(nodes) < 2:
        nodes = ["branch_a", "branch_b"]
    nodes = nodes[:4]

    branch_funcs: list[str] = []
    add_nodes_lines: list[str] = []
    fan_out_edges: list[str] = []
    fan_in_edges: list[str] = []
    for node_name in nodes:
        branch_funcs.append(textwrap.dedent(f"""\
            def {node_name}_fn(state: dict) -> dict:
                result = llm.invoke([
                    SystemMessage(content="You are the {node_name} branch of a parallel pipeline."),
                    HumanMessage(content=f"Process: {{state.get('input', 'topic')}}"),
                ])
                state["{node_name}_output"] = result.content
                return state
        """))
        add_nodes_lines.append(f'graph.add_node("{node_name}", {node_name}_fn)')
        fan_out_edges.append(f'graph.add_edge(START, "{node_name}")')
        fan_in_edges.append(f'graph.add_edge("{node_name}", "collector")')

    nl = chr(10)

    # Build body without textwrap.dedent to avoid indent mismatch with
    # embedded nl.join() content (which is already at 0-indent).
    body = (
        "from langgraph.graph import StateGraph, START, END\n"
        "from langchain_openai import ChatOpenAI\n"
        "from langchain_core.messages import HumanMessage, SystemMessage\n"
        "\n"
        f"llm = ChatOpenAI(\n"
        f"    model={vllm_model!r},\n"
        f"    base_url={vllm_url!r},\n"
        f"    api_key={api_key!r},\n"
        f")\n"
        "\n"
        + nl.join(branch_funcs) + "\n"
        "\n"
        'def collector_fn(state: dict) -> dict:\n'
        '    """Merge outputs from all branches."""\n'
        '    branch_keys = [k for k in state if k.endswith("_output")]\n'
        '    combined = "; ".join(str(state[k])[:200] for k in branch_keys)\n'
        '    result = llm.invoke([\n'
        '        SystemMessage(content="Summarise the following branch outputs."),\n'
        '        HumanMessage(content=combined),\n'
        '    ])\n'
        '    state["final_output"] = result.content\n'
        '    return state\n'
        "\n"
        "graph = StateGraph(dict)\n"
        + nl.join(add_nodes_lines) + "\n"
        'graph.add_node("collector", collector_fn)\n'
        "\n"
        + nl.join(fan_out_edges) + "\n"
        + nl.join(fan_in_edges) + "\n"
        'graph.add_edge("collector", END)\n'
        "\n"
        "app = graph.compile()\n"
        'result = app.invoke({"input": "Analyze the impact of AI on software development"})\n'
        'print("Graph completed (branching variant):", list(result.keys()))\n'
    )

    return preamble + body


def generate_autogen_script(
    agents: list[dict[str, str]],
    variant: str = "sequential",
) -> str:
    """Generate a self-contained AutoGen script with topology variety.

    Variants:
      - ``"sequential"`` : basic two-agent chat (original behaviour)
      - ``"twoway"``     : two-agent chat with a simple tool
      - ``"groupchat"``  : GroupChat with 3+ agents and GroupChatManager
    """
    if variant == "groupchat":
        return _generate_autogen_groupchat(agents)
    if variant == "twoway":
        return _generate_autogen_twoway(agents)
    # Default / "sequential" — original behaviour
    return _generate_autogen_sequential(agents)


# -- AutoGen variant: sequential (original) --------------------------------

def _generate_autogen_sequential(agents: list[dict[str, str]]) -> str:
    """Original two-agent AutoGen script generator (unchanged)."""
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


# -- AutoGen variant: twoway (basic with tool use) -------------------------

def _generate_autogen_twoway(agents: list[dict[str, str]]) -> str:
    """Two-agent chat with a simple registered tool to exercise
    ``tool.invoked`` events.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    a1 = agents[0] if agents else {"name": "assistant"}
    a2 = agents[1] if len(agents) > 1 else {"name": "user_proxy"}

    return (
        preamble
        + textwrap.dedent(f"""\
        import autogen

        config_list = [{{
            "model": {vllm_model!r},
            "base_url": {vllm_url!r},
            "api_key": {api_key!r},
        }}]

        def word_count(text: str) -> str:
            \"\"\"Count words in the given text.\"\"\"
            return f"Word count: {{len(text.split())}}"

        assistant = autogen.AssistantAgent(
            name={a1.get("name", "assistant")!r},
            system_message="You are a helpful assistant. Use the word_count tool when asked to analyse text.",
            llm_config={{"config_list": config_list}},
        )

        user_proxy = autogen.UserProxyAgent(
            name={a2.get("name", "user_proxy")!r},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
        )

        # Register the tool with both agents.
        assistant.register_for_llm(name="word_count", description="Count words in text")(word_count)
        user_proxy.register_for_execution(name="word_count")(word_count)

        user_proxy.initiate_chat(
            assistant,
            message="Use the word_count tool to count the words in: 'AI agents are transforming how we build software systems.'",
        )
        print("Chat completed (twoway variant)")
        """)
    )


# -- AutoGen variant: groupchat (exercises dynamic speaker selection) ------

def _generate_autogen_groupchat(agents: list[dict[str, str]]) -> str:
    """GroupChat with 3+ agents and a GroupChatManager to exercise
    dynamic speaker selection events.
    """
    vllm_url, vllm_model, api_key, repo_root = _env_config()
    preamble = _env_preamble(vllm_url, api_key, repo_root)

    # Ensure at least 3 agents.
    defaults = [
        {"name": "planner", "system_message": "You are a strategic planner."},
        {"name": "coder", "system_message": "You are a software developer."},
        {"name": "reviewer", "system_message": "You review code and plans for quality."},
    ]
    effective_agents = list(agents) if agents else []
    while len(effective_agents) < 3:
        effective_agents.append(defaults[len(effective_agents) % len(defaults)])

    # Cap at 5 agents.
    effective_agents = effective_agents[:5]

    agent_defs: list[str] = []
    agent_vars: list[str] = []
    for i, a in enumerate(effective_agents):
        var = f"agent_{i}"
        agent_defs.append(textwrap.dedent(f"""\
        {var} = autogen.AssistantAgent(
            name={a.get("name", f"agent_{i}")!r},
            system_message={a.get("system_message", "You are a helpful assistant.")!r},
            llm_config={{"config_list": config_list}},
        )"""))
        agent_vars.append(var)

    nl = chr(10)
    agents_list = ", ".join(agent_vars)

    # Build body without textwrap.dedent to avoid indent mismatch with
    # embedded nl.join() content (which is already at 0-indent).
    body = (
        "import autogen\n"
        "\n"
        f"config_list = [{{\n"
        f'    "model": {vllm_model!r},\n'
        f'    "base_url": {vllm_url!r},\n'
        f'    "api_key": {api_key!r},\n'
        f"}}]\n"
        "\n"
        + nl.join(agent_defs) + "\n"
        "\n"
        'user_proxy = autogen.UserProxyAgent(\n'
        '    name="user_proxy",\n'
        '    human_input_mode="NEVER",\n'
        '    max_consecutive_auto_reply=0,\n'
        '    code_execution_config=False,\n'
        ')\n'
        '\n'
        'group_chat = autogen.GroupChat(\n'
        f'    agents=[user_proxy, {agents_list}],\n'
        '    messages=[],\n'
        '    max_round=6,\n'
        ')\n'
        '\n'
        'manager = autogen.GroupChatManager(\n'
        '    groupchat=group_chat,\n'
        f'    llm_config={{"config_list": config_list}},\n'
        ')\n'
        '\n'
        'user_proxy.initiate_chat(\n'
        '    manager,\n'
        '    message="Design a simple REST API for a task management application. Plan it, write the code, then review it.",\n'
        ')\n'
        'print("GroupChat completed (groupchat variant)")\n'
    )

    return preamble + body


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
    """Entry point.  Returns 0 on success, 1 on failure.

    Supports ``--tier1.5`` flag to run only the import-and-call tier.
    Without the flag, runs full Tier 2 synthetic generation.
    """
    # Parse --tier1.5 flag
    tier_1_5_mode = "--tier1.5" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--tier1.5"]

    if len(args) < 3:
        print(
            "Usage: synthetic_harness.py [--tier1.5] <repo_path> <vllm_host> <output_dir>",
            file=sys.stderr,
        )
        return 1

    repo_path = Path(args[0])
    vllm_host = args[1]
    output_dir = args[2]
    events_file = os.environ.get(
        "STRATUM_EVENTS_FILE",
        os.path.join(output_dir, "stratum_events.jsonl"),
    )

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

    # -- Tier 1.5 mode: import-and-call only -------------------------------
    if tier_1_5_mode:
        return try_tier_1_5(repo_path, vllm_host, output_dir, timeout=180)

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

    # -- Select topology variant -------------------------------------------
    variant = select_variant(repo_path, best_framework)
    print(
        f"[synthetic] Selected variant: {variant} (for {best_framework})",
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
            script = generate_crewai_script(agents, tasks, variant=variant)

        elif best_framework == "langgraph":
            nodes = extract_langgraph_nodes(repo_path)
            print(
                f"[synthetic] Extracted {len(nodes)} graph nodes: {nodes}",
                file=sys.stderr,
            )
            script = generate_langgraph_script(nodes, variant=variant)

        elif best_framework == "autogen":
            agents = extract_autogen_agents(repo_path)
            print(
                f"[synthetic] Extracted {len(agents)} agents",
                file=sys.stderr,
            )
            script = generate_autogen_script(agents, variant=variant)

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
