#!/usr/bin/env python3
"""AST-based __main__ block detection and injection.

Parses a Python file to find orchestration objects (Crew, StateGraph,
GroupChat, etc.) and determines whether a __main__ execution block exists.
If no __main__ block is found, generates a wrapper script that imports and
calls the orchestration object.

Usage:
    python inject_main.py <file_path>

Output:
    Prints the path to run (original file if it has __main__, or a
    generated wrapper file).  Exit code 0 = success, 1 = nothing found.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple


class OrchestrationTarget(NamedTuple):
    """An orchestration object found via AST analysis."""
    framework: str      # crewai, langgraph, autogen
    variable: str       # variable name holding the object
    call_method: str    # method to call (.kickoff, .invoke, etc.)
    file_path: str      # source file
    module_path: str    # importable module path


# ── AST Analysis ─────────────────────────────────────────────────────────

def _has_main_guard(tree: ast.AST) -> bool:
    """Check if AST has an ``if __name__ == "__main__"`` block."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                left = test.left
                if (isinstance(left, ast.Name) and left.id == "__name__"
                        and len(test.comparators) == 1):
                    comp = test.comparators[0]
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        return True
                # Also check reversed: "__main__" == __name__
                if isinstance(left, ast.Constant) and left.value == "__main__":
                    return True
    return False


def _find_orchestration_targets(
    tree: ast.AST,
    file_path: str,
    repo_root: str,
) -> list[OrchestrationTarget]:
    """Find orchestration objects in the AST and return targets."""
    targets: list[OrchestrationTarget] = []

    # Compute importable module path from file path
    rel = os.path.relpath(file_path, repo_root)
    module_path = rel.replace(os.sep, ".").replace("/", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    if module_path.endswith(".__init__"):
        module_path = module_path[:-9]

    # Track what's imported from which framework
    crewai_imports: set[str] = set()
    langgraph_imports: set[str] = set()
    autogen_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in (node.names or []):
                name = alias.asname or alias.name
                if "crewai" in mod:
                    crewai_imports.add(name)
                elif "langgraph" in mod:
                    langgraph_imports.add(name)
                elif "autogen" in mod:
                    autogen_imports.add(name)

    # Track variable assignments
    assigned_vars: dict[str, str] = {}  # var_name -> class_name

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        assigned_vars[target.id] = func.id
                    elif isinstance(func, ast.Attribute):
                        assigned_vars[target.id] = func.attr

    # ── CrewAI: look for Crew() assignments ──
    for var_name, class_name in assigned_vars.items():
        if class_name == "Crew" and ("Crew" in crewai_imports or "crewai" in str(crewai_imports)):
            targets.append(OrchestrationTarget(
                framework="crewai",
                variable=var_name,
                call_method="kickoff",
                file_path=file_path,
                module_path=module_path,
            ))

    # ── CrewAI: look for methods that return Crew() ──
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value is not None:
                    if isinstance(child.value, ast.Call):
                        func = child.value.func
                        func_name = None
                        if isinstance(func, ast.Name):
                            func_name = func.id
                        elif isinstance(func, ast.Attribute):
                            func_name = func.attr
                        if func_name == "Crew":
                            targets.append(OrchestrationTarget(
                                framework="crewai",
                                variable=f"{node.name}()",
                                call_method="kickoff",
                                file_path=file_path,
                                module_path=module_path,
                            ))

    # ── LangGraph: look for .compile() calls ──
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "compile":
                        targets.append(OrchestrationTarget(
                            framework="langgraph",
                            variable=target.id,
                            call_method="invoke",
                            file_path=file_path,
                            module_path=module_path,
                        ))

    # ── LangGraph: look for functions that build and return a graph ──
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_state_graph = False
            has_compile = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name) and func.id == "StateGraph":
                        has_state_graph = True
                    if isinstance(func, ast.Attribute) and func.attr == "compile":
                        has_compile = True
            if has_state_graph and has_compile:
                targets.append(OrchestrationTarget(
                    framework="langgraph",
                    variable=f"{node.name}()",
                    call_method="invoke",
                    file_path=file_path,
                    module_path=module_path,
                ))

    # ── AutoGen: look for GroupChat or initiate_chat ──
    for var_name, class_name in assigned_vars.items():
        if class_name == "GroupChat":
            targets.append(OrchestrationTarget(
                framework="autogen",
                variable=var_name,
                call_method="initiate_chat",
                file_path=file_path,
                module_path=module_path,
            ))

    # ── AutoGen: look for UserProxyAgent with initiate_chat calls ──
    for var_name, class_name in assigned_vars.items():
        if class_name in ("UserProxyAgent", "ConversableAgent"):
            # Check if there's an initiate_chat call on this variable
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (isinstance(func, ast.Attribute)
                            and func.attr == "initiate_chat"
                            and isinstance(func.value, ast.Name)
                            and func.value.id == var_name):
                        targets.append(OrchestrationTarget(
                            framework="autogen",
                            variable=var_name,
                            call_method="initiate_chat",
                            file_path=file_path,
                            module_path=module_path,
                        ))
                        break

    return targets


def analyze_file(file_path: str, repo_root: str) -> tuple[bool, list[OrchestrationTarget]]:
    """Analyze a Python file for orchestration targets.

    Returns (has_main_guard, targets).
    """
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError, OSError):
        return False, []

    has_main = _has_main_guard(tree)
    targets = _find_orchestration_targets(tree, file_path, repo_root)
    return has_main, targets


def scan_repo(repo_root: str) -> list[OrchestrationTarget]:
    """Scan all .py files in repo for orchestration targets without __main__."""
    skip_dirs = {".git", "venv", ".venv", "node_modules", "__pycache__",
                 "test", "tests", ".tox", ".mypy_cache"}
    results: list[OrchestrationTarget] = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            has_main, targets = analyze_file(fpath, repo_root)
            if targets and not has_main:
                results.extend(targets)

    return results


# ── Wrapper Script Generation ────────────────────────────────────────────

def generate_wrapper(target: OrchestrationTarget, repo_root: str) -> str:
    """Generate a minimal wrapper script that imports and calls the target.

    Uses line-by-line construction to avoid multi-line f-string indentation
    bugs when call_block spans multiple lines.
    """
    var = target.variable
    is_function_call = var.endswith("()")
    func_name = var[:-2] if is_function_call else None

    # Build import + call lines (each element is one line of code)
    import_lines: list[str] = []
    call_lines: list[str] = []

    if is_function_call and func_name:
        import_lines.append(f"from {target.module_path} import {func_name}")
        if target.framework == "crewai":
            call_lines.append(f"result = {func_name}().{target.call_method}()")
        elif target.framework == "langgraph":
            call_lines.append(f"graph = {func_name}()")
            call_lines.append(f'result = graph.invoke({{"input": "Analyze AI agent frameworks"}})')
        else:
            call_lines.append(f"result = {func_name}()")
    else:
        import_lines.append("import importlib")
        import_lines.append(f"_mod = importlib.import_module({target.module_path!r})")
        if target.framework == "crewai":
            call_lines.append(f"result = _mod.{var}.{target.call_method}()")
        elif target.framework == "langgraph":
            call_lines.append(
                f'result = _mod.{var}.{target.call_method}'
                f'({{"input": "Analyze AI agent frameworks"}})')
        elif target.framework == "autogen":
            call_lines.append("# AutoGen: initiate_chat needs a target agent")
            call_lines.append("_agents = [")
            call_lines.append("    v for k, v in vars(_mod).items()")
            call_lines.append(f"    if hasattr(v, 'generate_reply') and k != {var!r}")
            call_lines.append("]")
            call_lines.append("if _agents:")
            call_lines.append(f"    _mod.{var}.initiate_chat(")
            call_lines.append(f'        _agents[0], message="Analyze AI agent frameworks"')
            call_lines.append("    )")
        else:
            call_lines.append(f"result = _mod.{var}.{target.call_method}()")

    # Build the script line by line
    lines: list[str] = []
    lines.append("#!/usr/bin/env python3")
    lines.append(f'"""Auto-generated wrapper for {target.framework} orchestration.')
    lines.append(f"Source: {target.file_path}")
    lines.append('"""')
    lines.append("import os")
    lines.append("import sys")
    lines.append("import signal")
    lines.append("")
    lines.append("# Timeout protection")
    lines.append("def _timeout_handler(signum, frame):")
    lines.append('    raise SystemExit("STRATUM TIMEOUT in inject_main wrapper")')
    lines.append("")
    lines.append('if hasattr(signal, "SIGALRM"):')
    lines.append("    signal.signal(signal.SIGALRM, _timeout_handler)")
    lines.append('    signal.alarm(int(os.environ.get("STRATUM_TIMEOUT_SECONDS", "180")))')
    lines.append("")
    lines.append("# Ensure repo is on path")
    lines.append(f"sys.path.insert(0, {repo_root!r})")
    lines.append(f"src_dir = os.path.join({repo_root!r}, 'src')")
    lines.append("if os.path.isdir(src_dir):")
    lines.append("    sys.path.insert(0, src_dir)")
    lines.append("")
    lines.append("# Set env vars")
    lines.append('os.environ.setdefault("STRATUM_EVENTS_FILE", "/app/stratum_events.jsonl")')
    lines.append('os.environ.setdefault("STRATUM_CAPTURE_PROMPTS", "1")')
    lines.append("")
    lines.append("# Mock unavailable service dependencies so imports don't crash")
    lines.append("import importlib as _il")
    lines.append("from unittest.mock import MagicMock")
    lines.append("for _mod_name in ['pinecone', 'tavily', 'chromadb', 'weaviate', 'qdrant_client',")
    lines.append("                   'firebase_admin', 'supabase', 'notion_client',")
    lines.append("                   'google', 'google.cloud', 'google.auth', 'google.oauth2',")
    lines.append("                   'firecrawl', 'browserbase', 'e2b', 'composio']:")
    lines.append("    if _mod_name not in sys.modules:")
    lines.append("        try:")
    lines.append("            _il.import_module(_mod_name)")
    lines.append("        except ImportError:")
    lines.append("            _m = MagicMock()")
    lines.append("            _m.__path__ = []")
    lines.append("            _m.__file__ = f'mock_{_mod_name}'")
    lines.append("            _m.__spec__ = None")
    lines.append("            sys.modules[_mod_name] = _m")
    lines.append("")
    lines.append("try:")
    for il in import_lines:
        lines.append(f"    {il}")
    for cl in call_lines:
        lines.append(f"    {cl}")
    lines.append('    print("Wrapper execution completed successfully")')
    lines.append("except ImportError as e:")
    lines.append('    print(f"Import failed: {e}", file=sys.stderr)')
    lines.append("    sys.exit(1)")
    lines.append("except Exception as e:")
    lines.append('    print(f"Execution failed: {e}", file=sys.stderr)')
    lines.append("    sys.exit(1)")
    lines.append("")

    script = "\n".join(lines)

    # Compile check — catch indentation bugs before writing
    try:
        compile(script, "<inject_main_wrapper>", "exec")
    except SyntaxError as exc:
        _fallback = (
            "import sys\n"
            f'print("SyntaxError in generated wrapper: {exc}", file=sys.stderr)\n'
            "sys.exit(1)\n"
        )
        return _fallback

    return script


def main() -> int:
    """CLI entry point.

    Takes a file path, analyzes it, and either prints the original path
    (if it already has __main__) or writes a wrapper and prints that path.
    """
    if len(sys.argv) < 2:
        print("Usage: inject_main.py <file_path> [repo_root]", file=sys.stderr)
        return 1

    file_path = sys.argv[1]
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "/tmp/repo"

    if not os.path.isfile(file_path):
        print(f"Error: {file_path} not found", file=sys.stderr)
        return 1

    has_main, targets = analyze_file(file_path, repo_root)

    if has_main:
        # File already has __main__ — use as-is
        print(file_path)
        return 0

    if not targets:
        # No orchestration objects found — nothing to inject
        print(file_path)
        return 1

    # Pick the best target (prefer the first one found)
    target = targets[0]

    # Generate wrapper
    wrapper_content = generate_wrapper(target, repo_root)
    wrapper_path = "/tmp/stratum_wrapper.py"

    try:
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(wrapper_content)
    except OSError as e:
        print(f"Error writing wrapper: {e}", file=sys.stderr)
        return 1

    print(wrapper_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
