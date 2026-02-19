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

    # ── CrewAI: look for TOP-LEVEL functions that return Crew() ──
    for node in ast.iter_child_nodes(tree):
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

    # ── LangGraph: look for .compile() on StateGraph objects ──
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if (isinstance(func, ast.Attribute) and func.attr == "compile"
                            and isinstance(func.value, ast.Name)
                            and assigned_vars.get(func.value.id) == "StateGraph"):
                        targets.append(OrchestrationTarget(
                            framework="langgraph",
                            variable=target.id,
                            call_method="invoke",
                            file_path=file_path,
                            module_path=module_path,
                        ))

    # ── LangGraph: look for TOP-LEVEL functions that build and return a graph ──
    # Only check direct children of the module (not methods inside classes)
    for node in ast.iter_child_nodes(tree):
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

    # ── Extended patterns for class-based and indirect orchestration ──

    # Collect top-level class and function names
    top_classes: dict[str, ast.ClassDef] = {}
    top_functions: dict[str, ast.FunctionDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            top_classes[node.name] = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_functions[node.name] = node

    # Pattern A: Classes whose methods contain Crew(), StateGraph(), GroupChat()
    _ORCH_CLASSES = {"Crew": ("crewai", "kickoff"),
                     "StateGraph": ("langgraph", "invoke"),
                     "GroupChat": ("autogen", "initiate_chat"),
                     "GroupChatManager": ("autogen", "initiate_chat")}
    for cls_name, cls_node in top_classes.items():
        found_fw = found_call = None
        for child in ast.walk(cls_node):
            if isinstance(child, ast.Call):
                f = child.func
                fname = (f.id if isinstance(f, ast.Name) else
                         f.attr if isinstance(f, ast.Attribute) else None)
                if fname in _ORCH_CLASSES:
                    found_fw, found_call = _ORCH_CLASSES[fname]
                    break
        if not found_fw:
            continue
        # Look for convenience functions that instantiate this class
        factory_func = None
        for fn_name, fn_node in top_functions.items():
            for child in ast.walk(fn_node):
                if isinstance(child, ast.Call):
                    f = child.func
                    fname = (f.id if isinstance(f, ast.Name) else
                             f.attr if isinstance(f, ast.Attribute) else None)
                    if fname == cls_name:
                        factory_func = fn_name
                        break
            if factory_func:
                break
        if factory_func:
            # Check if factory calls methods internally (→ direct)
            fn_node = top_functions[factory_func]
            _RUN_METHODS = {"run", "execute", "kickoff", "invoke",
                            "initiate_chat", "process", "start"}
            calls_method = any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr in _RUN_METHODS
                for c in ast.walk(fn_node))
            targets.append(OrchestrationTarget(
                found_fw, f"{factory_func}()",
                "direct" if calls_method else found_call,
                file_path, module_path))
        else:
            targets.append(OrchestrationTarget(
                found_fw, f"{cls_name}()", found_call,
                file_path, module_path))

    # Pattern B: Module-level functions calling .kickoff()/.invoke()/.run() etc.
    _EXEC_METHODS = {"kickoff": "crewai", "invoke": "langgraph",
                     "initiate_chat": "autogen"}
    has_fw = bool(crewai_imports or langgraph_imports or autogen_imports)
    if has_fw:
        _EXEC_METHODS["run"] = (
            "crewai" if crewai_imports else
            "langgraph" if langgraph_imports else "autogen")
    for fn_name, fn_node in top_functions.items():
        if any(t.variable == f"{fn_name}()" for t in targets):
            continue
        for child in ast.walk(fn_node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in _EXEC_METHODS):
                fw = _EXEC_METHODS[child.func.attr]
                targets.append(OrchestrationTarget(
                    fw, f"{fn_name}()", "direct",
                    file_path, module_path))
                break

    # Pattern C: Functions that create StateGraph (even without compile)
    for fn_name, fn_node in top_functions.items():
        if any(t.variable == f"{fn_name}()" for t in targets):
            continue
        for child in ast.walk(fn_node):
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == "StateGraph"):
                targets.append(OrchestrationTarget(
                    "langgraph", f"{fn_name}()", "invoke",
                    file_path, module_path))
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
        if target.call_method == "direct":
            # Function handles orchestration internally — just call it
            call_lines.append(f"result = {func_name}()")
        elif target.framework == "crewai":
            call_lines.append(f"result = {func_name}().{target.call_method}()")
        elif target.framework == "langgraph":
            call_lines.append(f"graph = {func_name}()")
            call_lines.append("if hasattr(graph, 'compile'):")
            call_lines.append("    graph = graph.compile()")
            call_lines.append("try:")
            call_lines.append('    result = graph.invoke({"messages": [{"role": "user", "content": "Analyze AI agent frameworks"}]})')
            call_lines.append("except TypeError as _te:")
            call_lines.append('    if "synchronous" in str(_te).lower() or "ainvoke" in str(_te).lower():')
            call_lines.append("        import asyncio")
            call_lines.append('        result = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "Analyze AI agent frameworks"}]}))')
            call_lines.append("    else:")
            call_lines.append("        raise")
        else:
            call_lines.append(f"result = {func_name}()")
    else:
        import_lines.append("import importlib")
        import_lines.append(f"_mod = importlib.import_module({target.module_path!r})")
        if target.framework == "crewai":
            call_lines.append(f"result = _mod.{var}.{target.call_method}()")
        elif target.framework == "langgraph":
            call_lines.append("try:")
            call_lines.append(
                f'    result = _mod.{var}.{target.call_method}'
                f'({{"messages": [{{"role": "user", "content": "Analyze AI agent frameworks"}}]}})')
            call_lines.append("except TypeError as _te:")
            call_lines.append('    if "synchronous" in str(_te).lower() or "ainvoke" in str(_te).lower():')
            call_lines.append("        import asyncio")
            call_lines.append(
                f'        result = asyncio.run(_mod.{var}.ainvoke'
                f'({{"messages": [{{"role": "user", "content": "Analyze AI agent frameworks"}}]}}))')
            call_lines.append("    else:")
            call_lines.append("        raise")
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
    lines.append("# Override non-OpenAI LLM constructors to use vLLM")
    lines.append("try:")
    lines.append("    from langchain_openai import ChatOpenAI as _StratumLLM")
    lines.append("    _stratum_llm = _StratumLLM(")
    lines.append('        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),')
    lines.append('        api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-stratum000000000000000000000000000000000000000000000000"),')
    lines.append('        model=os.environ.get("STRATUM_VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),')
    lines.append("        max_tokens=512,")
    lines.append("    )")
    lines.append("    try:")
    lines.append("        import langchain.chat_models as _chat_mod")
    lines.append("        if hasattr(_chat_mod, 'init_chat_model'):")
    lines.append("            _chat_mod.init_chat_model = lambda *a, **kw: _stratum_llm")
    lines.append("    except ImportError:")
    lines.append("        pass")
    lines.append("except Exception:")
    lines.append("    pass")
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


def _pick_best_target(targets: list[OrchestrationTarget]) -> OrchestrationTarget:
    """Pick the best target, preferring module-level vars over functions/classes.

    Priority (lower = better):
    1. Module-level compiled graph variables (e.g. ``graph``)
    2. Functions with call_method="direct" (they handle execution internally)
    3. Functions returning framework objects (e.g. ``create_crew()``)
    4. Class instantiation targets (e.g. ``Workflow()``) — risky, often needs args
    """
    if not targets:
        raise ValueError("No targets")

    def _score(t: OrchestrationTarget) -> tuple:
        is_func = t.variable.endswith("()")
        is_direct = t.call_method == "direct"
        # Check if the variable name looks like a class (starts with uppercase)
        var_base = t.variable.rstrip("()")
        is_class = var_base[0:1].isupper() if var_base else False
        is_private = var_base.startswith("_")

        # Score tuple: lower is better
        # (is_function_call, is_class_instantiation, not_direct, is_private)
        if not is_func:
            # Module-level variable — best option
            return (0, is_private, 0, 0)
        elif is_direct:
            # Direct-call function — second best
            return (1, is_private, 0, 0)
        elif not is_class:
            # Function returning framework object — third
            return (2, is_private, 0, 0)
        else:
            # Class instantiation — worst (often needs constructor args)
            return (3, is_private, 0, 0)

    return sorted(targets, key=_score)[0]


def main() -> int:
    """CLI entry point.

    Takes a file path, analyzes it, and either prints the original path
    (if it already has __main__) or writes a wrapper and prints that path.

    Falls back to scanning the entire repo if the primary file yields no
    usable targets.
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

    # Also scan the whole repo for additional targets (broader search)
    repo_targets = scan_repo(repo_root)

    # Merge: primary file targets + repo-wide targets (deduplicate)
    seen = {(t.module_path, t.variable) for t in targets}
    for rt in repo_targets:
        key = (rt.module_path, rt.variable)
        if key not in seen:
            targets.append(rt)
            seen.add(key)

    if not targets:
        print(file_path)
        return 1

    # Pick the best target
    target = _pick_best_target(targets)

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
