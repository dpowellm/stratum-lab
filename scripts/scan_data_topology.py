#!/usr/bin/env python3
"""Phase 0b: Static source code scan for tool registrations, data domain
indicators, and permission patterns. Runs per-repo after clone.

Usage:
    python scan_data_topology.py <repo_path>

Writes JSON to stdout with keys: tool_registrations, state_keys, patterns_detected.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Skip rules (same as scan_defensive_patterns.py)
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__", "env"}
MAX_FILES = 500

# ---------------------------------------------------------------------------
# Tool registration detection patterns
# ---------------------------------------------------------------------------

# 1. @tool decorator
RE_TOOL_DECORATOR = re.compile(r"@tool\b")
RE_DEF_AFTER_DECORATOR = re.compile(r"def\s+(\w+)\s*\(")

# 2. Tool() constructor
RE_TOOL_CONSTRUCTOR = re.compile(r'Tool\(\s*name\s*=\s*["\'](\w+)["\']')

# 3. StructuredTool
RE_STRUCTURED_TOOL = re.compile(
    r'StructuredTool(?:\.from_function)?\(\s*(?:func\s*=\s*(\w+)|name\s*=\s*["\'](\w+)["\'])'
)

# 4. tools list assignment
RE_TOOLS_LIST = re.compile(r'tools\s*=\s*\[([^\]]+)\]')

# 5. MCP server
RE_MCP_TOOL = re.compile(r'(?:mcp|server)\.tool\(\)')

# 6. API patterns
RE_HTTP_CLIENT = re.compile(r'(?:requests|httpx)\.\s*(?:get|post|put|patch|delete)\s*\(')

# 7. Database patterns
RE_DATABASE = re.compile(r'(?:cursor\.execute|session\.query|\.find\(|\.aggregate\()')

# 8. File write patterns
RE_FILE_WRITE = re.compile(r'(?:open\([^)]*["\']w|Path\([^)]*\.write)')

# ---------------------------------------------------------------------------
# State key detection patterns
# ---------------------------------------------------------------------------

# Dict access: state["key_name"] or state['key_name']
RE_STATE_DICT = re.compile(r'state\s*\[\s*["\'](\w+)["\']\s*\]')

# Attribute access: state.key_name
RE_STATE_ATTR = re.compile(r'state\.(\w+)\b')

# TypedDict field: class XState(TypedDict): field_name: type
RE_TYPEDDICT_CLASS = re.compile(r'class\s+\w+\(TypedDict\)')
RE_TYPEDDICT_FIELD = re.compile(r'^\s+(\w+)\s*:\s*\w+')

# Pydantic field: class State(BaseModel): field_name: type
RE_PYDANTIC_CLASS = re.compile(r'class\s+\w+\(BaseModel\)')

# ---------------------------------------------------------------------------
# Agent/scope detection for attribution
# ---------------------------------------------------------------------------
RE_CLASS_DEF = re.compile(r'^class\s+(\w+)')
RE_AGENT_ASSIGN = re.compile(r'(\w+)\s*=\s*(?:Agent|CrewAgent|ChatAgent)\s*\(')


def _collect_py_files(repo_path: str) -> list[Path]:
    """Collect .py files respecting skip dirs and cap."""
    py_files: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(Path(root) / fname)
                if len(py_files) >= MAX_FILES:
                    return py_files
    return py_files


def _find_enclosing_scope(lines: list[str], line_idx: int) -> str:
    """Find the enclosing class or agent variable name for attribution."""
    for i in range(line_idx, -1, -1):
        m = RE_CLASS_DEF.match(lines[i])
        if m:
            return m.group(1)
        m = RE_AGENT_ASSIGN.search(lines[i])
        if m:
            return m.group(1)
    return "unattributed"


def scan_tool_registrations(repo_path: str) -> dict:
    """Walk all .py files. Detect tool/function registrations."""
    tool_registrations: dict[str, list[str]] = {}
    unattributed_tools: list[str] = []
    patterns_detected = {
        "decorator_tools": 0,
        "constructor_tools": 0,
        "mcp_tools": 0,
        "api_patterns": 0,
        "database_patterns": 0,
        "file_patterns": 0,
    }

    py_files = _collect_py_files(repo_path)

    for fpath in py_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        lines = content.splitlines()

        for i, line in enumerate(lines):
            scope = None
            tool_name = None

            # 1. @tool decorator
            if RE_TOOL_DECORATOR.search(line):
                # Look for def on next few lines
                for j in range(i + 1, min(i + 4, len(lines))):
                    m = RE_DEF_AFTER_DECORATOR.search(lines[j])
                    if m:
                        tool_name = m.group(1)
                        scope = _find_enclosing_scope(lines, i)
                        patterns_detected["decorator_tools"] += 1
                        break

            # 2. Tool() constructor
            m = RE_TOOL_CONSTRUCTOR.search(line)
            if m:
                tool_name = m.group(1)
                scope = _find_enclosing_scope(lines, i)
                patterns_detected["constructor_tools"] += 1

            # 3. StructuredTool
            m = RE_STRUCTURED_TOOL.search(line)
            if m:
                tool_name = m.group(1) or m.group(2)
                scope = _find_enclosing_scope(lines, i)
                patterns_detected["constructor_tools"] += 1

            # 4. tools list
            m = RE_TOOLS_LIST.search(line)
            if m:
                items = m.group(1)
                names = re.findall(r'(\w+)', items)
                scope = _find_enclosing_scope(lines, i)
                for name in names:
                    if name[0].islower():  # likely a function name
                        if scope and scope != "unattributed":
                            tool_registrations.setdefault(scope, []).append(name)
                        else:
                            unattributed_tools.append(name)

            # 5. MCP server
            if RE_MCP_TOOL.search(line):
                for j in range(i + 1, min(i + 4, len(lines))):
                    m2 = RE_DEF_AFTER_DECORATOR.search(lines[j])
                    if m2:
                        tool_name = m2.group(1)
                        scope = _find_enclosing_scope(lines, i)
                        patterns_detected["mcp_tools"] += 1
                        break

            # 6. API patterns
            if RE_HTTP_CLIENT.search(line):
                tool_name = "http_client"
                scope = _find_enclosing_scope(lines, i)
                patterns_detected["api_patterns"] += 1

            # 7. Database patterns
            if RE_DATABASE.search(line):
                tool_name = "database_query"
                scope = _find_enclosing_scope(lines, i)
                patterns_detected["database_patterns"] += 1

            # 8. File write patterns
            if RE_FILE_WRITE.search(line):
                tool_name = "file_write"
                scope = _find_enclosing_scope(lines, i)
                patterns_detected["file_patterns"] += 1

            # Attribute tool to scope
            if tool_name:
                if scope and scope != "unattributed":
                    tool_registrations.setdefault(scope, []).append(tool_name)
                else:
                    unattributed_tools.append(tool_name)

    # Deduplicate lists
    for key in tool_registrations:
        tool_registrations[key] = sorted(set(tool_registrations[key]))
    unattributed_tools = sorted(set(unattributed_tools))

    tools_found = sum(len(v) for v in tool_registrations.values()) + len(unattributed_tools)

    return {
        "tools_found": tools_found,
        "tool_registrations": tool_registrations,
        "unattributed_tools": unattributed_tools,
        "patterns_detected": patterns_detected,
    }


def scan_state_keys(repo_path: str) -> list[str]:
    """Extract state/shared-memory key names from source code."""
    keys: set[str] = set()
    py_files = _collect_py_files(repo_path)

    for fpath in py_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        lines = content.splitlines()
        in_state_class = False

        for i, line in enumerate(lines):
            # Dict access
            for m in RE_STATE_DICT.finditer(line):
                keys.add(m.group(1))

            # Attribute access
            for m in RE_STATE_ATTR.finditer(line):
                attr = m.group(1)
                # Filter out common methods
                if attr not in ("get", "set", "update", "pop", "keys", "values",
                                "items", "copy", "clear", "__init__"):
                    keys.add(attr)

            # TypedDict / Pydantic class fields
            if RE_TYPEDDICT_CLASS.search(line) or RE_PYDANTIC_CLASS.search(line):
                in_state_class = True
                continue

            if in_state_class:
                fm = RE_TYPEDDICT_FIELD.match(line)
                if fm:
                    keys.add(fm.group(1))
                elif line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    in_state_class = False

    return sorted(keys)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scan_data_topology.py <repo_path>", file=sys.stderr)
        sys.exit(1)

    repo_path = sys.argv[1]
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    tool_result = scan_tool_registrations(repo_path)
    state_keys = scan_state_keys(repo_path)

    output = {
        "tool_registrations": tool_result["tool_registrations"],
        "unattributed_tools": tool_result["unattributed_tools"],
        "state_keys": state_keys,
        "tools_found": tool_result["tools_found"],
        "patterns_detected": tool_result["patterns_detected"],
    }

    json.dump(output, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
