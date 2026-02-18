#!/usr/bin/env python3
"""Static defensive pattern scanner for AI agent repositories.

Walks all .py files in a cloned repo BEFORE execution. Detects seven categories
of defensive coding patterns. Maps each category to the STRAT- finding it defends
against. Outputs defensive_patterns.json.

Usage:
    python scan_defensive_patterns.py <repo_dir> <output_json_path>
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Compiled regex patterns (module level for performance)
# ---------------------------------------------------------------------------

DELEGATION_PATTERNS = [
    re.compile(r'\.kickoff\('),
    re.compile(r'\.invoke\('),
    re.compile(r'agent\.run\('),
    re.compile(r'run_agent\('),
    re.compile(r'\.execute\('),
    re.compile(r'graph\.stream\('),
    re.compile(r'initiate_chat\('),
    re.compile(r'\.send\('),
    re.compile(r'crew\.kickoff\('),
]

TIMEOUT_PATTERNS = [
    (re.compile(r'timeout\s*=\s*(\d+)'), 'timeout'),
    (re.compile(r'max_retries\s*=\s*(\d+)'), 'max_retries'),
    (re.compile(r'max_iterations?\s*=\s*(\d+)'), 'max_iterations'),
    (re.compile(r'recursion_limit\s*=\s*(\d+)'), 'recursion_limit'),
    (re.compile(r'max_rpm\s*=\s*(\d+)'), 'max_rpm'),
    (re.compile(r'max_execution_time\s*=\s*(\d+)'), 'max_execution_time'),
]

OUTPUT_VALIDATION_PATTERNS = [
    (re.compile(r'class\s+(\w+)\(BaseModel\)'), 'pydantic_model'),
    (re.compile(r'class\s+(\w+)\(TypedDict\)'), 'typed_dict'),
    (re.compile(r'output_pydantic\s*='), 'crewai_output_pydantic'),
    (re.compile(r'output_json\s*='), 'crewai_output_json'),
    (re.compile(r'json\.loads\('), 'json_parse'),
    (re.compile(r'jsonschema\.validate\('), 'json_schema'),
    (re.compile(r're\.(match|search|fullmatch)\('), 'regex_validation'),
    (re.compile(r'\.model_validate\('), 'pydantic_v2_validate'),
    (re.compile(r'\.parse_obj\('), 'pydantic_v1_validate'),
    (re.compile(r'StructuredOutput'), 'langchain_structured'),
]

CONCURRENCY_PATTERNS = [
    (re.compile(r'threading\.Lock\(\)'), 'lock'),
    (re.compile(r'threading\.RLock\(\)'), 'rlock'),
    (re.compile(r'asyncio\.Lock\(\)'), 'async_lock'),
    (re.compile(r'Semaphore\('), 'semaphore'),
    (re.compile(r'Queue\('), 'queue'),
    (re.compile(r'\.acquire\(\)'), 'acquire'),
    (re.compile(r'\.release\(\)'), 'release'),
    (re.compile(r'atomic'), 'atomic_reference'),
]

RATE_LIMIT_PATTERNS = [
    (re.compile(r'time\.sleep\('), 'sleep'),
    (re.compile(r'backoff\.'), 'backoff_decorator'),
    (re.compile(r'@retry'), 'retry_decorator'),
    (re.compile(r'tenacity\.'), 'tenacity'),
    (re.compile(r'exponential_backoff'), 'exponential_backoff'),
    (re.compile(r'rate_limit'), 'rate_limit_reference'),
    (re.compile(r'RateLimiter'), 'rate_limiter_class'),
]

INPUT_SANITIZATION_PATTERNS = [
    (re.compile(r'\.strip\(\)'), 'strip'),
    (re.compile(r'\.replace\('), 'replace'),
    (re.compile(r'html\.escape\('), 'html_escape'),
    (re.compile(r'sanitize'), 'sanitize_reference'),
    (re.compile(r'validate_input'), 'validate_input'),
    (re.compile(r'clean_'), 'clean_function'),
    (re.compile(r'assert\s+isinstance\('), 'type_assertion'),
    (re.compile(r'if\s+not\s+isinstance\('), 'type_check'),
]

PROMPT_CONSTRAINT_PATTERNS = [
    (re.compile(r'you are ONLY responsible for', re.IGNORECASE), 'role_boundary'),
    (re.compile(r'do NOT', re.IGNORECASE), 'prohibition'),
    (re.compile(r'you must NOT', re.IGNORECASE), 'prohibition'),
    (re.compile(r'never\s+(attempt|try|generate|fabricate)', re.IGNORECASE), 'prohibition'),
    (re.compile(r'output.*format', re.IGNORECASE), 'output_format'),
    (re.compile(r'respond.*JSON', re.IGNORECASE), 'structured_output_requirement'),
    (re.compile(r'confidence.*level', re.IGNORECASE), 'confidence_requirement'),
    (re.compile(r'if.*unsure.*say', re.IGNORECASE), 'uncertainty_instruction'),
    (re.compile(r'cite.*source', re.IGNORECASE), 'attribution_requirement'),
    (re.compile(r'do not.*assume', re.IGNORECASE), 'anti_assumption'),
]

# API call patterns for rate limiting proximity
API_CALL_RE = re.compile(r'requests\.|httpx\.|aiohttp\.')

# Shared state patterns for concurrency proximity
SHARED_STATE_RE = re.compile(r'(?:global\s+\w|cls\.\w|self\.\w+\s*=)')

# Try/except detection
TRY_RE = re.compile(r'^\s*try\s*:')
EXCEPT_RE = re.compile(r'^\s*except\s*(.*?)\s*:')

# Broad exception types
BROAD_EXCEPTIONS = {'Exception', 'BaseException'}

# Directories to skip
SKIP_DIRS = {'.git', 'venv', '.venv', 'node_modules', '__pycache__'}

MAX_FILES = 500


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------

def find_delegation_lines(lines: list[str]) -> set[int]:
    """Find line numbers containing delegation calls (0-indexed)."""
    delegation_lines: set[int] = set()
    for i, line in enumerate(lines):
        for pat in DELEGATION_PATTERNS:
            if pat.search(line):
                delegation_lines.add(i)
                break
    return delegation_lines


def is_near_delegation(line_idx: int, delegation_lines: set[int], radius: int = 10) -> bool:
    """Check if a line is within `radius` lines of any delegation call."""
    for dl in delegation_lines:
        if abs(line_idx - dl) <= radius:
            return True
    return False


def is_in_string(line: str, match_start: int) -> bool:
    """Heuristic: check if match position is inside a string literal."""
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < match_start:
        remaining = line[i:]
        if not in_single and not in_double:
            if remaining.startswith('"""'):
                in_triple_double = not in_triple_double
                i += 3
                continue
            elif remaining.startswith("'''"):
                in_triple_single = not in_triple_single
                i += 3
                continue
        if not in_triple_single and not in_triple_double:
            if line[i] == '"' and not in_single:
                in_double = not in_double
            elif line[i] == "'" and not in_double:
                in_single = not in_single
        i += 1
    return in_single or in_double or in_triple_single or in_triple_double


def assess_timeout_value(value: int) -> str:
    """Assess timeout effectiveness based on value."""
    if value <= 120:
        return "effective"
    elif value <= 300:
        return "weak"
    else:
        return "ineffective"


def is_from_external_source(line: str, lines: list[str], line_idx: int) -> bool:
    """Heuristic: check if sanitized variable comes from function params."""
    # Check if we're in a function with parameters
    for i in range(max(0, line_idx - 20), line_idx):
        if re.match(r'\s*def\s+\w+\(', lines[i]):
            return True
    return False


def scan_file(file_path: Path, repo_root: Path) -> tuple[list[dict], int]:
    """Scan a single Python file for defensive patterns.

    Returns (patterns_list, files_skipped_count).
    """
    rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return [], 1

    lines = content.splitlines()
    delegation_lines = find_delegation_lines(lines)
    patterns: list[dict] = []

    # --- Category 1: timeout_iteration_guards ---
    for i, line in enumerate(lines):
        for pat, param_name in TIMEOUT_PATTERNS:
            m = pat.search(line)
            if m:
                value = int(m.group(1))
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "timeout_iteration_guards",
                    "pattern_detail": {
                        "parameter": param_name,
                        "value": value,
                        "value_assessment": assess_timeout_value(value),
                    },
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    # --- Category 2: exception_handling_topology ---
    i = 0
    while i < len(lines):
        if TRY_RE.match(lines[i]):
            # Find matching except block(s)
            try_indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines):
                stripped = lines[j].lstrip()
                current_indent = len(lines[j]) - len(lines[j].lstrip()) if stripped else try_indent + 1
                if current_indent <= try_indent and stripped:
                    if EXCEPT_RE.match(lines[j]):
                        em = EXCEPT_RE.match(lines[j])
                        except_clause = em.group(1).strip() if em else ""

                        # Determine except type
                        if not except_clause or except_clause == ':':
                            except_type = "bare"
                            exception_classes = []
                        else:
                            # Remove 'as ...' part
                            except_clause = re.sub(r'\s+as\s+\w+', '', except_clause).strip()
                            exception_classes = [c.strip() for c in except_clause.split(',') if c.strip()]
                            if any(c in BROAD_EXCEPTIONS for c in exception_classes):
                                except_type = "broad"
                            else:
                                except_type = "specific"

                        # Analyze except body
                        k = j + 1
                        except_body_lines = 0
                        has_logging = False
                        has_fallback = False
                        has_reraise = False
                        except_indent = len(lines[j]) - len(lines[j].lstrip())

                        while k < len(lines):
                            body_stripped = lines[k].lstrip()
                            body_indent = len(lines[k]) - len(lines[k].lstrip()) if body_stripped else except_indent + 1
                            if body_indent <= except_indent and body_stripped:
                                break
                            if body_stripped:
                                except_body_lines += 1
                                if re.search(r'logging\.(error|warning|info|exception)|print\(', lines[k]):
                                    has_logging = True
                                if re.search(r'\breturn\b|=\s*.+', lines[k]):
                                    has_fallback = True
                                if re.search(r'\braise\b', lines[k]):
                                    has_reraise = True
                            k += 1

                        patterns.append({
                            "file_path": rel_path,
                            "line_number": j + 1,
                            "pattern_category": "exception_handling_topology",
                            "pattern_detail": {
                                "except_type": except_type,
                                "exception_classes": exception_classes,
                                "has_logging": has_logging,
                                "has_fallback": has_fallback,
                                "has_reraise": has_reraise,
                                "except_body_lines": except_body_lines,
                            },
                            "near_delegation_boundary": is_near_delegation(j, delegation_lines),
                        })
                        j = k
                        continue
                    elif not stripped.startswith('#'):
                        break
                j += 1
        i += 1

    # --- Category 3: output_validation ---
    for i, line in enumerate(lines):
        for pat, val_type in OUTPUT_VALIDATION_PATTERNS:
            m = pat.search(line)
            if m:
                detail: dict = {
                    "validation_type": val_type,
                    "structural_position": "agent_output_boundary" if is_near_delegation(i, delegation_lines) else "internal",
                }
                # Extract model name for pydantic/typed_dict
                if val_type in ('pydantic_model', 'typed_dict') and m.lastindex and m.lastindex >= 1:
                    detail["model_name"] = m.group(1)
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "output_validation",
                    "pattern_detail": detail,
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    # --- Category 4: concurrency_controls ---
    for i, line in enumerate(lines):
        for pat, control_type in CONCURRENCY_PATTERNS:
            if pat.search(line):
                near_shared = bool(SHARED_STATE_RE.search(
                    '\n'.join(lines[max(0, i - 10):i + 10])
                ))
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "concurrency_controls",
                    "pattern_detail": {
                        "control_type": control_type,
                        "near_shared_state": near_shared,
                    },
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    # --- Category 5: rate_limiting_backoff ---
    for i, line in enumerate(lines):
        for pat, mechanism in RATE_LIMIT_PATTERNS:
            if pat.search(line):
                has_exponential = bool(re.search(r'backoff|exponential', line, re.IGNORECASE))
                near_api = bool(API_CALL_RE.search(
                    '\n'.join(lines[max(0, i - 10):i + 10])
                ))
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "rate_limiting_backoff",
                    "pattern_detail": {
                        "mechanism": mechanism,
                        "has_exponential": has_exponential,
                        "near_api_call": near_api,
                    },
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    # --- Category 6: input_sanitization ---
    for i, line in enumerate(lines):
        for pat, mechanism in INPUT_SANITIZATION_PATTERNS:
            if pat.search(line):
                from_ext = is_from_external_source(line, lines, i)
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "input_sanitization",
                    "pattern_detail": {
                        "mechanism": mechanism,
                        "target_variable": "",
                        "from_external_source": from_ext,
                    },
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    # --- Category 7: prompt_constraints ---
    for i, line in enumerate(lines):
        # Only detect within string literals
        for pat, constraint_type in PROMPT_CONSTRAINT_PATTERNS:
            m = pat.search(line)
            if m and is_in_string(line, m.start()):
                # Check if in system prompt variable
                in_system = bool(re.search(
                    r'system_prompt|system_message|SystemMessage',
                    '\n'.join(lines[max(0, i - 5):i + 1]),
                    re.IGNORECASE,
                ))
                patterns.append({
                    "file_path": rel_path,
                    "line_number": i + 1,
                    "pattern_category": "prompt_constraints",
                    "pattern_detail": {
                        "constraint_type": constraint_type,
                        "constraint_text_preview": line.strip()[:80],
                        "in_system_prompt": in_system,
                    },
                    "near_delegation_boundary": is_near_delegation(i, delegation_lines),
                })

    return patterns, 0


def build_summary(patterns: list[dict]) -> dict:
    """Build summary rollups from patterns list."""
    summary: dict = {}

    # timeout_iteration_guards
    tg = [p for p in patterns if p["pattern_category"] == "timeout_iteration_guards"]
    summary["timeout_iteration_guards"] = {
        "count": len(tg),
        "near_boundary_count": sum(1 for p in tg if p["near_delegation_boundary"]),
        "effective_count": sum(1 for p in tg if p["pattern_detail"].get("value_assessment") == "effective"),
    }

    # exception_handling_topology
    eh = [p for p in patterns if p["pattern_category"] == "exception_handling_topology"]
    summary["exception_handling_topology"] = {
        "count": len(eh),
        "near_boundary_count": sum(1 for p in eh if p["near_delegation_boundary"]),
        "bare_except_count": sum(1 for p in eh if p["pattern_detail"].get("except_type") == "bare"),
        "has_logging_count": sum(1 for p in eh if p["pattern_detail"].get("has_logging")),
    }

    # output_validation
    ov = [p for p in patterns if p["pattern_category"] == "output_validation"]
    summary["output_validation"] = {
        "count": len(ov),
        "at_boundary_count": sum(1 for p in ov if p["pattern_detail"].get("structural_position") == "agent_output_boundary"),
        "pydantic_count": sum(1 for p in ov if p["pattern_detail"].get("validation_type") in ("pydantic_model", "pydantic_v2_validate", "pydantic_v1_validate")),
        "typed_dict_count": sum(1 for p in ov if p["pattern_detail"].get("validation_type") == "typed_dict"),
    }

    # concurrency_controls
    cc = [p for p in patterns if p["pattern_category"] == "concurrency_controls"]
    summary["concurrency_controls"] = {
        "count": len(cc),
        "near_shared_state_count": sum(1 for p in cc if p["pattern_detail"].get("near_shared_state")),
    }

    # rate_limiting_backoff
    rl = [p for p in patterns if p["pattern_category"] == "rate_limiting_backoff"]
    summary["rate_limiting_backoff"] = {
        "count": len(rl),
        "has_exponential_count": sum(1 for p in rl if p["pattern_detail"].get("has_exponential")),
    }

    # input_sanitization
    is_pats = [p for p in patterns if p["pattern_category"] == "input_sanitization"]
    summary["input_sanitization"] = {
        "count": len(is_pats),
        "from_external_count": sum(1 for p in is_pats if p["pattern_detail"].get("from_external_source")),
    }

    # prompt_constraints
    pc = [p for p in patterns if p["pattern_category"] == "prompt_constraints"]
    summary["prompt_constraints"] = {
        "count": len(pc),
        "role_boundary_count": sum(1 for p in pc if p["pattern_detail"].get("constraint_type") == "role_boundary"),
        "prohibition_count": sum(1 for p in pc if p["pattern_detail"].get("constraint_type") == "prohibition"),
        "uncertainty_instruction_count": sum(1 for p in pc if p["pattern_detail"].get("constraint_type") == "uncertainty_instruction"),
    }

    return summary


def scan_repo(repo_dir: str) -> dict:
    """Scan an entire repository for defensive patterns."""
    repo_path = Path(repo_dir)
    all_patterns: list[dict] = []
    files_scanned = 0
    files_skipped = 0
    delegation_boundaries_found = 0

    # Collect .py files, respecting skip dirs and cap
    py_files: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith('.py'):
                py_files.append(Path(root) / f)
                if len(py_files) >= MAX_FILES:
                    break
        if len(py_files) >= MAX_FILES:
            break

    for py_file in py_files:
        patterns, skipped = scan_file(py_file, repo_path)
        if skipped:
            files_skipped += skipped
        else:
            files_scanned += 1
        all_patterns.extend(patterns)

    # Count unique delegation boundary files
    delegation_files = set()
    for p in all_patterns:
        if p["near_delegation_boundary"]:
            delegation_files.add(p["file_path"])
    delegation_boundaries_found = len(delegation_files)

    # Derive repo name
    repo_name = repo_path.name

    return {
        "repo_full_name": repo_name,
        "scan_version": "1.0",
        "scan_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files_scanned": files_scanned,
        "files_skipped": files_skipped,
        "total_patterns_found": len(all_patterns),
        "delegation_boundaries_found": delegation_boundaries_found,
        "patterns": all_patterns,
        "summary": build_summary(all_patterns),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: scan_defensive_patterns.py <repo_dir> <output_json>", file=sys.stderr)
        sys.exit(1)

    repo_dir = sys.argv[1]
    output_path = sys.argv[2]

    try:
        result = scan_repo(repo_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        error_result = {
            "error": str(e),
            "patterns": [],
            "summary": {},
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
