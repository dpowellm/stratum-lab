"""Probe execution — lightweight viability test.

Runs a single 30-second execution to verify:
1. pip install succeeds
2. Primary imports resolve
3. Entry point reaches agent code
4. Patcher activates and emits >=1 event

Results: PROBE_PASS, PROBE_INSTALL_FAIL, PROBE_IMPORT_FAIL,
         PROBE_RUNTIME_FAIL, PROBE_NO_EVENTS, PROBE_TIMEOUT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProbeResult(Enum):
    PASS = "pass"
    INSTALL_FAIL = "install_fail"
    IMPORT_FAIL = "import_fail"
    RUNTIME_FAIL = "runtime_fail"
    NO_EVENTS = "no_events"
    TIMEOUT = "timeout"


@dataclass
class ProbeOutcome:
    result: ProbeResult
    events_emitted: int
    install_seconds: float
    runtime_seconds: float
    error_message: Optional[str]
    entry_point_used: str
    framework_detected: Optional[str]
    repair_applied: Optional[str]  # What we did to make it work


def run_probe(
    repo_scan: Dict[str, Any],
    entry_candidates: List[Dict[str, Any]],
    timeout_seconds: int = 30,
    vllm_url: str = "http://localhost:8000/v1",
) -> ProbeOutcome:
    """Run a single probe execution against a repo.

    Tries entry point candidates in order until one works.
    Applies repair strategies (env var injection, mock services) if needed.

    Uses the same Docker container infrastructure as full execution,
    but with a shorter timeout and only 1 run.
    """
    for candidate in entry_candidates:
        result = _execute_probe(repo_scan, candidate, timeout_seconds, vllm_url)

        if result.result == ProbeResult.PASS:
            return result

        if result.result == ProbeResult.INSTALL_FAIL:
            repaired = _attempt_dep_repair(repo_scan, result.error_message)
            if repaired:
                result = _execute_probe(
                    repo_scan, candidate, timeout_seconds, vllm_url,
                    modified_requirements=repaired,
                )
                if result.result == ProbeResult.PASS:
                    result.repair_applied = "removed_blocking_deps"
                    return result
            return result

        if result.result in (
            ProbeResult.IMPORT_FAIL,
            ProbeResult.RUNTIME_FAIL,
            ProbeResult.NO_EVENTS,
        ):
            continue

    return ProbeOutcome(
        result=ProbeResult.RUNTIME_FAIL,
        events_emitted=0,
        install_seconds=0,
        runtime_seconds=0,
        error_message="all entry point candidates exhausted",
        entry_point_used="none",
        framework_detected=None,
        repair_applied=None,
    )


def _execute_probe(
    repo_scan: Dict[str, Any],
    candidate: Dict[str, Any],
    timeout_seconds: int,
    vllm_url: str,
    modified_requirements: Optional[str] = None,
) -> ProbeOutcome:
    """Execute a single probe attempt with the given entry point.

    In production this calls container.run_container() with:
    - timeout=30s (not the full 120-300s)
    - env: OPENAI_API_KEY=sk-stratum, OPENAI_BASE_URL={vllm_url}
    - STRATUM_PROBE_MODE=1
    - Single input: "test" or first example input from README
    """
    # Stub: actual implementation uses Docker container infrastructure
    return ProbeOutcome(
        result=ProbeResult.PASS,
        events_emitted=0,
        install_seconds=0,
        runtime_seconds=0,
        error_message=None,
        entry_point_used=candidate.get("command", ""),
        framework_detected=repo_scan.get("primary_framework"),
        repair_applied=None,
    )


def _attempt_dep_repair(
    repo_scan: Dict[str, Any],
    error_msg: Optional[str],
) -> Optional[str]:
    """Try to fix dependency installation by removing problematic packages.

    Common fixes:
    - Remove torch/tensorflow (huge, irrelevant to agent behavior)
    - Replace psycopg2 with psycopg2-binary
    - Remove packages that need system libraries
    """
    requirements = repo_scan.get("requirements_text", "")
    if not requirements:
        return None

    lines = requirements.strip().split("\n")
    fixed_lines: List[str] = []
    removed: List[str] = []

    for line in lines:
        pkg = line.strip().split("==")[0].split(">=")[0].split("<=")[0].strip()

        if pkg.lower() in {
            "torch", "tensorflow", "tensorflow-gpu", "torchaudio", "torchvision",
        }:
            removed.append(pkg)
            continue

        if pkg.lower() == "psycopg2":
            fixed_lines.append(line.replace("psycopg2", "psycopg2-binary"))
            continue

        fixed_lines.append(line)

    if removed:
        return "\n".join(fixed_lines)

    return None


def probe_batch(
    qualified_repos: List[Dict[str, Any]],
    max_concurrent: int = 10,
    timeout: int = 30,
    vllm_url: str = "http://localhost:8000/v1",
) -> Dict[str, Any]:
    """Probe a batch of repos and return viability results.

    Runs probes concurrently (up to max_concurrent).
    Returns summary with pass/fail counts and per-repo outcomes.
    """
    results: List[Dict[str, Any]] = []
    for repo in qualified_repos:
        entry_candidates = repo.get("entry_point_candidates", [])
        outcome = run_probe(repo, entry_candidates, timeout, vllm_url)
        results.append({
            "repo_id": repo.get("repo_id", ""),
            "outcome": outcome,
        })

    passed = [r for r in results if r["outcome"].result == ProbeResult.PASS]

    by_result: Dict[str, int] = {}
    for r in results:
        res = r["outcome"].result.value
        by_result[res] = by_result.get(res, 0) + 1

    return {
        "total_probed": len(results),
        "passed": len(passed),
        "pass_rate": len(passed) / max(len(results), 1),
        "by_result": by_result,
        "passed_repos": passed,
        "results": results,
    }
