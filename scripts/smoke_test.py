#!/usr/bin/env python3
"""
Smoke test: run 3-5 known-good repos through the full pipeline with real Docker execution.

Prerequisites:
  - Docker running
  - vLLM serving at VLLM_URL
  - stratum-lab-runner image built (run: stratum-lab build-image)

Usage:
  python scripts/smoke_test.py --vllm-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXTURE_DIR = PROJECT_ROOT / "eval" / "fixtures" / "smoke_test"
DOCKER_IMAGE = "stratum-lab-runner"
SEPARATOR = "=" * 78
INDENT = "    "

# Map of fixture filename (without .json) to display name
SMOKE_REPOS = {
    "crewai_minimal": "crewai-minimal",
    "langgraph_react": "langgraph-react",
    "openai_bare": "openai-bare",
}


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def check_docker() -> tuple[bool, str]:
    """Check that Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, f"docker info failed (exit {result.returncode})"
    except FileNotFoundError:
        return False, "docker not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"
    except Exception as e:
        return False, str(e)


def check_vllm(vllm_url: str) -> tuple[bool, str]:
    """Check that vLLM endpoint is reachable and responding."""
    models_url = vllm_url.rstrip("/") + "/models"
    try:
        import urllib.request
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_count = len(data.get("data", []))
            return True, f"OK ({model_count} model{'s' if model_count != 1 else ''} available)"
    except ImportError:
        # Fall back to subprocess curl
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "10", models_url],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                model_count = len(data.get("data", []))
                return True, f"OK ({model_count} model{'s' if model_count != 1 else ''} available)"
            return False, f"curl failed (exit {result.returncode})"
        except Exception as e:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def check_runner_image() -> tuple[bool, str]:
    """Check that the stratum-lab-runner Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, f"image '{DOCKER_IMAGE}' not found (run: stratum-lab build-image)"
    except FileNotFoundError:
        return False, "docker not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "docker image inspect timed out"
    except Exception as e:
        return False, str(e)


def check_fixtures() -> tuple[bool, str]:
    """Check that all smoke test fixture files exist."""
    missing = []
    for fixture_name in SMOKE_REPOS:
        path = FIXTURE_DIR / f"{fixture_name}.json"
        if not path.is_file():
            missing.append(f"{fixture_name}.json")
    if missing:
        return False, f"missing fixtures: {', '.join(missing)}"
    return True, f"OK ({len(SMOKE_REPOS)} fixtures found)"


def run_preflight(vllm_url: str) -> bool:
    """Run all preflight checks. Returns True if all pass."""
    print(f"\n  Preflight:")

    checks = [
        ("Docker", check_docker),
        ("vLLM", lambda: check_vllm(vllm_url)),
        ("Runner image", check_runner_image),
        ("Fixtures", check_fixtures),
    ]

    all_ok = True
    for label, check_fn in checks:
        ok, detail = check_fn()
        status_char = "OK" if ok else "FAIL"
        # Pad the label for alignment
        padded_label = f"{label} ".ljust(20, ".")
        print(f"{INDENT}{padded_label} {detail}")
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(
    fixture_dir: str,
    output_dir: str,
    vllm_url: str,
    timeout_seconds: int = 120,
) -> tuple[int, str, str]:
    """Run stratum-lab pipeline on a fixture directory.

    Returns (returncode, stdout, stderr).
    """
    cmd = [
        sys.executable, "-m", "stratum_lab.cli",
        "pipeline",
        fixture_dir,
        "--target", "3",
        "--vllm-url", vllm_url,
        "--concurrent", "1",
        "--timeout", str(timeout_seconds),
        "--pilot",
        "--pilot-size", "3",
        "--output-dir", output_dir,
        "--no-triage",
        "--no-probe",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 60,  # extra buffer beyond per-repo timeout
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Pipeline timed out"
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def count_events(output_dir: str) -> int:
    """Count total events across all JSONL files in the raw_events directory."""
    events_dir = Path(output_dir) / "raw_events"
    if not events_dir.is_dir():
        return 0

    total = 0
    for jsonl_file in events_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        total += 1
        except Exception:
            pass
    return total


def validate_output(output_dir: str) -> tuple[bool, str]:
    """Run validate_scan.py on the output directory.

    Returns (passed, detail_message).
    """
    validate_script = SCRIPT_DIR / "validate_scan.py"

    if not validate_script.is_file():
        return False, "validate_scan.py not found"

    try:
        result = subprocess.run(
            [sys.executable, str(validate_script), output_dir],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        return passed, "VALID" if passed else "INVALID"
    except subprocess.TimeoutExpired:
        return False, "validation timed out"
    except Exception as e:
        return False, str(e)


def check_behavioral_records(output_dir: str) -> tuple[bool, int]:
    """Check if behavioral record files were produced.

    Returns (has_records, record_count).
    """
    reports_dir = Path(output_dir) / "reports"
    if not reports_dir.is_dir():
        # Also check enriched_graphs as a fallback
        reports_dir = Path(output_dir) / "enriched_graphs"
    if not reports_dir.is_dir():
        return False, 0

    records = list(reports_dir.glob("*_behavioral.json"))
    if not records:
        # Also check for any .json files in enriched_graphs
        enriched_dir = Path(output_dir) / "enriched_graphs"
        if enriched_dir.is_dir():
            records = list(enriched_dir.glob("*.json"))

    return len(records) > 0, len(records)


# ---------------------------------------------------------------------------
# Main smoke test runner
# ---------------------------------------------------------------------------

def run_smoke_test(vllm_url: str, timeout: int = 120) -> bool:
    """Run the full smoke test suite.

    Returns True if all repos pass.
    """
    print(SEPARATOR)
    print("STRATUM SMOKE TEST")
    print(SEPARATOR)

    # --- Preflight ---
    preflight_ok = run_preflight(vllm_url)

    if not preflight_ok:
        print(f"\n{SEPARATOR}")
        print("  Preflight FAILED. Fix the issues above before running smoke tests.")
        print(SEPARATOR)
        return False

    # --- Run each repo through pipeline ---
    print(f"\n  Running {len(SMOKE_REPOS)} repos through pipeline...")

    results: list[dict] = []
    tmpdir = tempfile.mkdtemp(prefix="stratum_smoke_")

    try:
        for fixture_name, display_name in SMOKE_REPOS.items():
            output_dir = os.path.join(tmpdir, fixture_name)
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.monotonic()

            # Run pipeline
            returncode, stdout, stderr = run_pipeline(
                fixture_dir=str(FIXTURE_DIR),
                output_dir=output_dir,
                vllm_url=vllm_url,
                timeout_seconds=timeout,
            )

            elapsed = time.monotonic() - start_time

            # Gather results
            event_count = count_events(output_dir)
            has_records, record_count = check_behavioral_records(output_dir)

            if returncode == 0:
                valid, valid_detail = validate_output(output_dir)
                status = "PASS" if valid else "PARTIAL"
            elif returncode == -1:
                status = "TIMEOUT"
                valid = False
                valid_detail = "N/A"
            else:
                status = "FAIL"
                valid = False
                valid_detail = "pipeline failed"

            result = {
                "fixture": fixture_name,
                "display_name": display_name,
                "status": status,
                "returncode": returncode,
                "event_count": event_count,
                "record_count": record_count,
                "valid": valid,
                "valid_detail": valid_detail,
                "elapsed_s": round(elapsed, 1),
                "stdout": stdout,
                "stderr": stderr,
            }
            results.append(result)

            # Print inline result
            padded_name = f"{display_name} ".ljust(20, ".")
            record_status = "VALID" if valid else valid_detail
            print(f"{INDENT}{padded_name} [{status}] {event_count} events, "
                  f"behavioral record: {record_status} ({elapsed:.0f}s)")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Summary ---
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)

    print(f"\n{SEPARATOR}")
    print(f"  Results: {passed_count}/{total_count} repos produced valid output")

    # Print failure details if any
    failed = [r for r in results if r["status"] != "PASS"]
    if failed:
        print(f"\n  Failures:")
        for r in failed:
            print(f"    {r['display_name']}: {r['status']} (exit={r['returncode']})")
            if r["stderr"]:
                # Print first few lines of stderr
                stderr_lines = r["stderr"].strip().split("\n")
                for line in stderr_lines[:5]:
                    print(f"      {line}")
                if len(stderr_lines) > 5:
                    print(f"      ... ({len(stderr_lines) - 5} more lines)")

    print(SEPARATOR)

    return passed_count == total_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: run known-good repos through the full stratum-lab pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  - Docker running
  - vLLM serving at --vllm-url
  - stratum-lab-runner image built (run: stratum-lab build-image)

This script runs 3 hardcoded fixture repos through the real Docker execution
path and validates the output. It is designed to catch regressions in the
pipeline before running a full 1000+ repo scan.

Fixture repos:
  crewai-minimal ..... 2 agents, 1 tool, delegation edge
  langgraph-react .... 1 agent, 2 tools, ReAct loop
  openai-bare ........ 1 agent, 0 tools, bare LLM calls
        """,
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible endpoint URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-repo execution timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    ok = run_smoke_test(vllm_url=args.vllm_url, timeout=args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
