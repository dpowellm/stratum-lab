#!/usr/bin/env python3
"""Tier Improvement Evaluation — run 4 test repos through the full
three-tier Docker pipeline and produce structured evaluation outputs.

Usage:
    python3 scripts/evaluate_tiers.py \
        --vllm-url https://aishvbx8prhm6k-8000.proxy.runpod.net \
        --output-dir /app/output/evaluation

Outputs:
    evaluation_summary.json     — structured per-repo results + totals
    evaluation_events_detail.jsonl — all events with repo name prepended
    evaluation_report.md        — human-readable analysis
    evaluation_payloads.json    — first instance of each event type per repo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Test repos ────────────────────────────────────────────────────────────

TEST_REPOS = [
    {
        "url": "https://github.com/binbakhsh/QBO-CrewAI",
        "name": "QBO-CrewAI",
        "expected_framework": "crewai",
    },
    {
        "url": "https://github.com/DevJadhav/agentic-doc-extraction-system",
        "name": "agentic-doc-extraction",
        "expected_framework": "langgraph",
    },
    {
        "url": "https://github.com/itay601/langGraph",
        "name": "langGraph",
        "expected_framework": "langchain",
    },
    {
        "url": "https://github.com/Sonlux/ESCAI",
        "name": "ESCAI",
        "expected_framework": "autogen",
    },
]

DEFAULT_IMAGE = "stratum-lab-base"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TIMEOUT = 600


# ── Helpers ───────────────────────────────────────────────────────────────

def find_script(name: str, search_dirs: list[str]) -> str | None:
    """Find a script file by searching multiple directories."""
    for d in search_dirs:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def read_file_safe(path: str) -> str:
    """Read a file, return empty string on error."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def parse_events(events_file: str) -> list[dict]:
    """Parse a JSONL events file into a list of dicts."""
    events = []
    try:
        with open(events_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return events


def analyze_events(events: list[dict]) -> dict:
    """Analyze events and return structured metrics."""
    event_types: Counter = Counter()
    source_nodes: set[str] = set()
    unique_agents: set[str] = set()
    has_system_prompt = False
    has_user_message = False
    has_output = False
    has_tokens = False
    has_edge = False
    has_delegation = False
    payloads_by_type: dict[str, dict] = {}

    for evt in events:
        et = evt.get("event_type", "unknown")
        event_types[et] += 1

        # Capture first payload of each type
        if et not in payloads_by_type:
            payloads_by_type[et] = {
                "payload": evt.get("payload", {}),
                "source_node": evt.get("source_node", {}),
                "target_node": evt.get("target_node"),
            }

        # Source node tracking
        src = evt.get("source_node", {})
        if src.get("node_id"):
            source_nodes.add(src["node_id"])
        if src.get("node_name"):
            name = src["node_name"]
            # Skip generic names
            if name not in ("stratum_runner", "unhandled_exception", ""):
                unique_agents.add(name)

        # I/O capture checks (field names match actual patcher output)
        payload = evt.get("payload", {})
        if payload.get("system_prompt_preview"):
            has_system_prompt = True
        if payload.get("last_user_message_preview"):
            has_user_message = True
        if payload.get("output_preview") or payload.get("output"):
            has_output = True
        if payload.get("input_tokens") or payload.get("output_tokens"):
            has_tokens = True

        # Topology events
        if et in ("edge.traversed", "routing.decision"):
            has_edge = True
        if et in ("delegation.initiated", "delegation.completed"):
            has_delegation = True

    io_complete_count = sum([has_system_prompt, has_user_message, has_output, has_tokens])

    return {
        "event_types": dict(event_types),
        "total_events": len(events),
        "unique_source_nodes": len(source_nodes),
        "unique_agents": sorted(unique_agents),
        "io_capture": {
            "system_prompt": has_system_prompt,
            "user_message": has_user_message,
            "output_preview": has_output,
            "tokens": has_tokens,
        },
        "io_completeness": io_complete_count / 4.0,
        "has_edge_events": has_edge,
        "has_delegation_events": has_delegation,
        "payloads_by_type": payloads_by_type,
    }


def analyze_container_log(log_path: str) -> dict:
    """Extract tier-level information from container log."""
    tiers_attempted: dict[str, dict] = {}
    entry_point = ""
    entry_point_score = 0
    framework_detected = "unknown"
    pip_failures: list[str] = []

    log_content = read_file_safe(log_path)
    if not log_content:
        return {
            "tiers_attempted": tiers_attempted,
            "entry_point": entry_point,
            "entry_point_score": entry_point_score,
            "framework_detected": framework_detected,
            "pip_failures": pip_failures,
        }

    # Parse tier results from log
    for line in log_content.split("\n"):
        # Entry point
        if "Entry point:" in line or "Winner:" in line:
            if "(score=" in line:
                try:
                    score_str = line.split("(score=")[1].split(")")[0]
                    entry_point_score = int(score_str)
                except (IndexError, ValueError):
                    pass
            if "Winner:" in line:
                parts = line.split("Winner:")[1].strip().split()
                if parts:
                    entry_point = parts[0]
            elif "Entry point:" in line:
                parts = line.split("Entry point:")[1].strip().split()
                if parts:
                    entry_point = parts[0]

        # Tier 1 result
        if "Tier 1 result:" in line:
            parts = line.split("Tier 1 result:")[1].strip()
            status = parts.split("(")[0].strip()
            exit_code = 0
            if "exit_code=" in parts:
                try:
                    exit_code = int(parts.split("exit_code=")[1].split(")")[0])
                except (IndexError, ValueError):
                    pass
            tiers_attempted["tier1"] = {"status": status, "exit_code": exit_code}

        # Tier 1.5
        if "Tier 1.5 succeeded" in line:
            tiers_attempted["tier1_5"] = {"status": "SUCCESS", "exit_code": 0}
        elif "Tier 1.5 failed" in line:
            exit_code = 0
            if "exit=" in line:
                try:
                    exit_code = int(line.split("exit=")[1].split(")")[0])
                except (IndexError, ValueError):
                    pass
            tiers_attempted["tier1_5"] = {"status": "FAILED", "exit_code": exit_code}

        # Tier 2
        if "Tier 2 succeeded" in line:
            tiers_attempted["tier2"] = {"status": "SUCCESS", "exit_code": 0}
        elif "Tier 2 also failed" in line:
            exit_code = 0
            if "exit=" in line:
                try:
                    exit_code = int(line.split("exit=")[1].split(")")[0])
                except (IndexError, ValueError):
                    pass
            tiers_attempted["tier2"] = {"status": "FAILED", "exit_code": exit_code}

        # Framework
        if "Framework:" in line:
            fw = line.split("Framework:")[1].strip()
            if fw:
                framework_detected = fw

        # Pip failures
        if "Failed to install" in line:
            pkg = line.split("Failed to install")[1].strip().split()[0] if "Failed to install" in line else ""
            if pkg:
                pip_failures.append(pkg)

    return {
        "tiers_attempted": tiers_attempted,
        "entry_point": entry_point,
        "entry_point_score": entry_point_score,
        "framework_detected": framework_detected,
        "pip_failures": pip_failures,
    }


# ── Docker execution ─────────────────────────────────────────────────────

def run_repo_in_docker(
    repo: dict,
    output_dir: str,
    vllm_url: str,
    model: str,
    image: str,
    timeout: int,
    script_dir: str,
) -> dict:
    """Run a single repo through the Docker pipeline and return results."""
    repo_name = repo["name"]
    repo_url = repo["url"]
    repo_output = os.path.join(output_dir, repo_name)
    os.makedirs(repo_output, exist_ok=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Testing: {repo_name} ({repo['expected_framework']})", file=sys.stderr)
    print(f"  URL: {repo_url}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Locate scripts
    run_repo_sh = find_script("run_repo.sh", [script_dir, os.path.join(script_dir, "..")])
    synthetic_py = find_script("synthetic_harness.py", [
        os.path.join(script_dir, "scripts"),
        script_dir,
        os.path.join(script_dir, ".."),
    ])
    inject_main_py = find_script("inject_main.py", [
        os.path.join(script_dir, "scripts"),
        script_dir,
        os.path.join(script_dir, ".."),
    ])
    scan_defensive_py = find_script("scan_defensive_patterns.py", [
        os.path.join(script_dir, "scripts"),
        script_dir,
    ])

    if not run_repo_sh:
        print(f"  ERROR: run_repo.sh not found", file=sys.stderr)
        return {"error": "run_repo.sh not found"}

    # Build docker command
    run_id = f"eval_{uuid.uuid4().hex[:12]}"

    # Convert Windows paths to Docker-compatible format
    def to_docker_path(p: str) -> str:
        """Convert Windows path to Docker bind mount format."""
        if not p:
            return p
        # Convert C:\... to /c/... for Docker on Windows
        p = p.replace("\\", "/")
        if len(p) > 1 and p[1] == ":":
            p = "/" + p[0].lower() + p[2:]
        return p

    mount_args = [
        "-v", f"{to_docker_path(run_repo_sh)}:/app/run_repo.sh:ro",
        "-v", f"{to_docker_path(repo_output)}:/app/output",
        "-v", "/tmp/pip-cache:/root/.cache/pip",
    ]
    if synthetic_py:
        mount_args += ["-v", f"{to_docker_path(synthetic_py)}:/app/synthetic_harness.py:ro"]
    if inject_main_py:
        mount_args += ["-v", f"{to_docker_path(inject_main_py)}:/app/inject_main.py:ro"]
    if scan_defensive_py:
        mount_args += ["-v", f"{to_docker_path(scan_defensive_py)}:/app/scripts/scan_defensive_patterns.py:ro"]

    # Strip protocol for VLLM_HOST (run_repo.sh expects base URL without /v1)
    vllm_host = vllm_url.rstrip("/")
    if vllm_host.endswith("/v1"):
        vllm_host = vllm_host[:-3]

    env_args = [
        "-e", f"STRATUM_EVENTS_FILE=/app/output/events_run_1.jsonl",
        "-e", "STRATUM_RUN_NUMBER=1",
        "-e", f"STRATUM_VLLM_MODEL={model}",
        "-e", f"STRATUM_RUN_ID={run_id}",
        "-e", f"STRATUM_REPO_ID={repo_url}",
        "-e", "STRATUM_FRAMEWORK=auto",
        "-e", "STRATUM_CAPTURE_PROMPTS=1",
        "-e", f"VLLM_HOST={vllm_host}",
        "-e", "VLLM_TIMEOUT=60",
        "-e", "RUN_NUMBER=1",
    ]

    cmd = [
        "docker", "run", "--rm",
        "--network=host",
        "--entrypoint", "bash",
        *mount_args,
        *env_args,
        image,
        "/app/run_repo.sh", repo_url, vllm_host, "/app/output", str(timeout),
    ]

    print(f"  Running Docker container (timeout={timeout}s)...", file=sys.stderr)
    print(f"  Command: docker run --rm --network=host --entrypoint bash ... {image} /app/run_repo.sh {repo_url} ...", file=sys.stderr)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # extra 30s buffer for Docker overhead
        )
        duration = time.time() - t0
        container_stdout = result.stdout
        container_stderr = result.stderr
        docker_exit = result.returncode
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        container_stdout = ""
        container_stderr = "Docker timeout exceeded"
        docker_exit = 124
    except FileNotFoundError:
        duration = time.time() - t0
        container_stdout = ""
        container_stderr = "Docker not found in PATH"
        docker_exit = -1

    # Write container log
    container_log_path = os.path.join(repo_output, "container.log")
    try:
        with open(container_log_path, "w", encoding="utf-8") as f:
            f.write(container_stdout)
            if container_stderr:
                f.write("\n--- STDERR ---\n")
                f.write(container_stderr)
    except OSError:
        pass

    print(f"  Docker exit code: {docker_exit} ({duration:.1f}s)", file=sys.stderr)

    # Read outputs
    status = read_file_safe(os.path.join(repo_output, "status.txt")) or "UNKNOWN"
    # Prefer tier_detail.txt (has "1.5") over tier.txt (integer only)
    tier = read_file_safe(os.path.join(repo_output, "tier_detail.txt")) or \
        read_file_safe(os.path.join(repo_output, "tier.txt")) or "0"
    exit_code = read_file_safe(os.path.join(repo_output, "exit_code.txt")) or str(docker_exit)

    # Parse tier as float to handle "1.5"
    try:
        tier_num = float(tier)
    except ValueError:
        tier_num = 0

    # Find events file
    events_file = os.path.join(repo_output, "events_run_1.jsonl")
    if not os.path.isfile(events_file) or os.path.getsize(events_file) == 0:
        # Try alternate location
        alt = os.path.join(repo_output, "stratum_events.jsonl")
        if os.path.isfile(alt) and os.path.getsize(alt) > 0:
            events_file = alt

    events = parse_events(events_file)
    analysis = analyze_events(events)
    log_analysis = analyze_container_log(container_log_path)

    print(f"  Status: {status}", file=sys.stderr)
    print(f"  Tier: {tier}", file=sys.stderr)
    print(f"  Events: {len(events)}", file=sys.stderr)
    if analysis["unique_agents"]:
        print(f"  Agents: {analysis['unique_agents']}", file=sys.stderr)

    return {
        "repo_url": repo_url,
        "repo_name": repo_name,
        "expected_framework": repo["expected_framework"],
        "tier_succeeded": tier_num,
        "status": status,
        "exit_code": int(exit_code) if exit_code.lstrip("-").isdigit() else -1,
        "docker_exit_code": docker_exit,
        "duration_s": round(duration, 1),
        "tiers_attempted": log_analysis["tiers_attempted"],
        "behavioral_events": len(events),
        "event_types": analysis["event_types"],
        "io_capture": analysis["io_capture"],
        "io_completeness": analysis["io_completeness"],
        "unique_agents": analysis["unique_agents"],
        "unique_source_nodes": analysis["unique_source_nodes"],
        "has_edge_events": analysis["has_edge_events"],
        "has_delegation_events": analysis["has_delegation_events"],
        "entry_point": log_analysis["entry_point"],
        "entry_point_score": log_analysis["entry_point_score"],
        "framework_detected": log_analysis["framework_detected"],
        "pip_failures": log_analysis["pip_failures"],
        "events": events,
        "payloads_by_type": analysis["payloads_by_type"],
        "output_dir": repo_output,
    }


# ── Output generators ────────────────────────────────────────────────────

def write_evaluation_summary(results: list[dict], output_dir: str) -> None:
    """Write evaluation_summary.json."""
    repos_data: dict[str, Any] = {}
    tier1_success = 0
    tier1_5_success = 0
    tier2_success = 0
    total_events = 0
    io_scores: list[float] = []

    for r in results:
        tier = r["tier_succeeded"]
        if tier == 1:
            tier1_success += 1
        elif tier == 1.5:
            tier1_5_success += 1
        elif tier == 2:
            tier2_success += 1

        total_events += r["behavioral_events"]
        io_scores.append(r["io_completeness"])

        repos_data[r["repo_name"]] = {
            "tier_succeeded": tier,
            "tiers_attempted": r["tiers_attempted"],
            "behavioral_events": r["behavioral_events"],
            "event_types": r["event_types"],
            "io_capture": r["io_capture"],
            "unique_agents": r["unique_agents"],
            "unique_source_nodes": r["unique_source_nodes"],
            "has_edge_events": r["has_edge_events"],
            "has_delegation_events": r["has_delegation_events"],
            "entry_point": r["entry_point"],
            "entry_point_score": r["entry_point_score"],
            "framework_detected": r["framework_detected"],
            "pip_failures": r["pip_failures"],
            "status": r["status"],
            "duration_s": r["duration_s"],
        }

    total_success = tier1_success + tier1_5_success + tier2_success

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vllm_url": "",  # filled by caller
        "model": "",
        "repos": repos_data,
        "totals": {
            "repos_tested": len(results),
            "tier1_success": tier1_success,
            "tier1_5_success": tier1_5_success,
            "tier2_success": tier2_success,
            "total_success": total_success,
            "total_failed": len(results) - total_success,
            "total_behavioral_events": total_events,
            "avg_io_completeness": round(sum(io_scores) / len(io_scores), 3) if io_scores else 0,
        },
    }

    path = os.path.join(output_dir, "evaluation_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote {path}", file=sys.stderr)
    return summary


def write_events_detail(results: list[dict], output_dir: str) -> None:
    """Write evaluation_events_detail.jsonl."""
    path = os.path.join(output_dir, "evaluation_events_detail.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            tier = r["tier_succeeded"]
            for evt in r.get("events", []):
                record = {
                    "repo": r["repo_name"],
                    "tier": tier,
                    "event_type": evt.get("event_type", "unknown"),
                    "timestamp_ns": evt.get("timestamp_ns"),
                    "source_node": evt.get("source_node"),
                    "target_node": evt.get("target_node"),
                    "payload": evt.get("payload"),
                    "event_id": evt.get("event_id"),
                }
                f.write(json.dumps(record, default=str) + "\n")
    print(f"  Wrote {path}", file=sys.stderr)


def write_payloads(results: list[dict], output_dir: str) -> None:
    """Write evaluation_payloads.json — first instance of each event type per repo."""
    payloads: dict[str, dict] = {}
    for r in results:
        payloads[r["repo_name"]] = r.get("payloads_by_type", {})

    path = os.path.join(output_dir, "evaluation_payloads.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, default=str)
    print(f"  Wrote {path}", file=sys.stderr)


def write_report(results: list[dict], summary: dict, output_dir: str) -> None:
    """Write evaluation_report.md — human-readable analysis."""
    lines: list[str] = []
    totals = summary.get("totals", {}) if isinstance(summary, dict) else {}

    lines.append("# Stratum Lab — Tier Improvement Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Repos tested:** {len(results)}")
    lines.append("")

    # Overall yield summary
    lines.append("## Overall Yield Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Tier 1 success | {totals.get('tier1_success', 0)} |")
    lines.append(f"| Tier 1.5 success | {totals.get('tier1_5_success', 0)} |")
    lines.append(f"| Tier 2 success | {totals.get('tier2_success', 0)} |")
    lines.append(f"| Total success | {totals.get('total_success', 0)}/{totals.get('repos_tested', 0)} |")
    lines.append(f"| Total failed | {totals.get('total_failed', 0)} |")
    lines.append(f"| Total behavioral events | {totals.get('total_behavioral_events', 0)} |")
    lines.append(f"| Avg I/O completeness | {totals.get('avg_io_completeness', 0):.1%} |")
    lines.append("")

    # Per-repo breakdown
    lines.append("## Per-Repo Breakdown")
    lines.append("")

    for r in results:
        lines.append(f"### {r['repo_name']} ({r['expected_framework']})")
        lines.append("")
        lines.append(f"- **URL:** {r['repo_url']}")
        lines.append(f"- **Tier succeeded:** {r['tier_succeeded']}")
        lines.append(f"- **Status:** {r['status']}")
        lines.append(f"- **Duration:** {r['duration_s']}s")
        lines.append(f"- **Framework detected:** {r['framework_detected']}")
        lines.append(f"- **Entry point:** `{r['entry_point']}` (score={r['entry_point_score']})")
        lines.append(f"- **Behavioral events:** {r['behavioral_events']}")
        lines.append("")

        # Tier attempts
        tiers = r.get("tiers_attempted", {})
        if tiers:
            lines.append("**Tier attempts:**")
            lines.append("")
            for tier_name, tier_data in sorted(tiers.items()):
                status = tier_data.get("status", "?")
                exit_c = tier_data.get("exit_code", "?")
                lines.append(f"- {tier_name}: {status} (exit={exit_c})")
            lines.append("")

        # Why earlier tiers failed
        if r["tier_succeeded"] > 1:
            lines.append("**Why earlier tiers failed:**")
            lines.append("")
            if "tier1" in tiers:
                t1 = tiers["tier1"]
                lines.append(f"- Tier 1: {t1.get('status', 'unknown')}")
            if r["tier_succeeded"] > 1.5 and "tier1_5" in tiers:
                t15 = tiers["tier1_5"]
                lines.append(f"- Tier 1.5: {t15.get('status', 'unknown')}")
            lines.append("")

        # Event types
        if r["event_types"]:
            lines.append("**Event types:**")
            lines.append("")
            for et, count in sorted(r["event_types"].items()):
                lines.append(f"- `{et}`: {count}")
            lines.append("")

        # Agents
        if r["unique_agents"]:
            lines.append(f"**Unique agents ({len(r['unique_agents'])}):** {', '.join(r['unique_agents'])}")
            lines.append("")

        # Pip failures
        if r["pip_failures"]:
            lines.append(f"**Pip failures:** {', '.join(r['pip_failures'])}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # I/O capture completeness table
    lines.append("## I/O Capture Completeness")
    lines.append("")
    lines.append("| Repo | system_prompt | user_message | output | tokens | Score |")
    lines.append("|------|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        io = r.get("io_capture", {})
        sp = "Y" if io.get("system_prompt") else "-"
        um = "Y" if io.get("user_message") else "-"
        op = "Y" if io.get("output_preview") else "-"
        tk = "Y" if io.get("tokens") else "-"
        score = f"{r.get('io_completeness', 0):.0%}"
        lines.append(f"| {r['repo_name']} | {sp} | {um} | {op} | {tk} | {score} |")
    lines.append("")

    # Agent topology summary
    lines.append("## Agent Topology Summary")
    lines.append("")
    lines.append("| Repo | Unique Agents | Source Nodes | Edge Events | Delegation Events |")
    lines.append("|------|:---:|:---:|:---:|:---:|")
    for r in results:
        agents = len(r.get("unique_agents", []))
        nodes = r.get("unique_source_nodes", 0)
        edges = "Y" if r.get("has_edge_events") else "-"
        deleg = "Y" if r.get("has_delegation_events") else "-"
        lines.append(f"| {r['repo_name']} | {agents} | {nodes} | {edges} | {deleg} |")
    lines.append("")

    # Entry point detection analysis
    lines.append("## Entry Point Detection Analysis")
    lines.append("")
    lines.append("| Repo | Entry Point | Score | Framework |")
    lines.append("|------|------------|:---:|-----------|")
    for r in results:
        ep = f"`{r['entry_point']}`" if r["entry_point"] else "none"
        lines.append(f"| {r['repo_name']} | {ep} | {r['entry_point_score']} | {r['framework_detected']} |")
    lines.append("")

    # Comparison to baseline
    lines.append("## Comparison: Tier 1.5 vs Tier 2 Only")
    lines.append("")
    lines.append("Tier 1.5 preserves the repo's **real orchestration topology** — conditional routing,")
    lines.append("delegation chains, and multi-agent dynamics. Tier 2 generates synthetic scripts that")
    lines.append("exercise extracted agent definitions but lose the real graph structure.")
    lines.append("")

    tier15_repos = [r for r in results if r["tier_succeeded"] == 1.5]
    if tier15_repos:
        lines.append(f"**{len(tier15_repos)} repo(s) succeeded via Tier 1.5:**")
        for r in tier15_repos:
            lines.append(f"- {r['repo_name']}: {r['behavioral_events']} events, "
                        f"{len(r['unique_agents'])} agents, "
                        f"{'has' if r['has_edge_events'] else 'no'} edge events")
        lines.append("")
        lines.append("These would have fallen through to Tier 2 without the import-and-call mechanism,")
        lines.append("losing real topology signal.")
    else:
        lines.append("No repos were captured via Tier 1.5 in this evaluation run.")
    lines.append("")

    # Known issues
    lines.append("## Known Issues and Recommendations")
    lines.append("")

    failed_repos = [r for r in results if r["tier_succeeded"] == 0]
    if failed_repos:
        lines.append("### Failed Repos")
        for r in failed_repos:
            lines.append(f"- **{r['repo_name']}**: All tiers failed. "
                        f"Status: {r['status']}. Check container logs.")
        lines.append("")

    all_pip_failures = []
    for r in results:
        all_pip_failures.extend(r.get("pip_failures", []))
    if all_pip_failures:
        lines.append("### Pip Failures")
        lines.append(f"Packages that failed to install: {', '.join(set(all_pip_failures))}")
        lines.append("")

    lines.append("### Recommendations")
    lines.append("")
    lines.append("1. Monitor Tier 1.5 success rate across a larger sample (target: 25%+ of repos)")
    lines.append("2. Investigate repos where Tier 1 produces NO_EVENTS — inject_main.py may need")
    lines.append("   additional framework patterns")
    lines.append("3. Track I/O completeness — system_prompt and tokens capture often requires")
    lines.append("   framework-specific patcher improvements")
    lines.append("")

    path = os.path.join(output_dir, "evaluation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the three-tier execution pipeline with test repos"
    )
    parser.add_argument(
        "--vllm-url",
        required=True,
        help="vLLM server URL (e.g., https://aishvbx8prhm6k-8000.proxy.runpod.net)",
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_output",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image name (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"vLLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-repo timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Copy outputs to this directory after evaluation",
    )

    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find script directory (relative to this file)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)  # stratum-lab/

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Stratum Lab — Tier Improvement Evaluation", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  vLLM URL:   {args.vllm_url}", file=sys.stderr)
    print(f"  Model:      {args.model}", file=sys.stderr)
    print(f"  Image:      {args.image}", file=sys.stderr)
    print(f"  Timeout:    {args.timeout}s per repo", file=sys.stderr)
    print(f"  Output:     {output_dir}", file=sys.stderr)
    print(f"  Script dir: {repo_root}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Check Docker
    try:
        docker_check = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        if docker_check.returncode != 0:
            print("ERROR: Docker is not running or not accessible", file=sys.stderr)
            return 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: Docker not found in PATH. Install Docker and try again.", file=sys.stderr)
        return 1

    # Check image exists
    img_check = subprocess.run(
        ["docker", "images", "-q", args.image],
        capture_output=True, text=True, timeout=10,
    )
    if not img_check.stdout.strip():
        print(f"WARNING: Docker image '{args.image}' not found. Building...", file=sys.stderr)
        build_result = subprocess.run(
            ["docker", "build", "-t", args.image, repo_root],
            capture_output=True, text=True, timeout=300,
        )
        if build_result.returncode != 0:
            print(f"ERROR: Docker build failed:\n{build_result.stderr[:1000]}", file=sys.stderr)
            return 1
        print(f"  Docker image '{args.image}' built successfully", file=sys.stderr)

    # Run each repo
    results: list[dict] = []
    for repo in TEST_REPOS:
        result = run_repo_in_docker(
            repo=repo,
            output_dir=output_dir,
            vllm_url=args.vllm_url,
            model=args.model,
            image=args.image,
            timeout=args.timeout,
            script_dir=repo_root,
        )
        results.append(result)

    # Generate outputs
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Generating evaluation outputs...", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    summary = write_evaluation_summary(results, output_dir)
    # Patch in vllm/model info
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    try:
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        s["vllm_url"] = args.vllm_url
        s["model"] = args.model
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, default=str)
    except Exception:
        pass

    write_events_detail(results, output_dir)
    write_payloads(results, output_dir)
    write_report(results, summary, output_dir)

    # Copy to secondary location if requested
    if args.copy_to:
        copy_dir = os.path.abspath(args.copy_to)
        os.makedirs(copy_dir, exist_ok=True)
        for fname in [
            "evaluation_summary.json",
            "evaluation_events_detail.jsonl",
            "evaluation_report.md",
            "evaluation_payloads.json",
        ]:
            src = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(copy_dir, fname))
        print(f"  Copied outputs to {copy_dir}", file=sys.stderr)

    # Print summary to stdout
    totals = summary.get("totals", {}) if isinstance(summary, dict) else {}

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Results: {totals.get('total_success', 0)}/{totals.get('repos_tested', 0)} repos succeeded")
    print(f"")
    print(f"  Tier 1 success:   {totals.get('tier1_success', 0)}")
    print(f"  Tier 1.5 success: {totals.get('tier1_5_success', 0)}")
    print(f"  Tier 2 success:   {totals.get('tier2_success', 0)}")
    print(f"  Total events:     {totals.get('total_behavioral_events', 0)}")
    print(f"  Avg I/O:          {totals.get('avg_io_completeness', 0):.1%}")
    print(f"")

    for r in results:
        status_icon = "+" if r["tier_succeeded"] > 0 else "X"
        print(f"  [{status_icon}] {r['repo_name']:30s} tier={r['tier_succeeded']:<5} "
              f"events={r['behavioral_events']:<6} status={r['status']}")

    print(f"\n  Output dir: {output_dir}")
    print(f"{'='*60}")

    return 0 if totals.get("total_success", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
