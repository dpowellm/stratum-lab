"""Static execution viability analysis.

Examines structural scan metadata to classify repos into:
  - LIKELY_RUNNABLE: has clear entry point, known deps, framework detected
  - NEEDS_PROBE: might run but has risk factors (unknown deps, no Dockerfile)
  - LIKELY_BROKEN: missing critical files, stale, known-bad patterns
  - SKIP: tutorial/duplicate/toy, zero structural value
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


# Deps that break containerized execution
BLOCKING_DEPENDENCIES: Set[str] = {
    "mysql-connector-python", "psycopg2", "pymongo", "redis",
    "confluent-kafka", "pika", "celery",  # need running services
    "torch", "tensorflow",  # huge, slow install, unrelated to agent behavior
    "chromadb", "pinecone-client", "weaviate-client",  # vector DBs
    "streamlit", "gradio", "dash",  # UI frameworks - need browser
}

# Deps we know how to handle
MANAGEABLE_DEPENDENCIES: Set[str] = {
    "openai", "anthropic", "langchain", "langchain-core",
    "langchain-openai", "langchain-anthropic", "langchain-community",
    "crewai", "crewai-tools", "pyautogen", "autogen-agentchat",
    "langgraph", "langsmith",
    "requests", "httpx", "aiohttp",  # HTTP - patcher handles
    "python-dotenv", "pydantic", "pydantic-settings",
    "tiktoken", "tenacity", "rich", "click",
}

# Env vars that signal external service dependencies we can't mock
BLOCKING_ENV_VARS: Set[str] = {
    "DATABASE_URL", "MONGODB_URI", "REDIS_URL", "REDIS_HOST",
    "KAFKA_BOOTSTRAP_SERVERS", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "AZURE_STORAGE_CONNECTION_STRING",
    "PINECONE_API_KEY", "PINECONE_ENVIRONMENT",
    "WEAVIATE_URL", "CHROMA_HOST",
    "SUPABASE_URL", "SUPABASE_KEY",
}

# Env vars the patcher handles by redirecting to vLLM
HANDLED_ENV_VARS: Set[str] = {
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY",  # anthropic_patch translates to OpenAI
}


def analyze_static_viability(repo_scan: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a structural scan result for execution viability.

    Returns
    -------
    Dict with viability, viability_score, blockers, risks, signals,
    entry_point_candidates, api_dependencies, install_complexity.
    """
    blockers: List[str] = []
    risks: List[str] = []
    signals: List[str] = []

    # --- Entry point analysis ---
    files = set(repo_scan.get("file_list", []))
    entry_candidates = _rank_entry_points(repo_scan, files)

    if not entry_candidates:
        blockers.append("no_entry_point: no main.py, app.py, CLI, or test files detected")
    elif entry_candidates[0]["confidence"] >= 0.8:
        signals.append(
            f"strong_entry_point: {entry_candidates[0]['path']} "
            f"({entry_candidates[0]['strategy']})"
        )
    else:
        signals.append(
            f"weak_entry_point: best candidate {entry_candidates[0]['path']} "
            f"(confidence={entry_candidates[0]['confidence']:.1f})"
        )

    # --- Dependency analysis ---
    deps = set(repo_scan.get("dependencies", []))
    blocking_deps = deps & BLOCKING_DEPENDENCIES
    manageable_deps = deps & MANAGEABLE_DEPENDENCIES
    unknown_deps = deps - BLOCKING_DEPENDENCIES - MANAGEABLE_DEPENDENCIES

    if blocking_deps:
        core_blocking = blocking_deps - _detect_optional_deps(repo_scan)
        if core_blocking:
            blockers.append(f"blocking_deps: {', '.join(sorted(core_blocking))}")
        else:
            risks.append(f"optional_blocking_deps: {', '.join(sorted(blocking_deps))}")

    if len(unknown_deps) > 10:
        risks.append(f"many_unknown_deps: {len(unknown_deps)} unrecognized packages")

    if manageable_deps:
        signals.append(f"known_deps: {len(manageable_deps)} recognized packages")

    has_requirements = any(
        f.endswith("requirements.txt") or f.endswith("pyproject.toml")
        for f in files
    )
    if has_requirements:
        signals.append("has_dependency_file")
    else:
        risks.append("no_dependency_file: no requirements.txt or pyproject.toml")

    # --- Environment variable analysis ---
    env_refs = set(repo_scan.get("env_var_references", []))
    blocking_envs = env_refs & BLOCKING_ENV_VARS
    handled_envs = env_refs & HANDLED_ENV_VARS

    if blocking_envs:
        blockers.append(f"needs_external_services: {', '.join(sorted(blocking_envs))}")

    if handled_envs:
        signals.append(f"api_keys_handled: {', '.join(sorted(handled_envs))}")

    # --- Containerization signals ---
    has_dockerfile = any("dockerfile" in f.lower() for f in files)
    has_docker_compose = any("docker-compose" in f.lower() for f in files)
    has_ci = any(".github/workflows" in f or ".gitlab-ci" in f for f in files)
    has_tests = any(
        f.startswith("test") or "/test" in f or f.startswith("tests/")
        for f in files
    )

    if has_dockerfile:
        signals.append("has_dockerfile")
    if has_ci:
        signals.append("has_ci_pipeline")
    if has_tests:
        signals.append("has_tests")
    if has_docker_compose and not has_dockerfile:
        risks.append("docker_compose_only: may need multi-service setup")

    # --- Freshness and quality ---
    python_version = repo_scan.get("python_version_constraint", "")
    if python_version and "2." in python_version:
        blockers.append("python2: requires Python 2.x")

    last_commit = repo_scan.get("last_commit_date", "")
    if last_commit and last_commit < "2024-01-01":
        risks.append(f"stale: last commit {last_commit[:10]}")

    # --- Framework detection ---
    framework = repo_scan.get("primary_framework", "")
    if framework in ("crewai", "autogen", "langgraph", "langchain"):
        signals.append(f"known_framework: {framework}")
    elif framework == "custom":
        risks.append("custom_framework: no standard patcher, may miss events")
    elif not framework:
        blockers.append("no_framework_detected: structural scan found no AI agent framework")

    # --- Structural value filter ---
    agent_count = repo_scan.get("agent_count", 0)
    edge_count = repo_scan.get("edge_count", 0)

    if agent_count <= 1 and edge_count <= 2:
        risks.append("trivial_topology: single agent, minimal edges — low signal")
    if agent_count >= 3:
        signals.append(f"multi_agent: {agent_count} agents")

    # --- Score computation ---
    viability_score = 50  # Base
    viability_score += len(signals) * 8
    viability_score -= len(risks) * 5
    viability_score -= len(blockers) * 25
    viability_score = max(0, min(100, viability_score))

    # --- Install complexity ---
    if not deps or (deps <= MANAGEABLE_DEPENDENCIES):
        install_complexity = "trivial"
    elif blocking_deps:
        install_complexity = "impossible" if len(blocking_deps) > 2 else "complex"
    elif len(unknown_deps) > 10:
        install_complexity = "complex"
    else:
        install_complexity = "moderate"

    # --- Classify ---
    if blockers:
        viability = "likely_broken" if len(blockers) >= 2 else "needs_probe"
    elif viability_score >= 70:
        viability = "likely_runnable"
    elif viability_score >= 40:
        viability = "needs_probe"
    else:
        viability = "skip"

    return {
        "viability": viability,
        "viability_score": viability_score,
        "blockers": blockers,
        "risks": risks,
        "signals": signals,
        "entry_point_candidates": entry_candidates,
        "api_dependencies": sorted(handled_envs | blocking_envs),
        "install_complexity": install_complexity,
        "blocking_dep_count": len(blocking_deps),
        "unknown_dep_count": len(unknown_deps),
    }


def _rank_entry_points(
    repo_scan: Dict[str, Any],
    files: set,
) -> List[Dict[str, Any]]:
    """Rank possible entry points by execution likelihood.

    Priority order:
    1. Explicit CLI entry point in pyproject.toml (highest confidence)
    2. main.py / app.py at repo root
    3. __main__.py in the primary package
    4. Example scripts in examples/ or scripts/
    5. README-documented run commands
    6. Test files (lowest priority — but always available)
    """
    candidates: List[Dict[str, Any]] = []

    # Check pyproject.toml for scripts/entry_points
    entry_points = repo_scan.get("cli_entry_points", [])
    for ep in entry_points:
        candidates.append({
            "path": ep,
            "strategy": "cli_entry_point",
            "confidence": 0.9,
            "command": f"python -m {ep.split(':')[0].replace('.', '/')}",
        })

    # Check common entry point files
    for name, confidence in [
        ("main.py", 0.85), ("app.py", 0.8), ("run.py", 0.75), ("cli.py", 0.7),
    ]:
        if name in files or f"./{name}" in files:
            candidates.append({
                "path": name,
                "strategy": "standard_file",
                "confidence": confidence,
                "command": f"python {name}",
            })

    # Check for __main__.py
    primary_pkg = repo_scan.get("primary_package", "")
    if primary_pkg:
        main_path = f"{primary_pkg}/__main__.py"
        if main_path in files or f"./{main_path}" in files:
            candidates.append({
                "path": main_path,
                "strategy": "package_main",
                "confidence": 0.85,
                "command": f"python -m {primary_pkg}",
            })

    # Check example scripts
    for f in sorted(files):
        if ("example" in f.lower() or "demo" in f.lower()) and f.endswith(".py"):
            candidates.append({
                "path": f,
                "strategy": "example_script",
                "confidence": 0.6,
                "command": f"python {f}",
            })

    # Check test files (always available as fallback)
    test_files = sorted(
        f for f in files if ("test" in f.lower()) and f.endswith(".py")
    )
    if test_files:
        candidates.append({
            "path": test_files[0],
            "strategy": "test_execution",
            "confidence": 0.5,
            "command": f"pytest {test_files[0]} -x --timeout=120",
        })

    # Sort by confidence descending
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def _detect_optional_deps(repo_scan: Dict[str, Any]) -> set:
    """Detect dependencies that appear in optional/extras sections."""
    optional: set = set()
    extras = repo_scan.get("extras_require", {})
    for group_deps in extras.values():
        optional.update(group_deps)
    return optional


def triage_batch(repo_scans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run static analysis on all repos and return summary.

    Also performs cross-repo deduplication by topology_hash.
    """
    results: List[Dict[str, Any]] = []
    seen_topologies: Dict[str, str] = {}

    for scan in repo_scans:
        analysis = analyze_static_viability(scan)

        # Dedup: if we've seen this topology, mark as lower priority
        topo_hash = scan.get("topology_hash", "")
        if topo_hash and topo_hash in seen_topologies:
            analysis["duplicate_of"] = seen_topologies[topo_hash]
            if analysis["viability"] == "likely_runnable":
                analysis["viability"] = "needs_probe"
                analysis["risks"].append(
                    f"duplicate_topology: same structure as {seen_topologies[topo_hash]}"
                )
        elif topo_hash:
            seen_topologies[topo_hash] = scan.get("repo_id", "")

        analysis["repo_id"] = scan.get("repo_id", "")
        results.append(analysis)

    # Summary
    by_viability: Dict[str, int] = {}
    for r in results:
        v = r["viability"]
        by_viability[v] = by_viability.get(v, 0) + 1

    return {
        "total_analyzed": len(results),
        "by_viability": by_viability,
        "likely_runnable": [r for r in results if r["viability"] == "likely_runnable"],
        "needs_probe": [r for r in results if r["viability"] == "needs_probe"],
        "likely_broken": [r for r in results if r["viability"] == "likely_broken"],
        "skipped": [r for r in results if r["viability"] == "skip"],
        "unique_topologies": len(seen_topologies),
        "duplicate_count": sum(1 for r in results if "duplicate_of" in r),
        "results": results,
    }
