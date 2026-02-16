"""Main orchestrator for Phase 2 execution.

Reads the selection JSON, generates synthetic inputs, schedules runs across
a thread-pool of Docker containers, collects results, and writes metadata
CSVs for downstream pipeline phases.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency tracker for adaptive backpressure
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Track rolling average inference latency for adaptive backpressure."""

    def __init__(self, window_seconds: int = 60) -> None:
        self.window = window_seconds
        self.samples: list[tuple[float, float]] = []  # (timestamp, latency_ms)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record a latency sample."""
        with self._lock:
            self.samples.append((time.time(), latency_ms))
            self._prune()

    def avg_last_window(self) -> float:
        """Return average latency over the last window_seconds."""
        with self._lock:
            self._prune()
            if not self.samples:
                return 0.0
            return sum(s[1] for s in self.samples) / len(self.samples)

    def _prune(self) -> None:
        """Remove samples older than the window."""
        cutoff = time.time() - self.window
        self.samples = [(t, l) for t, l in self.samples if t > cutoff]

from stratum_lab.config import (
    DEFAULT_CONCURRENT_CONTAINERS,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_RUNS_PER_REPO,
    EXECUTION_META_DIR,
    ExecutionStatus,
    MIN_EVENT_THRESHOLD,
)
from stratum_lab.harness.container import RunResult, run_container
from stratum_lab.harness.input_generator import (
    generate_inputs,
    input_hash,
    plan_runs,
)

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Schedules and executes container runs for all selected repos.

    Parameters
    ----------
    selection_file:
        Path to the Phase 1 selection JSON.
    output_dir:
        Directory where raw event JSONL files are written.
    vllm_url:
        OpenAI-compatible vLLM endpoint.
    concurrent:
        Maximum number of containers running in parallel.
    timeout:
        Per-execution timeout in seconds.
    runs_per_repo:
        Total runs per repo (3 diverse + 2 repeat by default).
    max_concurrent:
        Maximum concurrent container executions (semaphore limit).
        Defaults to *concurrent* if not specified.
    """

    def __init__(
        self,
        selection_file: str | Path,
        output_dir: str | Path,
        vllm_url: str,
        concurrent: int = DEFAULT_CONCURRENT_CONTAINERS,
        timeout: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        runs_per_repo: int = DEFAULT_RUNS_PER_REPO,
        max_concurrent: int | None = None,
    ) -> None:
        self.selection_file = Path(selection_file)
        self.output_dir = Path(output_dir)
        self.vllm_url = vllm_url
        self.concurrent = concurrent
        self.max_concurrent = max_concurrent if max_concurrent is not None else concurrent
        self.timeout = timeout
        self.runs_per_repo = runs_per_repo

        # Concurrency control
        self._semaphore = threading.Semaphore(self.max_concurrent)
        self.active_count = 0
        self._active_lock = threading.Lock()
        self.latency_tracker = LatencyTracker()

        # Loaded at run-time
        self._selections: list[dict[str, Any]] = []
        self._results: list[RunResult] = []

        # Metadata tracking
        self._meta_dir = self.output_dir.parent / "execution_metadata"

        # Checkpoint for resume (v6.3 B2)
        self._checkpoint_file = self.output_dir.parent / "scan_checkpoint.json"
        self._completed_repos: set[str] = set()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> list[RunResult]:
        """Execute all selected repos and return collected results.

        Parameters
        ----------
        dry_run:
            If ``True``, print what would be executed without actually
            running any containers.

        Returns
        -------
        list[RunResult]
            All run results across every repo.
        """
        self._load_selections()
        self._ensure_directories()

        total_runs = len(self._selections) * self.runs_per_repo
        console.print(
            f"[bold]Execution plan:[/bold] {len(self._selections)} repos "
            f"x {self.runs_per_repo} runs = {total_runs} total runs"
        )
        console.print(
            f"[bold]Concurrency:[/bold] {self.concurrent} containers  |  "
            f"[bold]Timeout:[/bold] {self.timeout}s per run"
        )

        if dry_run:
            self._print_dry_run()
            return []

        all_results: list[RunResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]Executing repos", total=total_runs
            )

            with ThreadPoolExecutor(max_workers=self.concurrent) as executor:
                futures = {}

                for selection in self._selections:
                    repo_id = selection.get("repo_id", "unknown")
                    repo_url = selection.get("repo_url", "")
                    framework = selection.get("framework", "unknown")
                    entry_point = selection.get("detected_entry_point", "main.py")

                    # Generate inputs for this repo
                    inputs = self._generate_repo_inputs(selection)
                    run_schedule = plan_runs(inputs, total_runs=self.runs_per_repo)

                    for input_data, run_number in run_schedule:
                        run_id = f"{repo_id}_run_{run_number}_{uuid.uuid4().hex[:8]}"
                        future = executor.submit(
                            self.execute_repo,
                            selection,
                            run_number,
                            input_data,
                            run_id,
                        )
                        futures[future] = (repo_id, run_number)

                for future in as_completed(futures):
                    repo_id, run_number = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        status_color = (
                            "green"
                            if result.status == ExecutionStatus.SUCCESS
                            else "yellow"
                            if result.status == ExecutionStatus.PARTIAL_SUCCESS
                            else "red"
                        )
                        progress.console.print(
                            f"  [{status_color}]{result.status}[/{status_color}] "
                            f"{repo_id} run {run_number} "
                            f"({result.duration_ms}ms)"
                        )
                    except Exception as exc:
                        console.print(
                            f"  [red]EXCEPTION[/red] {repo_id} run {run_number}: {exc}"
                        )
                        all_results.append(
                            RunResult(
                                run_id=f"{repo_id}_run_{run_number}_err",
                                repo_id=repo_id,
                                status=ExecutionStatus.CRASH,
                                error_message=str(exc),
                            )
                        )
                    progress.advance(overall_task)

        self._results = all_results
        self._write_metadata_csvs(all_results)
        return all_results

    # ------------------------------------------------------------------
    # Single repo execution
    # ------------------------------------------------------------------

    def execute_repo(
        self,
        repo_selection: dict[str, Any],
        run_number: int,
        input_data: str,
        run_id: str | None = None,
        timeout_override: int | None = None,
    ) -> RunResult:
        """Execute a single run for a repo.

        1. Acquires semaphore to limit concurrency
        2. Applies adaptive backpressure if vLLM is overloaded
        3. Calls ``container.run_container()``
        4. Classifies the result status
        5. Saves the events file to output_dir
        6. Returns the ``RunResult``
        """
        # Adaptive backpressure: if average latency > 30s, sleep before proceeding
        avg_latency = self.latency_tracker.avg_last_window()
        if avg_latency > 30_000:  # 30s in ms
            logger.warning(
                f"vLLM overloaded (avg {avg_latency:.0f}ms). "
                f"Applying backpressure before next launch."
            )
            time.sleep(min(avg_latency / 1000.0, 30.0))

        self._semaphore.acquire()
        with self._active_lock:
            self.active_count += 1
        try:
            repo_id = repo_selection.get("repo_id", "unknown")
            repo_url = repo_selection.get("repo_url", "")
            framework = repo_selection.get("framework", "unknown")
            entry_point = repo_selection.get("detected_entry_point", "main.py")
            run_id = run_id or f"{repo_id}_run_{run_number}_{uuid.uuid4().hex[:8]}"
            timeout = timeout_override or self.timeout

            t0 = time.perf_counter()
            result = run_container(
                repo_url=repo_url,
                entry_point=entry_point,
                run_id=run_id,
                repo_id=repo_id,
                framework=framework,
                input_data=input_data,
                timeout=timeout,
                vllm_url=self.vllm_url,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.latency_tracker.record(latency_ms)

            # Refine status classification
            result.status = self.classify_status(result)

            # Copy events file to canonical output location
            if result.events_file_path:
                dest = self.output_dir / f"{repo_id}_run_{run_number}.jsonl"
                try:
                    src = Path(result.events_file_path)
                    if src.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(str(src), str(dest))
                        result.events_file_path = str(dest)
                except Exception:
                    pass

            return result
        finally:
            with self._active_lock:
                self.active_count -= 1
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Probe-before-commit (v6.3 B1)
    # ------------------------------------------------------------------

    def execute_repo_with_probe(
        self,
        repo_selection: dict[str, Any],
        planned_runs: list[dict] | None = None,
    ) -> list[RunResult]:
        """Run a quick probe first. Only proceed to full runs if probe produces events.

        Parameters
        ----------
        repo_selection:
            Repo selection dict from Phase 1.
        planned_runs:
            Optional pre-planned run schedule. If None, generates one internally.
        """
        repo_id = repo_selection.get("repo_id", "unknown")

        if planned_runs is None:
            inputs = self._generate_repo_inputs(repo_selection)
            run_schedule = plan_runs(inputs, total_runs=self.runs_per_repo)
        else:
            run_schedule = planned_runs

        if not run_schedule:
            return []

        # Probe with shorter timeout (60s or 1/5 of configured timeout, whichever is larger)
        probe_timeout = max(60, self.timeout // 5)
        probe_input, probe_run_number = run_schedule[0]
        probe_run_id = f"{repo_id}_probe_{uuid.uuid4().hex[:8]}"

        probe_result = self.execute_repo(
            repo_selection, probe_run_number, probe_input, probe_run_id,
            timeout_override=probe_timeout,
        )

        if probe_result.status in (
            ExecutionStatus.CRASH,
            ExecutionStatus.DEPENDENCY_FAILURE,
            ExecutionStatus.INSTRUMENTATION_FAILURE,
        ):
            logger.info(
                f"[PROBE FAIL] {repo_id}: {probe_result.status} "
                f"— skipping {len(run_schedule) - 1} remaining runs"
            )
            return [probe_result]

        # Probe passed — execute remaining runs with full timeout
        results: list[RunResult] = [probe_result]
        for input_data, run_number in run_schedule[1:]:
            run_id = f"{repo_id}_run_{run_number}_{uuid.uuid4().hex[:8]}"
            result = self.execute_repo(
                repo_selection, run_number, input_data, run_id,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Entry point resolution with fallback chain (Issue 21)
    # ------------------------------------------------------------------

    def resolve_entry_point(
        self,
        repo_selection: dict[str, Any],
        input_data: str,
        run_id: str,
    ) -> tuple[str, RunResult | None]:
        """Resolve the best entry point for a repo.

        Uses the entry point that succeeded in the probe (if available),
        then falls back to ranked entry candidates from static analysis.
        Returns (entry_point, first_run_result_or_None).
        """
        # Primary: probe-verified entry point
        entry_point = repo_selection.get("probe_entry_point") or repo_selection.get(
            "detected_entry_point", "main.py"
        )
        entry_candidates = repo_selection.get("entry_point_candidates", [])

        # If we don't have a fallback chain, just return the primary
        if not entry_candidates or len(entry_candidates) <= 1:
            return entry_point, None

        return entry_point, None

    # ------------------------------------------------------------------
    # Status classification
    # ------------------------------------------------------------------

    INTERACTION_EVENT_TYPES = frozenset({
        "delegation.initiated", "delegation.completed",
        "tool.invoked", "tool.completed",
        "data.read", "data.write",
    })

    def classify_status(self, run_result: RunResult) -> str:
        """Determine execution status from exit code, events, and errors.

        Refines the initial container-level status by inspecting the
        events file and stderr patterns.
        """
        # Timeout is already classified by the container layer
        if run_result.status == ExecutionStatus.TIMEOUT:
            return ExecutionStatus.TIMEOUT

        # Check events count
        events_count = _count_events(run_result.events_file_path)
        events = _load_events(run_result.events_file_path)

        stderr = run_result.stderr.lower()

        # Dependency failures
        if any(
            pattern in stderr
            for pattern in (
                "modulenotfounderror",
                "no module named",
                "importerror",
                "pkg_resources",
                "could not find a version",
                "pip install",
            )
        ):
            return ExecutionStatus.DEPENDENCY_FAILURE

        # Entry point failures
        if any(
            pattern in stderr
            for pattern in (
                "filenotfounderror",
                "no such file or directory",
                "syntaxerror",
                "indentationerror",
            )
        ):
            return ExecutionStatus.ENTRY_POINT_FAILURE

        # Model / API failures
        if any(
            pattern in stderr
            for pattern in (
                "openai",
                "api_key",
                "rate limit",
                "connection refused",
                "httpconnectionpool",
                "connection error",
            )
        ):
            return ExecutionStatus.MODEL_FAILURE

        # Instrumentation failures
        if any(
            pattern in stderr
            for pattern in (
                "stratum_patcher",
                "sitecustomize",
                "instrumentation",
            )
        ):
            return ExecutionStatus.INSTRUMENTATION_FAILURE

        # Check for interaction events
        has_interaction = any(
            e.get("event_type") in self.INTERACTION_EVENT_TYPES
            for e in events
        )

        # Success with rich data — enough events AND at least one interaction
        if run_result.exit_code == 0 and events_count >= MIN_EVENT_THRESHOLD and has_interaction:
            return ExecutionStatus.SUCCESS

        # Partial success — process ran but thin data
        if run_result.exit_code == 0 or events_count > 0:
            return ExecutionStatus.PARTIAL_SUCCESS

        return ExecutionStatus.CRASH

    def classify_status_detailed(self, run_result: RunResult) -> dict[str, Any]:
        """Classify execution outcome with detailed breakdown (Issue 23).

        Returns a dict with status, event type flags, semantic data presence,
        and usable_for_dataset flag.
        """
        status = self.classify_status(run_result)
        events_count = _count_events(run_result.events_file_path)
        events = _load_events(run_result.events_file_path)
        stderr = run_result.stderr or ""

        return {
            "status": status,
            "exit_code": run_result.exit_code,
            "events_emitted": events_count,
            "timeout_hit": run_result.status == ExecutionStatus.TIMEOUT,
            "has_agent_events": any(
                e.get("event_type", "").startswith("agent.") for e in events
            ),
            "has_llm_events": any(
                e.get("event_type", "").startswith("llm.") for e in events
            ),
            "has_delegation_events": any(
                e.get("event_type", "").startswith("delegation.") for e in events
            ),
            "has_error_events": any(
                e.get("event_type", "").startswith("error.") for e in events
            ),
            "has_semantic_data": any(
                e.get("payload", {}).get("output_hash") is not None
                for e in events
                if e.get("event_type") in ("llm.call_end", "agent.task_end")
            ),
            "stderr_summary": stderr[:500] if stderr else None,
            "usable_for_dataset": (
                status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS)
                and events_count >= 5
            ),
        }

    # ------------------------------------------------------------------
    # Input generation
    # ------------------------------------------------------------------

    def _generate_repo_inputs(self, selection: dict[str, Any]) -> list[str]:
        """Generate synthetic inputs for a repo, with error handling."""
        repo_url = selection.get("repo_url", "")
        framework = selection.get("framework", "unknown")

        try:
            inputs = generate_inputs(
                repo_url=repo_url,
                readme_content=selection.get("readme_content", ""),
                entry_point_code=selection.get("entry_point_code", ""),
                detected_input_type=selection.get("detected_input_type", "user prompt"),
                vllm_url=self.vllm_url,
                framework=framework,
                count=3,  # We only need 3 diverse; repeats reuse the first
            )
        except Exception as exc:
            console.print(
                f"  [yellow]Input generation failed for "
                f"{selection.get('repo_id', '?')}: {exc}[/yellow]"
            )
            inputs = [
                '{"task": "Perform the default task for this agent application."}',
                '{"task": "Run a simple test with minimal input."}',
                '{"task": "Execute the primary workflow end to end."}',
            ]
        return inputs

    # ------------------------------------------------------------------
    # Metadata output
    # ------------------------------------------------------------------

    def _write_metadata_csvs(self, results: list[RunResult]) -> None:
        """Write runs.csv and failures.csv from collected results."""
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        # runs.csv
        runs_path = self._meta_dir / "runs.csv"
        with open(runs_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "repo_id", "run_id", "status", "duration_ms",
                "events_count", "framework", "timestamp",
            ])
            for r in results:
                events_count = _count_events(r.events_file_path)
                framework = _extract_framework(r)
                writer.writerow([
                    r.repo_id,
                    r.run_id,
                    r.status,
                    r.duration_ms,
                    events_count,
                    framework,
                    datetime.now(timezone.utc).isoformat(),
                ])

        console.print(f"[green]Wrote {runs_path}[/green] ({len(results)} rows)")

        # failures.csv
        failures = [r for r in results if r.status not in (
            ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS
        )]
        failures_path = self._meta_dir / "failures.csv"
        with open(failures_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["repo_id", "run_id", "status", "error_message"])
            for r in failures:
                writer.writerow([
                    r.repo_id,
                    r.run_id,
                    r.status,
                    (r.error_message or "")[:500],
                ])

        console.print(
            f"[green]Wrote {failures_path}[/green] ({len(failures)} failures)"
        )

    # ------------------------------------------------------------------
    # Loading & setup helpers
    # ------------------------------------------------------------------

    def _load_selections(self) -> None:
        """Load and validate the selection JSON file."""
        console.print(f"Loading selections from [cyan]{self.selection_file}[/cyan]")
        with open(self.selection_file, encoding="utf-8") as fh:
            data = json.load(fh)

        self._selections = data.get("selections", [])
        summary = data.get("summary", {})

        if not self._selections:
            console.print("[red]No selections found in file.[/red]")
            return

        console.print(
            f"Loaded [bold]{len(self._selections)}[/bold] repos  |  "
            f"Frameworks: {summary.get('by_framework', {})}"
        )

    def _ensure_directories(self) -> None:
        """Create output and metadata directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint / resume (v7)
    # ------------------------------------------------------------------

    CHECKPOINT_FILENAME = "scan_checkpoint.json"

    def _load_checkpoint(self, output_dir: Path) -> dict:
        """Load checkpoint from output directory."""
        path = output_dir / self.CHECKPOINT_FILENAME
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._completed_repos = set(data.get("completed_repos", []))
                console.print(
                    f"[green]Resumed from checkpoint:[/green] "
                    f"{len(self._completed_repos)} repos already completed"
                )
                return data
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint: {exc}")
        return {"completed_repos": [], "status_counts": {}, "version": 1}

    def _save_checkpoint(self, output_dir: Path, checkpoint: dict) -> None:
        """Save checkpoint to output directory."""
        checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
        checkpoint["total_completed"] = len(checkpoint["completed_repos"])
        try:
            (output_dir / self.CHECKPOINT_FILENAME).write_text(
                json.dumps(checkpoint, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"Failed to save checkpoint: {exc}")

    def load_checkpoint(self) -> None:
        """Load completed repo IDs from checkpoint file (backward compat)."""
        self._load_checkpoint(self.output_dir.parent)

    def save_checkpoint(self, repo_id: str) -> None:
        """Save a completed repo to the checkpoint file (backward compat)."""
        self._completed_repos.add(repo_id)
        checkpoint = {
            "completed_repos": sorted(self._completed_repos),
            "status_counts": {},
            "version": 1,
        }
        self._save_checkpoint(self.output_dir.parent, checkpoint)

    def is_repo_completed(self, repo_id: str) -> bool:
        """Check if a repo has already been completed."""
        return repo_id in self._completed_repos

    def _print_dry_run(self) -> None:
        """Print dry-run summary without executing anything."""
        console.print("\n[bold yellow]DRY RUN — no containers will be started[/bold yellow]\n")
        for i, sel in enumerate(self._selections, 1):
            repo_id = sel.get("repo_id", "?")
            framework = sel.get("framework", "?")
            entry = sel.get("detected_entry_point", "?")
            console.print(
                f"  {i:4d}. [cyan]{repo_id}[/cyan]  "
                f"framework={framework}  entry={entry}  "
                f"runs={self.runs_per_repo}"
            )
        console.print(
            f"\n[bold]Total:[/bold] {len(self._selections)} repos x "
            f"{self.runs_per_repo} runs = "
            f"{len(self._selections) * self.runs_per_repo} executions"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _count_events(events_file_path: str | None) -> int:
    """Count the number of JSONL lines in an events file."""
    if not events_file_path:
        return 0
    path = Path(events_file_path)
    if not path.exists():
        return 0
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def _load_events(events_file_path: str | None) -> list[dict[str, Any]]:
    """Load all events from a JSONL file."""
    if not events_file_path:
        return []
    path = Path(events_file_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception:
        pass
    return events


def _extract_framework(run_result: RunResult) -> str:
    """Extract framework from a run result's run_id or default."""
    # The run_id format is {repo_id}_run_{n}_{uuid}
    # We don't embed framework there, so return from the repo_id pattern or empty.
    # The caller can override with the selection data; this is a fallback.
    return "unknown"
