"""Main orchestrator for Phase 2 execution.

Reads the selection JSON, generates synthetic inputs, schedules runs across
a thread-pool of Docker containers, collects results, and writes metadata
CSVs for downstream pipeline phases.
"""

from __future__ import annotations

import csv
import json
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

from stratum_lab.config import (
    DEFAULT_CONCURRENT_CONTAINERS,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_RUNS_PER_REPO,
    EXECUTION_META_DIR,
    ExecutionStatus,
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
    """

    def __init__(
        self,
        selection_file: str | Path,
        output_dir: str | Path,
        vllm_url: str,
        concurrent: int = DEFAULT_CONCURRENT_CONTAINERS,
        timeout: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        runs_per_repo: int = DEFAULT_RUNS_PER_REPO,
    ) -> None:
        self.selection_file = Path(selection_file)
        self.output_dir = Path(output_dir)
        self.vllm_url = vllm_url
        self.concurrent = concurrent
        self.timeout = timeout
        self.runs_per_repo = runs_per_repo

        # Loaded at run-time
        self._selections: list[dict[str, Any]] = []
        self._results: list[RunResult] = []

        # Metadata tracking
        self._meta_dir = self.output_dir.parent / "execution_metadata"

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
    ) -> RunResult:
        """Execute a single run for a repo.

        1. Calls ``container.run_container()``
        2. Classifies the result status
        3. Saves the events file to output_dir
        4. Returns the ``RunResult``
        """
        repo_id = repo_selection.get("repo_id", "unknown")
        repo_url = repo_selection.get("repo_url", "")
        framework = repo_selection.get("framework", "unknown")
        entry_point = repo_selection.get("detected_entry_point", "main.py")
        run_id = run_id or f"{repo_id}_run_{run_number}_{uuid.uuid4().hex[:8]}"

        result = run_container(
            repo_url=repo_url,
            entry_point=entry_point,
            run_id=run_id,
            repo_id=repo_id,
            framework=framework,
            input_data=input_data,
            timeout=self.timeout,
            vllm_url=self.vllm_url,
        )

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

    # ------------------------------------------------------------------
    # Status classification
    # ------------------------------------------------------------------

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

        # Success with events
        if run_result.exit_code == 0 and events_count > 0:
            return ExecutionStatus.SUCCESS

        # Partial success — process exited OK but no/few events
        if run_result.exit_code == 0 and events_count == 0:
            return ExecutionStatus.PARTIAL_SUCCESS

        # Non-zero exit with some events captured
        if run_result.exit_code is not None and run_result.exit_code != 0 and events_count > 0:
            return ExecutionStatus.PARTIAL_SUCCESS

        return ExecutionStatus.CRASH

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


def _extract_framework(run_result: RunResult) -> str:
    """Extract framework from a run result's run_id or default."""
    # The run_id format is {repo_id}_run_{n}_{uuid}
    # We don't embed framework there, so return from the repo_id pattern or empty.
    # The caller can override with the selection data; this is a fallback.
    return "unknown"
