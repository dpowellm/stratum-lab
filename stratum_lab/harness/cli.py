"""CLI runner for Phase 2: Execution Harness.

Called from the main ``stratum-lab execute`` CLI command.  Loads the
selection JSON, creates output directories, instantiates the Orchestrator,
runs all executions, and prints a summary.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from stratum_lab.config import ExecutionStatus
from stratum_lab.harness.orchestrator import Orchestrator

console = Console()


def run_execution(
    selection_file: str,
    output_dir: str,
    vllm_url: str,
    concurrent: int,
    timeout: int,
    runs: int,
    dry_run: bool,
) -> None:
    """Entry point for Phase 2 execution.

    Parameters
    ----------
    selection_file:
        Path to the Phase 1 selection JSON file.
    output_dir:
        Directory for raw event JSONL output files.
    vllm_url:
        OpenAI-compatible vLLM endpoint URL.
    concurrent:
        Number of concurrent Docker containers.
    timeout:
        Per-execution timeout in seconds.
    runs:
        Total runs per repo (default 5: 3 diverse + 2 repeat).
    dry_run:
        If ``True``, print the plan without executing.
    """
    selection_path = Path(selection_file)
    output_path = Path(output_dir)

    # Validate selection file
    if not selection_path.is_file():
        console.print(f"[red]Selection file not found:[/red] {selection_path}")
        return

    try:
        with open(selection_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Failed to load selection file:[/red] {exc}")
        return

    selections = data.get("selections", [])
    if not selections:
        console.print("[red]No repos found in selection file.[/red]")
        return

    console.print()
    console.rule("[bold]Phase 2: Execution Harness[/bold]")
    console.print()
    console.print(f"  Selection file : [cyan]{selection_path}[/cyan]")
    console.print(f"  Output dir     : [cyan]{output_path}[/cyan]")
    console.print(f"  vLLM endpoint  : [cyan]{vllm_url}[/cyan]")
    console.print(f"  Concurrency    : {concurrent}")
    console.print(f"  Timeout        : {timeout}s")
    console.print(f"  Runs per repo  : {runs}")
    console.print(f"  Total repos    : {len(selections)}")
    console.print(f"  Total runs     : {len(selections) * runs}")
    console.print()

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    meta_dir = output_path.parent / "execution_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate and run orchestrator
    orchestrator = Orchestrator(
        selection_file=selection_file,
        output_dir=output_dir,
        vllm_url=vllm_url,
        concurrent=concurrent,
        timeout=timeout,
        runs_per_repo=runs,
    )

    results = orchestrator.run(dry_run=dry_run)

    if dry_run:
        console.print("\n[bold yellow]Dry run complete. No containers were started.[/bold yellow]")
        return

    # Print summary
    _print_summary(results, selections)


def _print_summary(results: list, selections: list[dict]) -> None:
    """Print a rich summary table of execution results."""
    if not results:
        console.print("[yellow]No results to summarise.[/yellow]")
        return

    console.print()
    console.rule("[bold]Execution Summary[/bold]")
    console.print()

    # Status counts
    status_counts = Counter(r.status for r in results)

    status_table = Table(title="Status Breakdown", show_lines=True)
    status_table.add_column("Status", style="bold")
    status_table.add_column("Count", justify="right")
    status_table.add_column("Percentage", justify="right")

    total = len(results)
    for status in [
        ExecutionStatus.SUCCESS,
        ExecutionStatus.PARTIAL_SUCCESS,
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CRASH,
        ExecutionStatus.DEPENDENCY_FAILURE,
        ExecutionStatus.ENTRY_POINT_FAILURE,
        ExecutionStatus.MODEL_FAILURE,
        ExecutionStatus.INSTRUMENTATION_FAILURE,
    ]:
        count = status_counts.get(status, 0)
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0.0%"
        color = (
            "green" if status == ExecutionStatus.SUCCESS
            else "yellow" if status == ExecutionStatus.PARTIAL_SUCCESS
            else "red"
        )
        status_table.add_row(f"[{color}]{status}[/{color}]", str(count), pct)

    console.print(status_table)

    # Timing stats
    durations = [r.duration_ms for r in results if r.duration_ms > 0]
    if durations:
        console.print()
        console.print(f"  [bold]Timing:[/bold]")
        console.print(f"    Mean duration  : {sum(durations) / len(durations):,.0f} ms")
        console.print(f"    Median duration: {sorted(durations)[len(durations) // 2]:,} ms")
        console.print(f"    Min duration   : {min(durations):,} ms")
        console.print(f"    Max duration   : {max(durations):,} ms")
        total_time_s = sum(durations) / 1000
        console.print(f"    Total wall time: {total_time_s:,.1f} s")

    # Repo-level success rate
    repo_successes: Counter[str] = Counter()
    repo_totals: Counter[str] = Counter()
    for r in results:
        repo_totals[r.repo_id] += 1
        if r.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS):
            repo_successes[r.repo_id] += 1

    repos_with_at_least_one_success = sum(
        1 for rid in repo_totals if repo_successes.get(rid, 0) > 0
    )
    total_repos = len(repo_totals)

    console.print()
    console.print(f"  [bold]Repos:[/bold]")
    console.print(
        f"    Total repos executed       : {total_repos}"
    )
    console.print(
        f"    Repos with >= 1 success    : "
        f"[green]{repos_with_at_least_one_success}[/green] / {total_repos}"
    )
    console.print(
        f"    Repo-level success rate    : "
        f"{repos_with_at_least_one_success / max(total_repos, 1) * 100:.1f}%"
    )
    console.print()
