"""CLI runner for Phase 3: Data Collection.

Scans a directory of JSONL event files, parses events into structured
run records, aggregates by repo, and writes the output JSON.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from stratum_lab.collection.parser import (
    aggregate_run_records,
    build_run_record,
    parse_events_file,
)

console = Console()


def _discover_jsonl_files(events_dir: Path) -> list[Path]:
    """Find all .jsonl files in the events directory."""
    files = sorted(events_dir.rglob("*.jsonl"))
    return files


def _group_events_by_run(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group events by run_id."""
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        run_id = ev.get("run_id", "unknown")
        by_run[run_id].append(ev)
    return dict(by_run)


def _print_summary(
    repo_summaries: dict[str, dict[str, Any]],
    total_files: int,
    total_events: int,
    total_runs: int,
    elapsed: float,
) -> None:
    """Pretty-print the collection summary."""
    overview = (
        f"Files parsed: {total_files}  |  "
        f"Total events: [bold]{total_events:,}[/bold]  |  "
        f"Total runs: [bold]{total_runs}[/bold]  |  "
        f"Repos: [bold]{len(repo_summaries)}[/bold]\n"
        f"Elapsed: {elapsed:.1f}s"
    )
    console.print(Panel(overview, title="Collection Summary", border_style="green"))

    # Per-repo table
    table = Table(title="Repo Run Summaries", show_header=True, header_style="bold cyan")
    table.add_column("Repo ID", style="dim", max_width=40)
    table.add_column("Framework", justify="center")
    table.add_column("Runs", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Avg Duration", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("LLM Calls", justify="right")

    # Show up to 30 repos, sorted by total events descending
    sorted_repos = sorted(
        repo_summaries.items(),
        key=lambda kv: kv[1].get("total_events", 0),
        reverse=True,
    )
    for repo_id, summary in sorted_repos[:30]:
        timeline = summary.get("execution_timeline", {})
        avg_dur = f"{timeline.get('avg_duration_ms', 0):.0f}ms"
        errs = str(summary.get("error_summary", {}).get("total_errors", 0))
        llm = str(summary.get("llm_calls", {}).get("total_count", 0))

        table.add_row(
            repo_id,
            summary.get("framework", "?"),
            str(summary.get("num_runs", 0)),
            f"{summary.get('total_events', 0):,}",
            avg_dur,
            errs,
            llm,
        )

    if len(sorted_repos) > 30:
        table.add_row("...", "", "", "", "", "", "", style="dim")

    console.print(table)


def run_collection(events_dir: str, output_file: str) -> None:
    """Parse raw event files into structured run records.

    This is the function invoked by ``stratum-lab collect``.

    Parameters
    ----------
    events_dir:
        Directory containing JSONL event files from the execution phase.
    output_file:
        Path to write the structured run records JSON.
    """
    console.print(
        Panel(
            f"Events dir: [bold]{events_dir}[/bold]\n"
            f"Output: [bold]{output_file}[/bold]",
            title="Phase 3: Data Collection",
            border_style="blue",
        )
    )

    events_path = Path(events_dir)
    if not events_path.is_dir():
        console.print(f"[red]Error: {events_dir} is not a directory[/red]")
        return

    # Discover JSONL files
    jsonl_files = _discover_jsonl_files(events_path)
    if not jsonl_files:
        console.print(f"[red]No .jsonl files found in {events_dir}[/red]")
        return

    console.print(f"  Found [bold]{len(jsonl_files)}[/bold] JSONL files")

    start = time.perf_counter()

    # Parse all files
    all_run_records: list[dict[str, Any]] = []
    total_events = 0
    parse_errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing event files", total=len(jsonl_files))

        for filepath in jsonl_files:
            try:
                events = parse_events_file(filepath)
                total_events += len(events)

                # Group events by run_id and build a record per run
                by_run = _group_events_by_run(events)
                for run_id, run_events in by_run.items():
                    record = build_run_record(run_events)
                    all_run_records.append(record)

            except OSError as exc:
                parse_errors += 1
                if parse_errors <= 5:
                    console.print(
                        f"  [yellow]Warning:[/yellow] Failed to read {filepath.name}: {exc}"
                    )
            progress.advance(task)

    if parse_errors > 5:
        console.print(f"  [yellow]... and {parse_errors - 5} more file read errors[/yellow]")

    console.print(
        f"\n  Parsed [bold]{total_events:,}[/bold] events into "
        f"[bold]{len(all_run_records)}[/bold] run records"
    )

    # Aggregate by repo
    console.print("\n[bold]Aggregating by repo...[/bold]")
    runs_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_run_records:
        repo_id = record.get("repo_id", "unknown")
        runs_by_repo[repo_id].append(record)

    repo_summaries: dict[str, dict[str, Any]] = {}
    for repo_id, records in runs_by_repo.items():
        repo_summaries[repo_id] = aggregate_run_records(records)

    elapsed = time.perf_counter() - start

    # Print summary
    console.print()
    _print_summary(repo_summaries, len(jsonl_files), total_events, len(all_run_records), elapsed)

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "run_records": all_run_records,
        "repo_summaries": repo_summaries,
        "stats": {
            "total_files_parsed": len(jsonl_files),
            "total_events": total_events,
            "total_runs": len(all_run_records),
            "total_repos": len(repo_summaries),
            "parse_errors": parse_errors,
            "elapsed_seconds": round(elapsed, 2),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n  Run records written to [bold green]{output_path}[/bold green]")
    console.print(
        f"  {len(all_run_records)} runs across {len(repo_summaries)} repos collected."
    )
