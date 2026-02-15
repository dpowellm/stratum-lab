"""CLI runner for Phase 1: Repo Selection.

Loads structural scan JSONs, scores and selects repos, and writes the
selection output to a JSON file.
"""

from __future__ import annotations

import json
import time
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

from stratum_lab.selection.selector import score_and_select

console = Console()


def _load_scan_jsons(input_dir: Path) -> list[dict[str, Any]]:
    """Load all ``*.json`` files from *input_dir* as structural scan dicts."""
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        console.print(f"[red]No JSON files found in {input_dir}[/red]")
        return []

    repos: list[dict[str, Any]] = []
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading structural scans", total=len(json_files))
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Accept both single-repo dicts and lists of repos
                if isinstance(data, list):
                    repos.extend(data)
                elif isinstance(data, dict):
                    repos.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                errors += 1
                if errors <= 5:
                    console.print(f"  [yellow]Warning:[/yellow] Failed to load {path.name}: {exc}")
            progress.advance(task)

    if errors > 5:
        console.print(f"  [yellow]... and {errors - 5} more load errors[/yellow]")

    console.print(f"  Loaded [bold]{len(repos)}[/bold] repos from {len(json_files)} files")
    return repos


def _print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print the selection summary to the console."""
    # Overview panel
    overview = (
        f"Scanned: {summary['total_scanned']}  |  "
        f"Eligible: {summary['total_eligible']}  |  "
        f"[bold green]Selected: {summary['total_selected']}[/bold green]\n"
        f"Avg structural value: {summary['avg_structural_value']}  |  "
        f"Avg runnability: {summary['avg_runnability_score']}  |  "
        f"Avg agent count: {summary['avg_agent_count']}\n"
        f"Taxonomy preconditions covered: {summary['total_taxonomy_preconditions_covered']}  |  "
        f"Frameworks: {summary['frameworks_represented']}  |  "
        f"Archetypes: {summary['archetypes_represented']}"
    )
    console.print(Panel(overview, title="Selection Summary", border_style="green"))

    # Framework table
    fw_table = Table(title="Framework Distribution", show_header=True, header_style="bold cyan")
    fw_table.add_column("Framework", style="dim")
    fw_table.add_column("Count", justify="right")
    fw_table.add_column("Percentage", justify="right")
    total = max(summary["total_selected"], 1)
    for fw, count in summary.get("by_framework", {}).items():
        pct = f"{count / total * 100:.1f}%"
        fw_table.add_row(fw, str(count), pct)
    console.print(fw_table)

    # Archetype table
    arch_table = Table(title="Archetype Distribution", show_header=True, header_style="bold cyan")
    arch_table.add_column("Archetype", style="dim")
    arch_table.add_column("Count", justify="right")
    arch_table.add_column("Percentage", justify="right")
    for arch, count in summary.get("by_archetype", {}).items():
        pct = f"{count / total * 100:.1f}%"
        arch_table.add_row(arch, str(count), pct)
    console.print(arch_table)


def run_selection(
    input_dir: str,
    output_file: str,
    target: int,
    min_runnability: int,
    max_per_archetype: int,
) -> None:
    """Load structural scans, score, select, and write output JSON.

    This is the function invoked by ``stratum-lab select``.

    Parameters
    ----------
    input_dir:
        Directory containing structural scan JSON files from stratum-cli.
    output_file:
        Path to write the selection output JSON.
    target:
        Target number of repos to select (1500-2000).
    min_runnability:
        Minimum runnability score for eligibility.
    max_per_archetype:
        Maximum repos per archetype.
    """
    console.print(
        Panel(
            f"Input: [bold]{input_dir}[/bold]\n"
            f"Output: [bold]{output_file}[/bold]\n"
            f"Target: {target}  |  Min runnability: {min_runnability}  |  "
            f"Max/archetype: {max_per_archetype}",
            title="Phase 1: Repo Selection",
            border_style="blue",
        )
    )

    input_path = Path(input_dir)
    if not input_path.is_dir():
        console.print(f"[red]Error: {input_dir} is not a directory[/red]")
        return

    # Load
    start = time.perf_counter()
    raw_repos = _load_scan_jsons(input_path)
    if not raw_repos:
        console.print("[red]No repos loaded. Aborting selection.[/red]")
        return

    # Score and select
    console.print("\n[bold]Scoring and selecting...[/bold]")
    selected, summary = score_and_select(
        raw_repos,
        target=target,
        min_runnability=min_runnability,
        max_per_archetype=max_per_archetype,
    )

    elapsed = time.perf_counter() - start

    # Print summary
    console.print()
    _print_summary(summary)
    console.print(f"\n  Completed in [bold]{elapsed:.1f}s[/bold]")

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "selection": selected,
        "summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n  Selection written to [bold green]{output_path}[/bold green]")
    console.print(f"  {len(selected)} repos selected for behavioral scanning.")
