"""CLI runner for Phase 1: Repo Selection.

Loads reliability scanner output JSONs (or legacy structural scans),
scores repos for graph discovery coverage, and writes the selection output.
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

from stratum_lab.selection.discovery_optimizer import (
    select_for_discovery,
    validate_coverage,
    FRAMEWORK_TARGETS,
    COMPLEXITY_TARGETS,
)

console = Console()


def _load_scan_jsons(input_dir: Path) -> list[dict[str, Any]]:
    """Load all ``*.json`` files from *input_dir* as repo scan dicts."""
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
        task = progress.add_task("Loading scan results", total=len(json_files))
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
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


def _print_summary(selected: list[dict], coverage: dict) -> None:
    """Pretty-print the selection summary to the console."""
    overview = (
        f"[bold green]Selected: {coverage['total']}[/bold green]  |  "
        f"Unique topologies: {coverage['unique_topologies']}\n"
    )
    if coverage.get("gaps"):
        overview += f"[yellow]Gaps: {len(coverage['gaps'])}[/yellow]  |  "
    else:
        overview += "[green]All targets met[/green]  |  "
    overview += f"All targets met: {coverage.get('all_targets_met', False)}"
    console.print(Panel(overview, title="Selection Summary", border_style="green"))

    # Framework table
    fw_table = Table(title="Framework Coverage", show_header=True, header_style="bold cyan")
    fw_table.add_column("Framework", style="dim")
    fw_table.add_column("Count", justify="right")
    fw_table.add_column("Target", justify="right")
    fw_table.add_column("Met", justify="center")
    for fw, data in coverage.get("framework_coverage", {}).items():
        met = "[green]YES[/green]" if data["met"] else "[red]NO[/red]"
        fw_table.add_row(fw, str(data["count"]), str(data["target"]), met)
    console.print(fw_table)

    # Complexity table
    cx_table = Table(title="Complexity Coverage", show_header=True, header_style="bold cyan")
    cx_table.add_column("Bracket", style="dim")
    cx_table.add_column("Count", justify="right")
    cx_table.add_column("Target", justify="right")
    cx_table.add_column("Met", justify="center")
    for bracket, data in coverage.get("complexity_coverage", {}).items():
        met = "[green]YES[/green]" if data["met"] else "[red]NO[/red]"
        cx_table.add_row(bracket, str(data["count"]), str(data["target"]), met)
    console.print(cx_table)


def run_selection(
    input_dir: str,
    output_file: str,
    target: int = 1000,
    min_runnability: int = 15,
    max_per_archetype: int = 200,
    *,
    use_triage: bool = False,
    use_probe: bool = False,
    probe_timeout: int = 30,
    probe_batch_size: int = 5000,
) -> None:
    """Load scan results, apply discovery optimizer, and write output JSON.

    This is the function invoked by ``stratum-lab select``.
    """
    console.print(
        Panel(
            f"Input: [bold]{input_dir}[/bold]\n"
            f"Output: [bold]{output_file}[/bold]\n"
            f"Target: {target}",
            title="Phase 1: Repo Selection (Graph Discovery)",
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

    qualified = raw_repos

    # Optional triage filtering
    if use_triage:
        console.print("\n[bold]Running static triage...[/bold]")
        try:
            from stratum_lab.triage.static_analyzer import triage_batch
            triage_results = triage_batch(qualified)
            qualified = (
                triage_results["likely_runnable"]
                + triage_results["needs_probe"][:probe_batch_size]
            )
            console.print(f"  {triage_results['total_analyzed']} -> {len(qualified)} qualified")
        except Exception as e:
            console.print(f"  [yellow]Triage skipped: {e}[/yellow]")

    # Optional probe filtering
    if use_probe and qualified:
        console.print("\n[bold]Running probe execution...[/bold]")
        try:
            from stratum_lab.triage.probe import probe_batch
            probe_results = probe_batch(qualified, timeout=probe_timeout)
            console.print(
                f"  {probe_results['total_probed']} probed -> "
                f"{probe_results['passed']} passed"
            )
        except Exception as e:
            console.print(f"  [yellow]Probe skipped: {e}[/yellow]")

    # Discovery-optimized selection
    console.print("\n[bold]Running discovery optimizer...[/bold]")
    selected = select_for_discovery(qualified, target_count=target)
    coverage = validate_coverage(selected)

    elapsed = time.perf_counter() - start

    # Print summary
    console.print()
    _print_summary(selected, coverage)
    console.print(f"\n  Completed in [bold]{elapsed:.1f}s[/bold]")

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "selection": selected,
        "coverage": coverage,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\n  Selection written to [bold green]{output_path}[/bold green]")
    console.print(f"  {len(selected)} repos selected for graph discovery.")
