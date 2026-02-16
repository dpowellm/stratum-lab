"""CLI runner for Phase 4: Graph Overlay.

Loads structural graphs and runtime events, enriches graphs with behavioral
data, detects emergent and dead edges, and writes enriched graph JSONs.
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

from stratum_lab.collection.parser import build_run_record, parse_events_file
from stratum_lab.overlay.edges import detect_dead_edges, detect_emergent_edges, detect_emergent_edges_v2
from stratum_lab.overlay.enricher import enrich_graph, compute_edge_validation, compute_node_activation
from stratum_lab.overlay.error_propagation import trace_error_propagation
from stratum_lab.overlay.failure_modes import classify_failure_modes
from stratum_lab.overlay.monitoring_baselines import extract_monitoring_baselines

console = Console()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_structural_graphs(structural_dir: Path) -> dict[str, dict[str, Any]]:
    """Load structural graph JSON files from a directory.

    Returns a dict mapping repo_id -> structural graph dict.
    """
    graphs: dict[str, dict[str, Any]] = {}
    json_files = sorted(structural_dir.glob("*.json"))
    errors = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            repo_id = data.get("repo_id") or path.stem
            graphs[repo_id] = data
        except (json.JSONDecodeError, OSError) as exc:
            errors += 1
            if errors <= 5:
                console.print(
                    f"  [yellow]Warning:[/yellow] Failed to load {path.name}: {exc}"
                )

    if errors > 5:
        console.print(f"  [yellow]... and {errors - 5} more load errors[/yellow]")

    return graphs


def _load_events_by_repo(events_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and parse JSONL event files, grouped by repo_id.

    Returns a dict mapping repo_id -> list of event dicts.
    """
    events_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    jsonl_files = sorted(events_dir.rglob("*.jsonl"))
    errors = 0

    for path in jsonl_files:
        try:
            events = parse_events_file(path)
            for ev in events:
                repo_id = ev.get("repo_id", "unknown")
                events_by_repo[repo_id].append(ev)
        except OSError as exc:
            errors += 1
            if errors <= 5:
                console.print(
                    f"  [yellow]Warning:[/yellow] Failed to read {path.name}: {exc}"
                )

    if errors > 5:
        console.print(f"  [yellow]... and {errors - 5} more read errors[/yellow]")

    return dict(events_by_repo)


def _build_run_records_from_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group events by run_id and build run records.

    Each run record also carries its raw events under the ``events`` key
    so the enricher can access individual events.
    """
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        run_id = ev.get("run_id", "unknown")
        runs[run_id].append(ev)

    records: list[dict[str, Any]] = []
    for run_id, run_events in runs.items():
        record = build_run_record(run_events)
        record["events"] = run_events  # Attach raw events for enricher

        # Propagate input_hash from events into metadata so the enricher
        # can group runs by input for determinism analysis.
        if "metadata" not in record:
            record["metadata"] = {}
        if not record["metadata"].get("input_hash"):
            for ev in run_events:
                ih = (ev.get("payload") or {}).get("input_hash")
                if ih:
                    record["metadata"]["input_hash"] = ih
                    break

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    results: dict[str, dict[str, Any]],
    total_structural: int,
    total_event_repos: int,
    elapsed: float,
) -> None:
    """Pretty-print the overlay summary."""
    enriched_count = len(results)
    total_emergent = sum(
        len(r.get("emergent_edges", [])) for r in results.values()
    )
    total_dead = sum(
        len(r.get("dead_edges", [])) for r in results.values()
    )

    overview = (
        f"Structural graphs: {total_structural}  |  "
        f"Event repos: {total_event_repos}  |  "
        f"[bold green]Enriched: {enriched_count}[/bold green]\n"
        f"Emergent edges found: {total_emergent}  |  "
        f"Dead edges found: {total_dead}  |  "
        f"Elapsed: {elapsed:.1f}s"
    )
    console.print(Panel(overview, title="Overlay Summary", border_style="green"))

    # Per-repo table
    table = Table(title="Enriched Graphs", show_header=True, header_style="bold cyan")
    table.add_column("Repo ID", style="dim", max_width=40)
    table.add_column("Framework", justify="center")
    table.add_column("Nodes", justify="right")
    table.add_column("Edges", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Emergent", justify="right")
    table.add_column("Dead", justify="right")

    sorted_results = sorted(results.items())
    for repo_id, graph in sorted_results[:30]:
        table.add_row(
            repo_id,
            graph.get("framework", "?"),
            str(len(graph.get("nodes", {}))),
            str(len(graph.get("edges", {}))),
            str(graph.get("total_runs", 0)),
            str(len(graph.get("emergent_edges", []))),
            str(len(graph.get("dead_edges", []))),
        )

    if len(sorted_results) > 30:
        table.add_row("...", "", "", "", "", "", "", style="dim")

    console.print(table)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_overlay(structural_dir: str, events_dir: str, output_dir: str) -> None:
    """Merge behavioral data onto structural graphs.

    This is the function invoked by ``stratum-lab overlay``.

    Parameters
    ----------
    structural_dir:
        Directory containing structural graph JSON files.
    events_dir:
        Directory containing JSONL event files from execution.
    output_dir:
        Directory to write enriched graph JSON files.
    """
    console.print(
        Panel(
            f"Structural: [bold]{structural_dir}[/bold]\n"
            f"Events: [bold]{events_dir}[/bold]\n"
            f"Output: [bold]{output_dir}[/bold]",
            title="Phase 4: Graph Overlay",
            border_style="blue",
        )
    )

    structural_path = Path(structural_dir)
    events_path = Path(events_dir)
    output_path = Path(output_dir)

    if not structural_path.is_dir():
        console.print(f"[red]Error: {structural_dir} is not a directory[/red]")
        return
    if not events_path.is_dir():
        console.print(f"[red]Error: {events_dir} is not a directory[/red]")
        return

    start = time.perf_counter()

    # Load structural graphs
    console.print("\n[bold]Loading structural graphs...[/bold]")
    structural_graphs = _load_structural_graphs(structural_path)
    console.print(f"  Loaded [bold]{len(structural_graphs)}[/bold] structural graphs")

    # Load and group events by repo
    console.print("\n[bold]Loading event files...[/bold]")
    events_by_repo = _load_events_by_repo(events_path)
    console.print(f"  Loaded events for [bold]{len(events_by_repo)}[/bold] repos")

    # Find repos with both structural and event data
    common_repos = set(structural_graphs.keys()) & set(events_by_repo.keys())
    if not common_repos:
        console.print(
            "[yellow]Warning: No repos have both structural and event data.[/yellow]"
        )
        console.print(
            f"  Structural repos: {sorted(structural_graphs.keys())[:10]}"
        )
        console.print(
            f"  Event repos: {sorted(events_by_repo.keys())[:10]}"
        )

    console.print(
        f"\n  Repos with both structural + events: [bold]{len(common_repos)}[/bold]"
    )

    # Enrich each repo
    results: dict[str, dict[str, Any]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching graphs", total=len(common_repos))

        for repo_id in sorted(common_repos):
            structural_graph = structural_graphs[repo_id]
            repo_events = events_by_repo[repo_id]

            # Build run records with raw events attached
            run_records = _build_run_records_from_events(repo_events)

            # Enrich the structural graph (legacy overlay)
            enriched = enrich_graph(structural_graph, run_records)

            # Detect emergent and dead edges (legacy)
            structural_edges = structural_graph.get("edges", {})
            structural_nodes = structural_graph.get("nodes", {})
            total_runs = len(run_records) if run_records else 1

            runtime_interactions = [
                ev for ev in repo_events
                if ev.get("source_node") and ev.get("target_node")
            ]

            emergent = detect_emergent_edges(
                structural_edges, runtime_interactions, structural_nodes, total_runs
            )
            dead = detect_dead_edges(
                structural_edges, runtime_interactions, total_runs, structural_nodes
            )

            enriched["emergent_edges"] = emergent
            enriched["dead_edges"] = dead

            # ---- v6 analyses ----
            # 4a. Edge validation matrix
            enriched["edge_validation"] = compute_edge_validation(
                structural_graph, repo_events, total_runs
            )

            # 4b. Emergent edge detection v2 with classification
            enriched["emergent_edges_v2"] = detect_emergent_edges_v2(
                structural_graph, repo_events, total_runs
            )

            # 4c. Node activation topology
            enriched["node_activation"] = compute_node_activation(
                structural_graph, repo_events, total_runs
            )

            # 4d. Error propagation tracing
            enriched["error_propagation"] = trace_error_propagation(
                structural_graph, repo_events
            )

            # 4e. Failure mode classification
            findings = structural_graph.get("findings", [])
            enriched["failure_modes"] = classify_failure_modes(
                findings, repo_events, structural_graph,
                enriched["error_propagation"]
            )

            # 4f. Monitoring baseline extraction
            enriched["monitoring_baselines"] = extract_monitoring_baselines(
                findings, repo_events
            )

            results[repo_id] = enriched
            progress.advance(task)

    elapsed = time.perf_counter() - start

    # Print summary
    console.print()
    _print_summary(results, len(structural_graphs), len(events_by_repo), elapsed)

    # Write output
    output_path.mkdir(parents=True, exist_ok=True)
    written = 0

    for repo_id, enriched in results.items():
        out_file = output_path / f"{repo_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False, default=str)
        written += 1

    console.print(
        f"\n  Wrote [bold green]{written}[/bold green] enriched graphs to "
        f"[bold]{output_path}[/bold]"
    )
