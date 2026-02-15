"""CLI runner for the Knowledge Base build phase.

Loads enriched graphs, builds the pattern knowledge base, computes
taxonomy probabilities, detects novel patterns, compares frameworks,
and builds the fragility map.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stratum_lab.knowledge.fragility import build_fragility_map
from stratum_lab.knowledge.patterns import (
    build_pattern_knowledge_base,
    compare_frameworks,
    detect_novel_patterns,
)
from stratum_lab.knowledge.taxonomy import (
    TAXONOMY_PRECONDITIONS,
    compute_manifestation_probabilities,
    compute_structural_metric_correlations,
)

console = Console()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_enriched_graphs(enriched_dir: Path) -> list[dict[str, Any]]:
    """Load all enriched graph JSON files from a directory."""
    graphs: list[dict[str, Any]] = []
    json_files = sorted(enriched_dir.glob("*.json"))
    errors = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            graphs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            errors += 1
            if errors <= 5:
                console.print(
                    f"  [yellow]Warning:[/yellow] Failed to load {path.name}: {exc}"
                )

    if errors > 5:
        console.print(f"  [yellow]... and {errors - 5} more load errors[/yellow]")

    return graphs


def _write_json(data: Any, path: Path) -> None:
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    patterns: list[dict[str, Any]],
    taxonomy_probs: dict[str, dict[str, Any]],
    novel: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    fragility: list[dict[str, Any]],
    total_graphs: int,
    elapsed: float,
) -> None:
    """Pretty-print the knowledge base build summary."""
    overview = (
        f"Enriched graphs loaded: [bold]{total_graphs}[/bold]\n"
        f"Patterns identified: [bold]{len(patterns)}[/bold]  |  "
        f"Novel patterns: [bold]{len(novel)}[/bold]\n"
        f"Framework comparisons: [bold]{len(comparisons)}[/bold]  |  "
        f"Fragility entries: [bold]{len(fragility)}[/bold]\n"
        f"Elapsed: {elapsed:.1f}s"
    )
    console.print(Panel(overview, title="Knowledge Base Summary", border_style="green"))

    # Pattern table
    if patterns:
        pat_table = Table(
            title="Structural Patterns", show_header=True, header_style="bold cyan"
        )
        pat_table.add_column("Pattern", style="dim", max_width=40)
        pat_table.add_column("Repos", justify="right")
        pat_table.add_column("Prevalence", justify="right")
        pat_table.add_column("Failure Rate", justify="right")
        pat_table.add_column("Risk", justify="center")

        for pat in patterns[:15]:
            prev = pat.get("prevalence", {})
            bd = pat.get("behavioral_distribution", {})
            risk = pat.get("risk_assessment", {})

            pat_table.add_row(
                pat.get("pattern_name", "?"),
                str(prev.get("repos_count", 0)),
                f"{prev.get('prevalence_rate', 0) * 100:.1f}%",
                f"{bd.get('failure_rate', 0) * 100:.1f}%",
                risk.get("risk_level", "?"),
            )

        console.print(pat_table)

    # Taxonomy probabilities table
    active_preconditions = {
        k: v for k, v in taxonomy_probs.items()
        if isinstance(v, dict) and v.get("sample_size", 0) > 0
    }
    if active_preconditions:
        tax_table = Table(
            title="Taxonomy Manifestation Probabilities (sampled)",
            show_header=True,
            header_style="bold cyan",
        )
        tax_table.add_column("Precondition", style="dim", max_width=40)
        tax_table.add_column("Sample", justify="right")
        tax_table.add_column("Probability", justify="right")
        tax_table.add_column("CI (95%)", justify="right")
        tax_table.add_column("Severity", justify="center")

        # Sort by probability descending
        sorted_prec = sorted(
            active_preconditions.items(),
            key=lambda kv: kv[1].get("probability", 0) or 0,
            reverse=True,
        )
        for precondition_id, data in sorted_prec[:15]:
            prob = data.get("probability")
            ci = data.get("confidence_interval", [None, None])
            severity = data.get("severity_when_manifested", {})
            sev_label = severity.get("severity_label", "?") if severity else "?"

            prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
            ci_str = (
                f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"
                if ci[0] is not None
                else "N/A"
            )

            tax_table.add_row(
                precondition_id,
                str(data.get("sample_size", 0)),
                prob_str,
                ci_str,
                sev_label,
            )

        console.print(tax_table)

    # Fragility table
    if fragility:
        frag_table = Table(
            title="Fragility Map", show_header=True, header_style="bold cyan"
        )
        frag_table.add_column("Position", style="dim")
        frag_table.add_column("Nodes", justify="right")
        frag_table.add_column("Repos", justify="right")
        frag_table.add_column("Avg Failure Rate", justify="right")
        frag_table.add_column("Sensitivity", justify="right")
        frag_table.add_column("Quality Dep", justify="right")

        for entry in fragility:
            frag_table.add_row(
                entry.get("structural_position", "?"),
                str(entry.get("total_nodes_analyzed", 0)),
                str(entry.get("affected_repos_count", 0)),
                f"{entry.get('avg_tool_call_failure_rate', 0) * 100:.1f}%",
                f"{entry.get('sensitivity_score', 0):.3f}",
                f"{entry.get('quality_dependent_rate', 0) * 100:.1f}%",
            )

        console.print(frag_table)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_knowledge_build(enriched_dir: str, output_dir: str) -> None:
    """Build the cross-repo pattern knowledge base.

    This is the function invoked by ``stratum-lab knowledge``.

    Parameters
    ----------
    enriched_dir:
        Directory containing enriched graph JSON files.
    output_dir:
        Directory to write knowledge base output files.
    """
    console.print(
        Panel(
            f"Enriched: [bold]{enriched_dir}[/bold]\n"
            f"Output: [bold]{output_dir}[/bold]",
            title="Knowledge Base Build",
            border_style="blue",
        )
    )

    enriched_path = Path(enriched_dir)
    output_path = Path(output_dir)

    if not enriched_path.is_dir():
        console.print(f"[red]Error: {enriched_dir} is not a directory[/red]")
        return

    start = time.perf_counter()

    # Load enriched graphs
    console.print("\n[bold]Loading enriched graphs...[/bold]")
    enriched_graphs = _load_enriched_graphs(enriched_path)

    if not enriched_graphs:
        console.print("[red]No enriched graphs found. Aborting.[/red]")
        return

    console.print(f"  Loaded [bold]{len(enriched_graphs)}[/bold] enriched graphs")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Build pattern knowledge base
    console.print("\n[bold]Building pattern knowledge base...[/bold]")
    patterns = build_pattern_knowledge_base(enriched_graphs)
    _write_json(patterns, output_path / "patterns.json")
    console.print(f"  Identified [bold]{len(patterns)}[/bold] structural patterns")

    # 2. Compute taxonomy probabilities
    console.print("\n[bold]Computing taxonomy manifestation probabilities...[/bold]")
    taxonomy_probs = compute_manifestation_probabilities(enriched_graphs)
    _write_json(taxonomy_probs, output_path / "taxonomy_probabilities.json")
    active_count = sum(
        1 for v in taxonomy_probs.values()
        if isinstance(v, dict) and v.get("sample_size", 0) > 0
    )
    console.print(
        f"  Computed probabilities for [bold]{active_count}[/bold] of "
        f"{len(TAXONOMY_PRECONDITIONS)} preconditions"
    )

    # 3. Compute structural metric correlations
    console.print("\n[bold]Computing structural metric correlations...[/bold]")
    correlations = compute_structural_metric_correlations(enriched_graphs)
    _write_json(correlations, output_path / "structural_correlations.json")
    corr_count = sum(
        1 for v in correlations.values()
        if isinstance(v, dict) and v.get("correlation") is not None
    )
    console.print(f"  Computed [bold]{corr_count}[/bold] significant correlations")

    # 4. Detect novel patterns
    console.print("\n[bold]Detecting novel patterns...[/bold]")
    novel_patterns = detect_novel_patterns(enriched_graphs)
    _write_json(novel_patterns, output_path / "novel_patterns.json")
    console.print(f"  Found [bold]{len(novel_patterns)}[/bold] novel behavioral patterns")

    # 5. Compare frameworks
    console.print("\n[bold]Comparing frameworks...[/bold]")
    comparisons = compare_frameworks(enriched_graphs)
    _write_json(comparisons, output_path / "framework_comparisons.json")
    console.print(f"  Generated [bold]{len(comparisons)}[/bold] framework comparisons")

    # 6. Build fragility map
    console.print("\n[bold]Building fragility map...[/bold]")
    fragility_map = build_fragility_map(enriched_graphs)
    _write_json(fragility_map, output_path / "fragility_map.json")
    console.print(f"  Mapped [bold]{len(fragility_map)}[/bold] fragility entries")

    # 7. Compute and store graph fingerprints for similarity search
    console.print("\n[bold]Computing graph fingerprints...[/bold]")
    try:
        from stratum_lab.query.fingerprint import (
            compute_graph_fingerprint,
            compute_normalization_constants,
        )
        fingerprints: dict[str, Any] = {}
        for graph in enriched_graphs:
            repo_id = graph.get("repo_id", "unknown")
            structural = graph.get("structural", graph)
            fingerprints[repo_id] = compute_graph_fingerprint(structural)

        _write_json(fingerprints, output_path / "fingerprints.json")

        norm_constants = compute_normalization_constants(list(fingerprints.values()))
        _write_json(norm_constants, output_path / "normalization.json")
        console.print(
            f"  Stored [bold]{len(fingerprints)}[/bold] fingerprints + normalization constants"
        )
    except Exception as exc:
        console.print(f"  [yellow]Warning: fingerprint computation failed: {exc}[/yellow]")

    elapsed = time.perf_counter() - start

    # Print summary
    console.print()
    _print_summary(
        patterns, taxonomy_probs, novel_patterns, comparisons,
        fragility_map, len(enriched_graphs), elapsed,
    )

    console.print(
        f"\n  Knowledge base written to [bold green]{output_path}[/bold green]"
    )
    console.print(
        f"  Files: patterns.json, taxonomy_probabilities.json, "
        f"structural_correlations.json, novel_patterns.json, "
        f"framework_comparisons.json, fragility_map.json, "
        f"fingerprints.json, normalization.json"
    )
