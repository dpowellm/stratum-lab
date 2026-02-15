"""Main CLI entry point for stratum-lab pipeline."""

import click

from stratum_lab import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Stratum Lab — Behavioral scan infrastructure for AI agent repos."""


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", default="data/execution_metadata/selection.json",
              help="Output selection file path.")
@click.option("-n", "--target", default=1500, help="Target number of repos to select.")
@click.option("--min-runnability", default=15, help="Minimum runnability score filter.")
@click.option("--max-per-archetype", default=200, help="Max repos per archetype.")
def select(input_dir, output_file, target, min_runnability, max_per_archetype):
    """Phase 1: Select repos from 50k structural scan output.

    INPUT_DIR is the directory containing structural scan JSON files.
    """
    from stratum_lab.selection.cli import run_selection
    run_selection(input_dir, output_file, target, min_runnability, max_per_archetype)


@main.command()
@click.argument("selection_file", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="data/raw_events",
              help="Output directory for event files.")
@click.option("--vllm-url", default="http://host.docker.internal:8000/v1",
              help="vLLM OpenAI-compatible endpoint URL.")
@click.option("--concurrent", default=5, help="Number of concurrent containers.")
@click.option("--timeout", default=600, help="Per-execution timeout in seconds.")
@click.option("--runs", default=5, help="Runs per repo (3 diverse + 2 repeat).")
@click.option("--dry-run", is_flag=True, help="Print what would be executed without running.")
def execute(selection_file, output_dir, vllm_url, concurrent, timeout, runs, dry_run):
    """Phase 2: Execute selected repos in sandboxed containers.

    SELECTION_FILE is the JSON file from the 'select' phase.
    """
    from stratum_lab.harness.cli import run_execution
    run_execution(selection_file, output_dir, vllm_url, concurrent, timeout, runs, dry_run)


@main.command()
@click.argument("events_dir", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", default="data/execution_metadata/runs.json",
              help="Output run records file.")
def collect(events_dir, output_file):
    """Phase 3: Parse raw event files into structured run records.

    EVENTS_DIR is the directory containing JSONL event files.
    """
    from stratum_lab.collection.cli import run_collection
    run_collection(events_dir, output_file)


@main.command()
@click.argument("structural_dir", type=click.Path(exists=True))
@click.argument("events_dir", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="data/enriched_graphs",
              help="Output directory for enriched graph files.")
def overlay(structural_dir, events_dir, output_dir):
    """Phase 4: Merge behavioral data onto structural graphs.

    STRUCTURAL_DIR has structural graph JSONs. EVENTS_DIR has JSONL event files.
    """
    from stratum_lab.overlay.cli import run_overlay
    run_overlay(structural_dir, events_dir, output_dir)


@main.command()
@click.argument("enriched_dir", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="data/pattern_knowledge_base",
              help="Output directory for knowledge base files.")
def knowledge(enriched_dir, output_dir):
    """Build cross-repo pattern knowledge base from enriched graphs.

    ENRICHED_DIR is the directory containing enriched graph JSON files.
    """
    from stratum_lab.knowledge.cli import run_knowledge_build
    run_knowledge_build(enriched_dir, output_dir)


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--vllm-url", default="http://host.docker.internal:8000/v1",
              help="vLLM endpoint URL.")
@click.option("--concurrent", default=5, help="Concurrent containers.")
@click.option("--timeout", default=600, help="Execution timeout seconds.")
def pipeline(input_dir, vllm_url, concurrent, timeout):
    """Run the full pipeline: select -> execute -> collect -> overlay -> knowledge."""
    from pathlib import Path
    from rich.console import Console

    console = Console()
    data_dir = Path("data")

    console.print("[bold]Phase 1: Repo Selection[/bold]")
    from stratum_lab.selection.cli import run_selection
    selection_file = str(data_dir / "execution_metadata" / "selection.json")
    run_selection(input_dir, selection_file, 1500, 15, 200)

    console.print("\n[bold]Phase 2: Execution[/bold]")
    from stratum_lab.harness.cli import run_execution
    events_dir = str(data_dir / "raw_events")
    run_execution(selection_file, events_dir, vllm_url, concurrent, timeout, 5, False)

    console.print("\n[bold]Phase 3: Data Collection[/bold]")
    from stratum_lab.collection.cli import run_collection
    runs_file = str(data_dir / "execution_metadata" / "runs.json")
    run_collection(events_dir, runs_file)

    console.print("\n[bold]Phase 4: Graph Overlay[/bold]")
    from stratum_lab.overlay.cli import run_overlay
    enriched_dir = str(data_dir / "enriched_graphs")
    run_overlay(input_dir, events_dir, enriched_dir)

    console.print("\n[bold]Phase 5: Knowledge Base[/bold]")
    from stratum_lab.knowledge.cli import run_knowledge_build
    kb_dir = str(data_dir / "pattern_knowledge_base")
    run_knowledge_build(enriched_dir, kb_dir)

    console.print("\n[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    main()
