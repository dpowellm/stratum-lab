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
@click.option("--max-concurrent", default=5, type=int,
              help="Max concurrent container executions.")
@click.option("--timeout", default=600, help="Per-execution timeout in seconds.")
@click.option("--runs", default=5, help="Runs per repo (3 diverse + 2 repeat).")
@click.option("--dry-run", is_flag=True, help="Print what would be executed without running.")
def execute(selection_file, output_dir, vllm_url, concurrent, max_concurrent,
            timeout, runs, dry_run):
    """Phase 2: Execute selected repos in sandboxed containers.

    SELECTION_FILE is the JSON file from the 'select' phase.
    """
    from stratum_lab.harness.cli import run_execution
    run_execution(selection_file, output_dir, vllm_url, max_concurrent, timeout, runs, dry_run)


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
@click.argument("structural_graph", type=click.Path(exists=True))
@click.option("--knowledge-base", "-kb", type=click.Path(exists=True), required=True,
              help="Path to knowledge base directory.")
@click.option("--output-format", type=click.Choice(["json", "markdown"]), default="json",
              help="Output format.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path (stdout if not specified).")
def query(structural_graph, knowledge_base, output_format, output):
    """Query the behavioral dataset with a structural graph to predict risks."""
    import json
    from pathlib import Path

    from stratum_lab.query.fingerprint import compute_graph_fingerprint
    from stratum_lab.query.matcher import match_against_dataset
    from stratum_lab.query.predictor import predict_risks
    from stratum_lab.query.report import generate_risk_report

    # Load structural graph
    with open(structural_graph, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # Compute fingerprint
    fingerprint = compute_graph_fingerprint(graph)

    # Match against dataset
    kb_path = Path(knowledge_base)
    matches = match_against_dataset(fingerprint, kb_path)

    # Extract taxonomy preconditions from graph
    preconditions = list(graph.get("taxonomy_preconditions", []))
    for node_data in graph.get("nodes", {}).values():
        structural = node_data.get("structural", node_data)
        for pc in structural.get("taxonomy_preconditions", []):
            if pc not in preconditions:
                preconditions.append(pc)

    # Predict risks
    prediction = predict_risks(graph, matches, preconditions, kb_path)

    # Generate report
    report = generate_risk_report(prediction, graph, output_format)

    # Output
    if output:
        with open(output, "w", encoding="utf-8") as f:
            if isinstance(report, dict):
                json.dump(report, f, indent=2, default=str)
            else:
                f.write(report)
    else:
        if isinstance(report, dict):
            click.echo(json.dumps(report, indent=2, default=str))
        else:
            click.echo(report)


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--vllm-url", default="http://host.docker.internal:8000/v1",
              help="vLLM endpoint URL.")
@click.option("--concurrent", default=5, help="Concurrent containers.")
@click.option("--max-concurrent", default=5, type=int,
              help="Max concurrent container executions.")
@click.option("--timeout", default=600, help="Execution timeout seconds.")
@click.option("--pilot", is_flag=True,
              help="Run pilot batch with quality gate before full scan.")
@click.option("--pilot-size", default=20, type=int,
              help="Number of repos in pilot batch.")
@click.option("--max-instrumentation-failure-rate", default=0.3, type=float,
              help="Abort if instrumentation failure rate exceeds this in pilot.")
@click.option("--max-model-failure-rate", default=0.4, type=float,
              help="Abort if model failure rate exceeds this in pilot.")
@click.option("--enterprise-outreach", is_flag=True, default=False,
              help="Run enterprise classification + contact extraction + outreach queue generation.")
@click.option("--triage/--no-triage", default=True,
              help="Run Phase 0 static triage before selection.")
@click.option("--probe/--no-probe", default=True,
              help="Run Phase 0.5 probe execution before selection.")
@click.option("--probe-batch-size", default=5000, type=int,
              help="Max repos to probe (from triage pool).")
@click.option("--probe-timeout", default=30, type=int,
              help="Probe timeout in seconds.")
def pipeline(input_dir, vllm_url, concurrent, max_concurrent, timeout,
             pilot, pilot_size, max_instrumentation_failure_rate, max_model_failure_rate,
             enterprise_outreach, triage, probe, probe_batch_size, probe_timeout):
    """Run the full pipeline: triage -> probe -> select -> execute -> collect -> overlay -> knowledge -> reports."""
    import json
    from pathlib import Path
    from rich.console import Console

    console = Console()
    data_dir = Path("data")

    # Phase 0: Static Triage
    if triage:
        console.print("[bold]Phase 0: Static Triage[/bold]")
        from stratum_lab.triage.static_analyzer import triage_batch
        # Load structural scans from input_dir
        structural_scans = _load_structural_scans(input_dir)
        triage_results = triage_batch(structural_scans)
        qualified = (
            triage_results["likely_runnable"]
            + triage_results["needs_probe"][:probe_batch_size]
        )
        console.print(
            f"  {triage_results['total_analyzed']} scanned -> "
            f"{len(qualified)} qualified for probing"
        )
    else:
        qualified = None  # Will use normal selection flow

    # Phase 0.5: Probe Execution
    if probe and qualified is not None:
        console.print("[bold]Phase 0.5: Probe Execution[/bold]")
        from stratum_lab.triage.probe import probe_batch as run_probe_batch
        probe_results = run_probe_batch(qualified, timeout=probe_timeout)
        console.print(
            f"  {probe_results['total_probed']} probed -> "
            f"{probe_results['passed']} passed ({probe_results['pass_rate']:.0%})"
        )

    console.print("[bold]Phase 1: Repo Selection[/bold]")
    from stratum_lab.selection.cli import run_selection
    selection_file = str(data_dir / "execution_metadata" / "selection.json")
    target = pilot_size if pilot else 1500
    run_selection(input_dir, selection_file, target, 15, 200)

    if pilot:
        console.print(f"\n[bold yellow]PILOT MODE: using first {pilot_size} repos[/bold yellow]")

    console.print("\n[bold]Phase 2: Execution[/bold]")
    from stratum_lab.harness.cli import run_execution
    events_dir = str(data_dir / "raw_events")
    run_execution(selection_file, events_dir, vllm_url,
                  max_concurrent, timeout, 5, False)

    if pilot:
        # Quality gate after execution
        console.print("\n[bold]Pilot Quality Gate[/bold]")
        import json
        meta_dir = data_dir / "execution_metadata"
        runs_csv = meta_dir / "runs.csv"
        run_records = []
        if runs_csv.exists():
            import csv
            with open(runs_csv, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                run_records = list(reader)
        passed = _check_pilot_quality(
            run_records,
            {
                "instr": max_instrumentation_failure_rate,
                "model": max_model_failure_rate,
            },
        )
        if not passed:
            console.print("[bold red]Pilot failed. Aborting pipeline.[/bold red]")
            return

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

    console.print("\n[bold]Phase 6: Per-Repo Risk Reports[/bold]")
    from stratum_lab.query.batch_report import generate_batch_reports
    reports_dir = str(data_dir / "reports")
    report_summary = generate_batch_reports(
        enriched_graphs_dir=Path(enriched_dir),
        knowledge_base_dir=Path(kb_dir),
        output_dir=Path(reports_dir),
    )
    console.print(f"  Reports generated: {report_summary['reports_generated']}")
    console.print(f"  Errors: {report_summary['errors']}")

    console.print("\n[bold]Phase 6.5: Dataset Quality Validation[/bold]")
    from stratum_lab.knowledge.dataset_quality import validate_dataset_quality
    # Load enriched graphs and run records for quality check
    enriched_graph_list = _load_enriched_graphs(enriched_dir)
    run_record_list = _load_run_records(data_dir / "execution_metadata" / "runs.json")
    quality = validate_dataset_quality(enriched_graph_list, run_record_list, {})
    console.print(f"  Verdict: {quality['verdict']}")
    console.print(
        f"  Usable runs: {quality['usable_runs']}/{quality['total_runs']} "
        f"({quality['usable_run_rate']:.0%})"
    )
    claims_met = sum(
        1 for c in quality["per_claim_quality"].values() if c.get("sufficient", False)
    )
    console.print(
        f"  Claims supported: {claims_met}/{len(quality['per_claim_quality'])}"
    )

    if enterprise_outreach:
        console.print("\n[bold]Phase 1.5: GitHub Metadata Enrichment[/bold]")
        console.print("  (Requires GITHUB_TOKEN env var for API access)")
        from stratum_lab.outreach.github_enricher import GitHubEnricher
        enricher_gh = GitHubEnricher()
        metadata_dir = data_dir / "metadata"
        # Load selected repos
        with open(selection_file, "r", encoding="utf-8") as fh:
            sel_data = json.load(fh)
        sel_repos = sel_data.get("selections", sel_data.get("repos", []))
        enrichment = enricher_gh.enrich_batch(
            repos=sel_repos, output_dir=metadata_dir,
        )
        console.print(f"  Enriched: {enrichment['enriched']}/{enrichment['total']}")

        console.print("\n[bold]Phase 7: Enterprise Outreach[/bold]")
        from stratum_lab.outreach.enterprise_classifier import classify_batch
        from stratum_lab.outreach.contact_extractor import extract_contacts
        from stratum_lab.outreach.teaser_report import generate_teaser
        from stratum_lab.outreach.queue import build_outreach_queue, save_queue

        # Load metadata files
        repo_metadata_by_id = {}
        for md_file in metadata_dir.glob("*_metadata.json"):
            with open(md_file) as fh:
                md = json.load(fh)
            repo_metadata_by_id[md.get("repo_id", md_file.stem)] = md

        enterprise_results = classify_batch(list(repo_metadata_by_id.values()))
        console.print(f"  Enterprise repos: {enterprise_results['enterprise_count']}/{enterprise_results['total_repos']}")

        all_contacts = {}
        for repo in enterprise_results["classifications"]:
            if repo["is_enterprise"]:
                rid = repo.get("repo_id", "")
                if rid in repo_metadata_by_id:
                    all_contacts[rid] = extract_contacts(repo_metadata_by_id[rid])

        teasers = {}
        for repo_id in all_contacts:
            report_path = Path(reports_dir) / f"{repo_id}_report.json"
            if report_path.exists():
                with open(report_path) as fh:
                    full_report = json.load(fh)
                teasers[repo_id] = generate_teaser(full_report, repo_metadata_by_id.get(repo_id, {}))

        enterprise_list = [r for r in enterprise_results["classifications"] if r["is_enterprise"]]
        all_reports = {}
        for r in enterprise_list:
            rp = Path(reports_dir) / f"{r['repo_id']}_report.json"
            if rp.exists():
                with open(rp) as fh:
                    all_reports[r["repo_id"]] = json.load(fh)

        queue = build_outreach_queue(enterprise_list, all_reports, all_contacts, teasers)
        outreach_dir = data_dir / "outreach"
        save_queue(queue, outreach_dir / "outreach_queue.json")
        console.print(f"  Outreach queue: {len(queue)} records")

    console.print("\n[bold green]Pipeline complete.[/bold green]")


def _check_pilot_quality(run_records, thresholds):
    """Check failure classification distribution from pilot batch.

    Parameters
    ----------
    run_records:
        List of run record dicts with a 'status' field.
    thresholds:
        Dict with 'instr' and 'model' rate thresholds.

    Returns
    -------
    bool
        True if pilot passed quality gate.
    """
    from collections import Counter

    if not run_records:
        print("\nNo run records found for pilot quality gate.")
        return True

    classifications = Counter(r.get("status", "UNKNOWN") for r in run_records)
    total = len(run_records)

    instr_rate = classifications.get("INSTRUMENTATION_FAILURE", 0) / total
    model_rate = classifications.get("MODEL_FAILURE", 0) / total
    timeout_rate = classifications.get("TIMEOUT", 0) / total
    success_rate = (
        classifications.get("SUCCESS", 0) + classifications.get("PARTIAL_SUCCESS", 0)
    ) / total

    print(f"\n{'='*60}")
    print(f"PILOT QUALITY GATE ({total} runs)")
    print(f"{'='*60}")
    print(f"  SUCCESS + PARTIAL:       {success_rate:.0%}")
    print(f"  INSTRUMENTATION_FAILURE: {instr_rate:.0%}  (threshold: {thresholds['instr']:.0%})")
    print(f"  MODEL_FAILURE:           {model_rate:.0%}  (threshold: {thresholds['model']:.0%})")
    print(f"  TIMEOUT:                 {timeout_rate:.0%}")
    other_rate = 1 - success_rate - instr_rate - model_rate - timeout_rate
    print(f"  Other:                   {other_rate:.0%}")

    issues = []
    if instr_rate > thresholds["instr"]:
        issues.append(
            f"Instrumentation failure rate {instr_rate:.0%} exceeds {thresholds['instr']:.0%}"
        )
    if model_rate > thresholds["model"]:
        issues.append(
            f"Model failure rate {model_rate:.0%} exceeds {thresholds['model']:.0%}"
        )

    if issues:
        print(f"\n⚠ PILOT FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\nRecommendation: Fix patcher/vLLM before running full scan.")
        return False
    else:
        print(f"\n✓ PILOT PASSED: proceeding to full scan.")
        return True


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--graph-json", type=click.Path(exists=True),
              help="Structural graph JSON from stratum-cli.")
@click.option("--events-dir", type=click.Path(exists=True),
              help="Directory of JSONL event files from patcher.")
def validate_ids(repo_path, graph_json, events_dir):
    """Validate node ID compatibility between structural scan and patcher."""
    from pathlib import Path
    from stratum_lab.validation.id_checker import load_structural_ids, load_event_ids, compare_ids

    structural = load_structural_ids(Path(graph_json)) if graph_json else set()
    behavioral = load_event_ids(Path(events_dir)) if events_dir else set()

    result = compare_ids(structural, behavioral)
    click.echo(f"Match rate: {result['match_rate']:.0%}")
    click.echo(f"  Matched: {result['matched_count']}")
    click.echo(f"  Structural only: {result['structural_only_count']}")
    click.echo(f"  Behavioral only: {result['behavioral_only_count']}")
    if not result["compatible"]:
        click.echo("\n⚠ INCOMPATIBLE: behavioral events reference nodes not in structural graph.")
        click.echo("  Fix node_ids.py or stratum-cli before running mass scan.")
        for nid in result["behavioral_only"][:10]:
            click.echo(f"    unmapped: {nid}")
    else:
        click.echo("\n✓ COMPATIBLE: all behavioral node IDs map to structural graph nodes.")


def _load_structural_scans(input_dir: str) -> list:
    """Load structural scan JSON files from a directory."""
    import json
    from pathlib import Path

    scans = []
    for p in Path(input_dir).glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                scans.append(json.load(f))
        except Exception:
            pass
    return scans


def _load_enriched_graphs(enriched_dir: str) -> list:
    """Load enriched graph JSON files from a directory."""
    import json
    from pathlib import Path

    graphs = []
    for p in Path(enriched_dir).glob("*_enriched.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                graphs.append(json.load(f))
        except Exception:
            pass
    return graphs


def _load_run_records(runs_file) -> list:
    """Load run records from a JSON file."""
    import json
    from pathlib import Path

    path = Path(runs_file)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("records", data.get("runs", []))
    except Exception:
        return []


if __name__ == "__main__":
    main()
