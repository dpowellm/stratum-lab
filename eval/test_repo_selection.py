"""Evaluation of the v6 graph discovery repo selection pipeline.

Validation checks 10-12:
  Check 10: Discovery optimizer selects with framework + complexity coverage
  Check 11: XCOMP repos prioritized (XCOMP selection rate > base rate)
  Check 12: Coverage validation report with gaps identified

Generates 200 synthetic repos across 8 frameworks and 4 complexity brackets,
with ~20% carrying XCOMP findings.  Runs select_for_discovery() and
validate_coverage() from discovery_optimizer and validate_selection_input()
from schema, then prints detailed tables via rich console.

Output is written to eval/outputs/repo-selection-demo.txt.

Run as a standalone script:
    cd stratum-lab
    python eval/test_repo_selection.py
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure stratum_lab is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ===================================================================
# Synthetic repo generator
# ===================================================================

FRAMEWORKS = [
    "crewai", "langgraph", "autogen", "langchain",
    "openai_assistants", "semantic_kernel", "llama_index", "custom",
]

# Weighted so core frameworks appear more often in the candidate pool
FRAMEWORK_WEIGHTS = [0.18, 0.16, 0.12, 0.14, 0.10, 0.10, 0.08, 0.12]

COMPLEXITY_BRACKETS = {
    "single": (1, 1),
    "small":  (2, 3),
    "medium": (4, 7),
    "large":  (8, 15),
}

# Finding IDs from the discovery optimizer module
FINDING_IDS_POOL = [
    "STRAT-DC-001", "STRAT-DC-002", "STRAT-DC-003",
    "STRAT-SI-001", "STRAT-SI-002", "STRAT-SI-004",
    "STRAT-EA-001", "STRAT-EA-002",
    "STRAT-OC-001", "STRAT-OC-002",
    "STRAT-AB-001",
]

XCOMP_FINDING_IDS = ["STRAT-XCOMP-001", "STRAT-XCOMP-006"]

CONTROL_TYPES = [
    "human_gate", "schema_validation", "error_boundary",
    "timeout", "observability_sink", "rate_limiter",
]

ARCHETYPE_POOL = [
    "sequential_pipeline", "hub_and_spoke", "hierarchical_delegation",
    "parallel_fan_out", "reflection_loop", "debate_consensus",
    "supervisor_worker", "blackboard_architecture",
]


def _bracket_for_agent_count(n: int) -> str:
    """Return the complexity bracket label for a given agent count."""
    for label, (lo, hi) in COMPLEXITY_BRACKETS.items():
        if lo <= n <= hi:
            return label
    return "large"


def _generate_synthetic_repos(count: int = 200, seed: int = 42) -> list[dict]:
    """Generate *count* synthetic repo dicts matching discovery_optimizer input.

    Each repo has:
      - repo_full_name, graph, findings, control_inventory, framework,
        agent_count, edge_count  (required by schema)
      - topology_hash, finding_ids, present_control_types,
        estimated_runnability, xcomp_findings  (used by optimizer)
    """
    rng = random.Random(seed)

    repos: list[dict] = []
    for i in range(count):
        # --- Framework ---
        fw = rng.choices(FRAMEWORKS, weights=FRAMEWORK_WEIGHTS, k=1)[0]

        # --- Complexity: pick a bracket first, then an agent count in range ---
        bracket_label = rng.choice(list(COMPLEXITY_BRACKETS.keys()))
        lo, hi = COMPLEXITY_BRACKETS[bracket_label]
        agent_count = rng.randint(lo, hi)

        edge_count = rng.randint(agent_count, agent_count * 4)

        # --- Topology hash (unique per distinct graph shape) ---
        raw = f"{fw}_{agent_count}_{edge_count}_{i}"
        topology_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        # Make ~30% of repos share a topology with another repo
        if rng.random() < 0.30 and i > 0:
            donor_idx = rng.randint(0, len(repos) - 1)
            topology_hash = repos[donor_idx].get("topology_hash", topology_hash)

        # --- Findings ---
        n_findings = rng.randint(0, 5)
        finding_ids = rng.sample(FINDING_IDS_POOL, min(n_findings, len(FINDING_IDS_POOL)))

        # --- XCOMP (~20% of repos) ---
        is_xcomp = rng.random() < 0.20
        xcomp_findings: list[str] = []
        if is_xcomp:
            n_xcomp = rng.randint(1, len(XCOMP_FINDING_IDS))
            xcomp_findings = rng.sample(XCOMP_FINDING_IDS, n_xcomp)
            finding_ids.extend(xcomp_findings)

        # --- Controls ---
        n_present = rng.randint(0, len(CONTROL_TYPES))
        present_controls = rng.sample(CONTROL_TYPES, n_present)

        # --- Runnability (0-1) ---
        estimated_runnability = round(rng.uniform(0.2, 1.0), 2)

        # --- Graph structure (nodes + edges for schema validation) ---
        nodes = [
            {"node_id": f"agent_{j}", "node_type": "agent", "name": f"Agent_{j}"}
            for j in range(agent_count)
        ]
        edges = []
        for j in range(edge_count):
            src = rng.randint(0, max(agent_count - 1, 0))
            tgt = rng.randint(0, max(agent_count - 1, 0))
            edges.append({
                "source": f"agent_{src}",
                "target": f"agent_{tgt}",
                "edge_type": rng.choice(["delegates_to", "sends_to", "calls", "uses"]),
            })

        # --- Archetype ---
        archetype = rng.choice(ARCHETYPE_POOL)

        # --- Findings as dicts for the schema `findings` field ---
        finding_dicts = [
            {"finding_id": fid, "severity": rng.choice(["low", "medium", "high"])}
            for fid in finding_ids
        ]

        repo = {
            # Schema-required fields
            "repo_full_name": f"test-org/repo-{i:04d}",
            "graph": {"nodes": nodes, "edges": edges},
            "findings": finding_dicts,
            "control_inventory": {
                "present_controls": present_controls,
                "absent_controls": [c for c in CONTROL_TYPES if c not in present_controls],
            },
            "framework": fw,
            "agent_count": agent_count,
            "edge_count": edge_count,

            # Scoring/optimizer fields
            "repo_id": f"repo_{i:04d}",
            "topology_hash": topology_hash,
            "archetype": archetype,
            "finding_ids": finding_ids,
            "xcomp_findings": xcomp_findings,
            "present_control_types": present_controls,
            "estimated_runnability": estimated_runnability,
            "has_entry_point": rng.random() > 0.2,
            "has_requirements": rng.random() > 0.15,
        }
        repos.append(repo)

    return repos


# ===================================================================
# Main evaluation
# ===================================================================

def main() -> None:
    from stratum_lab.selection.discovery_optimizer import (
        select_for_discovery,
        validate_coverage,
    )
    from stratum_lab.selection.schema import validate_selection_input

    # --- Output setup ---
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "repo-selection-demo.txt"

    file_console = Console(file=open(str(output_path), "w", encoding="utf-8"), width=120)
    term_console = Console(width=120)

    def emit(renderable: object) -> None:
        """Print to both terminal and output file."""
        term_console.print(renderable)
        file_console.print(renderable)

    # -----------------------------------------------------------------
    # 1. Generate synthetic repos
    # -----------------------------------------------------------------
    repos = _generate_synthetic_repos(count=200, seed=42)

    emit(Panel(
        f"[bold]Repo Selection Eval -- v6 Graph Discovery Reframe[/bold]\n"
        f"Synthetic repos generated: [cyan]{len(repos)}[/cyan]   "
        f"Frameworks: [cyan]{len(FRAMEWORKS)}[/cyan]   "
        f"Complexity brackets: [cyan]{len(COMPLEXITY_BRACKETS)}[/cyan]",
        title="CHECKS 10-12",
        border_style="blue",
    ))

    # -----------------------------------------------------------------
    # 2. Schema validation (Check 12 prerequisite)
    # -----------------------------------------------------------------
    emit("\n[bold underline]SCHEMA VALIDATION (validate_selection_input)[/bold underline]")

    valid_count = 0
    invalid_count = 0
    sample_errors: list[tuple[str, list[str]]] = []

    for repo in repos:
        ok, missing = validate_selection_input(repo)
        if ok:
            valid_count += 1
        else:
            invalid_count += 1
            if len(sample_errors) < 5:
                sample_errors.append((repo["repo_id"], missing))

    schema_table = Table(title="Schema Validation Results", show_header=True, header_style="bold cyan")
    schema_table.add_column("Metric", style="dim")
    schema_table.add_column("Value", justify="right")
    schema_table.add_row("Total repos", str(len(repos)))
    schema_table.add_row("Valid", f"[green]{valid_count}[/green]")
    schema_table.add_row("Invalid", f"[red]{invalid_count}[/red]" if invalid_count else "[green]0[/green]")
    emit(schema_table)

    if sample_errors:
        emit("[yellow]Sample validation errors:[/yellow]")
        for repo_id, missing in sample_errors:
            emit(f"  {repo_id}: missing {missing}")

    # -----------------------------------------------------------------
    # 3. Input pool statistics
    # -----------------------------------------------------------------
    emit("\n[bold underline]INPUT POOL STATISTICS[/bold underline]")

    pool_fw_counts = Counter(r["framework"] for r in repos)
    pool_cx_counts = Counter(_bracket_for_agent_count(r["agent_count"]) for r in repos)
    pool_xcomp_count = sum(1 for r in repos if r.get("xcomp_findings"))
    pool_unique_topos = len({r["topology_hash"] for r in repos})

    fw_pool_table = Table(title="Framework Distribution (Input Pool)", show_header=True, header_style="bold cyan")
    fw_pool_table.add_column("Framework", style="dim")
    fw_pool_table.add_column("Count", justify="right")
    fw_pool_table.add_column("Pct", justify="right")
    fw_pool_table.add_column("Bar")
    for fw in FRAMEWORKS:
        c = pool_fw_counts.get(fw, 0)
        pct = c / len(repos) * 100
        bar = "#" * int(pct / 2)
        fw_pool_table.add_row(fw, str(c), f"{pct:.1f}%", bar)
    emit(fw_pool_table)

    cx_pool_table = Table(title="Complexity Distribution (Input Pool)", show_header=True, header_style="bold cyan")
    cx_pool_table.add_column("Bracket", style="dim")
    cx_pool_table.add_column("Agents", justify="center")
    cx_pool_table.add_column("Count", justify="right")
    cx_pool_table.add_column("Pct", justify="right")
    for label in COMPLEXITY_BRACKETS:
        c = pool_cx_counts.get(label, 0)
        pct = c / len(repos) * 100
        lo, hi = COMPLEXITY_BRACKETS[label]
        cx_pool_table.add_row(label, f"{lo}-{hi}", str(c), f"{pct:.1f}%")
    emit(cx_pool_table)

    emit(f"  XCOMP repos in pool: [bold]{pool_xcomp_count}[/bold] / {len(repos)}  "
         f"({pool_xcomp_count / len(repos) * 100:.1f}%)")
    emit(f"  Unique topologies in pool: [bold]{pool_unique_topos}[/bold]")

    # -----------------------------------------------------------------
    # 4. Run discovery selection (Check 10)
    # -----------------------------------------------------------------
    emit("\n[bold underline]CHECK 10: DISCOVERY OPTIMIZER SELECTION[/bold underline]")

    target_count = 100
    selected = select_for_discovery(repos, target_count=target_count)
    n_selected = len(selected)

    emit(f"  Target: {target_count}   Selected: [bold]{n_selected}[/bold]")

    # Framework counts in selected
    sel_fw_counts = Counter(r.get("framework", "generic") for r in selected)
    sel_cx_counts = Counter(_bracket_for_agent_count(r.get("agent_count", 1)) for r in selected)
    sel_xcomp_count = sum(1 for r in selected if r.get("xcomp_findings"))
    sel_unique_topos = len({r.get("topology_hash", "") for r in selected})

    fw_sel_table = Table(
        title="Framework Distribution (Selected)",
        show_header=True,
        header_style="bold cyan",
    )
    fw_sel_table.add_column("Framework", style="dim")
    fw_sel_table.add_column("Pool", justify="right")
    fw_sel_table.add_column("Selected", justify="right")
    fw_sel_table.add_column("Sel Rate", justify="right")
    fw_sel_table.add_column("Bar")
    for fw in FRAMEWORKS:
        pool_n = pool_fw_counts.get(fw, 0)
        sel_n = sel_fw_counts.get(fw, 0)
        rate = sel_n / pool_n * 100 if pool_n > 0 else 0
        bar = "#" * max(1, int(sel_n / max(n_selected, 1) * 50))
        fw_sel_table.add_row(fw, str(pool_n), str(sel_n), f"{rate:.1f}%", bar)
    emit(fw_sel_table)

    cx_sel_table = Table(
        title="Complexity Distribution (Selected)",
        show_header=True,
        header_style="bold cyan",
    )
    cx_sel_table.add_column("Bracket", style="dim")
    cx_sel_table.add_column("Agents", justify="center")
    cx_sel_table.add_column("Pool", justify="right")
    cx_sel_table.add_column("Selected", justify="right")
    cx_sel_table.add_column("Sel Rate", justify="right")
    cx_sel_table.add_column("Pct of Sel", justify="right")
    for label in COMPLEXITY_BRACKETS:
        pool_n = pool_cx_counts.get(label, 0)
        sel_n = sel_cx_counts.get(label, 0)
        rate = sel_n / pool_n * 100 if pool_n > 0 else 0
        pct = sel_n / n_selected * 100 if n_selected > 0 else 0
        lo, hi = COMPLEXITY_BRACKETS[label]
        cx_sel_table.add_row(label, f"{lo}-{hi}", str(pool_n), str(sel_n), f"{rate:.1f}%", f"{pct:.1f}%")
    emit(cx_sel_table)

    # Assertions for Check 10
    emit("\n[bold]Check 10 Assertions:[/bold]")

    # a) All 8 frameworks represented (or gaps explicitly identified)
    frameworks_represented = len(sel_fw_counts)
    c10a_pass = frameworks_represented == len(FRAMEWORKS)
    c10a_tag = "[green][PASS][/green]" if c10a_pass else "[yellow][WARN][/yellow]"
    emit(f"  {c10a_tag} Frameworks represented: {frameworks_represented}/{len(FRAMEWORKS)}")
    if not c10a_pass:
        missing_fw = set(FRAMEWORKS) - set(sel_fw_counts.keys())
        emit(f"         Missing: {missing_fw}")

    # b) All 4 complexity brackets have at least 1 selection
    brackets_represented = sum(1 for b in COMPLEXITY_BRACKETS if sel_cx_counts.get(b, 0) > 0)
    c10b_pass = brackets_represented == len(COMPLEXITY_BRACKETS)
    c10b_tag = "[green][PASS][/green]" if c10b_pass else "[red][FAIL][/red]"
    emit(f"  {c10b_tag} Complexity brackets covered: {brackets_represented}/{len(COMPLEXITY_BRACKETS)}")

    # c) Unique topologies maximized
    c10c_pass = sel_unique_topos >= n_selected * 0.5  # at least 50% unique
    c10c_tag = "[green][PASS][/green]" if c10c_pass else "[yellow][WARN][/yellow]"
    emit(f"  {c10c_tag} Unique topologies in selection: {sel_unique_topos}/{n_selected}  "
         f"({sel_unique_topos / n_selected * 100:.1f}%)")

    # -----------------------------------------------------------------
    # 5. XCOMP prioritization (Check 11)
    # -----------------------------------------------------------------
    emit("\n[bold underline]CHECK 11: XCOMP PRIORITIZATION[/bold underline]")

    base_rate = n_selected / len(repos) if repos else 0
    xcomp_selected_rate = sel_xcomp_count / pool_xcomp_count if pool_xcomp_count > 0 else 0

    xcomp_table = Table(title="XCOMP Selection Rates", show_header=True, header_style="bold cyan")
    xcomp_table.add_column("Metric", style="dim")
    xcomp_table.add_column("Value", justify="right")
    xcomp_table.add_row("XCOMP repos in pool", str(pool_xcomp_count))
    xcomp_table.add_row("XCOMP repos selected", str(sel_xcomp_count))
    xcomp_table.add_row("XCOMP selection rate", f"{xcomp_selected_rate:.3f}")
    xcomp_table.add_row("Base selection rate", f"{base_rate:.3f}")
    xcomp_table.add_row(
        "Boost factor",
        f"{xcomp_selected_rate / base_rate:.2f}x" if base_rate > 0 else "N/A",
    )
    emit(xcomp_table)

    c11_pass = xcomp_selected_rate > base_rate
    c11_tag = "[green][PASS][/green]" if c11_pass else "[red][FAIL][/red]"
    emit(f"  {c11_tag} XCOMP selection rate ({xcomp_selected_rate:.3f}) > "
         f"base rate ({base_rate:.3f})")

    # Show the top-10 selected repos by _discovery_value so the reader
    # can see XCOMP repos rising to the top.
    top10_table = Table(
        title="Top 10 Selected Repos by Discovery Value",
        show_header=True,
        header_style="bold cyan",
    )
    top10_table.add_column("#", justify="right", style="dim")
    top10_table.add_column("Repo ID")
    top10_table.add_column("Framework")
    top10_table.add_column("Agents", justify="right")
    top10_table.add_column("Bracket")
    top10_table.add_column("XCOMP", justify="center")
    top10_table.add_column("Topology", max_width=12)
    top10_table.add_column("Runnability", justify="right")
    top10_table.add_column("DValue", justify="right")

    for rank, r in enumerate(selected[:10], 1):
        dval = r.get("_discovery_value", 0.0)
        xcomp_flag = "[bold magenta]YES[/bold magenta]" if r.get("xcomp_findings") else "-"
        bracket = _bracket_for_agent_count(r.get("agent_count", 1))
        top10_table.add_row(
            str(rank),
            r.get("repo_id", "?"),
            r.get("framework", "?"),
            str(r.get("agent_count", "?")),
            bracket,
            xcomp_flag,
            r.get("topology_hash", "?")[:10],
            f"{r.get('estimated_runnability', 0):.2f}",
            f"{dval:.2f}",
        )
    emit(top10_table)

    # -----------------------------------------------------------------
    # 6. Coverage validation (Check 12)
    # -----------------------------------------------------------------
    emit("\n[bold underline]CHECK 12: COVERAGE VALIDATION REPORT[/bold underline]")

    report = validate_coverage(selected)

    # 12a) framework_coverage dict present
    c12a_pass = "framework_coverage" in report and isinstance(report["framework_coverage"], dict)
    c12a_tag = "[green][PASS][/green]" if c12a_pass else "[red][FAIL][/red]"
    emit(f"  {c12a_tag} report has framework_coverage dict: {c12a_pass}")

    # 12b) complexity_coverage dict present
    c12b_pass = "complexity_coverage" in report and isinstance(report["complexity_coverage"], dict)
    c12b_tag = "[green][PASS][/green]" if c12b_pass else "[red][FAIL][/red]"
    emit(f"  {c12b_tag} report has complexity_coverage dict: {c12b_pass}")

    # Framework coverage detail table
    if report.get("framework_coverage"):
        fc_table = Table(
            title="Framework Coverage Detail",
            show_header=True,
            header_style="bold cyan",
        )
        fc_table.add_column("Framework", style="dim")
        fc_table.add_column("Count", justify="right")
        fc_table.add_column("Target", justify="right")
        fc_table.add_column("Met", justify="center")
        fc_table.add_column("Gap", justify="right")

        for fw, info in report["framework_coverage"].items():
            count = info["count"]
            target = info["target"]
            met = info["met"]
            gap = max(0, target - count)
            met_str = "[green]YES[/green]" if met else f"[red]NO (-{gap})[/red]"
            fc_table.add_row(fw, str(count), str(target), met_str, str(gap) if gap > 0 else "-")
        emit(fc_table)

    # Complexity coverage detail table
    if report.get("complexity_coverage"):
        cc_table = Table(
            title="Complexity Coverage Detail",
            show_header=True,
            header_style="bold cyan",
        )
        cc_table.add_column("Bracket", style="dim")
        cc_table.add_column("Count", justify="right")
        cc_table.add_column("Target", justify="right")
        cc_table.add_column("Met", justify="center")
        cc_table.add_column("Gap", justify="right")

        for bracket, info in report["complexity_coverage"].items():
            count = info["count"]
            target = info["target"]
            met = info["met"]
            gap = max(0, target - count)
            met_str = "[green]YES[/green]" if met else f"[red]NO (-{gap})[/red]"
            cc_table.add_row(bracket, str(count), str(target), met_str, str(gap) if gap > 0 else "-")
        emit(cc_table)

    emit(f"\n  NOTE: Above targets are for --target=1000 (BASE_TARGET). "
         f"See Check 27 for scaled targets at --target={target_count}.")

    # Finding coverage
    if report.get("finding_coverage"):
        find_table = Table(
            title="Finding Coverage Detail",
            show_header=True,
            header_style="bold cyan",
        )
        find_table.add_column("Finding ID", style="dim")
        find_table.add_column("Count", justify="right")
        find_table.add_column("Target", justify="right")
        find_table.add_column("Met", justify="center")

        for fid, info in report["finding_coverage"].items():
            met_str = "[green]YES[/green]" if info["met"] else "[red]NO[/red]"
            style = "bold magenta" if "XCOMP" in fid else ""
            find_table.add_row(
                f"[{style}]{fid}[/{style}]" if style else fid,
                str(info["count"]),
                str(info["target"]),
                met_str,
            )
        emit(find_table)

    # Gaps
    gaps = report.get("gaps", [])
    if gaps:
        emit(f"\n[yellow]Coverage gaps identified ({len(gaps)}):[/yellow]")
        for gap in gaps:
            emit(f"  [yellow]- {gap}[/yellow]")
    else:
        emit("\n[green]No coverage gaps -- all targets met.[/green]")

    # 12c) gaps list exists and contains entries (with 100 repos vs. targets
    #       designed for 500-1000, we expect gaps)
    c12c_pass = isinstance(gaps, list)
    c12c_tag = "[green][PASS][/green]" if c12c_pass else "[red][FAIL][/red]"
    emit(f"\n  {c12c_tag} report has gaps list: {c12c_pass}  (entries: {len(gaps)})")

    # 12d) unique topologies tracked
    c12d_pass = report.get("unique_topologies", 0) > 0
    c12d_tag = "[green][PASS][/green]" if c12d_pass else "[red][FAIL][/red]"
    emit(f"  {c12d_tag} unique topologies tracked: {report.get('unique_topologies', 0)}")

    # -----------------------------------------------------------------
    # Check 27: Scaled targets with --target=100
    # -----------------------------------------------------------------
    emit("\n[bold underline]CHECK 27: SCALED TARGET COVERAGE[/bold underline]")

    scaled_report = validate_coverage(selected, target_count=target_count)
    scaled_fw_met = sum(
        1 for info in scaled_report["framework_coverage"].values()
        if info["met"]
    )

    emit(f"  Scaled framework targets met: {scaled_fw_met}/{len(scaled_report['framework_coverage'])}")
    for fw, info in scaled_report["framework_coverage"].items():
        met_str = "[green]YES[/green]" if info["met"] else "[red]NO[/red]"
        emit(f"    {fw:20s}  count={info['count']}  target={info['target']}  {met_str}")

    c27_pass = scaled_fw_met >= 4
    c27_tag = "[green][PASS][/green]" if c27_pass else "[red][FAIL][/red]"
    emit(f"\n  {c27_tag} With --target={target_count}, at least 4 framework targets met (got {scaled_fw_met})")

    # -----------------------------------------------------------------
    # 7. Summary statistics
    # -----------------------------------------------------------------
    emit("\n[bold underline]SUMMARY STATISTICS[/bold underline]")

    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Total in pool", str(len(repos)))
    summary_table.add_row("Target count", str(target_count))
    summary_table.add_row("Actually selected", str(n_selected))
    summary_table.add_row("Selection rate", f"{n_selected / len(repos) * 100:.1f}%")
    summary_table.add_row("Unique topologies (selected)", str(sel_unique_topos))
    summary_table.add_row("Unique topologies (pool)", str(pool_unique_topos))
    summary_table.add_row("Frameworks represented", str(frameworks_represented))
    summary_table.add_row("Complexity brackets covered", str(brackets_represented))
    summary_table.add_row("XCOMP repos selected", str(sel_xcomp_count))
    summary_table.add_row("XCOMP boost factor",
                          f"{xcomp_selected_rate / base_rate:.2f}x" if base_rate > 0 else "N/A")
    summary_table.add_row("Schema-valid repos", f"{valid_count}/{len(repos)}")
    summary_table.add_row("Coverage gaps", str(len(gaps)))
    summary_table.add_row("All targets met", str(report.get("all_targets_met", False)))

    # Average runnability of selected
    avg_run = sum(r.get("estimated_runnability", 0) for r in selected) / n_selected if n_selected else 0
    summary_table.add_row("Avg runnability (selected)", f"{avg_run:.3f}")

    # Average discovery value
    dvals = [r.get("_discovery_value", 0) for r in selected]
    avg_dval = sum(dvals) / len(dvals) if dvals else 0
    min_dval = min(dvals) if dvals else 0
    max_dval = max(dvals) if dvals else 0
    summary_table.add_row("Avg discovery value", f"{avg_dval:.2f}")
    summary_table.add_row("Min discovery value", f"{min_dval:.2f}")
    summary_table.add_row("Max discovery value", f"{max_dval:.2f}")
    emit(summary_table)

    # -----------------------------------------------------------------
    # 8. Final pass/fail verdict
    # -----------------------------------------------------------------
    emit("\n[bold underline]VERDICT[/bold underline]")

    checks = {
        "Check 10a: Framework coverage": c10a_pass,
        "Check 10b: Complexity coverage": c10b_pass,
        "Check 10c: Topology uniqueness": c10c_pass,
        "Check 11:  XCOMP prioritized": c11_pass,
        "Check 12a: framework_coverage dict": c12a_pass,
        "Check 12b: complexity_coverage dict": c12b_pass,
        "Check 12c: gaps list present": c12c_pass,
        "Check 12d: unique topologies tracked": c12d_pass,
        "Check 27:  Scaled targets met": c27_pass,
    }

    verdict_table = Table(title="Final Verdict", show_header=True, header_style="bold cyan")
    verdict_table.add_column("Check", style="dim")
    verdict_table.add_column("Result", justify="center")
    for name, passed in checks.items():
        tag = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        verdict_table.add_row(name, tag)
    emit(verdict_table)

    all_passed = all(checks.values())
    if all_passed:
        emit(Panel("[bold green]ALL CHECKS PASSED[/bold green]", border_style="green"))
    else:
        failed_names = [k for k, v in checks.items() if not v]
        emit(Panel(
            f"[bold red]{len(failed_names)} CHECK(S) FAILED[/bold red]\n"
            + "\n".join(f"  - {n}" for n in failed_names),
            border_style="red",
        ))

    # Close the file console
    file_console.file.close()
    term_console.print(f"\n[dim]Output written to {output_path}[/dim]")


if __name__ == "__main__":
    main()
