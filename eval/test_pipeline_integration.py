"""Pipeline integration eval -- feedback export + pilot quality gate.

Validation checks covered:
  13: Feedback: emergent heuristics (emergent_heuristics.json written)
  14: Feedback: edge weights (edge_confidence_weights.json written)
  15: Feedback: failure catalog (failure_mode_catalog.json written)
  16: Feedback: monitoring baselines (monitoring_baselines.json written)
  17: Feedback: prediction match (prediction_match_report.json written)
  22: Pilot quality gate metrics shown

Creates 10 synthetic behavioral records via build_behavioral_record(),
calls export_feedback(), verifies all 5 output files, then exercises
_check_pilot_quality with synthetic run records.

Run:
    cd stratum-lab
    python eval/test_pipeline_integration.py
"""

from __future__ import annotations

import io
import json
import os
import random
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Output setup -- write to both terminal and file
# ---------------------------------------------------------------------------
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EVAL_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pipeline-integration-demo.txt")

_file_buf = io.StringIO()
console = Console(record=True)
file_console = Console(file=_file_buf, width=120, no_color=True)

SEPARATOR = "=" * 78


def dual_print(msg: str = "") -> None:
    """Print to both the rich console and the file buffer."""
    console.print(msg, highlight=False)
    file_console.print(msg, highlight=False)


def pretty(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


def _sanitize(text: str) -> str:
    """Replace unicode symbols that break the Windows legacy console."""
    return (
        text
        .replace("\u2713", "OK")
        .replace("\u2717", "X")
        .replace("\u26a0", "!")
        .replace("\u2714", "OK")
        .replace("\u2716", "X")
    )


# =========================================================================
# Imports under test
# =========================================================================
from stratum_lab.output.behavioral_record import build_behavioral_record
from stratum_lab.feedback.exporter import export_feedback
from stratum_lab.cli import _check_pilot_quality
from stratum_lab.config import METRIC_TO_FINDING, SCANNER_METRIC_NAMES, FINDING_NAMES

# =========================================================================
# 1. Generate 10 synthetic behavioral records
# =========================================================================

random.seed(42)

FRAMEWORKS = ["crewai", "langgraph", "autogen"]
FINDING_IDS = [
    "STRAT-DC-001",  # max_delegation_depth
    "STRAT-SI-001",  # error_swallow_rate
    "STRAT-AB-001",  # total_llm_calls_per_run
    "STRAT-OC-002",  # concurrent_state_write_rate
    "STRAT-EA-001",  # tool_call_failure_rate
]
EDGE_TYPES = ["delegates_to", "uses", "writes_to", "reads_from", "calls", "filtered_by"]
DISCOVERY_TYPES = [
    "error_triggered_fallback",
    "dynamic_delegation",
    "implicit_data_sharing",
]
VALID_DISCOVERY_TYPES = {
    "error_triggered_fallback",
    "dynamic_delegation",
    "implicit_data_sharing",
    "framework_internal_routing",
}
METRICS = [
    "max_delegation_depth",
    "error_swallow_rate",
    "total_llm_calls_per_run",
    "concurrent_state_write_rate",
    "tool_call_failure_rate",
]


def _make_edge_validation(repo_idx: int) -> dict:
    """Build a realistic edge_validation section."""
    total = 8 + repo_idx
    activated = total - random.randint(0, 3)
    dead = total - activated
    activation_rates = {}
    for etype in EDGE_TYPES:
        activation_rates[etype] = round(random.uniform(0.4, 1.0), 4)
    return {
        "structural_edges_total": total,
        "structural_edges_activated": activated,
        "structural_edges_dead": dead,
        "activation_rates": activation_rates,
    }


def _make_emergent_edges(repo_idx: int) -> list[dict]:
    """Build a list of emergent edges with varied discovery types."""
    count = random.randint(1, 4)
    edges = []
    for j in range(count):
        dtype = DISCOVERY_TYPES[j % len(DISCOVERY_TYPES)]
        edges.append({
            "source_node": f"agent_{repo_idx}_a{j}",
            "target_node": f"agent_{repo_idx}_a{j + 1}",
            "discovery_type": dtype,
            "detection_heuristic": f"Detected via {dtype} analysis on repo {repo_idx}",
            "activation_rate": round(random.uniform(0.1, 0.9), 4),
            "interaction_count": random.randint(1, 20),
        })
    return edges


def _make_node_activation(repo_idx: int) -> dict:
    """Build a realistic node_activation section."""
    match_rate = round(random.uniform(0.5, 1.0), 4)
    return {
        "total_nodes": 6 + repo_idx,
        "activated_nodes": 4 + repo_idx,
        "activation_rate": round((4 + repo_idx) / (6 + repo_idx), 4),
        "structural_prediction_match_rate": match_rate,
        "per_node": {
            f"agent_{repo_idx}_a0": {"activation_count": random.randint(3, 15)},
            f"agent_{repo_idx}_a1": {"activation_count": random.randint(1, 10)},
        },
    }


def _make_error_propagation(repo_idx: int) -> list[dict]:
    """Build diverse error propagation traces."""
    agents = [f"agent_{repo_idx}_a{j}" for j in range(3)]

    # Vary stop_mechanism and path length by repo_idx
    mechanisms = ["propagated_through", "swallowed_at_source", "retry", "absorbed"]
    error_types = ["RuntimeError", "TimeoutError", "ValueError", "SchemaError"]

    entries = []

    # First trace: longer path, varies by repo
    mech = mechanisms[repo_idx % len(mechanisms)]
    depth = (repo_idx % 3) + 1  # 1, 2, or 3
    path = agents[:depth + 1] if depth < len(agents) else agents

    entries.append({
        "trace_id": f"trace_{repo_idx:03d}_001",
        "error_source_node": agents[0],
        "error_type": error_types[repo_idx % len(error_types)],
        "structural_predicted_path": path,
        "actual_observed_path": path if mech != "swallowed_at_source" else [agents[0]],
        "propagation_stopped_by": path[-1] if mech != "propagated_through" else "",
        "stop_mechanism": mech,
        "downstream_impact": {
            "nodes_affected": 0 if mech == "swallowed_at_source" else max(0, len(path) - 1),
            "downstream_errors": 0 if mech == "swallowed_at_source" else max(0, len(path) - 1),
            "downstream_tasks_failed": 1 if mech == "propagated_through" and depth > 1 else 0,
            "cascade_depth": 0 if mech == "swallowed_at_source" else depth,
        },
        "structural_prediction_match": (mech != "swallowed_at_source"),
        "finding_id": FINDING_IDS[0],
        "run_id": f"repo_{repo_idx:03d}_run_0",
    })

    # Second trace: single-node swallowed error
    entries.append({
        "trace_id": f"trace_{repo_idx:03d}_002",
        "error_source_node": agents[1],
        "error_type": "TimeoutError",
        "structural_predicted_path": [agents[1]],
        "actual_observed_path": [agents[1]],
        "propagation_stopped_by": agents[1],
        "stop_mechanism": "swallowed_at_source",
        "downstream_impact": {
            "nodes_affected": 0,
            "downstream_errors": 0,
            "downstream_tasks_failed": 0,
            "cascade_depth": 0,
        },
        "structural_prediction_match": True,
        "finding_id": FINDING_IDS[1],
        "run_id": f"repo_{repo_idx:03d}_run_1",
    })

    return entries


def _make_failure_modes(repo_idx: int) -> list[dict]:
    """Build failure mode observations for the repo."""
    modes = []
    failure_types = ["swallowed_error", "degraded_output_propagation", "timeout_cascade", "schema_mismatch"]
    for i, fid in enumerate(FINDING_IDS):
        manifested = random.random() < 0.6
        modes.append({
            "finding_id": fid,
            "finding_name": FINDING_NAMES.get(fid, fid),
            "manifestation_observed": manifested,
            "occurrences": random.randint(1, 8) if manifested else 0,
            "failure_type": failure_types[i % len(failure_types)] if manifested else "",
            "failure_description": (
                f"Observed {fid} in repo_{repo_idx:03d} during run execution"
                if manifested else ""
            ),
        })
    return modes


def _make_monitoring_baselines(repo_idx: int) -> list[dict]:
    """Build monitoring baseline entries with canonical metric-to-finding mapping."""
    baselines = []
    for metric in METRICS:
        baselines.append({
            "metric": metric,
            "finding_id": METRIC_TO_FINDING.get(metric, "STRAT-UNKNOWN"),
            "scanner_metric": SCANNER_METRIC_NAMES.get(metric, metric),
            "observed_baseline": round(random.uniform(0.01, 1.0), 4),
            "repo": f"test-org/repo_{repo_idx:03d}",
        })
    return baselines


def build_synthetic_records(count: int = 10) -> list[dict]:
    """Build *count* synthetic behavioral records."""
    records = []
    for i in range(count):
        fw = FRAMEWORKS[i % len(FRAMEWORKS)]
        record = build_behavioral_record(
            repo_full_name=f"test-org/repo_{i:03d}",
            execution_metadata={
                "framework": fw,
                "runs_completed": 3,
                "total_events": random.randint(80, 300),
                "run_ids": [f"repo_{i:03d}_run_{r}" for r in range(3)],
            },
            edge_validation=_make_edge_validation(i),
            emergent_edges=_make_emergent_edges(i),
            node_activation=_make_node_activation(i),
            error_propagation=_make_error_propagation(i),
            failure_modes=_make_failure_modes(i),
            monitoring_baselines=_make_monitoring_baselines(i),
        )
        records.append(record)
    return records


# =========================================================================
# 2. Main evaluation
# =========================================================================

def main() -> None:
    passed = 0
    failed = 0
    tmpdir = tempfile.mkdtemp(prefix="stratum_pipeline_eval_")

    dual_print(f"Working directory: {tmpdir}\n")

    # -----------------------------------------------------------------
    # Build synthetic behavioral records
    # -----------------------------------------------------------------
    dual_print(SEPARATOR)
    dual_print("STEP 1: BUILD SYNTHETIC BEHAVIORAL RECORDS")
    dual_print(SEPARATOR)

    records = build_synthetic_records(10)
    dual_print(f"  Records created:     {len(records)}")
    dual_print(f"  Schema version:      {records[0]['schema_version']}")

    tbl = Table(title="Behavioral Records Summary")
    tbl.add_column("Repo", style="cyan")
    tbl.add_column("Framework")
    tbl.add_column("Struct Edges")
    tbl.add_column("Emergent")
    tbl.add_column("Failure Modes")
    tbl.add_column("Baselines")

    for rec in records:
        tbl.add_row(
            rec["repo_full_name"],
            rec["execution_metadata"]["framework"],
            str(rec["edge_validation"]["structural_edges_total"]),
            str(len(rec["emergent_edges"])),
            str(len(rec["failure_modes"])),
            str(len(rec["monitoring_baselines"])),
        )

    console.print(tbl)
    file_console.print(tbl)

    # Show a sample record
    dual_print(f"\n  Sample record (repo_000):")
    sample = records[0]
    dual_print(f"    edge_validation.structural_edges_total:    {sample['edge_validation']['structural_edges_total']}")
    dual_print(f"    edge_validation.structural_edges_activated:{sample['edge_validation']['structural_edges_activated']}")
    dual_print(f"    edge_validation.activation_rates keys:     {list(sample['edge_validation']['activation_rates'].keys())}")
    dual_print(f"    emergent_edges count:                      {len(sample['emergent_edges'])}")
    dual_print(f"    node_activation.structural_prediction_match_rate: {sample['node_activation']['structural_prediction_match_rate']}")
    dual_print(f"    error_propagation entries:                 {len(sample['error_propagation'])}")
    dual_print(f"    failure_modes entries:                     {len(sample['failure_modes'])}")
    dual_print(f"    monitoring_baselines entries:              {len(sample['monitoring_baselines'])}")

    # -----------------------------------------------------------------
    # Export feedback
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("STEP 2: EXPORT FEEDBACK")
    dual_print(SEPARATOR)

    feedback_dir = os.path.join(tmpdir, "feedback")
    files_written = export_feedback(records, feedback_dir)

    dual_print(f"  Output directory:    {feedback_dir}")
    dual_print(f"  Files written:       {len(files_written)}")
    for fname, fpath in files_written.items():
        size = os.path.getsize(fpath)
        dual_print(f"    {fname:40s} {size:>6,} bytes")

    # -----------------------------------------------------------------
    # Check 13: emergent_heuristics.json
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 13: emergent_heuristics.json")
    dual_print(SEPARATOR)

    eh_path = os.path.join(feedback_dir, "emergent_heuristics.json")
    assert os.path.isfile(eh_path), "emergent_heuristics.json not written"

    with open(eh_path, "r", encoding="utf-8") as f:
        eh_data = json.load(f)

    has_total = "total_emergent_edges" in eh_data
    has_by_type = "by_discovery_type" in eh_data
    has_heuristics = "heuristics" in eh_data and len(eh_data["heuristics"]) > 0
    by_type = eh_data.get("by_discovery_type", {})
    grouped_ok = all(dt in by_type for dt in DISCOVERY_TYPES)

    check_13_pass = has_total and has_by_type and has_heuristics and grouped_ok

    dual_print(f"  File exists:                  YES")
    dual_print(f"  Has total_emergent_edges:      {has_total}")
    dual_print(f"  Has by_discovery_type:          {has_by_type}")
    dual_print(f"  Has non-empty heuristics list:  {has_heuristics} (count={len(eh_data.get('heuristics', []))})")
    dual_print(f"  Grouped by discovery_type:      {grouped_ok}")
    dual_print(f"    by_discovery_type:             {by_type}")

    dual_print(f"\n  Sample heuristics (first 3):")
    for h in eh_data.get("heuristics", [])[:3]:
        dual_print(f"    {h['repo']}: {h['source_node']} -> {h['target_node']} "
                   f"[{h['discovery_type']}] rate={h['activation_rate']}")

    if check_13_pass:
        passed += 1
        dual_print(f"\n  CHECK 13: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 13: FAIL")

    # -----------------------------------------------------------------
    # Check 24: by_discovery_type keys are subset of valid taxonomy
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 24: DISCOVERY TYPE TAXONOMY VALID")
    dual_print(SEPARATOR)

    by_type_keys = set(by_type.keys())
    taxonomy_valid = by_type_keys.issubset(VALID_DISCOVERY_TYPES)

    dual_print(f"  by_discovery_type keys:  {by_type_keys}")
    dual_print(f"  Valid taxonomy set:       {VALID_DISCOVERY_TYPES}")
    dual_print(f"  Is subset:                {taxonomy_valid}")

    check_24_pass = taxonomy_valid and len(by_type_keys) > 0
    if check_24_pass:
        passed += 1
        dual_print(f"\n  CHECK 24: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 24: FAIL")

    # -----------------------------------------------------------------
    # Check 25: All 5 feedback files have model_context.model_tier
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 25: MODEL CONTEXT IN ALL FEEDBACK FILES")
    dual_print(SEPARATOR)

    check_25_pass = True
    for fname in [
        "emergent_heuristics.json",
        "edge_confidence_weights.json",
        "failure_mode_catalog.json",
        "monitoring_baselines.json",
        "prediction_match_report.json",
    ]:
        fpath = os.path.join(feedback_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        mc = data.get("model_context", {})
        has_mt = isinstance(mc, dict) and "model_tier" in mc
        dual_print(f"  {fname:40s} model_context.model_tier: {has_mt}")
        if not has_mt:
            check_25_pass = False

    if check_25_pass:
        passed += 1
        dual_print(f"\n  CHECK 25: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 25: FAIL")

    # -----------------------------------------------------------------
    # Check 14: edge_confidence_weights.json
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 14: edge_confidence_weights.json")
    dual_print(SEPARATOR)

    ecw_path = os.path.join(feedback_dir, "edge_confidence_weights.json")
    assert os.path.isfile(ecw_path), "edge_confidence_weights.json not written"

    with open(ecw_path, "r", encoding="utf-8") as f:
        ecw_data = json.load(f)

    has_edge_types = "edge_types" in ecw_data
    has_repos_analyzed = "repos_analyzed" in ecw_data
    edge_types_data = ecw_data.get("edge_types", {})
    has_per_type_rates = all(
        "mean_activation_rate" in v and "sample_size" in v
        for v in edge_types_data.values()
    ) if edge_types_data else False

    check_14_pass = has_edge_types and has_repos_analyzed and has_per_type_rates and len(edge_types_data) > 0

    dual_print(f"  File exists:                  YES")
    dual_print(f"  Has edge_types:                {has_edge_types}")
    dual_print(f"  Has repos_analyzed:             {has_repos_analyzed} (value={ecw_data.get('repos_analyzed')})")
    dual_print(f"  Per-type activation rates:      {has_per_type_rates}")
    dual_print(f"  Edge types found:               {len(edge_types_data)}")

    dual_print(f"\n  Per-type weights:")
    for etype, weights in edge_types_data.items():
        dual_print(f"    {etype:20s}  mean={weights['mean_activation_rate']:.4f}  "
                   f"min={weights['min']:.4f}  max={weights['max']:.4f}  n={weights['sample_size']}")

    if check_14_pass:
        passed += 1
        dual_print(f"\n  CHECK 14: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 14: FAIL")

    # -----------------------------------------------------------------
    # Check 15: failure_mode_catalog.json
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 15: failure_mode_catalog.json")
    dual_print(SEPARATOR)

    fmc_path = os.path.join(feedback_dir, "failure_mode_catalog.json")
    assert os.path.isfile(fmc_path), "failure_mode_catalog.json not written"

    with open(fmc_path, "r", encoding="utf-8") as f:
        fmc_data = json.load(f)

    catalog = fmc_data.get("catalog", fmc_data.get("findings", []))
    has_findings = len(catalog) > 0
    has_examples = any(len(f.get("examples", [])) > 0 for f in catalog)

    check_15_pass = has_findings and has_examples

    dual_print(f"  File exists:                  YES")
    dual_print(f"  Has non-empty catalog:         {has_findings} (count={len(catalog)})")
    dual_print(f"  Has examples:                  {has_examples}")
    dual_print(f"  repos_analyzed:                {fmc_data.get('repos_analyzed')}")

    dual_print(f"\n  Failure mode catalog:")
    for finding in catalog:
        examples_count = len(finding.get("examples", []))
        dual_print(f"    {finding['finding_id']:40s}  repos_affected={finding.get('repos_affected', 0)}  "
                   f"total_manifestations={finding.get('total_manifestations', 0)}  examples={examples_count}")

    if check_15_pass:
        passed += 1
        dual_print(f"\n  CHECK 15: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 15: FAIL")

    # -----------------------------------------------------------------
    # Check 19: Catalog example has error_propagation.structural_predicted_path
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 19: CATALOG EXAMPLE ERROR PROPAGATION")
    dual_print(SEPARATOR)

    has_ep_trace = False
    for finding in catalog:
        for example in finding.get("examples", []):
            ep = example.get("error_propagation", {})
            if ep and isinstance(ep, dict) and ep.get("structural_predicted_path"):
                has_ep_trace = True
                dual_print(f"  Found in {finding['finding_id']}: "
                           f"path={ep['structural_predicted_path']}")
                break
        if has_ep_trace:
            break

    check_19_pass = has_ep_trace
    if check_19_pass:
        passed += 1
        dual_print(f"\n  CHECK 19: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 19: FAIL -- no catalog example has error_propagation.structural_predicted_path")

    # -----------------------------------------------------------------
    # Check 20: Catalog has metadata.model_tier + metadata.caveat
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 20: CATALOG METADATA")
    dual_print(SEPARATOR)

    catalog_meta = fmc_data.get("metadata", {})
    has_model_tier = "model_tier" in catalog_meta
    has_caveat = "caveat" in catalog_meta

    dual_print(f"  metadata.model_tier: {has_model_tier} (value={catalog_meta.get('model_tier')})")
    dual_print(f"  metadata.caveat:     {has_caveat}")
    if has_caveat:
        dual_print(f"    caveat: {catalog_meta.get('caveat', '')[:100]}...")

    check_20_pass = has_model_tier and has_caveat

    # Also verify finding_names are human-readable (not raw IDs)
    bad_names = []
    for entry in catalog:
        fname = entry.get("finding_name", "")
        if fname.startswith("STRAT-") or len(fname) <= 5:
            bad_names.append(f"{entry.get('finding_id')}: '{fname}'")
    if bad_names:
        check_20_pass = False
        dual_print(f"  finding_name human-readable:  False")
        for bn in bad_names[:3]:
            dual_print(f"    BAD: {bn}")
    else:
        dual_print(f"  finding_name human-readable:  True")

    if check_20_pass:
        passed += 1
        dual_print(f"\n  CHECK 20: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 20: FAIL")

    # -----------------------------------------------------------------
    # Check 16: monitoring_baselines.json
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 16: monitoring_baselines.json")
    dual_print(SEPARATOR)

    mb_path = os.path.join(feedback_dir, "monitoring_baselines.json")
    assert os.path.isfile(mb_path), "monitoring_baselines.json not written"

    with open(mb_path, "r", encoding="utf-8") as f:
        mb_data = json.load(f)

    has_baselines = "baselines" in mb_data and len(mb_data["baselines"]) > 0
    baselines = mb_data.get("baselines", [])
    has_per_finding_agg = all(
        "mean_baseline" in b and "sample_repos" in b and "finding_id" in b
        for b in baselines
    ) if baselines else False

    check_16_pass = has_baselines and has_per_finding_agg

    dual_print(f"  File exists:                  YES")
    dual_print(f"  Has non-empty baselines list:  {has_baselines} (count={len(baselines)})")
    dual_print(f"  Has per-finding aggregates:    {has_per_finding_agg}")
    dual_print(f"  repos_analyzed:                {mb_data.get('repos_analyzed')}")

    dual_print(f"\n  Monitoring baselines:")
    for bl in baselines:
        dual_print(f"    {bl['metric']:45s}  mean={bl['mean_baseline']:>10.4f}  "
                   f"sample_repos={bl['sample_repos']}  finding={bl['finding_id']}")

    if check_16_pass:
        passed += 1
        dual_print(f"\n  CHECK 16: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 16: FAIL")

    # -----------------------------------------------------------------
    # Check 23: ALL finding_ids in catalog + baselines start with "STRAT-"
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 23: STRAT- PREFIX ON ALL FINDING IDS")
    dual_print(SEPARATOR)

    all_fids_valid = True
    bad_fids = []

    for finding in catalog:
        fid = finding.get("finding_id", "")
        if not fid.startswith("STRAT-"):
            all_fids_valid = False
            bad_fids.append(f"catalog: {fid}")

    for bl in baselines:
        fid = bl.get("finding_id", "")
        if fid and not fid.startswith("STRAT-"):
            all_fids_valid = False
            bad_fids.append(f"baseline: {fid}")

    dual_print(f"  All finding_ids start with STRAT-: {all_fids_valid}")
    if bad_fids:
        for bf in bad_fids[:5]:
            dual_print(f"    BAD: {bf}")

    check_23_pass = all_fids_valid
    if check_23_pass:
        passed += 1
        dual_print(f"\n  CHECK 23: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 23: FAIL")

    # -----------------------------------------------------------------
    # Check 17: prediction_match_report.json
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 17: prediction_match_report.json")
    dual_print(SEPARATOR)

    pmr_path = os.path.join(feedback_dir, "prediction_match_report.json")
    assert os.path.isfile(pmr_path), "prediction_match_report.json not written"

    with open(pmr_path, "r", encoding="utf-8") as f:
        pmr_data = json.load(f)

    has_overall = "overall_edge_activation_rate" in pmr_data
    has_per_cat = "mean_node_prediction_match_rate" in pmr_data
    has_totals = (
        "total_structural_edges" in pmr_data
        and "activated_edges" in pmr_data
        and "dead_edges" in pmr_data
    )

    check_17_pass = has_overall and has_per_cat and has_totals

    dual_print(f"  File exists:                         YES")
    dual_print(f"  Has overall_edge_activation_rate:      {has_overall} (value={pmr_data.get('overall_edge_activation_rate')})")
    dual_print(f"  Has mean_node_prediction_match_rate:   {has_per_cat} (value={pmr_data.get('mean_node_prediction_match_rate')})")
    dual_print(f"  Has total/activated/dead edges:         {has_totals}")
    dual_print(f"    total_structural_edges:               {pmr_data.get('total_structural_edges')}")
    dual_print(f"    activated_edges:                      {pmr_data.get('activated_edges')}")
    dual_print(f"    dead_edges:                           {pmr_data.get('dead_edges')}")
    dual_print(f"    repos_analyzed:                       {pmr_data.get('repos_analyzed')}")

    if check_17_pass:
        passed += 1
        dual_print(f"\n  CHECK 17: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 17: FAIL")

    # -----------------------------------------------------------------
    # Check 22: Pilot quality gate
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("CHECK 22: PILOT QUALITY GATE")
    dual_print(SEPARATOR)

    # Build synthetic run records with a mix of statuses
    random.seed(99)
    pilot_statuses = [
        "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS",
        "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS",
        "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS",
        "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS",
        "PARTIAL_SUCCESS", "PARTIAL_SUCCESS",
        "INSTRUMENTATION_FAILURE", "INSTRUMENTATION_FAILURE",
        "MODEL_FAILURE",
        "TIMEOUT",
        "CRASH",
        "DEPENDENCY_FAILURE",
        "SUCCESS", "SUCCESS",
    ]
    pilot_run_records = [
        {"run_id": f"pilot_run_{i:03d}", "status": s}
        for i, s in enumerate(pilot_statuses)
    ]

    dual_print(f"  Synthetic pilot runs:          {len(pilot_run_records)}")

    # Count statuses for display
    from collections import Counter
    status_counts = Counter(r["status"] for r in pilot_run_records)
    for status, count in sorted(status_counts.items()):
        dual_print(f"    {status:30s}  {count}")

    # Test case A: should PASS (low failure rates)
    dual_print(f"\n  --- Test A: thresholds instr=20%, model=15% (expect PASS) ---")
    thresholds_pass = {"instr": 0.20, "model": 0.15}

    # Capture _check_pilot_quality output
    old_stdout = sys.stdout
    capture_a = io.StringIO()
    sys.stdout = capture_a
    result_a = _check_pilot_quality(pilot_run_records, thresholds_pass)
    sys.stdout = old_stdout
    gate_output_a = capture_a.getvalue()

    dual_print(_sanitize(gate_output_a.rstrip()))
    dual_print(f"  _check_pilot_quality returned: {result_a}")

    # Test case B: should FAIL (very tight thresholds)
    dual_print(f"\n  --- Test B: thresholds instr=1%, model=1% (expect FAIL) ---")
    thresholds_fail = {"instr": 0.01, "model": 0.01}

    capture_b = io.StringIO()
    sys.stdout = capture_b
    result_b = _check_pilot_quality(pilot_run_records, thresholds_fail)
    sys.stdout = old_stdout
    gate_output_b = capture_b.getvalue()

    dual_print(_sanitize(gate_output_b.rstrip()))
    dual_print(f"  _check_pilot_quality returned: {result_b}")

    # Test case C: empty run records (should pass by default)
    dual_print(f"\n  --- Test C: empty run records (expect PASS) ---")
    capture_c = io.StringIO()
    sys.stdout = capture_c
    result_c = _check_pilot_quality([], {"instr": 0.20, "model": 0.15})
    sys.stdout = old_stdout
    gate_output_c = capture_c.getvalue()

    dual_print(_sanitize(gate_output_c.rstrip()))
    dual_print(f"  _check_pilot_quality returned: {result_c}")

    check_22_pass = result_a is True and result_b is False and result_c is True

    if check_22_pass:
        passed += 1
        dual_print(f"\n  CHECK 22: PASS")
    else:
        failed += 1
        dual_print(f"\n  CHECK 22: FAIL")
        dual_print(f"    result_a (expected True):  {result_a}")
        dual_print(f"    result_b (expected False): {result_b}")
        dual_print(f"    result_c (expected True):  {result_c}")

    # -----------------------------------------------------------------
    # Full file content dump
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("FEEDBACK FILE CONTENTS (FULL)")
    dual_print(SEPARATOR)

    for fname in [
        "emergent_heuristics.json",
        "edge_confidence_weights.json",
        "failure_mode_catalog.json",
        "monitoring_baselines.json",
        "prediction_match_report.json",
    ]:
        fpath = os.path.join(feedback_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = json.load(f)
        dual_print(f"\n--- {fname} ---")
        dual_print(pretty(content))

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    dual_print(f"\n{SEPARATOR}")
    dual_print("PIPELINE INTEGRATION EVAL SUMMARY")
    dual_print(SEPARATOR)

    tbl_summary = Table(title="Validation Check Results")
    tbl_summary.add_column("Check", style="bold")
    tbl_summary.add_column("Description")
    tbl_summary.add_column("Result")

    check_results = [
        ("13", "Feedback: emergent heuristics", check_13_pass),
        ("14", "Feedback: edge weights", check_14_pass),
        ("15", "Feedback: failure catalog", check_15_pass),
        ("16", "Feedback: monitoring baselines", check_16_pass),
        ("17", "Feedback: prediction match", check_17_pass),
        ("19", "Catalog example has error_propagation", check_19_pass),
        ("20", "Catalog metadata (model_tier + caveat)", check_20_pass),
        ("22", "Pilot quality gate metrics", check_22_pass),
        ("23", "STRAT- prefix on all finding IDs", check_23_pass),
        ("24", "Discovery type taxonomy valid", check_24_pass),
        ("25", "Model context in all feedback files", check_25_pass),
    ]

    for check_id, desc, ok in check_results:
        result_str = "PASS" if ok else "FAIL"
        tbl_summary.add_row(check_id, desc, result_str)

    console.print(tbl_summary)
    file_console.print(tbl_summary)

    dual_print(f"\n  Total:  {passed} passed, {failed} failed, {passed + failed} checks")
    dual_print(f"  RESULT: {'ALL CHECKS PASSED' if failed == 0 else 'SOME CHECKS FAILED'}")

    # -----------------------------------------------------------------
    # Clean up
    # -----------------------------------------------------------------
    shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Write output file
    # -----------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(_file_buf.getvalue())
    dual_print(f"\n  Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
