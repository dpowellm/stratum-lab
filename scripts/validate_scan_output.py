#!/usr/bin/env python3
"""Post-scan validation script.

Validates that all expected output artifacts from a stratum-lab pipeline run
are present, well-formed, and internally consistent.

Usage:
    python scripts/validate_scan_output.py ./results
    python scripts/validate_scan_output.py ./results --min-repos 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate(output_dir: str, min_repos: int = 100) -> bool:
    """Validate scan output artifacts.

    Returns True if all checks pass.
    """
    root = Path(output_dir)
    passed = 0
    failed = 0
    warnings = 0

    def check(label: str, condition: bool, detail: str = "") -> bool:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {label}")
        else:
            failed += 1
            msg = f"  [FAIL] {label}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
        return condition

    def warn(label: str, detail: str = "") -> None:
        nonlocal warnings
        warnings += 1
        msg = f"  [WARN] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)

    print("=" * 70)
    print(f"POST-SCAN VALIDATION: {root}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Directory structure
    # ------------------------------------------------------------------
    print("\n--- Directory Structure ---")
    check("Output directory exists", root.is_dir())

    expected_dirs = ["feedback", "reports", "enriched_graphs", "execution_metadata"]
    for d in expected_dirs:
        check(f"  {d}/ exists", (root / d).is_dir(), f"missing: {root / d}")

    # ------------------------------------------------------------------
    # 2. Feedback files
    # ------------------------------------------------------------------
    print("\n--- Feedback Artifacts ---")
    feedback_dir = root / "feedback"
    feedback_files = [
        "emergent_heuristics.json",
        "edge_confidence_weights.json",
        "failure_mode_catalog.json",
        "monitoring_baselines.json",
        "prediction_match_report.json",
    ]

    for fname in feedback_files:
        fpath = feedback_dir / fname
        exists = fpath.is_file()
        check(f"  {fname} exists", exists)
        if exists:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    data = json.load(fh)
                check(f"  {fname} valid JSON", True)

                # All feedback files should have model_context
                has_mc = isinstance(data.get("model_context"), dict)
                check(f"  {fname} has model_context", has_mc)
            except json.JSONDecodeError as exc:
                check(f"  {fname} valid JSON", False, str(exc))

    # ------------------------------------------------------------------
    # 3. Failure mode catalog specifics
    # ------------------------------------------------------------------
    print("\n--- Failure Mode Catalog ---")
    fmc_path = feedback_dir / "failure_mode_catalog.json"
    if fmc_path.is_file():
        with open(fmc_path, encoding="utf-8") as fh:
            fmc = json.load(fh)

        catalog = fmc.get("catalog", fmc.get("findings", []))
        check("Catalog has entries", len(catalog) > 0, f"count={len(catalog)}")

        # All finding_ids should be STRAT-prefixed
        bad_ids = [e["finding_id"] for e in catalog if not e.get("finding_id", "").startswith("STRAT-")]
        check("All finding_ids STRAT-prefixed", len(bad_ids) == 0,
              f"bad: {bad_ids[:3]}" if bad_ids else "")

        # finding_name should be human-readable
        bad_names = [e["finding_id"] for e in catalog
                     if e.get("finding_name", "").startswith("STRAT-") or len(e.get("finding_name", "")) <= 5]
        check("All finding_names human-readable", len(bad_names) == 0,
              f"bad: {bad_names[:3]}" if bad_names else "")

        # Metadata
        meta = fmc.get("metadata", {})
        check("Catalog has metadata.model_tier", "model_tier" in meta)
        check("Catalog has metadata.caveat", "caveat" in meta)

    # ------------------------------------------------------------------
    # 4. Monitoring baselines specifics
    # ------------------------------------------------------------------
    print("\n--- Monitoring Baselines ---")
    mb_path = feedback_dir / "monitoring_baselines.json"
    if mb_path.is_file():
        with open(mb_path, encoding="utf-8") as fh:
            mb = json.load(fh)

        baselines = mb.get("baselines", [])
        check("Baselines has entries", len(baselines) > 0, f"count={len(baselines)}")

        # All baselines should have finding_id with STRAT prefix
        bad_bl = [b["metric"] for b in baselines if not b.get("finding_id", "").startswith("STRAT-")]
        check("All baseline finding_ids STRAT-prefixed", len(bad_bl) == 0,
              f"bad: {bad_bl[:3]}" if bad_bl else "")

    # ------------------------------------------------------------------
    # 5. Behavioral reports
    # ------------------------------------------------------------------
    print("\n--- Behavioral Reports ---")
    reports_dir = root / "reports"
    if reports_dir.is_dir():
        reports = list(reports_dir.glob("*_behavioral.json"))
        check(f"Behavioral reports exist", len(reports) > 0, f"count={len(reports)}")
        check(f"At least {min_repos} reports", len(reports) >= min_repos,
              f"got {len(reports)}, expected >= {min_repos}")

        # Validate a sample report
        if reports:
            sample = reports[0]
            try:
                with open(sample, encoding="utf-8") as fh:
                    record = json.load(fh)
                check("Sample report has schema_version", record.get("schema_version") == "v6",
                      f"got: {record.get('schema_version')}")
                check("Sample report has error_propagation", isinstance(record.get("error_propagation"), list))
                check("Sample report has failure_modes", isinstance(record.get("failure_modes"), list))
            except Exception as exc:
                check("Sample report valid", False, str(exc))
    else:
        check("Reports directory exists", False)

    # ------------------------------------------------------------------
    # 6. Prediction match report
    # ------------------------------------------------------------------
    print("\n--- Prediction Match Report ---")
    pmr_path = feedback_dir / "prediction_match_report.json"
    if pmr_path.is_file():
        with open(pmr_path, encoding="utf-8") as fh:
            pmr = json.load(fh)
        check("Has overall_edge_activation_rate", "overall_edge_activation_rate" in pmr)
        check("Has mean_node_prediction_match_rate", "mean_node_prediction_match_rate" in pmr)
        check("repos_analyzed > 0", pmr.get("repos_analyzed", 0) > 0,
              f"repos_analyzed={pmr.get('repos_analyzed')}")

    # ------------------------------------------------------------------
    # 7. Checkpoint file
    # ------------------------------------------------------------------
    print("\n--- Checkpoint ---")
    ckpt = root / "scan_checkpoint.json"
    if ckpt.is_file():
        with open(ckpt, encoding="utf-8") as fh:
            ckpt_data = json.load(fh)
        completed = len(ckpt_data.get("completed_repos", []))
        check(f"Checkpoint has {completed} completed repos", completed > 0)
    else:
        warn("No checkpoint file found (expected if scan completed without interruption)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Warnings: {warnings}")

    if failed == 0:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {failed} CHECK(S) FAILED")

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stratum-lab scan output")
    parser.add_argument("output_dir", help="Path to pipeline output directory")
    parser.add_argument("--min-repos", type=int, default=100,
                        help="Minimum expected behavioral reports")
    args = parser.parse_args()

    ok = validate(args.output_dir, args.min_repos)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
