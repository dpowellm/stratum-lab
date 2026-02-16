#!/usr/bin/env python3
"""Post-scan validation script.

Checks that scan output meets minimum quality bars across behavioral records,
feedback files, finding IDs, metric naming, and checkpoint state.

Usage:
    python scripts/validate_scan.py ./data/scan-output
    python scripts/validate_scan.py ./data/scan-output --strict  (exits 1 on warnings)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing stratum_lab
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stratum_lab.output.behavioral_record import validate_behavioral_record

# ---------------------------------------------------------------------------
# Feedback files that must be present and non-empty
# ---------------------------------------------------------------------------
FEEDBACK_FILES = [
    "failure_mode_catalog.json",
    "monitoring_baselines.json",
    "error_propagation_catalog.json",
    "remediation_playbook.json",
    "behavioral_summary.json",
]


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def validate_scan(scan_dir: str, strict: bool = False) -> bool:
    """Run all validation checks against *scan_dir*.

    Returns True when the scan should be considered acceptable (exit 0).
    """
    root = Path(scan_dir)

    results: list[tuple[int, str, str]] = []  # (check_num, status, label)
    check_num = 0

    def _record(status: str, label: str) -> str:
        nonlocal check_num
        check_num += 1
        results.append((check_num, status, label))
        return status

    # ==================================================================
    # CHECK 1: Behavioral records directory exists and has files
    # ==================================================================
    br_dir = root / "behavioral_records"
    if not br_dir.is_dir():
        br_dir = root / "enriched_graphs"
    br_files: list[Path] = []
    if br_dir.is_dir():
        br_files = [f for f in br_dir.iterdir() if f.suffix == ".json"]

    if br_dir.is_dir() and len(br_files) > 0:
        _record("PASS", f"Behavioral records directory exists and has files ({len(br_files)} found)")
    else:
        _record("FAIL", "Behavioral records directory exists and has files")

    # ==================================================================
    # CHECK 2: Each behavioral record passes validate_behavioral_record()
    # ==================================================================
    bad_records: list[str] = []
    for bf in br_files:
        data = _load_json(bf)
        if data is None or not isinstance(data, dict):
            bad_records.append(bf.name)
            continue
        ok, errors = validate_behavioral_record(data)
        if not ok:
            bad_records.append(bf.name)

    if len(br_files) > 0 and len(bad_records) == 0:
        _record("PASS", "All behavioral records pass validation")
    elif len(br_files) == 0:
        _record("FAIL", "All behavioral records pass validation (no records found)")
    else:
        _record("FAIL", f"All behavioral records pass validation ({len(bad_records)} failed: {', '.join(bad_records[:5])})")

    # ==================================================================
    # CHECK 3: All 5 feedback files exist and are non-empty
    # ==================================================================
    feedback_dir = root / "feedback"
    missing_feedback: list[str] = []
    empty_feedback: list[str] = []
    feedback_data: dict[str, dict | list] = {}

    for fname in FEEDBACK_FILES:
        fpath = feedback_dir / fname
        if not fpath.is_file():
            missing_feedback.append(fname)
            continue
        if fpath.stat().st_size == 0:
            empty_feedback.append(fname)
            continue
        data = _load_json(fpath)
        if data is None:
            empty_feedback.append(fname)
        else:
            feedback_data[fname] = data

    problems = missing_feedback + empty_feedback
    if len(problems) == 0:
        _record("PASS", "All 5 feedback files exist and are non-empty")
    else:
        detail_parts = []
        if missing_feedback:
            detail_parts.append(f"missing: {', '.join(missing_feedback)}")
        if empty_feedback:
            detail_parts.append(f"empty/invalid: {', '.join(empty_feedback)}")
        _record("FAIL", f"All 5 feedback files exist and are non-empty ({'; '.join(detail_parts)})")

    # ==================================================================
    # CHECK 4: Failure catalog has >=1 finding with >=1 example
    # ==================================================================
    fmc = feedback_data.get("failure_mode_catalog.json")
    if isinstance(fmc, dict):
        catalog = fmc.get("catalog", fmc.get("findings", []))
        has_example = any(
            len(entry.get("examples", entry.get("sample_repos", []))) > 0
            for entry in catalog
            if isinstance(entry, dict)
        )
        if len(catalog) >= 1 and has_example:
            _record("PASS", f"Failure catalog has >=1 finding with >=1 example ({len(catalog)} findings)")
        elif len(catalog) == 0:
            _record("FAIL", "Failure catalog has >=1 finding with >=1 example (0 findings)")
        else:
            _record("FAIL", "Failure catalog has >=1 finding with >=1 example (no examples found)")
    else:
        _record("FAIL", "Failure catalog has >=1 finding with >=1 example (file not loaded)")

    # ==================================================================
    # CHECK 5: Monitoring baselines have >=1 metric with sample_repos > 0
    # ==================================================================
    mb = feedback_data.get("monitoring_baselines.json")
    if isinstance(mb, dict):
        baselines = mb.get("baselines", mb.get("metrics", []))
        has_samples = any(
            (entry.get("sample_repos", 0) if isinstance(entry.get("sample_repos"), int)
             else len(entry.get("sample_repos", []))) > 0
            for entry in baselines
            if isinstance(entry, dict)
        )
        if len(baselines) >= 1 and has_samples:
            _record("PASS", f"Monitoring baselines have >=1 metric with sample_repos > 0 ({len(baselines)} metrics)")
        elif len(baselines) == 0:
            _record("FAIL", "Monitoring baselines have >=1 metric with sample_repos > 0 (0 metrics)")
        else:
            _record("FAIL", "Monitoring baselines have >=1 metric with sample_repos > 0 (no metrics with samples)")
    else:
        _record("FAIL", "Monitoring baselines have >=1 metric with sample_repos > 0 (file not loaded)")

    # ==================================================================
    # CHECK 6: All finding IDs start with STRAT-
    # ==================================================================
    all_finding_ids: list[str] = []
    for fname, data in feedback_data.items():
        if not isinstance(data, dict):
            continue
        for key in ("catalog", "findings", "baselines", "metrics", "chains", "entries"):
            entries = data.get(key, [])
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and "finding_id" in entry:
                        all_finding_ids.append(entry["finding_id"])

    bad_ids = [fid for fid in all_finding_ids if not fid.startswith("STRAT-")]
    if len(all_finding_ids) > 0 and len(bad_ids) == 0:
        _record("PASS", f"All finding IDs start with STRAT- ({len(all_finding_ids)} checked)")
    elif len(all_finding_ids) == 0:
        _record("WARN", "All finding IDs start with STRAT- (no finding IDs found to check)")
    else:
        _record("FAIL", f"All finding IDs start with STRAT- ({len(bad_ids)} invalid: {', '.join(bad_ids[:5])})")

    # ==================================================================
    # CHECK 7: All finding names are human-readable (not STRAT-xxx, len > 5)
    # ==================================================================
    all_finding_names: list[tuple[str, str]] = []  # (finding_id, finding_name)
    for fname, data in feedback_data.items():
        if not isinstance(data, dict):
            continue
        for key in ("catalog", "findings", "baselines", "metrics", "chains", "entries"):
            entries = data.get(key, [])
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and "finding_name" in entry:
                        all_finding_names.append(
                            (entry.get("finding_id", "?"), entry["finding_name"])
                        )

    bad_names = [
        (fid, name) for fid, name in all_finding_names
        if name.startswith("STRAT-") or len(name) <= 5
    ]
    if len(all_finding_names) > 0 and len(bad_names) == 0:
        _record("PASS", f"All finding names are human-readable ({len(all_finding_names)} checked)")
    elif len(all_finding_names) == 0:
        _record("WARN", "All finding names are human-readable (no finding names found to check)")
    else:
        examples = ", ".join(f"{fid}='{name}'" for fid, name in bad_names[:3])
        _record("FAIL", f"All finding names are human-readable ({len(bad_names)} invalid: {examples})")

    # ==================================================================
    # CHECK 8: Scanner metric names don't have scanner_ prefix
    # ==================================================================
    metric_names: list[str] = []
    if isinstance(mb, dict):
        baselines = mb.get("baselines", mb.get("metrics", []))
        for entry in baselines:
            if isinstance(entry, dict) and "metric" in entry:
                metric_names.append(entry["metric"])

    bad_metrics = [m for m in metric_names if m.startswith("scanner_")]
    if len(metric_names) > 0 and len(bad_metrics) == 0:
        _record("PASS", f"Scanner metric names don't have scanner_ prefix ({len(metric_names)} checked)")
    elif len(metric_names) == 0:
        _record("WARN", "Scanner metric names don't have scanner_ prefix (no metrics found to check)")
    else:
        _record("FAIL", f"Scanner metric names don't have scanner_ prefix ({len(bad_metrics)} invalid: {', '.join(bad_metrics[:5])})")

    # ==================================================================
    # CHECK 9: Model context present in all feedback files
    # ==================================================================
    missing_mc: list[str] = []
    for fname in FEEDBACK_FILES:
        data = feedback_data.get(fname)
        if isinstance(data, dict):
            if not isinstance(data.get("model_context"), dict):
                missing_mc.append(fname)
        # If the file wasn't loaded, CHECK 3 already flagged it

    checked_count = sum(1 for f in FEEDBACK_FILES if f in feedback_data)
    if checked_count > 0 and len(missing_mc) == 0:
        _record("PASS", f"Model context present in all feedback files ({checked_count} checked)")
    elif checked_count == 0:
        _record("FAIL", "Model context present in all feedback files (no files loaded)")
    else:
        _record("FAIL", f"Model context present in all feedback files (missing in: {', '.join(missing_mc)})")

    # ==================================================================
    # CHECK 10: Checkpoint file exists and status_counts is populated
    # ==================================================================
    ckpt_path = root / "checkpoint.json"
    if ckpt_path.is_file():
        ckpt_data = _load_json(ckpt_path)
        if isinstance(ckpt_data, dict):
            sc = ckpt_data.get("status_counts", {})
            if isinstance(sc, dict) and len(sc) > 0:
                _record("PASS", f"Checkpoint file has status_counts ({sc})")
            else:
                _record("FAIL", "Checkpoint file has status_counts (status_counts missing or empty)")
        else:
            _record("FAIL", "Checkpoint file has status_counts (invalid JSON)")
    else:
        _record("FAIL", "Checkpoint file has status_counts (checkpoint.json not found)")

    # ==================================================================
    # Report
    # ==================================================================
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")

    print("=" * 78)
    print(f"POST-SCAN VALIDATION: {scan_dir}")
    print("=" * 78)
    print()

    for num, status, label in results:
        print(f"  CHECK {num:2d}: [{status}] {label}")

    print()
    print("=" * 78)
    print(f"  Results: {n_pass} passed, {n_warn} warnings, {n_fail} failed")

    if n_fail > 0:
        print("  VALIDATION FAILED")
    elif n_warn > 0 and strict:
        print("  VALIDATION FAILED (strict mode: warnings treated as failures)")
    elif n_warn > 0:
        print("  VALIDATION PASSED (with warnings)")
    else:
        print("  VALIDATION PASSED")

    print("=" * 78)

    # Determine exit code
    if n_fail > 0:
        return False
    if strict and n_warn > 0:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-scan validation: checks scan output meets minimum quality bars."
    )
    parser.add_argument(
        "scan_dir",
        help="Path to the scan output directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit 1 on any warning)",
    )
    args = parser.parse_args()

    ok = validate_scan(args.scan_dir, strict=args.strict)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
