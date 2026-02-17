"""Per-repo JSONL writer and cross-repo summary aggregation."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stratum_lab.judge.config import JUDGE_MODEL, CostTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-repo writer
# ---------------------------------------------------------------------------

def write_judge_results(
    results: list[dict[str, Any]],
    output_path: Path | str,
    run_number: int = 1,
) -> None:
    """Write judge results JSONL for a single repo run."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        for rec in results:
            rec.setdefault("run_number", run_number)
            rec.setdefault("timestamp", timestamp)
            rec.setdefault("judge_model", JUDGE_MODEL)
            f.write(json.dumps(rec, default=str) + "\n")
    logger.info("Wrote %d judge results to %s", len(results), path)


# ---------------------------------------------------------------------------
# Cross-repo summary builder
# ---------------------------------------------------------------------------

def build_summary(
    all_results: list[dict[str, Any]],
    cost_tracker: CostTracker | None = None,
) -> dict[str, Any]:
    """Build cross-repo judge_summary.json from all judge results."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Group by criterion
    by_criterion: dict[str, list[dict]] = defaultdict(list)
    repos_seen: set[str] = set()
    for rec in all_results:
        if "judge_error" in rec:
            continue
        crit = rec.get("criterion", "")
        by_criterion[crit].append(rec)
        repos_seen.add(rec.get("repo_url", ""))

    summary: dict[str, Any] = {
        "meta": {
            "total_repos_evaluated": len(repos_seen),
            "total_judge_calls": len(all_results),
            "judge_model": JUDGE_MODEL,
            "timestamp": timestamp,
            "total_cost_usd": (
                cost_tracker.cost_usd if cost_tracker else 0.0
            ),
        },
        "per_criterion_summary": {},
        "headline_findings": [],
    }

    # --- Task Adherence ---
    ta_results = by_criterion.get("task_adherence", [])
    if ta_results:
        scores = [r["score"] for r in ta_results if "score" in r]
        summary["per_criterion_summary"]["task_adherence"] = _scored_summary(
            scores, ta_results, score_levels=["1", "2", "3"],
        )

    # --- Hallucination ---
    hal_results = by_criterion.get("hallucination", [])
    if hal_results:
        detected = [r for r in hal_results if r.get("hallucination_detected")]
        high_conf = [r for r in detected if r.get("confidence") == "high"]
        n = len(hal_results)
        fw_breakdown = _binary_framework_breakdown(
            hal_results, key="hallucination_detected",
        )
        summary["per_criterion_summary"]["hallucination"] = {
            "detection_rate": round(len(detected) / n, 2) if n else 0,
            "by_framework": fw_breakdown,
            "high_confidence_rate": round(len(high_conf) / n, 2) if n else 0,
        }

    # --- Instruction Leakage ---
    il_results = by_criterion.get("instruction_leakage", [])
    if il_results:
        detected = [r for r in il_results if r.get("leakage_detected")]
        n = len(il_results)
        type_counts: dict[str, int] = defaultdict(int)
        for r in il_results:
            lt = r.get("leakage_type", "none")
            type_counts[lt] += 1
        summary["per_criterion_summary"]["instruction_leakage"] = {
            "detection_rate": round(len(detected) / n, 2) if n else 0,
            "by_leakage_type": dict(type_counts),
        }

    # --- Output Quality ---
    oq_results = by_criterion.get("output_quality", [])
    if oq_results:
        scores = [r["score"] for r in oq_results if "score" in r]
        n = len(scores)
        garbage = sum(1 for s in scores if s == 1)
        s = _scored_summary(scores, oq_results, score_levels=["1", "2", "3"])
        s["garbage_rate"] = round(garbage / n, 2) if n else 0
        summary["per_criterion_summary"]["output_quality"] = s

    # --- Delegation Fidelity ---
    df_results = by_criterion.get("delegation_fidelity", [])
    if df_results:
        scores = [r["score"] for r in df_results if "score" in r]
        n = len(scores)
        ignored = sum(1 for s in scores if s == 1)
        full_use = sum(1 for s in scores if s == 3)
        summary["per_criterion_summary"]["delegation_fidelity"] = {
            "mean_score": round(sum(scores) / n, 1) if n else 0,
            "n_evaluated": n,
            "ignored_rate": round(ignored / n, 2) if n else 0,
            "full_use_rate": round(full_use / n, 2) if n else 0,
        }

    # --- Error Propagation ---
    ep_results = by_criterion.get("error_propagation", [])
    if ep_results:
        n = len(ep_results)
        type_dist: dict[str, int] = defaultdict(int)
        for r in ep_results:
            pt = r.get("propagation_type", "not_applicable")
            type_dist[pt] += 1
        amplified = type_dist.get("amplified", 0)
        summary["per_criterion_summary"]["error_propagation"] = {
            "n_evaluated": n,
            "distribution": dict(type_dist),
            "amplification_rate": round(amplified / n, 2) if n else 0,
        }

    # --- Headline findings ---
    summary["headline_findings"] = _generate_headlines(summary)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scored_summary(
    scores: list[int],
    results: list[dict],
    score_levels: list[str],
) -> dict[str, Any]:
    """Build mean, distribution, and framework breakdown for a scored criterion."""
    n = len(scores)
    dist = {lvl: sum(1 for s in scores if str(s) == lvl) for lvl in score_levels}
    fw_groups: dict[str, list[int]] = defaultdict(list)
    for rec in results:
        if "score" not in rec:
            continue
        fw = rec.get("framework", "unknown")
        fw_groups[fw].append(rec["score"])
    by_fw = {
        fw: {"mean": round(sum(s) / len(s), 1), "n": len(s)}
        for fw, s in fw_groups.items()
    }
    return {
        "mean_score": round(sum(scores) / n, 1) if n else 0,
        "distribution": dist,
        "by_framework": by_fw,
    }


def _binary_framework_breakdown(
    results: list[dict],
    key: str,
) -> dict[str, dict]:
    """Framework breakdown for a binary criterion."""
    fw_groups: dict[str, list[bool]] = defaultdict(list)
    for rec in results:
        fw = rec.get("framework", "unknown")
        fw_groups[fw].append(bool(rec.get(key)))
    return {
        fw: {
            "rate": round(sum(vals) / len(vals), 2),
            "n": len(vals),
        }
        for fw, vals in fw_groups.items()
    }


def _generate_headlines(summary: dict) -> list[str]:
    """Generate human-readable headline findings from summary stats."""
    headlines: list[str] = []
    pcs = summary.get("per_criterion_summary", {})

    # Hallucination rate
    hal = pcs.get("hallucination", {})
    if "detection_rate" in hal:
        pct = int(hal["detection_rate"] * 100)
        headlines.append(f"{pct}% of agent outputs contain hallucinated content")

    # Error propagation amplification
    ep = pcs.get("error_propagation", {})
    if "amplification_rate" in ep and ep.get("n_evaluated", 0) > 0:
        pct = int(ep["amplification_rate"] * 100)
        headlines.append(
            f"{pct}% of error chains show amplification from upstream to downstream"
        )

    # Instruction leakage
    il = pcs.get("instruction_leakage", {})
    if "detection_rate" in il:
        pct = int(il["detection_rate"] * 100)
        headlines.append(f"{pct}% of outputs leak internal framework scaffolding")

    # Framework comparison for hallucination
    hal_fw = hal.get("by_framework", {})
    if len(hal_fw) >= 2:
        sorted_fw = sorted(hal_fw.items(), key=lambda x: x[1].get("rate", 0))
        best = sorted_fw[0]
        worst = sorted_fw[-1]
        headlines.append(
            f"{best[0]} systems hallucinate at {int(best[1]['rate'] * 100)}% "
            f"vs {worst[0]} at {int(worst[1]['rate'] * 100)}%"
        )

    # Output quality garbage rate
    oq = pcs.get("output_quality", {})
    if "garbage_rate" in oq:
        pct = int(oq["garbage_rate"] * 100)
        headlines.append(f"{pct}% of agent outputs are unusable garbage")

    return headlines


def write_summary(summary: dict, output_path: Path | str) -> None:
    """Write judge_summary.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote judge summary to %s", path)
