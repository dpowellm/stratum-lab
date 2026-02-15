"""Outreach queue builder.

Combines enterprise classification, contact extraction, and teaser
reports into a prioritised outreach queue with CAN-SPAM / GDPR
compliance fields.

Output formats: JSON and CSV.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------

def _compute_priority(
    classification: Dict[str, Any],
    contact_info: Dict[str, Any],
    teaser: Dict[str, Any],
) -> int:
    """Compute an outreach priority score (0-100).

    Factors
    -------
    - Enterprise score (0-50): direct mapping from classification score
      (capped at 50).
    - Contact quality (0-25): corporate email +15, role quality +10.
    - Risk urgency (0-25): based on teaser risk severity.
    """
    priority = 0

    # Enterprise score contribution (up to 50)
    ent_score = classification.get("score", 0)
    priority += min(int(ent_score * 0.5), 50)

    # Contact quality (up to 25)
    best = contact_info.get("best_contact")
    if best:
        if not best.get("is_personal", True):
            priority += 15  # corporate email
        role = best.get("role", "contributor")
        if role in ("cto", "founder"):
            priority += 10
        elif role in ("lead", "maintainer"):
            priority += 7
        elif role == "developer":
            priority += 3

    # Risk urgency (up to 25)
    top_risks = teaser.get("top_risks") or []
    severity_score_map = {"critical": 25, "high": 18, "medium": 10, "low": 5, "info": 1}
    if top_risks:
        worst = str(top_risks[0].get("severity", "info")).lower()
        priority += severity_score_map.get(worst, 0)

    return min(priority, 100)


# ---------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------

def build_outreach_queue(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a prioritised outreach queue.

    Parameters
    ----------
    records:
        List of dicts, each containing:
        - ``classification`` -- from ``enterprise_classifier.classify_repo``
        - ``contact_info``   -- from ``contact_extractor.extract_contacts``
        - ``teaser``         -- from ``teaser_report.generate_teaser``
        - ``repo_metadata``  -- enriched metadata

    Returns
    -------
    Sorted (descending priority) list of outreach-ready records.
    """
    queue: List[Dict[str, Any]] = []

    for rec in records:
        classification = rec.get("classification", {})
        contact_info = rec.get("contact_info", {})
        teaser = rec.get("teaser", {})
        metadata = rec.get("repo_metadata", {})

        meta = metadata.get("_meta", {})
        owner = meta.get("owner", "unknown")
        repo = meta.get("repo", "unknown")

        best_contact = contact_info.get("best_contact")
        priority = _compute_priority(classification, contact_info, teaser)

        now_iso = datetime.now(timezone.utc).isoformat()

        entry: Dict[str, Any] = {
            # Identity
            "owner": owner,
            "repo": repo,
            "repo_url": f"https://github.com/{owner}/{repo}",

            # Contact
            "contact_email": best_contact["email"] if best_contact else None,
            "contact_name": best_contact.get("name") if best_contact else None,
            "contact_role": best_contact.get("role") if best_contact else None,
            "contact_source": best_contact.get("source") if best_contact else None,
            "company_domain": contact_info.get("company_domain"),
            "outreach_ready": contact_info.get("outreach_ready", False),

            # Classification
            "enterprise_score": classification.get("score", 0),
            "tier": classification.get("tier", "unknown"),
            "is_enterprise": classification.get("is_enterprise", False),

            # Teaser
            "teaser_headline": teaser.get("headline", ""),
            "teaser_preview": teaser.get("preview", ""),
            "total_risks": teaser.get("total_risks", 0),

            # Priority
            "priority_score": priority,

            # Status tracking
            "status": "pending",
            "created_at": now_iso,
            "last_updated": now_iso,
            "sent_at": None,
            "opened_at": None,
            "replied_at": None,
            "notes": "",

            # CAN-SPAM / GDPR compliance
            "consent_basis": "legitimate_interest",
            "unsubscribed": False,
            "data_source": "public_github_profile",
            "retention_expires": None,
            "gdpr_right_to_erasure_requested": False,
            "physical_address_included": False,
            "unsubscribe_link_required": True,
        }

        queue.append(entry)

    # Sort descending by priority
    queue.sort(key=lambda e: e["priority_score"], reverse=True)

    return queue


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

def save_queue(
    queue: List[Dict[str, Any]],
    output_dir: str | Path,
    basename: str = "outreach_queue",
) -> Dict[str, str]:
    """Save the outreach queue as JSON and CSV.

    Parameters
    ----------
    queue:
        Output of ``build_outreach_queue``.
    output_dir:
        Directory to write files into.
    basename:
        Base filename (without extension).

    Returns
    -------
    dict with ``json_path`` and ``csv_path``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{basename}.json"
    csv_path = out / f"{basename}.csv"

    # JSON
    json_path.write_text(
        json.dumps(queue, indent=2, default=str),
        encoding="utf-8",
    )

    # CSV
    if queue:
        fieldnames = list(queue[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(queue)
    else:
        csv_path.write_text("", encoding="utf-8")

    logger.info("Saved queue: %s (%d records)", json_path, len(queue))

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def _tier_summary(queue: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of the queue broken down by tier.

    Returns
    -------
    dict with per-tier counts and overall stats.
    """
    tiers: Dict[str, int] = {}
    total_priority = 0
    outreach_ready_count = 0

    for entry in queue:
        tier = entry.get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
        total_priority += entry.get("priority_score", 0)
        if entry.get("outreach_ready"):
            outreach_ready_count += 1

    return {
        "total": len(queue),
        "by_tier": tiers,
        "outreach_ready": outreach_ready_count,
        "avg_priority": round(total_priority / len(queue), 1) if queue else 0,
    }
