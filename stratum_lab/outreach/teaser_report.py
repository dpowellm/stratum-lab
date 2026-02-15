"""Teaser report generator.

Produces a gated preview of a full Stratum risk report that exposes
enough detail to demonstrate value without giving away remediation
guidance.

A teaser includes:
- Headline and preview paragraph
- Top 2 risks (name + probability + severity, **no** remediation)
- CTA with a list of gated sections the full report would unlock
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sections that are gated (only available in the full report)
GATED_SECTIONS = [
    "Full risk inventory with remediation steps",
    "Dependency vulnerability deep-dive",
    "Agent capability graph with trust boundaries",
    "Data-flow analysis and exfiltration vectors",
    "Compliance checklist (SOC 2 / HIPAA / GDPR)",
    "Priority-ranked action plan",
]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _one_liner(risk: Dict[str, Any]) -> str:
    """Produce a single-line summary of a risk for the teaser.

    Format: ``<name> (probability: <p>, severity: <s>)``
    """
    name = risk.get("name") or risk.get("title") or "Unnamed risk"
    probability = risk.get("probability", "unknown")
    severity = risk.get("severity", "unknown")
    return f"{name} (probability: {probability}, severity: {severity})"


def _generate_headline(
    repo_name: str,
    top_risks: List[Dict[str, Any]],
    total_risk_count: int,
) -> str:
    """Generate an attention-grabbing headline for the teaser."""
    if total_risk_count == 0:
        return f"Stratum Security Scan: {repo_name} -- No Risks Detected"

    severity_counts: Dict[str, int] = {}
    for r in top_risks:
        sev = str(r.get("severity", "unknown")).lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    worst = "critical" if "critical" in severity_counts else (
        "high" if "high" in severity_counts else "medium"
    )
    return (
        f"Stratum Security Scan: {repo_name} -- "
        f"{total_risk_count} risks identified, "
        f"including {worst}-severity findings"
    )


def _generate_preview(
    repo_name: str,
    top_risks: List[Dict[str, Any]],
    total_risk_count: int,
) -> str:
    """Generate a short preview paragraph (2-3 sentences)."""
    if total_risk_count == 0:
        return (
            f"Our automated scan of {repo_name} found no significant risks. "
            "Request the full report for a detailed breakdown of your "
            "agent's capability graph and trust boundaries."
        )

    risk_summaries = " and ".join(_one_liner(r) for r in top_risks[:2])
    return (
        f"Our automated scan of {repo_name} identified "
        f"{total_risk_count} potential risks. "
        f"The most notable findings include {risk_summaries}. "
        "Unlock the full report for detailed remediation guidance "
        "and a complete risk inventory."
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def generate_teaser(
    full_report: Dict[str, Any],
    repo_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a teaser from a full Stratum risk report.

    Parameters
    ----------
    full_report:
        The complete risk-analysis report.  Expected to contain a
        ``risks`` key with a list of risk dicts, each having at least
        ``name``/``title``, ``probability``, ``severity``.
    repo_metadata:
        Optional enriched repo metadata for additional context.

    Returns
    -------
    dict with:
        headline      -- attention-grabbing title
        preview       -- 2-3 sentence summary
        top_risks     -- list of top 2 risk one-liners (no remediation)
        total_risks   -- total number of risks in full report
        gated_sections-- list of sections available in full report
        cta           -- call-to-action text
        repo          -- repo identifier
    """
    risks: List[Dict[str, Any]] = full_report.get("risks") or []

    # Sort risks by severity (critical > high > medium > low > info)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_risks = sorted(
        risks,
        key=lambda r: severity_order.get(
            str(r.get("severity", "info")).lower(), 5
        ),
    )

    top_risks = sorted_risks[:2]
    total_risk_count = len(risks)

    # Derive repo name
    repo_name = "your repository"
    if repo_metadata:
        meta = repo_metadata.get("_meta", {})
        owner = meta.get("owner", "")
        repo = meta.get("repo", "")
        if owner and repo:
            repo_name = f"{owner}/{repo}"
    elif full_report.get("repo"):
        repo_name = full_report["repo"]

    headline = _generate_headline(repo_name, top_risks, total_risk_count)
    preview = _generate_preview(repo_name, top_risks, total_risk_count)

    # Build sanitised risk previews (no remediation)
    risk_previews = []
    for r in top_risks:
        risk_previews.append({
            "name": r.get("name") or r.get("title", "Unnamed risk"),
            "probability": r.get("probability", "unknown"),
            "severity": r.get("severity", "unknown"),
            "one_liner": _one_liner(r),
        })

    cta = (
        "Request the full Stratum report to get actionable remediation "
        "steps, a complete risk inventory, and a detailed agent capability "
        "graph for your project."
    )

    return {
        "headline": headline,
        "preview": preview,
        "top_risks": risk_previews,
        "total_risks": total_risk_count,
        "gated_sections": GATED_SECTIONS,
        "cta": cta,
        "repo": repo_name,
    }
