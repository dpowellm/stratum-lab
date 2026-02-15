"""Metadata schema for enterprise classification and contact extraction.

This module defines what metadata the structural scan (stratum-cli) should
collect per repo, and validates that sufficient data exists for outreach.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------

REQUIRED_FOR_ENTERPRISE: list[str] = [
    "owner_type",        # User vs Organization
    "owner_login",       # GitHub org/user name
    "file_list",         # For CI/CD, Docker, test detection
    "readme_text",       # For production keyword scanning
    "last_commit_date",  # For recency check
]

REQUIRED_FOR_CONTACTS: list[str] = [
    "contributors",      # List with login, name, email, commits
]

OPTIONAL_BUT_VALUABLE: list[str] = [
    "org_info",          # Website, email, member count
    "requirements_text", # For monitoring package detection
    "license_spdx",      # For commercial license check
    "topics",            # GitHub topics
    "stars",             # Social proof signal
    "pyproject_raw",     # Author fields
]

ALL_KNOWN_FIELDS: list[str] = list(
    dict.fromkeys(
        REQUIRED_FOR_ENTERPRISE
        + REQUIRED_FOR_CONTACTS
        + OPTIONAL_BUT_VALUABLE
    )
)


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

def validate_metadata(repo_metadata: dict[str, Any]) -> dict[str, Any]:
    """Check what metadata fields are present and report coverage.

    Parameters
    ----------
    repo_metadata:
        A dict produced by ``GitHubEnricher.enrich_repo``.

    Returns
    -------
    dict with enterprise_ready, contact_ready, field breakdowns, coverage_pct.
    """
    metadata = repo_metadata.get("metadata", repo_metadata)

    def _is_present(key: str) -> bool:
        val = metadata.get(key)
        if val is None:
            return False
        if isinstance(val, (list, dict, str)) and len(val) == 0:
            return False
        return True

    enterprise_fields = {f: _is_present(f) for f in REQUIRED_FOR_ENTERPRISE}
    contact_fields = {f: _is_present(f) for f in REQUIRED_FOR_CONTACTS}
    optional_fields = {f: _is_present(f) for f in OPTIONAL_BUT_VALUABLE}

    enterprise_ready = all(enterprise_fields.values())
    contact_ready = all(contact_fields.values())

    all_fields = {**enterprise_fields, **contact_fields, **optional_fields}
    coverage_pct = (
        round(sum(1 for v in all_fields.values() if v) / len(all_fields) * 100, 1)
        if all_fields
        else 0.0
    )

    return {
        "enterprise_ready": enterprise_ready,
        "contact_ready": contact_ready,
        "enterprise_fields": enterprise_fields,
        "contact_fields": contact_fields,
        "optional_fields": optional_fields,
        "coverage_pct": coverage_pct,
    }
