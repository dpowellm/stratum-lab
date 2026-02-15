"""Contact extraction from multiple sources.

Sources
-------
1. Git contributors -- emails and role estimation from commit history.
2. GitHub org contact info -- blog, email, location from org profile.
3. Package manifest authors -- ``pyproject.toml`` ``[project]`` table.
4. README email patterns -- regex extraction of email addresses.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

PERSONAL_EMAIL_DOMAINS: Set[str] = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "aol.com",
    "icloud.com",
    "me.com",
    "mail.com",
    "protonmail.com",
    "proton.me",
    "yandex.com",
    "gmx.com",
    "zoho.com",
    "fastmail.com",
    "tutanota.com",
    "hey.com",
    "pm.me",
    "msn.com",
    "qq.com",
    "163.com",
    "126.com",
    "sina.com",
}

NOREPLY_PATTERNS = re.compile(
    r"(noreply|no-reply|users\.noreply\.github\.com)", re.IGNORECASE
)

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Role heuristics: if the contributor login or name matches these
# patterns we bump their estimated role.
ROLE_PATTERNS = {
    "cto": re.compile(r"\bcto\b", re.IGNORECASE),
    "founder": re.compile(r"\bfounder\b", re.IGNORECASE),
    "lead": re.compile(r"\b(lead|principal|staff|senior)\b", re.IGNORECASE),
    "maintainer": re.compile(r"\b(maintainer|owner)\b", re.IGNORECASE),
    "developer": re.compile(r"\b(dev|developer|engineer|contributor)\b", re.IGNORECASE),
}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _is_noreply(email: str) -> bool:
    """Return True if the email is a no-reply / bot address."""
    return bool(NOREPLY_PATTERNS.search(email))


def _email_domain(email: str) -> str:
    """Extract the domain from an email address."""
    return email.rsplit("@", 1)[-1].lower() if "@" in email else ""


def _is_personal(email: str) -> bool:
    """Return True if the email belongs to a personal domain."""
    return _email_domain(email) in PERSONAL_EMAIL_DOMAINS


def _estimate_role(
    login: Optional[str] = None,
    bio: Optional[str] = None,
    contributions: int = 0,
    is_top_contributor: bool = False,
) -> str:
    """Estimate a contributor's role from available signals.

    Returns one of: cto, founder, lead, maintainer, developer, contributor.
    """
    text = f"{login or ''} {bio or ''}"

    for role, pattern in ROLE_PATTERNS.items():
        if pattern.search(text):
            return role

    if is_top_contributor or contributions > 100:
        return "maintainer"
    if contributions > 10:
        return "developer"
    return "contributor"


def _select_best_contact(contacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the best contact for outreach.

    Priority: corporate email > personal email, maintainer > contributor.
    """
    if not contacts:
        return None

    role_priority = {
        "cto": 0,
        "founder": 1,
        "lead": 2,
        "maintainer": 3,
        "developer": 4,
        "contributor": 5,
    }

    def _sort_key(c: Dict[str, Any]) -> tuple:
        is_corporate = 0 if c.get("is_personal", True) else -1
        is_noreply = 1 if c.get("is_noreply", False) else 0
        role_rank = role_priority.get(c.get("role", "contributor"), 5)
        return (is_noreply, is_corporate, role_rank)

    ranked = sorted(contacts, key=_sort_key)
    return ranked[0] if ranked else None


# ---------------------------------------------------------------
# Source extractors
# ---------------------------------------------------------------

def _extract_from_contributors(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract contacts from commit emails + contributor list."""
    contacts: List[Dict[str, Any]] = []
    seen_emails: Set[str] = set()

    commit_emails: List[str] = metadata.get("commit_emails") or []
    contributors: List[Dict[str, Any]] = metadata.get("contributors") or []

    # Build a map: login -> contribution count
    contrib_map: Dict[str, int] = {}
    top_login: Optional[str] = None
    for i, c in enumerate(contributors):
        login = c.get("login", "")
        contrib_map[login] = c.get("contributions", 0)
        if i == 0:
            top_login = login

    for email in commit_emails:
        if _is_noreply(email):
            continue
        if email.lower() in seen_emails:
            continue
        seen_emails.add(email.lower())

        domain = _email_domain(email)
        contacts.append({
            "email": email,
            "source": "git_commits",
            "domain": domain,
            "is_personal": _is_personal(email),
            "is_noreply": False,
            "role": _estimate_role(contributions=0),
            "name": None,
        })

    # Augment role for known contributors
    for c in contributors:
        login = c.get("login", "")
        is_top = login == top_login
        role = _estimate_role(
            login=login,
            contributions=c.get("contributions", 0),
            is_top_contributor=is_top,
        )
        # Try to match to an existing contact by email domain heuristic
        for contact in contacts:
            if login.lower() in (contact.get("email", "").split("@")[0].lower()):
                contact["role"] = role
                contact["name"] = login
                break

    return contacts


def _extract_from_org(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract contact information from the GitHub org profile."""
    contacts: List[Dict[str, Any]] = []
    org_info = metadata.get("org_info")
    if not org_info:
        return contacts

    org_email = org_info.get("email")
    if org_email and not _is_noreply(org_email):
        contacts.append({
            "email": org_email,
            "source": "github_org",
            "domain": _email_domain(org_email),
            "is_personal": _is_personal(org_email),
            "is_noreply": False,
            "role": "org_contact",
            "name": org_info.get("name"),
        })

    return contacts


def _extract_from_manifest(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract author emails from pyproject.toml content."""
    contacts: List[Dict[str, Any]] = []
    pyproject = metadata.get("pyproject_toml") or ""
    if not pyproject:
        return contacts

    # Simple regex extraction -- not a full TOML parser
    for match in EMAIL_REGEX.finditer(pyproject):
        email = match.group(0)
        if not _is_noreply(email):
            contacts.append({
                "email": email,
                "source": "pyproject_toml",
                "domain": _email_domain(email),
                "is_personal": _is_personal(email),
                "is_noreply": False,
                "role": "package_author",
                "name": None,
            })

    return contacts


def _extract_from_readme(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract email addresses mentioned in the README."""
    contacts: List[Dict[str, Any]] = []
    readme = metadata.get("readme") or ""
    if not readme:
        return contacts

    for match in EMAIL_REGEX.finditer(readme):
        email = match.group(0)
        if not _is_noreply(email):
            contacts.append({
                "email": email,
                "source": "readme",
                "domain": _email_domain(email),
                "is_personal": _is_personal(email),
                "is_noreply": False,
                "role": "readme_contact",
                "name": None,
            })

    return contacts


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def extract_contacts(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and deduplicate contacts from all sources.

    Parameters
    ----------
    metadata:
        Enriched repository metadata (output of ``GitHubEnricher.enrich_repo``).

    Returns
    -------
    dict with:
        contacts       -- deduplicated list of contact dicts
        company_domain -- most likely corporate domain (or None)
        best_contact   -- single best contact for outreach (or None)
        outreach_ready -- bool indicating whether a viable contact exists
    """
    all_contacts: List[Dict[str, Any]] = []
    seen_emails: Set[str] = set()

    for extractor in (
        _extract_from_contributors,
        _extract_from_org,
        _extract_from_manifest,
        _extract_from_readme,
    ):
        for contact in extractor(metadata):
            email_lower = contact["email"].lower()
            if email_lower not in seen_emails:
                seen_emails.add(email_lower)
                all_contacts.append(contact)

    # Determine company domain
    corporate_domains: Dict[str, int] = {}
    for c in all_contacts:
        if not c["is_personal"] and not c["is_noreply"]:
            d = c["domain"]
            if d:
                corporate_domains[d] = corporate_domains.get(d, 0) + 1

    company_domain: Optional[str] = None
    if corporate_domains:
        company_domain = max(corporate_domains, key=corporate_domains.get)  # type: ignore[arg-type]

    best = _select_best_contact(all_contacts)
    outreach_ready = best is not None and not best.get("is_noreply", True)

    return {
        "contacts": all_contacts,
        "company_domain": company_domain,
        "best_contact": best,
        "outreach_ready": outreach_ready,
    }
