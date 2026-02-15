"""Enterprise classifier with a 0-100 scoring rubric.

Scoring breakdown
-----------------
Org signals       (0-30):  github_org +10, org_website +10, 10+ repos +5, 5+ members +5
Code maturity     (0-35):  CI/CD +10, Docker/K8s +5, monitoring +5, multi-contributor +5, recently active +5
Production signals(0-35):  README deploy/production +10, .env +5, secrets mgmt +5,
                           commercial license +5, has tests +5

Enterprise threshold: 45+

Tiers
-----
- high_value  (70+)
- enterprise  (45-69)
- prosumer    (25-44)
- community   (<25)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

COMMERCIAL_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "mpl-2.0", "lgpl-2.1", "lgpl-3.0", "isc",
}

CI_CD_PATHS = {
    ".github/workflows", ".circleci", "Jenkinsfile", ".travis.yml",
    ".gitlab-ci.yml", "azure-pipelines.yml", "bitbucket-pipelines.yml",
}

DOCKER_K8S_FILES = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "k8s/", "kubernetes/", "helm/", "Chart.yaml",
}

MONITORING_FILES = {
    "prometheus", "grafana", "datadog", "newrelic", "sentry",
    "opentelemetry", "otel", "jaeger",
}

SECRETS_MGMT_PATTERNS = {
    "vault", "aws-secretsmanager", "sops", ".sealed-secret",
    "external-secrets", "doppler",
}

DEPLOY_KEYWORDS = re.compile(
    r"\b(deploy|deployment|production|staging|infrastructure|terraform|pulumi|cloudformation)\b",
    re.IGNORECASE,
)

TEST_DIRS = {"tests", "test", "spec", "__tests__", "testing"}


# ---------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------

def _score_org_signals(metadata: Dict[str, Any]) -> int:
    """Score organisational signals (0-30)."""
    score = 0
    org_info = metadata.get("org_info")
    repo_info = metadata.get("repo_info") or {}

    # github_org: owner is an Organization
    owner_type = repo_info.get("owner", {}).get("type", "")
    if owner_type == "Organization":
        score += 10

    # org_website
    if org_info and org_info.get("blog"):
        score += 10
    elif repo_info.get("homepage"):
        score += 10

    # 10+ public repos in org
    if org_info and (org_info.get("public_repos", 0) >= 10):
        score += 5

    # 5+ members (approximated from org public_members_url or collaborators)
    contributors = metadata.get("contributors") or []
    if org_info:
        # GitHub orgs don't directly expose member count in REST;
        # we approximate via contributor count for the repo.
        if len(contributors) >= 5:
            score += 5
    elif len(contributors) >= 5:
        score += 5

    return min(score, 30)


def _score_code_maturity(metadata: Dict[str, Any]) -> int:
    """Score code maturity signals (0-35)."""
    score = 0
    file_tree: List[str] = metadata.get("file_tree") or []
    file_tree_lower = {f.lower() for f in file_tree}
    contributors = metadata.get("contributors") or []

    # CI/CD
    for ci_path in CI_CD_PATHS:
        if any(f.startswith(ci_path) or f == ci_path for f in file_tree):
            score += 10
            break

    # Docker/K8s
    for dk_file in DOCKER_K8S_FILES:
        if any(f.startswith(dk_file) or f == dk_file for f in file_tree):
            score += 5
            break

    # Monitoring
    for mon in MONITORING_FILES:
        if any(mon in f.lower() for f in file_tree):
            score += 5
            break

    # Multi-contributor (3+)
    if len(contributors) >= 3:
        score += 5

    # Recently active (pushed in last 90 days)
    repo_info = metadata.get("repo_info") or {}
    pushed_at = repo_info.get("pushed_at")
    if pushed_at:
        try:
            pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - pushed_dt < timedelta(days=90):
                score += 5
        except (ValueError, TypeError):
            pass

    return min(score, 35)


def _score_production_signals(metadata: Dict[str, Any]) -> int:
    """Score production-readiness signals (0-35)."""
    score = 0
    file_tree: List[str] = metadata.get("file_tree") or []
    file_tree_lower = {f.lower() for f in file_tree}
    readme: str = metadata.get("readme") or ""
    repo_info = metadata.get("repo_info") or {}

    # README mentions deploy/production
    if DEPLOY_KEYWORDS.search(readme):
        score += 10

    # .env file present
    if ".env" in file_tree or ".env.example" in file_tree:
        score += 5

    # Secrets management
    for secret_pat in SECRETS_MGMT_PATTERNS:
        if any(secret_pat in f.lower() for f in file_tree):
            score += 5
            break

    # Commercial-friendly license
    license_info = repo_info.get("license") or {}
    spdx = (license_info.get("spdx_id") or "").lower()
    if spdx in COMMERCIAL_LICENSES:
        score += 5

    # Has tests
    for test_dir in TEST_DIRS:
        if any(
            f == test_dir or f.startswith(f"{test_dir}/")
            for f in file_tree
        ):
            score += 5
            break

    return min(score, 35)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def classify_repo(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single repository and return scoring breakdown.

    Parameters
    ----------
    metadata:
        Enriched repository metadata (output of ``GitHubEnricher.enrich_repo``).

    Returns
    -------
    dict with ``score``, ``tier``, ``breakdown``, ``is_enterprise``.
    """
    org_score = _score_org_signals(metadata)
    maturity_score = _score_code_maturity(metadata)
    production_score = _score_production_signals(metadata)
    total = org_score + maturity_score + production_score

    if total >= 70:
        tier = "high_value"
    elif total >= 45:
        tier = "enterprise"
    elif total >= 25:
        tier = "prosumer"
    else:
        tier = "community"

    return {
        "score": total,
        "tier": tier,
        "is_enterprise": total >= 45,
        "breakdown": {
            "org_signals": org_score,
            "code_maturity": maturity_score,
            "production_signals": production_score,
        },
    }


def classify_batch(
    metadata_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify a list of enriched repo metadata dicts.

    Returns a list of classification results (same order), each
    augmented with ``owner`` and ``repo`` from ``_meta``.
    """
    results: List[Dict[str, Any]] = []
    for meta in metadata_list:
        classification = classify_repo(meta)
        owner = meta.get("_meta", {}).get("owner", "unknown")
        repo = meta.get("_meta", {}).get("repo", "unknown")
        classification["owner"] = owner
        classification["repo"] = repo
        results.append(classification)

    logger.info(
        "Classified %d repos: %d enterprise, %d prosumer, %d community",
        len(results),
        sum(1 for r in results if r["tier"] in ("high_value", "enterprise")),
        sum(1 for r in results if r["tier"] == "prosumer"),
        sum(1 for r in results if r["tier"] == "community"),
    )
    return results
