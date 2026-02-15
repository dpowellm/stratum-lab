"""GitHub API enrichment for repository metadata collection.

Collects repo info, org info, contributors, file tree, README,
pyproject.toml, and commit emails using only stdlib (urllib).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubEnricher:
    """Enrich repository metadata via the GitHub REST API.

    Uses urllib (no requests dependency). Supports GITHUB_TOKEN for
    authenticated requests and tracks rate-limit headers.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.rate_limit_remaining: Optional[int] = None
        self.rate_limit_reset: Optional[float] = None
        self._request_count = 0

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, url: str, accept: str = "application/vnd.github+json") -> Optional[Dict[str, Any]]:
        """Issue a GET against the GitHub API and return parsed JSON.

        Returns ``None`` on 404 / 403 / non-200 responses so callers can
        degrade gracefully.
        """
        headers: Dict[str, str] = {
            "Accept": accept,
            "User-Agent": "stratum-lab-enricher/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # Respect rate-limit back-off
        if (
            self.rate_limit_remaining is not None
            and self.rate_limit_remaining < 5
            and self.rate_limit_reset is not None
        ):
            wait = max(0, self.rate_limit_reset - time.time()) + 1
            logger.warning("Rate-limit nearly exhausted, sleeping %.1fs", wait)
            time.sleep(wait)

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Track rate-limit headers
                self.rate_limit_remaining = int(
                    resp.headers.get("X-RateLimit-Remaining", -1)
                )
                reset_ts = resp.headers.get("X-RateLimit-Reset")
                if reset_ts:
                    self.rate_limit_reset = float(reset_ts)
                self._request_count += 1
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            logger.debug("HTTP %s for %s", exc.code, url)
            return None
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("Network error for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Single-repo enrichment
    # ------------------------------------------------------------------

    def enrich_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        """Collect all available metadata for *owner/repo*.

        Returns a dict with keys:
            repo_info, org_info, contributors, file_tree, readme,
            pyproject_toml, commit_emails, _meta
        """
        result: Dict[str, Any] = {"_meta": {"owner": owner, "repo": repo}}

        # 1. Repo info
        repo_info = self._get(f"{GITHUB_API}/repos/{owner}/{repo}")
        result["repo_info"] = repo_info

        # 2. Org info (only if owner is an org)
        if repo_info and repo_info.get("owner", {}).get("type") == "Organization":
            result["org_info"] = self._get(f"{GITHUB_API}/orgs/{owner}")
        else:
            result["org_info"] = None

        # 3. Contributors (top 30)
        result["contributors"] = (
            self._get(f"{GITHUB_API}/repos/{owner}/{repo}/contributors?per_page=30")
            or []
        )

        # 4. File tree (default branch, first level via recursive tree)
        default_branch = (repo_info or {}).get("default_branch", "main")
        tree_resp = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        )
        if tree_resp and "tree" in tree_resp:
            result["file_tree"] = [entry["path"] for entry in tree_resp["tree"]]
        else:
            result["file_tree"] = []

        # 5. README
        readme_resp = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/readme"
        )
        if readme_resp and "content" in readme_resp:
            try:
                result["readme"] = b64decode(readme_resp["content"]).decode(
                    errors="replace"
                )
            except Exception:
                result["readme"] = None
        else:
            result["readme"] = None

        # 6. pyproject.toml
        pyproject_resp = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/pyproject.toml"
        )
        if pyproject_resp and "content" in pyproject_resp:
            try:
                result["pyproject_toml"] = b64decode(
                    pyproject_resp["content"]
                ).decode(errors="replace")
            except Exception:
                result["pyproject_toml"] = None
        else:
            result["pyproject_toml"] = None

        # 7. Commit emails (last 30 commits)
        commits = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits?per_page=30"
        )
        emails: List[str] = []
        if commits:
            for c in commits:
                commit_data = c.get("commit", {})
                for role in ("author", "committer"):
                    email = commit_data.get(role, {}).get("email")
                    if email and email not in emails:
                        emails.append(email)
        result["commit_emails"] = emails

        result["_meta"]["requests_used"] = self._request_count
        result["_meta"]["rate_limit_remaining"] = self.rate_limit_remaining
        return result

    # ------------------------------------------------------------------
    # Batch enrichment
    # ------------------------------------------------------------------

    def enrich_batch(
        self,
        repos: List[str],
        output_dir: str | Path,
    ) -> Dict[str, Any]:
        """Enrich a list of ``owner/repo`` strings.

        Saves one JSON file per repo under *output_dir* and returns
        summary statistics.

        Parameters
        ----------
        repos:
            List of ``"owner/repo"`` strings.
        output_dir:
            Directory where per-repo JSON files are written.

        Returns
        -------
        dict with ``total``, ``succeeded``, ``failed``, ``files``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary: Dict[str, Any] = {
            "total": len(repos),
            "succeeded": 0,
            "failed": 0,
            "files": [],
        }

        for slug in repos:
            parts = slug.strip().split("/")
            if len(parts) != 2:
                logger.warning("Skipping invalid repo slug: %s", slug)
                summary["failed"] += 1
                continue

            owner, repo = parts
            try:
                data = self.enrich_repo(owner, repo)
                fname = out / f"{owner}__{repo}.json"
                fname.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                summary["files"].append(str(fname))
                summary["succeeded"] += 1
            except Exception:
                logger.exception("Failed to enrich %s/%s", owner, repo)
                summary["failed"] += 1

        return summary
