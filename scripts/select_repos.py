#!/usr/bin/env python3
"""Select and rank repos for behavioral scanning.

Modes:
  A. scan-results (default): reads scan_results.jsonl, scores by graph structure
  B. url-list (--plain):     reads a text file of GitHub URLs, passes through at score 0
  C. discovery (--discover): searches GitHub API for AI agent repos

Usage:
    python select_repos.py scan_results.jsonl [--top N] [--min-score N] [-o out.txt]
    python select_repos.py repo_urls.txt --plain [--top N] [-o out.txt]
    python select_repos.py --discover [--top N] [--min-score N] [-o out.txt]

Flags:
    --github-token TOKEN   GitHub personal access token for API calls
    --include-forks        Include forked repos (default: skip)
    --deduplicate          Enable hash-based duplicate detection
    --topology-diversity   Rerank selection to maximize unique topology patterns
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FRAMEWORKS = {"crewai", "langgraph", "autogen", "langchain"}
PREFERRED_FRAMEWORKS = {"crewai", "langgraph", "autogen"}
SERVERS = {"fastapi", "flask", "uvicorn", "starlette", "django"}
EXT_APIS = {"exa", "tavily", "serper", "serpapi", "brave_search",
            "bing", "wolfram", "google_search"}
DISCOVER_QUERIES = ["crewai agent", "langgraph agent",
                    "autogen multi-agent", "langchain agent"]
MAX_DISCOVER_URLS = 5000
GH_RATE_UNAUTH = 60       # requests/hour unauthenticated
GH_RATE_AUTH = 5000        # requests/hour authenticated
MAX_BACKOFF_RETRIES = 5

# -- GitHub API helpers -------------------------------------------------------

def _gh_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle_rate_limit(resp, attempt: int) -> bool:
    """Return True if the caller should retry after backoff."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) == 0:
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_ts - int(time.time()), 1)
        print(f"Rate limit exhausted; sleeping {wait}s", file=sys.stderr)
        time.sleep(wait)
        return True
    if resp.status_code == 403 and attempt < MAX_BACKOFF_RETRIES:
        wait = 2 ** attempt
        print(f"403 received; backing off {wait}s (attempt {attempt + 1})",
              file=sys.stderr)
        time.sleep(wait)
        return True
    return False


def _gh_get(url: str, params: dict | None, token: str | None,
            timeout: int = 15) -> "requests.Response | None":
    """GET with exponential backoff on 403 / rate-limit."""
    import requests
    headers = _gh_headers(token)
    for attempt in range(MAX_BACKOFF_RETRIES + 1):
        resp = requests.get(url, params=params, headers=headers,
                            timeout=timeout)
        if resp.status_code in (200, 422):
            return resp
        if not _handle_rate_limit(resp, attempt):
            print(f"Warning: GitHub API {resp.status_code} for {url}",
                  file=sys.stderr)
            return resp
    return resp                    # return last response after retries


def _check_repo_meta(owner_repo: str, token: str | None) -> dict[str, Any]:
    """Fetch repo metadata via GitHub API.  Returns dict with keys:
       archived, is_fork, last_push_iso, skip_reason (or None).
    """
    resp = _gh_get(f"https://api.github.com/repos/{owner_repo}",
                   None, token)
    if resp is None or resp.status_code != 200:
        return {}
    data = resp.json()
    return {
        "archived": data.get("archived", False),
        "is_fork": data.get("fork", False),
        "last_push_iso": data.get("pushed_at", ""),
    }


def _owner_repo(url: str) -> str | None:
    """Extract 'owner/repo' from a GitHub URL."""
    url = url.rstrip("/")
    parts = url.split("github.com/")
    if len(parts) < 2:
        return None
    seg = parts[1].split("/")
    if len(seg) >= 2:
        return f"{seg[0]}/{seg[1]}"
    return None

# -- Deduplication helpers ----------------------------------------------------

def _dedup_key(r: dict[str, Any]) -> str:
    """Build a hash key from agent_count + frameworks + primary_framework."""
    agent_count = str(r.get("agent_count", ""))
    frameworks = ",".join(sorted(r.get("frameworks", [])))
    primary = r.get("primary_framework", "")
    raw = f"{agent_count}|{frameworks}|{primary}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def deduplicate(scored: list[tuple[float, str, dict]],
                ) -> list[tuple[float, str, dict]]:
    """Remove hash-based duplicates, keeping the highest-scored entry."""
    seen: dict[str, tuple[float, str, dict]] = {}
    dupes = 0
    for score, url, raw in scored:
        key = _dedup_key(raw)
        if key in seen:
            dupes += 1
            if score > seen[key][0]:
                seen[key] = (score, url, raw)
        else:
            seen[key] = (score, url, raw)
    if dupes:
        print(f"Dedup: removed {dupes} potential duplicate(s)", file=sys.stderr)
    return list(seen.values())

# -- XCOMP and topology helpers -----------------------------------------------

def is_xcomp(repo_metadata: dict[str, Any]) -> bool:
    """Check if repo has cross-component interactions.

    XCOMP repos produce the richest behavioral records because they exercise
    interactions between agents, tools, data stores, external services, and guards.
    A repo qualifies as XCOMP if it has at least 2 different component types.
    """
    has_tools = repo_metadata.get("tool_count", 0) > 0
    has_data_stores = repo_metadata.get("data_store_count", 0) > 0
    has_external = repo_metadata.get("external_service_count", 0) > 0
    has_guards = repo_metadata.get("guard_count", 0) > 0
    component_types = sum([has_tools, has_data_stores, has_external, has_guards])
    return component_types >= 2  # At least 2 different component types


def compute_topology_hash(repo_metadata: dict[str, Any]) -> str:
    """Hash based on agent_count, framework, tool_count, etc.

    Used to track unique topology patterns across the selection and ensure
    diversity (target 86%+ unique topologies).
    """
    key = (f"{repo_metadata.get('framework', repo_metadata.get('primary_framework', 'unknown'))}:"
           f"{repo_metadata.get('agent_count', 0)}:"
           f"{repo_metadata.get('tool_count', 0)}:"
           f"{repo_metadata.get('data_store_count', 0)}")
    return hashlib.md5(key.encode()).hexdigest()[:8]


def rerank_for_topology_diversity(
    scored: list[tuple[float, str, dict]],
    top_n: int = 0,
) -> list[tuple[float, str, dict]]:
    """Rerank selection to maximize unique topology patterns.

    Uses a greedy approach: iterate through score-sorted candidates and prefer
    repos that introduce a new topology hash over those that duplicate an
    existing one.  Repos with duplicate topologies are pushed to the end
    rather than dropped, so --top N still returns N results.
    """
    if not scored:
        return scored

    seen_hashes: set[str] = set()
    unique_picks: list[tuple[float, str, dict]] = []
    duplicate_picks: list[tuple[float, str, dict]] = []

    for entry in scored:
        _, _, raw = entry
        topo_hash = compute_topology_hash(raw)
        if topo_hash not in seen_hashes:
            seen_hashes.add(topo_hash)
            unique_picks.append(entry)
        else:
            duplicate_picks.append(entry)

    reranked = unique_picks + duplicate_picks
    return reranked


# -- Mode A: scan-results scoring ---------------------------------------------

def score_repo(r: dict[str, Any]) -> float:
    graph = r.get("graph", {})
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    imports = r.get("imports", [])
    files = r.get("files", [])
    meta = r.get("metadata", {})
    score = 0.0

    imp_strs = {str(i).lower() for i in imports} if imports else set()
    names = {str(n.get("node_name", "")).lower() for n in nodes.values()}
    blob = " ".join(imp_strs | names)

    # -- existing structural scoring --

    # +10 known framework
    has_fw = any(fw in blob for fw in FRAMEWORKS)
    if not has_fw:
        has_fw = any(n.get("node_type") in ("agent", "Agent")
                     or str(n.get("node_id", "")).startswith("agent_")
                     for n in nodes.values())
    if has_fw:
        score += 10
    # +5 entry point
    if any(f.endswith(("main.py", "app.py")) for f in files) or meta.get("has_entry_point"):
        score += 5
    # +5 requirements.txt
    if any(f.endswith("requirements.txt") for f in files) or meta.get("has_requirements"):
        score += 5
    # +3 >= 2 agents
    n_agents = sum(1 for n in nodes.values()
                   if n.get("node_type") in ("agent", "Agent")
                   or str(n.get("node_id", "")).startswith("agent_"))
    if n_agents >= 2:
        score += 3
    # +3 small repo
    n_py = sum(1 for f in files if f.endswith(".py"))
    if 0 < n_py < 20:
        score += 3
    # +2 per delegation edge (max +10)
    n_deleg = sum(1 for e in edges.values()
                  if e.get("edge_type") in ("delegates_to", "delegation"))
    score += min(n_deleg * 2, 10)
    # +1 per tool (max +5)
    n_tools = sum(1 for n in nodes.values()
                  if n.get("node_type") in ("capability", "tool", "Tool"))
    score += min(n_tools, 5)
    # -10 server-based
    if any(s in blob for s in SERVERS):
        score -= 10
    # -5 heavy external API usage
    if sum(1 for a in EXT_APIS if a in blob) > 5:
        score -= 5
    # -3/-5 complex monorepo
    if n_py > 50:
        score -= 5
    elif n_py > 30:
        score -= 3

    # -- new telemetry-based scoring (scan_results.jsonl fields) --

    primary_fw = (r.get("primary_framework", "") or "").lower()
    agent_count = r.get("agent_count", n_agents)
    deployment = r.get("deployment_signals", {}) or {}
    llm_providers = [p.lower() for p in (r.get("llm_providers", []) or [])]
    fw_versions = r.get("framework_versions", {}) or {}
    fws_list = [f.lower() for f in (r.get("frameworks", []) or [])]

    # +10 preferred primary framework
    if primary_fw in PREFERRED_FRAMEWORKS:
        score += 10
    # +8 agent_count between 2 and 10
    if 2 <= agent_count <= 10:
        score += 8
    # +5 has lockfile (proxy for runnability)
    if deployment.get("has_lockfile"):
        score += 5
    # +5 llm_providers includes openai (easier to redirect to vLLM)
    if "openai" in llm_providers:
        score += 5
    # +3 agent_count sweet spot (2-5)
    if 2 <= agent_count <= 5:
        score += 3
    # +2 framework_versions present (active maintenance)
    if fw_versions:
        score += 2
    # -10 framework suggests server pattern
    server_fws = {"fastapi", "flask", "django"}
    if any(sf in fws_list or sf in primary_fw for sf in server_fws):
        score -= 10
    # -5 agent_count > 20 (too complex)
    if agent_count > 20:
        score -= 5
    # -5 agent_count == 0
    if agent_count == 0:
        score -= 5
    # -3 only non-OpenAI llm_providers
    if llm_providers and "openai" not in llm_providers:
        score -= 3

    # fork penalty (if the field is present in scan data)
    if r.get("fork", False):
        score -= 15

    # -- event-diversity scoring (PATCH_SPEC_V2 section 4.4) --

    tool_count = r.get("tool_count", 0)
    data_store_count = r.get("data_store_count", 0)
    capabilities = [str(c).lower() for c in (r.get("capabilities", []) or [])]
    patterns = [str(p).lower() for p in (r.get("patterns", []) or [])]
    deployment_strs = " ".join(str(v) for v in deployment.values()) if deployment else ""

    # +15 has tool definitions (will trigger tool.invoked events)
    if tool_count > 0 or n_tools > 0 or "tool" in deployment_strs.lower():
        score += 15

    # +12 has delegation patterns (will trigger delegation events)
    has_delegation = (n_deleg > 0
                      or any("delegation" in c or "delegat" in c for c in capabilities)
                      or any("delegation" in p or "delegat" in p for p in patterns))
    if has_delegation:
        score += 12

    # +10 has shared state management (will trigger state.access events)
    if data_store_count > 0:
        score += 10

    # +8 has conditional routing (will trigger routing.decision events)
    has_routing = (any("routing" in c or "conditional" in c or "router" in c for c in capabilities)
                   or any("routing" in p or "conditional" in p or "router" in p for p in patterns)
                   or "routing" in deployment_strs.lower()
                   or "router" in blob)
    if has_routing:
        score += 8

    # +8 agent_count between 3-7 (non-trivial topology, simple enough to run)
    if 3 <= agent_count <= 7:
        score += 8

    # +5 has error handling patterns (will trigger error propagation traces)
    has_error_handling = (any("error" in c or "exception" in c or "retry" in c for c in capabilities)
                         or any("error" in p or "exception" in p or "retry" in p for p in patterns)
                         or any("error" in c or "fallback" in c for c in capabilities))
    if has_error_handling:
        score += 5

    # XCOMP multiplier: repos with cross-component interactions score 1.7x
    if is_xcomp(r):
        score *= 1.7

    return round(score, 2)


def load_scan_results(path: Path, *, include_forks: bool = False,
                      do_dedup: bool = False,
                      ) -> list[tuple[float, str, dict]]:
    """Load scan_results.jsonl entries, score them, and return (score, url, raw_data) tuples.

    The raw_data dict is preserved for downstream XCOMP and topology analysis.
    """
    raw_entries: list[tuple[float, str, dict]] = []
    with open(path) as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: bad JSON at line {num}", file=sys.stderr)
                continue
            # skip forks unless --include-forks
            if r.get("fork", False) and not include_forks:
                continue
            url = (r.get("repo_url", "") or r.get("metadata", {}).get("repo_url", "")
                   or r.get("url", ""))
            if url:
                raw_entries.append((score_repo(r), url, r))
    if do_dedup:
        raw_entries = deduplicate(raw_entries)
    return raw_entries

# -- Mode B: plain URL list ---------------------------------------------------

def load_plain_urls(path: Path, *, token: str | None = None,
                    include_forks: bool = False,
                    ) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    with open(path) as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            if token:
                or_seg = _owner_repo(url)
                if or_seg:
                    meta = _check_repo_meta(or_seg, token)
                    if meta.get("archived"):
                        print(f"Skipping archived: {url}", file=sys.stderr)
                        continue
                    if meta.get("is_fork") and not include_forks:
                        print(f"Skipping fork: {url}", file=sys.stderr)
                        continue
                    push_iso = meta.get("last_push_iso", "")
                    if push_iso:
                        try:
                            pushed = datetime.fromisoformat(
                                push_iso.replace("Z", "+00:00"))
                            age_days = (datetime.now(timezone.utc) - pushed).days
                            if age_days > 730:      # 2 years
                                print(f"Skipping stale (>{age_days}d): {url}",
                                      file=sys.stderr)
                                continue
                        except ValueError:
                            pass
            out.append((0.0, url))
    return out

# -- Mode C: GitHub discovery -------------------------------------------------

def _gh_search(query: str, token: str | None, per_page: int = 100,
               max_pages: int = 10) -> list[dict]:
    """Paginated GitHub search with rate-limit handling."""
    import requests
    now = datetime.now(timezone.utc)
    since = now.replace(year=now.year - 1).strftime("%Y-%m-%d")
    all_items: list[dict] = []

    for page in range(1, max_pages + 1):
        resp = _gh_get(
            "https://api.github.com/search/repositories",
            params={"q": f"{query} language:python stars:>10 pushed:>{since}",
                    "sort": "stars", "order": "desc",
                    "per_page": per_page, "page": page},
            token=token,
        )
        if resp is None or resp.status_code != 200:
            break
        items = resp.json().get("items", [])
        if not items:
            break
        all_items.extend(items)
        # GitHub search caps at 1000 results per query
        if len(all_items) >= 1000:
            break
    return all_items


def _score_gh(repo: dict) -> float:
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    score = 15.0 if stars >= 1000 else 10.0 if stars >= 100 else 5.0 if stars >= 30 else 2.0
    score += min(forks * 0.5, 5)
    pushed = repo.get("pushed_at", "")
    if pushed:
        try:
            days = (datetime.now(timezone.utc)
                    - datetime.fromisoformat(pushed.replace("Z", "+00:00"))).days
            score += 5 if days < 30 else 3 if days < 90 else 1 if days < 180 else 0
        except ValueError:
            pass
    return round(score, 2)


def discover_repos(*, token: str | None = None,
                   include_forks: bool = False) -> list[tuple[float, str]]:
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: --github-token or GITHUB_TOKEN env var required for "
              "--discover", file=sys.stderr)
        sys.exit(1)
    seen: dict[str, float] = {}
    for q in DISCOVER_QUERIES:
        for repo in _gh_search(q, token):
            # skip forks unless --include-forks
            if repo.get("fork", False) and not include_forks:
                continue
            # skip archived
            if repo.get("archived", False):
                continue
            url = repo.get("html_url", "")
            if url:
                s = _score_gh(repo)
                if url not in seen or s > seen[url]:
                    seen[url] = s
            if len(seen) >= MAX_DISCOVER_URLS:
                break
        if len(seen) >= MAX_DISCOVER_URLS:
            break
    return [(s, u) for u, s in seen.items()]

# -- CLI ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Select and rank repos for behavioral scanning.")
    p.add_argument("input_file", nargs="?", default=None,
                   help="Path to scan_results.jsonl or repo_urls.txt")
    p.add_argument("--plain", action="store_true",
                   help="Input is a plain text file of URLs (one per line)")
    p.add_argument("--discover", action="store_true",
                   help="Search GitHub API for AI-agent repos")
    p.add_argument("--top", type=int, default=0,
                   help="Output top N repos (0=all)")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Minimum score threshold")
    p.add_argument("-o", "--output", default="-",
                   help="Output file (default: stdout)")
    p.add_argument("--github-token", default=None,
                   help="GitHub personal access token for API calls")
    p.add_argument("--include-forks", action="store_true",
                   help="Include forked repos (default: skip)")
    p.add_argument("--deduplicate", action="store_true",
                   help="Enable hash-based duplicate detection")
    p.add_argument("--topology-diversity", action="store_true",
                   help="Rerank selection to maximize unique topology patterns")
    args = p.parse_args()

    if args.discover and args.input_file:
        p.error("--discover does not take an input file")
    if not args.discover and not args.input_file:
        p.error("an input file is required (or use --discover)")

    token = args.github_token or os.environ.get("GITHUB_TOKEN", "")

    # scored_full carries (score, url, raw_data) for XCOMP/topology analysis.
    # Modes that lack raw scan data use an empty dict as the third element.
    has_raw_data = False

    if args.discover:
        scored_2 = discover_repos(token=token,
                                  include_forks=args.include_forks)
        scored_full = [(s, u, {}) for s, u in scored_2]
    elif args.plain:
        scored_2 = load_plain_urls(Path(args.input_file), token=token or None,
                                   include_forks=args.include_forks)
        scored_full = [(s, u, {}) for s, u in scored_2]
    else:
        scored_full = load_scan_results(Path(args.input_file),
                                        include_forks=args.include_forks,
                                        do_dedup=args.deduplicate)
        has_raw_data = True

    scored_full.sort(key=lambda x: x[0], reverse=True)
    if args.min_score > 0:
        scored_full = [(s, u, r) for s, u, r in scored_full if s >= args.min_score]

    # Apply topology diversity reranking before --top cutoff
    if args.topology_diversity and has_raw_data:
        scored_full = rerank_for_topology_diversity(scored_full, top_n=args.top)

    if args.top > 0:
        scored_full = scored_full[:args.top]

    # Write output (score + url)
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        for score, url, _ in scored_full:
            out.write(f"{score}\t{url}\n")
    finally:
        if out is not sys.stdout:
            out.close()

    # -- Summary stats --
    n_selected = len(scored_full)
    print(f"Selected {n_selected} repos", file=sys.stderr)

    # XCOMP and topology stats (only meaningful with scan-results data)
    if has_raw_data and n_selected > 0:
        xcomp_count = sum(1 for _, _, r in scored_full if is_xcomp(r))
        topo_hashes = {compute_topology_hash(r) for _, _, r in scored_full}
        unique_topos = len(topo_hashes)
        diversity_ratio = unique_topos / n_selected if n_selected > 0 else 0.0

        print(f"XCOMP repos: {xcomp_count}/{n_selected} "
              f"({100 * xcomp_count / n_selected:.1f}%)", file=sys.stderr)
        print(f"Unique topologies: {unique_topos}/{n_selected} "
              f"(diversity {100 * diversity_ratio:.1f}%"
              f"{' -- target 86%+ met' if diversity_ratio >= 0.86 else ' -- below 86% target'})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
