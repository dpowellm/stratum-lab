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
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FRAMEWORKS = {"crewai", "langgraph", "autogen", "langchain"}
SERVERS = {"fastapi", "flask", "uvicorn", "starlette", "django"}
EXT_APIS = {"exa", "tavily", "serper", "serpapi", "brave_search",
            "bing", "wolfram", "google_search"}
DISCOVER_QUERIES = ["crewai agent", "langgraph agent",
                    "autogen multi-agent", "langchain agent"]

# -- Mode A: scan-results scoring ------------------------------------------

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
    return round(score, 2)

def load_scan_results(path: Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
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
            url = (r.get("repo_url", "") or r.get("metadata", {}).get("repo_url", "")
                   or r.get("url", ""))
            if url:
                out.append((score_repo(r), url))
    return out

# -- Mode B: plain URL list ------------------------------------------------

def load_plain_urls(path: Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    with open(path) as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith("#"):
                out.append((0.0, url))
    return out

# -- Mode C: GitHub discovery ----------------------------------------------

def _gh_search(query: str, token: str, per_page: int = 30) -> list[dict]:
    import requests
    now = datetime.now(timezone.utc)
    since = now.replace(year=now.year - 1).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": f"{query} language:python stars:>10 pushed:>{since}",
                "sort": "stars", "order": "desc", "per_page": per_page},
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"Warning: GitHub search failed ({resp.status_code}) for '{query}'",
              file=sys.stderr)
        return []
    return resp.json().get("items", [])

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

def discover_repos() -> list[tuple[float, str]]:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: GITHUB_TOKEN env var required for --discover", file=sys.stderr)
        sys.exit(1)
    seen: dict[str, float] = {}
    for q in DISCOVER_QUERIES:
        for repo in _gh_search(q, token):
            url = repo.get("html_url", "")
            if url:
                s = _score_gh(repo)
                if url not in seen or s > seen[url]:
                    seen[url] = s
    return [(s, u) for u, s in seen.items()]

# -- CLI --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Select and rank repos for behavioral scanning.")
    p.add_argument("input_file", nargs="?", default=None,
                   help="Path to scan_results.jsonl or repo_urls.txt")
    p.add_argument("--plain", action="store_true",
                   help="Input is a plain text file of URLs (one per line)")
    p.add_argument("--discover", action="store_true",
                   help="Search GitHub API for AI-agent repos (requires GITHUB_TOKEN)")
    p.add_argument("--top", type=int, default=0, help="Output top N repos (0=all)")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold")
    p.add_argument("-o", "--output", default="-", help="Output file (default: stdout)")
    args = p.parse_args()

    if args.discover and args.input_file:
        p.error("--discover does not take an input file")
    if not args.discover and not args.input_file:
        p.error("an input file is required (or use --discover)")

    if args.discover:
        scored = discover_repos()
    elif args.plain:
        scored = load_plain_urls(Path(args.input_file))
    else:
        scored = load_scan_results(Path(args.input_file))

    scored.sort(key=lambda x: x[0], reverse=True)
    if args.min_score > 0:
        scored = [(s, u) for s, u in scored if s >= args.min_score]
    if args.top > 0:
        scored = scored[:args.top]

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        for score, url in scored:
            out.write(f"{score}\t{url}\n")
    finally:
        if out is not sys.stdout:
            out.close()
    print(f"Selected {len(scored)} repos", file=sys.stderr)

if __name__ == "__main__":
    main()
