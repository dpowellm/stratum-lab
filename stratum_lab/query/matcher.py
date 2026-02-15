"""Pattern matcher — matches a customer's graph fingerprint against the behavioral dataset.

Compares a structural graph fingerprint against the cross-repo knowledge base
produced by the ``knowledge`` phase.  Three matching strategies are used:

1. **Exact motif match** — the fingerprint contains motifs found in the KB.
2. **Structural similarity** — cosine similarity of feature vectors.
3. **Archetype match** — classify the fingerprint into a behaviorally-known
   archetype (hub-and-spoke, linear chain, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Match:
    """A single match result pairing a fingerprint to a known pattern."""

    pattern_id: str
    pattern_name: str
    similarity_score: float          # 0-1
    match_type: str                  # "exact_motif" | "structural_similarity" | "archetype"
    matched_repos: int
    behavioral_summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Vectors of different lengths are zero-padded to the longer length.
    Returns 0.0 when either vector has zero magnitude.
    """
    # Zero-pad to equal length if needed
    max_len = max(len(a), len(b))
    a_padded = list(a) + [0.0] * (max_len - len(a))
    b_padded = list(b) + [0.0] * (max_len - len(b))

    va = np.asarray(a_padded, dtype=np.float64)
    vb = np.asarray(b_padded, dtype=np.float64)

    dot = float(np.dot(va, vb))
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Knowledge-base file loaders (graceful on missing files)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Feature-vector extraction
# ---------------------------------------------------------------------------

# The feature dimensions used for structural similarity.  Order matters and
# must be identical between the fingerprint being queried and the stored
# knowledge-base fingerprints.
_FEATURE_KEYS: list[str] = [
    "agent_count",
    "capability_count",
    "data_store_count",
    "external_count",
    "edge_count",
    "delegation_depth",
    "has_shared_state_without_arbitration",
    "has_linear_delegation_chain",
    "has_hub_and_spoke",
    "has_feedback_loop",
    "has_trust_boundary_crossing",
    "guardrail_count",
    "mcp_server_count",
]


def _extract_feature_vector(fingerprint: dict) -> list[float]:
    """Turn a fingerprint dict into a fixed-length float vector.

    If the fingerprint already has a ``feature_vector`` key (from
    ``compute_graph_fingerprint``), use it directly.  Otherwise derive
    features from node/edge/motif data.
    """
    # Pre-computed vector takes priority (KB fingerprints have this)
    precomputed = fingerprint.get("feature_vector")
    if precomputed and isinstance(precomputed, list) and len(precomputed) > 0:
        return [float(v) for v in precomputed]

    nodes = fingerprint.get("nodes", {})
    edges = fingerprint.get("edges", {})
    motifs = fingerprint.get("motifs", [])
    motif_names: set[str] = set()
    if isinstance(motifs, list):
        for m in motifs:
            if isinstance(m, str):
                motif_names.add(m)
            else:
                motif_names.add(m.get("motif_name", ""))
    taxonomy = set(fingerprint.get("taxonomy_preconditions", []))

    # Derive simple counts from node types
    def _count_type(ntype: str) -> int:
        return sum(
            1 for n in nodes.values()
            if (n.get("structural", n).get("node_type", "") == ntype)
        )

    # Delegation depth heuristic: longest chain in motifs
    delegation_depth = 0
    for m in (motifs if isinstance(motifs, list) else []):
        if isinstance(m, dict):
            sig = m.get("structural_signature", {})
            delegation_depth = max(delegation_depth, sig.get("chain_length", 0))

    features: dict[str, float] = {
        "agent_count": float(_count_type("agent")),
        "capability_count": float(_count_type("capability")),
        "data_store_count": float(_count_type("data_store")),
        "external_count": float(_count_type("external")),
        "edge_count": float(len(edges)),
        "delegation_depth": float(delegation_depth),
        "has_shared_state_without_arbitration": 1.0 if "shared_state_without_arbitration" in motif_names else 0.0,
        "has_linear_delegation_chain": 1.0 if "linear_delegation_chain" in motif_names else 0.0,
        "has_hub_and_spoke": 1.0 if "hub_and_spoke" in motif_names else 0.0,
        "has_feedback_loop": 1.0 if "feedback_loop" in motif_names else 0.0,
        "has_trust_boundary_crossing": 1.0 if "trust_boundary_crossing" in motif_names else 0.0,
        "guardrail_count": float(_count_type("guardrail")),
        "mcp_server_count": float(_count_type("mcp_server")),
    }

    return [features.get(k, 0.0) for k in _FEATURE_KEYS]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_vector(
    vec: list[float],
    normalization: dict[str, Any] | None,
) -> list[float]:
    """Normalize a feature vector using stored min/max from normalization.json.

    Falls back to the raw vector when normalization constants are unavailable.
    """
    if normalization is None:
        return vec

    mins = normalization.get("min", [])
    maxs = normalization.get("max", [])

    if len(mins) != len(vec) or len(maxs) != len(vec):
        return vec

    normalized: list[float] = []
    for v, lo, hi in zip(vec, mins, maxs):
        span = hi - lo
        if span == 0.0:
            normalized.append(0.0)
        else:
            normalized.append((v - lo) / span)
    return normalized


# ---------------------------------------------------------------------------
# Archetype classification
# ---------------------------------------------------------------------------


def _motif_name(m: Any) -> str:
    """Extract motif name from either a string or dict."""
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        return m.get("motif_name", "")
    return ""


# Simple rule-based archetype classifier.  These correspond to the common
# multi-agent topology families observed in the dataset.  Combined-motif
# rules are checked first so that a graph with *both* a delegation chain
# and shared state isn't classified by whichever single-motif rule happens
# to fire first.
_ARCHETYPE_RULES: list[dict[str, Any]] = [
    # --- Combined motif rules (checked first) ---
    {
        "name": "hierarchical_delegation",
        "detect": lambda fp: (
            _fp_has_motif(fp, "linear_delegation_chain")
            and _fp_has_motif(fp, "shared_state_without_arbitration")
        ),
    },
    {
        "name": "hub_and_spoke_shared_state",
        "detect": lambda fp: (
            _fp_has_motif(fp, "hub_and_spoke")
            and _fp_has_motif(fp, "shared_state_without_arbitration")
        ),
    },
    # --- Single motif rules ---
    {
        "name": "hub_and_spoke",
        "detect": lambda fp: _fp_has_motif(fp, "hub_and_spoke"),
    },
    {
        "name": "linear_pipeline",
        "detect": lambda fp: _fp_has_motif(fp, "linear_delegation_chain"),
    },
    {
        "name": "feedback_loop_system",
        "detect": lambda fp: _fp_has_motif(fp, "feedback_loop"),
    },
    {
        "name": "shared_state_system",
        "detect": lambda fp: _fp_has_motif(fp, "shared_state_without_arbitration"),
    },
    {
        "name": "trust_boundary_system",
        "detect": lambda fp: _fp_has_motif(fp, "trust_boundary_crossing"),
    },
    {
        "name": "simple_agent",
        "detect": lambda fp: (
            sum(
                1 for n in fp.get("nodes", {}).values()
                if n.get("structural", n).get("node_type") == "agent"
            ) <= 1
        ),
    },
]


def _fp_has_motif(fingerprint: dict, motif_name: str) -> bool:
    """Check whether a fingerprint contains the given motif."""
    for m in fingerprint.get("motifs", []):
        if _motif_name(m) == motif_name:
            return True
    return False


def _classify_archetype(
    fingerprint: dict,
    kb_fingerprints: list[dict[str, Any]] | None = None,
    normalization: dict[str, Any] | None = None,
) -> str:
    """Return the archetype name for a fingerprint.

    Tries rule-based classification first, then nearest-neighbor fallback
    using KB fingerprints (if provided).  Only falls back to ``"generic"``
    when no KB repo has similarity > 0.5.
    """
    # 1. Rule-based classification
    for rule in _ARCHETYPE_RULES:
        try:
            if rule["detect"](fingerprint):
                return rule["name"]
        except Exception:
            continue

    # 2. Nearest-neighbor fallback using KB fingerprints
    if kb_fingerprints:
        query_vec = _extract_feature_vector(fingerprint)
        query_norm = _normalize_vector(query_vec, normalization)
        best_sim = 0.0
        best_archetype = "generic"
        for kb_fp in kb_fingerprints:
            kb_vec = _extract_feature_vector(kb_fp)
            kb_norm = _normalize_vector(kb_vec, normalization)
            sim = cosine_similarity(query_norm, kb_norm)
            if sim > best_sim:
                best_sim = sim
                # Classify the KB fingerprint itself to get its archetype
                for rule in _ARCHETYPE_RULES:
                    try:
                        if rule["detect"](kb_fp):
                            best_archetype = rule["name"]
                            break
                    except Exception:
                        continue
        if best_sim > 0.5 and best_archetype != "generic":
            return best_archetype

    return "generic"


# ---------------------------------------------------------------------------
# Strategy 1: Exact motif match
# ---------------------------------------------------------------------------

def _exact_motif_matches(
    fingerprint: dict,
    patterns: list[dict[str, Any]],
) -> list[Match]:
    """Find KB patterns whose motif name appears in the fingerprint's motifs."""
    fp_motif_names: set[str] = set()
    for m in fingerprint.get("motifs", []):
        if isinstance(m, str):
            name = m
        else:
            name = m.get("motif_name", "")
        if name:
            fp_motif_names.add(name)

    matches: list[Match] = []
    for pat in patterns:
        pat_name = pat.get("pattern_name", "")
        if pat_name in fp_motif_names:
            prevalence = pat.get("prevalence", {})
            behavioral = pat.get("behavioral_distribution", {})

            matches.append(Match(
                pattern_id=pat.get("pattern_id", ""),
                pattern_name=pat_name,
                similarity_score=1.0,
                match_type="exact_motif",
                matched_repos=prevalence.get("repos_count", 0),
                behavioral_summary={
                    "failure_rate": behavioral.get("failure_rate", 0.0),
                    "confidence_interval_95": behavioral.get("confidence_interval_95", [0.0, 0.0]),
                    "sample_size": behavioral.get("sample_size", 0),
                    "failure_modes": behavioral.get("failure_modes", {}),
                    "avg_error_rate": behavioral.get("avg_error_rate", 0.0),
                    "risk_level": pat.get("risk_assessment", {}).get("risk_level", "unknown"),
                },
            ))

    return matches


# ---------------------------------------------------------------------------
# Strategy 2: Structural similarity
# ---------------------------------------------------------------------------

def _structural_similarity_matches(
    fingerprint: dict,
    kb_fingerprints: list[dict[str, Any]],
    normalization: dict[str, Any] | None,
    patterns: list[dict[str, Any]],
    top_k: int,
) -> list[Match]:
    """Rank KB fingerprints by cosine similarity to the query fingerprint."""
    query_vec = _extract_feature_vector(fingerprint)
    query_norm = _normalize_vector(query_vec, normalization)

    # Build a lookup from repo_id to any matching pattern data
    repo_patterns: dict[str, list[dict[str, Any]]] = {}
    for pat in patterns:
        for rid in pat.get("prevalence", {}).get("repo_ids", []):
            repo_patterns.setdefault(rid, []).append(pat)

    scored: list[tuple[float, dict[str, Any]]] = []
    for kb_fp in kb_fingerprints:
        kb_vec = _extract_feature_vector(kb_fp)
        kb_norm = _normalize_vector(kb_vec, normalization)
        sim = cosine_similarity(query_norm, kb_norm)
        scored.append((sim, kb_fp))

    # Sort descending by similarity
    scored.sort(key=lambda t: t[0], reverse=True)

    matches: list[Match] = []
    seen_repos: set[str] = set()

    for sim, kb_fp in scored[:top_k]:
        if sim < 0.1:
            break  # Below meaningful similarity

        repo_id = kb_fp.get("repo_id", "unknown")
        if repo_id in seen_repos:
            continue
        seen_repos.add(repo_id)

        # Summarize behavioral data from patterns associated with this repo
        behavioral_summary: dict[str, Any] = {}
        associated_patterns = repo_patterns.get(repo_id, [])
        if associated_patterns:
            pat = associated_patterns[0]
            bd = pat.get("behavioral_distribution", {})
            behavioral_summary = {
                "failure_rate": bd.get("failure_rate", 0.0),
                "sample_size": bd.get("sample_size", 0),
                "risk_level": pat.get("risk_assessment", {}).get("risk_level", "unknown"),
            }

        matches.append(Match(
            pattern_id=f"sim_{repo_id}",
            pattern_name=f"structural_match_{repo_id}",
            similarity_score=round(sim, 4),
            match_type="structural_similarity",
            matched_repos=1,
            behavioral_summary=behavioral_summary,
        ))

    return matches


# ---------------------------------------------------------------------------
# Strategy 3: Archetype match
# ---------------------------------------------------------------------------

def _archetype_matches(
    fingerprint: dict,
    patterns: list[dict[str, Any]],
    kb_fingerprints: list[dict[str, Any]],
    normalization: dict[str, Any] | None = None,
) -> list[Match]:
    """Match fingerprint to patterns sharing the same archetype."""
    query_archetype = _classify_archetype(fingerprint, kb_fingerprints, normalization)

    if query_archetype == "generic":
        return []

    # Motif-to-archetype mapping
    archetype_motifs: dict[str, str] = {
        "hub_and_spoke": "hub_and_spoke",
        "linear_pipeline": "linear_delegation_chain",
        "feedback_loop_system": "feedback_loop",
        "shared_state_system": "shared_state_without_arbitration",
        "trust_boundary_system": "trust_boundary_crossing",
    }

    target_motif = archetype_motifs.get(query_archetype)

    # Collect all patterns whose motif matches the archetype
    matched_patterns: list[dict[str, Any]] = []
    if target_motif:
        for pat in patterns:
            if pat.get("pattern_name") == target_motif:
                matched_patterns.append(pat)

    # Also count KB fingerprints that share this archetype
    archetype_repos = 0
    for kb_fp in kb_fingerprints:
        if _classify_archetype(kb_fp) == query_archetype:
            archetype_repos += 1

    total_kb = len(kb_fingerprints) if kb_fingerprints else 1
    prevalence_rate = archetype_repos / total_kb

    # Aggregate behavioral data across matched patterns
    failure_rates: list[float] = []
    sample_sizes: list[int] = []
    for pat in matched_patterns:
        bd = pat.get("behavioral_distribution", {})
        fr = bd.get("failure_rate")
        if fr is not None:
            failure_rates.append(fr)
        ss = bd.get("sample_size", 0)
        if ss > 0:
            sample_sizes.append(ss)

    avg_failure_rate = float(np.mean(failure_rates)) if failure_rates else 0.0
    total_sample = sum(sample_sizes)

    matches: list[Match] = []
    if archetype_repos > 0:
        matches.append(Match(
            pattern_id=f"archetype_{query_archetype}",
            pattern_name=query_archetype,
            similarity_score=round(prevalence_rate, 4),
            match_type="archetype",
            matched_repos=archetype_repos,
            behavioral_summary={
                "archetype": query_archetype,
                "prevalence_rate": round(prevalence_rate, 4),
                "avg_failure_rate": round(avg_failure_rate, 4),
                "total_sample_size": total_sample,
                "patterns_count": len(matched_patterns),
            },
        ))

    return matches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def match_against_dataset(
    fingerprint: dict,
    knowledge_base_path: Path,
    top_k: int = 10,
) -> list[Match]:
    """Find the closest patterns in the knowledge base for a given fingerprint.

    Three strategies are applied in sequence:

    1. **Exact motif match** — the fingerprint contains motifs present in the
       knowledge base ``patterns.json``.
    2. **Structural similarity** — cosine similarity of feature vectors against
       all fingerprints stored in ``fingerprints.json``.
    3. **Archetype match** — classify the fingerprint into a known archetype
       and surface aggregate behavioral data.

    The following files are loaded from *knowledge_base_path* (missing files
    are silently skipped):

    - ``patterns.json`` — for motif matching
    - ``taxonomy_probabilities.json`` — for precondition lookup
    - ``framework_comparisons.json`` — for framework-specific data
    - ``fragility_map.json`` — for model sensitivity
    - ``fingerprints.json`` — all enriched graph fingerprints
    - ``normalization.json`` — normalization constants

    Parameters
    ----------
    fingerprint:
        A dict representing the customer graph's structural fingerprint.
        Expected keys include ``nodes``, ``edges``, ``motifs``, and
        ``taxonomy_preconditions``.
    knowledge_base_path:
        Path to the directory containing knowledge-base JSON files.
    top_k:
        Maximum number of structural-similarity matches to return.

    Returns
    -------
    List of :class:`Match` objects sorted by ``similarity_score`` descending.
    """
    kb = Path(knowledge_base_path)

    # Load knowledge-base assets
    patterns: list[dict[str, Any]] = _load_json(kb / "patterns.json") or []
    _raw_fps = _load_json(kb / "fingerprints.json") or {}
    # fingerprints.json may be a dict (repo_id -> fp) or a list of fp dicts
    if isinstance(_raw_fps, dict):
        kb_fingerprints: list[dict[str, Any]] = list(_raw_fps.values())
    else:
        kb_fingerprints: list[dict[str, Any]] = _raw_fps
    normalization: dict[str, Any] | None = _load_json(kb / "normalization.json")
    # The following are loaded for potential enrichment of match summaries
    _taxonomy_probs: dict[str, Any] | None = _load_json(kb / "taxonomy_probabilities.json")
    _framework_cmp: list[dict[str, Any]] | None = _load_json(kb / "framework_comparisons.json")
    _fragility_map: list[dict[str, Any]] | None = _load_json(kb / "fragility_map.json")

    all_matches: list[Match] = []

    # Strategy 1: exact motif match
    if patterns:
        all_matches.extend(_exact_motif_matches(fingerprint, patterns))

    # Strategy 2: structural similarity
    if kb_fingerprints:
        all_matches.extend(
            _structural_similarity_matches(
                fingerprint, kb_fingerprints, normalization, patterns, top_k,
            )
        )

    # Strategy 3: archetype match
    all_matches.extend(
        _archetype_matches(fingerprint, patterns, kb_fingerprints, normalization)
    )

    # Enrich matches with fragility flags where applicable
    if _fragility_map:
        _enrich_fragility(all_matches, _fragility_map)

    # Enrich matches with framework comparison data
    if _framework_cmp:
        _enrich_framework_data(all_matches, _framework_cmp)

    # Enrich matches with taxonomy probability data
    if _taxonomy_probs:
        _enrich_taxonomy_data(all_matches, _taxonomy_probs, fingerprint)

    # Deduplicate by pattern_id, keeping highest similarity
    seen: dict[str, Match] = {}
    for m in all_matches:
        existing = seen.get(m.pattern_id)
        if existing is None or m.similarity_score > existing.similarity_score:
            seen[m.pattern_id] = m
    deduped = list(seen.values())

    # Sort by similarity score descending
    deduped.sort(key=lambda m: m.similarity_score, reverse=True)

    return deduped


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def _enrich_fragility(
    matches: list[Match],
    fragility_map: list[dict[str, Any]],
) -> None:
    """Add fragility information to match behavioral summaries."""
    # Build a quick lookup: structural_position -> fragility entry
    fragility_by_role: dict[str, dict[str, Any]] = {}
    for entry in fragility_map:
        role = entry.get("structural_position", "")
        if role:
            fragility_by_role[role] = entry

    for m in matches:
        # For hub_and_spoke patterns, look up hub_node fragility
        if "hub" in m.pattern_name.lower():
            hub_frag = fragility_by_role.get("hub_node")
            if hub_frag:
                m.behavioral_summary["fragility"] = {
                    "position": "hub_node",
                    "sensitivity_score": hub_frag.get("sensitivity_score", 0.0),
                    "avg_tool_failure_rate": hub_frag.get("avg_tool_call_failure_rate", 0.0),
                }
        elif "chain" in m.pattern_name.lower() or "linear" in m.pattern_name.lower():
            chain_frag = fragility_by_role.get("chain_node")
            if chain_frag:
                m.behavioral_summary["fragility"] = {
                    "position": "chain_node",
                    "sensitivity_score": chain_frag.get("sensitivity_score", 0.0),
                    "avg_tool_failure_rate": chain_frag.get("avg_tool_call_failure_rate", 0.0),
                }


def _enrich_framework_data(
    matches: list[Match],
    framework_comparisons: list[dict[str, Any]],
) -> None:
    """Add per-framework comparison data to match summaries."""
    # Build lookup: motif_name -> comparison data
    cmp_by_motif: dict[str, dict[str, Any]] = {}
    for cmp in framework_comparisons:
        motif = cmp.get("motif_name", "")
        if motif:
            cmp_by_motif[motif] = cmp

    for m in matches:
        if m.match_type == "exact_motif":
            cmp = cmp_by_motif.get(m.pattern_name)
            if cmp:
                m.behavioral_summary["framework_comparison"] = {
                    "frameworks_compared": cmp.get("frameworks_compared", []),
                    "per_framework": {
                        fw: {
                            "repos_count": data.get("repos_count", 0),
                            "failure_rate": data.get("behavioral_distribution", {}).get("failure_rate", 0.0),
                        }
                        for fw, data in cmp.get("per_framework", {}).items()
                    },
                }


def _enrich_taxonomy_data(
    matches: list[Match],
    taxonomy_probs: dict[str, Any],
    fingerprint: dict,
) -> None:
    """Add taxonomy probability context to match summaries."""
    fp_preconditions = set(fingerprint.get("taxonomy_preconditions", []))

    if not fp_preconditions:
        return

    # Map motif names to related preconditions
    motif_precondition_map: dict[str, list[str]] = {
        "shared_state_without_arbitration": ["shared_state_no_arbitration"],
        "linear_delegation_chain": ["unbounded_delegation_depth", "no_timeout_on_delegation"],
        "hub_and_spoke": ["single_point_of_failure"],
        "feedback_loop": ["circular_delegation", "unbounded_iteration_loop"],
        "trust_boundary_crossing": ["trust_boundary_no_sanitization", "no_fallback_for_external"],
    }

    for m in matches:
        if m.match_type != "exact_motif":
            continue

        related_pcs = motif_precondition_map.get(m.pattern_name, [])
        taxonomy_context: list[dict[str, Any]] = []

        for pc_id in related_pcs:
            if pc_id in fp_preconditions:
                prob_data = taxonomy_probs.get(pc_id, {})
                if isinstance(prob_data, dict) and prob_data.get("sample_size", 0) > 0:
                    taxonomy_context.append({
                        "precondition_id": pc_id,
                        "manifestation_probability": prob_data.get("probability", 0.0),
                        "sample_size": prob_data.get("sample_size", 0),
                        "severity": prob_data.get("severity_when_manifested", {}),
                    })

        if taxonomy_context:
            m.behavioral_summary["taxonomy_context"] = taxonomy_context
