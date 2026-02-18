#!/usr/bin/env python3
"""Per-repo semantic analysis with five independent LLM analysis passes.

After all 5 runs complete for a repo, performs delegation fidelity, cross-run
consistency, uncertainty propagation chains, confidence escalation, and
topological vulnerability scoring.

Usage:
    python analyze_semantics.py \
        --results-dir <repo_results_dir> \
        --vllm-url <vllm_endpoint> \
        --model <model_id> \
        --output <output_json_path> \
        [--timeout 30] [--max-retries 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

# Add parent directory for stratum_lab imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from stratum_lab.semantic import (
    DELEGATION_FIDELITY_SYSTEM,
    DELEGATION_FIDELITY_USER,
    CONSISTENCY_SYSTEM,
    CONSISTENCY_USER,
    UNCERTAINTY_CHAIN_SYSTEM,
    UNCERTAINTY_CHAIN_USER,
    ESCALATION_SYSTEM,
    ESCALATION_USER,
    validate_pass1_response,
    validate_pass2_response,
    validate_pass3_response,
    validate_pass4_response,
    compute_stability_score,
)
from stratum_lab.vulnerability import compute_vulnerability_scores


# ---------------------------------------------------------------------------
# VLLMClient
# ---------------------------------------------------------------------------

class VLLMClient:
    """Thin wrapper for vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, timeout: int = 30, max_retries: int = 2):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def structured_query(self, system_prompt: str, user_prompt: str) -> dict:
        """Send prompt, parse JSON response. Retry on failure."""
        if requests is None:
            return {"parse_error": "requests library not available"}

        content = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 300,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Extract JSON from response (handle markdown fences)
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt == self.max_retries:
                    return {"parse_error": str(e), "raw_content": content[:200]}
                time.sleep(1)
        return {"parse_error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# Event loading helpers
# ---------------------------------------------------------------------------

def load_all_events(results_dir: str) -> dict:
    """Load events from all run files. Returns {run_id: [events]}."""
    events: dict[str, list] = {}
    for f in sorted(Path(results_dir).glob("events_run_*.jsonl")):
        run_id = f.stem
        run_events: list[dict] = []
        try:
            for line in f.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    try:
                        run_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except (OSError, UnicodeDecodeError):
            continue
        events[run_id] = run_events
    return events


def load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_json(path: str, data: dict) -> None:
    """Write dict as JSON to file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Delegation edge extraction
# ---------------------------------------------------------------------------

def extract_delegation_edges(events: list) -> list:
    """Extract delegation edges from events."""
    edges: list[dict] = []

    # Find explicit delegation pairs
    pending: dict[str, dict] = {}
    for evt in events:
        etype = evt.get("event_type", "")
        if etype == "delegation.initiated":
            key = evt.get("delegation_id") or evt.get("event_id", "")
            pending[key] = evt
        elif etype == "delegation.completed":
            key = evt.get("delegation_id") or evt.get("event_id", "")
            if key in pending:
                edges.append({
                    "initiated": pending.pop(key),
                    "completed": evt,
                })

    # Infer delegation from agent.task_end -> agent.task_start sequences
    task_ends = [e for e in events if e.get("event_type") == "agent.task_end"]
    task_starts = [e for e in events if e.get("event_type") == "agent.task_start"]

    for end_evt in task_ends:
        end_ts = end_evt.get("timestamp", "")
        end_nid = end_evt.get("node_id", end_evt.get("source_node", {}).get("node_id", ""))
        for start_evt in task_starts:
            start_ts = start_evt.get("timestamp", "")
            start_nid = start_evt.get("node_id", start_evt.get("source_node", {}).get("node_id", ""))
            if start_ts > end_ts and start_nid != end_nid:
                # Check not already covered by explicit delegation
                already_covered = any(
                    (e.get("initiated", {}).get("source_node", e.get("initiated", {}).get("node_id", "")) == end_nid and
                     e.get("completed", {}).get("target_node", e.get("completed", {}).get("node_id", "")) == start_nid)
                    for e in edges if "initiated" in e
                )
                if not already_covered:
                    edges.append({
                        "source_task_end": end_evt,
                        "target_task_start": start_evt,
                        "inferred": True,
                    })
                break

    return edges


def get_output_preview(event: dict) -> str:
    """Extract output preview from an event."""
    return (
        event.get("output_preview")
        or event.get("data", {}).get("output_preview")
        or event.get("payload", {}).get("output_preview")
        or ""
    )


def get_prompt_preview(events: list, after_timestamp: str, node_id: str = None) -> str:
    """Find the next llm.call_start after a timestamp, extract prompt preview."""
    for evt in events:
        if evt.get("timestamp", "") <= after_timestamp:
            continue
        if evt.get("event_type") == "llm.call_start":
            if node_id and evt.get("node_id") != node_id:
                continue
            return (
                evt.get("last_user_message_preview")
                or evt.get("data", {}).get("last_user_message_preview")
                or ""
            )
    return ""


def extract_node_outputs(events: list) -> dict:
    """Extract the last output_preview per node_id from a run's events."""
    outputs: dict[str, str] = {}
    for evt in events:
        nid = evt.get("node_id", evt.get("source_node", {}).get("node_id", ""))
        preview = get_output_preview(evt)
        if nid and preview and evt.get("event_type") in ("agent.task_end", "llm.call_end"):
            outputs[nid] = preview
    return outputs


def build_delegation_chain(events: list) -> list:
    """Reconstruct the agent delegation chain from events."""
    chain: list[dict] = []
    seen_nodes: set[str] = set()

    for evt in events:
        nid = evt.get("node_id", evt.get("source_node", {}).get("node_id", ""))
        etype = evt.get("event_type", "")

        if etype == "agent.task_end" and nid and nid not in seen_nodes:
            preview = get_output_preview(evt)
            if preview:
                chain.append({
                    "node_id": nid,
                    "output_preview": preview,
                    "timestamp": evt.get("timestamp", ""),
                })
                seen_nodes.add(nid)

    return chain


# ---------------------------------------------------------------------------
# Pass 1: Delegation Fidelity with MAST Classification
# ---------------------------------------------------------------------------

def pass_1_delegation_fidelity(events_by_run: dict, client: VLLMClient) -> tuple:
    """Evaluate delegation fidelity at each edge in each run."""
    results: list[dict] = []
    call_count = 0

    for run_id, events in events_by_run.items():
        edges = extract_delegation_edges(events)

        for edge in edges:
            if "source_task_end" in edge:
                source_evt = edge["source_task_end"]
                target_evt = edge["target_task_start"]
            else:
                source_evt = edge["initiated"]
                target_evt = edge["completed"]

            source_output = get_output_preview(source_evt)
            source_nid = source_evt.get("node_id", source_evt.get("source_node", {}).get("node_id", ""))
            source_role = source_nid.split(":")[-1] if ":" in source_nid else source_nid or "SourceAgent"

            target_nid = target_evt.get("node_id", target_evt.get("source_node", {}).get("node_id", ""))
            target_input = get_prompt_preview(events, source_evt.get("timestamp", ""), target_nid)
            target_role = target_nid.split(":")[-1] if ":" in target_nid else target_nid or "TargetAgent"

            # Skip if insufficient data
            if len(source_output) < 20 or len(target_input) < 20:
                continue

            source_output = source_output[:500]
            target_input = target_input[:500]

            prompt = DELEGATION_FIDELITY_USER.format(
                role_a=source_role,
                output_a=source_output,
                role_b=target_role,
                input_b=target_input,
            )

            response = client.structured_query(DELEGATION_FIDELITY_SYSTEM, prompt)
            response = validate_pass1_response(response) if "parse_error" not in response else response
            call_count += 1

            results.append({
                "edge": [source_nid, target_nid],
                "run_id": run_id,
                "source_output_len": len(source_output),
                "target_input_len": len(target_input),
                **response,
            })

    return results, call_count


# ---------------------------------------------------------------------------
# Pass 2: Cross-Run Semantic Consistency (ASI-inspired)
# ---------------------------------------------------------------------------

def pass_2_cross_run_consistency(events_by_run: dict, client: VLLMClient) -> tuple:
    """Compare agent outputs across repeat runs."""
    results: list[dict] = []
    call_count = 0

    run_ids = sorted(events_by_run.keys())
    if len(run_ids) < 4:
        return {"per_comparison": results, "node_stability": []}, 0

    run1_id = run_ids[0]
    repeat_ids = run_ids[3:]

    run1_node_outputs = extract_node_outputs(events_by_run[run1_id])

    for repeat_id in repeat_ids:
        repeat_node_outputs = extract_node_outputs(events_by_run[repeat_id])

        for node_id in run1_node_outputs:
            if node_id not in repeat_node_outputs:
                continue

            preview1 = run1_node_outputs[node_id][:500]
            preview_other = repeat_node_outputs[node_id][:500]

            if len(preview1) < 20 or len(preview_other) < 20:
                continue

            role = node_id.split(":")[-1] if ":" in node_id else node_id
            run_num = repeat_id.replace("events_run_", "")

            prompt = CONSISTENCY_USER.format(
                role=role,
                preview_run1=preview1,
                other_run_num=run_num,
                preview_other=preview_other,
            )

            response = client.structured_query(CONSISTENCY_SYSTEM, prompt)
            response = validate_pass2_response(response) if "parse_error" not in response else response
            call_count += 1

            results.append({
                "node_id": node_id,
                "run_pair": [run1_id, repeat_id],
                **response,
            })

    # Compute per-node stability scores
    node_scores: dict[str, list] = {}
    for r in results:
        nid = r["node_id"]
        if nid not in node_scores:
            node_scores[nid] = []

        fa = 1.0 if r.get("factual_agreement") else 0.0
        sa = 1.0 if r.get("structural_agreement") else 0.0
        so = float(r.get("semantic_overlap_estimate", 0.5))

        stability = compute_stability_score(bool(r.get("factual_agreement")), bool(r.get("structural_agreement")), so)
        node_scores[nid].append({
            "stability": stability,
            "novel_claims": r.get("novel_claims_in_other", 0),
            "dropped_claims": r.get("dropped_claims_from_run1", 0),
        })

    stability_summary: list[dict] = []
    for nid, scores in node_scores.items():
        stability_summary.append({
            "node_id": nid,
            "stability_score": round(sum(s["stability"] for s in scores) / len(scores), 3),
            "mean_novel_claims": round(sum(s["novel_claims"] for s in scores) / len(scores), 2),
            "mean_dropped_claims": round(sum(s["dropped_claims"] for s in scores) / len(scores), 2),
            "measurement_count": len(scores),
        })

    return {"per_comparison": results, "node_stability": stability_summary}, call_count


# ---------------------------------------------------------------------------
# Pass 3: Uncertainty Propagation Chain Analysis (UProp-inspired)
# ---------------------------------------------------------------------------

def pass_3_uncertainty_chains(events_by_run: dict, client: VLLMClient) -> tuple:
    """Trace uncertainty propagation through delegation chains."""
    results: list[dict] = []
    call_count = 0

    for run_id, events in events_by_run.items():
        chain = build_delegation_chain(events)

        if len(chain) < 2:
            continue

        chain_text_parts = []
        for i, node in enumerate(chain):
            role = node["node_id"].split(":")[-1] if ":" in node["node_id"] else node["node_id"]
            position = "FIRST" if i == 0 else ("LAST" if i == len(chain) - 1 else f"MIDDLE ({i + 1}/{len(chain)})")
            output = node["output_preview"][:400]
            chain_text_parts.append(f"AGENT {position} ({role}) output:\n{output}")

        chain_text = "\n\n".join(chain_text_parts)

        if len(chain_text) < 50:
            continue

        prompt = UNCERTAINTY_CHAIN_USER.format(chain_text=chain_text)
        response = client.structured_query(UNCERTAINTY_CHAIN_SYSTEM, prompt)
        response = validate_pass3_response(response) if "parse_error" not in response else response
        call_count += 1

        results.append({
            "chain": [n["node_id"] for n in chain],
            "chain_roles": [
                n["node_id"].split(":")[-1] if ":" in n["node_id"] else n["node_id"]
                for n in chain
            ],
            "run_id": run_id,
            "chain_length": len(chain),
            **response,
        })

    return results, call_count


# ---------------------------------------------------------------------------
# Pass 4: Confidence Escalation Under Failure
# ---------------------------------------------------------------------------

def pass_4_confidence_escalation(events_by_run: dict, client: VLLMClient) -> tuple:
    """Detect confidence escalation after failures within individual agents."""
    results: list[dict] = []
    call_count = 0

    for run_id, events in events_by_run.items():
        calls_by_node: dict[str, list] = {}
        error_events: set[tuple[str, str]] = set()

        for evt in events:
            etype = evt.get("event_type", "")
            nid = evt.get("node_id", evt.get("source_node", {}).get("node_id", ""))

            if etype.startswith("error.") and nid:
                error_events.add((nid, evt.get("timestamp", "")))

            if etype == "llm.call_end" and nid:
                if nid not in calls_by_node:
                    calls_by_node[nid] = []
                calls_by_node[nid].append(evt)

        for nid, calls in calls_by_node.items():
            if len(calls) < 2:
                continue

            calls_text_parts = []
            for i, call_evt in enumerate(calls[:5]):
                preview = get_output_preview(call_evt)[:300]
                call_ts = call_evt.get("timestamp", "")
                preceding_error = any(
                    err_nid == nid and err_ts < call_ts
                    for err_nid, err_ts in error_events
                )
                preceding_label = "after ERROR" if preceding_error else "normal"
                calls_text_parts.append(f"CALL {i + 1} ({preceding_label}):\n{preview}")

            calls_text = "\n\n".join(calls_text_parts)

            if len(calls_text) < 30:
                continue

            prompt = ESCALATION_USER.format(calls_text=calls_text)
            response = client.structured_query(ESCALATION_SYSTEM, prompt)
            response = validate_pass4_response(response) if "parse_error" not in response else response
            call_count += 1

            results.append({
                "node_id": nid,
                "run_id": run_id,
                "call_count": len(calls),
                "had_preceding_errors": any(
                    err_nid == nid for err_nid, _ in error_events
                ),
                **response,
            })

    return results, call_count


# ---------------------------------------------------------------------------
# Pass 5: Topological Vulnerability Scoring (Sherlock-inspired, no LLM)
# ---------------------------------------------------------------------------

def pass_5_topological_vulnerability(events_by_run: dict, defensive_patterns: dict) -> list:
    """Compute topology-aware vulnerability score per node."""
    all_nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    error_counts: dict[str, int] = {}

    for run_id, events in events_by_run.items():
        for evt in events:
            nid = evt.get("node_id", evt.get("source_node", {}).get("node_id", ""))
            if nid:
                all_nodes.add(nid)
            etype = evt.get("event_type", "")
            if etype.startswith("error.") and nid:
                error_counts[nid] = error_counts.get(nid, 0) + 1

        for edge in extract_delegation_edges(events):
            if "source_task_end" in edge:
                src = edge["source_task_end"].get("node_id", edge["source_task_end"].get("source_node", {}).get("node_id", ""))
                tgt = edge["target_task_start"].get("node_id", edge["target_task_start"].get("source_node", {}).get("node_id", ""))
            else:
                src = edge["initiated"].get("source_node", edge["initiated"].get("node_id", ""))
                tgt = edge["completed"].get("target_node", edge["completed"].get("node_id", ""))
            if src and tgt:
                edges.append((src, tgt))

    return compute_vulnerability_scores(all_nodes, edges, error_counts, defensive_patterns)


# ---------------------------------------------------------------------------
# Aggregate Score Computation
# ---------------------------------------------------------------------------

def compute_aggregate_scores(results: dict) -> dict:
    """Compute aggregate semantic scores from all passes."""
    scores: dict = {}

    # From Pass 1: delegation fidelity
    df = results.get("delegation_fidelity", [])
    if isinstance(df, list) and df:
        valid = [r for r in df if "parse_error" not in r]
        if valid:
            scores["hedging_preservation_rate"] = round(
                sum(1 for r in valid if r.get("hedging_preserved")) / len(valid), 3
            )
            scores["trust_elevation_rate"] = round(
                sum(
                    1 for r in valid
                    if not r.get("hedging_preserved") and r.get("factual_additions_detected")
                ) / len(valid), 3
            )
            scores["mean_delegation_fidelity"] = round(1.0 - scores["trust_elevation_rate"], 3)

            modes = [r.get("mast_failure_mode", "unknown") for r in valid]
            scores["mast_failure_distribution"] = {
                m: round(modes.count(m) / len(modes), 3) for m in set(modes)
            }

    # From Pass 2: cross-run consistency
    crc = results.get("cross_run_consistency", {})
    if isinstance(crc, dict):
        stability = crc.get("node_stability", [])
        if stability:
            scores["mean_stability_score"] = round(
                sum(s["stability_score"] for s in stability) / len(stability), 3
            )
            scores["min_stability_score"] = round(
                min(s["stability_score"] for s in stability), 3
            )
            scores["mean_novel_claims_per_repeat"] = round(
                sum(s["mean_novel_claims"] for s in stability) / len(stability), 2
            )

    # From Pass 3: uncertainty chains
    uc = results.get("uncertainty_chains", [])
    if isinstance(uc, list) and uc:
        valid = [r for r in uc if "parse_error" not in r]
        if valid:
            scores["mean_chain_fidelity"] = round(
                sum(float(r.get("chain_fidelity", 0.5)) for r in valid) / len(valid), 3
            )
            scores["information_accretion_rate"] = round(
                sum(1 for r in valid if r.get("information_accretion")) / len(valid), 3
            )
            scores["elevation_rate"] = round(
                sum(
                    1 for r in valid
                    if r.get("confidence_at_origin") == "hedged" and r.get("confidence_at_terminus") == "asserted"
                ) / len(valid), 3
            )

    # From Pass 4: confidence escalation
    ce = results.get("confidence_escalation", [])
    if isinstance(ce, list) and ce:
        valid = [r for r in ce if "parse_error" not in r]
        if valid:
            scores["confidence_escalation_rate"] = round(
                sum(1 for r in valid if r.get("confidence_trajectory") == "escalating") / len(valid), 3
            )
            scores["fabrication_risk_rate"] = round(
                sum(1 for r in valid if r.get("fabrication_risk") in ("high", "medium")) / len(valid), 3
            )
            scores["compensatory_assertion_rate"] = round(
                sum(1 for r in valid if r.get("compensatory_assertion")) / len(valid), 3
            )

    # From Pass 5: topological vulnerability
    tv = results.get("topological_vulnerability", [])
    if isinstance(tv, list) and tv:
        scores["max_vulnerability_score"] = round(max(n["vulnerability_score"] for n in tv), 3)
        scores["mean_vulnerability_score"] = round(
            sum(n["vulnerability_score"] for n in tv) / len(tv), 3
        )
        scores["undefended_vulnerable_nodes"] = sum(
            1 for n in tv
            if n["vulnerability_score"] > 0.7 and n["has_defenses"]["defense_count"] == 0
        )

    # Composite: OER estimate
    trust_elev = scores.get("trust_elevation_rate", 0)
    chain_fid = scores.get("mean_chain_fidelity", 1.0)
    fab_risk = scores.get("fabrication_risk_rate", 0)
    scores["oer_estimate"] = round(
        1.0 - ((1.0 - trust_elev) * chain_fid * (1.0 - fab_risk)),
        3,
    )

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-repo semantic analysis with 5 LLM passes.")
    parser.add_argument("--results-dir", required=True, help="Path to repo results directory.")
    parser.add_argument("--vllm-url", required=True, help="URL of vLLM endpoint.")
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-call timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per LLM call.")
    args = parser.parse_args()

    events_by_run = load_all_events(args.results_dir)
    defensive = load_json(os.path.join(args.results_dir, "defensive_patterns.json"))
    client = VLLMClient(args.vllm_url, args.model, args.timeout, args.max_retries)

    results: dict = {}
    total_calls = 0

    # Pass 1
    try:
        delegation_fidelity, calls = pass_1_delegation_fidelity(events_by_run, client)
        results["delegation_fidelity"] = delegation_fidelity
        total_calls += calls
    except Exception as e:
        results["delegation_fidelity"] = {"error": str(e)}

    # Pass 2
    try:
        consistency, calls = pass_2_cross_run_consistency(events_by_run, client)
        results["cross_run_consistency"] = consistency
        total_calls += calls
    except Exception as e:
        results["cross_run_consistency"] = {"error": str(e)}

    # Pass 3
    try:
        chains, calls = pass_3_uncertainty_chains(events_by_run, client)
        results["uncertainty_chains"] = chains
        total_calls += calls
    except Exception as e:
        results["uncertainty_chains"] = {"error": str(e)}

    # Pass 4
    try:
        escalation, calls = pass_4_confidence_escalation(events_by_run, client)
        results["confidence_escalation"] = escalation
        total_calls += calls
    except Exception as e:
        results["confidence_escalation"] = {"error": str(e)}

    # Pass 5 (no LLM)
    try:
        vulnerability = pass_5_topological_vulnerability(events_by_run, defensive)
        results["topological_vulnerability"] = vulnerability
    except Exception as e:
        results["topological_vulnerability"] = {"error": str(e)}

    results["aggregate_scores"] = compute_aggregate_scores(results)
    results["total_llm_calls"] = total_calls
    results["model_used"] = args.model
    results["semantic_analysis_version"] = "1.0"

    write_json(args.output, results)


if __name__ == "__main__":
    main()
