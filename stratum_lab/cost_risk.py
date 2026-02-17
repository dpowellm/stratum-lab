"""Cost risk scoring from event stream metrics for stratum-lab.

No LLM calls, no external APIs. Purely deterministic computation on event
streams. Research basis: BATS, industry token economics data.
"""
from __future__ import annotations

from datetime import datetime


def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to epoch seconds. Returns 0.0 on failure."""
    if not ts_str:
        return 0.0
    try:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(ts_str, fmt).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0


def _get_node_id(evt: dict) -> str:
    """Extract node_id from event, handling both nested and flat formats.

    Patcher format: source_node = {"node_type": ..., "node_id": ..., "node_name": ...}
    Flat format:    node_id = "..."
    """
    sn = evt.get("source_node")
    if isinstance(sn, dict):
        return sn.get("node_id", "")
    return evt.get("node_id", "")


def _get_delegation_nodes(evt: dict) -> tuple[str, str]:
    """Extract source and target node IDs from a delegation event.

    Patcher format: source_node/target_node are dicts with node_id key.
    Flat format:    source_node_id/target_node_id are strings, or
                    source_node/target_node are strings.
    """
    # Source
    sn = evt.get("source_node")
    if isinstance(sn, dict):
        src = sn.get("node_id", "")
    else:
        src = evt.get("source_node_id", sn or "")

    # Target
    tn = evt.get("target_node")
    if isinstance(tn, dict):
        tgt = tn.get("node_id", "")
    else:
        tgt = evt.get("target_node_id", tn or "")

    return src, tgt


def _get_input_tokens(evt: dict) -> int:
    """Extract input/prompt token count from event.

    Patcher format: payload.input_tokens
    Flat format:    prompt_tokens or input_tokens at top level
    """
    payload = evt.get("payload")
    if isinstance(payload, dict):
        val = payload.get("input_tokens", payload.get("prompt_tokens", 0))
        return val or 0
    return evt.get("prompt_tokens", evt.get("input_tokens", 0)) or 0


def _get_output_tokens(evt: dict) -> int:
    """Extract output/completion token count from event.

    Patcher format: payload.output_tokens
    Flat format:    completion_tokens or output_tokens at top level
    """
    payload = evt.get("payload")
    if isinstance(payload, dict):
        val = payload.get("output_tokens", payload.get("completion_tokens", 0))
        return val or 0
    return evt.get("completion_tokens", evt.get("output_tokens", 0)) or 0


def _get_latency_ms(evt: dict) -> float:
    """Extract latency_ms from event.

    Patcher format: payload.latency_ms
    Flat format:    latency_ms at top level
    """
    payload = evt.get("payload")
    if isinstance(payload, dict):
        val = payload.get("latency_ms", 0)
        if val:
            return val
    return evt.get("latency_ms", 0) or 0


def compute_token_amplification(runs: list[list[dict]]) -> dict:
    """Compute token amplification across delegation chains per run."""
    all_chains: list[dict] = []

    for events in runs:
        # Sum tokens per node
        node_prompt: dict[str, int] = {}
        node_completion: dict[str, int] = {}
        node_first_prompt: dict[str, int] = {}

        for evt in events:
            if evt.get("event_type") != "llm.call_end":
                continue
            nid = _get_node_id(evt)
            pt = _get_input_tokens(evt)
            ct = _get_output_tokens(evt)
            node_prompt[nid] = node_prompt.get(nid, 0) + pt
            node_completion[nid] = node_completion.get(nid, 0) + ct
            if nid not in node_first_prompt:
                node_first_prompt[nid] = pt

        # Build delegation chain from delegation.initiated events
        delegations: list[tuple[str, str, str]] = []
        for evt in events:
            if evt.get("event_type") == "delegation.initiated":
                src, tgt = _get_delegation_nodes(evt)
                ts = evt.get("timestamp", "")
                if src and tgt:
                    delegations.append((ts, src, tgt))

        delegations.sort()

        # Build chains via BFS from nodes with no incoming delegation
        adj: dict[str, list[str]] = {}
        targets_set: set[str] = set()
        for _, src, tgt in delegations:
            if src not in adj:
                adj[src] = []
            adj[src].append(tgt)
            targets_set.add(tgt)

        sources_set = set(adj.keys())
        roots = sources_set - targets_set

        for root in roots:
            chain = [root]
            current = root
            visited: set[str] = {root}
            while current in adj:
                next_nodes = [n for n in adj[current] if n not in visited]
                if not next_nodes:
                    break
                current = next_nodes[0]
                chain.append(current)
                visited.add(current)

            if len(chain) < 2:
                continue

            first_input = node_first_prompt.get(chain[0], 0)
            total = sum(
                node_prompt.get(n, 0) + node_completion.get(n, 0)
                for n in chain
            )
            ratio = total / max(first_input, 1)

            all_chains.append({
                "chain_nodes": chain,
                "chain_length": len(chain),
                "first_node_input_tokens": first_input,
                "total_chain_tokens": total,
                "amplification_ratio": round(ratio, 2),
            })

    ratios = [c["amplification_ratio"] for c in all_chains]
    return {
        "chains": all_chains,
        "max_amplification_ratio": round(max(ratios, default=0.0), 2),
        "mean_amplification_ratio": round(sum(ratios) / max(len(ratios), 1), 2),
        "high_amplification_chains": sum(1 for r in ratios if r > 20.0),
    }


def compute_tool_call_density(runs: list[list[dict]]) -> dict:
    """Compute LLM and tool call density per delegation hop."""
    node_llm: dict[str, list[int]] = {}
    node_tool: dict[str, list[int]] = {}
    node_deleg: dict[str, list[int]] = {}

    for run_idx, events in enumerate(runs):
        llm_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        deleg_counts: dict[str, int] = {}

        for evt in events:
            etype = evt.get("event_type", "")
            nid = _get_node_id(evt)
            if etype == "llm.call_end" and nid:
                llm_counts[nid] = llm_counts.get(nid, 0) + 1
            elif etype == "tool.call_end" and nid:
                tool_counts[nid] = tool_counts.get(nid, 0) + 1
            elif etype == "delegation.initiated":
                src, _ = _get_delegation_nodes(evt)
                if src:
                    deleg_counts[src] = deleg_counts.get(src, 0) + 1

        all_nodes = set(llm_counts) | set(tool_counts) | set(deleg_counts)
        for nid in all_nodes:
            node_llm.setdefault(nid, []).append(llm_counts.get(nid, 0))
            node_tool.setdefault(nid, []).append(tool_counts.get(nid, 0))
            node_deleg.setdefault(nid, []).append(deleg_counts.get(nid, 0))

    per_node: dict[str, dict] = {}
    densities: list[float] = []
    for nid in node_llm:
        ml = sum(node_llm[nid]) / max(len(node_llm[nid]), 1)
        mt = sum(node_tool.get(nid, [0])) / max(len(node_tool.get(nid, [1])), 1)
        md = sum(node_deleg.get(nid, [0])) / max(len(node_deleg.get(nid, [1])), 1)
        density = (ml + mt) / max(md, 1)
        per_node[nid] = {
            "mean_llm_calls": round(ml, 2),
            "mean_tool_calls": round(mt, 2),
            "mean_delegations": round(md, 2),
            "call_density": round(density, 2),
        }
        densities.append(density)

    return {
        "per_node": per_node,
        "max_call_density": round(max(densities, default=0.0), 2),
        "mean_call_density": round(sum(densities) / max(len(densities), 1), 2),
        "high_density_nodes": sum(1 for d in densities if d > 5.0),
    }


def compute_retry_waste(runs: list[list[dict]]) -> dict:
    """Estimate token waste from retries and errors."""
    node_errors: dict[str, list[int]] = {}
    node_calls: dict[str, list[int]] = {}
    node_prompt_tokens: dict[str, list[int]] = {}

    for events in runs:
        err_counts: dict[str, int] = {}
        call_counts: dict[str, int] = {}
        prompt_sums: dict[str, int] = {}

        for evt in events:
            etype = evt.get("event_type", "")
            nid = _get_node_id(evt)
            if etype == "error.llm_api" and nid:
                err_counts[nid] = err_counts.get(nid, 0) + 1
            elif etype == "llm.call_end" and nid:
                call_counts[nid] = call_counts.get(nid, 0) + 1
                prompt_sums[nid] = prompt_sums.get(nid, 0) + _get_input_tokens(evt)

        all_nodes = set(err_counts) | set(call_counts)
        for nid in all_nodes:
            node_errors.setdefault(nid, []).append(err_counts.get(nid, 0))
            node_calls.setdefault(nid, []).append(call_counts.get(nid, 0))
            node_prompt_tokens.setdefault(nid, []).append(prompt_sums.get(nid, 0))

    per_node: dict[str, dict] = {}
    total_errors = 0
    total_calls = 0
    total_wasted = 0

    for nid in node_errors:
        me = sum(node_errors[nid]) / max(len(node_errors[nid]), 1)
        mc = sum(node_calls.get(nid, [0])) / max(len(node_calls.get(nid, [1])), 1)
        retry_rate = me / max(mc, 1)
        mean_prompt = sum(node_prompt_tokens.get(nid, [0])) / max(sum(node_calls.get(nid, [1])), 1)
        wasted = int(me * mean_prompt)

        per_node[nid] = {
            "mean_error_count": round(me, 2),
            "mean_total_calls": round(mc, 2),
            "retry_rate": round(retry_rate, 3),
            "estimated_wasted_tokens": wasted,
        }
        total_errors += sum(node_errors[nid])
        total_calls += sum(node_calls.get(nid, [0]))
        total_wasted += wasted

    corpus_retry = total_errors / max(total_calls, 1)

    return {
        "per_node": per_node,
        "corpus_retry_rate": round(corpus_retry, 3),
        "total_estimated_wasted_tokens": total_wasted,
        "high_retry_nodes": sum(1 for n in per_node.values() if n["retry_rate"] > 0.2),
    }


def compute_latency_profile(runs: list[list[dict]]) -> dict:
    """Compute end-to-end and per-node latency profile."""
    run_durations: list[float] = []
    run_llm_totals: list[float] = []
    node_latencies: dict[str, list[float]] = {}

    for events in runs:
        timestamps = [_parse_ts(e.get("timestamp", "")) for e in events if e.get("timestamp")]
        timestamps = [t for t in timestamps if t > 0]
        if len(timestamps) >= 2:
            duration = max(timestamps) - min(timestamps)
            run_durations.append(duration)
        else:
            run_durations.append(0.0)

        llm_total_ms = 0.0
        for evt in events:
            if evt.get("event_type") == "llm.call_end":
                nid = _get_node_id(evt)
                lat = _get_latency_ms(evt)
                llm_total_ms += lat
                node_latencies.setdefault(nid, []).append(lat)
        run_llm_totals.append(llm_total_ms)

    mean_duration = sum(run_durations) / max(len(run_durations), 1)
    mean_llm_ms = sum(run_llm_totals) / max(len(run_llm_totals), 1)
    mean_llm_frac = (mean_llm_ms / 1000.0) / max(mean_duration, 0.001)
    mean_llm_frac = min(1.0, max(0.0, mean_llm_frac))

    per_node_lat: dict[str, dict] = {}
    bottleneck_node = ""
    bottleneck_lat = 0.0
    total_lat = sum(sum(v) for v in node_latencies.values())

    for nid, lats in node_latencies.items():
        mean_lat = sum(lats) / max(len(lats), 1)
        frac = sum(lats) / max(total_lat, 1)
        per_node_lat[nid] = {
            "mean_latency_ms": round(mean_lat, 2),
            "fraction_of_total": round(frac, 3),
        }
        if mean_lat > bottleneck_lat:
            bottleneck_lat = mean_lat
            bottleneck_node = nid

    return {
        "mean_run_duration_seconds": round(mean_duration, 2),
        "mean_llm_fraction": round(mean_llm_frac, 3),
        "bottleneck_node": bottleneck_node,
        "bottleneck_latency_ms": round(bottleneck_lat, 2),
        "per_node_latency": per_node_lat,
    }


def compute_cost_risk(runs: list[list[dict]], events_by_run: dict) -> dict:
    """Top-level cost risk computation."""
    run_list = list(events_by_run.values()) if events_by_run else runs

    amplification = compute_token_amplification(run_list)
    density = compute_tool_call_density(run_list)
    retry = compute_retry_waste(run_list)
    latency = compute_latency_profile(run_list)

    total_nodes = len(density.get("per_node", {}))
    amp_risk = min(1.0, amplification["max_amplification_ratio"] / 50.0)
    dens_risk = min(1.0, density["high_density_nodes"] / max(total_nodes, 1))
    ret_risk = min(1.0, retry["corpus_retry_rate"] * 5.0)
    cost_score = round(amp_risk * 0.4 + dens_risk * 0.3 + ret_risk * 0.3, 3)
    monthly_mult = round(amplification["max_amplification_ratio"] * (1 + retry["corpus_retry_rate"]), 2)

    return {
        "token_amplification": amplification,
        "tool_call_density": density,
        "retry_waste": retry,
        "latency_profile": latency,
        "cost_risk_score": cost_score,
        "finding_triggered": cost_score > 0.4,
        "estimated_monthly_cost_multiplier": monthly_mult,
    }
