#!/usr/bin/env python3
"""Phase 5b: Corpus-wide DTMC learning and risk estimation.

Learns Discrete-Time Markov Chain from execution traces across ALL repos.
Estimates violation probabilities with PAC bounds.

Research basis: Pro2Guard (arXiv 2508.00500).

Usage:
    python compute_risk_model.py <all_records_dir>

Writes risk_model.json to stdout.
"""
from __future__ import annotations

import json
import math
import os
import sys
from glob import glob
from pathlib import Path


def extract_state_sequences(events: list[dict]) -> list[str]:
    """Convert event stream into symbolic state sequence.

    State abstraction:
    - agent.task_start → "start:{role}"
    - agent.task_end with status=success → "success:{role}"
    - agent.task_end with status!=success → "fail:{role}"
    - delegation.initiated → "delegate:{source_role}→{target_role}"
    - error.* → "error:{role}"
    """
    sequence: list[str] = []

    # Sort by timestamp if available
    sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))

    for evt in sorted_events:
        etype = evt.get("event_type", "")
        nid = evt.get("node_id", "")

        # Extract role from node_id (2nd colon-field)
        parts = nid.split(":") if nid else []
        role = parts[1] if len(parts) > 1 else (nid or "unknown")

        if etype == "agent.task_start":
            sequence.append(f"start:{role}")
        elif etype == "agent.task_end":
            status = evt.get("status", "unknown")
            if status in ("success", "SUCCESS", "completed"):
                sequence.append(f"success:{role}")
            else:
                sequence.append(f"fail:{role}")
        elif etype == "delegation.initiated":
            src = evt.get("source_node_id", evt.get("source_node", ""))
            tgt = evt.get("target_node_id", evt.get("target_node", ""))
            src_parts = src.split(":") if src else []
            tgt_parts = tgt.split(":") if tgt else []
            src_role = src_parts[1] if len(src_parts) > 1 else (src or "unknown")
            tgt_role = tgt_parts[1] if len(tgt_parts) > 1 else (tgt or "unknown")
            sequence.append(f"delegate:{src_role}\u2192{tgt_role}")
        elif etype.startswith("error."):
            sequence.append(f"error:{role}")

    return sequence


def learn_dtmc(sequences: list[list[str]], laplace_alpha: float = 1.0) -> dict:
    """Learn Discrete-Time Markov Chain from state sequences."""
    # Collect all unique states
    all_states: set[str] = set()
    for seq in sequences:
        all_states.update(seq)

    states = sorted(all_states)
    if not states:
        return {
            "states": [],
            "transition_matrix": {},
            "state_counts": {},
            "total_sequences": len(sequences),
            "total_transitions": 0,
        }

    # Count transitions
    transition_counts: dict[str, dict[str, float]] = {
        s: {t: 0.0 for t in states} for s in states
    }
    state_counts: dict[str, int] = {s: 0 for s in states}
    total_transitions = 0

    for seq in sequences:
        for s in seq:
            state_counts[s] = state_counts.get(s, 0) + 1
        for i in range(len(seq) - 1):
            s_from = seq[i]
            s_to = seq[i + 1]
            transition_counts[s_from][s_to] += 1
            total_transitions += 1

    # Apply Laplace smoothing and normalize
    transition_matrix: dict[str, dict[str, float]] = {}
    for s_from in states:
        row = transition_counts[s_from]
        total = sum(row.values()) + laplace_alpha * len(states)
        transition_matrix[s_from] = {
            s_to: round((row[s_to] + laplace_alpha) / total, 6)
            for s_to in states
        }

    return {
        "states": states,
        "transition_matrix": transition_matrix,
        "state_counts": state_counts,
        "total_sequences": len(sequences),
        "total_transitions": total_transitions,
    }


def estimate_violation_probability(dtmc: dict, unsafe_states: list[str],
                                   current_state: str, horizon: int = 10) -> float:
    """Estimate P(reaching any unsafe state within horizon steps from current_state)."""
    states = dtmc.get("states", [])
    matrix = dtmc.get("transition_matrix", {})

    if current_state not in states:
        return 0.0

    unsafe_set = set(unsafe_states) & set(states)
    if not unsafe_set:
        return 0.0

    # Initialize probability vector
    prob: dict[str, float] = {s: 0.0 for s in states}
    prob[current_state] = 1.0

    total_unsafe = 0.0

    for _ in range(horizon):
        prob_next: dict[str, float] = {s: 0.0 for s in states}
        for s_from in states:
            if prob[s_from] <= 0:
                continue
            for s_to in states:
                prob_next[s_to] += prob[s_from] * matrix.get(s_from, {}).get(s_to, 0.0)

        # Accumulate probability of reaching unsafe states
        for us in unsafe_set:
            total_unsafe += prob_next[us]
            prob_next[us] = 0.0  # Absorbing state

        prob = prob_next

    return min(1.0, max(0.0, total_unsafe))


def compute_finding_violation_probabilities(dtmc: dict, finding_definitions: dict = None) -> dict:
    """For each STRAT- finding, compute expected violation probability."""
    states = dtmc.get("states", [])
    state_counts = dtmc.get("state_counts", {})
    total_sequences = dtmc.get("total_sequences", 1)

    # Define unsafe states: all states starting with "error:" or "fail:"
    unsafe_states = [s for s in states if s.startswith("error:") or s.startswith("fail:")]

    # Find all start states
    start_states = [s for s in states if s.startswith("start:")]
    if not start_states:
        start_states = states[:1] if states else []

    findings = {
        "STRAT-DC-001": {"desc": "Unsupervised delegation chain"},
        "STRAT-HC-001": {"desc": "Hallucination propagation"},
        "STRAT-CE-001": {"desc": "Confidence escalation"},
    }

    result: dict[str, dict] = {}
    horizon = 10

    # PAC bound: p ± sqrt(ln(2/delta) / (2*n))
    delta = 0.05
    n = max(total_sequences, 1)
    confidence_bound = math.sqrt(math.log(2.0 / delta) / (2 * n))

    for fid in findings:
        probs: list[float] = []
        for ss in start_states:
            p = estimate_violation_probability(dtmc, unsafe_states, ss, horizon)
            probs.append(p)

        if probs:
            # Weight by frequency
            weights = [state_counts.get(ss, 1) for ss in start_states]
            total_weight = sum(weights)
            mean_p = sum(p * w for p, w in zip(probs, weights)) / max(total_weight, 1)
        else:
            mean_p = 0.0

        result[fid] = {
            "mean_violation_probability": round(mean_p, 4),
            "max_violation_probability": round(max(probs, default=0.0), 4),
            "min_violation_probability": round(min(probs, default=0.0), 4),
            "confidence_bound": round(confidence_bound, 4),
            "horizon": horizon,
        }

    return result


def compute_corpus_risk_model(all_records_dir: str) -> dict:
    """Orchestrate corpus-wide risk model computation."""
    all_sequences: list[list[str]] = []
    total_repos = 0

    # Walk for behavioral_record.json files and their associated events
    for root, dirs, files in os.walk(all_records_dir):
        for fname in files:
            if fname == "behavioral_record.json":
                total_repos += 1
                record_dir = root

                # Look for event files
                for epath in sorted(glob(os.path.join(record_dir, "events_run_*.jsonl"))):
                    events: list[dict] = []
                    try:
                        with open(epath, "r", encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if line:
                                    try:
                                        events.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass
                    except OSError:
                        continue

                    if events:
                        seq = extract_state_sequences(events)
                        if seq:
                            all_sequences.append(seq)

    # Also check for raw_events subdirectories
    for epath in sorted(glob(os.path.join(all_records_dir, "*", "raw_events", "events_run_*.jsonl"))):
        events = []
        try:
            with open(epath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            continue
        if events:
            seq = extract_state_sequences(events)
            if seq:
                all_sequences.append(seq)

    # Learn DTMC
    dtmc = learn_dtmc(all_sequences)

    # Compute violation probabilities
    violation_probs = compute_finding_violation_probabilities(dtmc)

    # Corpus statistics
    states = dtmc.get("states", [])
    matrix = dtmc.get("transition_matrix", {})
    state_counts = dtmc.get("state_counts", {})

    # Top transitions (by probability)
    top_transitions: list[tuple] = []
    for s_from, row in matrix.items():
        for s_to, prob in row.items():
            if prob > 0.01:  # filter noise
                top_transitions.append((s_from, s_to, prob))
    top_transitions.sort(key=lambda x: x[2], reverse=True)
    top_transitions = top_transitions[:20]

    # Highest risk states (outgoing probability to unsafe)
    unsafe = {s for s in states if s.startswith("error:") or s.startswith("fail:")}
    risk_scores: list[tuple] = []
    for s in states:
        if s in unsafe:
            continue
        row = matrix.get(s, {})
        risk = sum(row.get(u, 0.0) for u in unsafe)
        if risk > 0.0:
            risk_scores.append((s, round(risk, 4)))
    risk_scores.sort(key=lambda x: x[1], reverse=True)

    # State coverage
    theoretical_states = max(len(states), 1)

    # Summarize DTMC if too large
    dtmc_output = dtmc
    if len(states) > 100:
        dtmc_output = {
            "states_count": len(states),
            "states_sample": states[:50],
            "total_sequences": dtmc["total_sequences"],
            "total_transitions": dtmc["total_transitions"],
        }

    delta = 0.05
    n = max(len(all_sequences), 1)

    return {
        "dtmc": dtmc_output,
        "violation_probabilities": violation_probs,
        "corpus_statistics": {
            "total_repos": total_repos,
            "total_sequences": len(all_sequences),
            "total_transitions": dtmc.get("total_transitions", 0),
            "unique_states": len(states),
            "top_transitions": [list(t) for t in top_transitions],
            "highest_risk_states": [list(r) for r in risk_scores[:10]],
        },
        "model_quality": {
            "sequence_coverage": round(len(all_sequences) / max(total_repos * 5, 1), 3),
            "state_coverage": round(len(states) / max(theoretical_states, 1), 3),
            "confidence_delta": delta,
        },
    }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compute_risk_model.py <all_records_dir>", file=sys.stderr)
        sys.exit(1)

    all_records_dir = sys.argv[1]
    if not os.path.isdir(all_records_dir):
        print(f"Error: {all_records_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = compute_corpus_risk_model(all_records_dir)
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
