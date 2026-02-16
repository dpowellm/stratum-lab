#!/usr/bin/env python3
"""Heuristic output classifier for AI agent outputs.

Classifies output_preview text into categories without ML dependencies.

Usage:
    python output_classifier.py <graph_json>
    python output_classifier.py --text "some text" --type str --size 100
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


def classify_output(preview: str, output_type: str, size_bytes: int) -> dict:
    """Returns {"primary": "factual", "confidence": 0.7, "signals": [...]}"""
    if size_bytes < 20:
        return {"primary": "error_empty", "confidence": 0.9, "signals": ["very_short"]}

    signals = []
    scores = {
        "factual": 0, "recommendation": 0, "speculative": 0,
        "refusal": 0, "structured": 0, "generative": 0,
        "delegative": 0, "error_empty": 0
    }

    text = preview.lower()

    # Structured
    if output_type in ("dict", "list") or preview.strip().startswith(("{", "[")):
        scores["structured"] += 3
        signals.append("structured_format")

    # Refusal
    refusal_phrases = ["i cannot", "i can't", "i don't have", "unable to", "i'm not able", "outside my"]
    if any(p in text for p in refusal_phrases):
        scores["refusal"] += 3
        signals.append("refusal_language")

    # Recommendation
    rec_phrases = ["should", "recommend", "suggest", "consider", "advise"]
    rec_count = sum(1 for p in rec_phrases if p in text)
    if rec_count >= 2:
        scores["recommendation"] += 3
        signals.append("recommendation_language")
    elif rec_count == 1:
        scores["recommendation"] += 1

    # Speculative
    spec_phrases = ["might", "could", "possibly", "perhaps", "it seems", "likely", "may be", "unclear", "uncertain"]
    spec_count = sum(1 for p in spec_phrases if p in text)
    if spec_count >= 2:
        scores["speculative"] += 3
        signals.append("hedging_language")
    elif spec_count == 1:
        scores["speculative"] += 1

    # Factual
    has_numbers = bool(re.search(r'\d{2,}', preview))
    has_proper_nouns = bool(re.search(r'[A-Z][a-z]+\s[A-Z]', preview))
    definitive_verbs = ["is", "was", "are", "were", "has", "had", "shows", "indicates"]
    def_count = sum(1 for v in definitive_verbs if f" {v} " in text)
    if has_numbers:
        scores["factual"] += 1
        signals.append("contains_numbers")
    if has_proper_nouns:
        scores["factual"] += 1
        signals.append("proper_nouns")
    if def_count >= 2:
        scores["factual"] += 2
        signals.append("definitive_verbs")

    # Generative
    if size_bytes > 200 and scores["structured"] == 0:
        scores["generative"] += 2
        signals.append("long_narrative")

    # Delegative
    delegation_phrases = ["i'll ask", "let me delegate", "passing to", "forwarding to"]
    if any(p in text for p in delegation_phrases):
        scores["delegative"] += 3
        signals.append("delegation_language")

    primary = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[primary] / max(total, 1)

    return {"primary": primary, "confidence": round(confidence, 2), "signals": signals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heuristic output classifier for AI agent outputs."
    )
    parser.add_argument(
        "graph_json",
        nargs="?",
        default=None,
        help="Path to a graph JSON file (from graph_builder.py output).",
    )
    parser.add_argument("--text", type=str, help="Direct text to classify.")
    parser.add_argument("--type", dest="output_type", type=str, default="str",
                        help="Output type (default: str).")
    parser.add_argument("--size", dest="size_bytes", type=int, default=None,
                        help="Size in bytes (default: len of --text).")

    args = parser.parse_args()

    # Direct classification mode
    if args.text is not None:
        size = args.size_bytes if args.size_bytes is not None else len(args.text.encode("utf-8"))
        result = classify_output(args.text, args.output_type, size)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(0)

    # Graph JSON mode
    if args.graph_json is None:
        parser.print_help()
        sys.exit(1)

    graph_path = Path(args.graph_json)
    if not graph_path.exists():
        print(f"Error: file not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    results = []

    for node in nodes:
        metadata = node.get("metadata", {})
        preview = metadata.get("output_preview")
        output_type = metadata.get("output_type", "str")
        size_bytes = metadata.get("output_size_bytes")

        if preview is None or size_bytes is None:
            continue

        classification = classify_output(preview, output_type, size_bytes)
        results.append({
            "node_id": node.get("id", "unknown"),
            "classification": classification,
        })

    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
