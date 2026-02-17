"""Node ID generation matching stratum-cli's structural graph IDs.

Stratum-cli generates node IDs as:
  - Agents:    agent_{name_lower_underscored}
  - Caps:      cap_{class}_{kind}
  - DataStore: ds_{friendly_name_lower_underscored}
  - External:  ext_{friendly_name_lower_underscored}
  - MCP:       mcp_{name}
  - Guardrail: guard_{kind}_{line}

The patcher generates runtime IDs as:
  framework:ClassName:source_file:line_number

This module provides utilities to map between the two formats.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Convert a name to lowercase underscore form matching stratum-cli."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    name = name.lower().replace(" ", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def structural_agent_id(agent_name: str) -> str:
    """Generate the structural graph agent node ID."""
    return f"agent_{normalize_name(agent_name)}"


def structural_capability_id(class_name: str, kind: str) -> str:
    """Generate the structural graph capability node ID."""
    return f"cap_{class_name}_{kind}"


def structural_data_store_id(friendly_name: str) -> str:
    """Generate the structural graph data store node ID."""
    clean = friendly_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return f"ds_{clean}"


def structural_external_id(service_name: str) -> str:
    """Generate the structural graph external service node ID."""
    clean = service_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return f"ext_{clean}"


def runtime_node_id(framework: str, class_name: str, source_file: str, line_number: int) -> str:
    """Generate a runtime node ID from patcher context.

    Format: framework:ClassName:source_file:line_number
    """
    return f"{framework}:{class_name}:{source_file}:{line_number}"


def match_runtime_to_structural(
    runtime_id: str,
    structural_nodes: dict[str, dict],
) -> str | None:
    """Attempt to match a runtime node ID to a structural graph node.

    Tries multiple matching strategies:
    1. Exact name match (runtime class name -> structural agent name)
    2. Normalized name match
    3. Source file + line number match
    """
    parts = runtime_id.split(":")
    if len(parts) < 4:
        return None

    _framework, class_name, source_file, line_str = parts[0], parts[1], parts[2], parts[3]

    # Strategy 1: Match by normalized agent name
    normalized = normalize_name(class_name)
    agent_id = f"agent_{normalized}"
    if agent_id in structural_nodes:
        logger.debug("node_id match: strategy=exact_name runtime=%s -> %s", runtime_id, agent_id)
        return agent_id

    # Strategy 2: Try without common suffixes
    for suffix in ["_agent", "_crew", "_graph"]:
        if not normalized.endswith(suffix):
            candidate = f"agent_{normalized}{suffix}"
            if candidate in structural_nodes:
                logger.debug("node_id match: strategy=suffix_append runtime=%s -> %s", runtime_id, candidate)
                return candidate

    # Strategy 3: Match by source file and line number
    try:
        line = int(line_str)
    except ValueError:
        line = None

    for node_id, node_data in structural_nodes.items():
        node_file = node_data.get("source_file", "")
        node_line = node_data.get("line_number")
        if node_file and source_file.endswith(node_file) and node_line == line:
            logger.debug("node_id match: strategy=source_file_line runtime=%s -> %s", runtime_id, node_id)
            return node_id

    # Strategy 4: Fuzzy match on name contained in node ID
    for node_id in structural_nodes:
        if normalized in node_id:
            logger.debug("node_id match: strategy=fuzzy_contains runtime=%s -> %s", runtime_id, node_id)
            return node_id

    logger.debug("node_id match: strategy=none runtime=%s -> unmatched", runtime_id)
    return None
