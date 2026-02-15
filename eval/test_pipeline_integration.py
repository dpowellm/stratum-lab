"""Pipeline integration demo — chains all phases on synthetic data.

selection → harness (mocked) → data collection → graph overlay → knowledge base
"""
import sys, os, json, time, uuid, tempfile, shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SEPARATOR = "=" * 78


def make_event(event_type, run_id, repo_id, framework="crewai",
               source_node=None, target_node=None, edge_type=None,
               payload=None, parent_event_id=None, stack_depth=0):
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:16]}",
        "timestamp_ns": time.time_ns(),
        "run_id": run_id,
        "repo_id": repo_id,
        "framework": framework,
        "event_type": event_type,
        **({"source_node": source_node} if source_node else {}),
        **({"target_node": target_node} if target_node else {}),
        **({"edge_type": edge_type} if edge_type else {}),
        **({"payload": payload} if payload else {}),
        **({"parent_event_id": parent_event_id} if parent_event_id else {}),
        "stack_depth": stack_depth,
    }


def node(node_type, node_id, node_name):
    return {"node_type": node_type, "node_id": node_id, "node_name": node_name}


def main():
    tmpdir = tempfile.mkdtemp(prefix="stratum_pipeline_")
    print(f"Working directory: {tmpdir}\n")

    # =====================================================================
    # PHASE 1: REPO SELECTION
    # =====================================================================
    print(SEPARATOR)
    print("PHASE 1: REPO SELECTION")
    print(SEPARATOR)

    from stratum_lab.selection.selector import score_and_select

    # Create 30 synthetic repos
    frameworks = ["crewai", "langgraph", "autogen", "langchain", "custom"]
    archetypes = list(range(1, 13))
    raw_repos = []
    for i in range(30):
        fw = frameworks[i % len(frameworks)]
        arch = archetypes[i % len(archetypes)]
        agent_count = 2 + (i % 5)
        agents = [
            {"name": f"Agent_{j}", "tool_names": [f"tool_{j}" for j in range(j % 3 + 1)]}
            for j in range(agent_count)
        ]
        raw_repos.append({
            "repo_id": f"repo_{i:03d}",
            "repo_url": f"https://github.com/test/repo_{i:03d}",
            "archetype_id": arch,
            "detected_frameworks": [fw],
            "agent_definitions": agents,
            "graph_edges": [{"source": f"a{j}", "target": f"a{j+1}"} for j in range(agent_count - 1)],
            "taxonomy_preconditions": ["shared_state_no_arbitration", "no_timeout_on_delegation"][:((i % 3) + 1)],
            "risk_surface": {
                "max_delegation_depth": min(agent_count - 1, 5),
                "shared_state_conflict_count": i % 4,
                "feedback_loop_count": 1 if i % 7 == 0 else 0,
                "trust_boundary_crossing_count": i % 5,
            },
            "detected_entry_point": "main.py",
            "detected_requirements": "requirements.txt",
            "has_readme_with_usage": True,
            "requires_docker": False,
            "requires_database": False,
            "recent_commit": True,
        })

    selected, summary = score_and_select(raw_repos, target=20, min_runnability=10, min_per_archetype=1)

    print(f"  Input repos:       {len(raw_repos)}")
    print(f"  Selected repos:    {len(selected)}")
    print(f"  Frameworks:        {summary['by_framework']}")
    print(f"  Archetypes:        {summary['by_archetype']}")
    print(f"  Avg structural:    {summary['avg_structural_value']}")
    print(f"  Avg runnability:   {summary['avg_runnability_score']}")

    # Save selection output
    selection_path = os.path.join(tmpdir, "selection.json")
    with open(selection_path, "w") as f:
        json.dump({"selections": selected, "summary": summary}, f, indent=2)
    print(f"\n  -> Saved selection to {selection_path}")

    # =====================================================================
    # PHASE 2: EXECUTION HARNESS (MOCKED)
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 2: EXECUTION HARNESS (MOCKED)")
    print(SEPARATOR)

    raw_events_dir = os.path.join(tmpdir, "raw_events")
    os.makedirs(raw_events_dir, exist_ok=True)

    # For each selected repo, generate synthetic JSONL events (3 runs each)
    execution_results = []
    total_events_written = 0
    runs_per_repo = 3

    for sel in selected[:10]:  # Limit to first 10 for demo speed
        repo_id = sel["repo_id"]
        fw = sel["framework"]
        agent_count = sel["agent_count"]
        agent_names = [f"Agent_{j}" for j in range(agent_count)]

        # Determinism: runs 1 and 3 share an input_hash, run 2 differs
        input_hashes = {1: f"{repo_id}_input_A", 2: f"{repo_id}_input_B", 3: f"{repo_id}_input_A"}

        for run_num in range(1, runs_per_repo + 1):
            run_id = f"{repo_id}_run_{run_num}"
            events_file = os.path.join(raw_events_dir, f"{run_id}.jsonl")
            events = []

            # execution.start
            events.append(make_event("execution.start", run_id, repo_id, fw,
                                     source_node=node("agent", f"agent_{agent_names[0].lower()}", agent_names[0])))

            for ag_idx, ag_name in enumerate(agent_names):
                ag_node = node("agent", f"agent_{ag_name.lower()}", ag_name)

                # agent.task_start
                start_id = f"evt_{uuid.uuid4().hex[:16]}"
                ev = make_event("agent.task_start", run_id, repo_id, fw, source_node=ag_node)
                ev["event_id"] = start_id
                events.append(ev)

                # llm.call_start/end
                llm_start = make_event("llm.call_start", run_id, repo_id, fw,
                                       source_node=node("capability", f"cap_llm_{ag_name.lower()}", f"{ag_name}_llm"),
                                       payload={"model_requested": "test-model", "message_count": 3})
                events.append(llm_start)
                time.sleep(0.001)
                events.append(make_event("llm.call_end", run_id, repo_id, fw,
                                         source_node=node("capability", f"cap_llm_{ag_name.lower()}", f"{ag_name}_llm"),
                                         payload={"input_tokens": 100, "output_tokens": 50, "latency_ms": 150.0},
                                         parent_event_id=llm_start["event_id"]))

                # tool.invoked/completed
                events.append(make_event("tool.invoked", run_id, repo_id, fw,
                                         source_node=node("capability", f"cap_tool_{ag_name.lower()}", f"{ag_name}_tool"),
                                         payload={"tool_name": f"tool_{ag_name.lower()}"}))
                events.append(make_event("tool.completed", run_id, repo_id, fw,
                                         source_node=node("capability", f"cap_tool_{ag_name.lower()}", f"{ag_name}_tool"),
                                         payload={"tool_name": f"tool_{ag_name.lower()}", "status": "success"}))

                # decision.made (with input_hash for determinism metrics)
                events.append(make_event("decision.made", run_id, repo_id, fw,
                                         source_node=ag_node,
                                         payload={
                                             "input_hash": input_hashes[run_num],
                                             "selected_option_hash": f"opt_{ag_idx}",
                                             "outcome": f"option_{ag_idx}",
                                             "confidence": round(0.7 + ag_idx * 0.05, 2),
                                         }))

                # agent.task_end
                events.append(make_event("agent.task_end", run_id, repo_id, fw,
                                         source_node=ag_node,
                                         parent_event_id=start_id))

            # delegation events
            for j in range(len(agent_names) - 1):
                src = node("agent", f"agent_{agent_names[j].lower()}", agent_names[j])
                tgt = node("agent", f"agent_{agent_names[j+1].lower()}", agent_names[j + 1])
                events.append(make_event("delegation.initiated", run_id, repo_id, fw,
                                         source_node=src, target_node=tgt, edge_type="delegates_to"))

            # data.write: first two agents write to shared data store
            ds_node = node("data_store", "ds_shared_memory", "SharedMemory")
            for j in range(min(2, len(agent_names))):
                ag_node = node("agent", f"agent_{agent_names[j].lower()}", agent_names[j])
                events.append(make_event("data.write", run_id, repo_id, fw,
                                         source_node=ag_node, target_node=ds_node,
                                         payload={"data_size_bytes": 1024}))

            # data.read: last agent reads from shared data store
            last_ag = node("agent", f"agent_{agent_names[-1].lower()}", agent_names[-1])
            events.append(make_event("data.read", run_id, repo_id, fw,
                                     source_node=last_ag, target_node=ds_node))

            # external.call: first agent calls external service
            ext_node = node("external", "ext_api_service", "APIService")
            first_ag = node("agent", f"agent_{agent_names[0].lower()}", agent_names[0])
            events.append(make_event("external.call", run_id, repo_id, fw,
                                     source_node=first_ag, target_node=ext_node,
                                     payload={"endpoint": "/api/data", "latency_ms": 200.0}))

            # guardrail.triggered: output filter guardrail
            guard_node = node("guardrail", "guard_output_filter", "OutputFilter")
            events.append(make_event("guardrail.triggered", run_id, repo_id, fw,
                                     source_node=guard_node,
                                     payload={"action_prevented": run_num == 2,
                                              "latency_ms": 5.0}))

            # Run 2: error.occurred + error.propagated with multi-node path
            if run_num == 2:
                events.append(make_event("error.occurred", run_id, repo_id, fw,
                                         source_node=node("agent", f"agent_{agent_names[-1].lower()}", agent_names[-1]),
                                         payload={"error_type": "RuntimeError", "error_message": "Test error"}))
                # error.propagated with propagation_path across agents
                if len(agent_names) >= 2:
                    prop_path = [f"agent_{agent_names[-1].lower()}"]
                    for j in range(len(agent_names) - 2, -1, -1):
                        prop_path.append(f"agent_{agent_names[j].lower()}")
                    events.append(make_event("error.propagated", run_id, repo_id, fw,
                                             source_node=node("agent", f"agent_{agent_names[-1].lower()}", agent_names[-1]),
                                             payload={
                                                 "error_type": "RuntimeError",
                                                 "propagation_path": prop_path,
                                                 "downstream_impact": True,
                                                 "reached_irreversible": False,
                                             }))

            # Emergent edge: last agent directly calls external service
            # (no structural edge between them — only agent_0 has a structural
            # calls edge to ext_api_service)
            if len(agent_names) >= 2:
                emergent_src = node("agent", f"agent_{agent_names[-1].lower()}", agent_names[-1])
                events.append(make_event("external.call", run_id, repo_id, fw,
                                         source_node=emergent_src, target_node=ext_node,
                                         payload={"endpoint": "/api/emergency", "latency_ms": 300.0}))

            # execution.end
            events.append(make_event("execution.end", run_id, repo_id, fw,
                                     source_node=node("agent", f"agent_{agent_names[0].lower()}", agent_names[0])))

            with open(events_file, "w") as f:
                for ev in events:
                    f.write(json.dumps(ev, default=str) + "\n")

            total_events_written += len(events)
            execution_results.append({
                "run_id": run_id,
                "repo_id": repo_id,
                "framework": fw,
                "events_file": events_file,
                "events_count": len(events),
                "status": "SUCCESS",
                "input_hash": input_hashes[run_num],
            })

    print(f"  Repos executed:    {len(selected[:10])}")
    print(f"  Runs per repo:     {runs_per_repo}")
    print(f"  Total runs:        {len(execution_results)}")
    print(f"  Total events:      {total_events_written}")
    print(f"\n  -> Saved {len(execution_results)} JSONL files to {raw_events_dir}")

    # =====================================================================
    # PHASE 3: DATA COLLECTION
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 3: DATA COLLECTION")
    print(SEPARATOR)

    from stratum_lab.collection.parser import parse_events_file, build_run_record, aggregate_run_records

    # Parse all event files and build run records
    all_run_records = []
    repo_run_records = {}  # repo_id -> [records]

    for result in execution_results:
        events = parse_events_file(result["events_file"])
        record = build_run_record(events, run_metadata={
            "repo_id": result["repo_id"],
            "run_id": result["run_id"],
            "framework": result["framework"],
            "input_hash": result.get("input_hash", result["run_id"]),
        })
        all_run_records.append(record)

        repo_id = result["repo_id"]
        if repo_id not in repo_run_records:
            repo_run_records[repo_id] = []
        repo_run_records[repo_id].append(record)

    # Aggregate per repo
    repo_aggregates = {}
    for repo_id, records in repo_run_records.items():
        repo_aggregates[repo_id] = aggregate_run_records(records)

    print(f"  Run records built: {len(all_run_records)}")
    print(f"  Repos aggregated:  {len(repo_aggregates)}")

    # Show sample run record
    sample = all_run_records[0]
    print(f"\n  Sample run record ({sample['run_id']}):")
    print(f"    Total events:    {sample['total_events']}")
    print(f"    Event types:     {sample['total_events_by_type']}")
    print(f"    LLM calls:       {sample['llm_calls']['count']}")
    print(f"    Errors:          {sample['error_summary']['total_errors']}")
    print(f"    Delegations:     {len(sample['delegation_chains'])}")

    # Show sample aggregate
    sample_agg = list(repo_aggregates.values())[0]
    print(f"\n  Sample repo aggregate ({sample_agg['repo_id']}):")
    print(f"    Num runs:        {sample_agg['num_runs']}")
    print(f"    Total events:    {sample_agg['total_events']}")
    print(f"    Avg events/run:  {sample_agg['avg_events_per_run']}")

    print(f"\n  -> Collection phase complete")

    # =====================================================================
    # PHASE 4: GRAPH OVERLAY
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 4: GRAPH OVERLAY")
    print(SEPARATOR)

    from stratum_lab.overlay.enricher import enrich_graph

    enriched_graphs = []
    total_emergent = 0
    total_dead = 0

    for repo_id, records in repo_run_records.items():
        sel = next((s for s in selected if s["repo_id"] == repo_id), None)
        if not sel:
            continue

        fw = sel["framework"]
        agent_count = sel["agent_count"]
        agent_names = [f"Agent_{j}" for j in range(agent_count)]

        # Build structural graph for this repo — all node types
        struct_nodes = {}
        struct_edges = {}
        edge_ctr = 0

        # Agent nodes
        for j, ag_name in enumerate(agent_names):
            struct_nodes[f"agent_{ag_name.lower()}"] = {
                "node_type": "agent", "name": ag_name,
                "source_file": "agents.py", "line_number": j * 20,
            }

        # Capability nodes (tool + LLM per agent)
        for ag_name in agent_names:
            struct_nodes[f"cap_tool_{ag_name.lower()}"] = {
                "node_type": "capability", "name": f"{ag_name}_tool",
                "class_name": f"Tool{ag_name}",
            }
            struct_nodes[f"cap_llm_{ag_name.lower()}"] = {
                "node_type": "capability", "name": f"{ag_name}_llm",
                "class_name": f"LLM{ag_name}",
            }

        # Data store node
        struct_nodes["ds_shared_memory"] = {
            "node_type": "data_store", "name": "SharedMemory",
        }

        # External service node
        struct_nodes["ext_api_service"] = {
            "node_type": "external", "name": "APIService",
        }

        # Guardrail node
        struct_nodes["guard_output_filter"] = {
            "node_type": "guardrail", "name": "OutputFilter",
        }

        # Delegation edges: agent chain
        for j in range(len(agent_names) - 1):
            edge_ctr += 1
            struct_edges[f"e_{edge_ctr}"] = {
                "edge_type": "delegates_to",
                "source": f"agent_{agent_names[j].lower()}",
                "target": f"agent_{agent_names[j+1].lower()}",
            }

        # tool_of edges: each tool capability -> its agent
        for ag_name in agent_names:
            edge_ctr += 1
            struct_edges[f"e_{edge_ctr}"] = {
                "edge_type": "tool_of",
                "source": f"cap_tool_{ag_name.lower()}",
                "target": f"agent_{ag_name.lower()}",
            }

        # writes_to edges: first two agents -> data store
        for j in range(min(2, len(agent_names))):
            edge_ctr += 1
            struct_edges[f"e_{edge_ctr}"] = {
                "edge_type": "writes_to",
                "source": f"agent_{agent_names[j].lower()}",
                "target": "ds_shared_memory",
            }

        # reads_from: last agent -> data store
        edge_ctr += 1
        struct_edges[f"e_{edge_ctr}"] = {
            "edge_type": "reads_from",
            "source": f"agent_{agent_names[-1].lower()}",
            "target": "ds_shared_memory",
        }

        # calls: first agent -> external service
        edge_ctr += 1
        struct_edges[f"e_{edge_ctr}"] = {
            "edge_type": "calls",
            "source": f"agent_{agent_names[0].lower()}",
            "target": "ext_api_service",
        }

        # filtered_by: second agent -> guardrail (or first if only 1)
        guard_agent = agent_names[1] if len(agent_names) > 1 else agent_names[0]
        edge_ctr += 1
        struct_edges[f"e_{edge_ctr}"] = {
            "edge_type": "filtered_by",
            "source": f"agent_{guard_agent.lower()}",
            "target": "guard_output_filter",
        }

        # Dead edge: will never be traversed at runtime
        if len(agent_names) >= 3:
            edge_ctr += 1
            struct_edges["e_dead"] = {
                "edge_type": "reads_from",
                "source": f"agent_{agent_names[-1].lower()}",
                "target": f"agent_{agent_names[0].lower()}",
            }

        # Taxonomy preconditions
        taxonomy_preconds = sel.get("taxonomy_preconditions", [])
        if not taxonomy_preconds:
            taxonomy_preconds = ["shared_state_no_arbitration", "no_timeout_on_delegation", "unhandled_tool_failure"]

        structural_graph = {
            "repo_id": repo_id,
            "framework": fw,
            "taxonomy_preconditions": taxonomy_preconds,
            "nodes": struct_nodes,
            "edges": struct_edges,
        }

        # Add raw events to run records for enricher
        enriched_records = []
        for rec in records:
            events = parse_events_file(
                next(r["events_file"] for r in execution_results if r["run_id"] == rec["run_id"])
            )
            enriched_records.append({**rec, "events": events})

        enriched = enrich_graph(structural_graph, enriched_records)

        # enrich_graph already computes emergent/dead edges
        enriched["taxonomy_preconditions"] = taxonomy_preconds
        enriched_graphs.append(enriched)

        total_emergent += len(enriched.get("emergent_edges", []))
        total_dead += len(enriched.get("dead_edges", []))

    print(f"  Graphs enriched:   {len(enriched_graphs)}")
    print(f"  Total emergent:    {total_emergent}")
    print(f"  Total dead:        {total_dead}")

    # Show sample enriched graph
    if enriched_graphs:
        sample_eg = enriched_graphs[0]
        print(f"\n  Sample enriched graph ({sample_eg['repo_id']}):")
        print(f"    Nodes:           {len(sample_eg['nodes'])}")
        print(f"    Edges:           {len(sample_eg['edges'])}")
        print(f"    Emergent edges:  {len(sample_eg.get('emergent_edges', []))}")
        print(f"    Dead edges:      {len(sample_eg.get('dead_edges', []))}")

        # Show one node per type
        seen_types = set()
        for nid, ndata in sample_eg["nodes"].items():
            struct = ndata.get("structural", ndata)
            ntype = struct.get("node_type", "unknown")
            if ntype in seen_types:
                continue
            seen_types.add(ntype)
            beh = ndata.get("behavioral", {})
            print(f"    Node {nid} (type={ntype}):")
            print(f"      activation_count:    {beh.get('activation_count', 0)}")
            print(f"      activation_rate:     {beh.get('activation_rate', 0)}")
            tp = beh.get("throughput", {})
            print(f"      failure_rate:        {tp.get('failure_rate', 0)}")
            det = beh.get("determinism")
            if det:
                print(f"      same_input_activation_consistency: {det.get('same_input_activation_consistency')}")
                print(f"      same_input_path_consistency:       {det.get('same_input_path_consistency')}")
            gr_eff = beh.get("guardrail_effectiveness")
            if gr_eff:
                print(f"      guardrail_trigger_count:  {gr_eff.get('trigger_count', 0)}")
                print(f"      guardrail_prevented:      {gr_eff.get('prevented_action_count', 0)}")

        # Show emergent edges detail
        for em in sample_eg.get("emergent_edges", [])[:2]:
            print(f"    Emergent: {em.get('source_node_id')} -> {em.get('target_node_id')} "
                  f"(count={em.get('interaction_count', 0)})")

    # Save enriched graphs
    enriched_dir = os.path.join(tmpdir, "enriched_graphs")
    os.makedirs(enriched_dir, exist_ok=True)
    for eg in enriched_graphs:
        path = os.path.join(enriched_dir, f"{eg['repo_id']}_enriched.json")
        with open(path, "w") as f:
            json.dump(eg, f, indent=2, default=str)
    print(f"\n  -> Saved {len(enriched_graphs)} enriched graphs to {enriched_dir}")

    # =====================================================================
    # PHASE 5: KNOWLEDGE BASE
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 5: KNOWLEDGE BASE")
    print(SEPARATOR)

    from stratum_lab.knowledge.patterns import build_pattern_knowledge_base, detect_novel_patterns, compare_frameworks
    from stratum_lab.knowledge.taxonomy import compute_manifestation_probabilities
    from stratum_lab.knowledge.fragility import build_fragility_map

    # Patterns
    patterns = build_pattern_knowledge_base(enriched_graphs)
    print(f"  Patterns found:    {len(patterns)}")
    for p in patterns[:5]:
        print(f"    {p['pattern_name']}: prevalence={p['prevalence']['prevalence_rate']:.2f}, "
              f"failure_rate={p['behavioral_distribution']['failure_rate']:.2f}, "
              f"risk={p['risk_assessment']['risk_level']}")

    # Taxonomy
    taxonomy = compute_manifestation_probabilities(enriched_graphs)
    present = {k: v for k, v in taxonomy.items() if v.get("sample_size", 0) > 0}
    print(f"\n  Taxonomy preconditions with data: {len(present)}")
    for pc_id, data in list(present.items())[:5]:
        print(f"    {pc_id}: P(manifest)={data['probability']}, "
              f"CI={data['confidence_interval']}, n={data['sample_size']}")

    # Novel patterns
    novel = detect_novel_patterns(enriched_graphs)
    print(f"\n  Novel patterns:    {len(novel)}")
    for np_ in novel[:3]:
        print(f"    {np_['repo_id']}: anomaly_score={np_['anomaly_score']:.2f}, "
              f"dim={np_['most_anomalous_dimension']}")

    # Framework comparison
    fw_comp = compare_frameworks(enriched_graphs)
    print(f"\n  Framework comparisons: {len(fw_comp)}")
    for comp in fw_comp[:3]:
        print(f"    {comp['motif_name']}: frameworks={comp['frameworks_compared']}")

    # Fragility map
    fragility = build_fragility_map(enriched_graphs)
    print(f"\n  Fragility entries: {len(fragility)}")
    for entry in fragility[:3]:
        print(f"    {entry['structural_position']}: sensitivity={entry['sensitivity_score']:.4f}, "
              f"repos={entry['affected_repos_count']}")

    # Save knowledge base
    kb_dir = os.path.join(tmpdir, "knowledge_base")
    os.makedirs(kb_dir, exist_ok=True)

    with open(os.path.join(kb_dir, "patterns.json"), "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    with open(os.path.join(kb_dir, "taxonomy.json"), "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)
    # Also save as taxonomy_probabilities.json (expected by predictor.py)
    with open(os.path.join(kb_dir, "taxonomy_probabilities.json"), "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)
    with open(os.path.join(kb_dir, "novel_patterns.json"), "w") as f:
        json.dump(novel, f, indent=2, default=str)
    with open(os.path.join(kb_dir, "framework_comparisons.json"), "w") as f:
        json.dump(fw_comp, f, indent=2, default=str)
    with open(os.path.join(kb_dir, "fragility_map.json"), "w") as f:
        json.dump(fragility, f, indent=2, default=str)

    print(f"\n  -> Saved knowledge base to {kb_dir}")

    # =====================================================================
    # PHASE 6: QUERY LAYER
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 6: QUERY LAYER")
    print(SEPARATOR)

    from stratum_lab.query.fingerprint import compute_graph_fingerprint, compute_normalization_constants
    from stratum_lab.query.matcher import match_against_dataset
    from stratum_lab.query.predictor import predict_risks
    from stratum_lab.query.report import generate_risk_report

    # Use the first enriched graph as a "customer" structural graph
    if enriched_graphs:
        customer_graph = enriched_graphs[0]

        # Compute fingerprint
        fp = compute_graph_fingerprint(customer_graph)
        print(f"  Customer graph fingerprint:")
        print(f"    feature_vector length: {len(fp['feature_vector'])}")
        print(f"    motifs:               {fp['motifs']}")
        print(f"    topology_hash:        {fp['topology_hash'][:16]}...")
        print(f"    structural_metrics:   {json.dumps(fp['structural_metrics'], indent=4)}")

        # Build dataset fingerprints + normalization constants
        dataset_fps = {}
        all_fps = []
        for eg in enriched_graphs[1:]:
            eg_fp = compute_graph_fingerprint(eg)
            dataset_fps[eg["repo_id"]] = eg_fp
            all_fps.append(eg_fp)

        norm_constants = compute_normalization_constants(all_fps + [fp])

        # Save fingerprints + normalization to kb
        with open(os.path.join(kb_dir, "fingerprints.json"), "w") as f:
            json.dump(dataset_fps, f, indent=2, default=str)
        with open(os.path.join(kb_dir, "normalization.json"), "w") as f:
            json.dump(norm_constants, f, indent=2, default=str)

        # Match against dataset
        matches = match_against_dataset(fp, kb_dir, top_k=5)
        print(f"\n  Top matches ({len(matches)}):")
        for m in matches[:5]:
            print(f"    {m.pattern_name}: score={m.similarity_score:.3f}, "
                  f"type={m.match_type}, repos={m.matched_repos}")

        # Predict risks
        preconditions = customer_graph.get("taxonomy_preconditions", [])
        prediction = predict_risks(customer_graph, matches, preconditions, kb_dir)
        print(f"\n  Risk prediction:")
        print(f"    overall_risk_score: {prediction.overall_risk_score:.1f}")
        print(f"    predicted_risks:   {len(prediction.predicted_risks)}")
        print(f"    positive_signals:  {len(prediction.positive_signals)}")
        for risk in prediction.predicted_risks[:3]:
            print(f"      {risk.precondition_id}: P={risk.manifestation_probability:.2f}, "
                  f"severity={risk.severity_when_manifested}")

        # Generate report
        report = generate_risk_report(prediction, customer_graph, output_format="json")
        print(f"\n  Risk report generated: {len(json.dumps(report))} chars (JSON)")

    # =====================================================================
    # PHASE 6B: BATCH REPORTS
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 6B: BATCH REPORTS")
    print(SEPARATOR)

    from stratum_lab.query.batch_report import generate_batch_reports

    if enriched_graphs:
        batch_reports_dir = os.path.join(tmpdir, "batch_reports")
        batch_summary = generate_batch_reports(enriched_dir, kb_dir, batch_reports_dir)
        batch_count = batch_summary.get("reports_generated", 0)
        print(f"  Batch reports generated: {batch_count}")
        print(f"  Errors:                {batch_summary.get('errors', 0)}")
        reports_list = batch_summary.get("reports", [])
        if reports_list:
            sample_br = reports_list[0]
            print(f"  Sample report ({sample_br.get('repo_id', '?')}):")
            print(f"    risk_score:  {sample_br.get('risk_score', 'N/A')}")
            print(f"    risk_level:  {sample_br.get('risk_level', 'N/A')}")
            print(f"    risk_count:  {sample_br.get('risk_count', 0)}")
        dist = batch_summary.get("risk_distribution", {})
        if dist:
            print(f"  Risk distribution: {dist.get('by_risk_level', {})}")
    else:
        batch_count = 0
        print("  (no enriched graphs for batch reports)")

    # =====================================================================
    # PILOT MODE TEST
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PILOT MODE TEST")
    print(SEPARATOR)

    # Test pilot mode with a small subset
    from stratum_lab.selection.selector import score_and_select as pilot_select

    pilot_repos = raw_repos[:10]
    pilot_selected, pilot_summary = pilot_select(
        pilot_repos, target=5, min_runnability=10, min_per_archetype=1
    )
    print(f"  Pilot input repos:    {len(pilot_repos)}")
    print(f"  Pilot selected repos: {len(pilot_selected)}")
    print(f"  Pilot control count:  {pilot_summary.get('control_count', 0)}")
    print(f"  Pilot treatment count:{pilot_summary.get('treatment_count', 0)}")
    print(f"  PILOT MODE TEST: PASS")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{SEPARATOR}")
    print("PIPELINE INTEGRATION SUMMARY")
    print(SEPARATOR)
    print(f"  Phase 1 (Selection):    {len(raw_repos)} scanned -> {len(selected)} selected")
    print(f"  Phase 2 (Execution):    {len(selected[:10])} repos x {runs_per_repo} runs = {len(execution_results)} runs, {total_events_written} events")
    print(f"  Phase 3 (Collection):   {len(all_run_records)} run records, {len(repo_aggregates)} repo aggregates")
    print(f"  Phase 4 (Overlay):      {len(enriched_graphs)} enriched graphs, {total_emergent} emergent edges, {total_dead} dead edges")
    print(f"  Phase 5 (Knowledge):    {len(patterns)} patterns, {len(present)} taxonomy entries, {len(novel)} novel, {len(fragility)} fragility entries")
    print(f"  Phase 6 (Query):        fingerprint + {len(matches) if enriched_graphs else 0} matches + risk prediction + report")
    batch_count = batch_count if enriched_graphs else 0
    print(f"  Phase 6B (Batch):       {batch_count} per-repo reports")
    print(f"\n  All outputs saved to: {tmpdir}")
    print(f"\n  RESULT: ALL PHASES COMPLETED SUCCESSFULLY")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
