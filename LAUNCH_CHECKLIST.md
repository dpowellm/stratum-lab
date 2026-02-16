# Stratum-Lab 1K Scan — Launch Checklist

## Infrastructure Prerequisites

### vLLM Server (RunPod)
- [ ] Pod running with RTX 3090 Ti (or better), 24GB+ VRAM
- [ ] vLLM serving `mistralai/Mistral-7B-Instruct-v0.3`
- [ ] Endpoint URL confirmed (e.g., `https://xxx-8000.proxy.runpod.net`)
- [ ] Test: `curl $VLLM_HOST/v1/models` returns model list
- [ ] Test: completion works:
  ```bash
  curl $VLLM_HOST/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
  ```

### Scan Runner (DigitalOcean Droplet)
- [ ] 8GB RAM, 4 vCPU, 50GB disk minimum
- [ ] Docker installed and running (`docker info`)
- [ ] stratum-lab repo cloned and up to date
- [ ] SSH access configured (for monitoring)
- [ ] `tmux` or `screen` installed (scan runs 3+ days)

---

## Build Steps (run on droplet)

```bash
# 1. Clone repo
git clone <repo-url> stratum-lab && cd stratum-lab

# 2. Build Docker image
docker build -f Dockerfile.stratum-lab -t stratum-lab-base .

# 3. Verify image
docker run --rm stratum-lab-base python -c "import stratum_patcher; print('Patcher OK')"
docker run --rm stratum-lab-base python -c "import crewai; print('CrewAI OK')"
docker run --rm stratum-lab-base python -c "import litellm; print('LiteLLM OK')"
docker run --rm stratum-lab-base python -c "import langgraph; print('LangGraph OK')"

# 4. Set vLLM host
export VLLM_HOST="https://xxx-8000.proxy.runpod.net"
export STRATUM_VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# 5. Run smoke test (requires vLLM)
bash scripts/smoke_test.sh "$VLLM_HOST"
# All 7+ tests should PASS

# 6. Prepare repo list (choose one):

#    Option A: From 30k security scan (best quality)
python scripts/select_repos.py scan_results.jsonl \
    --top 3000 --topology-diversity -o repos_ranked.txt

#    Option B: From URL list
python scripts/select_repos.py repos.txt --plain -o repos_ranked.txt

#    Option C: Discovery mode (GitHub API)
python scripts/select_repos.py --discover \
    --github-token "$GITHUB_TOKEN" --top 3000 -o repos_ranked.txt

# 7. Launch scan (in tmux!)
tmux new -s scan
./scripts/orchestrate.sh repos_ranked.txt scan_output \
    --concurrency 5 --timeout 300 \
    --vllm-host "$VLLM_HOST" --vllm-model "$STRATUM_VLLM_MODEL" \
    2>&1 | tee scan_output/orchestrate.log
```

---

## Monitoring During Scan

```bash
# Live progress
tail -f scan_output/scan_log.txt

# Overall status counts
tail -f scan_output/orchestrate.log | grep "Phase"

# Disk space (check every few hours)
df -h

# Running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"

# Success rate so far
grep "SUCCESS" scan_output/scan_log.txt | wc -l

# vLLM health
curl -s $VLLM_HOST/v1/models | python3 -m json.tool

# Check if orchestrator is still running
ps aux | grep orchestrate
```

---

## Expected Timeline

| Phase | Duration | Compute Cost |
|-------|----------|--------------|
| Pilot (30 repos x 1 run) | ~30 min | < $1 |
| Phase 1: Discovery (3000 repos x 1 run) | ~50 hours | ~$14 |
| Phase 2: Depth (500 repos x 4 more runs) | ~33 hours | ~$9 |
| Phase 3: Collection (events -> v6 records) | ~2 hours | < $1 |
| **Total** | **~85 hours (~3.5 days)** | **~$25** |

**Cost breakdown:**
- RunPod Community Cloud GPU: ~$0.27/hr x 83 hours = $22
- DigitalOcean 8GB Droplet: ~$0.07/hr x 83 hours = $6
- Total: ~$28

**Optimization:** Scale to 10 workers on 8GB droplet, halving wall-clock to ~42 hours.

---

## After Scan Completes

```bash
# 1. Check results
python scripts/aggregate_results.py scan_output/results \
    --behavioral-records-dir scan_output/behavioral_records

# 2. Validate sample records
python -c "
import json, pathlib
records = list(pathlib.Path('scan_output/behavioral_records').glob('*.json'))
print(f'Total records: {len(records)}')
valid = sum(1 for r in records if json.loads(r.read_text()).get('schema_version') == 'v6')
print(f'Valid v6 records: {valid}')
# Check coverage
for r in records[:3]:
    rec = json.loads(r.read_text())
    ev = rec.get('edge_validation', {})
    print(f'  {r.name}: {ev.get(\"structural_edges_total\",0)} edges, '
          f'{len(rec.get(\"emergent_edges\",[]))} emergent, '
          f'{len(rec.get(\"failure_modes\",[]))} failure modes')
"

# 3. Run validate_smoke.py on a sample
python scripts/validate_smoke.py scan_output/results/<hash> --verbose

# 4. Ingest into stratum-graph
# Copy behavioral_records/ to stratum-graph repo and run convergence pipeline
```

---

## Success Criteria

- [ ] Pilot quality gate passes (instrumentation failure <=20%, model failure <=15%)
- [ ] 500+ repos produce events in Phase 1 (from 3000 attempted)
- [ ] 450+ repos produce 5 valid run files in Phase 2
- [ ] 400+ valid v6 behavioral records produced (schema_version="v6")
- [ ] 300+ records have non-trivial edge validation data
- [ ] At least 3 STRAT- findings manifest across the corpus
- [ ] 5+ monitoring baseline metrics computed with confidence=medium
- [ ] 100+ unique topology hashes across records
- [ ] All feedback output includes model_tier="weak" caveat

---

## Troubleshooting

### vLLM pod preempted (RunPod Community Cloud)
The orchestrator auto-pauses when vLLM becomes unreachable (waits 5s between retries, up to 5 minutes). Restart the pod, verify the endpoint is alive, and the orchestrator resumes automatically. All completed repos are skipped via resume logic (checks for existing `run_metadata_N.json`).

### Disk full
Run `docker system prune -af` to reclaim space. The orchestrator does `docker system prune -f` every 100 containers, but aggressive manual cleanup may be needed if event files are large. At ~50KB per event file:
```
500 repos x 5 files x 50KB = 125MB total events
500 repos x 5 metadata files x 1KB = 2.5MB
500 behavioral records x 10KB = 5MB
Total: ~135MB (fits easily on 25GB disk)
```

### Container stuck / zombie
```bash
# List running containers
docker ps

# Kill stuck containers
docker ps -q | xargs -r docker stop

# Resume scan (orchestrator skips completed repos)
./scripts/orchestrate.sh repos_ranked.txt scan_output --skip-pilot ...
```

### High instrumentation failure rate
If pilot shows >20% instrumentation failures:
1. Verify patcher injection: `docker run --rm stratum-lab-base python -c "import stratum_patcher; print('OK')"`
2. Check sitecustomize.py in image: `docker run --rm stratum-lab-base cat /usr/local/lib/python3.11/site-packages/sitecustomize.py`
3. Check .pth file: `docker run --rm stratum-lab-base cat /usr/local/lib/python3.11/site-packages/stratum.pth`

### High model failure rate
If pilot shows >15% model failures:
1. Check vLLM health: `curl $VLLM_HOST/v1/models`
2. Check GPU memory: vLLM may be OOM. Restart with lower `--max-model-len`
3. Check vLLM logs for CUDA errors

### Phase 2 not running
If Phase 2 shows 0 successful repos:
1. Check `scan_output/successful_hashes.txt` — should contain repo hashes
2. Check `scan_output/scan_log.txt` for SUCCESS entries
3. Verify `run_metadata_1.json` files exist in `scan_output/results/*/`

---

## Architecture Notes

### Pipeline Flow
```
orchestrate.sh (Pilot -> Phase 1 -> Phase 2 -> Phase 3)
  |
  +-- Per container: run_repo.sh (clone -> install -> detect entry -> execute)
  |     |
  |     +-- Tier 1: Native execution (repo's own entry point)
  |     +-- Tier 2: synthetic_harness.py (fallback, varied topologies)
  |     +-- Output: events_run_N.jsonl + run_metadata_N.json
  |
  +-- Phase 3: build_behavioral_records.py
        |
        +-- Output: v6 behavioral records (9 top-level keys)
        +-- Feed into: stratum-graph convergence pipeline
```

### Per-Container Environment
```
STRATUM_EVENTS_FILE    = /app/output/events_run_N.jsonl
STRATUM_RUN_NUMBER     = N (1-5)
STRATUM_RUN_ID         = uuid
STRATUM_REPO_ID        = repo_url
STRATUM_VLLM_MODEL     = mistralai/Mistral-7B-Instruct-v0.3
STRATUM_FRAMEWORK      = auto
STRATUM_CAPTURE_PROMPTS = 1
VLLM_HOST              = https://...
RUN_NUMBER             = N
```

### Patcher Event Types Available
| Event Type | Source Patcher | v6 Section |
|---|---|---|
| llm.call_start / llm.call_end | openai_patch.py | node_activation |
| agent.task_start / agent.task_end | crewai_patch.py | node_activation, edge_validation |
| delegation.initiated / delegation.completed | crewai_patch.py | edge_validation, emergent_edges |
| tool.invoked / tool.completed | crewai_patch.py, autogen_patch.py | edge_validation |
| state.access | crewai_patch.py, langgraph_patch.py, autogen_patch.py | edge_validation, emergent_edges |
| routing.decision | crewai_patch.py, langgraph_patch.py, autogen_patch.py | edge_validation |
| error.occurred | all patchers | error_propagation, failure_modes |

**Known gap:** `autogen_patch.py` does not emit `delegation.initiated` for GroupChat speaker selection. AutoGen repos will have empty delegation edges in edge_validation.

---

## Pre-Flight Verification Commands

Run these before launching the scan to confirm everything is consistent:

```bash
# All Python scripts parse
for f in scripts/build_behavioral_records.py scripts/check_pilot_quality.py \
         scripts/synthetic_harness.py scripts/select_repos.py \
         scripts/aggregate_results.py scripts/validate_smoke.py; do
    python -c "import ast; ast.parse(open('$f').read()); print('OK: $f')"
done

# All bash scripts valid
bash -n run_repo.sh && echo "OK: run_repo.sh"
bash -n scripts/orchestrate.sh && echo "OK: orchestrate.sh"
bash -n scripts/smoke_test.sh && echo "OK: smoke_test.sh"

# No set -e in critical scripts
grep -rn "^set -e" run_repo.sh scripts/orchestrate.sh scripts/smoke_test.sh
# Should return nothing

# litellm in Docker image
grep -c "litellm" Dockerfile.stratum-lab  # Should be >= 1

# STRATUM_CAPTURE_PROMPTS in patcher
grep "STRATUM_CAPTURE_PROMPTS" stratum_patcher/openai_patch.py  # Should match

# No wrong field paths
grep -rn 'event\["data"\]' scripts/  # Should return nothing
grep -rn 'event\["source_node_id"\]' scripts/  # Should return nothing
grep -rn 'STRATUM_EVENTS_PATH' .  # Should return nothing

# Eval suites pass
python -m pytest tests/ -q  # 105/106 (1 pre-existing failure)
python eval/test_patchers.py 2>&1 | tail -3  # ALL CHECKS PASSED
python eval/test_end_to_end.py 2>&1 | tail -3  # ALL CHECKS PASSED
python eval/test_event_schema.py 2>&1 | tail -3  # ALL CHECKS PASSED
```
