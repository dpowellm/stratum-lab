# Mass Scan Preflight Checklist

## Infrastructure
- [ ] RunPod vLLM server is running and healthy
- [ ] VLLM_HOST env var is set
- [ ] Docker is running on droplet
- [ ] At least 50GB free disk space
- [ ] repos.txt file is ready

## Deployment
- [ ] deploy.sh ran successfully
- [ ] Docker image rebuilt (includes git + updated patcher)
- [ ] Smoke test passed (events > 0)
- [ ] No double-prefix bug in smoke test
- [ ] I/O capture working in smoke test

## Fixes Applied (Round 1 — Production Readiness)
- [ ] Dockerfile has `git` and `curl` installed (Issue A)
- [ ] `google` parent package mocked before `google.cloud` (Issue B)
- [ ] `tier_detail` field in run_metadata (Issue C)
- [ ] deploy.sh copies Dockerfile (Issue D)
- [ ] No `host.docker.internal` in Dockerfile (Issue E)

## Fixes Applied (Round 2 — Data Integrity)
- [ ] scan_id preserved before RUN_ID overwrite (Fix 1)
- [ ] scan_id present in status.json and run_metadata (Fix 1)
- [ ] Dynamic tier timeouts derived from $TIMEOUT (Fix 2)
- [ ] Host timeout buffer increased to +120s (Fix 2)
- [ ] Orphaned events recovery after Phase 1 (Fix 3)
- [ ] deploy.sh preserves smoke evidence on failure + exit 1 (Fix 4)
- [ ] validate_scan.py created for post-scan validation (Fix 5)

## Validation
- [ ] production_validation.json: all_passed = true
- [ ] compile_check_final.json: all 23 compile tests pass
- [ ] data_integrity_validation.json: all_passed = true
- [ ] All verification grep checks pass

## Mass Scan Command
```bash
export VLLM_HOST=https://aishvbx8prhm6k-8000.proxy.runpod.net
export STRATUM_VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
nohup bash ~/stratum-lab/scripts/orchestrate.sh \
    ~/stratum-lab/repos.txt \
    ~/scan_output \
    --concurrency 3 \
    --timeout 600 \
    --vllm-model mistralai/Mistral-7B-Instruct-v0.3 \
    > ~/scan_stdout.log 2>&1 &
```

## Monitoring
- `tail -f ~/scan_output/scan_log.txt` -- live progress
- `grep -c behavioral=FULL_BEHAVIORAL ~/scan_output/scan_log.txt` -- success count
- `grep -c tier=1.5 ~/scan_output/scan_log.txt` -- Tier 1.5 wins
- `grep -c RECOVERED ~/scan_output/scan_log.txt` -- recovered repos
- `df -h /var/lib/docker` -- disk space
- `docker ps | wc -l` -- active containers

## Post-Scan Validation
```bash
python3 ~/stratum-lab/scripts/validate_scan.py ~/scan_output/results \
    --output ~/scan_output/validation_report.json
```
- [ ] data_quality_score >= 50
- [ ] scan_ids contains exactly 1 entry
- [ ] double_prefix_count == 0
- [ ] corrupt_event_lines == 0

## Emergency Commands
- `kill %1` -- stop the mass scan (graceful shutdown via SIGINT trap)
- `docker ps -q | xargs docker stop` -- force-stop all containers
- `docker system prune -f` -- reclaim disk space
