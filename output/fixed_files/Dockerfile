FROM python:3.11-slim

# System dependencies (git needed for clone, curl for health checks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Framework dependencies with pinned versions
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

# Stratum instrumentation (injected, not part of the repo)
COPY stratum_patcher/ /opt/stratum/stratum_patcher/
COPY stratum_patcher/sitecustomize.py /opt/stratum/sitecustomize.py
ENV PYTHONPATH="/opt/stratum:$PYTHONPATH"

# Environment variables pointing to vLLM
ENV OPENAI_API_KEY="sk-stratum-local"
ENV OPENAI_API_BASE="http://VLLM_NOT_CONFIGURED:8000/v1"
ENV OPENAI_BASE_URL="http://VLLM_NOT_CONFIGURED:8000/v1"
ENV ANTHROPIC_API_KEY="sk-stratum-local"

# Execution timeout
ENV STRATUM_TIMEOUT_SECONDS=600

# Events output
ENV STRATUM_EVENTS_FILE="/app/stratum_events.jsonl"

WORKDIR /app

# Entry point is the stratum runner wrapper
COPY stratum_patcher/runner.py /opt/stratum/runner.py
ENTRYPOINT ["python", "/opt/stratum/runner.py"]
