FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Framework dependencies with pinned versions
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

# Stratum instrumentation
COPY stratum_patcher/ /opt/stratum/stratum_patcher/
COPY stratum_patcher/sitecustomize.py /opt/stratum/sitecustomize.py
ENV PYTHONPATH="/opt/stratum:$PYTHONPATH"

# Environment variables pointing to vLLM
ENV OPENAI_API_KEY="sk-stratum-local"
ENV OPENAI_API_BASE="http://host.docker.internal:8000/v1"
ENV OPENAI_BASE_URL="http://host.docker.internal:8000/v1"
ENV ANTHROPIC_API_KEY="sk-stratum-local"

# Execution timeout
ENV STRATUM_TIMEOUT_SECONDS=600
ENV STRATUM_EVENTS_FILE="/app/stratum_events.jsonl"

WORKDIR /app

COPY stratum_patcher/runner.py /opt/stratum/runner.py
ENTRYPOINT ["python", "/opt/stratum/runner.py"]
