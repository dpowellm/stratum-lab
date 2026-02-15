"""Docker container management for sandboxed repo execution.

Builds the stratum-lab-runner image and runs individual repo executions
inside isolated containers.  Each run produces a JSONL event stream,
stdout/stderr, and an exit code that the orchestrator collects.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.errors import BuildError, ContainerError, ImageNotFound
from rich.console import Console

from stratum_lab.config import (
    CONTAINER_WORKDIR,
    DOCKER_IMAGE_NAME,
    DOCKER_IMAGE_TAG,
    EVENTS_FILE_PATH,
    ExecutionStatus,
    VLLM_API_KEY,
)

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# RunResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single container execution run."""

    run_id: str
    repo_id: str
    status: str = ExecutionStatus.CRASH
    duration_ms: int = 0
    events_file_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    peak_memory_mb: float = 0.0
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Image building
# ---------------------------------------------------------------------------

def build_image(
    dockerfile_path: str | Path,
    tag: str | None = None,
) -> str:
    """Build the stratum-lab-runner Docker image.

    Parameters
    ----------
    dockerfile_path:
        Path to the Dockerfile (or directory containing it).
    tag:
        Image tag.  Defaults to ``stratum-lab-runner:latest``.

    Returns
    -------
    str
        The full image tag that was built.
    """
    dockerfile_path = Path(dockerfile_path)
    if dockerfile_path.is_file():
        context_path = str(dockerfile_path.parent)
        dockerfile_name = dockerfile_path.name
    else:
        context_path = str(dockerfile_path)
        dockerfile_name = "Dockerfile"

    tag = tag or f"{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"
    client = docker.from_env()

    console.print(f"[bold]Building image [cyan]{tag}[/cyan] from {context_path}[/bold]")
    try:
        _image, build_logs = client.images.build(
            path=context_path,
            dockerfile=dockerfile_name,
            tag=tag,
            rm=True,
        )
        for chunk in build_logs:
            if "stream" in chunk:
                line = chunk["stream"].rstrip()
                if line:
                    console.print(f"  [dim]{line}[/dim]")
    except BuildError as exc:
        console.print(f"[red]Image build failed:[/red] {exc}")
        raise

    console.print(f"[green]Image built successfully:[/green] {tag}")
    return tag


# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------

def run_container(
    repo_url: str,
    entry_point: str,
    run_id: str,
    repo_id: str,
    framework: str,
    input_data: str,
    timeout: int = 600,
    vllm_url: str = "http://host.docker.internal:8000/v1",
    environment: dict[str, str] | None = None,
) -> RunResult:
    """Run a single repo execution inside a Docker container.

    Creates a container from the stratum-lab-runner image, injects env vars
    for the run context, waits for it to finish (or time out), and collects
    the event file, stdout, stderr, and exit code.

    Parameters
    ----------
    repo_url:
        Git URL of the repo to execute.
    entry_point:
        Relative path to the Python entry point within the repo.
    run_id:
        Unique identifier for this execution run.
    repo_id:
        Identifier for the repository being executed.
    framework:
        Detected framework name (e.g. ``"crewai"``).
    input_data:
        Synthetic input data serialised as a string.
    timeout:
        Maximum execution time in seconds.
    vllm_url:
        OpenAI-compatible vLLM base URL.
    environment:
        Extra environment variables to inject into the container.

    Returns
    -------
    RunResult
        Collected execution data.
    """
    client = docker.from_env()
    image_tag = f"{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"

    # Ensure image exists locally
    try:
        client.images.get(image_tag)
    except ImageNotFound:
        console.print(
            f"[yellow]Image {image_tag} not found locally. "
            f"Build it first with 'build_image'.[/yellow]"
        )
        return RunResult(
            run_id=run_id,
            repo_id=repo_id,
            status=ExecutionStatus.INSTRUMENTATION_FAILURE,
            error_message=f"Docker image {image_tag} not found",
        )

    # Temporary directory for collecting output artifacts
    output_tmpdir = tempfile.mkdtemp(prefix=f"stratum_{run_id}_")

    # Environment variables for the container
    env = {
        "STRATUM_RUN_ID": run_id,
        "STRATUM_REPO_ID": repo_id,
        "STRATUM_FRAMEWORK": framework,
        "OPENAI_BASE_URL": vllm_url,
        "OPENAI_API_KEY": VLLM_API_KEY,
        "STRATUM_TIMEOUT_SECONDS": str(timeout),
        "STRATUM_INPUT_DATA": input_data,
    }
    if environment:
        env.update(environment)

    container = None
    start_time = time.monotonic()

    try:
        # Create and start the container.
        # The image ENTRYPOINT (runner.py) receives repo_url and entry_point
        # as positional arguments.
        container = client.containers.create(
            image=image_tag,
            command=[repo_url, entry_point],
            environment=env,
            volumes={
                output_tmpdir: {"bind": "/output", "mode": "rw"},
            },
            working_dir=CONTAINER_WORKDIR,
            mem_limit="2g",
            network_mode="bridge",
            detach=True,
        )

        container.start()

        # Wait for completion or timeout
        try:
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", -1)
            timed_out = False
        except Exception:
            # Timeout or connection error — kill the container
            timed_out = True
            exit_code = None
            try:
                container.kill()
            except Exception:
                pass

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Collect stdout and stderr
        stdout = _safe_logs(container, stdout=True, stderr=False)
        stderr = _safe_logs(container, stdout=False, stderr=True)

        # Collect events file from container
        events_path = _copy_events_from_container(container, output_tmpdir, run_id)

        # Try to read peak memory stats
        peak_memory_mb = _get_peak_memory_mb(container)

        # Build result
        if timed_out:
            status = ExecutionStatus.TIMEOUT
            error_message = f"Execution timed out after {timeout}s"
        elif exit_code == 0:
            status = ExecutionStatus.SUCCESS
            error_message = None
        else:
            status = ExecutionStatus.CRASH
            error_message = _extract_error_message(stderr)

        return RunResult(
            run_id=run_id,
            repo_id=repo_id,
            status=status,
            duration_ms=elapsed_ms,
            events_file_path=events_path,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            peak_memory_mb=peak_memory_mb,
            error_message=error_message,
        )

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return RunResult(
            run_id=run_id,
            repo_id=repo_id,
            status=ExecutionStatus.CRASH,
            duration_ms=elapsed_ms,
            error_message=str(exc),
        )

    finally:
        # Always clean up the container
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------

def collect_artifacts(
    container: Any,
    output_dir: str | Path,
    run_id: str,
) -> dict[str, str | None]:
    """Copy event file and logs from a (still-existing) container.

    Returns a dict with keys ``events_file``, ``stdout_file``, ``stderr_file``
    pointing to the collected file paths (or ``None`` if collection failed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, str | None] = {
        "events_file": None,
        "stdout_file": None,
        "stderr_file": None,
    }

    # Events JSONL
    events_path = _copy_events_from_container(container, str(output_dir), run_id)
    collected["events_file"] = events_path

    # Stdout
    try:
        stdout = _safe_logs(container, stdout=True, stderr=False)
        if stdout:
            stdout_path = output_dir / f"{run_id}_stdout.log"
            stdout_path.write_text(stdout, encoding="utf-8")
            collected["stdout_file"] = str(stdout_path)
    except Exception:
        pass

    # Stderr
    try:
        stderr = _safe_logs(container, stdout=False, stderr=True)
        if stderr:
            stderr_path = output_dir / f"{run_id}_stderr.log"
            stderr_path.write_text(stderr, encoding="utf-8")
            collected["stderr_file"] = str(stderr_path)
    except Exception:
        pass

    return collected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_logs(container: Any, *, stdout: bool = True, stderr: bool = False) -> str:
    """Return container logs as a decoded string, or empty string on error."""
    try:
        raw = container.logs(stdout=stdout, stderr=stderr)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception:
        return ""


def _copy_events_from_container(
    container: Any,
    output_dir: str,
    run_id: str,
) -> str | None:
    """Copy the JSONL events file out of the container via ``get_archive``.

    Returns the local file path, or ``None`` if copying failed.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    dest = output_dir_path / f"{run_id}_events.jsonl"

    try:
        import tarfile
        import io

        bits, _stat = container.get_archive(EVENTS_FILE_PATH)
        # bits is a generator of bytes chunks; combine into a tar stream
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        dest.write_bytes(extracted.read())
                        return str(dest)
    except Exception:
        pass

    # Fallback: check if the mounted /output dir has the file
    mounted_events = output_dir_path / "stratum_events.jsonl"
    if mounted_events.exists():
        shutil.copy2(str(mounted_events), str(dest))
        return str(dest)

    return None


def _get_peak_memory_mb(container: Any) -> float:
    """Try to read peak memory usage from container stats."""
    try:
        stats = container.stats(stream=False)
        memory_stats = stats.get("memory_stats", {})
        max_usage = memory_stats.get("max_usage", 0)
        if max_usage > 0:
            return round(max_usage / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def _extract_error_message(stderr: str, max_length: int = 500) -> str:
    """Extract a useful error message from stderr output.

    Looks for common Python exception patterns; falls back to the last
    non-empty lines of stderr.
    """
    if not stderr:
        return "Unknown error (no stderr)"

    lines = stderr.strip().splitlines()

    # Look for traceback — grab from the last "Traceback" line onward
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("Traceback"):
            tail = "\n".join(lines[i:])
            return tail[-max_length:] if len(tail) > max_length else tail

    # Fall back to last few lines
    tail = "\n".join(lines[-5:])
    return tail[-max_length:] if len(tail) > max_length else tail
