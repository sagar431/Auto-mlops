"""Verify the local golden image-classification slice, optionally through Docker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXAMPLE_DIR / "project"
REPOSITORY_ROOT = EXAMPLE_DIR.parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "golden"
IMAGE_TAG = "auto-mlops-golden-image:local"
FOCUSED_TESTS = (
    "examples/image_classification/tests/test_golden_dvc_lineage.py",
    "examples/image_classification/tests/test_golden_training.py",
    "examples/image_classification/tests/test_golden_mlflow.py",
    "examples/image_classification/tests/test_inference.py",
    "examples/image_classification/tests/test_serve.py",
)


def run_command(
    command: list[str], timeout: int, capture: bool = False, cwd: Path = REPOSITORY_ROOT
) -> str:
    """Run one bounded command and fail immediately on non-zero exit."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        timeout=timeout,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def train_artifact() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "examples.image_classification.project.golden_train",
        "--output-dir",
        str(ARTIFACT_DIR),
    ]
    output = run_command(command, timeout=30, capture=True)
    return json.loads(output.splitlines()[-1])


def run_focused_tests() -> None:
    run_command(["uv", "run", "pytest", *FOCUSED_TESTS, "-q"], timeout=60)


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def docker_prefix() -> list[str]:
    """Use direct Docker access, or bounded non-interactive sudo when available."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker CLI is unavailable")
    direct = subprocess.run(
        ["docker", "info"],
        cwd=REPOSITORY_ROOT,
        check=False,
        timeout=10,
        text=True,
        capture_output=True,
    )
    if direct.returncode == 0:
        return ["docker"]
    sudo = subprocess.run(
        ["sudo", "-n", "docker", "info"],
        cwd=REPOSITORY_ROOT,
        check=False,
        timeout=10,
        text=True,
        capture_output=True,
    )
    if sudo.returncode == 0:
        return ["sudo", "-n", "docker"]
    blocker = direct.stderr.strip() or direct.stdout.strip() or "Docker daemon is unavailable"
    raise RuntimeError(blocker)


def verify_docker(checkpoint_path: Path) -> dict[str, Any]:
    docker = docker_prefix()
    container_name = f"auto-mlops-golden-{os.getpid()}"
    port = available_port()
    container_started = False
    cleanup_status = "not_started"
    logs = ""
    try:
        run_command(
            [
                *docker,
                "build",
                "--file",
                "Dockerfile.golden",
                "--tag",
                IMAGE_TAG,
                ".",
            ],
            timeout=300,
            capture=False,
            cwd=PROJECT_DIR,
        )
        run_command(
            [
                *docker,
                "run",
                "--detach",
                "--name",
                container_name,
                "--publish",
                f"127.0.0.1:{port}:8000",
                "--volume",
                f"{checkpoint_path.parent}:/models:ro",
                IMAGE_TAG,
            ],
            timeout=30,
            capture=True,
        )
        container_started = True
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        health_payload = None
        with httpx.Client(timeout=3.0) as client:
            while time.monotonic() < deadline:
                try:
                    response = client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        health_payload = response.json()
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            if health_payload is None:
                raise RuntimeError("Container did not become healthy within 30 seconds")

            sample_path = ARTIFACT_DIR / "sample-red.png"
            Image.new("RGB", (16, 16), color="red").save(sample_path)
            with sample_path.open("rb") as sample_file:
                prediction_response = client.post(
                    f"{base_url}/predict",
                    files={"file": (sample_path.name, sample_file, "image/png")},
                )
            prediction_response.raise_for_status()
            prediction_payload = prediction_response.json()

        probabilities = prediction_payload.get("probabilities", {})
        if prediction_payload.get("predicted_class") != "red":
            raise RuntimeError("Docker prediction did not classify the deterministic red fixture")
        if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-5:
            raise RuntimeError("Docker prediction probabilities do not sum to one")
        logs = run_command([*docker, "logs", container_name], timeout=10, capture=True)
        return {
            "image_tag": IMAGE_TAG,
            "build": "succeeded",
            "container_name": container_name,
            "container_start": "succeeded",
            "health": health_payload,
            "prediction": prediction_payload,
            "logs": logs.splitlines()[-20:],
        }
    finally:
        if container_started:
            subprocess.run(
                [*docker, "rm", "--force", container_name],
                cwd=REPOSITORY_ROOT,
                check=False,
                timeout=20,
                text=True,
                capture_output=True,
            )
            cleanup_status = "removed"
        remaining = subprocess.run(
            [*docker, "container", "inspect", container_name],
            cwd=REPOSITORY_ROOT,
            check=False,
            timeout=10,
            text=True,
            capture_output=True,
        )
        if remaining.returncode == 0:
            raise RuntimeError(f"Verifier failed to remove container {container_name}")
        if cleanup_status == "removed":
            print(json.dumps({"container_cleanup": "stopped_and_removed"}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Explicitly opt in to bounded Docker build and runtime verification",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        training = train_artifact()
        run_focused_tests()
        report: dict[str, Any] = {
            "status": "succeeded",
            "training": training,
            "focused_tests": "passed",
            "docker": "not_requested",
        }
        if args.docker:
            report["docker"] = verify_docker(Path(training["checkpoint_path"]))
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
