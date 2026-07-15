"""Contract tests for the extracted basic MLflow MCP domain."""

import inspect
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mcp_mlops_tools
from mcp_servers.mlops.server import build_tool_registry

MLFLOW_TOOL_NAMES = (
    "init_mlflow_experiment",
    "start_mlflow_run",
    "log_mlflow_params",
    "log_mlflow_metrics",
    "log_mlflow_artifact",
    "register_mlflow_model",
    "get_best_mlflow_run",
    "end_mlflow_run",
)


class RecordingRun:
    """Small ActiveRun-compatible context manager."""

    def __init__(self, operations: list[tuple[Any, ...]], run_id: str) -> None:
        self._operations = operations
        self.info = SimpleNamespace(
            run_id=run_id,
            run_name="generated-run-name",
            artifact_uri=f"file:///artifacts/{run_id}",
        )

    def __enter__(self):
        self._operations.append(("enter_run", self.info.run_id))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._operations.append(("exit_run", self.info.run_id))
        return False


class RecordingMLflowSDK:
    """In-memory MLflow module substitute recording SDK calls."""

    def __init__(self, *, existing_experiment: bool = False) -> None:
        self.operations: list[tuple[Any, ...]] = []
        self.tracking_uri = "file:///default/mlruns"
        self.experiment = (
            SimpleNamespace(experiment_id="exp-existing") if existing_experiment else None
        )

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.operations.append(("set_tracking_uri", tracking_uri))
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, experiment_name: str):
        self.operations.append(("get_experiment_by_name", experiment_name))
        return self.experiment

    def create_experiment(
        self,
        experiment_name: str,
        artifact_location: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        self.operations.append(("create_experiment", experiment_name, artifact_location, tags))
        return "exp-created"

    def set_experiment(self, experiment_name: str) -> None:
        self.operations.append(("set_experiment", experiment_name))

    def get_tracking_uri(self) -> str:
        self.operations.append(("get_tracking_uri",))
        return self.tracking_uri

    def start_run(self, **kwargs):
        self.operations.append(("start_run", kwargs))
        return RecordingRun(self.operations, kwargs.get("run_id", "run-created"))

    def log_params(self, params: dict[str, Any]) -> None:
        self.operations.append(("log_params", params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.operations.append(("log_metrics", metrics, step))

    def log_artifact(self, artifact_path: str, artifact_dest: str | None = None) -> None:
        self.operations.append(("log_artifact", artifact_path, artifact_dest))

    def log_artifacts(self, artifact_path: str, artifact_dest: str | None = None) -> None:
        self.operations.append(("log_artifacts", artifact_path, artifact_dest))

    def register_model(self, model_uri: str, model_name: str, tags: dict[str, str] | None = None):
        self.operations.append(("register_model", model_uri, model_name, tags))
        return SimpleNamespace(name=model_name, version="7")

    def end_run(self, status: str = "FINISHED") -> None:
        self.operations.append(("end_run", status))


class RecordingMLflowClient:
    """In-memory MlflowClient substitute for best-run searches."""

    def __init__(
        self,
        *,
        experiment_exists: bool = True,
        runs: list[Any] | None = None,
    ) -> None:
        self.operations: list[tuple[Any, ...]] = []
        self.experiment = SimpleNamespace(experiment_id="exp-search") if experiment_exists else None
        self.runs = runs if runs is not None else [self.best_run()]

    @staticmethod
    def best_run():
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id="run-best",
                run_name="best-run",
                artifact_uri="file:///artifacts/run-best",
            ),
            data=SimpleNamespace(
                metrics={"accuracy": 0.91, "loss": 0.12},
                params={"learning_rate": "0.001"},
            ),
        )

    def get_experiment_by_name(self, experiment_name: str):
        self.operations.append(("get_experiment_by_name", experiment_name))
        return self.experiment

    def search_runs(self, **kwargs):
        self.operations.append(("search_runs", kwargs))
        return self.runs


def _dependencies(sdk=None, client=None, *, sdk_loader=None, client_factory=None):
    from mcp_servers.mlops.domains.mlflow_dependencies import MLflowDependencies

    return MLflowDependencies(
        sdk_loader=sdk_loader or (lambda: sdk),
        client_factory=client_factory or (lambda: client),
    )


@contextmanager
def _using(dependencies):
    from mcp_servers.mlops.domains.mlflow import use_dependencies

    with use_dependencies(dependencies):
        yield


def _text_result(contents) -> dict[str, Any]:
    assert len(contents) == 1
    assert contents[0].type == "text"
    return json.loads(contents[0].text)


def test_init_experiment_preserves_creation_existing_and_project_path_behavior():
    sdk = RecordingMLflowSDK()
    with _using(_dependencies(sdk=sdk)):
        created = mcp_mlops_tools.init_mlflow_experiment(
            "demo",
            artifact_location="file:///artifacts",
            tags={"team": "platform"},
            project_path="/project",
        )

    assert created == {
        "success": True,
        "experiment_id": "exp-created",
        "experiment_name": "demo",
        "tracking_uri": "/project/mlruns",
        "verification_results": [
            {
                "check_name": "mlflow_experiment_exists",
                "evidence_type": "declared",
                "source_step": "initialize_mlflow_experiment",
                "passed": True,
                "evidence": "MLflow experiment 'demo' is available.",
            }
        ],
        "message": "Experiment 'demo' initialized (ID: exp-created)",
    }
    assert sdk.operations == [
        ("set_tracking_uri", "/project/mlruns"),
        ("get_experiment_by_name", "demo"),
        (
            "create_experiment",
            "demo",
            "file:///artifacts",
            {"team": "platform"},
        ),
        ("set_experiment", "demo"),
        ("get_tracking_uri",),
    ]

    existing_sdk = RecordingMLflowSDK(existing_experiment=True)
    with _using(_dependencies(sdk=existing_sdk)):
        existing = mcp_mlops_tools.init_mlflow_experiment("existing")

    assert existing["experiment_id"] == "exp-existing"
    assert all(operation[0] != "create_experiment" for operation in existing_sdk.operations)


def test_start_run_preserves_name_tags_and_return_shape():
    sdk = RecordingMLflowSDK()
    with _using(_dependencies(sdk=sdk)):
        result = mcp_mlops_tools.start_mlflow_run("demo", run_name="trial", tags={"stage": "test"})

    assert result == {
        "success": True,
        "run_id": "run-created",
        "run_name": "trial",
        "experiment_name": "demo",
        "artifact_uri": "file:///artifacts/run-created",
        "message": "Started run run-created",
    }
    assert sdk.operations == [
        ("set_experiment", "demo"),
        ("start_run", {"run_name": "trial", "tags": {"stage": "test"}}),
    ]


def test_params_and_metrics_preserve_active_and_explicit_run_behavior():
    sdk = RecordingMLflowSDK()
    with _using(_dependencies(sdk=sdk)):
        active_params = mcp_mlops_tools.log_mlflow_params({"epochs": 3})
        explicit_params = mcp_mlops_tools.log_mlflow_params(
            {"optimizer": "adam"}, run_id="run-explicit"
        )
        active_metrics = mcp_mlops_tools.log_mlflow_metrics({"accuracy": 0.8}, step=4)
        explicit_metrics = mcp_mlops_tools.log_mlflow_metrics(
            {"loss": 0.2}, step=5, run_id="run-explicit"
        )

    assert active_params == {
        "success": True,
        "params_logged": ["epochs"],
        "message": "Logged 1 parameters",
    }
    assert explicit_params["params_logged"] == ["optimizer"]
    assert active_metrics == {
        "success": True,
        "metrics_logged": {"accuracy": 0.8},
        "step": 4,
        "message": "Logged 1 metrics",
    }
    assert explicit_metrics["step"] == 5
    assert sdk.operations == [
        ("log_params", {"epochs": 3}),
        ("start_run", {"run_id": "run-explicit"}),
        ("enter_run", "run-explicit"),
        ("log_params", {"optimizer": "adam"}),
        ("exit_run", "run-explicit"),
        ("log_metrics", {"accuracy": 0.8}, 4),
        ("start_run", {"run_id": "run-explicit"}),
        ("enter_run", "run-explicit"),
        ("log_metrics", {"loss": 0.2}, 5),
        ("exit_run", "run-explicit"),
    ]


def test_artifacts_preserve_missing_file_directory_destination_and_run_behavior(tmp_path):
    artifact_file = tmp_path / "metrics.json"
    artifact_file.write_text("{}")
    artifact_directory = tmp_path / "plots"
    artifact_directory.mkdir()
    sdk = RecordingMLflowSDK()

    with _using(_dependencies(sdk=sdk)):
        missing = mcp_mlops_tools.log_mlflow_artifact(str(tmp_path / "missing"))
        logged_file = mcp_mlops_tools.log_mlflow_artifact(
            str(artifact_file), artifact_dest="reports"
        )
        logged_directory = mcp_mlops_tools.log_mlflow_artifact(
            str(artifact_directory), artifact_dest="figures", run_id="run-explicit"
        )

    assert missing == {
        "success": False,
        "error": f"Artifact path {tmp_path / 'missing'} does not exist",
    }
    assert logged_file == {
        "success": True,
        "artifact_path": str(artifact_file),
        "message": f"Logged artifact from {artifact_file}",
    }
    assert logged_directory["success"] is True
    assert ("log_artifact", str(artifact_file), "reports") in sdk.operations
    assert ("log_artifacts", str(artifact_directory), "figures") in sdk.operations
    assert ("enter_run", "run-explicit") in sdk.operations
    assert ("exit_run", "run-explicit") in sdk.operations


def test_model_registration_preserves_uri_tags_and_result():
    sdk = RecordingMLflowSDK()
    with _using(_dependencies(sdk=sdk)):
        result = mcp_mlops_tools.register_mlflow_model(
            "model",
            "classifier",
            run_id="run-123",
            tags={"stage": "candidate"},
        )

    assert result == {
        "success": True,
        "model_name": "classifier",
        "model_version": "7",
        "model_uri": "runs:/run-123/model",
        "message": "Registered model 'classifier' version 7",
    }
    assert sdk.operations == [
        (
            "register_model",
            "runs:/run-123/model",
            "classifier",
            {"stage": "candidate"},
        )
    ]


def test_best_run_preserves_search_order_selection_and_empty_results():
    client = RecordingMLflowClient()
    with _using(_dependencies(client=client)):
        maximized = mcp_mlops_tools.get_best_mlflow_run(
            "demo", metric_name="accuracy", maximize=True
        )

    assert maximized == {
        "success": True,
        "run_id": "run-best",
        "run_name": "best-run",
        "metrics": {"accuracy": 0.91, "loss": 0.12},
        "params": {"learning_rate": "0.001"},
        "best_metric": {"accuracy": 0.91},
        "artifact_uri": "file:///artifacts/run-best",
    }
    assert client.operations == [
        ("get_experiment_by_name", "demo"),
        (
            "search_runs",
            {
                "experiment_ids": ["exp-search"],
                "order_by": ["metrics.accuracy DESC"],
                "max_results": 1,
            },
        ),
    ]

    minimizing_client = RecordingMLflowClient()
    with _using(_dependencies(client=minimizing_client)):
        mcp_mlops_tools.get_best_mlflow_run("demo", "loss", maximize=False)
    assert minimizing_client.operations[-1][1]["order_by"] == ["metrics.loss ASC"]

    with _using(_dependencies(client=RecordingMLflowClient(experiment_exists=False))):
        missing = mcp_mlops_tools.get_best_mlflow_run("missing")
    assert missing == {"success": False, "error": "Experiment 'missing' not found"}

    with _using(_dependencies(client=RecordingMLflowClient(runs=[]))):
        empty = mcp_mlops_tools.get_best_mlflow_run("demo")
    assert empty == {"success": False, "error": "No runs found in experiment"}


def test_end_run_preserves_status_and_historical_run_id_behavior():
    sdk = RecordingMLflowSDK()
    with _using(_dependencies(sdk=sdk)):
        result = mcp_mlops_tools.end_mlflow_run(run_id="historically-ignored", status="KILLED")

    assert result == {
        "success": True,
        "status": "KILLED",
        "message": "Run ended with status KILLED",
    }
    assert sdk.operations == [("end_run", "KILLED")]


@pytest.mark.parametrize(
    ("name", "arguments", "expected_error"),
    [
        (
            "init_mlflow_experiment",
            {"experiment_name": "demo"},
            "MLflow not installed. Run: pip install mlflow",
        ),
        ("start_mlflow_run", {"experiment_name": "demo"}, "MLflow not installed"),
        ("log_mlflow_params", {"params": {}}, "MLflow not installed"),
        ("log_mlflow_metrics", {"metrics": {}}, "MLflow not installed"),
        ("log_mlflow_artifact", {"artifact_path": "/missing"}, "MLflow not installed"),
        (
            "register_mlflow_model",
            {"model_path": "model", "model_name": "demo"},
            "MLflow not installed",
        ),
        ("get_best_mlflow_run", {"experiment_name": "demo"}, "MLflow not installed"),
        ("end_mlflow_run", {}, "MLflow not installed"),
    ],
)
def test_sdk_unavailable_preserves_each_historical_error(name, arguments, expected_error):
    def unavailable():
        raise ImportError("recorded unavailable SDK")

    dependencies = _dependencies(
        sdk_loader=unavailable,
        client_factory=unavailable,
    )
    with _using(dependencies):
        result = getattr(mcp_mlops_tools, name)(**arguments)

    assert result == {"success": False, "error": expected_error}


def test_dependency_failures_preserve_error_dictionary():
    def failing_loader():
        raise RuntimeError("tracking backend failed")

    with _using(_dependencies(sdk_loader=failing_loader)):
        result = mcp_mlops_tools.start_mlflow_run("demo")

    assert result == {"success": False, "error": "tracking backend failed"}


def test_dependency_overrides_are_immutable_nested_and_scoped():
    outer_sdk = RecordingMLflowSDK()
    inner_sdk = RecordingMLflowSDK()
    outer_dependencies = _dependencies(sdk=outer_sdk)
    inner_dependencies = _dependencies(sdk=inner_sdk)

    with pytest.raises(FrozenInstanceError):
        outer_dependencies.sdk_loader = lambda: inner_sdk

    with _using(outer_dependencies):
        mcp_mlops_tools.end_mlflow_run(status="FINISHED")
        with _using(inner_dependencies):
            mcp_mlops_tools.end_mlflow_run(status="FAILED")
        mcp_mlops_tools.end_mlflow_run(status="KILLED")

    assert outer_sdk.operations == [
        ("end_run", "FINISHED"),
        ("end_run", "KILLED"),
    ]
    assert inner_sdk.operations == [("end_run", "FAILED")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("init_mlflow_experiment", {"experiment_name": "demo"}),
        ("start_mlflow_run", {"experiment_name": "demo", "run_name": "trial"}),
        ("log_mlflow_params", {"params": {"epochs": 3}, "run_id": "run-1"}),
        (
            "log_mlflow_metrics",
            {"metrics": {"accuracy": 0.8}, "step": 2, "run_id": "run-1"},
        ),
        ("log_mlflow_artifact", {"artifact_path": __file__, "run_id": "run-1"}),
        (
            "register_mlflow_model",
            {"model_path": "model", "model_name": "demo", "run_id": "run-1"},
        ),
        ("get_best_mlflow_run", {"experiment_name": "demo"}),
        ("end_mlflow_run", {"run_id": "ignored", "status": "FAILED"}),
    ],
)
async def test_all_direct_and_mcp_calls_are_equivalent(name, arguments):
    direct_sdk = RecordingMLflowSDK()
    direct_client = RecordingMLflowClient()
    mcp_sdk = RecordingMLflowSDK()
    mcp_client = RecordingMLflowClient()

    with _using(_dependencies(direct_sdk, direct_client)):
        direct = getattr(mcp_mlops_tools, name)(**arguments)
    with _using(_dependencies(mcp_sdk, mcp_client)):
        through_mcp = _text_result(await mcp_mlops_tools.call_tool(name, arguments))

    assert through_mcp == direct
    assert mcp_sdk.operations == direct_sdk.operations
    assert mcp_client.operations == direct_client.operations


@pytest.mark.asyncio
async def test_mcp_dispatch_retains_dynamic_root_monkeypatch(monkeypatch):
    monkeypatch.setattr(
        mcp_mlops_tools,
        "start_mlflow_run",
        lambda experiment_name, run_name=None, tags=None: {
            "success": True,
            "source": "root-monkeypatch",
            "experiment_name": experiment_name,
        },
    )

    result = _text_result(
        await mcp_mlops_tools.call_tool("start_mlflow_run", {"experiment_name": "patched"})
    )

    assert result == {
        "success": True,
        "source": "root-monkeypatch",
        "experiment_name": "patched",
    }


def test_registry_construction_performs_no_mlflow_dependency_calls():
    calls: list[str] = []

    def loader():
        calls.append("sdk")
        return RecordingMLflowSDK()

    def client_factory():
        calls.append("client")
        return RecordingMLflowClient()

    with _using(_dependencies(sdk_loader=loader, client_factory=client_factory)):
        registry = build_tool_registry(mcp_mlops_tools)

    assert len(registry.specs) == 98
    assert calls == []


def test_imports_and_registry_construction_do_not_import_mlflow_or_mutate_cwd(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import builtins
import os
from pathlib import Path

real_import = builtins.__import__
def rejecting_import(name, *args, **kwargs):
    if name == 'mlflow' or name.startswith('mlflow.'):
        raise AssertionError(f'unexpected eager import: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = rejecting_import
before = sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob('*'))
import mcp_mlops_tools
from mcp_servers.mlops.domains.mlflow import MLflowDependencies
from mcp_servers.mlops.schemas.mlflow import InitMLflowExperimentInput
mcp_mlops_tools.build_tool_registry(mcp_mlops_tools)
after = sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob('*'))
assert before == after
assert MLflowDependencies is not None
assert InitMLflowExperimentInput is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={"PYTHONPATH": str(repository_root)},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_extracted_root_exports_models_and_historical_signatures():
    from mcp_servers.mlops.domains import mlflow as domain
    from mcp_servers.mlops.schemas import mlflow as schemas

    expected_signatures = {
        "init_mlflow_experiment": "(experiment_name: str, tracking_uri: str | None = None, artifact_location: str | None = None, tags: dict[str, str] | None = None, project_path: str | None = None) -> dict[str, typing.Any]",
        "start_mlflow_run": "(experiment_name: str, run_name: str | None = None, tags: dict[str, str] | None = None) -> dict[str, typing.Any]",
        "log_mlflow_params": "(params: dict[str, typing.Any], run_id: str | None = None) -> dict[str, typing.Any]",
        "log_mlflow_metrics": "(metrics: dict[str, float], step: int | None = None, run_id: str | None = None) -> dict[str, typing.Any]",
        "log_mlflow_artifact": "(artifact_path: str, artifact_dest: str | None = None, run_id: str | None = None) -> dict[str, typing.Any]",
        "register_mlflow_model": "(model_path: str, model_name: str, run_id: str | None = None, tags: dict[str, str] | None = None) -> dict[str, typing.Any]",
        "get_best_mlflow_run": "(experiment_name: str, metric_name: str = 'accuracy', maximize: bool = True) -> dict[str, typing.Any]",
        "end_mlflow_run": "(run_id: str | None = None, status: str = 'FINISHED') -> dict[str, typing.Any]",
    }
    schema_names = (
        "InitMLflowExperimentInput",
        "StartMLflowRunInput",
        "LogMLflowParamsInput",
        "LogMLflowMetricsInput",
        "LogMLflowArtifactInput",
        "RegisterMLflowModelInput",
        "GetBestMLflowRunInput",
        "EndMLflowRunInput",
    )

    for name in MLFLOW_TOOL_NAMES:
        assert getattr(mcp_mlops_tools, name) is getattr(domain, name)
        assert str(inspect.signature(getattr(mcp_mlops_tools, name))) == (expected_signatures[name])
    for name in schema_names:
        assert getattr(mcp_mlops_tools, name) is getattr(schemas, name)
