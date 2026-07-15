# Modular MCP server foundation

## Baseline inventory

Before this refactor, `mcp_mlops_tools.py` was 16,302 lines and exposed 98 MCP
tools. The canonical, machine-readable inventory is
`tests/fixtures/mcp_tool_contract.json`. It records every tool name, unchanged
description, Pydantic input-model name, complete JSON schema, and handler, plus
the root imports used elsewhere and the root attributes patched by tests.

The baseline audit found:

- 98 registered names and 98 unique names;
- no registration without a callable same-named handler;
- no implemented MCP handler omitted from registration;
- two unregistered input models, `TriggerGitHubWorkflowInput` and
  `CheckWorkflowRunInput`, with no matching implementation;
- two existing schema-to-function argument-name differences:
  `CreateHydraConfigInput.ml_model_config` maps to `model_config`, and
  `ValidateSchemaInput.schema_definition` maps to `schema`;
- `action.execute_step.AVAILABLE_TOOLS` contains 84 of the 98 MCP tools. The 14
  catalog-only tools are recorded as an existing direct-invocation limitation;
  changing that allowlist is outside this behavior-preserving refactor.

The root imports currently used by production code are
`update_hydra_config` in `agent/agent_loop.py`. Tests also directly import the
Hydra, MLflow, DVC, Docker, workflow, data-quality, monitoring, alerting, and
LitServe functions listed in the fixture. Tests patch root `run_command`,
`check_tool_installed`, `subprocess.run`, `shutil.which`,
`urllib.request.urlopen`, and several private capability helpers. These are
compatibility contracts even where they are not ideal long-term APIs.

## Registry and dispatch

`mcp_servers.mlops.registry.ToolSpec` pairs one public name and description with
one Pydantic input model and callable handler. `ToolRegistry` rejects duplicate
names and invalid models or handlers at registration time. Its ordered specs
generate the MCP `Tool` catalog and drive call validation and dispatch, so
catalog metadata cannot drift from a separate `if`/`elif` dispatch table.

`ToolRegistry.call_mcp()` validates the supplied mapping with the declared
model, invokes the paired handler, awaits it when necessary, and serializes the
result as the established single `TextContent` JSON response. Unknown names and
validation or handler exceptions retain the established error structures.

The four extracted Hydra specs and eight basic MLflow specs live with their
domains. The remaining 86 specs are declared in `mcp_servers.mlops.server` and
adapt implementations that intentionally remain in the root facade. Both
`list_tools()` and `call_tool()` on the root MCP server delegate to the one built
registry in deterministic historical order.

## Adding a tool

1. Define a Pydantic request model in the appropriate `schemas` module.
2. Implement the domain function with explicit dependencies at its boundary.
3. Add one `ToolSpec` to that domain's `tool_specs()` result.
4. Compose those specs in `build_tool_registry()`; do not add a second catalog
   or dispatch branch.
5. Add contract tests for schema, direct behavior, MCP serialization, errors,
   and any root compatibility export required by existing callers.

During incremental extraction, a root-owned implementation uses
`RootModuleHandler`. It resolves the handler attribute at call time and maps
only explicitly documented schema/function argument aliases. Consequently,
existing monkeypatches on `mcp_mlops_tools` still affect MCP dispatch.

## Hydra extraction and compatibility

`mcp_servers.mlops.schemas.hydra` owns these unchanged models:

- `AnalyzeProjectConfigInput`
- `CreateHydraConfigInput`
- `UpdateHydraConfigInput`
- `ValidateHydraConfigInput`

`mcp_servers.mlops.domains.hydra` owns project analysis, configuration
creation, recursive update, validation, and the immutable `HydraDependencies`.
`HydraDependencies` contains one `HydraFilesystem` implementation. The
production `LocalHydraFilesystem` adapter covers exactly the operations Hydra
uses: existence checks, directory creation, glob discovery, text reads, YAML
reads and writes, and project-relative paths.

Each historical public handler keeps its original signature and delegates to a
private implementation that receives the filesystem explicitly. Tests inject a
recording or failing implementation with the scoped `use_dependencies()`
context manager. The override is stored in a `ContextVar` and reset by token, so
nested or concurrent tests do not mutate the root facade's configured baseline
or leak state into later tests. Constructing the adapter, dependencies,
ToolSpecs, or registry performs no filesystem operation.

The root facade installs a `LocalHydraFilesystem` whose directory-creation and
project-relative callbacks resolve the root helpers dynamically. Patching
`mcp_mlops_tools.ensure_directory` or
`mcp_mlops_tools.relative_to_project` therefore remains observable until those
legacy seams are deliberately retired. `mcp_mlops_tools.py` also re-exports the
extracted models and functions under their historical names.

## Basic MLflow extraction and compatibility

`mcp_servers.mlops.schemas.mlflow` owns the unchanged input models for the eight
basic experiment-tracking tools: experiment initialization, run start,
parameter and metric logging, artifact logging, model registration, best-run
lookup, and run end. The later `track_training_in_mlflow` capstone orchestration
tool remains in the root facade and is deliberately outside this boundary.

`mcp_servers.mlops.domains.mlflow` owns the eight historical handler functions
and their ordered ToolSpecs. The root facade re-exports both handlers and input
models, and the ToolSpecs use dynamic root resolution so monkeypatches of those
root handler names remain observable during incremental migration. Public
function signatures, return dictionaries, error messages, explicit-run context
handling, metric steps, artifact destinations, model URIs, search ordering, and
the historically ignored `end_mlflow_run.run_id` argument remain unchanged.

`mcp_servers.mlops.domains.mlflow_dependencies` defines three narrow protocols:
the MLflow module calls used by seven handlers, the tracking-client calls used
by best-run lookup, and the two local artifact-path checks used before upload.
The frozen `MLflowDependencies` stores lazy SDK and client factories plus the
real local filesystem adapter. Scoped `use_dependencies()` overrides use a
resettable `ContextVar`, allowing recording or failing fakes without importing
MLflow, contacting a backend, or leaking dependency state between tests.

MLflow and `MlflowClient` are imported only inside the production factory that
needs them. Importing the modules or constructing dependencies, ToolSpecs, or
the registry therefore performs no MLflow call, filesystem write, network or
database access, experiment creation, or run creation.

## Dependency boundaries for later domains

Extracted domains must not capture mutable root globals implicitly. Follow the
Hydra pattern: define a narrow protocol from operations the domain actually
uses, provide an immutable real adapter, pass the protocol explicitly to
private implementations, and keep any compatibility override scoped and
automatically reset. Configure dynamic root-facing callbacks only for proven
legacy patch seams:

- subprocess: command runner, executable lookup, timeout, and result adapter;
- filesystem: path existence, directory creation, reads/writes, and templates;
- network: URL opener/client, retry clock, and timeout;
- Docker: availability, command execution, inspection, and cleanup;
- DVC and MLflow: CLI or SDK adapters and environment/config lookup;
- AWS: session/client factories, credential checks, and service calls.

Tests should inject recording and failing adapters at the domain boundary and
retain root callbacks until all established root monkeypatch callers migrate.
Importing modules or constructing dependencies and registries must be free of
filesystem mutation, network access, subprocesses, and privileged operations.

## Deliberate scope and extraction order

All schemas and implementations outside Hydra and the eight basic MLflow tools,
shared command/template helpers, legacy public and private helpers, other
optional SDK imports, the `Server` object, stdio startup, and executable
`main()` intentionally remain in `mcp_mlops_tools.py`. The basic MLflow
extraction reduces the facade from 14,613 to 14,320 lines. Empty
`common.commands` or `common.results` abstractions were not created before a
real domain needs them.

The recommended next order is:

1. basic DVC tools;
2. Docker and GitHub Actions tools;
3. data quality and monitoring;
4. training and capstone data;
5. LitServe and other serving targets;
6. Kubernetes, KServe, Helm, and AWS.

This foundation does not add tools, HPO, learning-rate finding, cloud or
deployment behavior, or alter agent/workflow routing.
