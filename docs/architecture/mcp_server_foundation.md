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

The four extracted Hydra specs live with the Hydra domain. The remaining 94
specs are declared in `mcp_servers.mlops.server` and adapt implementations that
intentionally remain in the root facade. Both `list_tools()` and `call_tool()`
on the root MCP server delegate to the one built registry in deterministic
historical order.

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
creation, recursive update, validation, and `HydraDependencies`. The latter
injects directory creation and project-relative path handling. The root facade
configures these through dynamic wrappers, so patching the corresponding root
helpers remains observable. `mcp_mlops_tools.py` imports and therefore
re-exports the extracted models and functions under their historical names.

## Dependency boundaries for later domains

Extracted domains must not capture mutable root globals implicitly. Define a
small immutable dependency object whose defaults perform the current behavior,
and configure root-facing wrappers during facade initialization:

- subprocess: command runner, executable lookup, timeout, and result adapter;
- filesystem: path existence, directory creation, reads/writes, and templates;
- network: URL opener/client, retry clock, and timeout;
- Docker: availability, command execution, inspection, and cleanup;
- DVC and MLflow: CLI or SDK adapters and environment/config lookup;
- AWS: session/client factories, credential checks, and service calls.

Tests should inject fakes at the domain boundary and retain root adapters until
all established root monkeypatch callers migrate. No remote or privileged
operation should happen merely by importing or constructing the registry.

## Deliberate scope and extraction order

All non-Hydra schemas and implementations, shared command/template helpers,
legacy public and private helpers, optional SDK imports, the `Server` object,
stdio startup, and executable `main()` intentionally remain in
`mcp_mlops_tools.py`. Empty `common.commands` or `common.results` abstractions
were not created before a real domain needs them.

The recommended next order is:

1. remaining Hydra cleanup;
2. basic MLflow tools;
3. basic DVC tools;
4. Docker and GitHub Actions tools;
5. data quality and monitoring;
6. training and capstone data;
7. LitServe and other serving targets;
8. Kubernetes, KServe, Helm, and AWS.

This foundation does not add tools, HPO, learning-rate finding, cloud or
deployment behavior, or alter agent/workflow routing.
