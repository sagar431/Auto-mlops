# Baseline Checkpoint - 2026-05-02

This checkpoint records the focused pre-Phase 0 test baseline after stabilizing the current agent and deployment e2e tests.

## Commands

```bash
.venv/bin/python -m pytest tests/unit/test_agent_loop.py -q
```

Result: `63 passed in 2.36s`

```bash
.venv/bin/python -m pytest tests/e2e/test_train_deploy_flow.py -q
```

Result: `11 passed in 3.73s`

```bash
.venv/bin/python -m pytest tests/unit/test_perception.py tests/unit/test_decision.py tests/unit/test_execute_step.py -q
```

Result: `125 passed in 4.08s`

## Notes

- This is not a full-suite baseline.
- `graphify update .` could not be run in this environment because `graphify` is not installed on PATH.
