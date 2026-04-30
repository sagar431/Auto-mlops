from pathlib import Path


def _prompt(name: str) -> str:
    return Path("prompts", name).read_text()


def test_perception_prompt_includes_course_workflow_signals():
    prompt = _prompt("perception_prompt.txt")

    assert "Reproducible training workflow" in prompt
    assert "Capstone production pipeline" in prompt
    assert "KServe/Knative canary stack" in prompt
    assert "health/predict smoke tests" in prompt


def test_decision_prompt_preserves_course_planning_rules():
    prompt = _prompt("decision_prompt.txt")

    assert "Course-Derived Planning Rules" in prompt
    assert "Course Capstone Pipeline" in prompt
    assert "Deployment is not complete until verified" in prompt
    assert "Production plans require rollback" in prompt


def test_deployment_selector_maps_course_targets():
    prompt = _prompt("deployment_selector_prompt.txt")

    assert "Course-Derived Target Mapping" in prompt
    assert "Session 09 style serving" in prompt
    assert "Session 16/17 style Kubernetes serving" in prompt
    assert "health endpoint, prediction smoke test, logs/metrics, and rollback path" in prompt


def test_improvement_and_summary_prompts_cover_hpo_and_verification():
    improvement_prompt = _prompt("improvement_prompt.txt")
    summarizer_prompt = _prompt("summarizer_prompt.txt")

    assert "Learning-rate finder" in improvement_prompt
    assert "HPO path" in improvement_prompt
    assert "Deployment Verification" in summarizer_prompt
    assert "Rollback Plan" in summarizer_prompt
