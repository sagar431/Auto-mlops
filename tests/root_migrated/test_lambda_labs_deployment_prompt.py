from pathlib import Path


def test_perception_prompt_disambiguates_lambda_labs_from_aws_lambda():
    prompt = Path("prompts/perception_prompt.txt").read_text()

    assert "Lambda Labs" in prompt
    assert "Lambda GPU" in prompt
    assert "do NOT choose fastapi_lambda" in prompt
    assert "AWS Lambda is CPU-only" in prompt


def test_deployment_selector_rejects_fastapi_lambda_for_gpu():
    prompt = Path("prompts/deployment_selector_prompt.txt").read_text()

    assert "gpu_required is false" in prompt
    assert "Never recommend FastAPI+Lambda for GPU-required inference" in prompt


def test_decision_prompt_routes_lambda_labs_gpu_to_litserve():
    prompt = Path("prompts/decision_prompt.txt").read_text()

    assert "Use this path for Lambda Labs / Lambda Cloud GPU instances" in prompt
    assert "Do not choose it for Lambda Labs, Lambda Cloud, or any GPU deployment request" in prompt
