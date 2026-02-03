import base64

import mcp_mlops_tools as tools


def test_create_k8s_manifests(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    dep = tools.create_k8s_deployment_yaml(
        str(project), name="demo", image="demo:latest", replicas=2
    )
    svc = tools.create_k8s_service_yaml(str(project), name="demo")
    ing = tools.create_k8s_ingress_yaml(
        str(project), name="demo", host="example.com", service_name="demo"
    )
    hpa = tools.create_k8s_hpa_yaml(str(project), name="demo", deployment_name="demo")
    cfg = tools.create_k8s_configmap_yaml(str(project), name="demo", data={"A": "1"})
    secret = tools.create_k8s_secret_yaml(str(project), name="demo", data={"K": "V"})

    assert dep["success"] is True
    assert svc["success"] is True
    assert ing["success"] is True
    assert hpa["success"] is True
    assert cfg["success"] is True
    assert secret["success"] is True

    secret_path = project / "deployment" / "k8s" / "secret.yaml"
    content = secret_path.read_text()
    assert base64.b64encode(b"V").decode("utf-8") in content


def test_create_helm_chart(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    result = tools.create_helm_chart(
        str(project),
        chart_name="demo",
        image="demo:latest",
        include_ingress=True,
        include_hpa=True,
        include_configmap=True,
        include_secret=True,
    )
    assert result["success"] is True
    chart_dir = project / "deployment" / "helm" / "demo"
    assert (chart_dir / "Chart.yaml").exists()
    assert (chart_dir / "values.yaml").exists()
    assert (chart_dir / "templates" / "deployment.yaml").exists()
    assert (chart_dir / "templates" / "service.yaml").exists()
    assert (chart_dir / "templates" / "ingress.yaml").exists()
    assert (chart_dir / "templates" / "hpa.yaml").exists()
    assert (chart_dir / "templates" / "configmap.yaml").exists()
    assert (chart_dir / "templates" / "secret.yaml").exists()


def test_rollback_deployment_dry_run(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    result = tools.rollback_deployment(str(project), target="k8s", dry_run=True)
    assert result["success"] is True
    assert "kubectl rollout undo" in result["command"]


def test_aws_tools_no_boto3(monkeypatch):
    monkeypatch.setattr(tools, "BOTO3_AVAILABLE", False)
    monkeypatch.setattr(tools, "boto3", None)

    result = tools.list_eks_clusters()
    assert result["success"] is False

    cost = tools.estimate_deployment_cost()
    assert cost["success"] is True
