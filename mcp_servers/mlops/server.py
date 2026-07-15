"""Build the MLOps MCP server registry from one declarative source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

from .compatibility import RootModuleHandler
from .domains.hydra import tool_specs as hydra_tool_specs
from .domains.mlflow import tool_specs as mlflow_tool_specs
from .registry import ToolRegistry, ToolSpec


@dataclass(frozen=True)
class RootToolDefinition:
    """Metadata needed to adapt a legacy root-facade tool into a ToolSpec."""

    name: str
    description: str
    input_model_name: str
    handler_name: str
    argument_aliases: dict[str, str] = field(default_factory=dict)


ROOT_TOOL_DEFINITIONS: tuple[RootToolDefinition, ...] = (
    RootToolDefinition(
        name='init_dvc_repo',
        description='Initialize DVC in a repository',
        input_model_name='InitDVCRepoInput',
        handler_name='init_dvc_repo',
    ),
    RootToolDefinition(
        name='configure_dvc_remote',
        description='Configure DVC remote storage (S3, GCS, Azure, etc.)',
        input_model_name='ConfigureDVCRemoteInput',
        handler_name='configure_dvc_remote',
    ),
    RootToolDefinition(
        name='add_data_to_dvc',
        description='Add data file or directory to DVC tracking',
        input_model_name='AddDataToDVCInput',
        handler_name='add_data_to_dvc',
    ),
    RootToolDefinition(
        name='create_dvc_pipeline',
        description='Create DVC pipeline with stages (dvc.yaml)',
        input_model_name='CreateDVCPipelineInput',
        handler_name='create_dvc_pipeline',
    ),
    RootToolDefinition(
        name='dvc_push',
        description='Push data to DVC remote storage',
        input_model_name='DVCPushInput',
        handler_name='dvc_push',
    ),
    RootToolDefinition(
        name='dvc_pull',
        description='Pull data from DVC remote storage',
        input_model_name='DVCPullInput',
        handler_name='dvc_pull',
    ),
    RootToolDefinition(
        name='dvc_reproduce',
        description='Reproduce DVC pipeline (run stages)',
        input_model_name='DVCReproduceInput',
        handler_name='dvc_reproduce',
    ),
    RootToolDefinition(
        name='create_ml_dockerfile',
        description='Create Dockerfile for ML project with GPU support option',
        input_model_name='CreateMLDockerfileInput',
        handler_name='create_ml_dockerfile',
    ),
    RootToolDefinition(
        name='build_ml_docker_image',
        description='Build Docker image for ML project',
        input_model_name='BuildMLDockerImageInput',
        handler_name='build_ml_docker_image',
    ),
    RootToolDefinition(
        name='run_training_container',
        description='Run training in Docker container with GPU and volume support',
        input_model_name='RunTrainingContainerInput',
        handler_name='run_training_container',
    ),
    RootToolDefinition(
        name='push_docker_image',
        description='Push Docker image to registry',
        input_model_name='PushDockerImageInput',
        handler_name='push_docker_image',
    ),
    RootToolDefinition(
        name='create_github_workflow',
        description='Create GitHub Actions workflow for ML pipeline with DVC, MLflow, and accuracy checks',
        input_model_name='CreateGitHubWorkflowInput',
        handler_name='create_github_workflow',
    ),
    RootToolDefinition(
        name='add_workflow_step',
        description='Add step to existing GitHub Actions workflow',
        input_model_name='AddWorkflowStepInput',
        handler_name='add_workflow_step',
    ),
    RootToolDefinition(
        name='detect_training_project',
        description='Detect supported Hydra/PyTorch/TIMM training project shape without running training',
        input_model_name='DetectTrainingProjectInput',
        handler_name='detect_training_project',
    ),
    RootToolDefinition(
        name='detect_capstone_data_layouts',
        description='Detect two user-provided canonical image-folder datasets without mutating files or DVC state',
        input_model_name='DetectCapstoneDataLayoutsInput',
        handler_name='detect_capstone_data_layouts',
    ),
    RootToolDefinition(
        name='generate_capstone_split_manifests',
        description='Generate deterministic capstone split manifests after approval',
        input_model_name='GenerateCapstoneSplitManifestsInput',
        handler_name='generate_capstone_split_manifests',
    ),
    RootToolDefinition(
        name='track_capstone_data_package',
        description='Validate/init local DVC and track generated capstone package paths',
        input_model_name='TrackCapstoneDataPackageInput',
        handler_name='track_capstone_data_package',
    ),
    RootToolDefinition(
        name='configure_validate_capstone_dvc_remote',
        description='Configure or validate local/S3 capstone DVC remote evidence without pushing or pulling data',
        input_model_name='ConfigureValidateCapstoneDVCRemoteInput',
        handler_name='configure_validate_capstone_dvc_remote',
    ),
    RootToolDefinition(
        name='push_capstone_data',
        description='Run approval-gated DVC push and record capstone transfer evidence',
        input_model_name='PushCapstoneDataInput',
        handler_name='push_capstone_data',
    ),
    RootToolDefinition(
        name='pull_capstone_data',
        description='Run approval-gated DVC pull and record capstone transfer evidence',
        input_model_name='PullCapstoneDataInput',
        handler_name='pull_capstone_data',
    ),
    RootToolDefinition(
        name='record_capstone_data_stage_evidence',
        description='Write durable Phase 4 data-stage evidence for capstone handoff',
        input_model_name='RecordCapstoneDataStageEvidenceInput',
        handler_name='record_capstone_data_stage_evidence',
    ),
    RootToolDefinition(
        name='prepare_capstone_container_ci_contract',
        description='Validate Phase 5 container/CI workflow inputs and record blocked deferred evidence without running Docker, registry, CI, or secret behavior',
        input_model_name='PrepareCapstoneContainerCIContractInput',
        handler_name='prepare_capstone_container_ci_contract',
    ),
    RootToolDefinition(
        name='resolve_capstone_container_upstream_evidence',
        description='Resolve Phase 5 upstream data-stage, training, MLflow, and model artifact evidence without running Docker, registry, CI, or secret behavior',
        input_model_name='ResolveCapstoneContainerUpstreamEvidenceInput',
        handler_name='resolve_capstone_container_upstream_evidence',
    ),
    RootToolDefinition(
        name='generate_validate_capstone_runtime_image_spec',
        description='Generate or validate Phase 5 Capstone Runtime Image build-spec evidence without running Docker, registry, CI, or secret behavior',
        input_model_name='GenerateValidateCapstoneRuntimeImageSpecInput',
        handler_name='generate_validate_capstone_runtime_image_spec',
    ),
    RootToolDefinition(
        name='build_smoke_check_capstone_container_image',
        description='Detect Docker availability and run approval-gated Phase 5 image build and bounded smoke evidence without registry, CI, deployment, or secret behavior',
        input_model_name='BuildSmokeCheckCapstoneContainerImageInput',
        handler_name='build_smoke_check_capstone_container_image',
    ),
    RootToolDefinition(
        name='configure_validate_capstone_registry_target',
        description='Configure and validate Phase 5 registry target evidence without login, push, CI generation, deployment, or secret mutation',
        input_model_name='ConfigureValidateCapstoneRegistryTargetInput',
        handler_name='configure_validate_capstone_registry_target',
    ),
    RootToolDefinition(
        name='approval_gated_capstone_registry_login_push',
        description='Run approval-gated Phase 5 registry login and image push while recording secret-safe observed push evidence',
        input_model_name='ApprovalGatedCapstoneRegistryLoginPushInput',
        handler_name='approval_gated_capstone_registry_login_push',
    ),
    RootToolDefinition(
        name='record_capstone_container_ci_evidence_handoff',
        description='Generate or validate Phase 5 Capstone CI evidence and write durable container_ci_evidence.json for orchestrator handoff',
        input_model_name='RecordCapstoneContainerCIEvidenceInput',
        handler_name='record_capstone_container_ci_evidence_handoff',
    ),
    RootToolDefinition(
        name='run_bounded_training',
        description='Run a detected training entrypoint with explicit bounded controls and capture metrics/artifacts',
        input_model_name='RunBoundedTrainingInput',
        handler_name='run_bounded_training',
    ),
    RootToolDefinition(
        name='track_training_in_mlflow',
        description='Track bounded training evidence in a verified local MLflow run',
        input_model_name='TrackTrainingInMLflowInput',
        handler_name='track_training_in_mlflow',
    ),
    RootToolDefinition(
        name='record_capstone_orchestrator_skeleton',
        description='Record the Capstone Orchestrator skeleton with blocked/deferred evidence',
        input_model_name='RecordCapstoneOrchestratorSkeletonInput',
        handler_name='record_capstone_orchestrator_skeleton',
    ),
    RootToolDefinition(
        name='analyze_training_results',
        description='Analyze training results and suggest improvements',
        input_model_name='AnalyzeTrainingResultsInput',
        handler_name='analyze_training_results',
    ),
    RootToolDefinition(
        name='suggest_improvements',
        description='Suggest configuration improvements based on current metrics',
        input_model_name='SuggestImprovementsInput',
        handler_name='suggest_improvements',
    ),
    RootToolDefinition(
        name='check_accuracy_threshold',
        description='Check if accuracy threshold is met in experiment',
        input_model_name='CheckAccuracyThresholdInput',
        handler_name='check_accuracy_threshold',
    ),
    RootToolDefinition(
        name='validate_dataset',
        description='Validate ML dataset for quality issues (missing values, duplicates, class balance, outliers, image validity)',
        input_model_name='ValidateDatasetInput',
        handler_name='validate_dataset',
    ),
    RootToolDefinition(
        name='create_expectation_suite',
        description='Create a Great Expectations expectation suite for data validation with customizable expectations',
        input_model_name='CreateExpectationSuiteInput',
        handler_name='create_expectation_suite',
    ),
    RootToolDefinition(
        name='check_data_quality',
        description='Check data quality using Great Expectations-based validation. Validates datasets against expectations or runs basic quality checks (nulls, duplicates, row count). Returns quality score, validation results, and recommendations.',
        input_model_name='CheckDataQualityInput',
        handler_name='check_data_quality',
    ),
    RootToolDefinition(
        name='profile_dataset',
        description='Profile a dataset to get comprehensive statistics including row/column counts, missing values, duplicates, memory usage, and per-column statistics (mean, median, std, quartiles for numeric; top values for categorical).',
        input_model_name='ProfileDatasetInput',
        handler_name='profile_dataset',
    ),
    RootToolDefinition(
        name='detect_anomalies',
        description='Detect anomalies in a dataset using statistical methods (IQR outliers, Z-score outliers, missing value patterns, duplicate rows). Returns detailed anomaly information with affected rows and severity levels.',
        input_model_name='DetectAnomaliesInput',
        handler_name='detect_anomalies',
    ),
    RootToolDefinition(
        name='validate_schema',
        description='Validate a dataset against a defined schema. Checks for missing columns, extra columns (in strict mode), type mismatches, and constraint violations (nullability, uniqueness, value ranges, patterns).',
        input_model_name='ValidateSchemaInput',
        handler_name='validate_schema',
        argument_aliases={'schema_definition': 'schema'},
    ),
    RootToolDefinition(
        name='compare_distributions',
        description='Compare distributions between a reference dataset and current dataset for drift detection. Uses Kolmogorov-Smirnov test to identify significant distribution shifts in numeric columns.',
        input_model_name='CompareDistributionsInput',
        handler_name='compare_distributions',
    ),
    RootToolDefinition(
        name='detect_data_drift',
        description='Detect data drift between reference and current datasets using Evidently AI. Provides comprehensive drift analysis with per-feature results, severity levels, and recommendations. Supports both numerical and categorical columns with configurable thresholds.',
        input_model_name='DetectDataDriftInput',
        handler_name='detect_data_drift',
    ),
    RootToolDefinition(
        name='monitor_model_performance',
        description='Monitor model performance metrics and detect degradation. Calculates classification metrics (accuracy, precision, recall, F1, AUC-ROC) or regression metrics (MSE, RMSE, MAE, R²), tracks performance over time, compares to baseline, and provides health status with recommendations.',
        input_model_name='MonitorModelPerformanceInput',
        handler_name='monitor_model_performance',
    ),
    RootToolDefinition(
        name='setup_alerting',
        description='Setup alerting configuration for model monitoring. Creates threshold, anomaly, drift, or composite alerts with configurable notification channels (email, Slack, PagerDuty, webhook). Generates alert rules, notification configs, and runner scripts.',
        input_model_name='SetupAlertingInput',
        handler_name='setup_alerting',
    ),
    RootToolDefinition(
        name='select_or_create_model_artifact',
        description='Select or create a local model artifact for LitServe preflight',
        input_model_name='SelectOrCreateModelArtifactInput',
        handler_name='select_or_create_model_artifact',
    ),
    RootToolDefinition(
        name='create_litserve_api',
        description='Create LitServe API for high-throughput model serving with batching and GPU support',
        input_model_name='CreateLitserveAPIInput',
        handler_name='create_litserve_api',
    ),
    RootToolDefinition(
        name='generate_litserve_dockerfile',
        description='Generate a Dockerfile for local LitServe preflight without building an image',
        input_model_name='GenerateLitserveDockerfileInput',
        handler_name='generate_litserve_dockerfile',
    ),
    RootToolDefinition(
        name='record_litserve_launch_command',
        description='Record the local LitServe launch command without starting the server',
        input_model_name='RecordLitserveLaunchCommandInput',
        handler_name='record_litserve_launch_command',
    ),
    RootToolDefinition(
        name='record_litserve_missing_live_evidence',
        description='Record GPU, server, /health, /predict, and endpoint evidence missing from preflight',
        input_model_name='RecordLitserveMissingLiveEvidenceInput',
        handler_name='record_litserve_missing_live_evidence',
    ),
    RootToolDefinition(
        name='detect_runtime_environment',
        description='Record local runtime context for LitServe GPU deployment',
        input_model_name='DetectRuntimeEnvironmentInput',
        handler_name='detect_runtime_environment',
    ),
    RootToolDefinition(
        name='detect_gpu_cuda',
        description='Detect GPU availability from observed nvidia-smi or PyTorch CUDA evidence',
        input_model_name='DetectGpuCudaInput',
        handler_name='detect_gpu_cuda',
    ),
    RootToolDefinition(
        name='select_best_model_artifact',
        description='Select an existing model artifact or LitServe preflight artifact',
        input_model_name='SelectBestModelArtifactInput',
        handler_name='select_best_model_artifact',
    ),
    RootToolDefinition(
        name='record_litserve_image_build_skipped',
        description='Record that Docker image build is optional and skipped by default',
        input_model_name='RecordLitserveImageBuildSkippedInput',
        handler_name='record_litserve_image_build_skipped',
    ),
    RootToolDefinition(
        name='start_litserve_server',
        description='Start LitServe server and record observed process evidence',
        input_model_name='StartLitserveServerInput',
        handler_name='start_litserve_server',
    ),
    RootToolDefinition(
        name='test_litserve_health_endpoint',
        description='Call LitServe /health and record observed HTTP evidence',
        input_model_name='TestLitserveHealthEndpointInput',
        handler_name='test_litserve_health_endpoint',
    ),
    RootToolDefinition(
        name='test_litserve_prediction_endpoint',
        description='Call LitServe /predict and record observed HTTP evidence',
        input_model_name='TestLitservePredictionEndpointInput',
        handler_name='test_litserve_prediction_endpoint',
    ),
    RootToolDefinition(
        name='capture_litserve_logs_and_endpoint',
        description='Record deployed LitServe endpoint URL and server log location',
        input_model_name='CaptureLitserveLogsAndEndpointInput',
        handler_name='capture_litserve_logs_and_endpoint',
    ),
    RootToolDefinition(
        name='record_litserve_gpu_rollback_readiness',
        description='Record LitServe cleanup command and manual Lambda Cloud stop instruction',
        input_model_name='RecordLitserveGpuRollbackReadinessInput',
        handler_name='record_litserve_gpu_rollback_readiness',
    ),
    RootToolDefinition(
        name='configure_litserver',
        description='Configure LitServe server settings (batch size, workers, accelerator)',
        input_model_name='ConfigureLitserverInput',
        handler_name='configure_litserver',
    ),
    RootToolDefinition(
        name='create_gradio_interface',
        description='Create Gradio interface for quick model demos and prototypes',
        input_model_name='CreateGradioInterfaceInput',
        handler_name='create_gradio_interface',
    ),
    RootToolDefinition(
        name='deploy_to_huggingface',
        description='Deploy Gradio app to Hugging Face Spaces',
        input_model_name='DeployToHuggingfaceInput',
        handler_name='deploy_to_huggingface',
    ),
    RootToolDefinition(
        name='create_fastapi_app',
        description='Create FastAPI application for serverless model serving',
        input_model_name='CreateFastAPIAppInput',
        handler_name='create_fastapi_app',
    ),
    RootToolDefinition(
        name='create_lambda_dockerfile',
        description='Create Dockerfile for AWS Lambda deployment with Lambda Web Adapter',
        input_model_name='CreateLambdaDockerfileInput',
        handler_name='create_lambda_dockerfile',
    ),
    RootToolDefinition(
        name='generate_cdk_stack',
        description='Generate AWS CDK stack for Lambda deployment with API Gateway',
        input_model_name='GenerateCDKStackInput',
        handler_name='generate_cdk_stack',
    ),
    RootToolDefinition(
        name='create_torchserve_handler',
        description='Create TorchServe custom handler for enterprise model serving',
        input_model_name='CreateTorchserveHandlerInput',
        handler_name='create_torchserve_handler',
    ),
    RootToolDefinition(
        name='create_mar_archive',
        description='Create TorchServe MAR (Model Archive) build script',
        input_model_name='CreateMARArchiveInput',
        handler_name='create_mar_archive',
    ),
    RootToolDefinition(
        name='generate_torchserve_config',
        description='Generate TorchServe configuration (ports, workers)',
        input_model_name='GenerateTorchserveConfigInput',
        handler_name='generate_torchserve_config',
    ),
    RootToolDefinition(
        name='create_inference_service_yaml',
        description='Create KServe InferenceService YAML for Kubernetes deployment',
        input_model_name='CreateInferenceServiceYAMLInput',
        handler_name='create_inference_service_yaml',
    ),
    RootToolDefinition(
        name='generate_kserve_config',
        description='Generate KServe scaling and resource configuration',
        input_model_name='GenerateKServeConfigInput',
        handler_name='generate_kserve_config',
    ),
    RootToolDefinition(
        name='create_k8s_deployment_yaml',
        description='Create Kubernetes Deployment YAML',
        input_model_name='CreateK8sDeploymentInput',
        handler_name='create_k8s_deployment_yaml',
    ),
    RootToolDefinition(
        name='create_k8s_service_yaml',
        description='Create Kubernetes Service YAML',
        input_model_name='CreateK8sServiceInput',
        handler_name='create_k8s_service_yaml',
    ),
    RootToolDefinition(
        name='create_k8s_ingress_yaml',
        description='Create Kubernetes Ingress YAML (ALB annotations for EKS)',
        input_model_name='CreateK8sIngressInput',
        handler_name='create_k8s_ingress_yaml',
    ),
    RootToolDefinition(
        name='create_k8s_hpa_yaml',
        description='Create Kubernetes HPA YAML',
        input_model_name='CreateK8sHPAInput',
        handler_name='create_k8s_hpa_yaml',
    ),
    RootToolDefinition(
        name='create_k8s_configmap_yaml',
        description='Create Kubernetes ConfigMap YAML',
        input_model_name='CreateK8sConfigMapInput',
        handler_name='create_k8s_configmap_yaml',
    ),
    RootToolDefinition(
        name='create_k8s_secret_yaml',
        description='Create Kubernetes Secret YAML',
        input_model_name='CreateK8sSecretInput',
        handler_name='create_k8s_secret_yaml',
    ),
    RootToolDefinition(
        name='generate_rollback_plan',
        description='Generate rollback plan for a deployment target',
        input_model_name='GenerateRollbackPlanInput',
        handler_name='generate_rollback_plan',
    ),
    RootToolDefinition(
        name='list_eks_clusters',
        description='List EKS clusters in a region',
        input_model_name='ListEKSClustersInput',
        handler_name='list_eks_clusters',
    ),
    RootToolDefinition(
        name='update_kubeconfig',
        description='Update kubeconfig for an EKS cluster using AWS CLI',
        input_model_name='UpdateKubeconfigInput',
        handler_name='update_kubeconfig',
    ),
    RootToolDefinition(
        name='create_ecr_repo',
        description='Create or get an ECR repository',
        input_model_name='CreateECRRepoInput',
        handler_name='create_ecr_repo',
    ),
    RootToolDefinition(
        name='get_ecr_login',
        description='Get ECR docker login command',
        input_model_name='GetECRLoginInput',
        handler_name='get_ecr_login',
    ),
    RootToolDefinition(
        name='generate_iam_policy',
        description='Generate least-privilege IAM policy',
        input_model_name='GenerateIAMPolicyInput',
        handler_name='generate_iam_policy',
    ),
    RootToolDefinition(
        name='estimate_deployment_cost',
        description='Estimate monthly deployment cost for Lambda/EKS',
        input_model_name='EstimateDeploymentCostInput',
        handler_name='estimate_deployment_cost',
    ),
    RootToolDefinition(
        name='create_helm_chart',
        description='Create Helm chart for Kubernetes deployment',
        input_model_name='CreateHelmChartInput',
        handler_name='create_helm_chart',
    ),
    RootToolDefinition(
        name='rollback_k8s_deployment',
        description='Rollback a Kubernetes deployment using kubectl',
        input_model_name='RollbackK8sDeploymentInput',
        handler_name='rollback_k8s_deployment',
    ),
    RootToolDefinition(
        name='rollback_lambda_stack',
        description='Rollback an AWS Lambda CDK stack',
        input_model_name='RollbackLambdaStackInput',
        handler_name='rollback_lambda_stack',
    ),
    RootToolDefinition(
        name='rollback_deployment',
        description='Rollback a deployment based on target type',
        input_model_name='RollbackDeploymentInput',
        handler_name='rollback_deployment',
    ),
)


def build_tool_registry(root_module: ModuleType) -> ToolRegistry:
    """Build the ordered registry used by both catalog and dispatch."""
    specs = list(hydra_tool_specs(root_module))
    specs.extend(mlflow_tool_specs(root_module))
    specs.extend(
        ToolSpec(
            name=definition.name,
            description=definition.description,
            input_model=getattr(root_module, definition.input_model_name),
            handler=RootModuleHandler(
                root_module, definition.handler_name, definition.argument_aliases
            ),
        )
        for definition in ROOT_TOOL_DEFINITIONS
    )
    return ToolRegistry(specs)
