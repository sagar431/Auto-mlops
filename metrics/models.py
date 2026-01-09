"""
Metrics Data Models

Pydantic models for metrics API responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SystemMetrics(BaseModel):
    """System-level resource metrics."""
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    memory_used_gb: float = Field(..., description="Memory used in GB")
    memory_total_gb: float = Field(..., description="Total memory in GB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    disk_used_gb: float = Field(..., description="Disk used in GB")
    disk_total_gb: float = Field(..., description="Total disk in GB")
    python_version: str = Field(..., description="Python version")
    platform: str = Field(..., description="Operating system platform")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    total_sessions: int = Field(..., description="Total sessions run")
    active_sessions: int = Field(..., description="Currently active sessions")
    successful_sessions: int = Field(..., description="Successfully completed sessions")
    failed_sessions: int = Field(..., description="Failed sessions")
    success_rate: float = Field(..., description="Success rate percentage")
    avg_execution_time_seconds: float = Field(..., description="Average execution time")
    total_steps_executed: int = Field(..., description="Total steps executed across all sessions")
    avg_steps_per_session: float = Field(..., description="Average steps per session")


class ToolUsageStats(BaseModel):
    """Statistics for a single tool."""
    tool_name: str
    invocations: int
    success_count: int
    failure_count: int
    avg_duration_ms: float


class PipelineMetrics(BaseModel):
    """Pipeline and tool usage metrics."""
    total_pipelines_run: int = Field(..., description="Total pipelines executed")
    pipelines_in_progress: int = Field(..., description="Currently running pipelines")
    completed_pipelines: int = Field(..., description="Completed pipelines")
    failed_pipelines: int = Field(..., description="Failed pipelines")
    tools_available: int = Field(..., description="Total available MCP tools")
    tool_invocations: int = Field(..., description="Total tool invocations")
    most_used_tools: List[ToolUsageStats] = Field(default_factory=list, description="Most frequently used tools")
    avg_pipeline_duration_seconds: float = Field(..., description="Average pipeline duration")


class AccuracyMetrics(BaseModel):
    """ML model accuracy metrics."""
    experiments_tracked: int = Field(..., description="Total MLflow experiments")
    total_training_runs: int = Field(..., description="Total training runs")
    best_accuracy: Optional[float] = Field(None, description="Best accuracy achieved")
    avg_accuracy: Optional[float] = Field(None, description="Average accuracy across runs")
    accuracy_improvement_rate: Optional[float] = Field(None, description="Rate of accuracy improvement")
    models_registered: int = Field(..., description="Models registered in MLflow")


class TimeSeriesDataPoint(BaseModel):
    """A single data point in a time series."""
    timestamp: str
    value: float


class MetricsSummary(BaseModel):
    """Complete metrics summary for dashboard."""
    timestamp: str = Field(..., description="Timestamp of metrics collection")
    system: SystemMetrics
    agent: AgentMetrics
    pipeline: PipelineMetrics
    accuracy: AccuracyMetrics

    # Time series data for charts
    cpu_history: List[TimeSeriesDataPoint] = Field(default_factory=list)
    memory_history: List[TimeSeriesDataPoint] = Field(default_factory=list)
    sessions_history: List[TimeSeriesDataPoint] = Field(default_factory=list)
    accuracy_history: List[TimeSeriesDataPoint] = Field(default_factory=list)


class LogEntry(BaseModel):
    """A single log entry."""
    id: str
    timestamp: str
    level: str = Field(..., description="Log level: info, warning, error, debug")
    source: str = Field(..., description="Source module/component")
    message: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogsResponse(BaseModel):
    """Response for logs endpoint."""
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


class RealtimeEvent(BaseModel):
    """Event for real-time WebSocket updates."""
    event_type: str = Field(..., description="Type: metric_update, log, alert, session_update")
    timestamp: str
    data: Dict[str, Any]
