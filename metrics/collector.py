"""
Metrics Collector

Collects system, agent, and pipeline metrics for monitoring dashboard.
"""

import json
import platform
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    AccuracyMetrics,
    AgentMetrics,
    LogEntry,
    LogsResponse,
    MetricsSummary,
    PipelineMetrics,
    SystemMetrics,
    TimeSeriesDataPoint,
    ToolUsageStats,
)


class MetricsCollector:
    """
    Collects and aggregates metrics from all system components.

    Tracks:
    - System resources (CPU, memory, disk)
    - Agent sessions and performance
    - Pipeline execution and tool usage
    - ML accuracy and experiments
    """

    def __init__(self, history_size: int = 100):
        self.start_time = time.time()
        self.history_size = history_size

        # Time series histories
        self._cpu_history: deque = deque(maxlen=history_size)
        self._memory_history: deque = deque(maxlen=history_size)
        self._sessions_history: deque = deque(maxlen=history_size)
        self._accuracy_history: deque = deque(maxlen=history_size)

        # Session tracking
        self._sessions: dict[str, dict] = {}
        self._total_sessions = 0
        self._successful_sessions = 0
        self._failed_sessions = 0
        self._total_execution_time = 0.0
        self._total_steps = 0

        # Tool usage tracking
        self._tool_usage: dict[str, dict[str, Any]] = {}

        # Pipeline tracking
        self._pipeline_count = 0
        self._completed_pipelines = 0
        self._failed_pipelines = 0
        self._total_pipeline_time = 0.0

        # Logs
        self._logs: deque = deque(maxlen=1000)

        # MLflow metrics cache
        self._mlflow_cache: dict[str, Any] = {}
        self._mlflow_cache_time: float = 0

    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
        except ImportError:
            # Fallback if psutil not available
            cpu_percent = 0.0
            memory = type("obj", (object,), {"percent": 0.0, "used": 0, "total": 1})()
            disk = type("obj", (object,), {"percent": 0.0, "used": 0, "total": 1})()

        uptime = time.time() - self.start_time

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / (1024**3), 2),
            memory_total_gb=round(memory.total / (1024**3), 2),
            disk_percent=disk.percent,
            disk_used_gb=round(disk.used / (1024**3), 2),
            disk_total_gb=round(disk.total / (1024**3), 2),
            python_version=platform.python_version(),
            platform=platform.system(),
            uptime_seconds=round(uptime, 2),
        )

    def get_agent_metrics(self) -> AgentMetrics:
        """Get agent performance metrics."""
        active_sessions = sum(1 for s in self._sessions.values() if s.get("status") == "running")

        success_rate = 0.0
        if self._total_sessions > 0:
            success_rate = round((self._successful_sessions / self._total_sessions) * 100, 2)

        avg_execution_time = 0.0
        if self._successful_sessions > 0:
            avg_execution_time = round(self._total_execution_time / self._successful_sessions, 2)

        avg_steps = 0.0
        if self._total_sessions > 0:
            avg_steps = round(self._total_steps / self._total_sessions, 2)

        return AgentMetrics(
            total_sessions=self._total_sessions,
            active_sessions=active_sessions,
            successful_sessions=self._successful_sessions,
            failed_sessions=self._failed_sessions,
            success_rate=success_rate,
            avg_execution_time_seconds=avg_execution_time,
            total_steps_executed=self._total_steps,
            avg_steps_per_session=avg_steps,
        )

    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get pipeline and tool usage metrics."""
        # Get available tools count
        tools_available = 28  # Default MCP tools count
        try:
            from action.execute_step import AVAILABLE_TOOLS

            tools_available = len(AVAILABLE_TOOLS)
        except ImportError:
            pass

        # Calculate most used tools
        most_used = []
        sorted_tools = sorted(
            self._tool_usage.items(), key=lambda x: x[1].get("invocations", 0), reverse=True
        )[:10]

        for tool_name, stats in sorted_tools:
            most_used.append(
                ToolUsageStats(
                    tool_name=tool_name,
                    invocations=stats.get("invocations", 0),
                    success_count=stats.get("success", 0),
                    failure_count=stats.get("failure", 0),
                    avg_duration_ms=round(
                        stats.get("total_duration", 0) / max(stats.get("invocations", 1), 1), 2
                    ),
                )
            )

        in_progress = self._pipeline_count - self._completed_pipelines - self._failed_pipelines
        avg_duration = 0.0
        if self._completed_pipelines > 0:
            avg_duration = round(self._total_pipeline_time / self._completed_pipelines, 2)

        return PipelineMetrics(
            total_pipelines_run=self._pipeline_count,
            pipelines_in_progress=max(0, in_progress),
            completed_pipelines=self._completed_pipelines,
            failed_pipelines=self._failed_pipelines,
            tools_available=tools_available,
            tool_invocations=sum(t.get("invocations", 0) for t in self._tool_usage.values()),
            most_used_tools=most_used,
            avg_pipeline_duration_seconds=avg_duration,
        )

    def get_accuracy_metrics(self) -> AccuracyMetrics:
        """Get ML accuracy and experiment metrics."""
        # Try to get from MLflow if available
        experiments = 0
        runs = 0
        best_accuracy = None
        avg_accuracy = None
        models_registered = 0

        try:
            # Check memory for past experiments
            memory_path = Path(__file__).parent.parent / "memory" / "session_logs"
            if memory_path.exists():
                for log_file in memory_path.glob("*.json"):
                    try:
                        with open(log_file) as f:
                            data = json.load(f)
                            exp_state = data.get("experiment_state", {})
                            if exp_state.get("best_accuracy"):
                                runs += 1
                                acc = exp_state["best_accuracy"]
                                if best_accuracy is None or acc > best_accuracy:
                                    best_accuracy = acc
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception:
            pass

        return AccuracyMetrics(
            experiments_tracked=experiments,
            total_training_runs=runs,
            best_accuracy=best_accuracy,
            avg_accuracy=avg_accuracy,
            accuracy_improvement_rate=None,
            models_registered=models_registered,
        )

    def get_metrics_summary(self) -> MetricsSummary:
        """Get complete metrics summary."""
        # Record current values in history
        now = datetime.utcnow().isoformat()
        system = self.get_system_metrics()

        self._cpu_history.append(TimeSeriesDataPoint(timestamp=now, value=system.cpu_percent))
        self._memory_history.append(TimeSeriesDataPoint(timestamp=now, value=system.memory_percent))
        self._sessions_history.append(
            TimeSeriesDataPoint(timestamp=now, value=float(self._total_sessions))
        )

        return MetricsSummary(
            timestamp=now,
            system=system,
            agent=self.get_agent_metrics(),
            pipeline=self.get_pipeline_metrics(),
            accuracy=self.get_accuracy_metrics(),
            cpu_history=list(self._cpu_history),
            memory_history=list(self._memory_history),
            sessions_history=list(self._sessions_history),
            accuracy_history=list(self._accuracy_history),
        )

    # =========================================================================
    # Event Recording
    # =========================================================================

    def record_session_start(self, session_id: str, query: str):
        """Record a new session starting."""
        self._sessions[session_id] = {
            "session_id": session_id,
            "query": query,
            "status": "running",
            "start_time": time.time(),
            "steps": 0,
        }
        self._total_sessions += 1
        self._log("info", "agent", f"Session started: {query[:50]}...", session_id)

    def record_session_complete(self, session_id: str, success: bool, steps: int = 0):
        """Record session completion."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            duration = time.time() - session["start_time"]
            session["status"] = "success" if success else "failed"
            session["duration"] = duration

            if success:
                self._successful_sessions += 1
                self._total_execution_time += duration
            else:
                self._failed_sessions += 1

            self._total_steps += steps
            status = "completed successfully" if success else "failed"
            self._log("info" if success else "error", "agent", f"Session {status}", session_id)

    def record_tool_invocation(self, tool_name: str, success: bool, duration_ms: float):
        """Record a tool being invoked."""
        if tool_name not in self._tool_usage:
            self._tool_usage[tool_name] = {
                "invocations": 0,
                "success": 0,
                "failure": 0,
                "total_duration": 0,
            }

        self._tool_usage[tool_name]["invocations"] += 1
        if success:
            self._tool_usage[tool_name]["success"] += 1
        else:
            self._tool_usage[tool_name]["failure"] += 1
        self._tool_usage[tool_name]["total_duration"] += duration_ms

    def record_pipeline_start(self):
        """Record pipeline starting."""
        self._pipeline_count += 1

    def record_pipeline_complete(self, success: bool, duration_seconds: float):
        """Record pipeline completion."""
        if success:
            self._completed_pipelines += 1
            self._total_pipeline_time += duration_seconds
        else:
            self._failed_pipelines += 1

    def record_accuracy(self, accuracy: float, session_id: str | None = None):
        """Record an accuracy measurement."""
        now = datetime.utcnow().isoformat()
        self._accuracy_history.append(TimeSeriesDataPoint(timestamp=now, value=accuracy))
        self._log("info", "training", f"Accuracy recorded: {accuracy:.4f}", session_id)

    # =========================================================================
    # Logging
    # =========================================================================

    def _log(
        self,
        level: str,
        source: str,
        message: str,
        session_id: str | None = None,
        metadata: dict = None,
    ):
        """Internal logging method."""
        entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            source=source,
            message=message,
            session_id=session_id,
            metadata=metadata or {},
        )
        self._logs.append(entry)

    def log(
        self,
        level: str,
        source: str,
        message: str,
        session_id: str | None = None,
        metadata: dict = None,
    ):
        """Public logging method."""
        self._log(level, source, message, session_id, metadata)

    def get_logs(
        self,
        page: int = 1,
        page_size: int = 50,
        level: str | None = None,
        source: str | None = None,
        session_id: str | None = None,
    ) -> LogsResponse:
        """Get paginated logs with optional filtering."""
        logs = list(self._logs)

        # Apply filters
        if level:
            logs = [l for l in logs if l.level == level]
        if source:
            logs = [l for l in logs if l.source == source]
        if session_id:
            logs = [l for l in logs if l.session_id == session_id]

        # Reverse to show newest first
        logs = logs[::-1]

        # Paginate
        total = len(logs)
        start = (page - 1) * page_size
        end = start + page_size
        page_logs = logs[start:end]

        return LogsResponse(
            logs=page_logs, total=total, page=page, page_size=page_size, has_more=end < total
        )

    # =========================================================================
    # Demo Data (for testing frontend)
    # =========================================================================

    def generate_demo_data(self):
        """Generate demo data for frontend testing."""
        import random

        # Generate history data
        now = datetime.utcnow()
        for i in range(50):
            ts = (now - timedelta(minutes=50 - i)).isoformat()
            self._cpu_history.append(
                TimeSeriesDataPoint(timestamp=ts, value=random.uniform(20, 80))
            )
            self._memory_history.append(
                TimeSeriesDataPoint(timestamp=ts, value=random.uniform(40, 70))
            )
            self._sessions_history.append(TimeSeriesDataPoint(timestamp=ts, value=float(i * 2)))
            if i > 30:
                self._accuracy_history.append(
                    TimeSeriesDataPoint(timestamp=ts, value=random.uniform(0.75, 0.95))
                )

        # Demo sessions
        self._total_sessions = 127
        self._successful_sessions = 118
        self._failed_sessions = 9
        self._total_execution_time = 4500.0
        self._total_steps = 890

        # Demo tool usage
        demo_tools = [
            ("create_hydra_config", 245, 240, 5),
            ("init_mlflow_experiment", 198, 195, 3),
            ("start_mlflow_run", 187, 185, 2),
            ("log_mlflow_metrics", 312, 310, 2),
            ("create_dvc_pipeline", 89, 87, 2),
            ("build_ml_docker_image", 45, 42, 3),
            ("analyze_training_results", 156, 154, 2),
        ]
        for tool, invocations, success, failure in demo_tools:
            self._tool_usage[tool] = {
                "invocations": invocations,
                "success": success,
                "failure": failure,
                "total_duration": invocations * random.uniform(100, 500),
            }

        # Demo pipelines
        self._pipeline_count = 89
        self._completed_pipelines = 82
        self._failed_pipelines = 4
        self._total_pipeline_time = 2800.0

        # Demo logs
        log_messages = [
            ("info", "agent", "Session started: Train ResNet model on CIFAR-10"),
            ("info", "perception", "Analyzing project structure..."),
            ("info", "decision", "Generated execution plan with 8 steps"),
            ("info", "action", "Executing: create_hydra_config"),
            ("info", "action", "Executing: init_mlflow_experiment"),
            ("warning", "training", "GPU memory usage high: 95%"),
            ("info", "action", "Training epoch 1/10 completed"),
            ("info", "training", "Accuracy improved: 0.82 -> 0.87"),
            ("info", "agent", "Session completed successfully"),
            ("error", "agent", "Connection timeout to MLflow server"),
        ]
        for level, source, msg in log_messages:
            self._log(level, source, msg, "demo-session-001")


# Global metrics collector instance
metrics_collector = MetricsCollector()
