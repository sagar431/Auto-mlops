"""
Pytest tests for setup_alerting MCP Tool

Tests for the setup_alerting MCP tool that creates alerting configurations
for model monitoring with support for threshold, anomaly, drift, and composite alerts.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from mcp_mlops_tools import setup_alerting

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="mlops_alerting_test_") as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestSetupAlertingBasic:
    """Basic functionality tests for setup_alerting."""

    def test_basic_threshold_alert(self, temp_dir):
        """Test creating a basic threshold alert."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="basic_alert",
            alert_type="threshold",
            metrics=["accuracy"],
        )

        assert result["success"] is True
        assert result["alert_name"] == "basic_alert"
        assert result["alert_type"] == "threshold"
        assert "config_path" in result
        assert Path(result["config_path"]).exists()

    def test_default_values(self, temp_dir):
        """Test that default values are applied correctly."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="default_test",
        )

        assert result["success"] is True
        assert result["alert_type"] == "threshold"
        assert result["evaluation_window"] == "5m"
        assert result["cooldown_period"] == "15m"
        assert result["severity"] == "warning"
        assert result["enabled"] is True
        assert "accuracy" in result["metrics_monitored"]
        assert "latency" in result["metrics_monitored"]
        assert "email" in result["notification_channels"]

    def test_custom_thresholds(self, temp_dir):
        """Test creating alert with custom thresholds."""
        thresholds = {"accuracy": 0.95, "latency": 50}
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="custom_threshold",
            alert_type="threshold",
            metrics=["accuracy", "latency"],
            thresholds=thresholds,
        )

        assert result["success"] is True
        assert len(result["alert_rules"]) == 2

        # Check that thresholds are applied
        for rule in result["alert_rules"]:
            metric = rule["metric"]
            if metric in thresholds:
                assert rule["threshold"] == thresholds[metric]

    def test_nonexistent_project_path(self):
        """Test error handling for non-existent project path."""
        result = setup_alerting(
            project_path="/nonexistent/path/to/project",
            alert_name="test_alert",
        )

        assert result["success"] is False
        assert "error" in result
        assert "does not exist" in result["error"]


# ============================================================================
# Alert Type Tests
# ============================================================================


class TestSetupAlertingTypes:
    """Tests for different alert types."""

    def test_anomaly_alert(self, temp_dir):
        """Test creating an anomaly detection alert."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="anomaly_alert",
            alert_type="anomaly",
            metrics=["accuracy", "f1_score"],
        )

        assert result["success"] is True
        assert result["alert_type"] == "anomaly"

        for rule in result["alert_rules"]:
            assert rule["alert_type"] == "anomaly"
            assert "detection_method" in rule
            assert "sensitivity" in rule

    def test_drift_alert(self, temp_dir):
        """Test creating a drift detection alert."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="drift_alert",
            alert_type="drift",
            metrics=["accuracy"],
            thresholds={"accuracy": 0.15},
        )

        assert result["success"] is True
        assert result["alert_type"] == "drift"

        for rule in result["alert_rules"]:
            assert rule["alert_type"] == "drift"
            assert "drift_threshold" in rule
            assert "statistical_test" in rule

    def test_composite_alert(self, temp_dir):
        """Test creating a composite alert."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="composite_alert",
            alert_type="composite",
            metrics=["accuracy", "latency"],
            thresholds={"accuracy": 0.9, "latency": 100},
        )

        assert result["success"] is True
        assert result["alert_type"] == "composite"

        for rule in result["alert_rules"]:
            assert rule["alert_type"] == "composite"
            assert "sub_rules" in rule
            assert "operator" in rule

    def test_invalid_alert_type(self, temp_dir):
        """Test error handling for invalid alert type."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="invalid_type",
            alert_type="invalid",
        )

        assert result["success"] is False
        assert "error" in result
        assert "Invalid alert_type" in result["error"]


# ============================================================================
# Notification Channel Tests
# ============================================================================


class TestSetupAlertingNotifications:
    """Tests for notification channel configuration."""

    def test_email_notification(self, temp_dir):
        """Test email notification configuration."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="email_alert",
            notification_channels=["email"],
        )

        assert result["success"] is True
        assert "email" in result["notification_setup"]
        email_config = result["notification_setup"]["email"]
        assert email_config["enabled"] is True
        assert "recipients" in email_config

    def test_slack_notification(self, temp_dir):
        """Test Slack notification configuration."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="slack_alert",
            notification_channels=["slack"],
        )

        assert result["success"] is True
        assert "slack" in result["notification_setup"]
        slack_config = result["notification_setup"]["slack"]
        assert "webhook_url" in slack_config
        assert "channel" in slack_config

    def test_pagerduty_notification(self, temp_dir):
        """Test PagerDuty notification configuration."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="pagerduty_alert",
            notification_channels=["pagerduty"],
        )

        assert result["success"] is True
        assert "pagerduty" in result["notification_setup"]
        pd_config = result["notification_setup"]["pagerduty"]
        assert "routing_key" in pd_config
        assert "severity_mapping" in pd_config

    def test_webhook_notification(self, temp_dir):
        """Test webhook notification configuration."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="webhook_alert",
            notification_channels=["webhook"],
        )

        assert result["success"] is True
        assert "webhook" in result["notification_setup"]
        webhook_config = result["notification_setup"]["webhook"]
        assert "url" in webhook_config
        assert "method" in webhook_config

    def test_multiple_notification_channels(self, temp_dir):
        """Test configuring multiple notification channels."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="multi_channel",
            notification_channels=["email", "slack", "pagerduty"],
        )

        assert result["success"] is True
        assert len(result["notification_channels"]) == 3
        assert "email" in result["notification_setup"]
        assert "slack" in result["notification_setup"]
        assert "pagerduty" in result["notification_setup"]

    def test_custom_notification_config(self, temp_dir):
        """Test custom notification configuration."""
        custom_config = {
            "email": {"recipients": ["team@example.com", "alerts@example.com"]},
            "slack": {"webhook_url": "https://hooks.slack.com/custom", "channel": "#custom"},
        }
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="custom_notif",
            notification_channels=["email", "slack"],
            notification_config=custom_config,
        )

        assert result["success"] is True
        assert (
            result["notification_setup"]["email"]["recipients"]
            == custom_config["email"]["recipients"]
        )
        assert (
            result["notification_setup"]["slack"]["webhook_url"]
            == custom_config["slack"]["webhook_url"]
        )
        assert result["notification_setup"]["slack"]["channel"] == custom_config["slack"]["channel"]

    def test_invalid_notification_channel(self, temp_dir):
        """Test error handling for invalid notification channel."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="invalid_channel",
            notification_channels=["invalid_channel"],
        )

        assert result["success"] is False
        assert "error" in result
        assert "Invalid notification channel" in result["error"]


# ============================================================================
# Severity Tests
# ============================================================================


class TestSetupAlertingSeverity:
    """Tests for severity configuration."""

    def test_info_severity(self, temp_dir):
        """Test info severity level."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="info_alert",
            severity="info",
        )

        assert result["success"] is True
        assert result["severity"] == "info"

    def test_warning_severity(self, temp_dir):
        """Test warning severity level."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="warning_alert",
            severity="warning",
        )

        assert result["success"] is True
        assert result["severity"] == "warning"

    def test_critical_severity(self, temp_dir):
        """Test critical severity level."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="critical_alert",
            severity="critical",
        )

        assert result["success"] is True
        assert result["severity"] == "critical"

    def test_invalid_severity(self, temp_dir):
        """Test error handling for invalid severity."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="invalid_severity",
            severity="invalid",
        )

        assert result["success"] is False
        assert "error" in result
        assert "Invalid severity" in result["error"]


# ============================================================================
# File Generation Tests
# ============================================================================


class TestSetupAlertingFiles:
    """Tests for file generation."""

    def test_config_file_created(self, temp_dir):
        """Test that configuration file is created."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="file_test",
        )

        assert result["success"] is True
        config_path = Path(result["config_path"])
        assert config_path.exists()

        # Verify YAML content
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "file_test"
        assert "rules" in config
        assert "notifications" in config

    def test_rule_files_created(self, temp_dir):
        """Test that individual rule files are created."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="rules_test",
            metrics=["accuracy", "precision", "recall"],
        )

        assert result["success"] is True
        assert len(result["rules_written"]) == 3

        for rule_path in result["rules_written"]:
            assert Path(rule_path).exists()

    def test_runner_script_created(self, temp_dir):
        """Test that runner script is created."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="runner_test",
        )

        assert result["success"] is True
        runner_path = Path(result["runner_script"])
        assert runner_path.exists()

        # Verify script content
        content = runner_path.read_text()
        assert "def run_alerting_check" in content
        assert "def check_alert_condition" in content

    def test_directory_structure_created(self, temp_dir):
        """Test that proper directory structure is created."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="structure_test",
        )

        assert result["success"] is True

        # Check directories exist
        monitoring_dir = temp_dir / "monitoring"
        alerting_dir = monitoring_dir / "alerting"
        configs_dir = alerting_dir / "configs"
        rules_dir = alerting_dir / "rules"

        assert monitoring_dir.exists()
        assert alerting_dir.exists()
        assert configs_dir.exists()
        assert rules_dir.exists()


# ============================================================================
# Time Window Tests
# ============================================================================


class TestSetupAlertingTimeWindows:
    """Tests for evaluation window and cooldown period."""

    def test_custom_evaluation_window(self, temp_dir):
        """Test custom evaluation window."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="eval_window_test",
            evaluation_window="15m",
        )

        assert result["success"] is True
        assert result["evaluation_window"] == "15m"

    def test_custom_cooldown_period(self, temp_dir):
        """Test custom cooldown period."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="cooldown_test",
            cooldown_period="1h",
        )

        assert result["success"] is True
        assert result["cooldown_period"] == "1h"


# ============================================================================
# Enabled/Disabled Tests
# ============================================================================


class TestSetupAlertingEnabled:
    """Tests for enabled/disabled state."""

    def test_alert_enabled_by_default(self, temp_dir):
        """Test that alert is enabled by default."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="enabled_test",
        )

        assert result["success"] is True
        assert result["enabled"] is True

    def test_alert_disabled(self, temp_dir):
        """Test creating a disabled alert."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="disabled_test",
            enabled=False,
        )

        assert result["success"] is True
        assert result["enabled"] is False


# ============================================================================
# Result Structure Tests
# ============================================================================


class TestSetupAlertingResultStructure:
    """Tests for the structure of setup_alerting results."""

    def test_result_contains_all_fields(self, temp_dir):
        """Test that result contains all expected fields."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="structure_test",
            metrics=["accuracy", "latency"],
            notification_channels=["email", "slack"],
        )

        assert result["success"] is True
        assert "alert_name" in result
        assert "alert_type" in result
        assert "config_path" in result
        assert "rules_dir" in result
        assert "runner_script" in result
        assert "rules_count" in result
        assert "rules_written" in result
        assert "metrics_monitored" in result
        assert "notification_channels" in result
        assert "alert_rules" in result
        assert "notification_setup" in result
        assert "evaluation_window" in result
        assert "cooldown_period" in result
        assert "severity" in result
        assert "enabled" in result
        assert "message" in result
        assert "next_steps" in result

    def test_alert_rules_structure(self, temp_dir):
        """Test the structure of alert rules."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="rules_structure",
            metrics=["accuracy"],
        )

        assert result["success"] is True
        assert len(result["alert_rules"]) == 1

        rule = result["alert_rules"][0]
        assert "name" in rule
        assert "metric" in rule
        assert "alert_type" in rule
        assert "evaluation_window" in rule
        assert "cooldown_period" in rule
        assert "severity" in rule
        assert "enabled" in rule

    def test_next_steps_provided(self, temp_dir):
        """Test that next steps are provided."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="next_steps_test",
        )

        assert result["success"] is True
        assert len(result["next_steps"]) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestSetupAlertingIntegration:
    """Integration tests for setup_alerting."""

    def test_full_alerting_workflow(self, temp_dir):
        """Test complete alerting setup workflow."""
        result = setup_alerting(
            project_path=str(temp_dir),
            alert_name="production_monitoring",
            alert_type="threshold",
            metrics=["accuracy", "precision", "recall", "latency"],
            thresholds={
                "accuracy": 0.95,
                "precision": 0.90,
                "recall": 0.90,
                "latency": 100,
            },
            notification_channels=["email", "slack", "pagerduty"],
            notification_config={
                "email": {"recipients": ["ml-team@example.com"]},
                "slack": {"channel": "#ml-production-alerts"},
            },
            evaluation_window="10m",
            cooldown_period="30m",
            severity="critical",
            enabled=True,
        )

        assert result["success"] is True
        assert result["alert_name"] == "production_monitoring"
        assert result["rules_count"] == 4
        assert len(result["notification_channels"]) == 3
        assert Path(result["config_path"]).exists()
        assert Path(result["runner_script"]).exists()

        # Verify config file content
        with open(result["config_path"]) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "production_monitoring"
        assert config["severity"] == "critical"
        assert len(config["rules"]) == 4

    def test_multiple_alerts_same_project(self, temp_dir):
        """Test creating multiple alerts in the same project."""
        # Create first alert
        result1 = setup_alerting(
            project_path=str(temp_dir),
            alert_name="accuracy_alert",
            metrics=["accuracy"],
        )

        # Create second alert
        result2 = setup_alerting(
            project_path=str(temp_dir),
            alert_name="latency_alert",
            metrics=["latency"],
        )

        assert result1["success"] is True
        assert result2["success"] is True

        # Both config files should exist
        assert Path(result1["config_path"]).exists()
        assert Path(result2["config_path"]).exists()

        # They should be different files
        assert result1["config_path"] != result2["config_path"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
