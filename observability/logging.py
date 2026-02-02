"""
Structured Logging Configuration

Provides JSON-formatted structured logging using structlog.
Supports both development (colored console) and production (JSON) output.
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogContext(BaseModel):
    """Context data attached to log entries."""

    session_id: str | None = None
    step_id: str | None = None
    tool_name: str | None = None
    phase: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StructuredLogEntry(BaseModel):
    """A structured log entry for JSON serialization."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    level: str
    logger_name: str
    message: str
    session_id: str | None = None
    step_id: str | None = None
    tool_name: str | None = None
    phase: str | None = None
    duration_ms: float | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StructuredLogger:
    """
    Structured logger with context binding.

    Provides JSON-formatted logging with support for:
    - Session and step context
    - Tool invocation tracking
    - Duration metrics
    - Error details
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True,
    ):
        self.name = name
        self._level = level
        self._json_output = json_output
        self._context = LogContext()

        # Set up Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value.upper()))

        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()

        # Add console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.value.upper()))

        if json_output:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind context values to the logger. Returns self for chaining."""
        if "session_id" in kwargs:
            self._context.session_id = kwargs.pop("session_id")
        if "step_id" in kwargs:
            self._context.step_id = kwargs.pop("step_id")
        if "tool_name" in kwargs:
            self._context.tool_name = kwargs.pop("tool_name")
        if "phase" in kwargs:
            self._context.phase = kwargs.pop("phase")
        self._context.extra.update(kwargs)
        return self

    def unbind(self, *keys: str) -> "StructuredLogger":
        """Remove context keys. Returns self for chaining."""
        if "session_id" in keys:
            self._context.session_id = None
        if "step_id" in keys:
            self._context.step_id = None
        if "tool_name" in keys:
            self._context.tool_name = None
        if "phase" in keys:
            self._context.phase = None
        for key in keys:
            self._context.extra.pop(key, None)
        return self

    def clear_context(self) -> "StructuredLogger":
        """Clear all context. Returns self for chaining."""
        self._context = LogContext()
        return self

    def _log(
        self,
        level: LogLevel,
        message: str,
        duration_ms: float | None = None,
        error: str | None = None,
        **extra,
    ):
        """Internal logging method."""
        merged_extra = {**self._context.extra, **extra}

        entry = StructuredLogEntry(
            level=level.value,
            logger_name=self.name,
            message=message,
            session_id=self._context.session_id,
            step_id=self._context.step_id,
            tool_name=self._context.tool_name,
            phase=self._context.phase,
            duration_ms=duration_ms,
            error=error,
            extra=merged_extra if merged_extra else {},
        )

        if self._json_output:
            # Output as JSON
            log_dict = entry.model_dump(exclude_none=True)
            if not log_dict.get("extra"):
                log_dict.pop("extra", None)
            import json

            log_line = json.dumps(log_dict)
        else:
            # Human-readable format
            parts = [message]
            if entry.session_id:
                parts.append(f"session={entry.session_id}")
            if entry.step_id:
                parts.append(f"step={entry.step_id}")
            if entry.tool_name:
                parts.append(f"tool={entry.tool_name}")
            if entry.phase:
                parts.append(f"phase={entry.phase}")
            if duration_ms is not None:
                parts.append(f"duration={duration_ms:.2f}ms")
            if error:
                parts.append(f"error={error}")
            if merged_extra:
                for k, v in merged_extra.items():
                    parts.append(f"{k}={v}")
            log_line = " | ".join(parts)

        python_level = getattr(logging, level.value.upper())
        self._logger.log(python_level, log_line)

    def debug(self, message: str, **kwargs):
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: str | None = None, **kwargs):
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, error=error, **kwargs)

    def critical(self, message: str, error: str | None = None, **kwargs):
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, error=error, **kwargs)

    # Aliases
    warn = warning


class LoggerFactory:
    """
    Factory for creating and managing structured loggers.

    Provides a centralized registry of loggers with consistent configuration.
    """

    _loggers: dict[str, StructuredLogger] = {}
    _default_level: LogLevel = LogLevel.INFO
    _json_output: bool = True

    @classmethod
    def configure(cls, level: LogLevel = LogLevel.INFO, json_output: bool = True):
        """Configure default settings for new loggers."""
        cls._default_level = level
        cls._json_output = json_output

    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """Get or create a logger with the given name."""
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(
                name=name,
                level=cls._default_level,
                json_output=cls._json_output,
            )
        return cls._loggers[name]

    @classmethod
    def clear(cls):
        """Clear all registered loggers."""
        cls._loggers.clear()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given name.

    This is the main entry point for getting loggers throughout the application.

    Args:
        name: Logger name, typically the module name (e.g., "agent.agent_loop")

    Returns:
        StructuredLogger instance

    Example:
        from observability import get_logger

        logger = get_logger("agent.agent_loop")
        logger.bind(session_id="abc123")
        logger.info("Session started", query="Train model")
        logger.info("Step completed", duration_ms=150.5)
    """
    return LoggerFactory.get_logger(name)


def configure_logging(level: str = "info", json_output: bool = True):
    """Configure the logging system.

    Args:
        level: Log level (debug, info, warning, error, critical)
        json_output: Whether to output JSON (True) or human-readable format (False)

    Example:
        from observability import configure_logging

        # Development
        configure_logging(level="debug", json_output=False)

        # Production
        configure_logging(level="info", json_output=True)
    """
    log_level = LogLevel(level.lower())
    LoggerFactory.configure(level=log_level, json_output=json_output)
