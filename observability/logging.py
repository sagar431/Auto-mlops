"""
Structured Logging Configuration

Provides JSON-formatted structured logging using structlog.
Supports both development (colored console) and production (JSON) output.
"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field
from structlog.types import EventDict, Processor


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

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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


# Context variable for request-scoped logging context
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


def get_log_context() -> dict[str, Any]:
    """Get the current logging context."""
    return _log_context.get().copy()


def set_log_context(**kwargs) -> None:
    """Set values in the logging context."""
    ctx = _log_context.get().copy()
    ctx.update(kwargs)
    _log_context.set(ctx)


def clear_log_context() -> None:
    """Clear the logging context."""
    _log_context.set({})


def _add_context_processor(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that adds context variables to log entries."""
    ctx = _log_context.get()
    for key, value in ctx.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def _add_timestamp_processor(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that adds ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _rename_event_to_message(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor that renames 'event' to 'message' for consistency."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def _add_logger_name(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Processor that adds logger name to log entries."""
    if logger:
        event_dict["logger_name"] = logger.name if hasattr(logger, "name") else str(logger)
    return event_dict


def _get_json_processors() -> list[Processor]:
    """Get processors for JSON output."""
    return [
        structlog.stdlib.add_log_level,
        _add_timestamp_processor,
        _add_logger_name,
        _add_context_processor,
        _rename_event_to_message,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]


def _get_console_processors() -> list[Processor]:
    """Get processors for console output."""
    return [
        structlog.stdlib.add_log_level,
        _add_timestamp_processor,
        _add_logger_name,
        _add_context_processor,
        _rename_event_to_message,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ]


# Track configuration state
_configured: bool = False


def configure_logging(level: str = "info", json_output: bool = True) -> None:
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
    global _configured

    log_level = getattr(logging, level.upper())

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Select processors based on output format
    processors = _get_json_processors() if json_output else _get_console_processors()

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Update factory settings
    LoggerFactory._default_level = LogLevel(level.lower())
    LoggerFactory._json_output = json_output

    _configured = True


def is_configured() -> bool:
    """Check if logging has been configured."""
    return _configured


class StructuredLogger:
    """
    Structured logger with context binding.

    Wraps structlog to provide a consistent interface with support for:
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
        self._bound_values: dict[str, Any] = {}

        # Ensure structlog is configured
        if not _configured:
            configure_logging(level=level.value, json_output=json_output)

        # Create structlog logger
        self._logger = structlog.get_logger(name)

    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind context values to the logger. Returns self for chaining."""
        self._bound_values.update(kwargs)
        self._logger = self._logger.bind(**kwargs)
        return self

    def unbind(self, *keys: str) -> "StructuredLogger":
        """Remove context keys. Returns self for chaining."""
        for key in keys:
            self._bound_values.pop(key, None)
        self._logger = self._logger.unbind(*keys)
        return self

    def clear_context(self) -> "StructuredLogger":
        """Clear all bound context. Returns self for chaining."""
        keys = list(self._bound_values.keys())
        self._bound_values.clear()
        if keys:
            self._logger = self._logger.unbind(*keys)
        return self

    def new(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional bound values."""
        new_logger = StructuredLogger(
            name=self.name,
            level=self._level,
            json_output=self._json_output,
        )
        new_logger._bound_values = {**self._bound_values, **kwargs}
        new_logger._logger = self._logger.bind(**kwargs)
        return new_logger

    def debug(self, message: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log at INFO level."""
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log at WARNING level."""
        self._logger.warning(message, **kwargs)

    def error(self, message: str, error: str | Exception | None = None, **kwargs) -> None:
        """Log at ERROR level."""
        if error is not None:
            if isinstance(error, Exception):
                kwargs["error"] = str(error)
                kwargs["error_type"] = type(error).__name__
            else:
                kwargs["error"] = error
        self._logger.error(message, **kwargs)

    def critical(self, message: str, error: str | Exception | None = None, **kwargs) -> None:
        """Log at CRITICAL level."""
        if error is not None:
            if isinstance(error, Exception):
                kwargs["error"] = str(error)
                kwargs["error_type"] = type(error).__name__
            else:
                kwargs["error"] = error
        self._logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log at ERROR level with exception info."""
        self._logger.exception(message, **kwargs)

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
    def configure(cls, level: LogLevel = LogLevel.INFO, json_output: bool = True) -> None:
        """Configure default settings for new loggers."""
        cls._default_level = level
        cls._json_output = json_output
        configure_logging(level=level.value, json_output=json_output)

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
    def clear(cls) -> None:
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


# Convenience function to get a structlog logger directly
def get_structlog_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a raw structlog logger for advanced use cases.

    Args:
        name: Logger name

    Returns:
        structlog BoundLogger instance
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
