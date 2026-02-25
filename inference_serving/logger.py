from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional, Union, Any
from .utils import *

# Project-wide root logger name
PROJECT_ROOT_LOGGER_NAME = "llmservingsim"


class LLMServingSimFormatter(logging.Formatter):
    """Formatter for LLMServingSim with component and optional node/instance.

    Example:
        [2025-12-01 14:47:13.208] [Scheduler] [node=0,inst=1] [INFO] iteration 0 finished
    """

    # ANSI colors for log levels
    COLORS = {
        "DEBUG": ANSI_CYAN,     # cyan
        "INFO": ANSI_BLUE,      # blue
        "WARNING": ANSI_YEL,    # yellow
        "ERROR": ANSI_RED,      # red
        "CRITICAL": ANSI_RED_BACK, # red background
    }
    RESET = ANSI_RESET

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Format time as 'YYYY-MM-DD HH:MM:SS.mmm' (milliseconds)."""
        dt = datetime.fromtimestamp(record.created)
        return f"{dt:%Y-%m-%d %H:%M:%S}.{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        """Build final log line with timestamp, component, node/inst, level, and message."""
        # Summary mode: no prefixes, just the message
        if getattr(record, "is_summary", False):
            record.message = record.getMessage()
            line = record.message

            if record.exc_info and not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                if not line.endswith("\n"):
                    line += "\n"
                line += record.exc_text
            return line

        # Normal log line
        record.message = record.getMessage()

        # Timestamp
        asctime = self.formatTime(record)

        # Component: class name like "Scheduler", "MemoryModel"
        component = getattr(record, "component", record.name)

        # Level in upper-case (INFO, WARNING, ERROR, CRITICAL)
        level = record.levelname.upper()
        color = self.COLORS.get(level)
        if color is not None:
            level_str = f"{color}{level}{self.RESET}"
        else:
            level_str = level

        # Optional node / instance info
        node_id = getattr(record, "node_id", None)
        instance_id = getattr(record, "instance_id", None)

        if node_id is not None and instance_id is not None:
            node_inst_tag = f"[node={node_id},inst={instance_id}] "
        elif node_id is not None:
            node_inst_tag = f"[node={node_id}] "
        elif instance_id is not None:
            node_inst_tag = f"[inst={instance_id}] "
        else:
            node_inst_tag = ""

        # Base line
        line = f"[{asctime}] [{component}] {node_inst_tag}[{level_str}] {record.message}"

        # Exception / stack information if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if not line.endswith("\n"):
                line += "\n"
            line += record.exc_text

        return line


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects component and optional node/instance into records."""

    def __init__(
        self,
        logger: logging.Logger,
        component: str,
        node_id: Optional[int] = None,
        instance_id: Optional[int] = None,
    ) -> None:
        # We do not use the base 'extra' directly here; we override process().
        super().__init__(logger, extra={})
        self.component = component
        self.node_id = node_id
        self.instance_id = instance_id

    def process(self, msg: Any, kwargs: dict) -> tuple[Any, dict]:
        """Inject component, node_id, and instance_id into log record."""
        extra = kwargs.get("extra", {})
        extra.setdefault("component", self.component)
        extra.setdefault("node_id", self.node_id)
        extra.setdefault("instance_id", self.instance_id)
        kwargs["extra"] = extra
        return msg, kwargs


_configured = False


def configure_logger(
    level: Union[str, int] = "INFO",
    *,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger for LLMServingSim.

    This should typically be called once per process (e.g., at process startup).

    Args:
        level: Logging level as string ("DEBUG", "INFO", ...) or int
               (logging.DEBUG, logging.INFO, ...).
        log_file: Optional path to a log file. If provided, logs go to both
                  console (stderr) and the file with the same format.
    """
    global _configured
    root = logging.getLogger(PROJECT_ROOT_LOGGER_NAME)

    # If already configured once, just update levels and return.
    if _configured:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)
        return

    # Clear any existing handlers for this project root logger.
    root.handlers.clear()

    # Normalize level string to int.
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root.setLevel(level)

    # Console handler (stderr) with project formatter.
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(LLMServingSimFormatter())
    root.addHandler(console_handler)

    # Optional file handler (same formatter, colors will still be present;
    # if you want no colors in file, you can make a second formatter class).
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(LLMServingSimFormatter())
        root.addHandler(file_handler)

    # Do not propagate to the global root logger to avoid duplicate logs.
    root.propagate = False
    _configured = True


def get_logger(
    component: Optional[Union[str, type]] = None,
    *,
    node_id: Optional[int] = None,
    instance_id: Optional[int] = None,
) -> ComponentLoggerAdapter:
    """Get a logger adapter with component (class name) and optional node/instance.

    Examples:
        >>> from llmservingsim.logger import configure_logger, get_logger
        >>> configure_logger("INFO")
        >>> logger = get_logger("Scheduler", node_id=0, instance_id=1)
        >>> logger.info("iteration 0 finished")

        >>> class MemoryModel:
        ...     def __init__(self):
        ...         self.logger = get_logger(self.__class__)
        ...     def init(self):
        ...         self.logger.info("Initialized memory configuration")

    Args:
        component:
            - If str: used directly (e.g., "Scheduler", "MemoryModel").
            - If type/class: uses component.__name__.
            - If None: defaults to "Global".
        node_id: Optional node index (e.g., 0, 1, 2 ...).
        instance_id: Optional instance index within a node.

    Returns:
        A ComponentLoggerAdapter that you can use like a normal logger.
    """
    if component is None:
        component_name = "Global"
    elif isinstance(component, str):
        component_name = component
    else:
        # Assume it is a class or type-like object.
        component_name = getattr(component, "__name__", str(component))

    base_logger = logging.getLogger(PROJECT_ROOT_LOGGER_NAME)
    return ComponentLoggerAdapter(
        base_logger,
        component=component_name,
        node_id=node_id,
        instance_id=instance_id,
    )


__all__ = ["configure_logger", "get_logger", "PROJECT_ROOT_LOGGER_NAME"]


# ---------------------------------------------------------------------------
# Example usage (for quick manual testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure logger once per process.
    configure_logger(level="DEBUG")

    # 1) Scheduler-like component with node / instance
    scheduler_logger = get_logger("Scheduler", node_id=0, instance_id=1)
    scheduler_logger.info("iteration 0 finished, exposed communication 0 cycles.")
    scheduler_logger.debug("debug info for scheduling decision.")

    # 2) MemoryModel-like global component (no node/instance)
    class MemoryModel:
        def __init__(self) -> None:
            self.logger = get_logger(self.__class__)

        def initialize(self) -> None:
            self.logger.info("Initialized memory configuration.")
            self.logger.warning("Using default DRAM latency; please double-check config.")

    mm = MemoryModel()
    mm.initialize()

    # 3) Completely global utility logger
    util_logger = get_logger()
    util_logger.error("Global error occurred before any node was initialized.")