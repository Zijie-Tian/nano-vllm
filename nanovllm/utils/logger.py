"""
Global logger for nano-vllm with colored output.

Usage:
    from nanovllm.utils.logger import logger

    logger.debug("Debug message")    # Gray
    logger.info("Info message")      # Blue
    logger.warning("Warning message") # Yellow
    logger.error("Error message")    # Red

Control log level via environment variable:
    NANOVLLM_LOG_LEVEL=DEBUG python your_script.py
    NANOVLLM_LOG_LEVEL=INFO python your_script.py  (default)
    NANOVLLM_LOG_LEVEL=WARNING python your_script.py
"""

import os
import sys
import logging
from typing import Optional


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"


# Level -> Color mapping
LEVEL_COLORS = {
    logging.DEBUG: Colors.GRAY,
    logging.INFO: Colors.BLUE,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.RED + Colors.BOLD,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = LEVEL_COLORS.get(record.levelno, Colors.RESET)
            # Color the level name
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            # Color the message
            record.msg = f"{color}{record.msg}{Colors.RESET}"
        return super().format(record)


# Log level from environment variable
_LOG_LEVEL = os.environ.get("NANOVLLM_LOG_LEVEL", "INFO").upper()

# Global logger instance
_logger: Optional[logging.Logger] = None


def _setup_logger() -> logging.Logger:
    """Setup and return the global logger."""
    log = logging.getLogger("nanovllm")

    # Avoid duplicate handlers if called multiple times
    if log.handlers:
        return log

    # Set level from environment
    level = getattr(logging, _LOG_LEVEL, logging.INFO)
    log.setLevel(level)

    # Create handler (stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # Check if terminal supports colors
    use_colors = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    # Format: [TIME] [LEVEL] [FILE:LINE] message
    formatter = ColoredFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        use_colors=use_colors,
    )
    handler.setFormatter(formatter)

    log.addHandler(handler)

    # Don't propagate to root logger
    log.propagate = False

    return log


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Optional sub-logger name. If provided, creates a child logger
              like 'nanovllm.attention'. If None, returns the root nanovllm logger.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = _setup_logger()

    if name:
        return _logger.getChild(name)
    return _logger


# Convenience: direct access to the root logger
logger = get_logger()
