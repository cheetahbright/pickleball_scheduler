"""Lightweight logging helpers used by the launcher and legacy scripts."""

from __future__ import annotations

import logging
from typing import Any, Mapping

LOGGER_NAME = "pickleball_scheduler"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Create or reuse a console logger."""
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_error_with_context(
    logger: logging.Logger | None,
    error: BaseException,
    message: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Log an error with simple key=value context rendering."""
    rendered_context = ""
    if context:
        rendered_context = " | " + ", ".join(f"{key}={value}" for key, value in sorted(context.items()))

    target_logger = logger or setup_logging()
    target_logger.error("%s: %s%s", message, error, rendered_context)
