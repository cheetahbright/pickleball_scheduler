"""Lightweight logging helpers used by the app, the launcher and legacy scripts."""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

LOGGER_NAME = "pickleball_scheduler"
# The app's modules log through logging.getLogger(__name__): "src.managers.history"
# when the app is imported as the src package (the deploy entry point, CI), and
# "managers.history" when it runs from inside src/. None of those is under
# LOGGER_NAME, and the root logger has no handler and passes only warnings, so
# each top-level name gets the console handler too. tests/unit/test_logging_config.py
# checks that every module that logs is covered.
APP_LOGGER_NAMES = ("src", "app_scheduler_flow", "app_tabs", "managers", "simple_auth")
# Each Streamlit session runs main() in its own thread, so two sessions' first runs
# could otherwise both find a logger without a handler and each add one.
_HANDLER_LOCK = threading.Lock()


def _add_console_handler(logger: logging.Logger) -> None:
    with _HANDLER_LOCK:
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Create or reuse a console logger, and send the app modules' log lines to the console too.

    The app's loggers still propagate, so a handler on the root logger (pytest's
    caplog) keeps receiving their records.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    for name in APP_LOGGER_NAMES:
        app_logger = logging.getLogger(name)
        _add_console_handler(app_logger)
        app_logger.setLevel(numeric_level)

    logger = logging.getLogger(LOGGER_NAME)
    _add_console_handler(logger)
    logger.setLevel(numeric_level)
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
