#!/usr/bin/env python3
"""Shared helper for the repo's dual import-path pattern.

The app supports being run both as `python -m src.main_app` (package-qualified,
used in CI/deployment) and as `python main_app.py` from inside `src/` (bare
module name, used in local development). Every affected module previously
repeated:

    try:
        from src.foo import bar
    except ImportError:
        from foo import bar

`import_module_with_fallback` consolidates that into one call that returns the
resolved module object, so callers do:

    _foo = import_module_with_fallback("foo")
    bar = _foo.bar
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def import_module_with_fallback(module_name: str) -> ModuleType:
    """Import `src.<module_name>` if available, else the bare `<module_name>`."""
    try:
        return import_module(f"src.{module_name}")
    except ImportError:
        return import_module(module_name)
