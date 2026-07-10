"""Shared storage-path and JSON-persistence plumbing used by every manager in this package."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _resolve_storage_path(env_var: str, filename: str) -> Path:
    """Resolve a mutable app-state path with optional env overrides."""
    explicit_path = os.environ.get(env_var)
    if explicit_path:
        return Path(explicit_path)

    data_dir = Path(os.environ.get("PICKLEBALL_DATA_DIR", "data"))
    return data_dir / filename


def _resolve_default_names_path() -> Path:
    """Resolve the tracked default player seed file."""
    explicit_path = os.environ.get("PICKLEBALL_DEFAULT_NAMES_FILE")
    if explicit_path:
        return Path(explicit_path)
    return Path("data/default_player_names.json")


def load_json_value(path: Path, expected_type: type[T], default: T, description: str) -> T:
    """Load JSON from ``path`` if it exists and matches ``expected_type``, else ``default``.

    Any missing file, parse error, or type mismatch is logged and treated as
    "nothing saved yet" rather than raised - callers apply their own
    per-entry validation/filtering on top of the raw loaded value.
    """
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, expected_type):
                return data
    except Exception:
        logger.exception("Failed to load %s from %s", description, path)
    return default


def save_json(path: Path, data: object, description: str) -> bool:
    """Persist ``data`` as JSON to ``path``, creating parent directories as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=2)
        return True
    except Exception:
        logger.exception("Failed to save %s to %s", description, path)
        return False
