"""Shared storage-path resolution used by every manager in this package."""

from __future__ import annotations

import os
from pathlib import Path


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
