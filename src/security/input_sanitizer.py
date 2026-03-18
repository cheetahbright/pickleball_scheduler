"""Minimal security helpers used by the launcher and JSON utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SecurityError(ValueError):
    """Raised when a path or payload fails basic security validation."""


class InputSanitizer:
    """Small compatibility surface for legacy launcher code."""

    @staticmethod
    def sanitize_file_path(path: str, base_dir: str | None = None) -> str:
        """Return a normalized path and optionally keep it inside ``base_dir``."""
        resolved_path = Path(path).expanduser()

        if base_dir:
            resolved_base = Path(base_dir).expanduser().resolve()
            if resolved_path.is_absolute() or path.startswith(("/", "\\")):
                return path

            resolved_target = (resolved_base / resolved_path).resolve()
            try:
                resolved_target.relative_to(resolved_base)
            except ValueError as exc:
                raise SecurityError(f"Path escapes allowed base directory: {path}") from exc
            return str(resolved_target)

        return str(resolved_path.resolve())


class SecureJSONLoader:
    """Compatibility JSON loader with lightweight validation."""

    @staticmethod
    def safe_load(file_path: str) -> Any:
        sanitized_path = InputSanitizer.sanitize_file_path(file_path)
        with open(sanitized_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def safe_loads(content: str) -> Any:
        if not isinstance(content, str):
            raise SecurityError("JSON content must be a string")
        return json.loads(content)
