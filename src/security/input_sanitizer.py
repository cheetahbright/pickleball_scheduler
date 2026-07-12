"""Minimal security helpers used by the launcher and JSON utilities."""

from __future__ import annotations

from pathlib import Path


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
            # Fold absolute paths into the same containment check instead of
            # returning them unchecked - an absolute path must still resolve
            # inside base_dir.
            resolved_target = (
                resolved_path if resolved_path.is_absolute() else (resolved_base / resolved_path)
            ).resolve()
            try:
                resolved_target.relative_to(resolved_base)
            except ValueError as exc:
                raise SecurityError(f"Path escapes allowed base directory: {path}") from exc
            return str(resolved_target)

        return str(resolved_path.resolve())
