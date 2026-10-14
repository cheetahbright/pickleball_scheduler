"""Shared storage-path and JSON-persistence plumbing used by every manager in this package."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The project root (this file is src/managers/_paths.py, here and in the deploy
# repo), so the app finds its data/ folder from any working directory - run
# from inside src/, a cwd-relative "data" was an empty src/data/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Windows refuses to replace a file another process has open (Python's open()
# doesn't grant delete-sharing), e.g. while another session is reading it or a
# sync client like OneDrive is scanning it. Those holds last milliseconds, so a
# few short retries turn a spurious "could not save" into a successful save.
_REPLACE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _replace_with_retry(source: str, target: Path) -> None:
    for delay in _REPLACE_RETRY_DELAYS_SECONDS:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(source, target)


def _read_bytes_with_retry(path: Path, retry: bool = True) -> bytes:
    """The reader's half of _replace_with_retry: opening a file at the moment
    another session's os.replace swaps it in fails with PermissionError on
    Windows, which must not read as a damaged file."""
    for delay in _REPLACE_RETRY_DELAYS_SECONDS if retry else ():
        try:
            return path.read_bytes()
        except PermissionError:
            time.sleep(delay)
    return path.read_bytes()


def read_json(path: Path, retry: bool = True) -> Any:
    """Parse the JSON file at ``path``, retrying a read blocked by a concurrent save.

    The app writes UTF-8, but a hand-edited file may come back as UTF-8 with
    a byte-order mark (Notepad's "UTF-8 with BOM", Windows PowerShell 5.1's
    -Encoding UTF8) or as UTF-16 (PowerShell 5.1's ``>`` and Out-File).
    json.loads detects all of these from the bytes, so none of them reads as
    an unreadable file. ``retry=False`` tries once, for a caller that already
    knows the file is held by something slower than a save.
    """
    return json.loads(_read_bytes_with_retry(path, retry))


def back_up_unreadable(path: Path) -> Path | None:
    """Copy a file the next save would replace aside, once per distinct content, and return the copy's path."""
    try:
        raw = path.read_bytes()
        backup = path.with_name(f"{path.name}.unreadable-{hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:8]}")
        if not (backup.exists() and backup.read_bytes() == raw):
            partial = backup.with_name(backup.name + ".tmp")
            partial.write_bytes(raw)
            os.replace(partial, backup)
        return backup
    except OSError:
        logger.exception("Could not back up %s", path)
        return None


def _resolve_storage_path(env_var: str, filename: str) -> Path:
    """Resolve a mutable app-state path with optional env overrides."""
    explicit_path = os.environ.get(env_var)
    if explicit_path:
        return Path(explicit_path)

    data_dir = os.environ.get("PICKLEBALL_DATA_DIR")
    return (Path(data_dir) if data_dir else PROJECT_ROOT / "data") / filename


def _resolve_default_names_path() -> Path:
    """Resolve the tracked default player seed file."""
    explicit_path = os.environ.get("PICKLEBALL_DEFAULT_NAMES_FILE")
    if explicit_path:
        return Path(explicit_path)
    return PROJECT_ROOT / "data" / "default_player_names.json"


def load_json_value(path: Path, expected_type: type[T], default: T, description: str) -> T:
    """Load JSON from ``path`` if it exists and matches ``expected_type``, else ``default``.

    Any missing file, parse error, or type mismatch is logged and treated as
    "nothing saved yet" rather than raised - callers apply their own
    per-entry validation/filtering on top of the raw loaded value.
    """
    try:
        if path.exists():
            data = read_json(path)
            if isinstance(data, expected_type):
                return data
            logger.warning(
                "Ignored %s in %s: expected a JSON %s, found %s",
                description,
                path,
                expected_type.__name__,
                type(data).__name__,
            )
    except Exception:
        logger.exception("Failed to load %s from %s", description, path)
    return default


def save_json(path: Path, data: object, description: str) -> bool:
    """Persist ``data`` as JSON to ``path``, creating parent directories as needed.

    Writes atomically: the new content lands in a temp file in the same
    directory first, and is only swapped onto ``path`` via os.replace() after
    a fully successful write. A failure mid-write (disk full, interrupted
    process) then leaves the previous file exactly as it was, instead of
    truncating it in place.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                json.dump(data, file_handle, indent=2)
            _replace_with_retry(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise
        return True
    except Exception:
        logger.exception("Failed to save %s to %s", description, path)
        return False
