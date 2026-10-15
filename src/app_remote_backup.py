#!/usr/bin/env python3
"""Streamlit glue for the encrypted cloud backup (src/managers/remote_backup.py).

Reads the settings from ``st.secrets`` as tolerantly as src/simple_auth.py reads
the password (no secrets file, a byte-order mark, an unreadable file), finds
the process-wide sync, and runs the start-of-run step: restoring a fresh start
from the cloud copy. Nothing here raises for a cloud problem.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

_remote = import_module_with_fallback("managers.remote_backup")
_backup = import_module_with_fallback("managers.backup")
_app_managers = import_module_with_fallback("app_managers")
_schedule_helpers = import_module_with_fallback("app_schedule_helpers")

logger = logging.getLogger(__name__)

# The restore generation this session has shown and adopted (see before_tabs).
ADOPTED_GENERATION_KEY = "_remote_backup_adopted_generation"
_PAUSE_NOTICE_KEY = "_remote_backup_pause_noticed"
_logged_problem: str | None = None


@dataclass(frozen=True)
class CloudBackupView:
    """What the page knows about the cloud backup: the sync when it is on, or why it isn't."""

    sync: Any | None = None  # a RemoteBackupSync
    problem: str | None = None  # a settings problem, shown as a warning; None when it is simply not set up


def _is_missing_secrets(exc: BaseException) -> bool:
    """Whether Streamlit raised its "no secrets file" error (matched by name: the class may be a stand-in in tests)."""
    return any(cls.__name__ == "StreamlitSecretNotFoundError" for cls in type(exc).__mro__)


def _secrets_table(st_module) -> tuple[Mapping[str, Any] | None, str | None]:
    """The ``[remote_backup]`` secrets table and a problem with reading the secrets, either of which may be None."""
    secrets = getattr(st_module, "secrets", None)
    section = _remote.SECRETS_SECTION
    if secrets is None or not hasattr(secrets, "get"):
        return None, None
    try:
        table = secrets.get(section)
        if table is None and f"\ufeff{section}" in secrets:
            return None, (
                "The secrets file (secrets.toml) starts with a byte-order mark, so its [remote_backup] table can't "
                "be read. Save it as UTF-8 without a BOM and reload."
            )
    except UnicodeDecodeError:
        return None, "The secrets file (secrets.toml) could not be read as UTF-8 text. Save it as UTF-8 and reload."
    except Exception as exc:
        if _is_missing_secrets(exc) and exc.__cause__ is None:
            return None, None  # no secrets file: the environment variables may still hold the settings
        if _is_missing_secrets(exc):
            return None, "The secrets file (secrets.toml) could not be read, so cloud backup is off. Fix the file."
        logger.warning("Cloud backup: reading the secrets failed (%s)", type(exc).__name__)
        return None, "The secrets couldn't be read, so cloud backup is off."
    if isinstance(table, (str, bytes, int, float, list)):
        return None, f"[{section}] in the secrets must be a table: a [{section}] heading with the settings under it."
    return (table if isinstance(table, Mapping) else None), None


def full_backup_data_paths(session_state) -> dict[str, Path]:
    """Where each JSON file of a full backup is saved, as this session's managers resolve it."""
    skill_manager = session_state.get("skill_manager") or _app_managers.SkillRatingManager()
    elo_manager = session_state.get("elo_manager") or _app_managers.EloRatingManager()
    return {
        _backup.CONFIG_NAME: session_state.config_manager.config_path,
        _backup.SKILLS_NAME: skill_manager.skills_path,
        _backup.ELO_NAME: elo_manager.ratings_path,
    }


def viewer_time_zone(st_module):
    """The time zone of the viewer's browser, or None (UTC is shown) when it isn't known."""
    try:
        name = st_module.context.timezone
        return ZoneInfo(name) if isinstance(name, str) and name else None
    except (AttributeError, OSError, ValueError, ZoneInfoNotFoundError):
        return None


def cloud_backup(st_module) -> CloudBackupView:
    """The cloud backup for this page: its sync, or why there is none ("Not set up" has neither). Never raises."""
    try:
        return _cloud_backup(st_module)
    except Exception as exc:
        problem = f"Cloud backup failed to start ({type(exc).__name__})."
        logger.warning("Cloud backup: %s", problem)
        return CloudBackupView(problem=problem)


def _is_e2e_run() -> bool:
    """Whether this is a browser-automation run (E2E_TEST, as simple_auth reads it)."""
    return os.environ.get("E2E_TEST", "").lower() in ("1", "true", "yes")


def _cloud_backup(st_module) -> CloudBackupView:
    global _logged_problem
    if _is_e2e_run():
        # A test server started from a developer's checkout reads the developer's own secrets.toml: it must never
        # restore from, or send test data to, the real cloud copy.
        return CloudBackupView()
    table, problem = _secrets_table(st_module)
    if problem is None:
        try:
            settings = _remote.load_settings(table, os.environ)
        except _remote.RemoteBackupConfigError as exc:
            problem = str(exc)
        else:
            if settings is None:
                return CloudBackupView()
            session_state = st_module.session_state
            history_manager = session_state.get("history_manager")
            config_manager = session_state.get("config_manager")
            if history_manager is None or config_manager is None:
                return CloudBackupView(problem="The schedule history can't be opened, so there is nothing to back up.")
            sync = _remote.get_process_sync(
                settings, history_manager, full_backup_data_paths(session_state), config_manager
            )
            return CloudBackupView(sync=sync)
    if problem != _logged_problem:  # every rerun asks again; the log hears about it once
        _logged_problem = problem
        logger.warning("Cloud backup is not working: %s", problem)
    return CloudBackupView(problem=problem)


def before_tabs(st_module) -> str | None:
    """Start of a script run, once the managers exist: restore a fresh start from the cloud copy.

    Runs in the session's script thread, before the configuration is loaded, so
    the restored data is what this run shows. Another session may be the one
    that restored (or may restore later, after a retry): this session then
    drops its loaded configuration so it is read again, and says what happened.
    Returns a warning for the page when automatic backups are paused, once per
    reason in this session.
    """
    view = cloud_backup(st_module)
    sync = view.sync
    if sync is None:
        return None
    session_state = st_module.session_state
    adopted = session_state.get(ADOPTED_GENERATION_KEY)
    if adopted is None:
        adopted = sync.restore_generation  # a session that starts after a restore reads the restored data anyway
    sync.ensure_started()
    status = sync.status()
    if status.restore_generation > adopted and status.restored_from is not None:
        message = _remote.restore_message(status.restored_from, viewer_time_zone(st_module), status.restored_made_at)
        if status.restored_kept is not None:
            kept_name = _schedule_helpers.escape_markdown(status.restored_kept.name)
            message += f" The data it replaced is kept as {kept_name}."
        session_state.global_status_message = f"✅ {message}"
        session_state.pop("app_config", None)  # loaded again below, from the restored file
    session_state[ADOPTED_GENERATION_KEY] = max(adopted, status.restore_generation)
    pause = status.pause_text(viewer_time_zone(st_module))
    if not pause:
        session_state.pop(_PAUSE_NOTICE_KEY, None)
    elif session_state.get(_PAUSE_NOTICE_KEY) != status.paused_reason:
        session_state[_PAUSE_NOTICE_KEY] = status.paused_reason
        return f"⚠️ {pause} See Configuration > Backup / Restore > Cloud backup."
    return None


def after_tabs(st_module) -> None:
    """End of a script run: ask for a cloud backup if the saved data changed during it."""
    sync = cloud_backup(st_module).sync
    if sync is not None:
        sync.note_run()
