#!/usr/bin/env python3
"""Full backup and restore of everything the app saves.

The hosted app keeps its data in local files that every reboot deletes, so a
full backup, downloaded before a reboot and restored after it, is how that
data survives. One ZIP file holds:

- schedule_history.db: a consistent snapshot of the history database
  (schedules, scores, weekly partner history);
- app_config.json, player_skills.json and player_elo.json, each one that is
  saved;
- manifest.json: the format, when the backup was made, and each file's size
  and SHA-256 (or null for a file that wasn't saved), so a damaged or edited
  backup is refused, never half restored.

Restoring makes the saved data exactly what the backup holds, so it also
removes a config or rating file the backup has none of. The copy of the
current data that a restore keeps first holds whatever it replaced.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from src.managers._paths import _REPLACE_RETRY_DELAYS_SECONDS, _read_bytes_with_retry, _replace_with_retry
    from src.managers.config import MAX_CONFIG_UPLOAD_BYTES, import_config_json
    from src.managers.history import describe_history_db
except ImportError:
    from managers._paths import _REPLACE_RETRY_DELAYS_SECONDS, _read_bytes_with_retry, _replace_with_retry
    from managers.config import MAX_CONFIG_UPLOAD_BYTES, import_config_json
    from managers.history import describe_history_db

logger = logging.getLogger(__name__)

BACKUP_FORMAT = "pickleball-scheduler-full-backup"
BACKUP_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
HISTORY_DB_NAME = "schedule_history.db"
CONFIG_NAME = "app_config.json"
SKILLS_NAME = "player_skills.json"
ELO_NAME = "player_elo.json"
JSON_FILE_NAMES = (CONFIG_NAME, SKILLS_NAME, ELO_NAME)

# Limits on what a restore unpacks, so a small ZIP can't expand into gigabytes.
# Five years of weekly 16-player sessions make a 2.2 MB database.
MAX_BACKUP_DB_BYTES = 50_000_000
MAX_BACKUP_JSON_BYTES = MAX_CONFIG_UPLOAD_BYTES
_MAX_MANIFEST_BYTES = 64_000

# The copy of the current data a restore keeps (next to the history database) before replacing it.
KEPT_COPY_PREFIX = "full_backup_before_restore_"

# One restore at a time: two at once would interleave their steps, leaving the
# database of one backup with the JSON files of the other.
_RESTORE_LOCK = threading.Lock()

# How to rewrite a saved file that a backup can't include as it is.
_REWRITE_HINTS = {
    SKILLS_NAME: "Saving the skill ratings in Player Management rewrites it.",
    ELO_NAME: "Recomputing the ELO ratings on the Leaderboard tab rewrites it.",
}


class BackupError(Exception):
    """A full backup couldn't be made, read or restored. The message says why, for the user."""


@dataclass(frozen=True)
class FullBackup:
    """A full backup that passed every check in read_full_backup, ready for restore_full_backup."""

    created_at: str
    history_db: bytes
    json_files: Dict[str, bytes]
    schedules: int
    scores: int
    presets: int | None
    skill_ratings: int | None
    elo_ratings: int | None

    def describe(self) -> list[str]:
        """One line per part of the backup, saying what restoring it brings back."""
        return [
            f"Made {self.created_at[:16].replace('T', ' ')}",
            f"Schedule history: {self.schedules} schedule(s), {self.scores} recorded game score(s)",
            (
                f"Configuration, with {self.presets} player preset(s)"
                if self.presets is not None
                else "Configuration: none saved, so the default settings apply"
            ),
            (
                f"Skill ratings for {self.skill_ratings} player(s)"
                if self.skill_ratings is not None
                else "Skill ratings: none saved"
            ),
            (
                f"ELO ratings for {self.elo_ratings} player(s)"
                if self.elo_ratings is not None
                else "ELO ratings: none saved"
            ),
        ]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_saved_file(path: Path) -> bytes | None:
    """The file's bytes, or None if it doesn't exist (a read blocked by a concurrent save is retried)."""
    try:
        return _read_bytes_with_retry(path)
    except FileNotFoundError:
        return None


def _json_object(name: str, data: bytes) -> Dict[str, Any]:
    try:
        value = json.loads(data)  # detects a BOM or UTF-16 itself, as read_json does
    except (ValueError, RecursionError) as exc:
        raise BackupError(f"{name} isn't valid JSON ({exc}).") from exc
    if not isinstance(value, dict):
        raise BackupError(f"{name} doesn't hold a JSON object.")
    return value


def build_full_backup(history_db: bytes, json_files: Mapping[str, bytes], created_at: datetime | None = None) -> bytes:
    """Zip a history snapshot and the given JSON files (by name) with their manifest.

    A name of JSON_FILE_NAMES missing from ``json_files`` is listed as null:
    that file wasn't saved, and restoring the backup removes it.
    """
    files = {HISTORY_DB_NAME: history_db, **json_files}
    manifest = {
        "format": BACKUP_FORMAT,
        "version": BACKUP_FORMAT_VERSION,
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
        "files": {
            name: {"size": len(files[name]), "sha256": _sha256(files[name])} if name in files else None
            for name in (HISTORY_DB_NAME, *JSON_FILE_NAMES)
        },
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def make_full_backup(
    history_manager, data_paths: Mapping[str, Path], config_json: str, default_config: Dict[str, Any]
) -> bytes:
    """The "Download full backup" file: a history snapshot, ``config_json``, and the saved rating files.

    ``config_json`` is the configuration as "Download config" exports it.
    ``data_paths`` maps SKILLS_NAME and ELO_NAME to where those files are
    saved; one that doesn't exist yet is left out. One that exists but can't
    be read as JSON stops the backup with a BackupError, rather than leaving
    it out unnoticed or making a backup that restore would refuse.

    The finished file goes through read_full_backup (with ``default_config``)
    before it is returned, so a backup that restore would refuse (a damaged
    database, a config the import check rejects) fails here, while the data
    it should hold still exists, not after a reboot has deleted it.
    """
    json_files = {CONFIG_NAME: config_json.encode("utf-8")}
    for name in (SKILLS_NAME, ELO_NAME):
        data = _read_saved_file(data_paths[name])
        if data is None:
            continue
        try:
            _json_object(name, data)
        except BackupError as exc:
            raise BackupError(f"No full backup was made: the saved {exc} {_REWRITE_HINTS[name]}") from exc
        json_files[name] = data
    history_db = history_manager.export_db_bytes()
    if len(history_db) > MAX_BACKUP_DB_BYTES:
        raise BackupError(
            f"No full backup was made: the schedule history ({len(history_db)} bytes) is larger than a restore "
            f"accepts ({MAX_BACKUP_DB_BYTES} bytes)."
        )
    raw = build_full_backup(history_db, json_files)
    try:
        read_full_backup(raw, default_config)
    except BackupError as exc:
        raise BackupError(
            f"No full backup was made, because restoring it would fail: {exc} Don't reboot the hosted app "
            "until a full backup downloads: the reboot would delete this data."
        ) from exc
    return raw


def _read_manifest(archive: zipfile.ZipFile) -> Dict[str, Any]:
    try:
        info = archive.getinfo(MANIFEST_NAME)
    except KeyError:
        raise BackupError(f"This isn't a full backup: it has no {MANIFEST_NAME}.") from None
    if info.file_size > _MAX_MANIFEST_BYTES:
        raise BackupError(f"This isn't a full backup: its {MANIFEST_NAME} is too large.")
    manifest = _json_object(MANIFEST_NAME, archive.read(info))
    if manifest.get("format") != BACKUP_FORMAT:
        raise BackupError("This isn't a full backup from this app.")
    version = manifest.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise BackupError(f"This backup's {MANIFEST_NAME} has no valid format version.")
    if version > BACKUP_FORMAT_VERSION:
        raise BackupError("This backup was made by a newer version of the app. Update the app, then restore it.")
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get(HISTORY_DB_NAME) is None:
        raise BackupError(f"This backup's {MANIFEST_NAME} doesn't list a {HISTORY_DB_NAME}.")
    for name in JSON_FILE_NAMES:
        if name not in files:
            raise BackupError(f"This backup's {MANIFEST_NAME} doesn't say whether it holds {name}.")
    for name, entry in files.items():
        if name not in (HISTORY_DB_NAME, *JSON_FILE_NAMES):
            raise BackupError(f"This backup holds a file this app doesn't know: {name}.")
        if entry is not None and not (
            isinstance(entry, dict)
            and isinstance(entry.get("size"), int)
            and not isinstance(entry.get("size"), bool)
            and isinstance(entry.get("sha256"), str)
        ):
            raise BackupError(f"This backup's {MANIFEST_NAME} has no valid size and checksum for {name}.")
    if not isinstance(manifest.get("created_at"), str):
        raise BackupError(f"This backup's {MANIFEST_NAME} doesn't say when it was made.")
    return manifest


def _read_listed_file(archive: zipfile.ZipFile, name: str, entry: Dict[str, Any]) -> bytes:
    """A file the manifest lists, checked against its size and checksum (and the size limits)."""
    limit = MAX_BACKUP_DB_BYTES if name == HISTORY_DB_NAME else MAX_BACKUP_JSON_BYTES
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise BackupError(f"This backup is incomplete: {name} is missing.") from None
    if info.file_size > limit or entry["size"] > limit:
        raise BackupError(f"{name} in this backup is larger than a restore accepts ({limit} bytes).")
    with archive.open(info) as member:
        data = member.read(limit + 1)  # zipfile also checks the member's CRC as it reads
    if len(data) != entry["size"] or _sha256(data) != entry["sha256"]:
        raise BackupError(f"This backup is damaged: {name} doesn't match its checksum.")
    return data


def read_full_backup(raw: bytes, default_config: Dict[str, Any]) -> FullBackup:
    """Check an uploaded full backup end to end, without changing anything. Raises BackupError.

    Every listed file must match its size and checksum, the history database
    must pass describe_history_db, the rating files must be JSON objects, and
    the configuration must import with no repairs (as "Upload config"
    requires).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            manifest = _read_manifest(archive)
            files = {
                name: _read_listed_file(archive, name, entry)
                for name, entry in manifest["files"].items()
                if entry is not None
            }
    except BackupError:
        raise
    except zipfile.BadZipFile as exc:
        raise BackupError(f"This isn't a full backup file: it isn't a readable ZIP archive ({exc}).") from exc
    except Exception as exc:  # an encrypted or oddly compressed member, a truncated stream
        raise BackupError(f"This backup couldn't be read ({exc}).") from exc

    try:
        schedules, scores = describe_history_db(files[HISTORY_DB_NAME])
    except ValueError as exc:
        raise BackupError(f"{HISTORY_DB_NAME} in this backup can't be restored: {exc}.") from exc

    presets = None
    if CONFIG_NAME in files:
        config, problems = import_config_json(files[CONFIG_NAME], default_config)
        if problems:
            raise BackupError(f"{CONFIG_NAME} in this backup has problems: {' '.join(problems)}")
        presets = len(config.get("player_presets", {}))
    ratings = {name: len(_json_object(name, files[name])) for name in (SKILLS_NAME, ELO_NAME) if name in files}

    return FullBackup(
        created_at=manifest["created_at"],
        history_db=files[HISTORY_DB_NAME],
        json_files={name: data for name, data in files.items() if name in JSON_FILE_NAMES},
        schedules=schedules,
        scores=scores,
        presets=presets,
        skill_ratings=ratings.get(SKILLS_NAME),
        elo_ratings=ratings.get(ELO_NAME),
    )


def _write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file beside ``path``, then swap it in, so ``path`` is never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file_handle:
            file_handle.write(data)
        _replace_with_retry(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _keep_copy(directory: Path, archive: bytes) -> Path:
    """Save ``archive`` in ``directory`` under a new name that sorts after every earlier kept copy; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    while True:
        path = directory / f"{KEPT_COPY_PREFIX}{stamp:%Y%m%d-%H%M%S-%f}.zip"
        try:
            file_handle = path.open("xb")
        except FileExistsError:
            # Made within the same clock tick as an earlier copy: the next free microsecond still sorts after it.
            stamp += timedelta(microseconds=1)
            continue
        try:
            with file_handle:
                file_handle.write(archive)
        except BaseException:
            with contextlib.suppress(OSError):
                path.unlink()
            raise
        return path


def latest_kept_copy(directory: Path) -> Path | None:
    """The copy of the data that the most recent restore replaced, if one is in ``directory``.

    Ordered by when each was written, and by name (when it was made, to the
    microsecond) among copies the file system dates the same.
    """
    try:
        return max(
            directory.glob(f"{KEPT_COPY_PREFIX}*.zip"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            default=None,
        )
    except OSError:
        return None


def _unlink_with_retry(path: Path) -> None:
    """Remove ``path`` if it exists, retrying while another session's read holds it open (see _replace_with_retry)."""
    for delay in _REPLACE_RETRY_DELAYS_SECONDS:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
    path.unlink(missing_ok=True)


def restore_full_backup(backup: FullBackup, history_manager, data_paths: Mapping[str, Path]) -> Path:
    """Replace the saved data with ``backup``'s, keeping a copy of the current data first.

    ``data_paths`` maps each of JSON_FILE_NAMES to where it is saved. The copy
    is a full backup of the files as they are now, saved next to the history
    database, so restoring it undoes this restore; its path is returned. The
    history database is copied in within one transaction, each JSON file is
    swapped in whole, and one the backup has none of is removed. If any step
    fails, everything already replaced is put back and BackupError says what
    happened. A restore started while another is running is refused.
    """
    if not _RESTORE_LOCK.acquire(blocking=False):
        raise BackupError(
            "Another full restore is running right now, so nothing was restored. Try again when it has finished."
        )
    try:
        return _restore(backup, history_manager, data_paths)
    finally:
        _RESTORE_LOCK.release()


def _restore(backup: FullBackup, history_manager, data_paths: Mapping[str, Path]) -> Path:
    try:
        previous_db = history_manager.export_db_bytes()
        previous_files = {name: _read_saved_file(data_paths[name]) for name in JSON_FILE_NAMES}
        kept = _keep_copy(
            history_manager.db_path.parent,
            build_full_backup(previous_db, {name: data for name, data in previous_files.items() if data is not None}),
        )
    except Exception as exc:
        logger.exception("Full restore: could not keep a copy of the current data")
        raise BackupError(f"Couldn't keep a copy of the current data ({exc}), so nothing was restored.") from exc
    logger.info("Full restore: kept the current data as %s", kept)

    # The files changed so far, to put back if a later step fails. A step that
    # fails leaves its file as it was (the database copy-in is one transaction,
    # a JSON file is swapped in whole), so a file is listed once its step succeeds.
    replaced: list[str] = []
    step = HISTORY_DB_NAME
    try:
        history_manager.import_db_bytes(backup.history_db)
        replaced.append(HISTORY_DB_NAME)
        history_manager.ensure_db_exists()  # a database from an older version may lack a table or column
        for step in JSON_FILE_NAMES:
            if step in backup.json_files:
                _write_atomically(data_paths[step], backup.json_files[step])
            else:
                _unlink_with_retry(data_paths[step])  # the kept copy holds it
            replaced.append(step)
    except Exception as exc:
        logger.exception("Full restore failed at %s; putting the previous data back", step)
        failed_put_back = _put_back(history_manager, data_paths, replaced, previous_db, previous_files)
        if failed_put_back:
            raise BackupError(
                f"Couldn't restore the backup ({exc}), and putting the previous {', '.join(failed_put_back)} back "
                f"failed too. A copy of the previous data is kept as {kept.name}: restore that file to get it back."
            ) from exc
        outcome = "so the previous data was put back" if replaced else "and nothing was changed"
        raise BackupError(
            f"Couldn't restore the backup ({exc}), {outcome}. A copy of the previous data is also kept as {kept.name}."
        ) from exc
    logger.info("Full restore: restored a backup made %s", backup.created_at)
    return kept


def _put_back(
    history_manager,
    data_paths: Mapping[str, Path],
    replaced: list[str],
    previous_db: bytes,
    previous_files: Mapping[str, bytes | None],
) -> list[str]:
    """Undo a failed restore's steps; return the names of the files that couldn't be put back."""
    failed = []
    for name in replaced:
        previous = previous_files.get(name)
        try:
            if name == HISTORY_DB_NAME:
                history_manager.import_db_bytes(previous_db)
            elif previous is None:
                _unlink_with_retry(data_paths[name])  # it didn't exist before the restore
            else:
                _write_atomically(data_paths[name], previous)
        except Exception:
            logger.exception("Full restore: could not put back %s", name)
            failed.append(name)
    return failed
