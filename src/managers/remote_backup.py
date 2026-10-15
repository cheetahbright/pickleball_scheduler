#!/usr/bin/env python3
"""An encrypted cloud copy of the full backup, restored automatically on a fresh start.

The hosted app keeps its data in local files that every reboot (and the
platform's own sleep/restart) deletes, so the "Download full backup" ZIP has to
be downloaded and uploaded by hand. This module also keeps that same, verified
full-backup ZIP in a hosted Postgres database (Neon), encrypted on this side,
and restores the newest copy when the app starts with no data. See
docs/REMOTE_BACKUP.md for the setup and the threat model.

Rules the code below keeps:

- The local files stay the source of truth. The cloud holds snapshots only, and
  nothing the cloud does (an outage, a wrong key, a bad password, a timeout, any
  exception) may break or noticeably slow the app: every remote call is
  wrapped, bounded by short timeouts, logged without the connection string,
  password or key, and turned into a one-line status for the UI.
- The database only ever sees AES-256-GCM ciphertext, its size, its time and the
  namespace. The key never leaves the settings object, which hides it (and the
  password) from repr() and str().
- TLS is always on: an sslmode weaker than "require" is refused, and with none
  given the connection is verified (verify-full against the system's roots).
- It is off unless configured, from the ``[remote_backup]`` Streamlit secrets
  table or the PICKLEBALL_REMOTE_BACKUP_* environment variables.
- The ZIP is built, read and restored by the existing, verified code in
  backup.py, through the three small wrappers below.

This module has no Streamlit imports; src/app_remote_backup.py is the glue.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import secrets
import socket
import sqlite3
import threading
import time
import urllib.parse
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Protocol

try:
    from src import features as _features
    from src.managers import backup as _backup
    from src.managers.config import export_config_json
except ImportError:
    import features as _features  # type: ignore[no-redef]
    from managers import backup as _backup
    from managers.config import export_config_json

logger = logging.getLogger(__name__)

SECRETS_SECTION = "remote_backup"
ENV_URL = "PICKLEBALL_REMOTE_BACKUP_URL"
ENV_KEY = "PICKLEBALL_REMOTE_BACKUP_KEY"
ENV_NAMESPACE = "PICKLEBALL_REMOTE_BACKUP_NAMESPACE"
ENV_KEEP = "PICKLEBALL_REMOTE_BACKUP_KEEP"
ENV_AUTO_RESTORE = "PICKLEBALL_REMOTE_BACKUP_AUTO_RESTORE"
ENV_VARS = (ENV_URL, ENV_KEY, ENV_NAMESPACE, ENV_KEEP, ENV_AUTO_RESTORE)
SETTING_NAMES = ("database_url", "encryption_key", "namespace", "keep", "auto_restore")

DEFAULT_NAMESPACE = "default"
DEFAULT_KEEP = 20
MIN_KEEP, MAX_KEEP = 1, 100
# Namespaces the helper tool (scripts/remote_backup_tool.py) uses and empties, so a real one can't be named so.
RESERVED_NAMESPACES = ("smoke-test", "check-privileges")
_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# The stored blob: 1 byte format version, 12 bytes nonce, then AES-256-GCM ciphertext and its 16-byte tag.
FORMAT_VERSION = 1
KEY_BYTES = 32
MIN_DISTINCT_KEY_BYTES = 8
NONCE_BYTES = 12
TAG_BYTES = 16
# The largest blob stored or restored (a year of weekly sessions is well under 1 MB).
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

CONNECT_TIMEOUT_SECONDS = 8
STATEMENT_TIMEOUT_MS = 15_000
# The statement timeout is enforced by the server, which doesn't help when the server or the network goes quiet
# after the connection is made, so a whole database call (connect, statement, commit) also gets this deadline.
OPERATION_DEADLINE_SECONDS = 25.0
# Calls that missed their deadline and are still running (each one is cut off, but a wedged network may keep it).
MAX_STUCK_CALLS = 3
# libpq's own limits for a connection that goes quiet; the deadline above is the bound that always holds.
CONNECT_OPTIONS = {
    "connect_timeout": CONNECT_TIMEOUT_SECONDS,
    "prepare_threshold": None,
    "keepalives": 1,
    "keepalives_idle": 10,
    "keepalives_interval": 5,
    "keepalives_count": 3,
    "tcp_user_timeout": 20_000,
}

PUSH_MIN_INTERVAL_SECONDS = 20.0
BACKOFF_BASE_SECONDS = 30.0
BACKOFF_CAP_SECONDS = 15 * 60.0
# How long a manual push or restore waits for one already running before giving up.
MANUAL_WAIT_SECONDS = 45.0
# How long the first page run of a process waits for the startup restore; a slower one finishes in the background.
STARTUP_WAIT_SECONDS = 6.0
# A restored copy's own made-time and the time its row says differ by more than this: worth a warning.
MADE_AT_SLACK_SECONDS = 3600.0
# A cloud copy is "newer" than local data only by more than this (the two clocks differ a little).
CLOCK_SLACK_SECONDS = 120.0
WORKER_POLL_SECONDS = 30.0
WORKER_IDLE_EXIT_SECONDS = 600.0

TABLE_NAME = "pickleball_backups"
SETUP_SQL = (
    "CREATE TABLE IF NOT EXISTS pickleball_backups (id BIGSERIAL PRIMARY KEY, namespace TEXT NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), format_version INTEGER NOT NULL, "
    "size_bytes INTEGER NOT NULL, payload BYTEA NOT NULL)",
    "CREATE INDEX IF NOT EXISTS pickleball_backups_namespace_created "
    "ON pickleball_backups (namespace, created_at DESC)",
)
_INSERT_SQL = (
    "INSERT INTO pickleball_backups (namespace, format_version, size_bytes, payload) "
    "VALUES (%s, %s, %s, %b) RETURNING id, created_at"
)
# The app's own rows are stamped by the database when they are inserted, so a row dated in the future (or before
# 2000, or "infinity") was written by hand. Reads ignore such a row, so it can't outrank every later backup or
# crash the page, and the prune deletes it.
_PRUNE_SQL = (
    "DELETE FROM pickleball_backups WHERE namespace = %s AND (created_at < '2000-01-01' OR created_at > now() "
    "OR id NOT IN (SELECT id FROM pickleball_backups WHERE namespace = %s AND created_at >= '2000-01-01' "
    "AND created_at <= now() ORDER BY created_at DESC, id DESC LIMIT %s))"
)
_LATEST_SQL = (
    "SELECT id, created_at, format_version, size_bytes, octet_length(payload), "
    "CASE WHEN octet_length(payload) <= %s THEN payload END "
    "FROM pickleball_backups WHERE namespace = %s AND created_at >= '2000-01-01' AND created_at <= now() "
    "ORDER BY created_at DESC, id DESC LIMIT 1"
)
_LATEST_INFO_SQL = (
    "SELECT id, created_at, format_version, size_bytes FROM pickleball_backups "
    "WHERE namespace = %s AND created_at >= '2000-01-01' AND created_at <= now() "
    "ORDER BY created_at DESC, id DESC LIMIT 1"
)
# MemoryStore applies the same rule to its own rows.
_EARLIEST_SANE_ROW = datetime(2000, 1, 1, tzinfo=timezone.utc)
_DELETE_NAMESPACE_SQL = "DELETE FROM pickleball_backups WHERE namespace = %s"
_SELECT_PROBE_SQL = "SELECT id FROM pickleball_backups WHERE namespace = %s LIMIT 1"

TABLE_MISSING_MESSAGE = (
    "The cloud database has no backup table yet: run the setup SQL in docs/REMOTE_BACKUP.md (in the project's source "
    "repository, not in the deployed app) in the Neon SQL editor."
)
SETUP_DOC = "docs/REMOTE_BACKUP.md in the source repository"


class RemoteBackupError(Exception):
    """A cloud backup step failed. The message is one line for the user and never holds a secret."""


class RemoteBackupConfigError(RemoteBackupError):
    """The cloud backup settings are missing something or wrong."""


class RemoteBackupDecryptError(RemoteBackupError):
    """A cloud copy couldn't be decrypted: the wrong key, or a damaged or altered copy."""


class _DataAppeared(RemoteBackupError):
    """An automatic restore found data where there was none a moment ago, and left it alone."""


# --- The three calls into backup.py -------------------------------------------------------------
# Every call to make_full_backup, read_full_backup and restore_full_backup goes through these, so a
# change to their signatures touches one place. make and restore pass the score_tracking feature flag, read
# at call time like everywhere else: with it off, a damaged player_elo.json (nothing in the app can rewrite it
# then) is left out of the cloud copy and of the copy a restore keeps, instead of blocking every push.


def _make_backup(history_manager, data_paths: Mapping[str, Path], config_json: str, default_config: dict) -> bytes:
    return _backup.make_full_backup(
        history_manager, data_paths, config_json, default_config, score_tracking=_features.SCORE_TRACKING
    )


def _read_backup(raw: bytes, default_config: dict):
    return _backup.read_full_backup(raw, default_config)


def _restore_backup(backup, history_manager, data_paths: Mapping[str, Path], config_manager) -> Path:
    return _backup.restore_full_backup(
        backup, history_manager, data_paths, config_manager, score_tracking=_features.SCORE_TRACKING
    )


# --- Settings ------------------------------------------------------------------------------------

_TLS_MODES = ("require", "verify-ca", "verify-full")
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+={0,2}")


@dataclass(frozen=True, repr=False)
class RemoteBackupSettings:
    """Where the cloud copy lives and how it is encrypted. Its repr() and str() hide the password and key."""

    database_url: str
    encryption_key: bytes
    namespace: str = DEFAULT_NAMESPACE
    keep: int = DEFAULT_KEEP
    auto_restore: bool = True

    def __repr__(self) -> str:
        return (
            f"RemoteBackupSettings(database_url=<hidden>, encryption_key=<hidden>, namespace={self.namespace!r}, "
            f"keep={self.keep}, auto_restore={self.auto_restore})"
        )

    def conninfo(self) -> str:
        """The connection URL to use: ``database_url`` with the TLS rules applied (see secure_conninfo)."""
        return secure_conninfo(self.database_url)

    def secret_strings(self) -> tuple[str, ...]:
        """Every text that must never reach a log or a message: the URL, its password, and the key in each spelling."""
        found = [self.database_url]
        try:
            split = urllib.parse.urlsplit(self.database_url)
            userinfo, password = split.netloc.rpartition("@")[0], split.password
        except ValueError:
            userinfo, password = "", None
        if password:
            found += [password, urllib.parse.unquote(password)]
        # The driver may split the user info at another "@" than urllib does, and echo only a part of the password.
        found += [part for part in userinfo.partition(":")[2].split("@") if part]
        found += [part for part in _libpq_passwords(self.database_url) if part]
        if self.encryption_key:
            found += [
                base64.urlsafe_b64encode(self.encryption_key).decode("ascii"),
                base64.urlsafe_b64encode(self.encryption_key).decode("ascii").rstrip("="),
                base64.b64encode(self.encryption_key).decode("ascii"),
                self.encryption_key.hex(),
            ]
        return tuple(dict.fromkeys(text for text in found if len(text) >= 3))


def _libpq_passwords(url: str) -> list[str]:
    """The password(s) the database driver reads in ``url``, which can differ from urllib's reading (never raises)."""
    try:
        from psycopg.conninfo import conninfo_to_dict

        password = conninfo_to_dict(url).get("password")
    except Exception:  # noqa: BLE001 - the driver may be absent or refuse the URL; either way there is nothing to add
        return []
    return [password] if isinstance(password, str) else []


_URL_PASSWORD = re.compile(r"(\w+://[^\s:/@]*):[^\s@]*@")
_PASSWORD_ASSIGNMENT = re.compile(r"(password\s*=\s*)\S+", re.IGNORECASE)


def redact(text: str, settings: RemoteBackupSettings | None = None) -> str:
    """``text`` with the settings' secrets, and any user:password@ or password= text, replaced by ***."""
    if settings is not None:
        for secret in sorted(settings.secret_strings(), key=len, reverse=True):
            text = text.replace(secret, "***")
    text = _URL_PASSWORD.sub(r"\1:***@", text)
    return _PASSWORD_ASSIGNMENT.sub(r"\1***", text)


def generate_key() -> str:
    """A new encryption key: 32 random bytes as base64url text without padding."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode("ascii").rstrip("=")


def parse_key(value: object) -> bytes:
    """The 32 key bytes in ``value`` (base64url text). Raises RemoteBackupConfigError, never quoting the value."""
    if not isinstance(value, str):
        raise RemoteBackupConfigError("encryption_key must be text in quotes.")
    text = value.strip()
    if not text:
        raise RemoteBackupConfigError("encryption_key is empty.")
    make_one = f"Make one with: python scripts/remote_backup_tool.py new-key (see {SETUP_DOC})."
    if not _BASE64URL.fullmatch(text):
        raise RemoteBackupConfigError(f"encryption_key isn't base64url text (letters, digits, - and _). {make_one}")
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise RemoteBackupConfigError(f"encryption_key isn't valid base64url text. {make_one}") from None
    if len(raw) != KEY_BYTES:
        raise RemoteBackupConfigError(
            f"encryption_key must decode to exactly {KEY_BYTES} bytes, but it decodes to {len(raw)}. {make_one}"
        )
    if len(set(raw)) < MIN_DISTINCT_KEY_BYTES:
        # Random bytes have about 30 different values; a typed phrase or a repeated character has far fewer.
        raise RemoteBackupConfigError(f"encryption_key doesn't look random enough to be a key. {make_one}")
    return raw


def _sslmode_refusal() -> RemoteBackupConfigError:
    return RemoteBackupConfigError(
        "database_url's sslmode must be require, verify-ca or verify-full: TLS is mandatory, and anything "
        "weaker than 'require' is refused. Use sslmode=verify-full (or leave sslmode out)."
    )


def _check_against_libpq(conninfo: str, parts: urllib.parse.SplitResult) -> None:
    """Refuse a URL the database driver reads differently from this module.

    The TLS rule is checked on this module's reading of the URL, but libpq
    parses it too, and the two disagree about a few inputs (an unencoded "@" in
    the password, a "#"): a weaker sslmode or another host could then ride
    through. So the driver's own reading must give an allowed sslmode and the
    same host, user, password and port. Without the driver nothing can connect
    either, so nothing is checked.
    """
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        return
    try:
        read = conninfo_to_dict(conninfo)
    except Exception:  # noqa: BLE001 - the driver's message could quote the password, so it is dropped
        raise RemoteBackupConfigError("database_url isn't a valid URL.") from None
    if read.get("sslmode") not in _TLS_MODES:
        raise _sslmode_refusal()
    expected = {
        "host": (parts.hostname or "").lower(),
        "user": urllib.parse.unquote(parts.username or ""),
        "password": urllib.parse.unquote(parts.password or ""),
        "port": str(parts.port) if parts.port else "",
    }
    seen = {
        "host": str(read.get("host") or "").lower(),
        "user": str(read.get("user") or ""),
        "password": str(read.get("password") or ""),
        "port": str(read.get("port") or ""),
    }
    if seen != expected:
        raise RemoteBackupConfigError(
            "database_url is read differently by the database driver: write each special character in the user "
            "name and password as %XX (for example @ as %40, : as %3A, / as %2F, # as %23), and give the user and "
            "password only before the host, not as ?user= or ?password=."
        )


def secure_conninfo(url: str) -> str:
    """``url`` with TLS enforced: sslmode weaker than "require" is refused, and a missing one becomes verify-full.

    A verify-full URL with no sslrootcert gets ``sslrootcert=system`` (the
    operating system's trusted roots), since libpq would otherwise look for a
    root.crt file in the user's home folder, which the hosted app doesn't
    have. Raises RemoteBackupConfigError, without quoting the URL.
    """
    text = url.strip()
    if "#" in text:
        raise RemoteBackupConfigError(
            "database_url must not contain '#' (a fragment is read differently by the database driver); write "
            "a '#' inside the password as %23."
        )
    try:
        parts = urllib.parse.urlsplit(text)
        host = parts.hostname
        _ = parts.port  # raises ValueError on a port that isn't a number
    except ValueError:
        raise RemoteBackupConfigError("database_url isn't a valid URL.") from None
    if parts.scheme not in ("postgresql", "postgres"):
        raise RemoteBackupConfigError("database_url must be a URL that starts with postgresql://")
    if not host or host.startswith("/"):
        raise RemoteBackupConfigError("database_url needs the database server's host name in the URL.")
    pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    keys = {key for key, _value in pairs}
    if keys & {"host", "hostaddr"}:
        raise RemoteBackupConfigError("database_url must name the server in the URL itself, not in host= or hostaddr=.")
    modes = [value for key, value in pairs if key == "sslmode"]
    for mode in modes:
        if mode not in _TLS_MODES:
            raise _sslmode_refusal()
    extra = []
    mode = modes[-1] if modes else "verify-full"
    if not modes:
        extra.append("sslmode=verify-full")
    if mode == "verify-ca" and "sslrootcert" not in keys:
        # libpq refuses sslrootcert=system with verify-ca, so there is no default root to offer.
        raise RemoteBackupConfigError(
            "database_url's sslmode=verify-ca needs sslrootcert=<a root certificate file>; use sslmode=verify-full "
            "to check the server against the system's trusted roots."
        )
    if mode == "verify-full" and "sslrootcert" not in keys:
        extra.append("sslrootcert=system")
    query = "&".join(part for part in (parts.query, *extra) if part)
    conninfo = urllib.parse.urlunsplit(parts._replace(query=query))
    _check_against_libpq(conninfo, parts)
    return conninfo


def _is_set(value: object) -> bool:
    return value is not None and value != ""


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise RemoteBackupConfigError(f"{name} must be true or false.")


def _as_keep(value: object) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not MIN_KEEP <= value <= MAX_KEEP:
        raise RemoteBackupConfigError(f"keep must be a whole number from {MIN_KEEP} to {MAX_KEEP}.")
    return value


def load_settings(
    secrets_table: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None
) -> RemoteBackupSettings | None:
    """The cloud backup settings, or None when it isn't set up. Raises RemoteBackupConfigError for wrong settings.

    ``secrets_table`` is the ``[remote_backup]`` secrets table, and each
    setting it lacks falls back to its PICKLEBALL_REMOTE_BACKUP_* variable in
    ``environ`` (the process environment unless given). Nothing is set up when
    neither holds anything. The error messages name settings, never values.
    """
    environ = os.environ if environ is None else environ
    if secrets_table is not None and not isinstance(secrets_table, Mapping):
        raise RemoteBackupConfigError(f"[{SECRETS_SECTION}] must be a table of settings.")
    table = dict(secrets_table or {})
    unknown = sorted(str(name) for name in table if name not in SETTING_NAMES)
    if unknown:
        raise RemoteBackupConfigError(
            f"Unknown cloud backup setting(s): {', '.join(unknown)}. The settings are {', '.join(SETTING_NAMES)}."
        )
    env_names = dict(zip(SETTING_NAMES, ENV_VARS))
    values = {name: table[name] if _is_set(table.get(name)) else environ.get(env_names[name]) for name in SETTING_NAMES}
    if not any(_is_set(value) for value in values.values()):
        return None
    missing = [name for name in ("database_url", "encryption_key") if not _is_set(values[name])]
    if missing:
        raise RemoteBackupConfigError(f"Cloud backup is only partly set up: {' and '.join(missing)} missing.")
    url = values["database_url"]
    if not isinstance(url, str):
        raise RemoteBackupConfigError("database_url must be text in quotes.")
    secure_conninfo(url)  # refuses an unusable URL or a weak sslmode now, not at the first backup
    key = parse_key(values["encryption_key"])
    namespace = values["namespace"] if _is_set(values["namespace"]) else DEFAULT_NAMESPACE
    if not isinstance(namespace, str) or not _NAMESPACE_PATTERN.fullmatch(namespace):
        raise RemoteBackupConfigError(
            "namespace must be 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or digit."
        )
    if namespace in RESERVED_NAMESPACES:
        raise RemoteBackupConfigError(
            f"namespace '{namespace}' is reserved for the helper tool, which empties it: choose another name."
        )
    return RemoteBackupSettings(
        database_url=url.strip(),
        encryption_key=key,
        namespace=namespace,
        keep=_as_keep(values["keep"]) if _is_set(values["keep"]) else DEFAULT_KEEP,
        auto_restore=_as_bool(values["auto_restore"], "auto_restore") if _is_set(values["auto_restore"]) else True,
    )


# --- Encryption ----------------------------------------------------------------------------------


def _aesgcm(key: bytes):
    if len(key) != KEY_BYTES:
        raise RemoteBackupConfigError(f"The encryption key must be exactly {KEY_BYTES} bytes.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RemoteBackupError("The 'cryptography' package isn't installed, so cloud backup can't run.") from None
    return AESGCM(key)


def _associated_data(version: int, namespace: str) -> bytes:
    """Authenticated with every blob, so a ciphertext can't be replayed under another namespace or format."""
    return b"pickleball-scheduler remote backup\x00" + bytes([version]) + namespace.encode("utf-8")


def encrypt_snapshot(plaintext: bytes, key: bytes, namespace: str) -> bytes:
    """The blob to store: format version, a fresh random nonce, then the AES-256-GCM ciphertext and tag."""
    cipher = _aesgcm(key)
    nonce = os.urandom(NONCE_BYTES)
    return (
        bytes([FORMAT_VERSION]) + nonce + cipher.encrypt(nonce, plaintext, _associated_data(FORMAT_VERSION, namespace))
    )


def decrypt_snapshot(blob: bytes, key: bytes, namespace: str) -> bytes:
    """The plaintext of ``blob``. Raises RemoteBackupDecryptError for the wrong key or namespace, or a damaged blob."""
    cipher = _aesgcm(key)
    damaged = RemoteBackupDecryptError(
        "The cloud copy couldn't be decrypted: the encryption key is wrong, or the copy is damaged."
    )
    if len(blob) < 1 + NONCE_BYTES + TAG_BYTES:
        raise damaged
    version = blob[0]
    if FORMAT_VERSION < version <= FORMAT_VERSION + 4:  # a version a later release of the app could plausibly write
        raise RemoteBackupDecryptError("The cloud copy was made by a newer version of the app. Update the app.")
    if version != FORMAT_VERSION:
        raise damaged
    try:
        from cryptography.exceptions import InvalidTag
    except ImportError:
        raise RemoteBackupError("The 'cryptography' package isn't installed, so cloud backup can't run.") from None
    try:
        return cipher.decrypt(blob[1 : 1 + NONCE_BYTES], blob[1 + NONCE_BYTES :], _associated_data(version, namespace))
    except InvalidTag:
        raise damaged from None


# --- Stores --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotInfo:
    """One stored copy, without its contents."""

    id: int
    created_at: datetime
    size_bytes: int
    namespace: str
    format_version: int = FORMAT_VERSION


@dataclass(frozen=True)
class StoredSnapshot:
    info: SnapshotInfo
    payload: bytes


@dataclass
class StoreDiagnosis:
    """What a store check found: lines that passed, and problems."""

    passed: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


class RemoteStore(Protocol):
    """Where snapshots are kept. Every method raises RemoteBackupError, with a message that holds no secret."""

    def put(self, namespace: str, payload: bytes, format_version: int, keep: int) -> SnapshotInfo:
        """Store ``payload``, then drop the copies of this namespace beyond the newest ``keep``."""

    def latest(self, namespace: str) -> StoredSnapshot | None:
        """The newest copy of this namespace, or None."""

    def latest_info(self, namespace: str) -> SnapshotInfo | None:
        """What ``latest`` would return, without the contents."""

    def delete_namespace(self, namespace: str) -> int:
        """Remove every copy of this namespace; return how many."""

    def diagnose(self) -> StoreDiagnosis:
        """Check the connection, the table and the role's privileges, changing nothing."""


def check_payload_size(payload: bytes) -> None:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RemoteBackupError(
            f"The backup is {len(payload) / 1e6:.1f} MB, over the {MAX_PAYLOAD_BYTES // (1024 * 1024)} MB limit for "
            "cloud copies, so it was not sent."
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    """A RemoteStore in memory, for tests. Set ``outage`` to an exception to make every call raise it.

    Like the database, it ignores (and prunes) a row dated in the future or before 2000: see ``plant``.
    """

    def __init__(self, clock: Callable[[], datetime] = _utc_now):
        self._clock = clock
        self._lock = threading.Lock()
        self._rows: list[StoredSnapshot] = []
        self._next_id = 1
        self.outage: Exception | None = None
        self.calls: list[str] = []

    def _enter(self, operation: str) -> None:
        self.calls.append(operation)
        if self.outage is not None:
            raise self.outage

    def _is_sane(self, moment: datetime) -> bool:
        return _EARLIEST_SANE_ROW <= moment <= self._clock()

    def put(self, namespace: str, payload: bytes, format_version: int, keep: int) -> SnapshotInfo:
        self._enter("put")
        check_payload_size(payload)
        with self._lock:
            info = SnapshotInfo(self._next_id, self._clock(), len(payload), namespace, format_version)
            self._next_id += 1
            self._rows.append(StoredSnapshot(info, bytes(payload)))
            mine = [row for row in self._rows if row.info.namespace == namespace]
            sane = [row for row in mine if self._is_sane(row.info.created_at)]
            sane.sort(key=lambda row: (row.info.created_at, row.info.id), reverse=True)
            dropped = {row.info.id for row in mine} - {row.info.id for row in sane[:keep]}
            self._rows = [row for row in self._rows if row.info.id not in dropped]
        return info

    def plant(self, namespace: str, payload: bytes, created_at: datetime, format_version: int = 1) -> SnapshotInfo:
        """Add a row with a made-up time, as someone with write access to the table could (for tests)."""
        with self._lock:
            info = SnapshotInfo(self._next_id, created_at, len(payload), namespace, format_version)
            self._next_id += 1
            self._rows.append(StoredSnapshot(info, bytes(payload)))
        return info

    def _newest(self, namespace: str) -> StoredSnapshot | None:
        with self._lock:
            mine = [row for row in self._rows if row.info.namespace == namespace and self._is_sane(row.info.created_at)]
        return max(mine, key=lambda row: (row.info.created_at, row.info.id), default=None)

    def latest(self, namespace: str) -> StoredSnapshot | None:
        self._enter("latest")
        return self._newest(namespace)

    def latest_info(self, namespace: str) -> SnapshotInfo | None:
        self._enter("latest_info")
        newest = self._newest(namespace)
        return newest.info if newest else None

    def delete_namespace(self, namespace: str) -> int:
        self._enter("delete_namespace")
        with self._lock:
            before = len(self._rows)
            self._rows = [row for row in self._rows if row.info.namespace != namespace]
            return before - len(self._rows)

    def diagnose(self) -> StoreDiagnosis:
        self._enter("diagnose")
        return StoreDiagnosis(passed=["In-memory store: nothing to check."])

    def count(self, namespace: str) -> int:
        with self._lock:
            return sum(1 for row in self._rows if row.info.namespace == namespace)


_ROOT_CERTIFICATE_FAILURES = (
    "certificate verify failed",
    "unable to get local issuer certificate",
    "unable to get issuer certificate",
    "self-signed certificate",
    "root certificate",
)


def _is_root_certificate_failure(exc: Exception) -> bool:
    """Whether a connection failed for lack of a root certificate to verify the server against."""
    text = str(exc).lower()
    return isinstance(exc, _psycopg().OperationalError) and any(phrase in text for phrase in _ROOT_CERTIFICATE_FAILURES)


def _certifi_conninfo(database_url: str, conninfo: str) -> str | None:
    """``conninfo`` with certifi's CA bundle in place of the system roots, if the URL didn't name a root certificate."""
    if "sslrootcert=system" not in conninfo or "sslrootcert" in database_url:
        return None
    try:
        import certifi
    except ImportError:
        return None
    bundle = urllib.parse.quote(Path(certifi.where()).as_posix(), safe="/:")
    return conninfo.replace("sslrootcert=system", f"sslrootcert={bundle}")


def _psycopg() -> Any:
    try:
        import psycopg
    except ImportError:
        raise RemoteBackupError(
            "The 'psycopg' package isn't installed, so cloud backup can't run (pip install -r requirements.txt)."
        ) from None
    return psycopg


def _abort_connection(conn: Any) -> None:
    """Best effort: shut down the connection's socket, so a call blocked on it fails now instead of never.

    The file descriptor is only borrowed (``detach`` hands it back untouched),
    and nothing here raises: a connection that is already closed has no socket.
    """
    if conn is None or getattr(conn, "closed", False):
        return
    try:
        borrowed = socket.socket(fileno=conn.pgconn.socket)
    except Exception:  # noqa: BLE001 - not a connection with a socket
        return
    try:
        borrowed.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        borrowed.detach()


class PostgresStore:
    """A RemoteStore in a Postgres table (Neon), through psycopg 3.

    Each operation opens its own short-lived connection (no pool) with a
    connect timeout, runs in one transaction with a statement timeout, and
    closes. Needs no more than SELECT, INSERT and DELETE on the table and
    USAGE on its sequence: it never creates anything (see ``create_table``).
    Every failure becomes a RemoteBackupError whose message is fixed text.

    Every operation also has a hard deadline (``deadline`` seconds for the
    whole of it): it runs on a helper thread that the caller waits for, so a
    server that goes silent after connecting, which no libpq setting bounds,
    still returns control. The stuck call's socket is then shut down so it
    ends, and while ``MAX_STUCK_CALLS`` of them are still running no new call
    is started.

    ``conninfo`` and ``connect`` exist for tests: the CI database has no TLS,
    which ``settings.conninfo()`` would refuse, and a test injects the failures
    a real connection raises. The app passes neither.
    """

    def __init__(
        self,
        settings: RemoteBackupSettings,
        conninfo: str | None = None,
        connect: Callable[..., Any] | None = None,
        deadline: float = OPERATION_DEADLINE_SECONDS,
    ):
        self._settings = settings
        self._conninfo = conninfo if conninfo is not None else settings.conninfo()
        # Only a URL that named no root certificate gets this: see _open_connection.
        self._fallback_conninfo = _certifi_conninfo(settings.database_url, self._conninfo) if conninfo is None else None
        self._connect = connect
        self._deadline = deadline
        self._local = threading.local()
        self._stuck_lock = threading.Lock()
        self._stuck: list[threading.Thread] = []

    def _open(self) -> Any:
        conn = self._open_connection()
        slot = getattr(self._local, "slot", None)
        if slot is not None:
            slot["conn"] = conn  # so the caller can cut the call off if it misses its deadline
        return conn

    def _open_connection(self) -> Any:
        connect = self._connect or _psycopg().connect
        try:
            return connect(self._conninfo, **CONNECT_OPTIONS)
        except Exception as exc:
            fallback = self._fallback_conninfo
            if fallback is None or not _is_root_certificate_failure(exc):
                raise
            # The bundled libpq looks for the system's trusted roots through OpenSSL's default paths, which hold
            # none on every platform (Windows, for one). Verification stays on: the Mozilla roots in certifi
            # (already installed for requests) replace them.
            logger.info("Cloud backup: the system's trusted roots weren't usable, using certifi's bundle")
            conn = connect(fallback, **CONNECT_OPTIONS)
            self._conninfo, self._fallback_conninfo = fallback, None
            return conn

    def _bounded(self, operation: Callable[[], Any]) -> Any:
        """Run ``operation`` on a helper thread and wait for it at most ``deadline`` seconds."""
        with self._stuck_lock:
            self._stuck = [thread for thread in self._stuck if thread.is_alive()]
            if len(self._stuck) >= MAX_STUCK_CALLS:
                raise RemoteBackupError(
                    "The cloud database isn't answering: earlier calls to it are still waiting. Trying again later."
                )
        slot: dict[str, Any] = {}
        outcome: dict[str, Any] = {}
        finished = threading.Event()

        def run() -> None:
            self._local.slot = slot
            try:
                outcome["value"] = operation()
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below, whatever it is
                outcome["error"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=run, name="remote-backup-db", daemon=True)
        worker.start()
        if not finished.wait(self._deadline):
            logger.warning(
                "Cloud backup: a database call didn't finish within %s seconds and was cut off", self._deadline
            )
            _abort_connection(slot.get("conn"))
            with self._stuck_lock:
                self._stuck.append(worker)
            raise RemoteBackupError(f"The cloud database didn't answer within {self._deadline:g} seconds.")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    @contextmanager
    def _transaction(self, creating: bool = False) -> Iterator[Any]:
        try:
            with self._open() as conn:
                # Local to this transaction, so it also works through Neon's pooled (PgBouncer) endpoint.
                conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(STATEMENT_TIMEOUT_MS),))
                yield conn
        except RemoteBackupError:
            raise
        except Exception as exc:
            raise self._translate(exc, creating) from None

    def _translate(self, exc: Exception, creating: bool = False) -> RemoteBackupError:
        """The user's one-line message for a failure, with the details (redacted) in the log."""
        name = type(exc).__name__
        detail = redact(" ".join(str(exc).split()), self._settings)[:300]
        logger.warning("Cloud backup database call failed (%s): %s", name, detail)
        text = detail.lower()
        try:
            psycopg = _psycopg()
        except RemoteBackupError as unavailable:
            return unavailable
        errors = psycopg.errors
        if isinstance(exc, errors.UndefinedTable):
            message = TABLE_MISSING_MESSAGE
        elif isinstance(exc, errors.InsufficientPrivilege):
            if creating:
                message = (
                    "This role isn't allowed to create the backup table: run the setup SQL (see "
                    f"{SETUP_DOC}) as the project's owner role."
                )
            else:
                message = (
                    "The cloud database role isn't allowed to use the backup table: grant it SELECT, INSERT and "
                    f"DELETE on the table and USAGE on its sequence (see {SETUP_DOC})."
                )
        elif isinstance(exc, errors.QueryCanceled):
            message = f"The cloud database took longer than {STATEMENT_TIMEOUT_MS // 1000} seconds to answer."
        elif isinstance(exc, psycopg.OperationalError):
            if "database" in text and "does not exist" in text:
                message = (
                    "The cloud database server has no database with the name at the end of database_url: "
                    "check that name."
                )
            elif "authentication failed" in text or "password" in text or "does not exist" in text:
                message = "The cloud database refused the login: check the user name and password in database_url."
            elif "ssl" in text or "tls" in text or "certificate" in text:
                message = (
                    "Couldn't make a verified secure (TLS) connection to the cloud database; see the TLS notes in "
                    f"{SETUP_DOC}."
                )
            elif "timeout" in text or "timed out" in text:
                message = f"The cloud database didn't answer within {CONNECT_TIMEOUT_SECONDS} seconds."
            else:
                message = "Couldn't reach the cloud database."
        elif isinstance(exc, psycopg.DataError):
            message = (
                "The cloud table holds a value this app can't read (a row written by hand?); delete that row in the "
                f"Neon SQL editor (see {SETUP_DOC})."
            )
        elif isinstance(exc, psycopg.Error):
            message = f"The cloud database returned an error ({name})."
        elif isinstance(exc, UnicodeError):
            message = "The host name in database_url isn't valid: copy it again from the Neon console."
        else:
            message = f"The cloud backup failed unexpectedly ({name})."
        return RemoteBackupError(message)

    def put(self, namespace: str, payload: bytes, format_version: int, keep: int) -> SnapshotInfo:
        check_payload_size(payload)
        return self._bounded(lambda: self._put(namespace, payload, format_version, keep))

    def _put(self, namespace: str, payload: bytes, format_version: int, keep: int) -> SnapshotInfo:
        with self._transaction() as conn:
            row = conn.execute(_INSERT_SQL, (namespace, format_version, len(payload), payload)).fetchone()
            conn.execute(_PRUNE_SQL, (namespace, namespace, keep))
        return SnapshotInfo(int(row[0]), row[1], len(payload), namespace, format_version)

    def latest(self, namespace: str) -> StoredSnapshot | None:
        return self._bounded(lambda: self._latest(namespace))

    def _latest(self, namespace: str) -> StoredSnapshot | None:
        with self._transaction() as conn:
            row = conn.execute(_LATEST_SQL, (MAX_PAYLOAD_BYTES, namespace)).fetchone()
        if row is None:
            return None
        if row[5] is None:
            raise RemoteBackupError("The newest cloud copy is larger than this app accepts, so it was not downloaded.")
        return StoredSnapshot(SnapshotInfo(int(row[0]), row[1], int(row[4]), namespace, int(row[2])), bytes(row[5]))

    def latest_info(self, namespace: str) -> SnapshotInfo | None:
        return self._bounded(lambda: self._latest_info(namespace))

    def _latest_info(self, namespace: str) -> SnapshotInfo | None:
        with self._transaction() as conn:
            row = conn.execute(_LATEST_INFO_SQL, (namespace,)).fetchone()
        return None if row is None else SnapshotInfo(int(row[0]), row[1], int(row[3]), namespace, int(row[2]))

    def delete_namespace(self, namespace: str) -> int:
        return self._bounded(lambda: self._delete_namespace(namespace))

    def _delete_namespace(self, namespace: str) -> int:
        with self._transaction() as conn:
            return int(conn.execute(_DELETE_NAMESPACE_SQL, (namespace,)).rowcount)

    def create_table(self) -> None:
        """Create the table and index, for a role that happens to be allowed to. The app never calls this."""
        self._bounded(self._create_table)

    def _create_table(self) -> None:
        with self._transaction(creating=True) as conn:
            for statement in SETUP_SQL:
                conn.execute(statement)

    def diagnose(self) -> StoreDiagnosis:
        """Connect, report the TLS in use, and try SELECT, INSERT and DELETE on the table, rolled back."""
        return self._bounded(self._diagnose)

    def _diagnose(self) -> StoreDiagnosis:
        result = StoreDiagnosis()
        psycopg = _psycopg()
        with self._transaction() as conn:
            if conn.pgconn.ssl_in_use:
                result.passed.append("The connection is encrypted (TLS).")
            else:
                result.problems.append("The connection is NOT encrypted (TLS is not in use).")
            for label, statement, params in (
                ("SELECT", _SELECT_PROBE_SQL, ("check-privileges",)),
                ("INSERT", _INSERT_SQL, ("check-privileges", 0, 0, b"")),
                ("DELETE", _DELETE_NAMESPACE_SQL, ("check-privileges",)),
            ):
                try:
                    with conn.transaction():  # a savepoint, so one refusal doesn't end the others
                        conn.execute(statement, params)
                except psycopg.errors.UndefinedTable:
                    result.problems.append(TABLE_MISSING_MESSAGE)
                    break
                except psycopg.errors.InsufficientPrivilege:
                    result.problems.append(f"The role is not allowed to {label} on {TABLE_NAME} (or use its sequence).")
                else:
                    result.passed.append(f"The role can {label} on {TABLE_NAME}.")
            broad = self._broad_privileges(conn, psycopg)
            if broad:
                result.problems.append(
                    f"This role has more privileges than needed ({', '.join(broad)}), so a leaked connection string "
                    f"could do more than touch the backup table. See {SETUP_DOC} for a role limited to it."
                )
            conn.rollback()  # nothing the probes did is kept
        return result

    @staticmethod
    def _broad_privileges(conn: Any, psycopg: Any) -> list[str]:
        """What the connected role can do beyond the backup table (superuser-like attributes, broad built-in roles)."""
        broad = []
        row = conn.execute(
            "SELECT rolsuper, rolcreaterole, rolcreatedb, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if row and any(row):
            broad.append("superuser, CREATEROLE, CREATEDB or BYPASSRLS")
        for built_in in ("neon_superuser", "pg_write_all_data", "pg_read_all_data"):
            try:
                with conn.transaction():
                    if conn.execute("SELECT pg_has_role(current_user, %s, 'member')", (built_in,)).fetchone()[0]:
                        broad.append(f"member of {built_in}")
            except psycopg.Error:
                continue  # that role doesn't exist on this server
        return broad


def make_store(settings: RemoteBackupSettings) -> RemoteStore:
    """The store for the app: PostgresStore. (A test replaces this to avoid any database.)"""
    return PostgresStore(settings)


# --- What is on disk -----------------------------------------------------------------------------


def local_fingerprint(db_path: Path, config_path: Path, skills_path: Path, elo_path: Path) -> tuple:
    """(mtime_ns, size) of the history database, its -wal file and the three JSON files; None for one that's missing.

    A -wal file of zero bytes holds nothing (SQLite makes one for every open
    connection and deletes it at the last close), so it counts as missing.
    """
    parts: list[tuple[int, int] | None] = []
    for path in (db_path, db_path.with_name(db_path.name + "-wal"), config_path, skills_path, elo_path):
        try:
            stat = path.stat()
        except OSError:
            parts.append(None)
            continue
        parts.append(None if path.name.endswith("-wal") and stat.st_size == 0 else (stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def _holds_ratings(path: Path) -> bool:
    """Whether a ratings file (skills or ELO) holds any rating. One that can't be read counts as data."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        return bool(json.loads(raw.decode("utf-8-sig")))
    except ValueError:
        return True


def local_is_fresh(db_path: Path, config_path: Path, rating_paths: Iterable[Path] = ()) -> bool | None:
    """Whether this app has no data: the history database holds no schedule row (deleted ones count),
    app_config.json doesn't exist, and no ratings file in ``rating_paths`` holds a rating. None when that can't
    be told (an unreadable database): treated as not fresh.
    """
    try:
        if config_path.exists() or any(_holds_ratings(path) for path in rating_paths):
            return False
        if not db_path.exists():
            return True
        with closing(sqlite3.connect(str(db_path), timeout=5)) as conn:
            return conn.execute("SELECT EXISTS (SELECT 1 FROM schedule_history)").fetchone()[0] == 0
    except sqlite3.OperationalError as exc:
        return True if "no such table" in str(exc) else None
    except (sqlite3.Error, OSError):
        return None


def describe_time(moment: datetime, tz: tzinfo | None = None) -> str:
    """``moment`` as "2026-10-14 15:04 UTC" (or in ``tz``, with its abbreviation).

    A time too far from now to show in ``tz`` (year 9999 in a zone ahead of UTC) is shown in UTC, and one
    that can't be shown at all as its ISO text: a row written by hand must not break the page.
    """
    for zone in (tz or timezone.utc, timezone.utc):
        try:
            return moment.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z").strip()
        except (OverflowError, ValueError, OSError):
            continue
    return moment.isoformat()


def _parse_made_at(made_at: str | None) -> datetime | None:
    """The time in a backup's manifest (``2026-10-14T15:04:10``, local time of the machine that made it)."""
    if not made_at:
        return None
    try:
        return datetime.fromisoformat(made_at).astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def restore_message(info: SnapshotInfo, tz: tzinfo | None = None, made_at: str | None = None) -> str:
    """ "Restored your data from the cloud backup made <time>." ``made_at`` is the backup's own made-time (its
    manifest, which the encryption protects). The row's time is not: someone with write access to the table can
    put an old copy in as a new row, so when the two differ by more than an hour the message says so.
    """
    stored = describe_time(info.created_at, tz)
    made = _parse_made_at(made_at)
    if made is None or made_at is None:
        return f"Restored your data from the cloud backup made {stored}."
    message = f"Restored your data from the cloud backup made {made_at[:16].replace('T', ' ')}."
    try:
        apart = abs((info.created_at - made).total_seconds())
    except (OverflowError, TypeError):
        apart = 0.0
    if apart > MADE_AT_SLACK_SECONDS:
        message += (
            f" The cloud stored that copy on {stored}, which doesn't match: if you didn't expect that, someone with "
            "access to the database may have put an older copy in place of a newer one."
        )
    return message


# --- The sync policy -----------------------------------------------------------------------------

PAUSE_CLOUD_NEWER = "cloud_newer"
PAUSE_NOT_RESTORED = "not_restored"
PAUSE_RESTORE_FAILED = "restore_failed"
PAUSE_KEY_MISMATCH = "key_mismatch"


@dataclass(frozen=True)
class PushResult:
    ok: bool
    message: str
    snapshot: SnapshotInfo | None = None
    skipped: bool = False


@dataclass(frozen=True)
class RestoreResult:
    snapshot: SnapshotInfo
    kept_copy: Path
    lines: tuple[str, ...]
    made_at: str | None = None  # when the backup itself says it was made (see restore_message)


@dataclass(frozen=True)
class RemoteBackupStatus:
    """What the UI shows about the cloud backup."""

    configured: bool = True
    last_push_at: datetime | None = None
    last_error: str | None = None
    next_retry_at: datetime | None = None
    pending: bool = False
    in_flight: bool = False
    paused_reason: str | None = None
    cloud_copy: SnapshotInfo | None = None  # the copy a pause is about
    checking: bool = False  # the first comparison with the cloud copy hasn't finished
    restored_from: SnapshotInfo | None = None
    restored_kept: Path | None = None  # the copy of the data that restore replaced
    restored_made_at: str | None = None
    restore_generation: int = 0

    def pause_text(self, tz: tzinfo | None = None) -> str | None:
        """Why automatic backups are paused, if they are."""
        if not (self.paused_reason and self.cloud_copy):
            return None
        when = describe_time(self.cloud_copy.created_at, tz)
        if self.paused_reason == PAUSE_CLOUD_NEWER:
            return (
                f"The cloud has a copy made {when} that is newer than this app's data, so automatic backups are "
                "paused: restore it, or back up now to keep this app's data instead."
            )
        if self.paused_reason == PAUSE_NOT_RESTORED:
            return (
                f"The cloud has a copy made {when} that this app's data doesn't include (the app started with no "
                "data and didn't restore it), so automatic backups are paused so it isn't replaced: restore it, or "
                "back up now to replace it."
            )
        if self.paused_reason == PAUSE_KEY_MISMATCH:
            return (
                f"The newest cloud copy, made {when}, can't be opened with this app's encryption key (a different "
                "key, or a damaged copy), so automatic backups are paused so it isn't replaced: put the right key "
                "back, or back up now to start new copies with this key."
            )
        return (
            f"The cloud copy made {when} couldn't be restored, so automatic backups are paused so it isn't "
            "replaced: fix the problem below, or back up now to replace it."
        )

    def summary(self, tz: tzinfo | None = None) -> str:
        """One or two sentences for the Configuration tab."""
        if not self.configured:
            return "Cloud backup is off."
        parts = []
        if self.in_flight:
            parts.append("Backing up to the cloud now.")
        elif self.last_push_at:
            parts.append(f"Last backed up to the cloud: {describe_time(self.last_push_at, tz)}.")
        else:
            parts.append("Nothing has been backed up to the cloud yet by this session.")
        pause = self.pause_text(tz)
        if pause:
            parts.append(pause)
        elif self.checking and not self.last_push_at:
            parts.append("Checking the cloud copy before the first automatic backup.")
        elif self.pending and not self.in_flight:
            parts.append("A recent change is waiting to be backed up.")
        if self.last_error:
            retry = f" Trying again around {describe_time(self.next_retry_at, tz)}." if self.next_retry_at else ""
            parts.append(f"Problem: {self.last_error}{retry}")
        return " ".join(parts)


class RemoteBackupSync:
    """Keeps the cloud copy current and restores it on a fresh start. One per process (see get_process_sync).

    - ``note_run`` (once per script run) compares a cheap fingerprint of the
      local files with the last one sent; a change asks for a push, which runs
      on one background daemon thread, never in the caller's thread. Pushes are
      at least ``min_interval`` seconds apart, and a change made in that
      cooldown is still sent after it; after a failure the wait doubles up to 15
      minutes. One push runs at a time, from a SQLite backup-API snapshot.
    - ``ensure_started`` restores the newest cloud copy once per process when
      the app has no data (see local_is_fresh). It never restores over data;
      with data it compares with the cloud copy in the background instead, and
      automatic pushes stay paused while the cloud holds a copy this app's data
      doesn't include (a newer one, one it couldn't restore, or one this app's
      key doesn't open), so a push can't bury it. "Back up now" and "Restore"
      are the user's choices. The cloud work runs on the background thread: the
      first page run waits for it only a few seconds, and no lock is held while
      the cloud is being asked.
    - Nothing here raises into the caller for a cloud problem: failures become
      ``status().last_error``.

    ``clock`` (seconds, monotonic) and ``wall_clock`` are injectable for tests;
    ``run_worker=False`` leaves the background thread off so a test calls
    ``run_due`` itself.
    """

    def __init__(
        self,
        settings: RemoteBackupSettings,
        store: RemoteStore,
        history_manager,
        data_paths: Mapping[str, Path],
        config_manager,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = _utc_now,
        min_interval: float = PUSH_MIN_INTERVAL_SECONDS,
        backoff_base: float = BACKOFF_BASE_SECONDS,
        backoff_cap: float = BACKOFF_CAP_SECONDS,
        manual_wait: float = MANUAL_WAIT_SECONDS,
        startup_wait: float = STARTUP_WAIT_SECONDS,
        run_worker: bool = True,
    ):
        self.settings = settings
        self._store = store
        self._history = history_manager
        self._config = config_manager
        self._paths = {name: Path(path) for name, path in data_paths.items()}
        self._clock = clock
        self._wall_clock = wall_clock
        self._min_interval = min_interval
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._manual_wait = manual_wait
        self._startup_wait = startup_wait
        self._run_worker = run_worker

        self._lock = threading.Lock()  # guards the fields below
        self._wake = threading.Condition(self._lock)
        self._push_lock = threading.Lock()  # one push or restore at a time
        self._worker: threading.Thread | None = None
        self._closed = False

        self._settled: tuple | None = None  # the fingerprint the cloud copy (or "nothing to send") matches
        self._pending = False
        self._in_flight = False
        self._next_attempt = 0.0
        self._failures = 0
        self._last_error: str | None = None
        self._last_push: SnapshotInfo | None = None
        self._startup_done = False
        self._startup_running = False  # the startup work has been handed out and hasn't finished
        self._startup_requested = False  # ... and the worker hasn't picked it up yet
        self._startup_event: threading.Event | None = None  # set when the running startup work ends
        self._startup_wait_now = 0.0  # how long a session arriving during it waits
        self._startup_attempts = 0
        self._startup_next = 0.0
        self._startup_failures = 0
        self._compare_pending = False
        self._compare_next = 0.0
        self._checked = False  # the cloud copy has been compared with this app's data
        self._began_empty = False  # the app started with no data, which no cloud copy has been restored into
        self._paused: str | None = None
        self._cloud_copy: SnapshotInfo | None = None
        self._restored_from: SnapshotInfo | None = None
        self._restored_kept: Path | None = None
        self._restored_made_at: str | None = None
        self._restore_generation = 0

    # --- local state ---

    def _db_path(self) -> Path:
        return Path(self._history.db_path)

    def fingerprint(self) -> tuple:
        return local_fingerprint(
            self._db_path(),
            self._paths[_backup.CONFIG_NAME],
            self._paths[_backup.SKILLS_NAME],
            self._paths[_backup.ELO_NAME],
        )

    def _is_fresh(self) -> bool | None:
        return local_is_fresh(
            self._db_path(),
            self._paths[_backup.CONFIG_NAME],
            (self._paths[_backup.SKILLS_NAME], self._paths[_backup.ELO_NAME]),
        )

    def _local_modified(self) -> datetime | None:
        """When the newest of the local files was last changed."""
        times = []
        for path in (self._db_path(), self._db_path().with_name(self._db_path().name + "-wal"), *self._paths.values()):
            try:
                times.append(path.stat().st_mtime)
            except OSError:
                continue
        return datetime.fromtimestamp(max(times), timezone.utc) if times else None

    # --- status ---

    def status(self) -> RemoteBackupStatus:
        with self._lock:
            retry = None
            if self._last_error:
                if not self._startup_done:
                    due = self._startup_next
                elif self._compare_pending:
                    due = self._compare_next
                elif self._pending:
                    due = self._next_attempt
                else:
                    due = None
                if due is not None:
                    retry = self._wall_clock() + timedelta(seconds=max(0.0, due - self._clock()))
            return RemoteBackupStatus(
                last_push_at=self._last_push.created_at if self._last_push else None,
                last_error=self._last_error,
                next_retry_at=retry,
                pending=self._pending,
                in_flight=self._in_flight,
                paused_reason=self._paused,
                cloud_copy=self._cloud_copy,
                checking=not self._checked,
                restored_from=self._restored_from,
                restored_kept=self._restored_kept,
                restored_made_at=self._restored_made_at,
                restore_generation=self._restore_generation,
            )

    @property
    def restore_generation(self) -> int:
        with self._lock:
            return self._restore_generation

    # --- worker ---

    def _backoff(self, failures: int) -> float:
        return min(self._backoff_cap, self._backoff_base * 2 ** max(0, failures - 1))

    def _next_job_locked(self) -> tuple[str | None, float]:
        now = self._clock()
        if self._startup_requested:
            return "startup", 0.0
        if self._compare_pending:
            return "compare", max(0.0, self._compare_next - now)
        if self._pending and self._checked and self._paused is None:
            return "push", max(0.0, self._next_attempt - now)
        return None, 0.0

    def _start_worker_locked(self) -> None:
        if not self._run_worker or self._closed or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._worker_main, name="remote-backup", daemon=True)
        self._worker.start()

    def _worker_main(self) -> None:
        idle_since = self._clock()
        try:
            while True:
                with self._wake:
                    if self._closed:
                        self._worker = None
                        return
                    job, delay = self._next_job_locked()
                    if job is None:
                        if self._clock() - idle_since >= WORKER_IDLE_EXIT_SECONDS:
                            # Cleared in this same locked step, so a change requested right after it starts a
                            # new worker instead of finding one that is about to be gone.
                            self._worker = None
                            return
                        self._wake.wait(timeout=WORKER_POLL_SECONDS)
                        continue
                    if delay > 0:
                        self._wake.wait(timeout=delay)
                        continue
                idle_since = self._clock()
                self._run_job(job)
        finally:
            with self._wake:
                if self._worker is threading.current_thread():
                    self._worker = None

    def _run_job(self, job: str) -> None:
        try:
            if job == "startup":
                self._run_startup()
            elif job == "compare":
                self._compare_with_cloud()
            else:
                self._push(manual=False)
        except Exception as exc:  # the jobs catch their own failures; this keeps the thread alive if one slips through
            logger.warning("Cloud backup background job failed (%s)", type(exc).__name__)
            self._record_failure(f"The cloud backup failed unexpectedly ({type(exc).__name__}).")

    def run_due(self) -> str | None:
        """Run the background job that is due now, if any: returns "startup", "compare", "push" or None."""
        with self._lock:
            job, delay = self._next_job_locked()
        if job is None or delay > 0:
            return None
        self._run_job(job)
        return job

    def close(self) -> None:
        with self._wake:
            self._closed = True
            event = self._startup_event
            self._wake.notify_all()
        if event is not None:
            event.set()  # a session waiting for the startup work stops waiting

    # --- the per-run hooks ---

    def note_run(self) -> None:
        """Called once per script run: ask for a push if the local files changed since the last one. Never raises."""
        try:
            fingerprint = self.fingerprint()
            changed = fingerprint != self._settled
            if changed and self._is_fresh() is True:
                with self._lock:  # an app with no data has nothing to send (and must not send its emptiness)
                    self._settled = fingerprint
                changed = False
            with self._wake:
                if changed:
                    self._pending = True
                if self._pending or self._compare_pending:
                    self._start_worker_locked()
                    self._wake.notify_all()
        except Exception as exc:
            logger.warning("Cloud backup could not check the local files (%s)", type(exc).__name__)

    def ensure_started(self) -> None:
        """Once per process: restore a fresh start from the cloud, or schedule the comparison with its copy.

        The first call hands the work to the background thread and waits for it
        at most ``startup_wait`` seconds, so a slow or silent cloud can't hold
        the page: the app then starts as it is, and a restore that finishes
        later is adopted by the next page run (see restore_generation). A
        session arriving meanwhile waits on the same work, never on a lock, so
        it never sees a half-restored app. After a failure to reach the cloud,
        a later call retries once the backoff has passed, without waiting for
        the retry. Never raises.
        """
        owner = False
        try:
            with self._lock:
                if self._startup_done or self._closed:
                    return
                if not self._startup_running:
                    if self._clock() < self._startup_next:
                        return
                    self._startup_running = True
                    self._startup_event = threading.Event()
                    self._startup_wait_now = self._startup_wait if self._startup_attempts == 0 else 0.0
                    self._startup_attempts += 1
                    owner = True
                event, wait = self._startup_event, self._startup_wait_now
            if owner:
                if self._run_worker:
                    with self._wake:
                        self._startup_requested = True
                        self._start_worker_locked()
                        self._wake.notify_all()
                else:
                    self._run_startup()
            if event is not None and wait > 0:
                event.wait(wait)
        except Exception as exc:
            if owner:
                self._finish_startup()
            self._startup_failed(self._failure_text(exc))

    def _run_startup(self) -> None:
        """The startup work, on the background thread (inline when there is none). No lock is held across the cloud."""
        with self._lock:
            self._startup_requested = False
        try:
            self._startup()
        except Exception as exc:
            self._startup_failed(self._failure_text(exc))
        finally:
            self._finish_startup()

    def _finish_startup(self) -> None:
        with self._wake:
            self._startup_running = False
            self._startup_requested = False
            event = self._startup_event
            self._wake.notify_all()
        if event is not None:
            event.set()

    def _failure_text(self, exc: Exception) -> str:
        """The one-line message for a failure: a RemoteBackupError's own, else a generic one (details to the log)."""
        if isinstance(exc, (RemoteBackupError, _backup.BackupError)):
            return str(exc)
        logger.warning("Cloud backup call failed (%s): %s", type(exc).__name__, redact(str(exc), self.settings))
        return f"The cloud backup failed unexpectedly ({type(exc).__name__})."

    def _startup_failed(self, message: str) -> None:
        with self._lock:
            self._startup_failures += 1
            self._startup_next = self._clock() + self._backoff(self._startup_failures)
            self._last_error = message

    def _startup(self) -> None:
        fresh = self._is_fresh()
        namespace = self.settings.namespace
        if fresh:
            self._began_empty = True  # remembered if the cloud can't be reached now, or the restore is off
        if fresh and self.settings.auto_restore:
            try:
                snapshot = self._store.latest(namespace)
            except Exception as exc:
                self._startup_failed(self._failure_text(exc))
                return
            if snapshot is None:
                self._conclude_check()
                return
            try:
                self._restore(snapshot, only_if_fresh=True)
            except _DataAppeared:
                logger.info("Cloud backup: data appeared while the cloud copy was fetched, so it was not restored")
                self._conclude_check(PAUSE_NOT_RESTORED, snapshot.info)
            except RemoteBackupError as exc:
                logger.warning("Cloud backup: could not restore the newest copy on a fresh start: %s", exc)
                self._conclude_check(PAUSE_RESTORE_FAILED, snapshot.info, error=str(exc))
            return
        with self._wake:
            self._startup_done = True
            self._compare_pending = True
            self._compare_next = 0.0
            self._start_worker_locked()
            self._wake.notify_all()

    def _conclude_check(
        self, paused: str | None = None, cloud_copy: SnapshotInfo | None = None, error: str | None = None
    ) -> None:
        with self._wake:
            self._startup_done = True
            self._compare_pending = False
            self._checked = True
            self._paused = paused
            self._cloud_copy = cloud_copy
            self._startup_failures = 0
            if error is not None:
                self._last_error = error
            elif paused is None:
                self._last_error = None
            self._wake.notify_all()

    def _compare_failed(self, exc: Exception) -> None:
        """The comparison couldn't be made: try again after a growing wait."""
        with self._lock:
            self._startup_failures += 1
            self._compare_next = self._clock() + self._backoff(self._startup_failures)
            self._last_error = self._failure_text(exc)

    def _compare_with_cloud(self) -> None:
        """Look at the newest cloud copy (in the background) and decide whether automatic pushes may start."""
        try:
            info = self._store.latest_info(self.settings.namespace)
        except Exception as exc:
            self._compare_failed(exc)
            return
        if info is None:
            self._conclude_check()
            return
        with self._lock:
            ours = self._last_push
        local_modified = self._local_modified()
        known = ours.created_at if ours else None
        newest_known = max([moment for moment in (local_modified, known) if moment], default=None)
        if info.id == (ours.id if ours else None):
            self._conclude_check()
        elif self._began_empty or self._is_fresh():
            self._conclude_check(PAUSE_NOT_RESTORED, info)
        elif newest_known is None or info.created_at > newest_known + timedelta(seconds=CLOCK_SLACK_SECONDS):
            self._conclude_check(PAUSE_CLOUD_NEWER, info)
        else:
            self._compare_key(info, local_modified)

    def _compare_key(self, info: SnapshotInfo, local_modified: datetime | None) -> None:
        """The cloud holds a copy this process didn't make: pushes may go on only if this app's key opens it.

        With a different (or mistyped) key every push would drop the copies the
        right key opens, as the prune keeps only the newest few, and the page
        would say "backed up" throughout.
        """
        snapshot = None
        try:
            snapshot = self._store.latest(self.settings.namespace)
            if snapshot is not None:
                decrypt_snapshot(snapshot.payload, self.settings.encryption_key, self.settings.namespace)
        except RemoteBackupDecryptError as exc:
            self._conclude_check(PAUSE_KEY_MISMATCH, snapshot.info if snapshot else info, error=str(exc))
            return
        except Exception as exc:
            self._compare_failed(exc)
            return
        if local_modified is not None and local_modified <= info.created_at:
            # Nothing was saved here since that copy was made (a restart over the same files): it holds this data.
            fingerprint = self.fingerprint()
            with self._lock:
                self._settled, self._pending = fingerprint, False
        self._conclude_check()

    # --- pushing ---

    def _record_failure(self, message: str) -> None:
        with self._wake:
            self._failures += 1
            self._last_error = message
            self._next_attempt = self._clock() + self._backoff(self._failures)
            self._in_flight = False
            self._wake.notify_all()

    def push_now(self) -> PushResult:
        """Back up now, whatever the cooldown, backoff or pause (the user asked). Waits for a push already running."""
        return self._push(manual=True)

    def _push(self, *, manual: bool) -> PushResult:
        if manual:
            acquired = self._push_lock.acquire(timeout=self._manual_wait)
        else:
            acquired = self._push_lock.acquire(blocking=False)
        if not acquired:
            with self._lock:
                self._next_attempt = max(self._next_attempt, self._clock() + 1.0)
            return PushResult(False, "A cloud backup is already running.", skipped=True)
        try:
            return self._push_locked(manual)
        finally:
            self._push_lock.release()

    def _push_locked(self, manual: bool) -> PushResult:
        if self._is_fresh():
            fingerprint = self.fingerprint()
            with self._lock:
                self._pending = False
                self._settled = fingerprint
            return PushResult(
                False,
                "There is no data here yet to back up (no schedules and no saved configuration), so nothing was "
                "sent and the cloud copy is unchanged.",
                skipped=True,
            )
        started = self.fingerprint()
        with self._lock:
            self._in_flight = True
        try:
            blob = self._make_blob()
            info = self._store.put(self.settings.namespace, blob, FORMAT_VERSION, self.settings.keep)
        except Exception as exc:
            message = self._failure_text(exc)
            logger.info("Cloud backup push failed: %s", message)
            self._record_failure(message)
            return PushResult(False, message)
        changed_since = self.fingerprint() != started
        with self._wake:
            self._in_flight = False
            self._failures = 0
            self._last_error = None
            self._last_push = info
            self._began_empty = False
            self._settled = started
            self._pending = changed_since
            self._next_attempt = self._clock() + self._min_interval
            self._checked = True
            self._paused = None
            self._cloud_copy = None
            self._compare_pending = False
            self._startup_done = True
            self._wake.notify_all()
        logger.info("Cloud backup: sent copy %s (%d bytes)", info.id, info.size_bytes)
        return PushResult(True, f"Backed up to the cloud ({info.size_bytes / 1000:.0f} KB).", info)

    def _make_blob(self) -> bytes:
        config = self._config.read_saved_config()
        if config is None:
            if self._paths[_backup.CONFIG_NAME].exists():
                raise RemoteBackupError(
                    "app_config.json can't be read, so no cloud backup was made. Saving the configuration on the "
                    "Configuration tab rewrites it."
                )
            config = deepcopy(self._config.default_config)
        raw = _make_backup(self._history, self._paths, export_config_json(config), self._config.default_config)
        blob = encrypt_snapshot(raw, self.settings.encryption_key, self.settings.namespace)
        check_payload_size(blob)
        return blob

    # --- restoring ---

    def restore_latest(self) -> RestoreResult:
        """Restore the newest cloud copy over the local data (the user chose to). Raises RemoteBackupError."""
        snapshot = self._store.latest(self.settings.namespace)
        if snapshot is None:
            raise RemoteBackupError("No cloud backup has been made yet.")
        return self.restore_snapshot(snapshot)

    def restore_snapshot(self, snapshot: StoredSnapshot) -> RestoreResult:
        """Decrypt ``snapshot`` and restore it through the verified full-backup path, which keeps a copy of
        the data it replaces and puts it back if a step fails. Raises RemoteBackupError."""
        return self._restore(snapshot, only_if_fresh=False)

    def _restore(self, snapshot: StoredSnapshot, only_if_fresh: bool) -> RestoreResult:
        """The restore. With ``only_if_fresh`` (the automatic one), data that appeared since the caller looked is
        left alone: that check is made once the restore holds the push lock, just before anything is replaced."""
        plaintext = decrypt_snapshot(snapshot.payload, self.settings.encryption_key, self.settings.namespace)
        if not self._push_lock.acquire(timeout=self._manual_wait):
            raise RemoteBackupError("A cloud backup is running right now; try again in a moment.")
        try:
            if only_if_fresh and self._is_fresh() is not True:
                raise _DataAppeared("This app has data now, so the cloud copy was not restored over it.")
            try:
                backup = _read_backup(plaintext, self._config.default_config)
                kept = _restore_backup(backup, self._history, self._paths, self._config)
            except _backup.BackupError as exc:
                raise RemoteBackupError(f"The cloud copy can't be restored: {exc}") from exc
            fingerprint = self.fingerprint()
            with self._wake:
                self._settled = fingerprint
                self._pending = False
                self._began_empty = False
                self._last_push = snapshot.info
                self._failures = 0
                self._last_error = None
                self._checked = True
                self._paused = None
                self._cloud_copy = None
                self._compare_pending = False
                self._startup_done = True
                self._restored_from = snapshot.info
                self._restored_kept = kept
                self._restored_made_at = backup.created_at
                self._restore_generation += 1
                self._wake.notify_all()
            logger.info("Cloud backup: restored copy %s", snapshot.info.id)
            return RestoreResult(snapshot.info, kept, tuple(backup.describe()), backup.created_at)
        finally:
            self._push_lock.release()


# --- One sync per process ------------------------------------------------------------------------

_PROCESS_LOCK = threading.Lock()
_process_sync: RemoteBackupSync | None = None
_process_key: tuple | None = None


def get_process_sync(
    settings: RemoteBackupSettings, history_manager, data_paths: Mapping[str, Path], config_manager
) -> RemoteBackupSync:
    """The process-wide sync for these settings and data files, made on first use.

    Streamlit Community Cloud runs every session in one process, so one worker,
    one cooldown and one in-flight push serve them all.
    """
    global _process_sync, _process_key
    key = (
        settings,
        str(history_manager.db_path),
        tuple(sorted((name, str(path)) for name, path in data_paths.items())),
    )
    with _PROCESS_LOCK:
        if _process_sync is None or _process_key != key:
            if _process_sync is not None:
                _process_sync.close()
            _process_sync = RemoteBackupSync(
                settings, make_store(settings), history_manager, data_paths, config_manager
            )
            _process_key = key
        return _process_sync


def reset_process_state() -> None:
    """Forget the process-wide sync, stopping its worker (for tests)."""
    global _process_sync, _process_key
    with _PROCESS_LOCK:
        if _process_sync is not None:
            _process_sync.close()
        _process_sync = None
        _process_key = None
