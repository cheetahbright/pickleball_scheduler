#!/usr/bin/env python3
"""Stateful managers extracted from the main Streamlit app."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

_schedule_analytics_core = import_module_with_fallback("utils.schedule_analytics_core")
calculate_fairness_metrics = _schedule_analytics_core.calculate_fairness_metrics
serialize_schedule_for_json = _schedule_analytics_core.serialize_schedule_for_json

logger = logging.getLogger(__name__)


def validate_player_names(player_names: List[str]) -> bool:
    """Validate player names for scheduling.

    Args:
        player_names: List of player names to validate

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValueError: If validation fails with specific error message
    """
    if not player_names:
        raise ValueError("Player names list cannot be empty")

    if len(player_names) < 4:
        raise ValueError("At least 4 players are required for scheduling")

    if len(player_names) != len(set(player_names)):
        raise ValueError("Duplicate player names are not allowed")

    for name in player_names:
        if not name or not name.strip():
            raise ValueError("Player names cannot be empty or just whitespace")

    return True


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


def _parse_constraint_pair(raw_constraint: Any) -> list[str] | None:
    """Normalize a single stored constraint into a two-player pair."""
    if isinstance(raw_constraint, (list, tuple)) and len(raw_constraint) >= 2:
        first, second = raw_constraint[0], raw_constraint[1]
    elif isinstance(raw_constraint, str):
        stripped_value = raw_constraint.strip()
        if not stripped_value:
            return None

        parts: list[str] | None = None
        for pattern, flags in (
            (r"\s+vs\s+", re.IGNORECASE),
            (r"\s*&\s*", 0),
            (r"\s*,\s*", 0),
        ):
            split_parts = re.split(pattern, stripped_value, maxsplit=1, flags=flags)
            if len(split_parts) == 2:
                parts = split_parts
                break

        if parts is None:
            return None

        first, second = parts[0], parts[1]
    else:
        return None

    normalized_first = str(first).strip()
    normalized_second = str(second).strip()
    if not normalized_first or not normalized_second or normalized_first == normalized_second:
        return None

    return [normalized_first, normalized_second]


def normalize_constraint_pairs(raw_constraints: Any) -> list[list[str]]:
    """Normalize legacy or mixed constraint values into a consistent pair list."""
    if isinstance(raw_constraints, str):
        items = raw_constraints.splitlines()
    elif isinstance(raw_constraints, list):
        items = raw_constraints
    elif isinstance(raw_constraints, tuple):
        items = list(raw_constraints)
    else:
        return []

    normalized_pairs: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for raw_constraint in items:
        parsed_pair = _parse_constraint_pair(raw_constraint)
        if parsed_pair is None:
            continue

        pair_key = (parsed_pair[0], parsed_pair[1])
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        normalized_pairs.append(parsed_pair)

    return normalized_pairs


def serialize_constraint_pairs(raw_constraints: Any, separator: str = ", ") -> str:
    """Render stored constraint pairs for text-area editing."""
    return "\n".join(f"{first}{separator}{second}" for first, second in normalize_constraint_pairs(raw_constraints))


def normalize_config_constraints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure constraint settings use the pair structure expected by the scheduler UI."""
    constraints = config.setdefault("constraints", {})
    constraints["do_not_pair"] = normalize_constraint_pairs(constraints.get("do_not_pair", []))
    constraints["do_not_oppose"] = normalize_constraint_pairs(constraints.get("do_not_oppose", []))
    return config


class HistoryManager:
    """Manage schedule history and persistence."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else _resolve_storage_path(
                "PICKLEBALL_HISTORY_DB",
                "schedule_history.db",
            )
        )
        self.ensure_db_exists()

    def ensure_db_exists(self):
        """Create the history database if it does not exist."""
        self.db_path.parent.mkdir(exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedule_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    num_players INTEGER,
                    num_rounds INTEGER,
                    algorithm TEXT,
                    fairness_score REAL,
                    schedule_data TEXT,
                    settings_data TEXT,
                    deleted_at TEXT
                )
            """)

            try:
                cursor.execute("PRAGMA table_info(schedule_history)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                if "deleted_at" not in existing_columns:
                    cursor.execute("ALTER TABLE schedule_history ADD COLUMN deleted_at TEXT")
            except (sqlite3.OperationalError, TypeError):
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS partner_history (
                    week_date TEXT,
                    player1 TEXT,
                    player2 TEXT,
                    PRIMARY KEY (week_date, player1, player2)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opponent_history (
                    week_date TEXT,
                    player1 TEXT,
                    player2 TEXT,
                    PRIMARY KEY (week_date, player1, player2)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schedule_id INTEGER NOT NULL,
                    round_num INTEGER NOT NULL,
                    court INTEGER NOT NULL,
                    team1_players TEXT NOT NULL,
                    team2_players TEXT NOT NULL,
                    team1_score INTEGER,
                    team2_score INTEGER,
                    recorded_at TEXT NOT NULL,
                    UNIQUE (schedule_id, round_num, court)
                )
            """)

            conn.commit()

    def save_schedule(self, schedule: List[Dict], players: List[str], settings: Dict) -> int | None:
        """Save schedule data and weekly relationships. Returns the new row's id, or
        None on failure - callers use the id to attach game scores to this schedule."""
        try:
            metrics = calculate_fairness_metrics(schedule)
            serializable_schedule = serialize_schedule_for_json(schedule)
            stored_settings = dict(settings)
            stored_settings["players"] = list(players)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO schedule_history
                    (timestamp, num_players, num_rounds, algorithm, fairness_score, schedule_data, settings_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().isoformat(),
                        len(players),
                        len(schedule),
                        "genetic",
                        metrics.get("overall_fairness", 0),
                        json.dumps(serializable_schedule),
                        json.dumps(stored_settings),
                    ),
                )
                schedule_id = cursor.lastrowid

                current_week = datetime.now().strftime("%Y-%m-%d")
                self._save_weekly_relationships(cursor, schedule, current_week)

                conn.commit()

            logger.info("History save: rounds=%d players=%d schedule_id=%s", len(schedule), len(players), schedule_id)
            return schedule_id

        except Exception as e:
            logger.exception("History save failed")
            st.error(f"Failed to save history: {e}")
            return None

    def save_game_score(
        self,
        schedule_id: int,
        round_num: int,
        court: int,
        team1_players: List[str],
        team2_players: List[str],
        team1_score: int | None,
        team2_score: int | None,
    ) -> bool:
        """Record or update the score for one game. A schedule_id+round+court is unique -
        re-entering a score for the same game updates it rather than duplicating."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO game_scores
                    (schedule_id, round_num, court, team1_players, team2_players, team1_score, team2_score, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (schedule_id, round_num, court) DO UPDATE SET
                        team1_players = excluded.team1_players,
                        team2_players = excluded.team2_players,
                        team1_score = excluded.team1_score,
                        team2_score = excluded.team2_score,
                        recorded_at = excluded.recorded_at
                    """,
                    (
                        schedule_id,
                        round_num,
                        court,
                        json.dumps(list(team1_players)),
                        json.dumps(list(team2_players)),
                        team1_score,
                        team2_score,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            return True
        except Exception:
            logger.exception(
                "Failed to save game score: schedule_id=%s round=%s court=%s", schedule_id, round_num, court
            )
            return False

    def import_scores_csv(self, schedule_id: int, games: List[Dict], csv_text: str) -> Dict[str, Any]:
        """Bulk-import scores from CSV text with columns round,court,team1_score,team2_score.

        `games` is the output of list_games_for_scoring(schedule) - used to validate
        that a (round, court) pair exists and to look up the team rosters for the
        upsert. Re-importing the same round/court updates it (see save_game_score).
        Malformed or unmatched rows are reported individually; valid rows still apply.

        Returns {"applied": int, "errors": [str, ...]}.
        """
        games_by_key = {(g["round_num"], g["court"]): g for g in games}
        applied = 0
        errors: List[str] = []

        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            reader.fieldnames = [name.strip().lower() if name else name for name in (reader.fieldnames or [])]
        except Exception as exc:
            return {"applied": 0, "errors": [f"Could not parse CSV: {exc}"]}

        required = {"round", "court", "team1_score", "team2_score"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            return {"applied": 0, "errors": [f"CSV missing required column(s): {', '.join(sorted(missing))}"]}

        for line_num, row in enumerate(reader, start=2):
            try:
                round_num = int(row["round"])
                court = int(row["court"])
                team1_score = int(row["team1_score"])
                team2_score = int(row["team2_score"])
            except (TypeError, ValueError):
                errors.append(f"Row {line_num}: non-numeric round/court/score")
                continue

            game = games_by_key.get((round_num, court))
            if game is None:
                errors.append(f"Row {line_num}: no game at round {round_num}, court {court}")
                continue

            success = self.save_game_score(
                schedule_id, round_num, court, game["team1"], game["team2"], team1_score, team2_score
            )
            if success:
                applied += 1
            else:
                errors.append(f"Row {line_num}: failed to save round {round_num}, court {court}")

        logger.info("CSV score import: schedule_id=%s applied=%d errors=%d", schedule_id, applied, len(errors))
        return {"applied": applied, "errors": errors}

    def get_game_scores(self, schedule_id: int) -> List[Dict]:
        """Return every recorded score for a schedule, ordered by round then court."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM game_scores WHERE schedule_id = ? ORDER BY round_num, court",
                    (schedule_id,),
                )
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for row in rows:
                row["team1_players"] = json.loads(row["team1_players"])
                row["team2_players"] = json.loads(row["team2_players"])
            return rows
        except Exception:
            logger.exception("Failed to read game scores for schedule_id=%s", schedule_id)
            return []

    def get_leaderboard(self) -> List[Dict]:
        """Aggregate wins/losses/win-rate per player across every recorded score.

        Games without both scores entered are skipped; a tie (equal scores)
        counts toward games_played but not wins or losses.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gs.team1_players, gs.team2_players, gs.team1_score, gs.team2_score "
                    "FROM game_scores gs JOIN schedule_history sh ON sh.id = gs.schedule_id "
                    "WHERE gs.team1_score IS NOT NULL AND gs.team2_score IS NOT NULL "
                    "AND sh.deleted_at IS NULL"
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to read leaderboard data")
            return []

        stats: Dict[str, Dict[str, int]] = {}

        def _player_stats(name: str) -> Dict[str, int]:
            return stats.setdefault(name, {"wins": 0, "losses": 0, "games_played": 0})

        for team1_json, team2_json, team1_score, team2_score in rows:
            team1 = json.loads(team1_json)
            team2 = json.loads(team2_json)

            for player in team1 + team2:
                _player_stats(player)["games_played"] += 1

            if team1_score == team2_score:
                continue

            winners, losers = (team1, team2) if team1_score > team2_score else (team2, team1)
            for player in winners:
                _player_stats(player)["wins"] += 1
            for player in losers:
                _player_stats(player)["losses"] += 1

        leaderboard = []
        for player, record in stats.items():
            games = record["games_played"]
            win_rate = record["wins"] / games if games else 0.0
            leaderboard.append(
                {
                    "player": player,
                    "wins": record["wins"],
                    "losses": record["losses"],
                    "games_played": games,
                    "win_rate": win_rate,
                }
            )

        leaderboard.sort(key=lambda entry: (-entry["win_rate"], -entry["wins"], entry["player"]))
        return leaderboard

    def _save_weekly_relationships(self, cursor, schedule, week_date):
        """Save weekly partner and opponent relationships."""
        for round_data in schedule:
            games = round_data.get("games", []) if hasattr(round_data, "get") else [round_data]

            for game in games:
                if hasattr(game, "team1"):
                    team1 = [str(p) for p in game.team1]
                    team2 = [str(p) for p in game.team2]
                else:
                    team1 = [str(p) for p in game.get("team1", [])]
                    team2 = [str(p) for p in game.get("team2", [])]

                if len(team1) == 2:
                    p1, p2 = sorted(team1)
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO partner_history (week_date, player1, player2)
                        VALUES (?, ?, ?)
                    """,
                        (week_date, p1, p2),
                    )

                if len(team2) == 2:
                    p1, p2 = sorted(team2)
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO partner_history (week_date, player1, player2)
                        VALUES (?, ?, ?)
                    """,
                        (week_date, p1, p2),
                    )

                for t1_player in team1:
                    for t2_player in team2:
                        p1, p2 = sorted([t1_player, t2_player])
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO opponent_history (week_date, player1, player2)
                            VALUES (?, ?, ?)
                        """,
                            (week_date, p1, p2),
                        )

    def get_recent_schedules(self, limit: int = 10) -> List[Dict]:
        """Get recent (non-deleted) schedules from history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM schedule_history
                    WHERE deleted_at IS NULL
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (limit,),
                )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to read recent schedules from history")
            return []

    def get_schedule(self, schedule_id: int) -> Dict | None:
        """Fetch a single history entry by id, or None if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM schedule_history WHERE id = ?", (schedule_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        except Exception:
            logger.exception("Failed to read history entry %s", schedule_id)
            return None

    def delete_schedule(self, schedule_id: int) -> bool:
        """Soft-delete a single history entry by id (sets deleted_at).

        Scores in game_scores are preserved and only re-appear if restored.
        Returns True if a non-deleted row was found and marked deleted.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE schedule_history SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                    (datetime.now().isoformat(), schedule_id),
                )
                conn.commit()
                deleted = cursor.rowcount > 0
            if deleted:
                logger.info("History soft-delete: id=%s", schedule_id)
            return deleted
        except Exception:
            logger.exception("Failed to delete history entry %s", schedule_id)
            return False

    def restore_schedule(self, schedule_id: int) -> bool:
        """Undo a soft-delete. Returns True if a deleted row was found and restored."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE schedule_history SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
                    (schedule_id,),
                )
                conn.commit()
                restored = cursor.rowcount > 0
            if restored:
                logger.info("History restore: id=%s", schedule_id)
            return restored
        except Exception:
            logger.exception("Failed to restore history entry %s", schedule_id)
            return False

    def get_deleted_schedules(self, limit: int = 10) -> List[Dict]:
        """Get recently soft-deleted schedules, most recently deleted first."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM schedule_history WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT ?",
                    (limit,),
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to read deleted schedules from history")
            return []

    def purge_deleted(self, older_than_days: int = 30) -> int:
        """Permanently remove soft-deleted schedules (and their scores) older than
        the given number of days. Returns the number of schedules purged."""
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM schedule_history WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (cutoff,),
                )
                ids = [row[0] for row in cursor.fetchall()]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    cursor.execute(f"DELETE FROM game_scores WHERE schedule_id IN ({placeholders})", ids)
                    cursor.execute(f"DELETE FROM schedule_history WHERE id IN ({placeholders})", ids)
                conn.commit()
            if ids:
                logger.info("History purge: removed %d schedules older than %d days", len(ids), older_than_days)
            return len(ids)
        except Exception:
            logger.exception("Failed to purge deleted history entries")
            return 0

    def get_weekly_partners(self, weeks_back: int = 4) -> Dict[str, List]:
        """Get partner history keyed by week."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT week_date, player1, player2 FROM partner_history
                    ORDER BY week_date DESC LIMIT ?
                """,
                    (weeks_back * 100,),
                )

                history: Dict[str, List[Any]] = {}
                for week, p1, p2 in cursor.fetchall():
                    history.setdefault(week, []).append((p1, p2))

                return history
        except Exception:
            logger.exception("Failed to read weekly partner history")
            return {}


def _is_valid_objectives(objectives: Any) -> bool:
    """An objectives section is a non-empty dict of {weight, min, max} int entries."""
    if not isinstance(objectives, dict) or not objectives:
        return False
    for entry in objectives.values():
        if not isinstance(entry, dict):
            return False
        for key in ("weight", "min", "max"):
            if not isinstance(entry.get(key), int) or isinstance(entry.get(key), bool):
                return False
    return True


def _is_valid_scheduling(scheduling: Any) -> bool:
    """A scheduling section has an in-range round count and parseable HH:MM times."""
    if not isinstance(scheduling, dict):
        return False

    default_rounds = scheduling.get("default_rounds")
    if not isinstance(default_rounds, int) or isinstance(default_rounds, bool) or not (1 <= default_rounds <= 20):
        return False

    for key in ("start_time", "end_time"):
        value = scheduling.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
            return False
        hours, minutes = value.split(":")
        if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
            return False

    return True


def _is_valid_player_presets(player_presets: Any) -> bool:
    """player_presets must be dict[str, list[str]] (an empty dict is valid - no presets saved)."""
    if not isinstance(player_presets, dict):
        return False
    for name, players in player_presets.items():
        if not isinstance(name, str) or not isinstance(players, list):
            return False
        if not all(isinstance(player, str) for player in players):
            return False
    return True


_CONFIG_SECTION_VALIDATORS = {
    "objectives": _is_valid_objectives,
    "scheduling": _is_valid_scheduling,
    "player_presets": _is_valid_player_presets,
}

CONFIG_REPAIR_MESSAGES_KEY = "_config_repair_messages"
CONFIG_SCHEMA_VERSION_KEY = "schema_version"

# Bump this and add a new v(N-1)->vN migration function below whenever the
# config shape changes in a way that would otherwise silently drop user data
# on validate_config_shape's default-reset repair path.
CURRENT_CONFIG_SCHEMA_VERSION = 2


def _migrate_v1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
    """v1 configs predate the 'must_pair' constraint list - add it, empty."""
    constraints = config.setdefault("constraints", {})
    constraints.setdefault("must_pair", [])
    return config


# Keyed by the version being migrated FROM (i.e. _CONFIG_MIGRATIONS[1] takes a
# v1 config and returns a v2 config). Applied in order until the config
# reaches CURRENT_CONFIG_SCHEMA_VERSION.
_CONFIG_MIGRATIONS: Dict[int, Any] = {
    1: _migrate_v1_to_v2,
}


def migrate_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], list[int]]:
    """Migrate a config forward to CURRENT_CONFIG_SCHEMA_VERSION in place of resetting it.

    Configs written before schema_version existed are treated as v1. Returns
    (migrated_config, versions_migrated_from) - the second is empty if no
    migration was needed. A config from an unknown future version is left
    untouched (not migrated backward, not reset) with a warning logged by the
    caller; a config stuck on a version with no registered migration path
    also stops there rather than raising.
    """
    version = config.get(CONFIG_SCHEMA_VERSION_KEY, 1)
    if not isinstance(version, int):
        version = 1

    applied: list[int] = []
    while version < CURRENT_CONFIG_SCHEMA_VERSION:
        migration = _CONFIG_MIGRATIONS.get(version)
        if migration is None:
            break
        config = migration(config)
        applied.append(version)
        version += 1

    config[CONFIG_SCHEMA_VERSION_KEY] = version
    return config, applied


def validate_config_shape(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Repair any top-level config section that doesn't match its expected shape.

    app_config.json can be hand-edited, synced, or corrupted; without this, a
    wrong type (e.g. "objectives": "high") would propagate until some tab
    crashed with an opaque TypeError. Constraints are already covered by
    normalize_config_constraints and are not re-checked here. Returns a new
    dict; on any repairs, CONFIG_REPAIR_MESSAGES_KEY holds one human-readable
    message per repaired section for the caller to surface and then discard -
    it must never be persisted.
    """
    repaired = deepcopy(config)
    repaired_sections = []

    for section, is_valid in _CONFIG_SECTION_VALIDATORS.items():
        if not is_valid(repaired.get(section)):
            repaired[section] = deepcopy(default_config[section])
            repaired_sections.append(section)

    if repaired_sections:
        repaired[CONFIG_REPAIR_MESSAGES_KEY] = [
            f"Config issue fixed: '{section}' was invalid and reset to defaults." for section in repaired_sections
        ]

    return repaired


MAX_CONFIG_UPLOAD_BYTES = 1_000_000  # 1 MB - a hand-edited app_config.json is a few KB


def export_config_json(config: Dict[str, Any]) -> str:
    """Serialize config to pretty JSON for download, stripping the repair-messages channel."""
    exportable = deepcopy(config)
    exportable.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
    return json.dumps(exportable, indent=2)


def import_config_json(raw_bytes: bytes, default_config: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Parse and repair an uploaded config file. Never raises.

    Returns (config, messages). `messages` is empty on a clean import; on any
    problem (invalid JSON, oversized file, wrong shape) it holds human-readable
    reasons and `config` falls back to validate_config_shape's repair-and-warn
    path (or, for unparseable input, a full default copy).
    """
    if len(raw_bytes) > MAX_CONFIG_UPLOAD_BYTES:
        return deepcopy(default_config), [
            f"File too large ({len(raw_bytes)} bytes) - must be under {MAX_CONFIG_UPLOAD_BYTES} bytes."
        ]

    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        return deepcopy(default_config), [f"Not valid JSON: {exc}"]

    if not isinstance(parsed, dict):
        return deepcopy(default_config), ["Uploaded file must contain a JSON object."]

    normalized = normalize_config_constraints(parsed)
    repaired = validate_config_shape(normalized, default_config)
    messages = repaired.pop(CONFIG_REPAIR_MESSAGES_KEY, [])
    return repaired, messages


class ConfigurationManager:
    """Manage application configuration and constraints."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        default_names_path: str | Path | None = None,
    ):
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else _resolve_storage_path(
                "PICKLEBALL_APP_CONFIG",
                "app_config.json",
            )
        )
        self.default_names_path = (
            Path(default_names_path) if default_names_path is not None else _resolve_default_names_path()
        )
        self.config_path.parent.mkdir(exist_ok=True)
        self.default_config = {
            CONFIG_SCHEMA_VERSION_KEY: CURRENT_CONFIG_SCHEMA_VERSION,
            "objectives": {
                "equal_games": {"weight": 5000, "min": 1000, "max": 10000},
                "equal_variety": {"weight": 4000, "min": 1000, "max": 10000},
                "constraints": {"weight": 6000, "min": 2000, "max": 10000},
            },
            "constraints": {"do_not_pair": [], "do_not_oppose": [], "must_pair": []},
            "player_presets": {
                "Regular Group": [
                    "Teresa",
                    "Katie",
                    "Chris",
                    "Lisa",
                    "Mary",
                    "Michelle",
                    "Beth",
                    "Betsi",
                    "Kat",
                    "Nancy",
                    "Laura",
                    "Denise",
                    "LizBeth",
                    "MichelleM",
                    "BethL",
                ],
                "12 Player League": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "Diana",
                    "Eve",
                    "Frank",
                    "Grace",
                    "Henry",
                    "Ivy",
                    "Jack",
                    "Kate",
                    "Liam",
                ],
                "16 Player Tournament": [
                    "P1",
                    "P2",
                    "P3",
                    "P4",
                    "P5",
                    "P6",
                    "P7",
                    "P8",
                    "P9",
                    "P10",
                    "P11",
                    "P12",
                    "P13",
                    "P14",
                    "P15",
                    "P16",
                ],
            },
            "scheduling": {
                "default_rounds": 8,
                "start_time": "14:00",
                "end_time": "16:00",
                "game_duration_minutes": 15,
            },
        }

    def load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    raw_config = json.load(f)
                # Migrate on the raw on-disk shape - self._merge_configs would otherwise
                # backfill schema_version from self.default_config, masking an older version.
                migrated, applied_from = migrate_config(deepcopy(raw_config))
                if applied_from:
                    logger.info(
                        "Config migrated: schema_version %s -> %s", applied_from[0], migrated[CONFIG_SCHEMA_VERSION_KEY]
                    )
                elif migrated[CONFIG_SCHEMA_VERSION_KEY] > CURRENT_CONFIG_SCHEMA_VERSION:
                    logger.warning(
                        "Config schema_version %s is newer than this app supports (%s) - using as-is",
                        migrated[CONFIG_SCHEMA_VERSION_KEY],
                        CURRENT_CONFIG_SCHEMA_VERSION,
                    )
                loaded = normalize_config_constraints(self._merge_configs(self.default_config, migrated))
            else:
                loaded = normalize_config_constraints(deepcopy(self.default_config))
        except Exception:
            logger.exception("Failed to load app_config.json - falling back to defaults")
            loaded = normalize_config_constraints(deepcopy(self.default_config))

        repaired = validate_config_shape(loaded, self.default_config)
        repair_messages = repaired.get(CONFIG_REPAIR_MESSAGES_KEY)
        if repair_messages:
            logger.warning("Config repaired on load: %s", "; ".join(repair_messages))
        return repaired

    def save_config(self, config: Dict) -> bool:
        """Save configuration to file. Returns True on success, False on failure."""
        try:
            normalized_config = normalize_config_constraints(deepcopy(config))
            normalized_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
            normalized_config[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(normalized_config, f, indent=2)
            return True
        except Exception as e:
            logger.exception("Failed to save configuration")
            st.error(f"Failed to save configuration: {e}")
            return False

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        result = deepcopy(default)
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def load_default_names(self) -> List[str]:
        """Load default player names from file."""
        try:
            if self.default_names_path.exists():
                with open(self.default_names_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception:
            return []


class SkillRatingManager:
    """Store per-player skill ratings (1-5) for team-balance analytics.

    Deliberately does not feed into the genetic scheduler's fitness function -
    that would require dedicated algorithm design and verification work
    (see #124/#126 discussion). This gives visibility into how skill-balanced
    a generated schedule happens to be, as a first step.
    """

    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, skills_path: str | Path | None = None):
        self.skills_path = (
            Path(skills_path)
            if skills_path is not None
            else _resolve_storage_path("PICKLEBALL_SKILLS_FILE", "player_skills.json")
        )

    def load_skills(self) -> Dict[str, int]:
        """Return {player_name: rating}. Missing/corrupted file yields an empty dict."""
        try:
            if self.skills_path.exists():
                with open(self.skills_path, "r", encoding="utf-8") as f:
                    skills = json.load(f)
                if isinstance(skills, dict):
                    return {
                        str(player): rating
                        for player, rating in skills.items()
                        if isinstance(rating, int) and self.MIN_RATING <= rating <= self.MAX_RATING
                    }
            return {}
        except Exception:
            logger.exception("Failed to load player skills")
            return {}

    def save_skills(self, skills: Dict[str, int]) -> bool:
        """Persist the full skills dict, overwriting whatever was there before."""
        try:
            self.skills_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.skills_path, "w", encoding="utf-8") as f:
                json.dump(skills, f, indent=2)
            return True
        except Exception:
            logger.exception("Failed to save player skills")
            return False

    def set_skill(self, player: str, rating: int) -> bool:
        """Set one player's rating and persist immediately."""
        if not (self.MIN_RATING <= rating <= self.MAX_RATING):
            raise ValueError(f"Rating must be between {self.MIN_RATING} and {self.MAX_RATING}")
        skills = self.load_skills()
        skills[player] = rating
        return self.save_skills(skills)


class PlayerManager:
    """Advanced player management with substitutions and availability."""

    player_data_path = Path("data/default_player_names.json")

    @staticmethod
    def create_substitution_interface():
        """Create UI for player substitutions."""
        st.subheader("🔄 Player Substitutions")
        st.info("Configure players for different time periods (e.g., early vs late players)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**First Half Players**")
            first_half = st.text_area(
                "Players available for early rounds:",
                value=st.session_state.get("first_half_players", ""),
                height=100,
                key="first_half_input",
            )
            st.session_state.first_half_players = first_half

        with col2:
            st.markdown("**Second Half Players**")
            second_half = st.text_area(
                "Players available for later rounds:",
                value=st.session_state.get("second_half_players", ""),
                height=100,
                key="second_half_input",
            )
            st.session_state.second_half_players = second_half

        sub_round = st.number_input(
            "Substitution round (when to switch players):",
            min_value=1,
            max_value=10,
            value=st.session_state.get("substitution_round", 4),
        )
        st.session_state.substitution_round = sub_round

        return first_half, second_half, sub_round

    @staticmethod
    def apply_availability_constraints(players: List[str], constraints: Dict) -> List[str]:
        """Apply availability constraints to a player list."""
        available_players = []
        unavailable = constraints.get("unavailable_players", [])

        for player in players:
            if player not in unavailable:
                available_players.append(player)

        return available_players

    @classmethod
    def load_player_data(cls) -> List[str]:
        """Load persisted player defaults for compatibility with older callers."""
        try:
            if cls.player_data_path.exists():
                with open(cls.player_data_path, "r", encoding="utf-8") as file_handle:
                    data = json.load(file_handle)
                if isinstance(data, list):
                    return [str(player) for player in data if str(player).strip()]
        except Exception:
            pass
        return []

    @classmethod
    def save_player_data(cls, players: List[str]) -> Dict[str, Any]:
        """Persist player defaults and return a small operation summary."""
        try:
            cleaned_players = [player.strip() for player in players if player and player.strip()]
            cls.player_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.player_data_path, "w", encoding="utf-8") as file_handle:
                json.dump(cleaned_players, file_handle, indent=2)
            return {"success": True, "count": len(cleaned_players)}
        except Exception as exc:
            return {"success": False, "error": str(exc), "count": 0}

    @staticmethod
    def validate_player_list(players: List[str]) -> bool:
        """Validate a player list without raising, for older test helpers."""
        try:
            validate_player_names(players)
            return True
        except ValueError:
            return False
