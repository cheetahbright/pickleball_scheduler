#!/usr/bin/env python3
"""Schedule/score history persistence (SQLite)."""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from src.managers._paths import _resolve_storage_path
except ImportError:
    from managers._paths import _resolve_storage_path

_schedule_analytics_core = import_module_with_fallback("utils.schedule_analytics_core")
calculate_fairness_metrics = _schedule_analytics_core.calculate_fairness_metrics
serialize_schedule_for_json = _schedule_analytics_core.serialize_schedule_for_json

logger = logging.getLogger(__name__)


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

    def get_all_scored_games(self) -> List[Dict]:
        """Return every recorded score across all non-deleted schedules, oldest first.

        Ordering by recorded_at (rather than schedule_id/round/court, or
        whatever order SQLite happens to return rows in) is what makes ELO
        replay deterministic (#145): the same score history always produces
        the same ratings, regardless of the order scores were entered in.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gs.team1_players, gs.team2_players, gs.team1_score, gs.team2_score, gs.recorded_at "
                    "FROM game_scores gs JOIN schedule_history sh ON sh.id = gs.schedule_id "
                    "WHERE gs.team1_score IS NOT NULL AND gs.team2_score IS NOT NULL "
                    "AND sh.deleted_at IS NULL "
                    "ORDER BY gs.recorded_at ASC"
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to read all scored games")
            return []

        return [
            {
                "team1_players": json.loads(team1_json),
                "team2_players": json.loads(team2_json),
                "team1_score": team1_score,
                "team2_score": team2_score,
                "recorded_at": recorded_at,
            }
            for team1_json, team2_json, team1_score, team2_score, recorded_at in rows
        ]

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

                for team in (team1, team2):
                    if len(team) == 2:
                        p1, p2 = sorted(team)
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
