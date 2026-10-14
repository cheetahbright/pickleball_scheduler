#!/usr/bin/env python3
"""Schedule/score history persistence (SQLite)."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from src.managers._paths import _resolve_storage_path
except ImportError:
    from managers._paths import _resolve_storage_path

try:
    from src.utils.schedule_shapes import extract_game_teams, games_in_round
except ImportError:
    from utils.schedule_shapes import extract_game_teams, games_in_round

_schedule_analytics_core = import_module_with_fallback("utils.schedule_analytics_core")
calculate_fairness_metrics = _schedule_analytics_core.calculate_fairness_metrics
serialize_schedule_for_json = _schedule_analytics_core.serialize_schedule_for_json

logger = logging.getLogger(__name__)


# A game_scores row that records an actual result. 0-0 is excluded because no
# pickleball game ends 0-0: those rows were written by an earlier "Save Scores"
# that stored every unplayed game as 0-0, and must not count as played ties.
def _result_row_condition(prefix: str) -> str:
    return (
        f"{prefix}team1_score IS NOT NULL AND {prefix}team2_score IS NOT NULL "
        f"AND NOT ({prefix}team1_score = 0 AND {prefix}team2_score = 0)"
    )


# A module constant with no input in it, so Bandit's B608 (SQL injection) is
# suppressed where queries splice it in; their values are all bound as ?.
_RESULT_ROW_SQL = _result_row_condition("gs.")


@dataclass(frozen=True)
class ReplanSaveResult:
    """Outcome of HistoryManager.update_schedule_after_replan.

    status is "saved", or why nothing was written: "missing", "changed",
    "scored" (with the scored round numbers), or "failed".
    """

    status: str
    scored_rounds: tuple[int, ...] = ()

    @property
    def saved(self) -> bool:
        return self.status == "saved"


@dataclass(frozen=True)
class ScoreSaveResult:
    """Outcome of HistoryManager.save_game_scores.

    status is "saved", or why nothing was written: "missing", "changed"
    (with the (round, court) of each game the stored schedule no longer
    has), "invalid" (a negative score), or "failed".
    """

    status: str
    stale_games: tuple[tuple[int, int], ...] = ()

    @property
    def saved(self) -> bool:
        return self.status == "saved"


def _fairness_score(schedule, players, roster_by_round=None) -> float:
    """The schedule's fairness, scored the same way as the Main and Analytics
    tabs (roster, dimensions and a replan's per-round rosters included), so the
    History "Fairness" column agrees with them. Used for both a new row and a
    replanned one."""
    courts = max((len(r.get("games", [])) for r in schedule if hasattr(r, "get")), default=0)
    metrics = calculate_fairness_metrics(
        schedule,
        num_players=len(players),
        num_rounds=len(schedule),
        num_courts=courts,
        players=list(players),
        roster_by_round=roster_by_round,
    )
    return metrics.get("overall_fairness", 0)


def _games_by_round(schedule) -> list[list[tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    """Each round's games as comparable (court, team1, team2) tuples, whatever the schedule's shape."""
    rounds = []
    for round_data in serialize_schedule_for_json(schedule):
        games = []
        for game in games_in_round(round_data):
            team1, team2, court = extract_game_teams(game)
            games.append((str(court), tuple(sorted(map(str, team1))), tuple(sorted(map(str, team2)))))
        rounds.append(sorted(games))
    return rounds


def stored_time(timestamp: str, tz: tzinfo | None = None) -> datetime:
    """A stored timestamp as the time it was in tz (None: the server's time zone).

    Timestamps are stored as the server's local time with no zone
    (datetime.now().isoformat()), which is how astimezone() reads a naive time.
    """
    return datetime.fromisoformat(timestamp).astimezone(tz)


def _lineup(team1, team2) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(map(str, team1))), tuple(sorted(map(str, team2)))


def _lineups_by_slot(schedule) -> dict[tuple[int, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Each game's teams keyed by (round number, court), numbered as list_games_for_scoring numbers them."""
    lineups = {}
    for round_num, round_data in enumerate(serialize_schedule_for_json(schedule), 1):
        for game in games_in_round(round_data):
            team1, team2, court = extract_game_teams(game)
            lineups[(round_num, str(court))] = _lineup(team1, team2)
    return lineups


# The schedule_history columns the app reads. deleted_at is left out: a
# database from before soft-delete lacks it, and ensure_db_exists adds it.
_HISTORY_COLUMNS = frozenset(
    {"id", "timestamp", "num_players", "num_rounds", "algorithm", "fairness_score", "schedule_data", "settings_data"}
)


def describe_history_db(data: bytes) -> tuple[int, int]:
    """Return (schedules, recorded game results) in a history database given as the bytes of its file.

    Raises ValueError, with a reason for the user, unless the bytes are an
    intact SQLite database holding this app's schedule_history table - and no
    triggers or views, which this app never creates and a restore would carry
    into the live database.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "check.db"
        path.write_bytes(data)
        try:
            with closing(sqlite3.connect(path)) as conn:
                if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("it is damaged (it fails SQLite's integrity check)")
                objects = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
                if any(kind in ("trigger", "view") for kind, _name in objects):
                    raise ValueError("it holds triggers or views, which this app never creates")
                tables = {name for kind, name in objects if kind == "table"}
                if "schedule_history" not in tables:
                    raise ValueError("it has no schedule history table")
                columns = {row[1] for row in conn.execute("PRAGMA table_info(schedule_history)")}
                missing = sorted(_HISTORY_COLUMNS - columns)
                if missing:
                    raise ValueError(f"its schedule history table lacks column(s): {', '.join(missing)}")
                active = " WHERE deleted_at IS NULL" if "deleted_at" in columns else ""
                count_schedules = f"SELECT COUNT(*) FROM schedule_history{active}"  # nosec B608
                count_scores = f"SELECT COUNT(*) FROM game_scores WHERE {_result_row_condition('')}"  # nosec B608
                schedules = conn.execute(count_schedules).fetchone()[0]
                scores = conn.execute(count_scores).fetchone()[0] if "game_scores" in tables else 0
        except sqlite3.Error as exc:
            raise ValueError(f"it isn't a readable SQLite database ({exc})") from exc
    return schedules, scores


def _give_up_when_locked(status: int, _remaining: int, _total: int) -> None:
    """Progress callback for Connection.backup: stop once a lock wait has outlasted the busy timeout.

    Connection.backup itself retries a locked database forever, so a
    connection that keeps the database locked would otherwise hang the caller.
    """
    if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        raise sqlite3.OperationalError("database is locked")


# Leading byte-order mark that Excel and Notepad put on a saved CSV.
_BOM = "\ufeff"
# Comma is the score sheet's own delimiter; tab is what copying cells out of
# Excel or Google Sheets puts on the clipboard, and semicolon is how Excel
# saves a CSV in locales that use a decimal comma.
_SCORE_CSV_DELIMITERS = (",", "\t", ";")


def _team_cell_names(cell: str) -> list[str]:
    # Spacing and case are compared as find_player_name_issues compares names.
    return sorted(re.sub(r"\s+", " ", name.strip()).casefold() for name in cell.split("&"))


def _before_excel_ansi(cell: str) -> str:
    """The cell as it was before Excel read the UTF-8 sheet as Windows-1252 ("JosÃ©" was "José"), else the cell.

    Excel opens a CSV without a byte-order mark in the ANSI code page, as it
    did every score sheet downloaded before the sheet had one.
    """
    raw = bytearray()
    for char in cell:
        try:
            raw += char.encode("cp1252")
        except UnicodeEncodeError:
            if ord(char) > 0xFF:
                return cell
            raw.append(ord(char))  # a byte Windows-1252 leaves undefined, which Windows reads as U+0081 and so on
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return cell


def _team_cell_matches(cell: str, team) -> bool:
    """Whether a score sheet's team cell names this team, as blank_score_sheet_csv writes it ("A & B").

    The cell may have lost the apostrophe the sheet puts before a name a
    spreadsheet would read as a formula, or kept it, and Excel may have
    garbled its accented letters (see _before_excel_ansi); spacing, case and
    the order of the names don't matter.
    """
    expected = _team_cell_names(" & ".join(map(str, team)))
    readings = {cell, _before_excel_ansi(cell)}
    readings |= {text.removeprefix("'") for text in readings}
    return any(_team_cell_names(text) == expected for text in readings)


def _sheet_names_teams(sheet_teams: Sequence[str], teams: Sequence) -> bool:
    """Whether a score sheet row's team1 and team2 cells name these teams, in this order. A blank cell names either."""
    return all(not cell or _team_cell_matches(cell, team) for cell, team in zip(sheet_teams, teams))


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

    # sqlite3's default busy timeout (5s) is short enough that two Streamlit
    # sessions saving near-simultaneously can hit "database is locked" and
    # fail outright; WAL mode lets readers proceed without blocking on a
    # writer at all, and a longer timeout gives a genuine writer/writer
    # collision more room to resolve on its own before giving up.
    _BUSY_TIMEOUT_SECONDS = 15.0

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that commits/rolls back on exit (sqlite3.Connection's
        own context-manager behavior) and is always explicitly closed afterward,
        rather than relying on CPython refcounting to close it."""
        with closing(sqlite3.connect(self.db_path, timeout=self._BUSY_TIMEOUT_SECONDS)) as raw_conn:
            raw_conn.execute("PRAGMA journal_mode=WAL")
            with raw_conn as conn:
                yield conn

    def ensure_db_exists(self):
        """Create the history database if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
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

            # Nothing writes or reads opponent_history any more; the table stays
            # so the schema matches databases earlier versions wrote rows into.
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

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_schedule_history_active " "ON schedule_history(deleted_at, timestamp)"
            )

            conn.commit()

    def export_db_bytes(self) -> bytes:
        """Return a consistent snapshot of the whole database, as the bytes of a standalone SQLite file.

        Made with SQLite's backup API, which copies the database as of one
        moment even while other sessions write to it. Copying the file itself
        could catch a write half done, and would miss whatever still sits in
        the -wal file. Raises on failure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / self.db_path.name
            with self._connect() as conn, closing(sqlite3.connect(snapshot_path)) as snapshot:
                conn.backup(snapshot, progress=_give_up_when_locked)
                # The copy inherits WAL mode; out of it, the one file holds everything.
                snapshot.execute("PRAGMA journal_mode=DELETE")
            return snapshot_path.read_bytes()

    def import_db_bytes(self, data: bytes) -> None:
        """Replace the whole database with the one in ``data`` (the bytes of a SQLite file).

        Copied in with SQLite's backup API, in one write transaction, rather
        than by swapping the file: other sessions then see either the old
        database or the new one, and the live -wal file can't be replayed onto
        a database it doesn't belong to. Check ``data`` with describe_history_db
        first, and run ensure_db_exists afterwards: a database from an older
        version may lack a table or column. Raises on failure (also when another
        connection keeps the database locked past the busy timeout), leaving
        the database as it was.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / self.db_path.name
            source_path.write_bytes(data)
            with closing(sqlite3.connect(source_path)) as source, self._connect() as conn:
                source.backup(conn, progress=_give_up_when_locked)

    def holds_schedule(self, schedule_id: int, schedule) -> bool:
        """Whether history entry ``schedule_id`` exists (deleted or not) and holds ``schedule``'s games."""
        entry = self.get_schedule(schedule_id)
        if entry is None:
            return False
        try:
            return _games_by_round(json.loads(entry["schedule_data"])) == _games_by_round(schedule)
        except Exception:
            logger.exception("Could not compare history entry %s with a schedule", schedule_id)
            return False

    def save_schedule(
        self, schedule: List[Dict], players: List[str], settings: Dict, algorithm: str = "genetic"
    ) -> int | None:
        """Save schedule data and weekly relationships. Returns the new row's id, or
        None on failure - callers use the id to attach game scores to this schedule,
        and are responsible for surfacing a None return to the user."""
        try:
            fairness_score = _fairness_score(schedule, players, settings.get("roster_by_round"))
            serializable_schedule = serialize_schedule_for_json(schedule)
            stored_settings = dict(settings)
            stored_settings["players"] = list(players)

            with self._connect() as conn:
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
                        algorithm,
                        fairness_score,
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

        except Exception:
            logger.exception("History save failed")
            return None

    def rounds_with_results(self, schedule_id: int, from_round: int) -> List[int] | None:
        """Round numbers >= from_round that have a recorded result for this schedule.

        Returns None if the scores can't be read, so callers can refuse to
        act rather than assume there are none.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT gs.round_num FROM game_scores gs "
                    f"WHERE gs.schedule_id = ? AND gs.round_num >= ? AND {_RESULT_ROW_SQL} "  # nosec B608
                    "ORDER BY gs.round_num",
                    (schedule_id, from_round),
                ).fetchall()
            return [row[0] for row in rows]
        except Exception:
            logger.exception("Failed to read scored rounds for schedule_id=%s", schedule_id)
            return None

    def update_schedule_after_replan(
        self,
        schedule_id: int,
        previous_schedule: List[Dict],
        schedule: List[Dict],
        players: List[str],
        replanned_from_round: int,
        roster_by_round: List[List[str]],
    ) -> ReplanSaveResult:
        """Persist a mid-session replan onto the schedule's existing history row.

        The row keeps its id, so scores recorded for the played rounds stay
        attached to the teams that actually played them, and a later load
        from history shows the replanned rounds instead of the originals.
        The settings also record each round's roster, so "sitting out" stays
        right after a reload.

        Checked and written in one write-locked transaction, so nothing can
        change in between. The save is refused (nothing written) when:
        - "missing": the row is gone or deleted;
        - "changed": the stored schedule no longer matches previous_schedule,
          the one this replan was computed from (another session replanned
          or the row was otherwise rewritten since this session loaded it);
        - "scored": a round being replaced now has a recorded result.

        Score rows for the replaced rounds that hold no result (blank, or the
        0-0 an older Save Scores wrote for every unplayed game) are deleted,
        since the games they name no longer exist. The replanned rounds' pairs
        are added to the partner_history table, which keeps the replaced
        rounds' pairs too, because it doesn't record which schedule a pair
        came from. Weekly Partner History doesn't read that table:
        get_weekly_partners works the pairs out from the schedules, so the
        replaced rounds leave it.
        """
        try:
            serializable_schedule = serialize_schedule_for_json(schedule)
            fairness_score = _fairness_score(schedule, players, roster_by_round)
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    "SELECT timestamp, settings_data, schedule_data FROM schedule_history "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (schedule_id,),
                ).fetchone()
                if row is None:
                    logger.warning("Replan not saved: schedule_id=%s is missing or deleted", schedule_id)
                    return ReplanSaveResult("missing")
                timestamp, settings_json, stored_schedule_json = row

                try:
                    stored_schedule = json.loads(stored_schedule_json)
                except (TypeError, ValueError):
                    stored_schedule = None
                if stored_schedule is None or _games_by_round(stored_schedule) != _games_by_round(previous_schedule):
                    logger.warning("Replan not saved: schedule_id=%s changed since it was loaded", schedule_id)
                    return ReplanSaveResult("changed")

                scored_rounds = tuple(
                    row[0]
                    for row in cursor.execute(
                        "SELECT DISTINCT gs.round_num FROM game_scores gs "
                        f"WHERE gs.schedule_id = ? AND gs.round_num >= ? AND {_RESULT_ROW_SQL} "  # nosec B608
                        "ORDER BY gs.round_num",
                        (schedule_id, replanned_from_round),
                    ).fetchall()
                )
                if scored_rounds:
                    logger.warning("Replan not saved: schedule_id=%s rounds %s have scores", schedule_id, scored_rounds)
                    return ReplanSaveResult("scored", scored_rounds)

                try:
                    settings = json.loads(settings_json) if settings_json else {}
                except (TypeError, ValueError):
                    settings = {}
                if not isinstance(settings, dict):
                    settings = {}
                settings["players"] = list(players)
                settings["roster_by_round"] = [list(roster) for roster in roster_by_round]
                settings["replanned_from_round"] = replanned_from_round

                cursor.execute(
                    "UPDATE schedule_history SET num_players = ?, num_rounds = ?, fairness_score = ?, "
                    "schedule_data = ?, settings_data = ? WHERE id = ?",
                    (
                        len(players),
                        len(schedule),
                        fairness_score,
                        json.dumps(serializable_schedule),
                        json.dumps(settings),
                        schedule_id,
                    ),
                )
                cursor.execute(
                    "DELETE FROM game_scores WHERE schedule_id = ? AND round_num >= ? "
                    f"AND NOT ({_result_row_condition('')})",  # nosec B608
                    (schedule_id, replanned_from_round),
                )
                try:
                    week_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
                except (TypeError, ValueError):
                    week_date = datetime.now().strftime("%Y-%m-%d")
                self._save_weekly_relationships(cursor, schedule[replanned_from_round - 1 :], week_date)
                conn.commit()

            logger.info(
                "History replan saved: schedule_id=%s rounds=%d replanned_from=%d",
                schedule_id,
                len(schedule),
                replanned_from_round,
            )
            return ReplanSaveResult("saved")
        except Exception:
            logger.exception("Failed to save replan for schedule_id=%s", schedule_id)
            return ReplanSaveResult("failed")

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
        """Record or update the score for one game (see save_game_scores)."""
        game = {"round_num": round_num, "court": court, "team1": team1_players, "team2": team2_players}
        return self.save_game_scores(schedule_id, [(game, team1_score, team2_score)]).saved

    def save_game_scores(
        self, schedule_id: int, scores: Sequence[tuple[Dict, int | None, int | None]]
    ) -> ScoreSaveResult:
        """Record or update the scores of several games, all or none of them.

        Each entry is (game, team1_score, team2_score), with game a
        list_games_for_scoring row of the schedule the caller shows. A
        schedule_id+round+court is unique - re-entering a score for the same
        game updates it rather than duplicating.

        Checked and written in one write-locked transaction, so a busy
        database is waited for once rather than once per game, and nothing
        can change in between. Nothing is written when:
        - "invalid": a score is negative;
        - "missing": the schedule is gone or deleted;
        - "changed": a game is no longer in the stored schedule with these
          teams (another session replanned since the caller loaded it), so
          its score would credit players who didn't play it.
        """
        try:
            invalid_games = [
                (game["round_num"], game["court"])
                for game, team1_score, team2_score in scores
                if any(score is not None and score < 0 for score in (team1_score, team2_score))
            ]
            if invalid_games:
                logger.warning("Scores not saved: schedule_id=%s negative score at %s", schedule_id, invalid_games)
                return ScoreSaveResult("invalid")
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    "SELECT schedule_data FROM schedule_history WHERE id = ? AND deleted_at IS NULL",
                    (schedule_id,),
                ).fetchone()
                if row is None:
                    logger.warning("Scores not saved: schedule_id=%s is missing or deleted", schedule_id)
                    return ScoreSaveResult("missing")

                try:
                    stored_lineups = _lineups_by_slot(json.loads(row[0]))
                except (TypeError, ValueError):
                    stored_lineups = {}
                stale_games = tuple(
                    (game["round_num"], game["court"])
                    for game, _team1_score, _team2_score in scores
                    if stored_lineups.get((game["round_num"], str(game["court"])))
                    != _lineup(game["team1"], game["team2"])
                )
                if stale_games:
                    logger.warning("Scores not saved: schedule_id=%s changed at %s", schedule_id, stale_games)
                    return ScoreSaveResult("changed", stale_games)

                recorded_at = datetime.now().isoformat()
                cursor.executemany(
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
                    [
                        (
                            schedule_id,
                            game["round_num"],
                            game["court"],
                            json.dumps(list(game["team1"])),
                            json.dumps(list(game["team2"])),
                            team1_score,
                            team2_score,
                            recorded_at,
                        )
                        for game, team1_score, team2_score in scores
                    ],
                )
                conn.commit()
            return ScoreSaveResult("saved")
        except Exception:
            logger.exception("Failed to save %d game score(s) for schedule_id=%s", len(scores), schedule_id)
            return ScoreSaveResult("failed")

    def import_scores_csv(self, schedule_id: int, games: List[Dict], csv_text: str) -> Dict[str, Any]:
        """Bulk-import scores from CSV text with columns round,court,team1_score,team2_score.

        `games` is the output of list_games_for_scoring(schedule) - used to validate
        that a (round, court) pair exists and to look up the team rosters for the
        upsert. Re-importing the same round/court updates it (see save_game_scores).
        Where a row's team1/team2 cells (which the blank score sheet fills in) are
        filled in, they must name that game's teams: a sheet from an older
        schedule would otherwise credit its scores to whoever plays that court
        now. Teams named the other way round get their scores swapped.
        Malformed or unmatched rows are reported individually; the valid rows are
        then saved together, all or none of them. Tab- or semicolon-separated
        text (cells copied out of a spreadsheet) is read too.

        Returns {"applied": int, "errors": [str, ...]}.
        """
        games_by_key = {(g["round_num"], g["court"]): g for g in games}
        to_save: List[tuple[Dict, int, int]] = []
        errors: List[str] = []

        required = {"round", "court", "team1_score", "team2_score"}
        lines = csv_text.removeprefix(_BOM).splitlines(keepends=True)
        while lines and not lines[0].strip():  # else the first blank line would be read as the header
            lines.pop(0)
        try:
            # The delimiter that parses the header into the required columns;
            # comma when none does, so the error names the missing ones.
            delimiter = next(
                (
                    candidate
                    for candidate in _SCORE_CSV_DELIMITERS
                    if required.issubset(
                        name.strip().lower() for name in next(csv.reader(lines[:1], delimiter=candidate), [])
                    )
                ),
                ",",
            )
            reader = csv.DictReader(io.StringIO("".join(lines)), delimiter=delimiter)
            reader.fieldnames = [name.strip().lower() if name else name for name in (reader.fieldnames or [])]
        except Exception as exc:
            return {"applied": 0, "errors": [f"Could not parse CSV: {exc}"]}

        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            return {"applied": 0, "errors": [f"CSV missing required column(s): {', '.join(sorted(missing))}"]}

        # Given once, with the first row whose teams aren't the game's.
        older_sheet_advice = (
            " The sheet may be from an older schedule - download a new score sheet. If these are this schedule's "
            "results anyway, clear the team1 and team2 cells of rows like this to import them by round and court."
        )
        for line_num, row in enumerate(reader, start=2):
            try:
                round_num = int(row["round"])
                court = int(row["court"])
                team1_score = int(row["team1_score"])
                team2_score = int(row["team2_score"])
            except (TypeError, ValueError):
                errors.append(f"Row {line_num}: non-numeric round/court/score")
                continue

            if team1_score < 0 or team2_score < 0:
                errors.append(f"Row {line_num}: a score can't be negative")
                continue

            if team1_score == 0 and team2_score == 0:
                errors.append(f"Row {line_num}: 0-0 isn't a game result - leave the row out until it's played")
                continue

            game = games_by_key.get((round_num, court))
            if game is None:
                errors.append(f"Row {line_num}: no game at round {round_num}, court {court}")
                continue

            sheet_teams = [(row.get(column) or "").strip() for column in ("team1", "team2")]
            if not _sheet_names_teams(sheet_teams, (game["team1"], game["team2"])):
                if not _sheet_names_teams(sheet_teams, (game["team2"], game["team1"])):
                    errors.append(
                        f"Row {line_num}: the sheet has {sheet_teams[0] or '?'} vs {sheet_teams[1] or '?'} at round "
                        f"{round_num}, court {court}, but this schedule has {' & '.join(game['team1'])} vs "
                        f"{' & '.join(game['team2'])} there.{older_sheet_advice}"
                    )
                    older_sheet_advice = ""
                    continue
                team1_score, team2_score = team2_score, team1_score

            to_save.append((game, team1_score, team2_score))

        applied = 0
        if to_save:
            result = self.save_game_scores(schedule_id, to_save)
            if result.saved:
                applied = len(to_save)
            elif result.status == "changed":
                errors.append(
                    "This schedule was changed in another session since you opened it, so nothing was imported. "
                    "Load it again from History, then import the scores."
                )
            elif result.status == "missing":
                errors.append(
                    "This schedule was deleted from History, so nothing was imported. Restore it from "
                    "History > Recently deleted first."
                )
            else:
                errors.append("Could not save the imported scores, so nothing was imported. Please try again.")

        logger.info("CSV score import: schedule_id=%s applied=%d errors=%d", schedule_id, applied, len(errors))
        return {"applied": applied, "errors": errors}

    def get_game_scores(self, schedule_id: int) -> List[Dict] | None:
        """Return every recorded score for a schedule, ordered by round then court.

        Returns None if the scores can't be read, so the UI can say so rather
        than show the games as unplayed.
        """
        try:
            with self._connect() as conn:
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
            return None

    def get_leaderboard(self) -> List[Dict] | None:
        """Aggregate wins/losses/win-rate per player across every recorded score.

        Games without both scores entered are skipped, as are 0-0 rows (see
        _RESULT_ROW_SQL); a tie (equal scores) counts toward games_played but
        not wins or losses. Returns None if the scores can't be read, so the
        UI can say so rather than show an empty leaderboard.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gs.team1_players, gs.team2_players, gs.team1_score, gs.team2_score "
                    "FROM game_scores gs JOIN schedule_history sh ON sh.id = gs.schedule_id "
                    f"WHERE {_RESULT_ROW_SQL} "  # nosec B608
                    "AND sh.deleted_at IS NULL"
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to read leaderboard data")
            return None

        stats: Dict[str, Dict[str, int]] = {}

        def _player_stats(name: str) -> Dict[str, int]:
            return stats.setdefault(name, {"wins": 0, "losses": 0, "games_played": 0})

        for team1_json, team2_json, team1_score, team2_score in rows:
            try:
                team1 = json.loads(team1_json)
                team2 = json.loads(team2_json)
            except (TypeError, json.JSONDecodeError):
                # One corrupted row must not take down the whole leaderboard -
                # skip it and keep aggregating the rest.
                logger.exception("Skipping unparseable leaderboard row")
                continue

            for player in team1 + team2:
                _player_stats(player)["games_played"] += 1

            if team1_score == team2_score:
                continue

            winners, losers = (team1, team2) if team1_score > team2_score else (team2, team1)
            for player in winners:
                _player_stats(player)["wins"] += 1
            for player in losers:
                _player_stats(player)["losses"] += 1

        leaderboard: List[Dict[str, Any]] = []
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

    def get_all_scored_games(self) -> List[Dict] | None:
        """Return every recorded score across all non-deleted schedules, in the order they were played.

        That is the order of the schedules' creation times, then round and
        court - never the order scores were entered in (recorded_at), which
        entering an old week late, correcting a score or re-importing a sheet
        changes. So ELO replay (#145) gives the same ratings for the same
        results, however and whenever they were entered.

        Returns None if the scores can't be read, so the ELO recompute can
        leave the ratings alone rather than replay no games.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gs.team1_players, gs.team2_players, gs.team1_score, gs.team2_score, gs.recorded_at "
                    "FROM game_scores gs JOIN schedule_history sh ON sh.id = gs.schedule_id "
                    f"WHERE {_RESULT_ROW_SQL} "  # nosec B608
                    "AND sh.deleted_at IS NULL "
                    "ORDER BY sh.timestamp, gs.schedule_id, gs.round_num, gs.court"
                )
                rows = cursor.fetchall()
        except Exception:
            logger.exception("Failed to read all scored games")
            return None

        games = []
        for team1_json, team2_json, team1_score, team2_score, recorded_at in rows:
            try:
                team1_players = json.loads(team1_json)
                team2_players = json.loads(team2_json)
            except (TypeError, json.JSONDecodeError):
                # One corrupted row must not take down ELO recompute for
                # every other schedule - skip it and keep replaying the rest.
                logger.exception("Skipping unparseable scored-game row")
                continue
            games.append(
                {
                    "team1_players": team1_players,
                    "team2_players": team2_players,
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "recorded_at": recorded_at,
                }
            )
        return games

    def _save_weekly_relationships(self, cursor, schedule, week_date):
        """Save weekly partner relationships."""
        partner_rows: list[tuple[str, str, str]] = []

        for round_data in schedule:
            for game in round_data.get("games", []):
                raw_team1, raw_team2, _court = extract_game_teams(game)
                team1 = [str(p) for p in raw_team1]
                team2 = [str(p) for p in raw_team2]

                for team in (team1, team2):
                    if len(team) == 2:
                        p1, p2 = sorted(team)
                        partner_rows.append((week_date, p1, p2))

        if partner_rows:
            cursor.executemany(
                "INSERT OR IGNORE INTO partner_history (week_date, player1, player2) VALUES (?, ?, ?)", partner_rows
            )

    def get_recent_schedules(self, limit: int = 10) -> List[Dict] | None:
        """Get recent (non-deleted) schedules from history, or None if they can't be read."""
        try:
            with self._connect() as conn:
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
            return None

    def count_schedules(self, deleted: bool = False) -> int | None:
        """How many schedules History holds (the soft-deleted ones if deleted), or None if they can't be counted."""
        condition = "deleted_at IS NOT NULL" if deleted else "deleted_at IS NULL"
        query = f"SELECT COUNT(*) FROM schedule_history WHERE {condition}"  # nosec B608
        try:
            with self._connect() as conn:
                return conn.execute(query).fetchone()[0]
        except Exception:
            logger.exception("Failed to count history entries")
            return None

    def get_schedule(self, schedule_id: int) -> Dict | None:
        """Fetch a single history entry by id, or None if it doesn't exist."""
        try:
            with self._connect() as conn:
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
            with self._connect() as conn:
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
            with self._connect() as conn:
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

    def get_deleted_schedules(self, limit: int = 10) -> List[Dict] | None:
        """Get recently soft-deleted schedules, most recently deleted first, or None if they can't be read."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM schedule_history WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT ?",
                    (limit,),
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to read deleted schedules from history")
            return None

    # Chunk size for the IN (?,?,...) clauses below - comfortably under
    # SQLite's bound-variable limit (999 on older builds, 32766 on newer
    # ones) so a large soft-deleted backlog always makes forward progress
    # instead of the whole purge silently failing past that limit.
    _PURGE_CHUNK_SIZE = 500

    def count_purgeable(self, older_than_days: int = 30) -> int:
        """Count soft-deleted schedules that purge_deleted(older_than_days)
        would permanently remove, without removing them - lets the UI show
        the user what an irreversible purge is actually about to do."""
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM schedule_history WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (cutoff,),
                )
                return cursor.fetchone()[0]
        except Exception:
            logger.exception("Failed to count purgeable history entries")
            return 0

    def purge_deleted(self, older_than_days: int = 30) -> int:
        """Permanently remove soft-deleted schedules (and their scores) older than
        the given number of days. Returns the number of schedules purged."""
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM schedule_history WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                    (cutoff,),
                )
                ids = [row[0] for row in cursor.fetchall()]
                purged = 0
                for start in range(0, len(ids), self._PURGE_CHUNK_SIZE):
                    chunk = ids[start : start + self._PURGE_CHUNK_SIZE]
                    placeholders = ",".join("?" * len(chunk))
                    # Re-check deleted_at IS NOT NULL/cutoff at delete time,
                    # not just at the SELECT above - a concurrent
                    # restore_schedule() between the two could otherwise
                    # still get deleted here despite no longer being
                    # soft-deleted. Both deletes share the same live
                    # condition so a schedule's scores and its history row
                    # are purged (or spared) together.
                    # placeholders is only "?,?,...", with the ids bound as params,
                    # so these f-strings carry no input and B608 is suppressed.
                    still_purgeable = (
                        f"SELECT id FROM schedule_history WHERE id IN ({placeholders}) "  # nosec B608
                        "AND deleted_at IS NOT NULL AND deleted_at < ?"
                    )
                    params = (*chunk, cutoff)
                    delete_scores = f"DELETE FROM game_scores WHERE schedule_id IN ({still_purgeable})"  # nosec B608
                    delete_history = f"DELETE FROM schedule_history WHERE id IN ({still_purgeable})"  # nosec B608
                    cursor.execute(delete_scores, params)
                    cursor.execute(delete_history, params)
                    purged += cursor.rowcount
                conn.commit()
            if purged:
                logger.info("History purge: removed %d schedules older than %d days", purged, older_than_days)
            return purged
        except Exception:
            logger.exception("Failed to purge deleted history entries")
            return 0

    def get_weekly_partners(self, weeks_back: int = 4, tz: tzinfo | None = None) -> Dict[str, List] | None:
        """Every partner pair of the schedules in History, keyed by the date each was generated, for the
        weeks_back most recent dates that have any - or None if they can't be read.

        Worked out from the schedules themselves rather than from partner_history,
        which keeps the pairs of deleted and purged schedules and of the rounds a
        replan replaced. Dates are in tz (None: the server's time zone).
        """
        history: Dict[str, set[tuple[str, ...]]] = {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, schedule_data FROM schedule_history "
                    "WHERE deleted_at IS NULL ORDER BY timestamp DESC"
                )
                for schedule_id, timestamp, schedule_json in rows:
                    try:
                        week = stored_time(timestamp, tz).strftime("%Y-%m-%d")
                        if week not in history and len(history) == weeks_back:
                            break  # newest first, so every later row is older still
                        rounds = _games_by_round(json.loads(schedule_json))
                    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
                        # One unreadable row must not hide every other schedule's pairs.
                        logger.warning("Weekly partners: skipping unreadable schedule_id=%s", schedule_id)
                        continue
                    pairs = {
                        team
                        for games in rounds
                        for _court, team1, team2 in games
                        for team in (team1, team2)
                        if len(team) == 2
                    }
                    if pairs:  # a date with none would show as an empty week, and take one of the weeks_back
                        history.setdefault(week, set()).update(pairs)
        except Exception:
            logger.exception("Failed to read weekly partner history")
            return None
        return {week: sorted(pairs) for week, pairs in history.items()}
