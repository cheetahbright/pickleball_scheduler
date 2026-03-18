#!/usr/bin/env python3
"""Stateful managers extracted from the main Streamlit app."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.app_core_functions import validate_player_names
from src.utils.schedule_analytics_core import (
    calculate_fairness_metrics,
    serialize_schedule_for_json,
)


class HistoryManager:
    """Manage schedule history and persistence."""

    def __init__(self):
        self.db_path = Path("data/schedule_history.db")
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
                    settings_data TEXT
                )
            """)

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

            conn.commit()

    def save_schedule(self, schedule: List[Dict], players: List[str], settings: Dict):
        """Save schedule data and weekly relationships."""
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

                current_week = datetime.now().strftime("%Y-%m-%d")
                self._save_weekly_relationships(cursor, schedule, current_week)

                conn.commit()

        except Exception as e:
            st.error(f"Failed to save history: {e}")

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
        """Get recent schedules from history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM schedule_history
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (limit,),
                )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            return []

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
            return {}


class ConfigurationManager:
    """Manage application configuration and constraints."""

    def __init__(self):
        self.config_path = Path("data/app_config.json")
        self.default_names_path = Path("data/default_player_names.json")
        self.config_path.parent.mkdir(exist_ok=True)
        self.default_config = {
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
                    config = json.load(f)
                return self._merge_configs(self.default_config, config)
            return self.default_config.copy()
        except Exception:
            return self.default_config.copy()

    def save_config(self, config: Dict):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
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
