#!/usr/bin/env python3
"""Player-name validation and advanced player management (substitutions, availability)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


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
