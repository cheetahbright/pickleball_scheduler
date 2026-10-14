#!/usr/bin/env python3
"""Advanced player management (substitutions, availability)."""

from __future__ import annotations


class PlayerManager:
    """Advanced player management with substitutions and availability."""

    @staticmethod
    def create_substitution_interface(st_module):
        """Create UI for player substitutions."""
        st_module.subheader("🔄 Player Substitutions")
        st_module.info("Configure players for different time periods (e.g., early vs late players)")

        col1, col2 = st_module.columns(2)

        with col1:
            st_module.markdown("**First Half Players**")
            first_half = st_module.text_area(
                "Players available for early rounds:",
                value=st_module.session_state.get("first_half_players", ""),
                height=100,
                key="first_half_input",
            )
            st_module.session_state.first_half_players = first_half

        with col2:
            st_module.markdown("**Second Half Players**")
            second_half = st_module.text_area(
                "Players available for later rounds:",
                value=st_module.session_state.get("second_half_players", ""),
                height=100,
                key="second_half_input",
            )
            st_module.session_state.second_half_players = second_half

        sub_round = st_module.number_input(
            "Substitution round (when to switch players):",
            min_value=1,
            max_value=10,
            value=st_module.session_state.get("substitution_round", 4),
        )
        st_module.session_state.substitution_round = sub_round

        return first_half, second_half, sub_round
