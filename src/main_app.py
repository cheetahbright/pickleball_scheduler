#!/usr/bin/env python3
"""
Compatibility facade for the Streamlit app entrypoint.

The app logic now lives in focused modules (`app_managers`, `app_tabs`,
`app_schedule_helpers`, `app_scheduler_flow`, and `app_analytics`), but this
module intentionally keeps the legacy `main_app` import surface stable for the
Streamlit launcher, older tests, and downstream scripts.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _register_main_app_aliases() -> None:
    """Expose only the opposite import alias without clobbering existing modules."""
    current_module = sys.modules[__name__]
    if __name__ == "src.main_app":
        sys.modules.setdefault("main_app", current_module)
    elif __name__ == "main_app":
        sys.modules.setdefault("src.main_app", current_module)


_register_main_app_aliases()

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

GeneticPickleballScheduler = import_module_with_fallback("algorithms.genetic_scheduler").GeneticPickleballScheduler

_app_managers = import_module_with_fallback("app_managers")
ConfigurationManager = _app_managers.ConfigurationManager
EloRatingManager = _app_managers.EloRatingManager
HistoryManager = _app_managers.HistoryManager
PlayerManager = _app_managers.PlayerManager
SkillRatingManager = _app_managers.SkillRatingManager

_app_schedule_helpers = import_module_with_fallback("app_schedule_helpers")
_display_enhanced_schedule = _app_schedule_helpers.display_enhanced_schedule
_schedule_to_csv = _app_schedule_helpers.schedule_to_csv
_validate_schedule_integrity = _app_schedule_helpers.validate_schedule_integrity

_app_tabs = import_module_with_fallback("app_tabs")
_analytics_tab = _app_tabs.render_analytics_tab
_configuration_tab = _app_tabs.render_configuration_tab
_history_tab = _app_tabs.render_history_tab
_player_management_tab = _app_tabs.render_player_management_tab
_stress_test_tab = _app_tabs.render_stress_test_tab
_leaderboard_tab = _app_tabs.render_leaderboard_tab

_app_scheduler_flow = import_module_with_fallback("app_scheduler_flow")
_enhanced_scheduler_page = _app_scheduler_flow.render_enhanced_scheduler_page
_main_scheduler_tab = _app_scheduler_flow.render_main_scheduler_tab

_app_analytics = import_module_with_fallback("app_analytics")
_build_pairing_matrices = _app_analytics.build_pairing_matrices
_calculate_fairness_metrics = _app_analytics.calculate_fairness_metrics
_create_fairness_visualization = _app_analytics.create_fairness_visualization
_create_pairing_heatmap = _app_analytics.create_pairing_heatmap
_create_player_stats_chart = _app_analytics.create_player_stats_chart
_serialize_schedule_for_json = _app_analytics.serialize_schedule_for_json

_simple_auth = import_module_with_fallback("simple_auth")
logout = _simple_auth.logout
simple_auth = _simple_auth.simple_auth

setup_logging = import_module_with_fallback("utils.logging_config").setup_logging
inject_mobile_css = import_module_with_fallback("mobile_styles").inject_mobile_css
inject_pwa_manifest = import_module_with_fallback("pwa").inject_pwa_manifest

# Try to import plotly for visualizations
go = None

try:
    import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class ScheduleAnalytics:
    """Public analytics facade preserved for `main_app` compatibility."""

    @staticmethod
    def serialize_schedule_for_json(schedule):
        """Convert schedule with Game objects to JSON-serializable format"""
        return _serialize_schedule_for_json(schedule)

    @staticmethod
    def calculate_fairness_metrics(
        schedule: List[Dict],
        num_players: Optional[int] = None,
        num_rounds: Optional[int] = None,
        num_courts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics normalized by mathematical optimality"""
        return _calculate_fairness_metrics(
            schedule,
            num_players=num_players,
            num_rounds=num_rounds,
            num_courts=num_courts,
        )

    @staticmethod
    def compute_fairness_metrics(
        schedule: List[Dict],
        num_players: Optional[int] = None,
        num_rounds: Optional[int] = None,
        num_courts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for older tests and scripts."""
        return ScheduleAnalytics.calculate_fairness_metrics(
            schedule,
            num_players=num_players,
            num_rounds=num_rounds,
            num_courts=num_courts,
        )

    @staticmethod
    def create_fairness_visualization(metrics: Dict[str, Any]):
        """Create plotly visualization of fairness metrics"""
        return _create_fairness_visualization(metrics, HAS_PLOTLY, go)

    @staticmethod
    def create_player_stats_chart(metrics: Dict[str, Any]):
        """Create player statistics chart"""
        return _create_player_stats_chart(metrics, HAS_PLOTLY, go)

    @staticmethod
    def build_pairing_matrices(schedule: List[Dict], players: List[str]):
        """Return (partner_counts, opponent_counts) NxN matrices for the given player order."""
        return _build_pairing_matrices(schedule, players)

    @staticmethod
    def create_pairing_heatmap(matrix: List[List[int]], players: List[str], title: str):
        """Create a plotly heatmap for a partner/opponent count matrix."""
        return _create_pairing_heatmap(matrix, players, title, HAS_PLOTLY, go)

    @staticmethod
    def generate_radar_chart_data(metrics: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Return a simple radar-chart payload for compatibility helpers."""
        categories = [
            "Games",
            "Partners",
            "Opponents",
            "Courts",
        ]
        variance_keys = [
            ("games_balance", "games_variance"),
            ("partners_balance", "partners_variance"),
            ("opponents_balance", "opponents_variance"),
            ("courts_balance", "courts_variance"),
        ]
        values: List[float] = []

        for nested_key, flat_key in variance_keys:
            variance = 1.0
            nested_metric = metrics.get(nested_key)
            if isinstance(nested_metric, dict):
                raw_variance = nested_metric.get("variance", 1.0)
                if isinstance(raw_variance, (int, float)):
                    variance = float(raw_variance)
            else:
                raw_variance = metrics.get(flat_key, 1.0)
                if isinstance(raw_variance, (int, float)):
                    variance = float(raw_variance)

            values.append(max(0.0, 1.0 - variance))

        return {"categories": categories, "values": values}


def enhanced_scheduler_page():
    """Enhanced scheduler page with full features"""
    return _enhanced_scheduler_page(
        st,
        HistoryManager,
        ConfigurationManager,
        logout,
        main_scheduler_tab,
        analytics_tab,
        history_tab,
        configuration_tab,
        player_management_tab,
        stress_test_tab_fn=stress_test_tab,
        leaderboard_tab_fn=leaderboard_tab,
        skill_manager_cls=SkillRatingManager,
        elo_manager_cls=EloRatingManager,
    )


def main_scheduler_tab():
    """Enhanced main scheduling interface with all features"""
    return _main_scheduler_tab(
        st,
        datetime,
        json,
        GeneticPickleballScheduler,
        ScheduleAnalytics,
        validate_schedule_integrity,
        display_enhanced_schedule,
        schedule_to_csv,
    )


def analytics_tab():
    """Analytics and visualization tab"""
    return _analytics_tab(st, pd, ScheduleAnalytics, HAS_PLOTLY)


def history_tab():
    """Schedule history tab"""
    return _history_tab(st, pd, json)


def configuration_tab():
    """Configuration and constraints tab"""
    return _configuration_tab(st)


def player_management_tab():
    """Advanced player management tab"""
    return _player_management_tab(st, PlayerManager)


def stress_test_tab():
    """Stress-test sweep across player/round/constraint combinations"""
    return _stress_test_tab(st, GeneticPickleballScheduler, validate_schedule_integrity, pd)


def leaderboard_tab():
    """Score entry and all-time win/loss standings"""
    return _leaderboard_tab(st, pd)


def validate_schedule_integrity(schedule, all_players=None):
    """Comprehensive validation of schedule integrity."""
    return _validate_schedule_integrity(schedule, all_players)


def display_enhanced_schedule(schedule, all_players=None, round_times=None):
    """Display schedule with enhanced formatting and error checking."""
    return _display_enhanced_schedule(schedule, st, pd, all_players, round_times)


def schedule_to_csv(schedule):
    """Convert schedule to CSV format."""
    return _schedule_to_csv(schedule)


def main():
    """Main application"""

    # setup_logging is idempotent (no-op if handlers already exist), which
    # matters because Streamlit re-executes this whole script on every rerun.
    setup_logging()

    # Page config
    st.set_page_config(page_title="Pickleball Scheduler", page_icon="🎾", layout="wide")
    inject_mobile_css(st)
    inject_pwa_manifest(st)

    # Check authentication
    if not simple_auth():
        return

    # Main scheduler
    enhanced_scheduler_page()


if __name__ == "__main__":
    main()


__all__ = [
    "ConfigurationManager",
    "EloRatingManager",
    "HistoryManager",
    "PlayerManager",
    "ScheduleAnalytics",
    "SkillRatingManager",
    "analytics_tab",
    "configuration_tab",
    "display_enhanced_schedule",
    "enhanced_scheduler_page",
    "history_tab",
    "main",
    "main_scheduler_tab",
    "player_management_tab",
    "schedule_to_csv",
    "validate_schedule_integrity",
]
