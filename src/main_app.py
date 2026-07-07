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
    from src.algorithms.genetic_scheduler import GeneticPickleballScheduler
except ImportError:
    from algorithms.genetic_scheduler import GeneticPickleballScheduler

try:
    from src.app_managers import ConfigurationManager, HistoryManager, PlayerManager
except ImportError:
    from app_managers import ConfigurationManager, HistoryManager, PlayerManager

try:
    from src.app_schedule_helpers import display_enhanced_schedule as _display_enhanced_schedule
    from src.app_schedule_helpers import schedule_to_csv as _schedule_to_csv
    from src.app_schedule_helpers import validate_schedule_integrity as _validate_schedule_integrity
except ImportError:
    from app_schedule_helpers import display_enhanced_schedule as _display_enhanced_schedule
    from app_schedule_helpers import schedule_to_csv as _schedule_to_csv
    from app_schedule_helpers import validate_schedule_integrity as _validate_schedule_integrity

try:
    from src.app_tabs import render_analytics_tab as _analytics_tab
    from src.app_tabs import render_configuration_tab as _configuration_tab
    from src.app_tabs import render_history_tab as _history_tab
    from src.app_tabs import render_player_management_tab as _player_management_tab
except ImportError:
    from app_tabs import render_analytics_tab as _analytics_tab
    from app_tabs import render_configuration_tab as _configuration_tab
    from app_tabs import render_history_tab as _history_tab
    from app_tabs import render_player_management_tab as _player_management_tab

try:
    from src.app_scheduler_flow import render_enhanced_scheduler_page as _enhanced_scheduler_page
    from src.app_scheduler_flow import render_main_scheduler_tab as _main_scheduler_tab
except ImportError:
    from app_scheduler_flow import render_enhanced_scheduler_page as _enhanced_scheduler_page
    from app_scheduler_flow import render_main_scheduler_tab as _main_scheduler_tab

try:
    from src.app_analytics import calculate_fairness_metrics as _calculate_fairness_metrics
    from src.app_analytics import create_fairness_visualization as _create_fairness_visualization
    from src.app_analytics import create_player_stats_chart as _create_player_stats_chart
    from src.app_analytics import serialize_schedule_for_json as _serialize_schedule_for_json
except ImportError:
    from app_analytics import calculate_fairness_metrics as _calculate_fairness_metrics
    from app_analytics import create_fairness_visualization as _create_fairness_visualization
    from app_analytics import create_player_stats_chart as _create_player_stats_chart
    from app_analytics import serialize_schedule_for_json as _serialize_schedule_for_json

try:
    from src.simple_auth import logout, simple_auth
except ImportError:
    from simple_auth import logout, simple_auth

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


def validate_schedule_integrity(schedule, all_players=None):
    """Comprehensive validation of schedule integrity."""
    return _validate_schedule_integrity(schedule, all_players)


def display_enhanced_schedule(schedule, all_players=None):
    """Display schedule with enhanced formatting and error checking."""
    return _display_enhanced_schedule(schedule, st, pd, all_players)


def schedule_to_csv(schedule):
    """Convert schedule to CSV format."""
    return _schedule_to_csv(schedule)


def main():
    """Main application"""

    # Page config
    st.set_page_config(page_title="Pickleball Scheduler", page_icon="🎾", layout="wide")

    # Check authentication
    if not simple_auth():
        return

    # Main scheduler
    enhanced_scheduler_page()


if __name__ == "__main__":
    main()


__all__ = [
    "ConfigurationManager",
    "HistoryManager",
    "PlayerManager",
    "ScheduleAnalytics",
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
