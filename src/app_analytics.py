#!/usr/bin/env python3
"""Schedule analytics helpers behind the main_app compatibility facade."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from src.utils.schedule_analytics_core import calculate_fairness_metrics as _calculate_fairness_metrics
    from src.utils.schedule_analytics_core import serialize_schedule_for_json as _serialize_schedule_for_json
except ImportError:
    from utils.schedule_analytics_core import calculate_fairness_metrics as _calculate_fairness_metrics
    from utils.schedule_analytics_core import serialize_schedule_for_json as _serialize_schedule_for_json


def serialize_schedule_for_json(schedule):
    """Convert schedule data into a JSON-serializable structure."""
    return _serialize_schedule_for_json(schedule)


def calculate_fairness_metrics(
    schedule: List[Dict],
    num_players: Optional[int] = None,
    num_rounds: Optional[int] = None,
    num_courts: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate fairness metrics using the shared core implementation."""
    return _calculate_fairness_metrics(
        schedule,
        num_players=num_players,
        num_rounds=num_rounds,
        num_courts=num_courts,
    )


def create_fairness_visualization(metrics: Dict[str, Any], has_plotly: bool, go_module):
    """Create the radar-chart fairness visualization."""
    if not has_plotly or go_module is None or not metrics:
        return None

    categories = [
        "Games Range",
        "Partners Range",
        "Opponents Range",
        "Courts Range",
    ]
    values = [
        10 - metrics.get("games_range", 0),
        10 - metrics.get("partners_range", 0),
        10 - metrics.get("opponents_range", 0),
        10 - metrics.get("courts_range", 0),
    ]

    fig = go_module.Figure(
        data=go_module.Scatterpolar(r=values, theta=categories, fill="toself", name="Fairness Score")
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Schedule Fairness Analysis",
    )

    return fig


def create_player_stats_chart(metrics: Dict[str, Any], has_plotly: bool, go_module):
    """Create the grouped bar chart for player schedule statistics."""
    if not has_plotly or go_module is None or not metrics.get("player_stats"):
        return None

    player_stats = metrics["player_stats"]
    players = list(player_stats.keys())
    games = [stats["games_played"] for stats in player_stats.values()]
    partners = [len(stats["partners"]) for stats in player_stats.values()]
    opponents = [len(stats["opponents"]) for stats in player_stats.values()]

    fig = go_module.Figure()
    fig.add_trace(go_module.Bar(name="Games Played", x=players, y=games))
    fig.add_trace(go_module.Bar(name="Unique Partners", x=players, y=partners))
    fig.add_trace(go_module.Bar(name="Unique Opponents", x=players, y=opponents))

    fig.update_layout(
        barmode="group",
        title="Player Statistics",
        xaxis_title="Players",
        yaxis_title="Count",
    )

    return fig
