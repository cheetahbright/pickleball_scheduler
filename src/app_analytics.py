#!/usr/bin/env python3
"""Schedule analytics helpers behind the main_app compatibility facade."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

_schedule_analytics_core = import_module_with_fallback("utils.schedule_analytics_core")
_build_pairing_matrices = _schedule_analytics_core.build_pairing_matrices
_calculate_fairness_metrics = _schedule_analytics_core.calculate_fairness_metrics
_serialize_schedule_for_json = _schedule_analytics_core.serialize_schedule_for_json


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


def build_pairing_matrices(schedule: List[Dict], players: List[str]):
    """Return (partner_counts, opponent_counts) NxN matrices using the shared core implementation."""
    return _build_pairing_matrices(schedule, players)


def create_pairing_heatmap(matrix: List[List[int]], players: List[str], title: str, has_plotly: bool, go_module):
    """Create a plotly heatmap for a partner/opponent count matrix.

    Cell-count labels are drawn as per-cell annotations (rather than the
    Heatmap trace's own text, whose textfont color is a single global value)
    so each label can switch between white and black based on that cell's
    own value - a fixed color choice reads fine against the light end of the
    "Blues" colorscale but goes low-contrast on the darkest, highest-count
    cells, or vice versa.
    """
    if not has_plotly or go_module is None or not players:
        return None

    fig = go_module.Figure(
        data=go_module.Heatmap(
            z=matrix,
            x=players,
            y=players,
            colorscale="Blues",
            hovertemplate="%{y} & %{x}: %{z}<extra></extra>",
        )
    )

    flat_values = [value for row in matrix for value in row]
    threshold = (max(flat_values) if flat_values else 0) / 2
    annotations = [
        go_module.layout.Annotation(
            x=players[col],
            y=players[row],
            text=str(value),
            showarrow=False,
            font={"color": "white" if value > threshold else "black"},
        )
        for row, row_values in enumerate(matrix)
        for col, value in enumerate(row_values)
    ]

    fig.update_layout(title=title, xaxis_title="Player", yaxis_title="Player", annotations=annotations)
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
