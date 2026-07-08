"""Shared helpers for the two schedule/game shapes used throughout the app.

Games appear in two forms depending on where they came from:
- attribute-style `Game` objects straight from the genetic scheduler
  (`game.team1`, `game.team2`, `game.court`)
- dict-style games from JSON-serialized or history-loaded schedules
  (`game["team1"]`, `game["team2"]`, `game["court"]`)

Every reader of a schedule needs to handle both. Before this module existed,
each site re-implemented the same `hasattr(game, "team1")` branch.
"""

from __future__ import annotations

from typing import Any


def extract_game_teams(game: Any) -> tuple[list, list, Any]:
    """Return (team1, team2, court) for either game shape."""
    if hasattr(game, "team1"):
        return game.team1, game.team2, game.court
    return game.get("team1", []), game.get("team2", []), game.get("court", 1)


def games_in_round(round_data: Any) -> list:
    """Return the games list for one round.

    round_data is normally a round dict (`{"games": [...]}`), but some
    callers pass a schedule that is just a flat list of bare games with no
    round wrapper - in that case treat round_data itself as the one game.
    """
    if hasattr(round_data, "get"):
        return round_data.get("games", [])
    return [round_data]
