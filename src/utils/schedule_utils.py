"""Schedule utilities: normalization helpers for GUI and reporting."""

from __future__ import annotations

from typing import Any, Dict, List


def normalize_schedule(schedule: Any) -> Dict[int, List[Dict[str, Any]]]:
    """Normalize various schedule shapes to a consistent dict[int, list[dict]].

    Output shape:
      {
        round_number: [
          {"players": [p1, p2, p3, p4], "court": int},
          ...
        ],
        ...
      }

    Accepted inputs:
      - list of round dicts with key "games" containing Game-like objects (team1/team2 attrs)
      - dict of round -> list of games where games are dicts with "players" or team1/team2
      - list of games (fallback: treated as single round)
    """
    # Already normalized dict[int, list]
    if isinstance(schedule, dict):
        normalized: Dict[int, List[Dict[str, Any]]] = {}
        for rnd, games in schedule.items():
            norm_games: List[Dict[str, Any]] = []
            for i, g in enumerate(games, start=1):
                # Dict with players
                if isinstance(g, dict):
                    players = g.get("players")
                    if not players and "team1" in g and "team2" in g:
                        team1 = g.get("team1", [])
                        team2 = g.get("team2", [])
                        players = list(team1[:2] + team2[:2])
                    court = g.get("court", i)
                else:
                    # Object with attributes (team1/team2/court)
                    team1 = getattr(g, "team1", [])
                    team2 = getattr(g, "team2", [])
                    players = list(team1[:2] + team2[:2])
                    court = getattr(g, "court", i)
                if players and len(players) == 4:
                    norm_games.append({"players": players, "court": court})
            # Ensure integer-ish round keys sorted later
            try:
                rnd_key = int(rnd)
            except (TypeError, ValueError):
                rnd_key = rnd
            normalized[rnd_key] = norm_games
        return normalized

    # list input – either [round_info,...] or [game,...]
    if isinstance(schedule, list):
        normalized = {}
        if schedule and isinstance(schedule[0], dict) and "games" in schedule[0]:
            # List of round dicts
            for idx, round_info in enumerate(schedule, start=1):
                games = round_info.get("games", [])
                norm_games: List[Dict[str, Any]] = []
                for i, g in enumerate(games, start=1):
                    if isinstance(g, dict):
                        players = g.get("players")
                        if not players and "team1" in g and "team2" in g:
                            players = list(g.get("team1", [])[:2] + g.get("team2", [])[:2])
                        court = g.get("court", i)
                    else:
                        team1 = getattr(g, "team1", [])
                        team2 = getattr(g, "team2", [])
                        players = list(team1[:2] + team2[:2])
                        court = getattr(g, "court", i)
                    if players and len(players) == 4:
                        norm_games.append({"players": players, "court": court})
                normalized[idx] = norm_games
            return normalized
        else:
            # Treat as a single round composed of raw games/tuples
            norm_games = []
            for i, g in enumerate(schedule, start=1):
                players = None
                court = i
                if isinstance(g, dict):
                    players = g.get("players")
                    court = g.get("court", i)
                elif isinstance(g, (list, tuple)) and len(g) == 4:
                    players = list(g)
                else:
                    team1 = getattr(g, "team1", [])
                    team2 = getattr(g, "team2", [])
                    if team1 or team2:
                        players = list(team1[:2] + team2[:2])
                        court = getattr(g, "court", i)
                if players and len(players) == 4:
                    norm_games.append({"players": players, "court": court})
            return {1: norm_games}

    # Fallback: empty
    return {}


def validate_schedule_format(schedule: Any) -> bool:
    """Validate schedule format for consistency and correctness.

    Args:
        schedule: Schedule data in various formats

    Returns:
        bool: True if schedule format is valid, False otherwise

    Raises:
        ValueError: If schedule format is invalid with specific error message
    """
    if not schedule:
        return True  # Empty schedule is valid

    # Check for list format (list of rounds or games)
    if isinstance(schedule, list):
        if not schedule:
            return True

        # Check if it's a list of round dictionaries
        first_item = schedule[0]
        if isinstance(first_item, dict) and "round" in first_item and "games" in first_item:
            # Validate round-based format
            for round_data in schedule:
                if not isinstance(round_data, dict):
                    raise ValueError("Each round must be a dictionary")
                if "round" not in round_data or "games" not in round_data:
                    raise ValueError("Each round must have 'round' and 'games' keys")
                if not isinstance(round_data["games"], list):
                    raise ValueError("Games must be a list")

                # Validate each game in the round
                for game in round_data["games"]:
                    if not isinstance(game, dict):
                        raise ValueError("Each game must be a dictionary")
                    if not ("team1" in game and "team2" in game) and "players" not in game:
                        raise ValueError("Each game must have either team1/team2 or players")
        else:
            # Assume it's a list of games
            for i, game in enumerate(schedule):
                if hasattr(game, "team1") and hasattr(game, "team2"):
                    # Game object format - check attributes exist
                    continue
                elif isinstance(game, dict):
                    # Dictionary game format
                    if not ("team1" in game and "team2" in game) and "players" not in game:
                        raise ValueError(f"Game {i} must have either team1/team2 or players")
                else:
                    raise ValueError(f"Game {i} must be a dictionary or game object")

    # Check for dictionary format (round number -> games)
    elif isinstance(schedule, dict):
        for round_num, games in schedule.items():
            if not isinstance(games, list):
                raise ValueError(f"Round {round_num} must contain a list of games")

            for i, game in enumerate(games):
                if isinstance(game, dict):
                    if not ("team1" in game and "team2" in game) and "players" not in game:
                        raise ValueError(f"Game {i} in round {round_num} must have either team1/team2 or players")
                elif not hasattr(game, "team1") or not hasattr(game, "team2"):
                    raise ValueError(f"Game {i} in round {round_num} must have team1 and team2 attributes")

    else:
        raise ValueError("Schedule must be a list or dictionary")

    return True
