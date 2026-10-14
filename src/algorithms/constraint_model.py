#!/usr/bin/env python3
"""
Benchmark-only scoring helpers for scripts/algorithm_benchmark.py.

No scheduler optimizes with these: the live objective is scheduler_metrics
plus GeneticPickleballScheduler._fitness, and the app's do-not-pair /
do-not-oppose rules are enforced by the arrangement pool
(arrangement_generator.py).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

import numpy as np

PlayerName = str
TupleGame = Tuple[PlayerName, PlayerName, PlayerName, PlayerName]


@runtime_checkable
class TeamGameLike(Protocol):
    """Minimal protocol for game objects with team attributes."""

    team1: Iterable[object]
    team2: Iterable[object]


@dataclass(frozen=True, slots=True)
class SchedulingProblem:
    """The problem shape the benchmark scores a schedule against."""

    players: List[str]
    num_courts: int
    num_rounds: int
    num_games_per_round: int

    availability: Dict[str, Set[int]] = field(default_factory=dict)  # player -> available rounds

    def __post_init__(self):
        """Initialize defaults."""
        if not self.availability:
            object.__setattr__(self, "availability", {p: set(range(self.num_rounds)) for p in self.players})


def _coerce_team(players: object) -> List[str]:
    """Convert a supported team representation to a normalized string list."""
    if isinstance(players, (list, tuple, set)):
        return [str(player) for player in players]
    return []


def _extract_teams(game: object) -> Optional[Tuple[List[str], List[str]]]:
    """Extract two teams from tuple, dict, or object-style game data."""
    if isinstance(game, tuple) and len(game) == 4:
        p1, p2, p3, p4 = game
        return [str(p1), str(p2)], [str(p3), str(p4)]

    if isinstance(game, dict):
        return _coerce_team(game.get("team1", [])), _coerce_team(game.get("team2", []))

    team1 = getattr(game, "team1", None)
    team2 = getattr(game, "team2", None)
    if team1 is not None and team2 is not None:
        typed_game = game
        if isinstance(typed_game, TeamGameLike):
            return _coerce_team(typed_game.team1), _coerce_team(typed_game.team2)

    return None


class ObjectiveCalculator:
    """Scoring terms the benchmark reports for each algorithm's schedule."""

    @staticmethod
    def calculate_partner_variety(schedule: Sequence[Sequence[object]], players: List[str]) -> float:
        """Calculate partner variety score (higher is better)."""
        partner_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for round_games in schedule:
            for game in round_games:
                teams = _extract_teams(game)
                if teams is None:
                    continue
                team1, team2 = teams

                # Count partnerships
                for i, p1 in enumerate(team1):
                    for p2 in team1[i + 1 :]:
                        partner_counts[p1][p2] += 1
                        partner_counts[p2][p1] += 1

                for i, p1 in enumerate(team2):
                    for p2 in team2[i + 1 :]:
                        partner_counts[p1][p2] += 1
                        partner_counts[p2][p1] += 1

        # Calculate variety (lower variance is better)
        all_counts = []
        for p1 in players:
            for p2 in players:
                if p1 < p2:
                    count = partner_counts[p1].get(p2, 0)
                    all_counts.append(count)

        if all_counts:
            return float(1.0 / (1.0 + float(np.var(all_counts))))  # Normalize to 0-1
        return 1.0

    @staticmethod
    def calculate_opponent_variety(schedule: Sequence[Sequence[object]], players: List[str]) -> float:
        """Calculate opponent variety score (higher is better)."""
        opponent_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for round_games in schedule:
            for game in round_games:
                teams = _extract_teams(game)
                if teams is None:
                    continue
                team1, team2 = teams

                # Count oppositions
                for p1 in team1:
                    for p2 in team2:
                        opponent_counts[p1][p2] += 1
                        opponent_counts[p2][p1] += 1

        # Calculate variety
        all_counts = []
        for p1 in players:
            for p2 in players:
                if p1 < p2:
                    count = opponent_counts[p1].get(p2, 0)
                    all_counts.append(count)

        if all_counts:
            return float(1.0 / (1.0 + float(np.var(all_counts))))
        return 1.0

    @staticmethod
    def calculate_court_balance(schedule: Sequence[Sequence[object]], players: List[str], num_courts: int) -> float:
        """Calculate court usage balance (higher is better)."""
        court_usage: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for round_games in schedule:
            for game_idx, game in enumerate(round_games):
                teams = _extract_teams(game)
                if teams is None:
                    continue
                team1, team2 = teams
                game_players = team1 + team2
                court = game_idx
                if isinstance(game, dict):
                    court_value = game.get("court", game_idx + 1)
                    if isinstance(court_value, (int, float)):
                        court = max(int(court_value) - 1, 0)

                for player in game_players:
                    court_usage[player][court] += 1

        # Calculate balance
        variances = []
        for player in players:
            usage = [court_usage[player].get(c, 0) for c in range(num_courts)]
            if sum(usage) > 0:
                variances.append(float(np.var(usage)))

        if variances:
            return float(1.0 / (1.0 + float(np.mean(variances))))
        return 1.0

    @staticmethod
    def calculate_constraint_violations(schedule: Sequence[Sequence[object]], problem: SchedulingProblem) -> int:
        """Count each appearance of a player in a round outside their availability."""
        violations = 0
        availability = problem.availability

        for round_idx, round_games in enumerate(schedule):
            for game in round_games:
                teams = _extract_teams(game)
                if teams is None:
                    continue
                team1, team2 = teams
                game_players = team1 + team2

                for player in game_players:
                    if round_idx not in availability.get(player, set(range(problem.num_rounds))):
                        violations += 1

        return violations
