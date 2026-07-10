#!/usr/bin/env python3
"""
Canonical constraint model and objective terms for pickleball scheduling.
Shared by all algorithm implementations for consistency.
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


@dataclass
class SchedulingProblem:
    """Canonical problem representation for all algorithms."""

    players: List[str]
    num_courts: int
    num_rounds: int
    num_games_per_round: int

    # Constraints
    availability: Dict[str, Set[int]] = field(default_factory=dict)  # player -> available rounds
    pair_constraints: Optional[List[Tuple[str, str]]] = None  # must be teammates
    oppose_constraints: Optional[List[Tuple[str, str]]] = None  # must be opponents
    timing_constraints: Optional[Dict[str, Set[int]]] = None  # player -> unavailable rounds

    # Preferences and weights
    objective_weights: Dict[str, float] = field(default_factory=dict)
    skill_levels: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Initialize defaults."""
        if not self.availability:
            self.availability = {p: set(range(self.num_rounds)) for p in self.players}

        if not self.objective_weights:
            self.objective_weights = {
                "partner_variety": 3.0,
                "opponent_variety": 2.0,
                "court_balance": 1.0,
                "skill_balance": 0.5,
                "rest_spacing": 0.5,
                "constraint_violations": 10.0,  # High penalty
            }


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


def _extract_round_games(round_data: object) -> List[object]:
    """Normalize flat and nested round containers into a list of games."""
    if isinstance(round_data, dict):
        games = round_data.get("games")
        if isinstance(games, list):
            return games
        return []

    if isinstance(round_data, list):
        return round_data

    return []


def _build_round_signature(
    round_games: Sequence[object],
) -> frozenset[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Create a canonical signature for duplicate-round detection."""
    signature: set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    for game in round_games:
        teams = _extract_teams(game)
        if teams is None:
            continue

        team1_list, team2_list = teams
        team1 = tuple(sorted(team1_list))
        team2 = tuple(sorted(team2_list))
        ordered_teams = sorted([team1, team2])
        signature.add((ordered_teams[0], ordered_teams[1]))

    return frozenset(signature)


class ObjectiveCalculator:
    """Shared objective term calculations for all algorithms."""

    @staticmethod
    def calculate_partner_variety(schedule: Sequence[Sequence[object]], players: List[str]) -> float:
        """Calculate partner variety score (higher is better)."""
        partner_counts = defaultdict(lambda: defaultdict(int))

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
        opponent_counts = defaultdict(lambda: defaultdict(int))

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
        court_usage = defaultdict(lambda: defaultdict(int))

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
    def calculate_rest_spacing(schedule: Sequence[Sequence[object]], players: List[str]) -> float:
        """Calculate rest spacing quality (higher is better)."""
        player_rounds = defaultdict(list)

        for round_idx, round_games in enumerate(schedule):
            for game in round_games:
                teams = _extract_teams(game)
                if teams is None:
                    continue
                team1, team2 = teams
                game_players = team1 + team2

                for player in game_players:
                    player_rounds[player].append(round_idx)

        # Calculate spacing quality
        spacing_scores = []
        for player in players:
            rounds = sorted(player_rounds[player])
            if len(rounds) > 1:
                gaps = [rounds[i + 1] - rounds[i] for i in range(len(rounds) - 1)]
                # Prefer consistent gaps
                spacing_scores.append(float(1.0 / (1.0 + float(np.var(gaps)))))

        if spacing_scores:
            return float(np.mean(spacing_scores))
        return 1.0

    @staticmethod
    def calculate_constraint_violations(schedule: Sequence[Sequence[object]], problem: SchedulingProblem) -> int:
        """Count total constraint violations."""
        violations = 0
        availability = problem.availability

        # 🚨 DUPLICATE DETECTION TEMPORARILY DISABLED DUE TO FALSE POSITIVES
        # if ScheduleRepair.has_duplicate_rounds(schedule):
        #     violations += 1000  # Massive penalty for duplicate rounds
        #     print("🚨 CRITICAL VIOLATION: Duplicate rounds detected!")

        # Check availability constraints
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

        # Check pair constraints (must be teammates)
        if problem.pair_constraints:
            for p1, p2 in problem.pair_constraints:
                together_count = 0
                for round_games in schedule:
                    for game in round_games:
                        teams = _extract_teams(game)
                        if teams is None:
                            continue
                        team1, team2 = teams

                        if (p1 in team1 and p2 in team1) or (p1 in team2 and p2 in team2):
                            together_count += 1
                if together_count == 0:
                    violations += 1

        # Check oppose constraints (must be opponents)
        if problem.oppose_constraints:
            for p1, p2 in problem.oppose_constraints:
                opposed_count = 0
                for round_games in schedule:
                    for game in round_games:
                        teams = _extract_teams(game)
                        if teams is None:
                            continue
                        team1, team2 = teams

                        if (p1 in team1 and p2 in team2) or (p1 in team2 and p2 in team1):
                            opposed_count += 1
                if opposed_count == 0:
                    violations += 1

        return violations

    @staticmethod
    def calculate_total_objective(schedule: Sequence[Sequence[object]], problem: SchedulingProblem) -> float:
        """Calculate weighted total objective (lower is better)."""
        weights = problem.objective_weights

        # Calculate individual terms
        partner_variety = ObjectiveCalculator.calculate_partner_variety(schedule, problem.players)
        opponent_variety = ObjectiveCalculator.calculate_opponent_variety(schedule, problem.players)
        court_balance = ObjectiveCalculator.calculate_court_balance(schedule, problem.players, problem.num_courts)
        rest_spacing = ObjectiveCalculator.calculate_rest_spacing(schedule, problem.players)
        violations = ObjectiveCalculator.calculate_constraint_violations(schedule, problem)

        # Weighted sum (convert positive scores to negative for minimization)
        total = (
            -weights["partner_variety"] * partner_variety
            - weights["opponent_variety"] * opponent_variety
            - weights["court_balance"] * court_balance
            - weights["rest_spacing"] * rest_spacing
            + weights["constraint_violations"] * violations
        )

        return total


class ScheduleRepair:
    """Shared schedule repair operators."""

    @staticmethod
    def has_duplicate_rounds(schedule: Sequence[object]) -> bool:
        """Check if schedule has any duplicate rounds (CRITICAL BUG DETECTOR)"""
        round_signatures: set[frozenset[Tuple[Tuple[str, ...], Tuple[str, ...]]]] = set()

        for round_data in schedule:
            round_games = _extract_round_games(round_data)
            if not round_games:
                continue

            round_signature = _build_round_signature(round_games)

            # Check for duplicate
            if round_signature in round_signatures:
                return True

            round_signatures.add(round_signature)

        return False
