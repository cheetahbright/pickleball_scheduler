#!/usr/bin/env python3
"""Mathematical Feasibility Analysis for Pickleball Scheduling"""

from __future__ import annotations

from typing import TypedDict


class FeasibilityMinimums(TypedDict):
    min_partner_range: int
    min_games_range: int
    min_courts_range: int
    min_opponents_range: int
    total_theoretical_min: int
    range_0_possible: bool
    games_per_player: float
    sitting_players: int
    players_per_round: int
    total_games: int


class QualityAssessment(TypedDict):
    quality: str
    grade: str
    reason: str
    gap_from_theoretical: int
    theoretical_min: int
    actual_total: int
    critical_perfect: bool
    feasibility_analysis: FeasibilityMinimums


class OptimalParameters(TypedDict):
    population_size: int
    max_generations: int
    max_runtime: float
    convergence_patience: int
    tournament_size: int


class ScheduleFeasibilityAnalyzer:
    """Analyzes mathematical constraints and feasibility for pickleball scheduling."""

    @staticmethod
    def calculate_theoretical_minimums(
        num_players: int, num_courts: int | None = None, num_rounds: int = 8
    ) -> FeasibilityMinimums:
        """Proven lower bounds on each fairness range for this configuration.

        Every number here holds for every possible schedule, so a schedule
        that reaches total_theoretical_min is optimal. Only two things are
        proven: the game slots (rounds x courts x 4) either divide evenly
        among the players or force a games range of 1, and when there are
        fewer slots than players someone can't play at all - a player with no
        game has no partner, opponent or court, while everyone who plays has
        1 partner, 2 opponents and 1 court. Nothing else is ruled out, so
        range_0_possible means "not proven impossible", not "achievable".
        """

        if num_courts is None:
            num_courts = max(1, num_players // 4)

        # Basic calculations
        players_per_round = num_courts * 4
        sitting_players = max(0, num_players - players_per_round)
        total_games = num_rounds * num_courts
        total_player_spots = total_games * 4

        games_per_player: float = total_player_spots / num_players if num_players > 0 else float(num_rounds)
        min_games_range = 0 if num_players <= 0 or total_player_spots % num_players == 0 else 1

        # Fewer slots than players: someone gets no game at all.
        someone_cannot_play = 0 < total_player_spots < num_players
        min_partner_range = 1 if someone_cannot_play else 0
        min_opponents_range = 2 if someone_cannot_play else 0
        min_courts_range = 1 if someone_cannot_play else 0

        total_theoretical_min = min_partner_range + min_games_range + min_courts_range + min_opponents_range

        return {
            "min_partner_range": min_partner_range,
            "min_games_range": min_games_range,
            "min_courts_range": min_courts_range,
            "min_opponents_range": min_opponents_range,
            "total_theoretical_min": total_theoretical_min,
            "range_0_possible": total_theoretical_min == 0,
            "games_per_player": games_per_player,
            "sitting_players": sitting_players,
            "players_per_round": players_per_round,
            "total_games": total_games,
        }

    @staticmethod
    def assess_quality_with_feasibility(
        breakdown: dict[str, int], num_players: int, num_courts: int | None = None, num_rounds: int = 8
    ) -> QualityAssessment:
        """Grade a schedule by how far its total range is above the proven lower bound.

        At the bound (gap 0) the schedule is proven optimal. Above it, the
        grade only says how far above: the bound is not always reachable, so
        a gap does not prove the schedule could be better.
        """

        # Get the proven lower bounds
        feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(num_players, num_courts, num_rounds)

        # Calculate actual total range
        actual_total = (
            breakdown["games_range"]
            + breakdown["partners_range"]
            + breakdown["opponents_range"]
            + breakdown["courts_range"]
        )

        # Calculate gap from the lower bound
        gap = actual_total - feasibility["total_theoretical_min"]

        # Assess critical ranges (Games, Partners, Courts at their lower bounds)
        critical_perfect = (
            breakdown["games_range"] == feasibility["min_games_range"]
            and breakdown["partners_range"] == feasibility["min_partner_range"]
            and breakdown["courts_range"] == feasibility["min_courts_range"]
        )

        # Quality grading based on gap from the lower bound
        if gap <= 0:
            # Every range is at its lower bound, so nothing better exists
            quality = "PERFECT"
            grade = "A+"
            reason = "Reached the proven minimum total range"
        elif gap <= 2:
            # Within 2 of the lower bound
            quality = "VERY_GOOD"
            grade = "B+" if gap == 1 else "B"
            reason = f"{gap} above the proven lower bound"
        elif gap <= 4:
            # Within 4 of the lower bound
            quality = "GOOD"
            grade = "B-" if gap == 3 else "C+"
            reason = f"{gap} above the proven lower bound"
        else:
            # More than 4 above the lower bound
            quality = "NEEDS_IMPROVEMENT"
            grade = "C" if gap <= 6 else "D"
            reason = f"{gap} above the proven lower bound"

        return {
            "quality": quality,
            "grade": grade,
            "reason": reason,
            "gap_from_theoretical": gap,
            "theoretical_min": feasibility["total_theoretical_min"],
            "actual_total": actual_total,
            "critical_perfect": critical_perfect,
            "feasibility_analysis": feasibility,
        }

    @staticmethod
    def get_optimal_parameters(num_players: int) -> OptimalParameters:
        """Get optimal algorithm parameters based on player count and complexity.

        max_generations is a reference figure only: GeneticPickleballScheduler's
        desktop mode, which reads these tiers, has no generation cap.
        """

        # Base parameters
        base_params: OptimalParameters = {
            "population_size": 100,
            "max_generations": 1200,
            "max_runtime": 30.0,
            "convergence_patience": 40,
            "tournament_size": 5,
        }

        # Adjust based on player count and mathematical complexity
        if num_players == 16:
            # 16 players - optimal scenario, focus on achieving Range 0
            params = base_params.copy()
            params.update(
                {
                    "max_generations": 1500,  # More generations for Range 0 pursuit
                    "max_runtime": 40.0,
                    "convergence_patience": 50,
                }
            )
        elif num_players in [13, 14, 15]:
            # Smaller counts with moderate complexity
            params = base_params.copy()
            params.update(
                {
                    "population_size": 120,
                    "max_generations": 1500,
                    "max_runtime": 40.0,
                    "convergence_patience": 45,
                }
            )
        elif num_players in [17, 18]:
            # Larger counts with high complexity
            params = base_params.copy()
            params.update(
                {
                    "population_size": 150,
                    "max_generations": 1800,
                    "max_runtime": 50.0,
                    "convergence_patience": 60,
                }
            )
        elif num_players <= 12:
            # Smaller scenarios - standard parameters
            params = base_params.copy()
        else:
            # Very large scenarios - intensive parameters
            params = base_params.copy()
            params.update(
                {
                    "population_size": 180,
                    "max_generations": 2000,
                    "max_runtime": 60.0,
                    "convergence_patience": 80,
                }
            )

        return params


if __name__ == "__main__":
    # Example usage and testing
    analyzer = ScheduleFeasibilityAnalyzer()

    print("🧮 MATHEMATICAL FEASIBILITY ANALYZER TEST")
    print("=" * 50)

    for players in [13, 14, 15, 16, 17, 18]:
        analysis = analyzer.calculate_theoretical_minimums(players)
        params = analyzer.get_optimal_parameters(players)

        print(f"\n{players} players:")
        print(f"  Proven total range lower bound: {analysis['total_theoretical_min']}")
        print(f"  Range 0 ruled out: {not analysis['range_0_possible']}")
        print(f"  Recommended population: {params['population_size']}")
        print(f"  Recommended runtime: {params['max_runtime']}s")
