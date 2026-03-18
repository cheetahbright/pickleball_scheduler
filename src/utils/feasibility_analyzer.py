#!/usr/bin/env python3
"""Mathematical Feasibility Analysis for Pickleball Scheduling"""


class ScheduleFeasibilityAnalyzer:
    """Analyzes mathematical constraints and feasibility for pickleball scheduling."""

    @staticmethod
    def calculate_theoretical_minimums(num_players, num_courts=None, num_rounds=8):
        """Calculate realistic theoretical minimum ranges for given configuration."""

        if num_courts is None:
            num_courts = max(1, num_players // 4)

        # Basic calculations
        players_per_round = num_courts * 4
        sitting_players = max(0, num_players - players_per_round)
        total_games = num_rounds * num_courts

        # Games range analysis - most fundamental constraint
        if sitting_players == 0:
            # No sitting - everyone plays same number of games
            min_games_range = 0
            games_per_player = num_rounds
        else:
            # With sitting - games can't be perfectly equal
            total_player_spots = total_games * 4
            avg_games = total_player_spots / num_players
            min_games_range = 1 if avg_games != int(avg_games) else 0
            games_per_player = avg_games

        # Partner range analysis - realistic calculation
        # In practice, partner ranges are typically 0-2 even with constraints
        if num_players <= 4:
            min_partner_range = 0  # Too few players for range
        elif sitting_players == 0 and num_players % 4 == 0:
            min_partner_range = 0  # Perfect pairing possible
        elif num_players >= 16:
            min_partner_range = 1  # Larger groups have slight imbalances
        else:
            min_partner_range = 1  # Small imbalances likely

        # Courts range analysis
        if sitting_players == 0:
            min_courts_range = 0  # Everyone plays, court balance achievable
        elif num_players > 16:
            min_courts_range = 1  # Large groups create rotation imbalances
        else:
            min_courts_range = 0  # Small groups can balance courts well

        # Opponents range - usually achievable at 0-1
        if num_players <= 8:
            min_opponents_range = 0  # Small groups balance easily
        else:
            min_opponents_range = 0  # Good algorithms can balance this

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
    def assess_quality_with_feasibility(breakdown, num_players, num_courts=None, num_rounds=8):
        """Assess schedule quality relative to mathematical feasibility."""

        # Get theoretical minimums
        feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(num_players, num_courts, num_rounds)

        # Calculate actual total range
        actual_total = (
            breakdown["games_range"]
            + breakdown["partners_range"]
            + breakdown["opponents_range"]
            + breakdown["courts_range"]
        )

        # Calculate gap from theoretical minimum
        gap = actual_total - feasibility["total_theoretical_min"]

        # Assess critical ranges (Games, Partners, Courts must be optimal)
        critical_perfect = (
            breakdown["games_range"] == feasibility["min_games_range"]
            and breakdown["partners_range"] == feasibility["min_partner_range"]
            and breakdown["courts_range"] == feasibility["min_courts_range"]
        )

        # Quality grading based on gap from theoretical minimum
        if gap <= 0:
            # At or better than theoretical minimum
            if critical_perfect and breakdown["opponents_range"] == 0:
                quality = "PERFECT"
                grade = "A+"
                reason = "Achieved theoretical minimum with perfect ranges"
            elif critical_perfect:
                quality = "EXCELLENT"
                grade = "A"
                reason = "Critical ranges optimal, opponents minimal"
            else:
                quality = "VERY_GOOD"
                grade = "B+"
                reason = "At theoretical minimum but suboptimal distribution"
        elif gap <= 2:
            # Within 2 of theoretical minimum
            quality = "VERY_GOOD"
            grade = "B+" if gap == 1 else "B"
            reason = f"Close to optimal (gap: {gap})"
        elif gap <= 4:
            # Within 4 of theoretical minimum
            quality = "GOOD"
            grade = "B-" if gap == 3 else "C+"
            reason = f"Reasonable performance (gap: {gap})"
        else:
            # More than 4 above theoretical minimum
            quality = "NEEDS_IMPROVEMENT"
            grade = "C" if gap <= 6 else "D"
            reason = f"Significant gap from optimal (gap: {gap})"

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
    def get_optimal_parameters(num_players):
        """Get optimal algorithm parameters based on player count and complexity."""

        # Base parameters
        base_params = {
            "population_size": 100,
            "max_generations": 1200,
            "max_runtime": 30.0,
            "convergence_patience": 40,
            "elite_size": 10,
            "tournament_size": 5,
            "mutation_rate": 0.15,
            "crossover_rate": 0.85,
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

    @staticmethod
    def create_performance_report(results_list):
        """Create a comprehensive performance report for multiple test results."""

        if not results_list:
            return "No results to analyze"

        report = []
        report.append("🎯 MATHEMATICAL FEASIBILITY PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary table
        report.append("📊 RESULTS SUMMARY")
        report.append("-" * 40)
        report.append("Player Count | Theoretical | Actual | Gap  | Quality")
        report.append("-------------|-------------|--------|------|----------")

        total_gap = 0
        perfect_count = 0
        excellent_count = 0

        for result in results_list:
            players = result.get("players", "Unknown")
            theoretical = result.get("theoretical_min", 0)
            actual = result.get("actual_total", 0)
            gap = result.get("gap_from_theoretical", 0)
            quality = result.get("quality", "Unknown")

            total_gap += gap
            if quality == "PERFECT":
                perfect_count += 1
            elif quality == "EXCELLENT":
                excellent_count += 1

            report.append(f"{players:11} | {theoretical:11} | {actual:6} | {gap:4} | {quality}")

        report.append("")

        # Performance metrics
        avg_gap = total_gap / len(results_list)
        perfect_rate = perfect_count / len(results_list) * 100
        excellent_rate = (perfect_count + excellent_count) / len(results_list) * 100

        report.append("📈 PERFORMANCE METRICS")
        report.append("-" * 25)
        report.append(f"Average gap from theoretical: {avg_gap:.1f}")
        report.append(f"Perfect achievement rate: {perfect_rate:.0f}%")
        report.append(f"Excellent+ achievement rate: {excellent_rate:.0f}%")
        report.append("")

        # Recommendations
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 20)

        if avg_gap <= 0:
            report.append("✅ EXCELLENT: Algorithm consistently meets/exceeds theoretical limits")
        elif avg_gap <= 2:
            report.append("✅ VERY_GOOD: Algorithm performs close to theoretical optimum")
        elif avg_gap <= 4:
            report.append("⚠️  GOOD: Algorithm performance is reasonable but has room for improvement")
        else:
            report.append("❌ NEEDS_IMPROVEMENT: Algorithm significantly underperforms theoretical limits")

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage and testing
    analyzer = ScheduleFeasibilityAnalyzer()

    print("🧮 MATHEMATICAL FEASIBILITY ANALYZER TEST")
    print("=" * 50)

    for players in [13, 14, 15, 16, 17, 18]:
        analysis = analyzer.calculate_theoretical_minimums(players)
        params = analyzer.get_optimal_parameters(players)

        print(f"\n{players} players:")
        print(f"  Theoretical minimum: {analysis['total_theoretical_min']}")
        print(f"  Range 0 possible: {analysis['range_0_possible']}")
        print(f"  Recommended population: {params['population_size']}")
        print(f"  Recommended runtime: {params['max_runtime']}s")
