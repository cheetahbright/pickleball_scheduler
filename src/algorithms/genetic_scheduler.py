#!/usr/bin/env python3
"""High-Performance Genetic Pickleball Scheduler - Optimized for speed."""

import builtins
import os
import random
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from src.algorithms.scheduler_reporting import (
        Game,
        build_scheduler_failure_result,
        build_scheduler_result,
        format_schedule,
        invoke_progress_callback,
        print_scheduler_final_report,
        print_scheduler_start_banner,
        sum_range_metrics,
    )
except ImportError:
    from algorithms.scheduler_reporting import (  # noqa: F401  (Game re-exported for legacy imports)
        Game,
        build_scheduler_failure_result,
        build_scheduler_result,
        format_schedule,
        invoke_progress_callback,
        print_scheduler_final_report,
        print_scheduler_start_banner,
        sum_range_metrics,
    )

try:
    from src.algorithms.scheduler_metrics import (
        count_avoidable_duplicate_rounds,
        count_duplicate_rounds,
        evaluate_metrics_from_arrangement_stats,
        evaluate_schedule_metrics,
        precompute_arrangement_stats,
        round_signature,
    )
except ImportError:
    from algorithms.scheduler_metrics import (
        count_avoidable_duplicate_rounds,
        count_duplicate_rounds,
        evaluate_metrics_from_arrangement_stats,
        evaluate_schedule_metrics,
        precompute_arrangement_stats,
        round_signature,
    )

try:
    from src.algorithms.scheduler_repairs import (
        apply_targeted_repairs,
        count_partners,
        repair_invalid_schedule,
        repair_opponent_imbalance,
        repair_partner_imbalance,
    )
except ImportError:
    from algorithms.scheduler_repairs import (
        apply_targeted_repairs,
        count_partners,
        repair_invalid_schedule,
        repair_opponent_imbalance,
        repair_partner_imbalance,
    )

try:
    from src.algorithms.scheduler_evolution import (
        compute_general_stop_threshold,
        compute_perfect_stop_threshold,
    )
    from src.algorithms.scheduler_evolution import crossover as _crossover_fn
    from src.algorithms.scheduler_evolution import elitism as _elitism_fn
    from src.algorithms.scheduler_evolution import random_individual as _random_individual_fn
    from src.algorithms.scheduler_evolution import (
        run_evolution_loop,
    )
    from src.algorithms.scheduler_evolution import super_aggressive_mutate as _super_aggressive_mutate_fn
    from src.algorithms.scheduler_evolution import tournament_select as _tournament_select_fn
except ImportError:
    from algorithms.scheduler_evolution import (
        compute_general_stop_threshold,
        compute_perfect_stop_threshold,
    )
    from algorithms.scheduler_evolution import crossover as _crossover_fn
    from algorithms.scheduler_evolution import elitism as _elitism_fn
    from algorithms.scheduler_evolution import random_individual as _random_individual_fn
    from algorithms.scheduler_evolution import (
        run_evolution_loop,
    )
    from algorithms.scheduler_evolution import super_aggressive_mutate as _super_aggressive_mutate_fn
    from algorithms.scheduler_evolution import tournament_select as _tournament_select_fn

try:
    from src.algorithms.arrangement_generator import game_valid as _game_valid_fn
    from src.algorithms.arrangement_generator import generate_arrangements as _generate_arrangements_fn
except ImportError:
    from algorithms.arrangement_generator import game_valid as _game_valid_fn
    from algorithms.arrangement_generator import generate_arrangements as _generate_arrangements_fn

try:
    from src.utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
except ImportError:
    try:
        from utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
    except ImportError:
        ScheduleFeasibilityAnalyzer = None


def _console_print(*args, sep: str = " ", end: str = "\n", file=None, flush=False):
    """Print safely on terminals that cannot encode emoji-rich progress logs."""
    target = file if file is not None else sys.stdout
    text = sep.join("" if arg is None else str(arg) for arg in args)
    encoding = getattr(target, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    builtins.print(safe_text, end=end, file=target, flush=flush)


# The scheduler is often run directly from Windows terminals and diagnostic scripts.
print = _console_print


GameTuple = Tuple[str, str, str, str]


class GeneticPickleballScheduler:
    """Genetic algorithm scheduler with true Range-0 early stop when feasible.

    Contract
    - Inputs: players (list[str]), num_courts (int), num_rounds (int=8)
    - Output: dict with keys {schedule, success, fitness, score, fitness_score,
      generations, time_seconds, algorithm, total_range, raw_metrics, metrics,
      fitness_details}
    - Range-0 feasible when (num_rounds*num_courts*4) % num_players == 0
    """

    # Desktop optimization parameters - QUALITY FOCUSED
    DESKTOP_PARAMS = {
        "population_size": 200,  # Increased from 30 (quality > speed)
        "elite_size": 20,  # Increased from 3 (preserve best solutions)
        "mutation_rate": 0.15,  # Reduced from 0.18 (more stable)
        "crossover_rate": 0.95,  # Increased from 0.85 (better mixing)
        "max_generations": 5000,  # Increased from 1200
        "convergence_patience": 100,  # Increased from 12 (don't stop early)
        "tournament_size": 7,  # Increased from 3 (stronger selection)
        "fitness_scale_factor": 1.5,
        "performance_metrics": {
            "target_gen_per_sec": 20,  # Reduced from 51.5 (quality > speed)
            "population_size": 200,
            "speedup_vs_baseline": "2x",  # Reduced from 8x
        },
    }

    # Lexicographic penalty weights (games > partners > courts > opponents)
    LEXICOGRAPHIC_WEIGHTS = {
        "games_range": 10_000_000,  # Games Range 0 is top priority
        "partners_range": 100_000,  # Partners Range 0 is second priority
        "courts_range": 1_000,  # Courts Range 0 is third priority
        "opponents_range": 10,  # Opponents minimized last
    }

    # Fixed court assignment based on player count
    @classmethod
    def get_fixed_courts(cls, num_players: int) -> int:
        """Calculate fixed court count based on player count."""
        return max(1, num_players // 4)

    def __init__(
        self,
        players: List[Any],
        num_courts: int = 1,
        num_rounds: Optional[int] = None,  # Always hardcoded to 8 rounds
        *,
        population_size: int = 100,  # Will use DESKTOP_PARAMS for desktop mode
        max_generations: int = 1000,  # Reduced from 2000
        max_runtime: float = 300.0,  # 5 minutes default for Range 0 optimization
        min_runtime: float | None = None,
        use_desktop_params: bool = False,  # Enable desktop optimization
        progress_callback: Optional[Callable[..., object]] = None,
    ) -> None:
        # Validate inputs
        if int(num_courts) < 1:
            raise ValueError("num_courts must be >= 1")

        # Desktop parameter optimization with feasibility analysis
        if use_desktop_params:
            # Get optimal parameters based on player count and mathematical constraints
            if ScheduleFeasibilityAnalyzer:
                optimal_params = ScheduleFeasibilityAnalyzer.get_optimal_parameters(len(players))
                population_size = optimal_params["population_size"]
                max_generations = optimal_params["max_generations"]
                max_runtime = optimal_params["max_runtime"]
                self.mutation_rate = optimal_params["mutation_rate"]
                self.crossover_rate = optimal_params["crossover_rate"]
                self.convergence_patience = optimal_params["convergence_patience"]
                self.elite_size = optimal_params["elite_size"]
                self.tournament_size = optimal_params["tournament_size"]
            else:
                # Fallback to original desktop params
                population_size = self.DESKTOP_PARAMS["population_size"]
                self.mutation_rate = self.DESKTOP_PARAMS["mutation_rate"]
                self.crossover_rate = self.DESKTOP_PARAMS["crossover_rate"]
                self.convergence_patience = self.DESKTOP_PARAMS["convergence_patience"]
                self.elite_size = self.DESKTOP_PARAMS["elite_size"]
                self.tournament_size = self.DESKTOP_PARAMS["tournament_size"]

        # Store progress callback for real-time updates
        self.progress_callback = progress_callback

        # Detect Player objects vs. plain strings and build dual representations
        has_objects = bool(players) and hasattr(players[0], "name")
        if has_objects:
            # Keep original Player objects for legacy API
            self.players = list(players)  # legacy-visible (Player objects)
            self.player_names = [str(getattr(p, "name", None) or str(p)) for p in players]
        else:
            # No Player objects supplied
            self.players = []  # legacy-visible (expected [] for string inputs)
            self.player_names = [str(p) for p in players]

        # Internal immutable names used by the algorithm
        self._names = list(self.player_names)
        self.num_players = len(self._names)
        if self.num_players == 0:
            raise ValueError("players list cannot be empty")

        # Use fixed court assignment for desktop mode
        if use_desktop_params:
            self.num_courts = self.get_fixed_courts(self.num_players)
        else:
            self.num_courts = int(num_courts)

        self.num_rounds = int(num_rounds) if num_rounds is not None else 8

        # GA params
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)

        # Runtime controls - ALLOW IMMEDIATE Range 0 early termination!
        env_max = os.environ.get("GA_MAX_RUNTIME")
        in_e2e = os.environ.get("E2E_TEST") in ("1", "true", "True")

        # CRITICAL: Set minimum runtime to 0 for immediate Range 0 early stop
        if in_e2e:
            self.min_runtime = 0.5
            self.max_runtime = float(env_max) if env_max else 10.0
        else:
            # DISABLE minimum runtime - allow immediate Range 0 termination
            self.min_runtime = 0.0  # ZERO minimum - early stop when Range 0 found!
            self.max_runtime = float(env_max) if env_max else float(max_runtime)

        # Ensure max is at least min (but min is now 0)
        if self.max_runtime < self.min_runtime:
            self.max_runtime = max(1.0, self.min_runtime)  # At least 1 second max

        # Default parameters (overridden by desktop params if enabled)
        if not use_desktop_params:
            # Increase mutation for better exploration (non-desktop mode)
            self.mutation_rate = 0.20  # Increased from 0.12
            self.crossover_rate = 0.85  # Increased from 0.8
            self.elite_size = max(4, self.population_size // 5)  # Keep more elites
            self.tournament_size = 7  # Default tournament size for non-desktop mode

        # Constraints
        self.do_not_pair_map: Dict[str, set[str]] = defaultdict(set)
        self.do_not_oppose_map: Dict[str, set[str]] = defaultdict(set)
        self.must_pair_map: Dict[str, set[str]] = defaultdict(set)
        self.must_oppose_map: Dict[str, set[str]] = defaultdict(set)

        # Availability/timing (legacy compatibility)
        self.availability: Dict[str, Any] = {}

        # If Player objects provided, build constraint maps and availability
        if has_objects:
            for p in players:
                name = str(getattr(p, "name", None) or str(p))
                for x in getattr(p, "do_not_pair", []) or []:
                    self.do_not_pair_map[name].add(str(x))
                for x in getattr(p, "do_not_oppose", []) or []:
                    self.do_not_oppose_map[name].add(str(x))
                self.availability[name] = {
                    "rounds": list(getattr(p, "available_rounds", []) or []),
                    "from": getattr(p, "available_from", None),
                    "to": getattr(p, "available_to", None),
                }

        if ScheduleFeasibilityAnalyzer:
            feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(
                self.num_players,
                self.num_courts,
                self.num_rounds,
            )
        else:
            total_slots = self.num_rounds * self.num_courts * 4
            total_partnerships = (self.num_players * (self.num_players - 1)) // 2
            partnership_slots = total_slots // 2
            partner_range0_possible = (
                partnership_slots % total_partnerships == 0 and partnership_slots >= total_partnerships
            )
            games_balanced = total_slots % self.num_players == 0
            courts_balanced = total_slots % (self.num_players * self.num_courts) == 0
            feasibility = {
                "min_partner_range": 0 if partner_range0_possible else 1,
                "total_theoretical_min": 0 if partner_range0_possible and games_balanced and courts_balanced else 1,
                "range_0_possible": partner_range0_possible and games_balanced and courts_balanced,
            }

        self.feasibility_minimums = feasibility
        self.range0_possible = bool(feasibility["range_0_possible"])
        self.partner_range0_possible = int(feasibility["min_partner_range"]) == 0
        self.optimal_partner_range = int(feasibility["min_partner_range"])
        self.optimal_total_range = int(feasibility["total_theoretical_min"])

        # Precompute arrangement pool (one round templates)
        self.arrangements: List[List[Tuple[str, str, str, str]]] = self._generate_arrangements(max_arrangements=400)
        if not self.arrangements:
            raise ValueError("No valid arrangements generated for these players/courts")
        self.max_unique_rounds = len({self._round_signature(arrangement) for arrangement in self.arrangements})
        self.minimum_duplicate_rounds = max(0, self.num_rounds - self.max_unique_rounds)

        # Ensure constraint maps have entries for all players (legacy expectation)
        for name in self._names:
            _ = self.do_not_pair_map[name]
            _ = self.do_not_oppose_map[name]

        # Performance optimizations - Enhanced caching
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}
        self._metrics_cache: Dict[Tuple[int, ...], Dict[str, int]] = {}
        self._schedule_cache: Dict[Tuple[int, ...], List[List[Tuple]]] = {}
        self._signature_cache: Dict[Tuple[int, ...], Tuple] = {}
        self._refresh_arrangement_stats()

        # Progress tracking for reduced update frequency
        self._last_progress_time = 0.0
        self._last_logged_fitness = float("inf")
        self._progress_update_interval = 2.0  # Update every 2 seconds max
        # Last run diagnostics
        self.last_fitness_details: Dict[str, int] | None = None
        # Legacy/compatibility knobs - don't override convergence_patience if already set by desktop params
        if not hasattr(self, "convergence_patience"):
            self.convergence_patience: int = 60
        self.time_budget: float = float(self.max_runtime)
        self.substitution_info: Dict[str, Any] | None = None

    # ---------------- Public API ----------------
    def add_constraints(
        self,
        pair_constraints: List[Tuple[str, str]] | None = None,
        oppose_constraints: List[Tuple[str, str]] | None = None,
        must_pair: List[Tuple[str, str]] | None = None,
        must_oppose: List[Tuple[str, str]] | None = None,
    ) -> None:
        if pair_constraints:
            for a, b in pair_constraints:
                a, b = str(a), str(b)
                self.do_not_pair_map[a].add(b)
                self.do_not_pair_map[b].add(a)
        if oppose_constraints:
            for a, b in oppose_constraints:
                a, b = str(a), str(b)
                self.do_not_oppose_map[a].add(b)
                self.do_not_oppose_map[b].add(a)
        if pair_constraints or oppose_constraints:
            # The arrangement pool was built before these constraints existed,
            # so it can still contain forbidden pairings/oppositions. Rebuild it
            # so hard constraints are excluded from the search space entirely
            # instead of relying only on the fitness penalty.
            rebuilt = self._generate_arrangements(max_arrangements=400)
            if rebuilt:
                self.arrangements = rebuilt
                self.max_unique_rounds = len({self._round_signature(arrangement) for arrangement in self.arrangements})
                self.minimum_duplicate_rounds = max(0, self.num_rounds - self.max_unique_rounds)
            else:
                print("⚠️ Constraints leave no valid round arrangements - keeping unconstrained pool with penalties")
            # Constraint maps changed either way, so per-arrangement violation
            # summaries and every cached evaluation are stale.
            self._fitness_cache.clear()
            self._metrics_cache.clear()
            self._schedule_cache.clear()
            self._signature_cache.clear()
            self._refresh_arrangement_stats()
        if must_pair:
            for a, b in must_pair:
                a, b = str(a), str(b)
                self.must_pair_map[a].add(b)
                self.must_pair_map[b].add(a)
        if must_oppose:
            for a, b in must_oppose:
                a, b = str(a), str(b)
                self.must_oppose_map[a].add(b)
                self.must_oppose_map[b].add(a)

    def set_substitution_info(self, info: Dict[str, Any] | None) -> None:
        self.substitution_info = info

    def _minimum_runtime_before_general_stop(self) -> float:
        """Return the minimum runtime before non-perfect early exits are allowed."""
        return compute_general_stop_threshold(self.num_players, self.num_rounds, self.min_runtime)

    def _minimum_runtime_before_perfect_stop(self) -> float:
        """Return how long to keep searching after a perfect schedule is found."""
        return compute_perfect_stop_threshold(self.num_players, self.num_rounds, self.min_runtime, self.max_runtime)

    # Main entry
    def generate_schedule(
        self,
        verbose: bool = False,
        max_time: float | None = None,
        progress_callback: Optional[Callable[..., object]] = None,
        seed: Optional[int] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """Run GA with Range 0 early termination and optional progress callback.

        Pass ``seed`` (and optionally a fake ``clock``) to make a run fully
        reproducible; by default the seed is time-based for run-to-run variety.
        """
        import random
        import time

        # Use instance progress callback if not provided
        if progress_callback is None:
            progress_callback = self.progress_callback

        # Update runtime settings
        if max_time is not None and isinstance(max_time, (int, float)):
            self.max_runtime = float(max_time)
        else:
            self.max_runtime = self.max_runtime

        absolute_min_runtime = self._minimum_runtime_before_general_stop()
        perfect_stop_runtime = self._minimum_runtime_before_perfect_stop()

        now = clock if clock is not None else time.time
        start_time = now()

        # Time-based seed by default so unseeded runs stay varied.
        if seed is None:
            seed = int(time.time() * 1000) % 1000000
        rng = random.Random(seed)
        # Expose the run's rng/clock to repair callbacks for reproducibility.
        self._run_rng = rng
        self._run_clock = now

        print_scheduler_start_banner(
            print,
            absolute_min_runtime=absolute_min_runtime,
            perfect_stop_runtime=perfect_stop_runtime,
            max_runtime=self.max_runtime,
            partner_range0_possible=self.partner_range0_possible,
            optimal_partner_range=self.optimal_partner_range,
            range0_possible=self.range0_possible,
            optimal_total_range=self.optimal_total_range,
            max_unique_rounds=self.max_unique_rounds,
            minimum_duplicate_rounds=self.minimum_duplicate_rounds,
            seed=seed,
        )

        best_individual, best_fitness, generations_run = run_evolution_loop(
            self,
            rng=rng,
            now=now,
            start_time=start_time,
            verbose=verbose,
            progress_callback=progress_callback,
            absolute_min_runtime=absolute_min_runtime,
            perfect_stop_runtime=perfect_stop_runtime,
            printer=print,
            invoke_progress=invoke_progress_callback,
        )

        # Final results
        if best_individual is None:
            print("❌ No valid solution found!")
            return build_scheduler_failure_result(
                elapsed_seconds=now() - start_time,
                generations_run=generations_run,
                optimal_total_range=self.optimal_total_range,
                minimum_duplicate_rounds=self.minimum_duplicate_rounds,
                max_unique_rounds=self.max_unique_rounds,
            )

        final_schedule = self._decode(best_individual)
        final_metrics = self._evaluate_metrics(final_schedule)
        duplicate_rounds = self._count_duplicate_rounds(final_schedule)
        avoidable_duplicate_rounds = self._count_avoidable_duplicate_rounds(final_schedule)

        # POST-PROCESSOR: Apply targeted repairs if optimal range not achieved
        current_total_range = sum_range_metrics(final_metrics)
        optimal_total_range = self.optimal_total_range

        if current_total_range > optimal_total_range:
            print("\n🔧 POST-PROCESSOR: Attempting targeted repairs...")
            repaired_schedule = self._apply_targeted_repairs(final_schedule, max_time=0.5)
            if repaired_schedule:
                repair_metrics = self._evaluate_metrics(repaired_schedule)
                repair_total = sum_range_metrics(repair_metrics)
                if repair_total < current_total_range:
                    print("✅ POST-PROCESSOR: Improved from Range " f"{current_total_range} to Range {repair_total}")
                    final_schedule = repaired_schedule
                    final_metrics = repair_metrics
                    current_total_range = repair_total
                else:
                    print(f"   ⚠️ REPAIR PARTIAL: Reduced to total range {repair_total}")

        elapsed_seconds = now() - start_time
        print_scheduler_final_report(
            print,
            elapsed_seconds=elapsed_seconds,
            final_metrics=final_metrics,
            duplicate_rounds=duplicate_rounds,
            avoidable_duplicate_rounds=avoidable_duplicate_rounds,
            minimum_duplicate_rounds=self.minimum_duplicate_rounds,
            partner_range0_possible=self.partner_range0_possible,
            optimal_partner_range=self.optimal_partner_range,
            optimal_total_range=optimal_total_range,
            num_players=self.num_players,
            max_unique_rounds=self.max_unique_rounds,
        )

        # Store fitness details
        self.last_fitness_details = {
            "games_range": final_metrics["games_range"],
            "partners_range": final_metrics["partners_range"],
            "opponents_range": final_metrics["opponents_range"],
            "courts_range": final_metrics["courts_range"],
            "range0_possible": self.range0_possible,
        }

        # Return dict format expected by multi-algorithm scheduler
        result = build_scheduler_result(
            formatted_schedule=self._format_schedule(final_schedule),
            best_fitness=best_fitness,
            generations_run=generations_run,
            elapsed_seconds=elapsed_seconds,
            final_metrics=final_metrics,
            optimal_total_range=self.optimal_total_range,
            duplicate_rounds=duplicate_rounds,
            avoidable_duplicate_rounds=avoidable_duplicate_rounds,
            minimum_duplicate_rounds=self.minimum_duplicate_rounds,
            max_unique_rounds=self.max_unique_rounds,
            fitness_details=self.last_fitness_details,
            seed=seed,
        )

        # Quality check for small problems
        elapsed = elapsed_seconds
        if self.num_players <= 8 and best_fitness > 0 and elapsed < self.max_runtime:
            if verbose:
                print(
                    f"⚠️ Suboptimal result (Range {result['total_range']}). "
                    "Consider increasing runtime or population size."
                )

        # Add feasibility analysis if available
        if ScheduleFeasibilityAnalyzer:
            feasibility_assessment = ScheduleFeasibilityAnalyzer.assess_quality_with_feasibility(
                final_metrics, self.num_players, self.num_courts, self.num_rounds
            )
            result["feasibility_analysis"] = feasibility_assessment

        return result

    def validate_constraints(self, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compatibility helper for older tests and diagnostics."""
        duplicate_rounds = False
        round_duplicates = 0
        incomplete_games = 0

        try:
            from src.algorithms.constraint_model import ScheduleRepair
        except ImportError:
            from algorithms.constraint_model import ScheduleRepair

        normalized_rounds: List[List[Dict[str, Any]]] = []
        for round_data in schedule:
            games = round_data.get("games", []) if isinstance(round_data, dict) else []
            normalized_games: List[Dict[str, Any]] = []
            players_in_round: set[str] = set()

            for game in games:
                if not isinstance(game, dict):
                    continue

                team1 = [str(player) for player in game.get("team1", [])]
                team2 = [str(player) for player in game.get("team2", [])]
                all_players = team1 + team2

                if len(all_players) != 4:
                    incomplete_games += 1

                if len(all_players) != len(set(all_players)):
                    round_duplicates += 1

                overlap = players_in_round.intersection(all_players)
                if overlap:
                    round_duplicates += len(overlap)

                players_in_round.update(all_players)
                normalized_games.append(
                    {
                        "team1": team1,
                        "team2": team2,
                        "court": game.get("court"),
                    }
                )

            normalized_rounds.append(normalized_games)

        duplicate_rounds = ScheduleRepair.has_duplicate_rounds(normalized_rounds)
        total_violations = round_duplicates + incomplete_games + int(duplicate_rounds)

        return {
            "violations": total_violations,
            "duplicate_rounds": duplicate_rounds,
            "duplicate_players": round_duplicates,
            "incomplete_games": incomplete_games,
            "is_valid": total_violations == 0,
        }

    def _super_aggressive_mutate(self, ind: Tuple[int, ...], rng: random.Random) -> Tuple[int, ...]:
        """SUPER aggressive mutation for breaking out of local optima with diversity enforcement."""
        return _super_aggressive_mutate_fn(ind, rng, len(self.arrangements))

    def _random_individual(self, rng) -> Tuple[int, ...]:
        """Create a random individual ensuring maximum diversity."""
        return _random_individual_fn(rng, len(self.arrangements), self.num_rounds)

    def _crossover(
        self, parent1: Tuple[int, ...], parent2: Tuple[int, ...], rng
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Crossover two parents to create offspring."""
        return _crossover_fn(parent1, parent2, rng)

    def _generate_arrangements(self, max_arrangements: int = 400) -> List[List[Tuple[str, str, str, str]]]:
        """Generate the diverse round-arrangement pool (see arrangement_generator)."""
        return _generate_arrangements_fn(
            self._names,
            self.num_courts,
            do_not_pair_map=self.do_not_pair_map,
            do_not_oppose_map=self.do_not_oppose_map,
            max_arrangements=max_arrangements,
            printer=print,
        )

    def _game_valid(self, game: Tuple[str, str, str, str]) -> bool:
        """Check if a game satisfies all constraints."""
        return _game_valid_fn(game, self.do_not_pair_map, self.do_not_oppose_map)

    def _tournament_select(self, population, fitness_scores, rng, k=7):
        """Tournament selection with configurable tournament size."""
        return _tournament_select_fn(population, fitness_scores, rng, k)

    def _evaluate_population_optimized(self, population: List[Tuple[int, ...]]) -> List[Tuple[Tuple[int, ...], float]]:
        """Optimized population evaluation with enhanced caching (sequential for better performance)."""
        fitness_scores = []

        # Use pure sequential processing with optimized caching
        # Parallel processing has too much overhead for this use case
        for ind in population:
            if ind in self._fitness_cache:
                fitness_scores.append((ind, self._fitness_cache[ind]))
            else:
                fitness = self._fitness(ind)
                fitness_scores.append((ind, fitness))

        return fitness_scores

    def _refresh_arrangement_stats(self) -> None:
        """Precompute per-arrangement summaries powering the fast fitness path.

        Every metric and violation term is local to a single round, so each
        pool arrangement can be summarized once; per-individual evaluation then
        merges the per-round summaries instead of re-walking every game.
        """
        self._arrangement_stats = precompute_arrangement_stats(
            self.arrangements,
            num_courts=self.num_courts,
            do_not_pair_map=self.do_not_pair_map,
            do_not_oppose_map=self.do_not_oppose_map,
        )
        self._arrangement_signatures = [round_signature(arrangement) for arrangement in self.arrangements]
        duplicate_penalties: List[int] = []
        for arrangement in self.arrangements:
            penalty = 0
            round_players: set = set()
            for game in arrangement:
                game_players = set(game)
                if len(game_players) != 4:
                    penalty += 1000000000
                overlap = game_players & round_players
                if overlap:
                    penalty += 1000000000 * len(overlap)
                round_players.update(game_players)
            duplicate_penalties.append(penalty)
        self._arrangement_duplicate_penalties = duplicate_penalties

    def _decode_cached(self, individual: Tuple[int, ...]) -> List[List[Tuple]]:
        """Cached version of schedule decoding."""
        if individual in self._schedule_cache:
            return self._schedule_cache[individual]

        schedule = self._decode(individual)

        self._schedule_cache[individual] = schedule

        return schedule

    def _evaluate_metrics_cached(self, schedule: List[List[Tuple]], individual: Tuple[int, ...]) -> Dict[str, int]:
        """Cached metrics evaluation using precomputed per-arrangement stats."""
        _ = schedule  # The individual's pool indices fully determine the metrics.
        if individual in self._metrics_cache:
            return self._metrics_cache[individual]

        metrics = evaluate_metrics_from_arrangement_stats(
            [self._arrangement_stats[idx] for idx in individual],
            self._names,
        )

        self._metrics_cache[individual] = metrics

        return metrics

    @staticmethod
    def _round_signature(round_games: List[Tuple[str, str, str, str]] | List[Tuple]) -> Tuple:
        """Build a canonical signature for a round regardless of game/team ordering."""
        return round_signature(round_games)

    def _count_duplicate_rounds(self, schedule: List[List[Tuple[str, str, str, str]]]) -> int:
        """Count repeated round patterns in a schedule."""
        return count_duplicate_rounds(schedule)

    def _count_avoidable_duplicate_rounds(self, schedule: List[List[Tuple[str, str, str, str]]]) -> int:
        """Count duplicate rounds beyond the mathematically unavoidable minimum."""
        return count_avoidable_duplicate_rounds(schedule, self.minimum_duplicate_rounds)

    def _get_duplicate_signature_cached(self, individual: Tuple[int, ...], schedule: List[List[Tuple]]) -> Tuple:
        """Cached duplicate-round signature built from precomputed pool signatures."""
        _ = schedule  # The individual's pool indices fully determine the signature.
        if individual in self._signature_cache:
            return self._signature_cache[individual]

        full_signature = tuple(self._arrangement_signatures[idx] for idx in individual)

        self._signature_cache[individual] = full_signature

        return full_signature

    def _fitness(self, individual: Tuple[int, ...]) -> float:
        """Optimized lexicographic fitness function with enhanced caching."""
        if individual in self._fitness_cache:
            return self._fitness_cache[individual]

        # Metrics, structural penalties, and signatures all derive from the
        # precomputed per-arrangement summaries - no decode needed here.
        metrics = self._evaluate_metrics_cached([], individual)

        # CRITICAL: player-duplicate penalties (precomputed per arrangement)
        duplicate_penalty = sum(self._arrangement_duplicate_penalties[idx] for idx in individual)

        # Use optimized duplicate detection with caching
        signature = self._get_duplicate_signature_cached(individual, [])
        unique_rounds = len(set(signature))
        total_rounds = len(signature)
        duplicate_rounds = total_rounds - unique_rounds
        avoidable_duplicate_rounds = max(0, duplicate_rounds - self.minimum_duplicate_rounds)

        # MASSIVE penalty only for duplicate rounds beyond what the configuration forces.
        duplicate_penalty += avoidable_duplicate_rounds * 1000000000  # 1 BILLION per avoidable duplicate

        # Lexicographic optimization using LEXICOGRAPHIC_WEIGHTS
        penalty = 0.0

        # Layer 1: Games Range (Highest Priority)
        penalty += metrics["games_range"] * self.LEXICOGRAPHIC_WEIGHTS["games_range"]

        # Layer 2: Partners Range (Second Priority)
        penalty += metrics["partners_range"] * self.LEXICOGRAPHIC_WEIGHTS["partners_range"]

        # Layer 3: Courts Range (Third Priority)
        penalty += metrics["courts_range"] * self.LEXICOGRAPHIC_WEIGHTS["courts_range"]

        # Layer 4: Opponents Range (Lowest Priority)
        penalty += metrics["opponents_range"] * self.LEXICOGRAPHIC_WEIGHTS["opponents_range"]

        # Add constraint violations and duplicate penalties
        penalty += metrics["violations"] * 50000000  # 50M per violation
        penalty += duplicate_penalty

        # Range 0 achievement bonus for desktop optimization
        if self.range0_possible:
            total_range = (
                metrics["games_range"]
                + metrics["partners_range"]
                + metrics["opponents_range"]
                + metrics["courts_range"]
            )
            if total_range == 0:
                penalty -= 100000000  # 100M bonus for perfect Range 0
            elif total_range == 1:
                penalty -= 50000000  # 50M bonus for near-perfect Range 1

        # Cache the result
        self._fitness_cache[individual] = penalty

        return penalty

    def _decode(self, individual: Tuple[int, ...]) -> List[List[Tuple[str, str, str, str]]]:
        """Decode individual to schedule."""
        schedule = [self.arrangements[idx] for idx in individual]
        return schedule

    def _repair_invalid_schedule(
        self,
        schedule: List[List[Tuple[str, str, str, str]]],
        problematic_round_idx: int,
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Repair an invalid schedule by replacing problematic rounds."""
        return repair_invalid_schedule(schedule, problematic_round_idx, self.arrangements, print)

    def _evaluate_metrics(
        self, schedule: List[List[Tuple[str, str, str, str]]] | List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Evaluate fairness metrics from schedule in either raw tuple format or formatted dict format."""
        return evaluate_schedule_metrics(
            schedule,
            num_courts=self.num_courts,
            player_names=self._names,
            do_not_pair_map=self.do_not_pair_map,
            do_not_oppose_map=self.do_not_oppose_map,
        )

    def _apply_targeted_repairs(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float = 0.5
    ) -> Optional[List[List[Tuple[str, str, str, str]]]]:
        """Apply targeted local repairs to achieve Range 0 when mathematically possible."""
        return apply_targeted_repairs(
            schedule,
            max_time,
            evaluate_metrics=self._evaluate_metrics,
            printer=print,
            partner_repair=self._repair_partner_imbalance,
            opponent_repair=self._repair_opponent_imbalance,
            clock=getattr(self, "_run_clock", None),
        )

    def _repair_partner_imbalance(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Targeted repair for partner imbalance using smart swaps."""
        return repair_partner_imbalance(
            schedule,
            max_time,
            evaluate_metrics=self._evaluate_metrics,
            printer=print,
            rng=getattr(self, "_run_rng", None),
            clock=getattr(self, "_run_clock", None),
        )

    def _repair_opponent_imbalance(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Targeted repair for opponent imbalance using smart swaps."""
        return repair_opponent_imbalance(
            schedule,
            max_time,
            evaluate_metrics=self._evaluate_metrics,
            printer=print,
        )

    def _elitism(self, population: List[Tuple[int, ...]], fitnesses: List[float], k: int) -> List[Tuple[int, ...]]:
        """Select k best individuals for elitism."""
        return _elitism_fn(population, fitnesses, k)

    def _tournament(
        self, population: List[Tuple[int, ...]], fitnesses: List[float], rng, k: int = 3
    ) -> Tuple[int, ...]:
        """Tournament selection."""
        cand = rng.sample(range(len(population)), k=min(k, len(population)))
        best = min(cand, key=lambda i: fitnesses[i])
        return population[best]

    def _count_partners(self, schedule: List[List[Tuple[str, str, str, str]]]) -> Dict[str, Dict[str, int]]:
        """Count partner pairings for each player in the schedule."""
        return count_partners(schedule)

    def _smart_mutate(self, ind: Tuple[int, ...], rng) -> Tuple[int, ...]:
        """Smart mutation focusing on partner balance for desktop optimization."""
        arr_len = len(self.arrangements)
        out = list(ind)

        # Get current schedule for analysis
        current_schedule = self._decode(ind)
        partner_counts = self._count_partners(current_schedule)

        # Calculate partner imbalance
        max_partnerships = 0
        min_partnerships = float("inf")
        for player_partners in partner_counts.values():
            if player_partners:
                max_partnerships = max(max_partnerships, max(player_partners.values()))
                min_partnerships = min(min_partnerships, min(player_partners.values()))

        partner_range = max_partnerships - min_partnerships if min_partnerships != float("inf") else 0

        # Use smart mutation if partner range > 0
        if partner_range > 0:
            # Target rounds with high partner imbalance
            for i in range(len(out)):
                if rng.random() < self.mutation_rate * 1.5:  # Higher mutation rate for smart targeting
                    # Try multiple candidates and pick the best for partner balance
                    best_candidate = out[i]
                    best_improvement = 0

                    for _ in range(5):  # Test 5 random alternatives
                        candidate = rng.randrange(0, arr_len)
                        if candidate != out[i]:
                            # Test this candidate
                            test_out = out[:]
                            test_out[i] = candidate
                            test_schedule = self._decode(tuple(test_out))
                            test_partners = self._count_partners(test_schedule)

                            # Calculate improvement in partner balance
                            test_max = 0
                            test_min = float("inf")
                            for player_partners in test_partners.values():
                                if player_partners:
                                    test_max = max(test_max, max(player_partners.values()))
                                    test_min = min(test_min, min(player_partners.values()))

                            test_range = test_max - test_min if test_min != float("inf") else 0
                            improvement = partner_range - test_range

                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_candidate = candidate

                    out[i] = best_candidate
        else:
            # Standard mutation when partner balance is already good
            for i in range(len(out)):
                if rng.random() < self.mutation_rate:
                    out[i] = rng.randrange(0, arr_len)

        # Occasional swap mutation for diversity
        if rng.random() < 0.1 and len(out) > 1:
            i, j = rng.sample(range(len(out)), 2)
            out[i], out[j] = out[j], out[i]

        return tuple(out)

    def _mutate(self, ind: Tuple[int, ...], rng) -> Tuple[int, ...]:
        """Standard mutation with smart mutation option for desktop mode."""
        # Use smart mutation if desktop parameters are enabled
        if (
            hasattr(self, "convergence_patience")
            and self.convergence_patience == self.DESKTOP_PARAMS["convergence_patience"]
        ):
            return self._smart_mutate(ind, rng)

        # Standard mutation for non-desktop mode
        arr_len = len(self.arrangements)
        out = list(ind)
        for i in range(len(out)):
            if rng.random() < self.mutation_rate:
                out[i] = rng.randrange(0, arr_len)
        # occasional swap mutation
        if rng.random() < 0.1 and len(out) > 1:
            i, j = rng.sample(range(len(out)), 2)
            out[i], out[j] = out[j], out[i]
        return tuple(out)

    def _format_schedule(self, schedule: List[List[Tuple[str, str, str, str]]]) -> List[Dict[str, Any]]:
        """Format schedule to legacy format without silently dropping rounds."""
        return format_schedule(schedule)

    def fitness(self, individual: List[int]) -> float:
        """Legacy fitness method for compatibility."""
        return self._fitness(tuple(individual))

    def create_random_schedule(self) -> List[List[Tuple[str, str, str, str]]]:
        """Legacy: Create a random schedule by decoding a random individual."""
        rng = random.Random()
        ind = self._random_individual(rng)
        return self._decode(ind)

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Legacy crossover method."""
        rng = random.Random()
        c1, c2 = self._crossover(tuple(parent1), tuple(parent2), rng)
        return list(c1), list(c2)

    def mutate(self, individual: List[int]) -> List[int]:
        """Legacy mutate method."""
        rng = random.Random()
        return list(self._mutate(tuple(individual), rng))

    def tournament_selection(self, population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
        """Legacy tournament selection."""
        rng = random.Random()
        pop_tuples = [tuple(ind) for ind in population]
        selected = self._tournament(pop_tuples, fitnesses, rng, k)
        return list(selected)

    def evolve_generation(
        self, population: List[List[int]], fitnesses: List[float]
    ) -> Tuple[List[List[int]], List[float]]:
        """Legacy evolve method."""
        rng = random.Random()
        pop_tuples = [tuple(ind) for ind in population]
        new_pop = []

        # Elitism
        elites = self._elitism(pop_tuples, fitnesses, self.elite_size)
        new_pop.extend(elites)

        # Fill rest
        while len(new_pop) < self.population_size:
            if rng.random() < self.crossover_rate:
                p1 = self._tournament(pop_tuples, fitnesses, rng)
                p2 = self._tournament(pop_tuples, fitnesses, rng)
                c1, c2 = self._crossover(p1, p2, rng)
                c1 = self._mutate(c1, rng)
                c2 = self._mutate(c2, rng)
                new_pop.extend([c1, c2])
            else:
                p = self._tournament(pop_tuples, fitnesses, rng)
                new_pop.append(self._mutate(p, rng))

        new_pop = new_pop[: self.population_size]
        new_pop_lists = [list(ind) for ind in new_pop]
        new_fitnesses = [self.fitness(ind) for ind in new_pop_lists]

        return new_pop_lists, new_fitnesses
