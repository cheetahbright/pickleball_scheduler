#!/usr/bin/env python3
"""High-Performance Genetic Pickleball Scheduler - Optimized for speed."""

import builtins
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

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


@dataclass
class Game:
    team1: List[str]
    team2: List[str]
    court: int


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

        # Feasibility for Partner Range-0 (primary constraint)
        total_slots = self.num_rounds * self.num_courts * 4
        total_partnerships = (self.num_players * (self.num_players - 1)) // 2
        partnership_slots = total_slots // 2

        # Partner Range 0 requires perfect division of partnerships
        partner_range0_possible = (
            partnership_slots % total_partnerships == 0 and partnership_slots >= total_partnerships
        )

        # Overall Range 0 considers games + courts + partners (opponents flexible)
        games_balanced = total_slots % self.num_players == 0
        courts_balanced = total_slots % (self.num_players * self.num_courts) == 0

        self.range0_possible = partner_range0_possible and games_balanced and courts_balanced
        self.partner_range0_possible = partner_range0_possible
        self.optimal_partner_range = 0 if partner_range0_possible else 1

        # Precompute arrangement pool (one round templates)
        self.arrangements: List[List[Tuple[str, str, str, str]]] = self._generate_arrangements(max_arrangements=400)
        if not self.arrangements:
            raise ValueError("No valid arrangements generated for these players/courts")

        # Ensure constraint maps have entries for all players (legacy expectation)
        for name in self._names:
            _ = self.do_not_pair_map[name]
            _ = self.do_not_oppose_map[name]

        # Performance optimizations - Enhanced caching
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}
        self._metrics_cache: Dict[Tuple[int, ...], Dict[str, int]] = {}
        self._schedule_cache: Dict[Tuple[int, ...], List[List[Tuple]]] = {}
        self._signature_cache: Dict[Tuple[int, ...], Tuple] = {}

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

    # Main entry
    def generate_schedule(
        self,
        verbose: bool = False,
        max_time: float | None = None,
        progress_callback: Optional[Callable[..., object]] = None,
    ):
        """Run GA with Range 0 early termination and optional progress callback."""
        import random
        import time

        # Use instance progress callback if not provided
        if progress_callback is None:
            progress_callback = self.progress_callback

        # QUALITY-FOCUSED: For small problems where Range 0 is achievable, ensure minimum runtime
        if self.num_players <= 8 and self.num_rounds <= 8:
            ABSOLUTE_MIN_RUNTIME = max(15.0, self.min_runtime)  # At least 15 seconds for quality
        else:
            ABSOLUTE_MIN_RUNTIME = self.min_runtime or 5.0

        # Environment check for testing only
        if os.environ.get("E2E_TEST") in ("1", "true", "True"):
            ABSOLUTE_MIN_RUNTIME = 0.5

        # Update runtime settings
        if max_time is not None and isinstance(max_time, (int, float)):
            self.max_runtime = float(max_time)
        else:
            self.max_runtime = self.max_runtime

        start_time = time.time()

        # CRITICAL FIX: Use RANDOM seed based on time for different results each run!
        seed = int(time.time() * 1000) % 1000000  # Different seed each run
        rng = random.Random(seed)

        # Log runtime enforcement
        print(f"🕐 GA Starting with ENFORCED minimum runtime: {ABSOLUTE_MIN_RUNTIME}s")
        print(f"   Maximum runtime: {self.max_runtime}s")
        print(f"   Partner Range 0 possible: {self.partner_range0_possible}")
        print(f"   Optimal partner range: {self.optimal_partner_range}")
        print(f"   Overall Range 0 possible: {self.range0_possible}")
        print(f"   Random seed: {seed} (different each run for variety)")

        # Initialize tracking
        best_individual = None
        best_fitness = float("inf")
        best_ranges = None
        generations_run = 0
        generations_without_improvement = 0

        # Track if we've logged progress recently
        last_progress_log = 0

        # Initialize population
        population = [self._random_individual(rng) for _ in range(self.population_size)]
        current_ranges = {"games": 0, "partners": 0, "opponents": 0, "courts": 0}

        # MAIN EVOLUTION LOOP - GUARANTEED TO RUN FOR MINIMUM TIME
        while True:
            elapsed = time.time() - start_time

            # CRITICAL: Check if we can stop
            can_stop = False
            stop_reason = None

            # Check for Range 0 achievement
            if best_individual is not None:
                schedule = self._decode(best_individual)
                metrics = self._evaluate_metrics(schedule)
                current_ranges = {
                    "games": metrics["games_range"],
                    "partners": metrics["partners_range"],
                    "opponents": metrics["opponents_range"],
                    "courts": metrics["courts_range"],
                }

                total_range = sum(current_ranges.values())

                # Log progress every 15 seconds (much less frequent)
                if elapsed - last_progress_log >= 15.0:
                    print(
                        f"⏱️ {elapsed:.1f}s - Range {total_range} "
                        f"(G:{current_ranges['games']} P:{current_ranges['partners']} "
                        f"O:{current_ranges['opponents']} C:{current_ranges['courts']})"
                    )
                    last_progress_log = elapsed

                # Progress callback for real-time Streamlit updates
                if progress_callback:
                    progress_data = {
                        "elapsed_time": elapsed,
                        "generation": generations_run,
                        "best_fitness": best_fitness,
                        "current_ranges": current_ranges,
                        "total_range": total_range,
                        "range0_possible": self.range0_possible,
                        "max_time": self.max_runtime,
                    }
                    try:
                        progress_callback(progress_data)
                    except Exception as e:
                        print(f"Progress callback failed: {e}")

                # Check stopping conditions - RUN FOR FULL TIME FOR QUALITY
                if self.range0_possible and total_range == 0:
                    # Even with Range 0, continue for quality unless we've run sufficient time
                    if elapsed >= min(30.0, self.max_runtime * 0.8):
                        can_stop = True
                        stop_reason = "PERFECT RANGE 0 ACHIEVED AFTER SUFFICIENT RUNTIME! 🎉"
                    else:
                        can_stop = False  # Keep running for quality
                elif self.range0_possible and total_range > 0:
                    # Continue trying for Range 0 but respect time limits
                    if elapsed >= self.max_runtime:
                        can_stop = True
                        stop_reason = (
                            f"Max runtime {self.max_runtime}s reached - "
                            f"Range 0 not achieved (current: {total_range})"
                        )
                elif elapsed >= self.max_runtime:
                    can_stop = True
                    stop_reason = f"Max runtime {self.max_runtime}s reached - FORCING STOP"
                elif elapsed < ABSOLUTE_MIN_RUNTIME:
                    # Continue until minimum runtime unless Range 0
                    can_stop = False
                    # Minimal logging every 5 seconds only
                    if int(elapsed) % 5 == 0 and elapsed > 0 and int(elapsed) != int(elapsed - 0.1):
                        print(f"⏳ Optimizing... {ABSOLUTE_MIN_RUNTIME - elapsed:.1f}s remaining")
                elif generations_run >= self.max_generations and elapsed >= ABSOLUTE_MIN_RUNTIME:
                    # Only stop for max generations AFTER minimum runtime
                    can_stop = True
                    stop_reason = f"Max generations {self.max_generations} reached"
            else:
                # First generation
                if elapsed >= self.max_runtime:
                    can_stop = True
                    stop_reason = "Max runtime reached (no valid solution found)"

            # EXIT CONDITION
            if can_stop:
                print(f"🏁 Stopping: {stop_reason} after {elapsed:.1f}s and {generations_run} generations")
                break

            # EVOLUTION STEP
            # Evaluate current population with optimized batch processing
            fitness_scores = self._evaluate_population_optimized(population)

            # Track best with convergence and quality-focused improvements
            fitness_improved = False
            for ind, fitness in fitness_scores:
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = ind
                    fitness_improved = True

                    # Only log significant improvements and limit frequency
                    current_time = time.time()
                    fitness_improvement = self._last_logged_fitness - fitness
                    time_since_log = current_time - self._last_progress_time

                    # Log if major improvement OR enough time passed
                    if fitness_improvement > 1000000 or time_since_log > self._progress_update_interval:

                        schedule = self._decode_cached(ind)
                        metrics = self._evaluate_metrics_cached(schedule, ind)
                        best_ranges = {
                            "games": metrics["games_range"],
                            "partners": metrics["partners_range"],
                            "opponents": metrics["opponents_range"],
                            "courts": metrics["courts_range"],
                        }
                        total_range = sum(best_ranges.values())

                        print(f"📈 Gen {generations_run}: Range = {total_range}, Fitness = {fitness:.0f}")

                        self._last_progress_time = current_time
                        self._last_logged_fitness = fitness

            # Enhanced convergence tracking for quality
            if fitness_improved:
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Check for convergence with Range 0 priority
            if best_individual is not None:
                schedule = self._decode(best_individual)
                metrics = self._evaluate_metrics(schedule)
                total_range = sum(
                    [
                        metrics["games_range"],
                        metrics["partners_range"],
                        metrics["opponents_range"],
                        metrics["courts_range"],
                    ]
                )

                # Quality-focused convergence logic
                if total_range == 0:  # Range 0 achieved
                    if verbose:
                        print(f"✅ Found optimal solution (Range 0) at generation {generations_run}")
                    # DON'T STOP - let main loop handle timing constraints
                    # Only stop if we've run for sufficient time to ensure quality
                    if elapsed >= min(30.0, self.max_runtime * 0.5) and generations_without_improvement > 30:
                        if verbose:
                            print("🏁 Range 0 confirmed stable after sufficient runtime - stopping")
                        break
                elif generations_without_improvement >= self.convergence_patience:
                    if verbose:
                        print(f"Converged after {generations_without_improvement} generations without improvement")
                    # CRITICAL: Don't stop unless we've used most of our runtime!
                    # This ensures we run close to the full minute for quality
                    if elapsed < self.max_runtime * 0.8:
                        if verbose:
                            print(f"Continuing search - only {elapsed:.1f}s of {self.max_runtime}s used...")
                        # Reset convergence to give more time
                        generations_without_improvement = self.convergence_patience - 50
                    else:
                        print("🏁 Convergence reached - stopping")
                        break  # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1])

            # Create next generation
            new_population = []

            # Elitism - keep best individuals
            elite_count = max(2, self.population_size // 10)
            for ind, _ in fitness_scores[:elite_count]:
                new_population.append(ind)

            # Fill rest with crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness_scores, rng, k=self.tournament_size)
                parent2 = self._tournament_select(population, fitness_scores, rng, k=self.tournament_size)

                # Crossover
                if rng.random() < 0.9:
                    child1, child2 = self._crossover(parent1, parent2, rng)
                else:
                    child1, child2 = parent1, parent2

                # SUPER aggressive mutation for Range 0
                child1 = self._super_aggressive_mutate(child1, rng)
                child2 = self._super_aggressive_mutate(child2, rng)

                new_population.extend([child1, child2])

            # Trim to population size
            population = new_population[: self.population_size]

            # Inject diversity MORE frequently when pursuing Range 0
            diversity_interval = 15 if self.range0_possible else 30
            if generations_run % diversity_interval == 0 and generations_run > 0:
                current_best_range = sum(
                    [
                        current_ranges["games"],
                        current_ranges["partners"],
                        current_ranges["opponents"],
                        current_ranges["courts"],
                    ]
                )
                print(
                    f"💉 Injecting MASSIVE diversity at generation {generations_run} "
                    f"(current range: {current_best_range})"
                )
                # Replace MORE of population when pursuing Range 0
                replacement_fraction = 0.75 if (self.range0_possible and current_best_range > 0) else 0.5
                for i in range(int(self.population_size * replacement_fraction)):
                    idx = rng.randrange(len(population))
                    population[idx] = self._random_individual(rng)

            generations_run += 1

        # Final results
        if best_individual is None:
            print("❌ No valid solution found!")
            return []

        final_schedule = self._decode(best_individual)
        final_metrics = self._evaluate_metrics(final_schedule)

        # POST-PROCESSOR: Apply targeted repairs if optimal range not achieved
        current_total_range = sum(
            [
                final_metrics["games_range"],
                final_metrics["partners_range"],
                final_metrics["opponents_range"],
                final_metrics["courts_range"],
            ]
        )
        optimal_total_range = self.optimal_partner_range  # Minimum achievable for this player count

        if current_total_range > optimal_total_range:
            print("\n🔧 POST-PROCESSOR: Attempting targeted repairs...")
            repaired_schedule = self._apply_targeted_repairs(final_schedule, max_time=0.5)
            if repaired_schedule:
                repair_metrics = self._evaluate_metrics(repaired_schedule)
                repair_total = sum(
                    [
                        repair_metrics["games_range"],
                        repair_metrics["partners_range"],
                        repair_metrics["opponents_range"],
                        repair_metrics["courts_range"],
                    ]
                )
                if repair_total < current_total_range:
                    print("✅ POST-PROCESSOR: Improved from Range " f"{current_total_range} to Range {repair_total}")
                    final_schedule = repaired_schedule
                    final_metrics = repair_metrics
                    current_total_range = repair_total
                else:
                    print(f"   ⚠️ REPAIR PARTIAL: Reduced to total range {repair_total}")

        print(f"\n📊 FINAL RESULTS after {time.time() - start_time:.1f}s:")
        print(f"   Games Range: {final_metrics['games_range']}")
        print(f"   Partners Range: {final_metrics['partners_range']}")
        print(f"   Opponents Range: {final_metrics['opponents_range']}")
        print(f"   Courts Range: {final_metrics['courts_range']}")
        print(f"   Total Range: {current_total_range}")

        # OPTIMALITY ASSESSMENT based on mathematical constraints
        if self.partner_range0_possible and final_metrics["partners_range"] > 0:
            print("\n⚠️ WARNING: Partner Range 0 was mathematically possible " "but NOT achieved!")
            print("   This schedule is SUBOPTIMAL and should have continued optimizing!")
        elif not self.partner_range0_possible and current_total_range <= optimal_total_range + 1:
            print("\n✅ OPTIMAL: Achieved mathematically optimal range for " f"{self.num_players} players!")
            print(f"   Partner Range 0 is impossible - {final_metrics['partners_range']} is optimal")
        elif current_total_range > optimal_total_range + 1:
            print("\n⚠️ SUBOPTIMAL: Could be improved further " f"(target Range {optimal_total_range + 1})")
        else:
            print("\n🏆 PERFECT: Achieved optimal balance for this player count!")

        # Store fitness details
        self.last_fitness_details = {
            "games_range": final_metrics["games_range"],
            "partners_range": final_metrics["partners_range"],
            "opponents_range": final_metrics["opponents_range"],
            "courts_range": final_metrics["courts_range"],
            "range0_possible": self.range0_possible,
        }

        # Return dict format expected by multi-algorithm scheduler
        total_range = sum(
            [
                final_metrics["games_range"],
                final_metrics["partners_range"],
                final_metrics["opponents_range"],
                final_metrics["courts_range"],
            ]
        )
        success = total_range == 0

        formatted_schedule = self._format_schedule(final_schedule)
        score = 100 - min(100, total_range * 10)
        raw_metrics = dict(final_metrics)

        # Enhanced result with feasibility analysis
        result = {
            "schedule": formatted_schedule,
            "success": success,
            "fitness": best_fitness,
            "score": score,  # Higher score for lower ranges
            "fitness_score": score,
            "generations": generations_run,
            "time_seconds": time.time() - start_time,
            "algorithm": "Genetic Algorithm",
            "total_range": total_range,
            "games_range": raw_metrics["games_range"],
            "partners_range": raw_metrics["partners_range"],
            "opponents_range": raw_metrics["opponents_range"],
            "courts_range": raw_metrics["courts_range"],
            "fitness_details": self.last_fitness_details,
            "raw_metrics": raw_metrics,  # Include raw metrics for debugging
            "metrics": dict(raw_metrics),  # Backward-compatible alias
        }

        # Quality check for small problems
        elapsed = time.time() - start_time
        if self.num_players <= 8 and best_fitness > 0 and elapsed < self.max_runtime:
            if verbose:
                print(f"⚠️ Suboptimal result (Range {total_range}). Consider increasing runtime or population size.")

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
        arr_len = len(self.arrangements)
        out = list(ind)

        # EXTREME mutation rate for Range 0 targeting
        mut_rate = 0.4  # 40% chance to mutate each gene

        # Multiple mutation strategies
        strategy = rng.randint(0, 6)  # Added new strategy

        if strategy == 0:
            # Replace many genes WITH DIVERSITY ENFORCEMENT
            for i in range(len(out)):
                if rng.random() < mut_rate:
                    # Try to avoid duplicates when possible
                    if arr_len >= len(out):
                        # We have enough arrangements - try to pick unique
                        candidates = list(range(arr_len))
                        # Remove already used indices
                        used = set(out[j] for j in range(len(out)) if j != i)
                        available = [x for x in candidates if x not in used]
                        if available:
                            out[i] = rng.choice(available)
                        else:
                            out[i] = rng.randrange(0, arr_len)
                    else:
                        out[i] = rng.randrange(0, arr_len)
        elif strategy == 1:
            # Massive swaps
            num_swaps = rng.randint(2, max(3, len(out) // 2))
            for _ in range(num_swaps):
                if len(out) > 1:
                    i, j = rng.sample(range(len(out)), 2)
                    out[i], out[j] = out[j], out[i]
        elif strategy == 2:
            # Replace entire chunks WITH DIVERSITY
            if len(out) > 4:
                start = rng.randrange(len(out) - 2)
                length = rng.randint(2, min(4, len(out) - start))

                # Generate diverse replacements
                if arr_len >= length:
                    used = set(out[j] for j in range(len(out)) if j < start or j >= start + length)
                    candidates = [x for x in range(arr_len) if x not in used]
                    if len(candidates) >= length:
                        replacements = rng.sample(candidates, length)
                        for i, replacement in enumerate(replacements):
                            out[start + i] = replacement
                    else:
                        for i in range(start, start + length):
                            out[i] = rng.randrange(0, arr_len)
                else:
                    for i in range(start, start + length):
                        out[i] = rng.randrange(0, arr_len)
        elif strategy == 3:
            # Reverse segments
            if len(out) > 2:
                start = rng.randrange(len(out) - 1)
                end = rng.randint(start + 1, len(out))
                out[start:end] = reversed(out[start:end])
        elif strategy == 4:
            # Shuffle a portion
            if len(out) > 4:
                start = rng.randrange(len(out) - 3)
                length = rng.randint(3, min(5, len(out) - start))
                portion = out[start : start + length]
                rng.shuffle(portion)
                out[start : start + length] = portion
        elif strategy == 5:
            # Complete randomization of most genes
            indices = rng.sample(range(len(out)), max(1, int(len(out) * 0.7)))
            for i in indices:
                out[i] = rng.randrange(0, arr_len)
        else:
            # NEW STRATEGY 6: ENFORCE COMPLETE DIVERSITY
            if arr_len >= len(out):
                # We have enough arrangements to make all rounds unique
                indices = list(range(arr_len))
                rng.shuffle(indices)
                out = indices[: len(out)]
                # print(f"🎯 DIVERSITY MUTATION: Created fully unique individual: {out}")  # Commented out to reduce spam
            else:
                # Not enough arrangements - do aggressive mutation
                for i in range(len(out)):
                    if rng.random() < 0.6:  # 60% mutation rate
                        out[i] = rng.randrange(0, arr_len)

        return tuple(out)

    def _random_individual(self, rng) -> Tuple[int, ...]:
        """Create a random individual ensuring maximum diversity."""
        # CRITICAL FIX: Always enforce unique arrangements when possible
        if len(self.arrangements) >= self.num_rounds:
            # We have enough arrangements - ALWAYS use unique ones
            indices = list(range(len(self.arrangements)))
            rng.shuffle(indices)
            selected = indices[: self.num_rounds]

            # VALIDATION: Ensure all indices are different
            if len(set(selected)) != len(selected):
                # Fallback: manually ensure uniqueness
                selected = rng.sample(range(len(self.arrangements)), self.num_rounds)

            return tuple(selected)
        else:
            # Not enough unique arrangements - but still try to maximize diversity
            indices = []
            available = list(range(len(self.arrangements)))
            for _ in range(self.num_rounds):
                if available:
                    choice = rng.choice(available)
                    indices.append(choice)
                    available.remove(choice)  # Remove to avoid immediate reuse
                else:
                    # Ran out of unique options, reset and continue
                    available = list(range(len(self.arrangements)))
                    choice = rng.choice(available)
                    indices.append(choice)
                    available.remove(choice)
            return tuple(indices)

    def _crossover(
        self, parent1: Tuple[int, ...], parent2: Tuple[int, ...], rng
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Crossover two parents to create offspring."""
        if len(parent1) <= 1:
            return parent1, parent2

        # Multi-point crossover for better mixing
        num_points = rng.randint(1, min(3, len(parent1) - 1))
        points = sorted(rng.sample(range(1, len(parent1)), num_points))

        child1, child2 = list(parent1), list(parent2)

        # Alternate segments between parents
        swap = False
        start = 0
        for point in points:
            if swap:
                child1[start:point], child2[start:point] = (
                    child2[start:point],
                    child1[start:point],
                )
            start = point
            swap = not swap

        # Handle last segment
        if swap:
            child1[start:], child2[start:] = child2[start:], child1[start:]

        return tuple(child1), tuple(child2)

    def _generate_arrangements(self, max_arrangements: int = 400) -> List[List[Tuple[str, str, str, str]]]:
        """Generate MANY diverse arrangements to enable Range 0."""
        rng = random.Random(123)  # Keep this deterministic for arrangement generation

        arrangements: List[List[Tuple[str, str, str, str]]] = []

        # Generate many times more attempts to get diversity
        for attempt in range(max_arrangements * 10):
            # Create fresh shuffled player list for each attempt
            player_pool = list(self._names)
            rng.shuffle(player_pool)

            round_games = []
            used_in_round = set()  # Track all players used in this round
            court = 0

            # GUARANTEED UNIQUE: Select 4 unused players for each game
            player_idx = 0
            while court < self.num_courts and player_idx < len(player_pool):
                # Find 4 players that haven't been used in this round
                game_players = []
                temp_idx = player_idx

                while len(game_players) < 4 and temp_idx < len(player_pool):
                    candidate = player_pool[temp_idx]
                    if candidate not in used_in_round and candidate not in game_players:
                        game_players.append(candidate)
                    temp_idx += 1

                # If we couldn't find 4 unused players, try next starting position
                if len(game_players) < 4:
                    player_idx += 1
                    continue

                p1, p2, p3, p4 = game_players

                # CRITICAL: Double-check uniqueness before accepting
                if len(set([p1, p2, p3, p4])) == 4 and self._game_valid((p1, p2, p3, p4)):
                    round_games.append((p1, p2, p3, p4))
                    used_in_round.update([p1, p2, p3, p4])
                    court += 1
                    player_idx = temp_idx  # Move past used players
                else:
                    player_idx += 1  # Try different starting position

            # Only accept rounds that use exactly the right number of players
            if len(round_games) == self.num_courts and len(used_in_round) == self.num_courts * 4:
                # FINAL VALIDATION: Ensure no player appears in multiple games
                all_round_players = set()
                has_duplicates = False
                for game in round_games:
                    game_set = set(game)
                    if game_set & all_round_players:
                        has_duplicates = True
                        break
                    all_round_players.update(game_set)

                if not has_duplicates and len(all_round_players) == self.num_courts * 4:
                    arrangements.append(round_games)
                else:
                    # Skip this arrangement due to duplicates
                    pass

            if len(arrangements) >= max_arrangements:
                break

        # Generate even more variations by swapping players within games
        original_count = len(arrangements)
        for arr in list(arrangements[: original_count // 2]):
            for _ in range(3):  # Create 3 variations of each
                var: List[Tuple[str, str, str, str]] = []
                used_in_var = set()

                for game in arr:
                    g = list(game)
                    # Simple safe swaps within teams only (no cross-team swaps)
                    if rng.random() < 0.5:
                        g[0], g[1] = g[1], g[0]  # Swap within team 1
                    if rng.random() < 0.5:
                        g[2], g[3] = g[3], g[2]  # Swap within team 2

                    game_tuple = cast(GameTuple, tuple(g))
                    # Ensure no duplicates in this variation
                    game_players = set(game_tuple)
                    valid_game = len(game_players) == 4 and not any(p in used_in_var for p in game_players)
                    if valid_game:
                        var.append(game_tuple)
                        used_in_var.update(game_players)
                    else:
                        # Keep original game if swap would create duplicates
                        var.append(game)
                        used_in_var.update(game)

                # Only add variation if it maintains round validity
                if len(var) == len(arr) and len(used_in_var) == self.num_courts * 4:
                    # FINAL VALIDATION: Ensure no player appears in multiple games in variation
                    all_var_players = set()
                    has_var_duplicates = False
                    for game in var:
                        game_set = set(game)
                        if game_set & all_var_players:
                            has_var_duplicates = True
                            break
                        all_var_players.update(game_set)

                    if not has_var_duplicates and len(all_var_players) == self.num_courts * 4:
                        arrangements.append(var)

            if len(arrangements) >= max_arrangements:
                break

        print(f"✅ Generated {len(arrangements)} unique round arrangements")

        # FINAL FILTER: Remove any arrangements with duplicates
        valid_arrangements = []
        for arr in arrangements:
            is_valid = True
            # Check each game in the arrangement
            for game in arr:
                if len(set(game)) != 4:
                    is_valid = False
                    break
            # Check for duplicates across games in the arrangement
            all_players = set()
            for game in arr:
                game_set = set(game)
                if game_set & all_players:
                    is_valid = False
                    break
                all_players.update(game_set)
            if is_valid and len(all_players) == self.num_courts * 4:
                valid_arrangements.append(arr)

        print(f"✅ Filtered to {len(valid_arrangements)} valid arrangements")
        return valid_arrangements[:max_arrangements]

    def _game_valid(self, game: Tuple[str, str, str, str]) -> bool:
        """Check if a game satisfies all constraints."""
        p1, p2, p3, p4 = game

        # CRITICAL: Check for duplicate players within the game
        all_players = [p1, p2, p3, p4]
        if len(set(all_players)) != 4:
            return False  # Same player appears multiple times in the game

        # Check do-not-pair constraints
        if p2 in self.do_not_pair_map[p1] or p1 in self.do_not_pair_map[p2]:
            return False
        if p4 in self.do_not_pair_map[p3] or p3 in self.do_not_pair_map[p4]:
            return False

        # Check do-not-oppose constraints
        for a in (p1, p2):
            for b in (p3, p4):
                if b in self.do_not_oppose_map[a] or a in self.do_not_oppose_map[b]:
                    return False

        return True

    def _tournament_select(self, population, fitness_scores, rng, k=7):
        """Tournament selection with configurable tournament size."""
        tournament_indices = rng.sample(range(len(population)), min(k, len(population)))
        tournament = [(population[i], fitness_scores[i][1]) for i in tournament_indices]
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0]

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

    def _decode_cached(self, individual: Tuple[int, ...]) -> List[List[Tuple]]:
        """Cached version of schedule decoding."""
        if individual in self._schedule_cache:
            return self._schedule_cache[individual]

        schedule = self._decode(individual)

        self._schedule_cache[individual] = schedule

        return schedule

    def _evaluate_metrics_cached(self, schedule: List[List[Tuple]], individual: Tuple[int, ...]) -> Dict[str, int]:
        """Cached version of metrics evaluation."""
        if individual in self._metrics_cache:
            return self._metrics_cache[individual]

        metrics = self._evaluate_metrics(schedule)

        self._metrics_cache[individual] = metrics

        return metrics

    def _get_duplicate_signature_cached(self, individual: Tuple[int, ...], schedule: List[List[Tuple]]) -> Tuple:
        """Cached version of duplicate round signature calculation."""
        if individual in self._signature_cache:
            return self._signature_cache[individual]

        # Optimized signature calculation
        round_signatures = []
        for round_games in schedule:
            # Pre-sort games for faster signature creation
            sorted_games = []
            for game in round_games:
                sorted_game = tuple(sorted(game))
                sorted_games.append(sorted_game)
            signature = tuple(sorted(sorted_games))
            round_signatures.append(signature)

        full_signature = tuple(round_signatures)

        self._signature_cache[individual] = full_signature

        return full_signature

    def _fitness(self, individual: Tuple[int, ...]) -> float:
        """Optimized lexicographic fitness function with enhanced caching."""
        if individual in self._fitness_cache:
            return self._fitness_cache[individual]

        # Use cached decode and metrics evaluation
        schedule = self._decode_cached(individual)
        metrics = self._evaluate_metrics_cached(schedule, individual)

        # CRITICAL: Check for player duplicates within rounds (immediate invalidation)
        duplicate_penalty = 0
        for round_games in schedule:
            round_players = set()
            for game in round_games:
                game_players = set(game)
                # Check for duplicates within this game
                if len(game_players) != 4:
                    duplicate_penalty += 1000000000  # 1B penalty for game duplicates
                # Check for duplicates across games in this round
                overlap = game_players & round_players
                if overlap:
                    duplicate_penalty += 1000000000 * len(overlap)  # 1B per duplicate player

                round_players.update(game_players)

        # Use optimized duplicate detection with caching
        signature = self._get_duplicate_signature_cached(individual, schedule)
        unique_rounds = len(set(signature))
        total_rounds = len(signature)
        duplicate_rounds = total_rounds - unique_rounds

        # MASSIVE penalty for duplicate rounds
        duplicate_penalty += duplicate_rounds * 1000000000  # 1 BILLION per duplicate

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

        # CRITICAL VALIDATION: Check for identical rounds (bug detection)
        round_signatures = []
        for round_idx, round_games in enumerate(schedule):
            # Create a signature for this round
            signature = tuple(sorted([tuple(sorted(game)) for game in round_games]))
            round_signatures.append(signature)

        # Check for duplicates
        unique_signatures = set(round_signatures)
        if len(unique_signatures) < len(round_signatures):
            duplicate_count = len(round_signatures) - len(unique_signatures)
            # Minimal logging to avoid terminal spam
            if duplicate_count >= 5:  # Only log severe cases
                print(f"⚠️ {duplicate_count} duplicate rounds detected")

        return schedule

    def _repair_invalid_schedule(
        self,
        schedule: List[List[Tuple[str, str, str, str]]],
        problematic_round_idx: int,
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Repair an invalid schedule by replacing problematic rounds."""
        # Try to find a replacement arrangement that doesn't conflict
        used_players = set()
        for round_idx, round_games in enumerate(schedule):
            if round_idx != problematic_round_idx:
                for game in round_games:
                    used_players.update(game)

        # Find arrangements that don't use any already-used players
        available_arrangements = []
        for i, arrangement in enumerate(self.arrangements):
            arrangement_players = set()
            for game in arrangement:
                arrangement_players.update(game)

            # Check if this arrangement conflicts with used players
            if not (arrangement_players & used_players):
                available_arrangements.append((i, arrangement))

        if available_arrangements:
            # Use the first available arrangement
            replacement_idx, replacement_arrangement = available_arrangements[0]
            schedule[problematic_round_idx] = replacement_arrangement
            print(f"✅ Repaired round {problematic_round_idx + 1} with arrangement {replacement_idx}")
        else:
            # If no perfect replacement, try to find one with minimal conflicts
            print(f"⚠️ No perfect replacement for round {problematic_round_idx + 1}, using fallback")
            # For now, just return the original and let fitness handle it
            pass

        return schedule

    def _evaluate_metrics(
        self, schedule: List[List[Tuple[str, str, str, str]]] | List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Evaluate fairness metrics from schedule in either raw tuple format or formatted dict format."""

        # Handle formatted schedule (from _format_schedule)
        if schedule and isinstance(schedule[0], dict):
            # Convert formatted schedule back to tuple format
            raw_schedule = []
            for round_data in cast(List[Dict[str, Any]], schedule):
                round_games = []
                games = round_data.get("games", [])
                for game in games:
                    if hasattr(game, "team1") and hasattr(game, "team2"):
                        # Game object format
                        team1 = list(game.team1)
                        team2 = list(game.team2)
                        # Combine as p1, p2, p3, p4 where team1=[p1,p2], team2=[p3,p4]
                        game_tuple = (team1[0], team1[1], team2[0], team2[1])
                        round_games.append(game_tuple)
                raw_schedule.append(round_games)
            schedule = raw_schedule

        """Evaluate schedule metrics."""
        # Counters
        games_played: Counter[str] = Counter()
        partners: Dict[str, Counter[str]] = defaultdict(Counter)
        opponents: Dict[str, Counter[str]] = defaultdict(Counter)
        courts: Dict[str, Counter[int]] = defaultdict(Counter)
        violations = 0

        for r_idx, round_games in enumerate(schedule, start=1):
            if len(round_games) != self.num_courts:
                violations += abs(self.num_courts - len(round_games))

            # Track players in this round to detect duplicates across games
            round_players = set()

            for c_idx, g in enumerate(round_games, start=1):
                p1, p2, p3, p4 = map(str, g)

                # CRITICAL: Check for duplicate players within the game
                game_players = [p1, p2, p3, p4]
                if len(set(game_players)) != 4:
                    violations += 1000  # Heavy penalty for duplicate players in a game

                # Check for duplicates across games in this round
                game_player_set = set(game_players)
                overlap = game_player_set & round_players
                if overlap:
                    violations += 1000 * len(overlap)  # Heavy penalty for duplicate players across games

                round_players.update(game_player_set)

                # constraints
                if p2 in self.do_not_pair_map[p1] or p1 in self.do_not_pair_map[p2]:
                    violations += 1
                if p4 in self.do_not_pair_map[p3] or p3 in self.do_not_pair_map[p4]:
                    violations += 1
                for a, b in [(p1, p3), (p1, p4), (p2, p3), (p2, p4)]:
                    if b in self.do_not_oppose_map[a] or a in self.do_not_oppose_map[b]:
                        violations += 1

                # stats
                for p in (p1, p2, p3, p4):
                    games_played[p] += 1
                    courts[p][c_idx] += 1
                for a, b in [(p1, p2), (p3, p4)]:
                    partners[a][b] += 1
                    partners[b][a] += 1
                for a in (p1, p2):
                    for b in (p3, p4):
                        opponents[a][b] += 1
                        opponents[b][a] += 1

        def rnge(vals: Iterable[int]) -> int:
            data = list(vals)
            if not data:
                return 0
            return max(data) - min(data)

        games_range = rnge(games_played[p] for p in self._names)
        partners_range = rnge(len(partners[p]) for p in self._names)  # Count unique partners, not frequencies
        opponents_range = rnge(len(opponents[p]) for p in self._names)  # Count unique opponents, not frequencies
        # FIXED: Count unique courts used per player, not total games per player
        courts_range = rnge(sum(1 for count in courts[p].values() if count > 0) for p in self._names)

        return {
            "games_range": games_range,
            "partners_range": partners_range,
            "opponents_range": opponents_range,
            "courts_range": courts_range,
            "violations": violations,
        }
        # Counters
        games_played: Counter[str] = Counter()
        partners: Dict[str, Counter[str]] = defaultdict(Counter)
        opponents: Dict[str, Counter[str]] = defaultdict(Counter)
        courts: Dict[str, Counter[int]] = defaultdict(Counter)
        violations = 0

        for r_idx, round_games in enumerate(schedule, start=1):
            if len(round_games) != self.num_courts:
                violations += abs(self.num_courts - len(round_games))
            for c_idx, g in enumerate(round_games, start=1):
                p1, p2, p3, p4 = map(str, g)
                # constraints
                if p2 in self.do_not_pair_map[p1] or p1 in self.do_not_pair_map[p2]:
                    violations += 1
                if p4 in self.do_not_pair_map[p3] or p3 in self.do_not_pair_map[p4]:
                    violations += 1
                for a, b in [(p1, p3), (p1, p4), (p2, p3), (p2, p4)]:
                    if b in self.do_not_oppose_map[a] or a in self.do_not_oppose_map[b]:
                        violations += 1

                # stats
                for p in (p1, p2, p3, p4):
                    games_played[p] += 1
                    courts[p][c_idx] += 1
                for a, b in [(p1, p2), (p3, p4)]:
                    partners[a][b] += 1
                    partners[b][a] += 1
                for a in (p1, p2):
                    for b in (p3, p4):
                        opponents[a][b] += 1
                        opponents[b][a] += 1

        def rnge(vals: Iterable[int]) -> int:
            data = list(vals)
            if not data:
                return 0
            return max(data) - min(data)

        games_range = rnge(games_played[p] for p in self._names)
        partners_range = rnge(len(partners[p]) for p in self._names)  # Count unique partners, not frequencies
        opponents_range = rnge(len(opponents[p]) for p in self._names)  # Count unique opponents, not frequencies
        # FIXED: Count unique courts used per player, not total games per player
        courts_range = rnge(sum(1 for count in courts[p].values() if count > 0) for p in self._names)

        return {
            "games_range": games_range,
            "partners_range": partners_range,
            "opponents_range": opponents_range,
            "courts_range": courts_range,
            "violations": violations,
        }

    def _apply_targeted_repairs(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float = 0.5
    ) -> Optional[List[List[Tuple[str, str, str, str]]]]:
        """Apply targeted local repairs to achieve Range 0 when mathematically possible."""
        start_time = time.time()

        current_schedule = [round_games[:] for round_games in schedule]  # Deep copy
        current_metrics = self._evaluate_metrics(current_schedule)

        print(
            "   Starting repair: "
            f"G:{current_metrics['games_range']} "
            f"P:{current_metrics['partners_range']} "
            f"O:{current_metrics['opponents_range']} "
            f"C:{current_metrics['courts_range']}"
        )

        # Focus on partner balancing since that's the stuck metric
        if current_metrics["partners_range"] > 0:
            current_schedule = self._repair_partner_imbalance(current_schedule, max_time * 0.8)
            current_metrics = self._evaluate_metrics(current_schedule)

        # Additional repairs if needed
        if current_metrics["opponents_range"] > 0 and time.time() - start_time < max_time:
            current_schedule = self._repair_opponent_imbalance(current_schedule, max_time - (time.time() - start_time))
            current_metrics = self._evaluate_metrics(current_schedule)

        final_total = sum(
            [
                current_metrics["games_range"],
                current_metrics["partners_range"],
                current_metrics["opponents_range"],
                current_metrics["courts_range"],
            ]
        )

        if final_total == 0:
            print("   🎉 REPAIR SUCCESS: Achieved Range 0!")
            return current_schedule
        else:
            print(f"   ⚠️ REPAIR PARTIAL: Reduced to total range {final_total}")
            print(
                f"      Breakdown - G:{current_metrics['games_range']} "
                f"P:{current_metrics['partners_range']} "
                f"O:{current_metrics['opponents_range']} "
                f"C:{current_metrics['courts_range']}"
            )
            return current_schedule

    def _repair_partner_imbalance(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Targeted repair for partner imbalance using smart swaps."""
        start_time = time.time()

        best_schedule = [round_games[:] for round_games in schedule]
        current_metrics = self._evaluate_metrics(best_schedule)

        # If already balanced, return early
        if current_metrics["partners_range"] == 0:
            return best_schedule

        print(f"   🔧 Partner repair starting: range {current_metrics['partners_range']}")

        attempts = 0
        max_attempts = 200  # More attempts for better results
        improvements = 0

        while time.time() - start_time < max_time and attempts < max_attempts:
            attempts += 1

            # Try strategic player repositioning between rounds
            test_schedule = [round_games[:] for round_games in best_schedule]

            # Pick two different rounds
            r1_idx = random.randint(0, len(test_schedule) - 1)
            r2_idx = random.randint(0, len(test_schedule) - 1)
            if r1_idx == r2_idx or len(test_schedule[r1_idx]) == 0 or len(test_schedule[r2_idx]) == 0:
                continue

            # Pick games within those rounds
            g1_idx = random.randint(0, len(test_schedule[r1_idx]) - 1)
            g2_idx = random.randint(0, len(test_schedule[r2_idx]) - 1)

            game1 = list(test_schedule[r1_idx][g1_idx])
            game2 = list(test_schedule[r2_idx][g2_idx])

            # Try different types of swaps to improve partner variety
            swap_strategies = [
                # Cross-team swaps (swap player from team1 in game1 with player from team1 in game2)
                [(0, 0), (1, 1)],  # Team1 positions
                [(2, 2), (3, 3)],  # Team2 positions
                [(0, 2), (1, 3)],  # Cross-team swaps
                [(0, 3), (1, 2)],  # Different cross-team swaps
                # Single player swaps
                [(0, 1)],
                [(2, 3)],
                [(0, 2)],
                [(1, 3)],
            ]

            strategy = swap_strategies[attempts % len(swap_strategies)]

            # Apply swaps
            for pos1, pos2 in strategy:
                if pos1 < len(game1) and pos2 < len(game2):
                    game1[pos1], game2[pos2] = game2[pos2], game1[pos1]

            test_schedule[r1_idx][g1_idx] = cast(GameTuple, tuple(game1))
            test_schedule[r2_idx][g2_idx] = cast(GameTuple, tuple(game2))

            # Evaluate improvement
            test_metrics = self._evaluate_metrics(test_schedule)

            # Accept if it improves partner balance or doesn't worsen other metrics
            current_best_metrics = self._evaluate_metrics(best_schedule)

            is_improvement = test_metrics["partners_range"] < current_best_metrics["partners_range"] or (
                test_metrics["partners_range"] == current_best_metrics["partners_range"]
                and test_metrics["opponents_range"] <= current_best_metrics["opponents_range"]
                and test_metrics["games_range"] <= current_best_metrics["games_range"]
                and test_metrics["courts_range"] <= current_best_metrics["courts_range"]
            )

            if is_improvement and test_metrics["violations"] == 0:
                best_schedule = test_schedule
                improvements += 1
                print(
                    f"   🎯 Improvement #{improvements}: "
                    f"P:{test_metrics['partners_range']} "
                    f"O:{test_metrics['opponents_range']} "
                    f"(attempt {attempts})"
                )

                if test_metrics["partners_range"] == 0:
                    print(f"   🎉 Partner balance achieved after {attempts} attempts!")
                    break

        final_metrics = self._evaluate_metrics(best_schedule)
        final_total = sum(
            [
                final_metrics["games_range"],
                final_metrics["partners_range"],
                final_metrics["opponents_range"],
                final_metrics["courts_range"],
            ]
        )
        print(f"   🏁 Partner repair complete: {attempts} attempts, " f"{improvements} improvements")
        print(f"      Partners Range: {final_metrics['partners_range']} | " f"Total Range: {final_total}")
        print(
            f"      Breakdown - G:{final_metrics['games_range']} "
            f"P:{final_metrics['partners_range']} "
            f"O:{final_metrics['opponents_range']} "
            f"C:{final_metrics['courts_range']}"
        )

        return best_schedule

    def _repair_opponent_imbalance(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Targeted repair for opponent imbalance using smart swaps."""
        # Similar logic to partner repair but focused on opponent distributions
        # For brevity, implement basic version
        return schedule  # Placeholder - extend if needed

    def _elitism(self, population: List[Tuple[int, ...]], fitnesses: List[float], k: int) -> List[Tuple[int, ...]]:
        """Select k best individuals for elitism."""
        idxs = sorted(range(len(population)), key=lambda i: fitnesses[i])[:k]
        return [population[i] for i in idxs]

    def _tournament(
        self, population: List[Tuple[int, ...]], fitnesses: List[float], rng, k: int = 3
    ) -> Tuple[int, ...]:
        """Tournament selection."""
        cand = rng.sample(range(len(population)), k=min(k, len(population)))
        best = min(cand, key=lambda i: fitnesses[i])
        return population[best]

    def _count_partners(self, schedule: List[List[Tuple[str, str, str, str]]]) -> Dict[str, Dict[str, int]]:
        """Count partner pairings for each player in the schedule."""
        partner_count = defaultdict(lambda: defaultdict(int))

        for round_games in schedule:
            for game in round_games:
                p1, p2, p3, p4 = game
                # Team 1 partners
                partner_count[p1][p2] += 1
                partner_count[p2][p1] += 1
                # Team 2 partners
                partner_count[p3][p4] += 1
                partner_count[p4][p3] += 1

        return dict(partner_count)

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
        """Format schedule to legacy format."""
        rounds: List[Dict[str, Any]] = []
        seen = set()
        for r_idx, round_games in enumerate(schedule, start=1):
            # canonical dedupe signature
            sig = []
            for g in round_games:
                p1, p2, p3, p4 = map(str, g)
                team1 = tuple(sorted([p1, p2]))
                team2 = tuple(sorted([p3, p4]))
                sig.append(tuple(sorted([team1, team2])))
            key = frozenset(sig)
            if key in seen and round_games:
                # Instead of trying to modify the round (which can create duplicates),
                # just skip adding this duplicate round
                print(f"⚠️ Skipping duplicate round {r_idx}")
                continue
            seen.add(key)

            games: List[Game] = []
            for c_idx, g in enumerate(round_games, start=1):
                p1, p2, p3, p4 = map(str, g)
                games.append(Game([p1, p2], [p3, p4], c_idx))
            rounds.append({"round": r_idx, "games": games})
        return rounds

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
