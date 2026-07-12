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
        avoidable_duplicate_rounds_from_signature,
        count_avoidable_duplicate_rounds,
        count_duplicate_rounds,
        evaluate_metrics_from_arrangement_stats,
        evaluate_schedule_metrics,
        precompute_arrangement_stats,
        round_signature,
    )
except ImportError:
    from algorithms.scheduler_metrics import (
        avoidable_duplicate_rounds_from_signature,
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
        repair_opponent_imbalance,
        repair_partner_imbalance,
    )
except ImportError:
    from algorithms.scheduler_repairs import (
        apply_targeted_repairs,
        repair_opponent_imbalance,
        repair_partner_imbalance,
    )

try:
    from src.algorithms.scheduler_evolution import (
        compute_general_stop_threshold,
        compute_perfect_stop_threshold,
    )
    from src.algorithms.scheduler_evolution import crossover as _crossover_fn
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
    from src.algorithms.constructive_scheduler import build_perfect_schedule as _build_perfect_schedule_fn
except ImportError:
    from algorithms.constructive_scheduler import build_perfect_schedule as _build_perfect_schedule_fn

try:
    from src.utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
except ImportError:
    try:
        from utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
    except ImportError:
        ScheduleFeasibilityAnalyzer = None  # type: ignore[assignment,misc]


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

    # Defensive bound on the per-run evaluation caches (_fitness_cache,
    # _metrics_cache, _schedule_cache, _signature_cache). A caller-requested
    # max_generations budget (see _resolve_ga_sizing) can run long enough for
    # these to otherwise grow unbounded for the lifetime of one
    # generate_schedule() call - past this many distinct individuals seen,
    # they're cleared together rather than left to grow forever.
    _MAX_CACHE_ENTRIES = 300_000

    # Fixed court assignment based on player count
    @classmethod
    def get_fixed_courts(cls, num_players: int) -> int:
        """Calculate fixed court count based on player count."""
        return max(1, num_players // 4)

    def _resolve_ga_sizing(self, use_desktop_params, num_players, population_size, max_generations, max_runtime):
        """Resolve population_size/max_generations/max_runtime for desktop mode.

        max_generations/max_runtime of None mean "the caller has no explicit
        preference" - resolved to the desktop-tier default (or the plain
        constructor default outside desktop mode). An explicit value from the
        caller is never silently discarded: in desktop mode it's combined with
        the feasibility analyzer's tier via max(), so a caller asking for a
        longer budget than the analyzer's tier actually gets it, while a
        caller asking for less still benefits from the analyzer's recommended
        floor.

        Returns (population_size, max_generations, max_runtime, desktop_tunables),
        where desktop_tunables is the optimal_params/DESKTOP_PARAMS dict to apply
        verbatim via _resolve_ga_tunables when use_desktop_params is True, else None.
        """
        if not use_desktop_params:
            return (
                population_size,
                max_generations if max_generations is not None else 1000,
                max_runtime if max_runtime is not None else 300.0,
                None,
            )

        if ScheduleFeasibilityAnalyzer is not None:
            optimal_params = ScheduleFeasibilityAnalyzer.get_optimal_parameters(num_players)
            resolved_generations = optimal_params["max_generations"]
            if max_generations is not None:
                resolved_generations = max(resolved_generations, max_generations)
            resolved_runtime = optimal_params["max_runtime"]
            if max_runtime is not None:
                resolved_runtime = max(resolved_runtime, max_runtime)
            return (
                optimal_params["population_size"],
                resolved_generations,
                resolved_runtime,
                optimal_params,
            )

        # Fallback to original desktop params - max_generations/max_runtime are
        # intentionally left as the caller's values here, matching this
        # fallback's long-standing behavior (only population_size and the
        # tunables below come from DESKTOP_PARAMS in this branch).
        return (
            self.DESKTOP_PARAMS["population_size"],
            max_generations if max_generations is not None else self.DESKTOP_PARAMS["max_generations"],
            max_runtime if max_runtime is not None else 300.0,
            self.DESKTOP_PARAMS,
        )

    def _resolve_ga_tunables(self, use_desktop_params, desktop_tunables, population_size):
        """Resolve mutation_rate/crossover_rate/elite_size/tournament_size/
        convergence_patience for both desktop and non-desktop modes, so every
        GA tunable is always set here rather than via a trailing
        hasattr(self, "convergence_patience") fallback."""
        if use_desktop_params and desktop_tunables is not None:
            return {
                "mutation_rate": desktop_tunables["mutation_rate"],
                "crossover_rate": desktop_tunables["crossover_rate"],
                "convergence_patience": desktop_tunables["convergence_patience"],
                "elite_size": desktop_tunables["elite_size"],
                "tournament_size": desktop_tunables["tournament_size"],
            }
        return {
            "mutation_rate": 0.20,  # Increased from 0.12 for better exploration
            "crossover_rate": 0.85,  # Increased from 0.8
            "convergence_patience": 60,
            "elite_size": max(4, population_size // 5),  # Keep more elites
            "tournament_size": 7,
        }

    def __init__(
        self,
        players: List[Any],
        num_courts: int = 1,
        num_rounds: Optional[int] = None,
        *,
        population_size: int = 100,  # Will use DESKTOP_PARAMS for desktop mode
        max_generations: Optional[int] = None,  # None = no explicit preference; see _resolve_ga_sizing
        max_runtime: Optional[float] = None,  # None = no explicit preference; see _resolve_ga_sizing
        min_runtime: float | None = None,
        use_desktop_params: bool = False,  # Enable desktop optimization
        progress_callback: Optional[Callable[..., object]] = None,
    ) -> None:
        # Validate inputs
        if int(num_courts) < 1:
            raise ValueError("num_courts must be >= 1")

        # Desktop parameter optimization with feasibility analysis
        population_size, max_generations, max_runtime, desktop_tunables = self._resolve_ga_sizing(
            use_desktop_params, len(players), population_size, max_generations, max_runtime
        )

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

        # num_courts is always the caller's explicit choice (physical courts
        # available cannot be inferred from player count alone - a club with
        # 16 players and 3 courts is a real, common case). Desktop mode only
        # affects population/mutation tuning via _resolve_ga_sizing, never
        # how many courts are used.
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

        # GA tunables - resolved here for both desktop and non-desktop modes,
        # so every tunable (including convergence_patience) is always set.
        tunables = self._resolve_ga_tunables(use_desktop_params, desktop_tunables, self.population_size)
        self.mutation_rate = tunables["mutation_rate"]
        self.crossover_rate = tunables["crossover_rate"]
        self.convergence_patience = tunables["convergence_patience"]
        self.elite_size = tunables["elite_size"]
        self.tournament_size = tunables["tournament_size"]

        # Constraints
        self.do_not_pair_map: Dict[str, set[str]] = defaultdict(set)
        self.do_not_oppose_map: Dict[str, set[str]] = defaultdict(set)

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

        if ScheduleFeasibilityAnalyzer is not None:
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
        self.time_budget: float = float(self.max_runtime)

    # ---------------- Public API ----------------
    def add_constraints(
        self,
        pair_constraints: List[Tuple[str, str]] | None = None,
        oppose_constraints: List[Tuple[str, str]] | None = None,
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

        # A verified constructive design (see constructive_scheduler.py) may
        # exist for this exact player count/court/round combination. With no
        # active constraints it is already optimal, so skip the GA entirely;
        # with constraints active, use it to seed the initial population
        # instead of starting purely random.
        seed_individual = self._build_seed_individual()
        has_active_constraints = any(self.do_not_pair_map.values()) or any(self.do_not_oppose_map.values())

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

        if seed_individual is not None and not has_active_constraints:
            print("✨ Using a verified constructive design - already optimal, skipping search.")
            best_individual = seed_individual
            best_fitness = self._fitness(best_individual)
            generations_run = 0
        else:
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
                seed_individuals=[seed_individual] if seed_individual is not None else None,
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
                    print(f"   ⚠️ REPAIR PARTIAL: No improvement - still total range {repair_total}")

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
        if ScheduleFeasibilityAnalyzer is not None:
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

    def _build_seed_individual(self) -> Tuple[int, ...] | None:
        """Build a GA individual from a verified constructive design (see
        constructive_scheduler.py) for this player count, or None if no
        design applies here - in which case the caller falls back to a fully
        random initial population exactly as before this existed.

        With no active do_not_pair/do_not_oppose constraints, the returned
        individual decodes to a schedule that is already optimal (range 0),
        so the caller can use it directly instead of running the GA at all.
        With constraints active, up to 20 random player-to-group reorderings
        are tried until one produces a schedule where every game satisfies
        the constraints; if none do, seeding is skipped (None) rather than
        forcing an invalid schedule into the population.

        Only extends self.arrangements (and refreshes the pool stats derived
        from it) when a usable design is actually found - a no-op on every
        unsupported player count/court/round combination.
        """
        if self.num_players != self.num_courts * 4:
            return None  # the design assumes every player plays every round

        # Shuffle which physical player fills each design slot so repeated,
        # unseeded calls return varied schedules instead of the byte-identical
        # one every time - the construction is optimal for any name ordering,
        # so this can't regress range/fairness. A fixed seed still reproduces
        # the exact same shuffle, so seed-replay stays deterministic.
        rng = getattr(self, "_run_rng", None) or random.Random(0)
        names = list(self._names)
        rng.shuffle(names)
        probe = _build_perfect_schedule_fn(names, self.num_courts, self.num_rounds)
        if probe is None:
            return None  # no verified design for this (courts, rounds) combination

        has_constraints = any(self.do_not_pair_map.values()) or any(self.do_not_oppose_map.values())
        if not has_constraints:
            schedule = probe
        else:
            schedule = None
            for _ in range(20):
                candidate_names = rng.sample(names, len(names))
                candidate = _build_perfect_schedule_fn(candidate_names, self.num_courts, self.num_rounds)
                if candidate and all(self._game_valid(game) for round_games in candidate for game in round_games):
                    schedule = candidate
                    break
            if schedule is None:
                return None  # constraints conflict with the design in every tried ordering

        assert schedule is not None
        # Reuse existing pool slots for rounds already present (by signature)
        # instead of blindly appending - otherwise a long-lived instance whose
        # caller replays the same seed repeatedly (or regenerates without a
        # seed) grows self.arrangements by num_rounds on every single call.
        signature_to_index = {
            self._round_signature(arrangement): idx for idx, arrangement in enumerate(self.arrangements)
        }
        new_indices = []
        pool_grew = False
        for round_games in schedule:
            signature = self._round_signature(round_games)
            idx = signature_to_index.get(signature)
            if idx is None:
                idx = len(self.arrangements)
                self.arrangements.append(round_games)
                signature_to_index[signature] = idx
                pool_grew = True
            new_indices.append(idx)

        if pool_grew:
            self._refresh_arrangement_stats()
            self.max_unique_rounds = len({self._round_signature(arrangement) for arrangement in self.arrangements})
            self.minimum_duplicate_rounds = max(0, self.num_rounds - self.max_unique_rounds)
        return tuple(new_indices)

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

    def _enforce_cache_size_limit(self) -> None:
        """Clear the per-run evaluation caches together once they've grown
        past _MAX_CACHE_ENTRIES, so a long-running search (large
        max_generations) can't grow them without bound. A full clear (rather
        than per-cache LRU eviction) keeps the four caches - which all key on
        the same individual tuples - always in sync with each other."""
        if len(self._fitness_cache) <= self._MAX_CACHE_ENTRIES:
            return
        self._fitness_cache.clear()
        self._metrics_cache.clear()
        self._schedule_cache.clear()
        self._signature_cache.clear()

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
        avoidable_duplicate_rounds = avoidable_duplicate_rounds_from_signature(signature, self.minimum_duplicate_rounds)

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
        self._enforce_cache_size_limit()

        return penalty

    def _decode(self, individual: Tuple[int, ...]) -> List[List[Tuple[str, str, str, str]]]:
        """Decode individual to schedule."""
        schedule = [self.arrangements[idx] for idx in individual]
        return schedule

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
            rng=getattr(self, "_run_rng", None),
            clock=getattr(self, "_run_clock", None),
        )

    def _format_schedule(self, schedule: List[List[Tuple[str, str, str, str]]]) -> List[Dict[str, Any]]:
        """Format schedule to legacy format without silently dropping rounds."""
        return format_schedule(schedule)
