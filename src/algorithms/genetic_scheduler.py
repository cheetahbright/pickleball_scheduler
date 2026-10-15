#!/usr/bin/env python3
"""High-Performance Genetic Pickleball Scheduler - Optimized for speed."""

import builtins
import os
import random
import sys
from collections import defaultdict
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

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
        partner_repeat_floor,
        partner_repeat_score,
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
        partner_repeat_floor,
        partner_repeat_score,
        precompute_arrangement_stats,
        round_signature,
    )

try:
    from src.algorithms.scheduler_repairs import (
        apply_targeted_repairs,
        games_range_floor,
        order_rounds_for_rest,
        repair_games_balance,
        repair_opponent_imbalance,
        repair_partner_imbalance,
    )
except ImportError:
    from algorithms.scheduler_repairs import (
        apply_targeted_repairs,
        games_range_floor,
        order_rounds_for_rest,
        repair_games_balance,
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
    from src.algorithms.scheduler_evolution import run_evolution_loop
    from src.algorithms.scheduler_evolution import super_aggressive_mutate as _super_aggressive_mutate_fn
    from src.algorithms.scheduler_evolution import tournament_select as _tournament_select_fn
except ImportError:
    from algorithms.scheduler_evolution import (
        compute_general_stop_threshold,
        compute_perfect_stop_threshold,
    )
    from algorithms.scheduler_evolution import crossover as _crossover_fn
    from algorithms.scheduler_evolution import random_individual as _random_individual_fn
    from algorithms.scheduler_evolution import run_evolution_loop
    from algorithms.scheduler_evolution import super_aggressive_mutate as _super_aggressive_mutate_fn
    from algorithms.scheduler_evolution import tournament_select as _tournament_select_fn

try:
    from src.algorithms.arrangement_generator import ArrangementSearch
    from src.algorithms.arrangement_generator import game_valid as _game_valid_fn
    from src.algorithms.arrangement_generator import search_arrangements as _search_arrangements_fn
except ImportError:
    from algorithms.arrangement_generator import ArrangementSearch
    from algorithms.arrangement_generator import game_valid as _game_valid_fn
    from algorithms.arrangement_generator import search_arrangements as _search_arrangements_fn

try:
    from src.algorithms.constructive_scheduler import build_perfect_schedule as _build_perfect_schedule_fn
except ImportError:
    from algorithms.constructive_scheduler import build_perfect_schedule as _build_perfect_schedule_fn

try:
    from src.utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
except ImportError:
    from utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer


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
    """Genetic algorithm scheduler that stops once it reaches the proven minimum range (see run_evolution_loop).

    Contract
    - Inputs: players (list[str]), num_courts (int), num_rounds (int=8)
    - Output: dict with keys {schedule, success, fitness, score, fitness_score,
      generations, time_seconds, algorithm, total_range, raw_metrics, metrics,
      fitness_details, replayable}
    - The proven lower bound on the total range, and whether range 0 is ruled
      out, come from ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums.
    """

    # Lexicographic penalty weights (games > partners > courts > opponents)
    LEXICOGRAPHIC_WEIGHTS = {
        "games_range": 10_000_000,  # Games Range 0 is top priority
        "partners_range": 100_000,  # Partners Range 0 is second priority
        "courts_range": 1_000,  # Courts Range 0 is third priority
        "opponents_range": 10,  # Opponents minimized last
    }

    # Defensive bound on the per-run evaluation caches (_fitness_cache,
    # _schedule_cache, _partner_repeat_cache). A caller-requested
    # max_generations budget (see _resolve_ga_sizing) can run long enough for
    # these to otherwise grow unbounded for the lifetime of one
    # generate_schedule() call - past this many distinct individuals seen,
    # they're cleared together rather than left to grow forever.
    _MAX_CACHE_ENTRIES = 300_000

    # Time the post-processor's repairs get in a normal (clock-bound) run.
    _REPAIR_MAX_TIME = 0.5

    # Search steps the arrangement-pool search may take per second of
    # max_runtime (see _search_arrangements), so a longer Max generation time
    # also searches further for distinct valid rounds.
    _ARRANGEMENT_SEARCH_NODES_PER_SECOND = 10_000

    def _resolve_ga_sizing(self, use_desktop_params, num_players, population_size, max_generations, max_runtime):
        """Resolve population_size/max_generations/max_runtime for desktop mode.

        An explicit max_generations or max_runtime from the caller is used
        exactly as given, in either mode: max_generations caps the number of
        generations for every roster shape (see run_evolution_loop), and
        max_runtime caps the run's time, as generate_schedule(max_time) does.
        None means "no explicit preference": outside desktop mode that is
        1000 generations and 300 s. Desktop mode, which the app uses, is
        bounded by time - the analyzer's tier runtime, unless generate_schedule
        is given max_time - and has no generation cap.

        Returns (population_size, max_generations, max_runtime, desktop_tunables),
        where desktop_tunables is the feasibility analyzer's optimal_params dict
        to apply via _resolve_ga_tunables when use_desktop_params is True, else None.
        """
        if not use_desktop_params:
            return (
                population_size,
                max_generations if max_generations is not None else 1000,
                max_runtime if max_runtime is not None else 300.0,
                None,
            )

        optimal_params = ScheduleFeasibilityAnalyzer.get_optimal_parameters(num_players)
        return (
            optimal_params["population_size"],
            max_generations,
            max_runtime if max_runtime is not None else optimal_params["max_runtime"],
            optimal_params,
        )

    def _resolve_ga_tunables(self, use_desktop_params, desktop_tunables):
        """Resolve tournament_size/convergence_patience for both desktop and
        non-desktop modes, so every GA tunable is always set here rather than
        via a trailing hasattr(self, "convergence_patience") fallback.

        These are the only tunables run_evolution_loop reads: its elite count,
        crossover probability and mutation are fixed in the loop itself."""
        if use_desktop_params and desktop_tunables is not None:
            return {
                "convergence_patience": desktop_tunables["convergence_patience"],
                "tournament_size": desktop_tunables["tournament_size"],
            }
        return {
            "convergence_patience": 60,
            "tournament_size": 7,
        }

    def __init__(
        self,
        players: List[Any],
        num_courts: int = 1,
        num_rounds: Optional[int] = None,
        *,
        population_size: int = 100,  # Desktop mode uses the feasibility analyzer's tier instead
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
        # Who sat out each round played before the schedule this run makes,
        # oldest first: a mid-session replan sets it to its locked rounds'
        # sitters, so order_rounds_for_rest measures sit-out streaks across
        # that boundary. No fitness term reads it.
        self.preceding_sit_outs: Sequence[Collection[str]] = ()
        # Offsets added to players' games before the games range is taken: see
        # the baseline_games property. Empty means plain games played.
        self._baseline_games: Dict[str, int] = {}
        self._game_offsets: Tuple[int, ...] = ()

        # num_courts is always the caller's explicit choice (physical courts
        # available cannot be inferred from player count alone - a club with
        # 16 players and 3 courts is a real, common case). Desktop mode only
        # affects population/runtime/patience sizing via _resolve_ga_sizing, never
        # how many courts are used.
        self.num_courts = int(num_courts)

        self.num_rounds = int(num_rounds) if num_rounds is not None else 8
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")

        # GA params
        self.population_size = int(population_size)
        # None: no generation cap - the run is bounded by max_runtime alone.
        self.max_generations: Optional[int] = int(max_generations) if max_generations is not None else None

        # Runtime controls. A schedule at the proven minimum range with no
        # constraint violation and partnerships spread as evenly as possible
        # ends the search at once (see run_evolution_loop); min_runtime only
        # delays a stop at the generation cap (compute_general_stop_threshold).
        env_max = os.environ.get("GA_MAX_RUNTIME")
        in_e2e = os.environ.get("E2E_TEST") in ("1", "true", "True")

        if in_e2e:
            self.min_runtime = 0.5
            self.max_runtime = float(env_max) if env_max else 10.0
        else:
            self.min_runtime = float(min_runtime) if min_runtime is not None else 0.0
            self.max_runtime = float(env_max) if env_max else float(max_runtime)

        # Ensure max is at least min
        if self.max_runtime < self.min_runtime:
            self.max_runtime = max(1.0, self.min_runtime)  # At least 1 second max

        # GA tunables - resolved here for both desktop and non-desktop modes,
        # so every tunable (including convergence_patience) is always set.
        tunables = self._resolve_ga_tunables(use_desktop_params, desktop_tunables)
        self.convergence_patience = tunables["convergence_patience"]
        self.tournament_size = tunables["tournament_size"]

        # Constraints
        self.do_not_pair_map: Dict[str, set[str]] = defaultdict(set)
        self.do_not_oppose_map: Dict[str, set[str]] = defaultdict(set)

        # If Player objects provided, build constraint maps
        if has_objects:
            for p in players:
                name = str(getattr(p, "name", None) or str(p))
                # The search has no notion of a player missing some rounds, so
                # availability it can't honour is refused rather than dropped.
                if (
                    getattr(p, "available_rounds", None)
                    or getattr(p, "available_from", None) is not None
                    or getattr(p, "available_to", None) is not None
                ):
                    raise ValueError(f"Player availability is not supported (given for {name!r})")
                for x in getattr(p, "do_not_pair", []) or []:
                    self.do_not_pair_map[name].add(str(x))
                for x in getattr(p, "do_not_oppose", []) or []:
                    self.do_not_oppose_map[name].add(str(x))

        self._set_proven_bounds()

        # Precompute arrangement pool (one round templates)
        search = self._search_arrangements(max_arrangements=400)
        if not search.arrangements:
            raise ValueError("No valid arrangements generated for these players/courts")
        self._use_arrangement_pool(search)

        # Ensure constraint maps have entries for all players (legacy expectation)
        for name in self._names:
            _ = self.do_not_pair_map[name]
            _ = self.do_not_oppose_map[name]

        # Performance optimizations - Enhanced caching. The fitness and
        # partner-repeat caches are keyed on the sorted individual: both
        # values ignore round order, so a reordering of an individual already
        # seen is a cache hit rather than a fresh evaluation. A schedule's
        # metrics and round signatures are not cached: _fitness needs them
        # once per individual, and kept for every individual seen they held
        # over half of a long run's memory.
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}
        self._schedule_cache: Dict[Tuple[int, ...], List[List[Tuple]]] = {}
        self._partner_repeat_cache: Dict[Tuple[int, ...], Tuple[int, int]] = {}
        self._partner_repeat_floor = partner_repeat_floor(self.num_players, 2 * self.num_courts * self.num_rounds)
        self._refresh_arrangement_stats()

        # Progress tracking for reduced update frequency
        self._last_progress_time = 0.0
        self._last_logged_fitness = float("inf")
        self._progress_update_interval = 2.0  # Update every 2 seconds max
        # Last run diagnostics
        self.last_fitness_details: Dict[str, int] | None = None

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
            search = self._search_arrangements(max_arrangements=400)
            if search.arrangements:
                self._use_arrangement_pool(search)
            elif search.budget_reached:
                # Not proof that no valid round exists - only that the search
                # stopped before finding one.
                self.round_search_budget_reached = True
                print(
                    "⚠️ The round search ran out of budget before finding a round that meets every constraint - "
                    "keeping unconstrained pool with penalties (a longer max runtime searches further)"
                )
            else:
                # The search visited every round and none meets every
                # constraint, so a longer run would not help.
                self.round_search_budget_reached = False
                print("⚠️ Constraints leave no valid round arrangements - keeping unconstrained pool with penalties")
            # Constraint maps changed either way, so per-arrangement violation
            # summaries and every cached evaluation are stale.
            self._fitness_cache.clear()
            self._schedule_cache.clear()
            self._partner_repeat_cache.clear()
            self._refresh_arrangement_stats()

    @property
    def baseline_games(self) -> Mapping[str, int]:
        """Per-player offsets added to each player's games before the games range is taken.

        Empty (the default) leaves the games range as plain games played. A
        run that carries on from rounds already played sets minus each
        player's sit-outs so far: the games range, which the search puts
        first, is then the range of everyone's sit-outs to date, so the
        sitters of this run are the players who have sat out least. Players
        not named have 0. Setting it re-derives the proven bounds that early
        stops and repairs rely on, and drops every cached evaluation.
        """
        return self._baseline_games

    @baseline_games.setter
    def baseline_games(self, baseline: Mapping[str, int] | None) -> None:
        roster = set(self._names)
        offsets: Dict[str, int] = {}
        for name, offset in (baseline or {}).items():
            if int(offset) != offset:
                raise ValueError(f"baseline_games offsets must be whole numbers (got {offset!r} for {name!r})")
            if str(name) in roster and int(offset) != 0:
                offsets[str(name)] = int(offset)
        self._baseline_games = offsets
        self._game_offsets = tuple(offsets.get(name, 0) for name in self._names) if offsets else ()
        self._set_proven_bounds()
        # Every cached fitness was scored against the previous offsets.
        self._fitness_cache.clear()

    def _set_proven_bounds(self) -> None:
        """Set the proven lower bounds from the shape and the games offsets.

        range0_possible means range 0 isn't ruled out, and optimal_total_range
        is the lowest total range any schedule can have. Offsets can only move
        the games part of that (see min_games_range_with_offsets).
        """
        feasibility: Mapping[str, Any] = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(
            self.num_players,
            self.num_courts,
            self.num_rounds,
            self._game_offsets,
        )
        self.feasibility_minimums = feasibility
        self.range0_possible = bool(feasibility["range_0_possible"])
        # Not read by the GA itself; kept for callers and diagnostics that report it.
        self.partner_range0_possible = int(feasibility["min_partner_range"]) == 0
        self.optimal_partner_range = int(feasibility["min_partner_range"])
        self.optimal_total_range = int(feasibility["total_theoretical_min"])

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
        generations: Optional[int] = None,
    ):
        """Run the GA until it reaches the proven minimum range or its budget runs out.

        By default the seed is time-based for run-to-run variety, and the run
        stops on the clock (max_time), so the same seed can end at a different
        generation, and with a different schedule, on a busier computer. To
        replay a run exactly, pass its ``seed`` and the ``generations`` it
        reported: the search then runs exactly that many generations whatever
        the clock says, and the repairs are bounded by attempts, not time. The
        result's ``replayable`` says whether this run can be replayed that way.
        A fake ``clock`` makes a whole run reproducible too.
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
        # A replay stops the repairs' clock, so only their attempt limits
        # bound them, exactly as in a run that never ran out of repair time.
        self._run_rng = rng
        self._run_clock = now if generations is None else (lambda: 0.0)

        # A verified constructive design (see constructive_scheduler.py) may
        # exist for this exact player count/court/round combination. With no
        # active constraints it is already optimal, so skip the GA entirely;
        # with constraints active, use it to seed the initial population
        # instead of starting purely random.
        seed_individual = self._build_seed_individual()
        has_active_constraints = self._has_active_constraints()

        print_scheduler_start_banner(
            print,
            absolute_min_runtime=absolute_min_runtime,
            perfect_stop_runtime=perfect_stop_runtime,
            max_runtime=self.max_runtime,
            optimal_partner_range=self.optimal_partner_range,
            range0_possible=self.range0_possible,
            optimal_total_range=self.optimal_total_range,
            max_unique_rounds=self.max_unique_rounds,
            minimum_duplicate_rounds=self.minimum_duplicate_rounds,
            seed=seed,
            max_generations=self.max_generations,
            round_search_budget_reached=self.round_search_budget_reached,
        )

        if seed_individual is not None and not has_active_constraints:
            print("✨ Using a verified constructive design - already optimal, skipping search.")
            best_individual = seed_individual
            best_fitness = self._fitness(best_individual)
            generations_run = 0
            spread_best_individual = best_individual
        else:
            best_individual, best_fitness, generations_run, spread_best_individual = run_evolution_loop(
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
                generation_budget=generations,
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
                round_search_budget_reached=self.round_search_budget_reached,
            )

        final_schedule = self._decode(best_individual)
        final_metrics = self._evaluate_metrics(final_schedule)
        duplicate_rounds = self._count_duplicate_rounds(final_schedule)
        avoidable_duplicate_rounds = self._count_avoidable_duplicate_rounds(final_schedule)

        # POST-PROCESSOR: Apply targeted repairs unless the schedule is at the
        # proven minimum range - or if the games are split less evenly than
        # they can be.
        current_total_range = sum_range_metrics(final_metrics)
        optimal_total_range = self.optimal_total_range
        games_floor = games_range_floor(final_schedule, self.num_players, self._game_offsets)
        repaired = False
        repair_seconds = 0.0

        if current_total_range > optimal_total_range or final_metrics["games_range"] > games_floor:
            print("\n🔧 POST-PROCESSOR: Attempting targeted repairs...")
            repair_started = now()
            repaired_schedule = self._apply_targeted_repairs(final_schedule, max_time=self._REPAIR_MAX_TIME)
            repair_seconds = now() - repair_started
            if repaired_schedule:
                repair_metrics = self._evaluate_metrics(repaired_schedule)
                repair_total = sum_range_metrics(repair_metrics)
                # Lexicographic, like the GA's own objective: never trade in a
                # constraint violation or an avoidable duplicate round; then
                # an even games split first, then the total range.
                no_worse = repair_metrics["violations"] <= final_metrics["violations"] and (
                    self._count_avoidable_duplicate_rounds(repaired_schedule) <= avoidable_duplicate_rounds
                )
                improved = (repair_metrics["games_range"], repair_total) < (
                    final_metrics["games_range"],
                    current_total_range,
                )
                if no_worse and improved:
                    print("✅ POST-PROCESSOR: Improved from Range " f"{current_total_range} to Range {repair_total}")
                    final_schedule = repaired_schedule
                    final_metrics = repair_metrics
                    current_total_range = repair_total
                    duplicate_rounds = self._count_duplicate_rounds(final_schedule)
                    avoidable_duplicate_rounds = self._count_avoidable_duplicate_rounds(final_schedule)
                    best_fitness = self._schedule_fitness(final_schedule)
                    repaired = True
                else:
                    print(f"   ⚠️ REPAIR PARTIAL: No improvement - still total range {current_total_range}")

        if not repaired:
            # Same ranges, violations and duplicate rounds as best_individual
            # (see _spreads_partners_better), but fewer repeated partnerships.
            # Repairs above still start from best_individual, exactly as before.
            # Everything reported is recomputed from the schedule returned, so
            # the result can never describe a different one.
            final_schedule = self._decode(spread_best_individual)
            final_metrics = self._evaluate_metrics(final_schedule)
            duplicate_rounds = self._count_duplicate_rounds(final_schedule)
            avoidable_duplicate_rounds = self._count_avoidable_duplicate_rounds(final_schedule)
            best_fitness = self._fitness(spread_best_individual)

        # The last tiebreak, below the partner-repetition one above: put the
        # rounds in the order that spreads each player's sit-outs out most,
        # counting on from the rounds already played before these, if any.
        # Nothing reported above depends on round order.
        final_schedule = order_rounds_for_rest(final_schedule, self._names, preceding_sit_outs=self.preceding_sit_outs)

        constraint_violations = sum(
            0 if self._game_valid(game) else 1 for round_games in final_schedule for game in round_games
        )
        if constraint_violations and not self.round_search_budget_reached:
            print(
                f"⚠️ {constraint_violations} game(s) in the final schedule violate a "
                "requested do-not-pair/do-not-oppose constraint - the constraints "
                "may be infeasible together with this many players/courts/rounds."
            )
        elif constraint_violations:
            print(
                f"⚠️ {constraint_violations} game(s) in the final schedule violate a "
                "requested do-not-pair/do-not-oppose constraint - the round search ran "
                "out of budget before finding a round that meets them all; a longer "
                "max runtime searches further."
            )

        elapsed_seconds = now() - start_time
        print_scheduler_final_report(
            print,
            elapsed_seconds=elapsed_seconds,
            final_metrics=final_metrics,
            duplicate_rounds=duplicate_rounds,
            avoidable_duplicate_rounds=avoidable_duplicate_rounds,
            minimum_duplicate_rounds=self.minimum_duplicate_rounds,
            optimal_total_range=optimal_total_range,
            num_players=self.num_players,
            max_unique_rounds=self.max_unique_rounds,
            constraint_violations=constraint_violations,
            round_search_budget_reached=self.round_search_budget_reached,
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
            constraint_violations=constraint_violations,
            round_search_budget_reached=self.round_search_budget_reached,
        )
        # The search itself replays exactly from (seed, generations); the
        # repairs do too unless one of them ran out of time. No repair's time
        # limit is under half of _REPAIR_MAX_TIME, counted from no earlier
        # than repair_started, so repairs that all finished within that never
        # hit one.
        result["replayable"] = generations is not None or repair_seconds < self._REPAIR_MAX_TIME / 2

        # Quality check for small problems
        elapsed = elapsed_seconds
        if self.num_players <= 8 and best_fitness > 0 and elapsed < self.max_runtime:
            if verbose:
                print(
                    f"⚠️ Suboptimal result (Range {result['total_range']}). "
                    "Consider increasing runtime or population size."
                )

        # Add feasibility analysis
        result["feasibility_analysis"] = ScheduleFeasibilityAnalyzer.assess_quality_with_feasibility(
            final_metrics, self.num_players, self.num_courts, self.num_rounds, self._game_offsets
        )

        return result

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

    def _search_arrangements(self, max_arrangements: int = 400) -> ArrangementSearch:
        """Search for the diverse round-arrangement pool (see arrangement_generator).

        The search budget grows with max_runtime (up to its 300s ceiling in the
        app), so a longer run also searches further for distinct valid rounds.
        """
        runtime = min(max(self.max_runtime, 20.0), 300.0)
        return _search_arrangements_fn(
            self._names,
            self.num_courts,
            do_not_pair_map=self.do_not_pair_map,
            do_not_oppose_map=self.do_not_oppose_map,
            max_arrangements=max_arrangements,
            node_budget=int(runtime * self._ARRANGEMENT_SEARCH_NODES_PER_SECOND),
            printer=print,
        )

    def _use_arrangement_pool(self, search: ArrangementSearch) -> None:
        """Adopt a searched pool, and the duplicate-round floor it implies.

        minimum_duplicate_rounds is the fewest repeated rounds any schedule
        drawn from this pool can have. When the search ran out of budget with
        room left in the pool (round_search_budget_reached), more distinct
        valid rounds may exist than it found, so that is no true minimum.
        add_constraints also sets round_search_budget_reached when its search
        runs out of budget before finding any round that meets every
        constraint. Either way, a longer max runtime searches further.
        """
        self.arrangements = search.arrangements
        self.round_search_budget_reached = search.budget_reached
        self.max_unique_rounds = len({self._round_signature(arrangement) for arrangement in self.arrangements})
        self.minimum_duplicate_rounds = max(0, self.num_rounds - self.max_unique_rounds)

    def _game_valid(self, game: Tuple[str, str, str, str]) -> bool:
        """Check if a game satisfies all constraints."""
        return _game_valid_fn(game, self.do_not_pair_map, self.do_not_oppose_map)

    def _has_active_constraints(self) -> bool:
        """Whether any do_not_pair/do_not_oppose constraint can rule out a game.

        Only a pair of two different players who are both on this roster can.
        A constraint naming someone who isn't playing, or a player paired with
        themselves, never applies to any game, so it must not turn off the
        constructive fast path either.
        """
        roster = set(self._names)
        return any(
            other != name and other in roster
            for constraint_map in (self.do_not_pair_map, self.do_not_oppose_map)
            for name in roster
            for other in constraint_map.get(name, ())
        )

    def _round_stats(self, rounds: List[List[GameTuple]]) -> List[Dict[str, Any]]:
        """Per-round summaries of rounds, as the fitness function's fast path builds them."""
        return precompute_arrangement_stats(
            rounds,
            num_courts=self.num_courts,
            do_not_pair_map=self.do_not_pair_map,
            do_not_oppose_map=self.do_not_oppose_map,
        )

    def _seed_quality(self, round_stats: List[Dict[str, Any]], signatures: List[Tuple]) -> Tuple[int, ...]:
        """Rank a candidate seed schedule, given its per-round summaries
        (_round_stats) and round signatures, in the GA's own order (lower is
        better): violations, avoidable duplicate rounds, then games,
        partners, courts and opponents range."""
        metrics = evaluate_metrics_from_arrangement_stats(round_stats, self._names, self._baseline_games)
        return (
            metrics["violations"],
            avoidable_duplicate_rounds_from_signature(signatures, self.minimum_duplicate_rounds),
            metrics["games_range"],
            metrics["partners_range"],
            metrics["courts_range"],
            metrics["opponents_range"],
        )

    def _repair_seed_violations(self, schedule: List[List[GameTuple]]) -> Optional[List[List[GameTuple]]]:
        """Swap players within each round until no game breaks a constraint.

        Each step takes, among the swaps of two players in a round that leave
        fewer of its games violating, the one that keeps the schedule fairest
        (_seed_quality). Everyone still plays every round, so the games
        balance can't change. Returns None if some round can't be made valid
        this way.
        """
        repaired = [list(round_games) for round_games in schedule]
        round_stats = self._round_stats(repaired)
        signatures = [round_signature(round_games) for round_games in repaired]
        for round_index, round_games in enumerate(repaired):
            violating = sum(not self._game_valid(game) for game in round_games)
            while violating:
                slots = [(game_index, position) for game_index in range(len(round_games)) for position in range(4)]
                best: Tuple[Tuple[int, ...], List[GameTuple], int] | None = None
                for i, (game_a, position_a) in enumerate(slots):
                    for game_b, position_b in slots[i + 1 :]:
                        games = [list(game) for game in round_games]
                        games[game_a][position_a], games[game_b][position_b] = (
                            games[game_b][position_b],
                            games[game_a][position_a],
                        )
                        new_round = [(game[0], game[1], game[2], game[3]) for game in games]
                        new_violating = sum(not self._game_valid(game) for game in new_round)
                        if new_violating >= violating:
                            continue
                        key = self._seed_quality(
                            round_stats[:round_index] + self._round_stats([new_round]) + round_stats[round_index + 1 :],
                            signatures[:round_index] + [round_signature(new_round)] + signatures[round_index + 1 :],
                        )
                        if best is None or key < best[0]:
                            best = (key, new_round, new_violating)
                if best is None:
                    return None
                _, round_games, violating = best
                repaired[round_index] = round_games
                round_stats[round_index] = self._round_stats([round_games])[0]
                signatures[round_index] = round_signature(round_games)
        return repaired

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
        the constraints. The designs spread partners and opponents as widely
        as possible, so on many shapes every pair of players partners or
        faces each other somewhere and no reordering can avoid even one
        constraint. Then the reorderings with the fewest violating games are
        repaired by swapping players within the violating rounds
        (_repair_seed_violations), and the fairest valid result seeds the GA.
        If none can be repaired, seeding is skipped (None) rather than
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

        if not self._has_active_constraints():
            schedule = probe
        else:
            schedule = None
            fewest_violating: List[List[List[GameTuple]]] = []
            fewest_count = 0
            for _ in range(20):
                candidate_names = rng.sample(names, len(names))
                candidate = _build_perfect_schedule_fn(candidate_names, self.num_courts, self.num_rounds)
                if not candidate:
                    continue
                violating = sum(not self._game_valid(game) for round_games in candidate for game in round_games)
                if violating == 0:
                    schedule = candidate
                    break
                if not fewest_violating or violating < fewest_count:
                    fewest_violating, fewest_count = [candidate], violating
                elif violating == fewest_count:
                    fewest_violating.append(candidate)
            if schedule is None:
                best_quality: Tuple[int, ...] | None = None
                for candidate in fewest_violating:
                    repaired = self._repair_seed_violations(candidate)
                    if repaired is None:
                        continue
                    quality = self._seed_quality(
                        self._round_stats(repaired), [round_signature(round_games) for round_games in repaired]
                    )
                    if best_quality is None or quality < best_quality:
                        schedule, best_quality = repaired, quality
            if schedule is None:
                return None  # constraints conflict with the design in every tried ordering, even after repair

        assert schedule is not None
        # Reuse existing pool slots for rounds already present instead of
        # blindly appending - otherwise a long-lived instance whose caller
        # replays the same seed repeatedly (or regenerates without a seed)
        # grows self.arrangements by num_rounds on every single call. Match
        # the exact arrangement, court order included: the round signature
        # ignores court order, and reusing a same-signature slot with its
        # games on other courts broke the design's court balance.
        exact_to_index: dict[tuple[tuple, ...], int] = {}
        for pool_index, arrangement in enumerate(self.arrangements):
            exact_to_index.setdefault(tuple(tuple(game) for game in arrangement), pool_index)
        new_indices = []
        pool_grew = False
        for round_games in schedule:
            key = tuple(tuple(game) for game in round_games)
            idx = exact_to_index.get(key)
            if idx is None:
                idx = len(self.arrangements)
                self.arrangements.append(round_games)
                exact_to_index[key] = idx
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

        # Use pure sequential processing with optimized caching (_fitness
        # checks its cache first). Parallel processing has too much overhead
        # for this use case
        for ind in population:
            fitness_scores.append((ind, self._fitness(ind)))

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
        # Every round of a schedule built from this pool breaks at least as
        # many constraints as the pool's best arrangement, so no schedule can
        # have fewer violations than this: 0 whenever the constraints can be
        # met, more only when add_constraints had to keep the unconstrained
        # pool. The evolution loop treats a range-0 schedule as perfect only
        # once its violations are down to this bound.
        self.minimum_violations = self.num_rounds * min(
            (stat["violations"] for stat in self._arrangement_stats), default=0
        )
        self._arrangement_duplicate_penalties = [
            self._round_duplicate_penalty(arrangement) for arrangement in self.arrangements
        ]

    @staticmethod
    def _round_duplicate_penalty(round_games: List[Tuple]) -> int:
        """1 billion for each game without 4 different players and each
        player who appears in a second game of the same round."""
        penalty = 0
        round_players: set = set()
        for game in round_games:
            game_players = set(game)
            if len(game_players) != 4:
                penalty += 1000000000
            overlap = game_players & round_players
            if overlap:
                penalty += 1000000000 * len(overlap)
            round_players.update(game_players)
        return penalty

    def _decode_cached(self, individual: Tuple[int, ...]) -> List[List[Tuple]]:
        """Cached version of schedule decoding."""
        if individual in self._schedule_cache:
            return self._schedule_cache[individual]

        schedule = self._decode(individual)

        self._schedule_cache[individual] = schedule

        return schedule

    def _individual_metrics(self, individual: Tuple[int, ...]) -> Dict[str, int]:
        """Metrics of an individual, merged from the precomputed per-arrangement stats."""
        return evaluate_metrics_from_arrangement_stats(
            [self._arrangement_stats[idx] for idx in individual],
            self._names,
            self._baseline_games,
        )

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
        than per-cache LRU eviction) keeps the three caches always in sync
        with each other."""
        if len(self._fitness_cache) <= self._MAX_CACHE_ENTRIES:
            return
        self._fitness_cache.clear()
        self._schedule_cache.clear()
        self._partner_repeat_cache.clear()

    def _individual_signature(self, individual: Tuple[int, ...]) -> Tuple:
        """Duplicate-round signature of an individual, from the precomputed pool signatures."""
        return tuple(self._arrangement_signatures[idx] for idx in individual)

    def _fitness(self, individual: Tuple[int, ...]) -> float:
        """Optimized lexicographic fitness function with enhanced caching."""
        # Every term below ignores round order, so reorderings share an entry.
        key = tuple(sorted(individual))
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        # Metrics, structural penalties, and signatures all derive from the
        # precomputed per-arrangement summaries - no decode needed here.
        metrics = self._individual_metrics(individual)

        # CRITICAL: player-duplicate penalties (precomputed per arrangement)
        duplicate_penalty = sum(self._arrangement_duplicate_penalties[idx] for idx in individual)

        # Use optimized duplicate detection from the precomputed pool signatures
        signature = self._individual_signature(individual)
        avoidable_duplicate_rounds = avoidable_duplicate_rounds_from_signature(signature, self.minimum_duplicate_rounds)

        # MASSIVE penalty only for duplicate rounds beyond what the configuration forces.
        duplicate_penalty += avoidable_duplicate_rounds * 1000000000  # 1 BILLION per avoidable duplicate

        penalty = self._penalty(metrics, duplicate_penalty)

        # Cache the result
        self._fitness_cache[key] = penalty
        self._enforce_cache_size_limit()

        return penalty

    def _schedule_fitness(self, schedule: List[List[Tuple[str, str, str, str]]]) -> float:
        """_fitness for a schedule whose rounds need not come from the pool,
        such as one the post-processor repaired."""
        duplicate_penalty = sum(self._round_duplicate_penalty(round_games) for round_games in schedule)
        duplicate_penalty += self._count_avoidable_duplicate_rounds(schedule) * 1000000000
        return self._penalty(self._evaluate_metrics(schedule), duplicate_penalty)

    def _penalty(self, metrics: Mapping[str, int], duplicate_penalty: float) -> float:
        """The fitness of a schedule with these metrics and duplicate penalties (lower is better)."""
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

        # Range 0 achievement bonus for desktop optimization - only for a
        # schedule that breaks no constraint: the bonus is worth more than a
        # violation costs, so it would otherwise rank range 0 with an extra
        # violation above range 1 without it.
        if self.range0_possible and metrics["violations"] == 0:
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

        return penalty

    def _spreads_partners_better(self, candidate: Tuple[int, ...], incumbent: Tuple[int, ...]) -> bool:
        """Tiebreak between two individuals with the same fitness: True when
        candidate repeats partnerships less than incumbent and every fitness
        term - each range, violations, avoidable duplicate rounds - is equal.

        The ranges count each player's distinct partners, so once those are
        even they can't tell spread-out partnerships from the same pair
        partnering again and again. This never feeds back into the fitness or
        the selection, so the search runs exactly as it did without it.
        """
        # Cheapest tests first: once the incumbent can't be beaten, or the
        # candidate repeats as much, nothing else needs comparing.
        incumbent_score = self._partner_repeat_score_cached(incumbent)
        if (
            candidate == incumbent
            or incumbent_score <= self._partner_repeat_floor
            or self._partner_repeat_score_cached(candidate) >= incumbent_score
        ):
            return False
        candidate_metrics = self._individual_metrics(candidate)
        # A schedule that already breaks a constraint is left as the search
        # found it, so even its count of violating games can't change.
        if candidate_metrics["violations"] or candidate_metrics != self._individual_metrics(incumbent):
            return False
        return avoidable_duplicate_rounds_from_signature(
            self._individual_signature(candidate), self.minimum_duplicate_rounds
        ) == avoidable_duplicate_rounds_from_signature(
            self._individual_signature(incumbent), self.minimum_duplicate_rounds
        )

    def _partner_repeat_score_cached(self, individual: Tuple[int, ...]) -> Tuple[int, int]:
        """Cached partner_repeat_score from the precomputed per-arrangement stats
        (keyed like _fitness_cache: the score ignores round order)."""
        key = tuple(sorted(individual))
        if key not in self._partner_repeat_cache:
            self._partner_repeat_cache[key] = partner_repeat_score(self._arrangement_stats[idx] for idx in individual)
        return self._partner_repeat_cache[key]

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
            baseline_games=self._baseline_games,
        )

    def _apply_targeted_repairs(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float = 0.5
    ) -> Optional[List[List[Tuple[str, str, str, str]]]]:
        """Apply targeted local repairs to achieve Range 0 when mathematically possible.

        The games repair always finishes (each move evens the game counts out
        further), so it gets the larger of half of max_time and a tenth of the
        run's budget: a fixed quarter second left large rosters short of an
        even games split, worst on a slow host.
        """
        return apply_targeted_repairs(
            schedule,
            max_time,
            evaluate_metrics=self._evaluate_metrics,
            printer=print,
            partner_repair=self._repair_partner_imbalance,
            opponent_repair=self._repair_opponent_imbalance,
            clock=getattr(self, "_run_clock", None),
            games_repair=self._repair_games_balance,
            num_players=self.num_players,
            games_max_time=max(max_time * 0.5, self.max_runtime * 0.1),
            game_offsets=self._game_offsets,
        )

    def _repair_games_balance(
        self, schedule: List[List[Tuple[str, str, str, str]]], max_time: float
    ) -> List[List[Tuple[str, str, str, str]]]:
        """Even out games played by moving games from the most- to the least-played players."""
        return repair_games_balance(
            schedule,
            self._names,
            max_time,
            evaluate_metrics=self._evaluate_metrics,
            printer=print,
            clock=getattr(self, "_run_clock", None),
            baseline_games=self._baseline_games,
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
