"""GA search/evolution flow helpers for the genetic scheduler.

Extracted from genetic_scheduler.py following the same seam pattern used
by scheduler_repairs.py, scheduler_metrics.py, and scheduler_reporting.py.

Covers:
- Population initialisation (random_individual)
- Crossover and mutation operators (crossover, super_aggressive_mutate)
- Parent selection and elite preservation (tournament_select, elitism)
- Diversity injection / stagnation handling (inject_diversity)
- Termination / early-stop thresholds (compute_general_stop_threshold,
  compute_perfect_stop_threshold)
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Tuple

try:
    from src.algorithms.scheduler_metrics import avoidable_duplicate_rounds_from_signature
except ImportError:
    from algorithms.scheduler_metrics import avoidable_duplicate_rounds_from_signature

Individual = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------


def random_individual(rng: random.Random, num_arrangements: int, num_rounds: int) -> Individual:
    """Create a random individual that maximises arrangement diversity.

    When enough distinct arrangements exist every round gets a unique one.
    When the pool is smaller we cycle through with resets to avoid excessive
    immediate repetition.

    Args:
        rng: Seeded random instance.
        num_arrangements: Size of the pre-computed arrangement pool.
        num_rounds: Number of rounds the individual must represent.

    Returns:
        Tuple of arrangement indices, one per round.
    """
    if num_arrangements >= num_rounds:
        indices = list(range(num_arrangements))
        rng.shuffle(indices)
        selected = indices[:num_rounds]

        # Paranoia check – should never trigger, but guarantees uniqueness
        if len(set(selected)) != len(selected):
            selected = rng.sample(range(num_arrangements), num_rounds)

        return tuple(selected)
    else:
        # Not enough unique arrangements – cycle through available pool
        indices: List[int] = []
        available = list(range(num_arrangements))
        for _ in range(num_rounds):
            if available:
                choice = rng.choice(available)
                indices.append(choice)
                available.remove(choice)
            else:
                # Reset and continue to avoid running out of options
                available = list(range(num_arrangements))
                choice = rng.choice(available)
                indices.append(choice)
                available.remove(choice)
        return tuple(indices)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """Multi-point crossover producing two children from two parents.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.
        rng: Seeded random instance.

    Returns:
        Pair of child individuals with the same length as their parents.
    """
    if len(parent1) <= 1:
        return parent1, parent2

    num_points = rng.randint(1, min(3, len(parent1) - 1))
    points = sorted(rng.sample(range(1, len(parent1)), num_points))

    child1, child2 = list(parent1), list(parent2)

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

    if swap:
        child1[start:], child2[start:] = child2[start:], child1[start:]

    return tuple(child1), tuple(child2)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def super_aggressive_mutate(
    ind: Individual,
    rng: random.Random,
    arr_len: int,
) -> Individual:
    """SUPER aggressive mutation for breaking out of local optima.

    Applies one of seven mutation strategies at random, several of which
    enforce arrangement diversity within the individual.

    Args:
        ind: Current individual (tuple of arrangement indices).
        rng: Seeded random instance.
        arr_len: Size of the arrangement pool (upper bound for index sampling).

    Returns:
        Mutated individual of the same length as *ind*.
    """
    out = list(ind)
    mut_rate = 0.4  # 40% per-gene chance

    strategy = rng.randint(0, 6)

    if strategy == 0:
        # Replace many genes with diversity enforcement
        for i in range(len(out)):
            if rng.random() < mut_rate:
                if arr_len >= len(out):
                    used = set(out[j] for j in range(len(out)) if j != i)
                    available = [x for x in range(arr_len) if x not in used]
                    if available:
                        out[i] = rng.choice(available)
                    else:
                        out[i] = rng.randrange(0, arr_len)
                else:
                    out[i] = rng.randrange(0, arr_len)

    elif strategy == 1:
        # Massive random swaps
        num_swaps = rng.randint(2, max(3, len(out) // 2))
        for _ in range(num_swaps):
            if len(out) > 1:
                i, j = rng.sample(range(len(out)), 2)
                out[i], out[j] = out[j], out[i]

    elif strategy == 2:
        # Replace a random chunk with diverse alternatives
        if len(out) > 4:
            start = rng.randrange(len(out) - 2)
            length = rng.randint(2, min(4, len(out) - start))
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
        # Reverse a random segment
        if len(out) > 2:
            start = rng.randrange(len(out) - 1)
            end = rng.randint(start + 1, len(out))
            out[start:end] = reversed(out[start:end])

    elif strategy == 4:
        # Shuffle a random portion
        if len(out) > 4:
            start = rng.randrange(len(out) - 3)
            length = rng.randint(3, min(5, len(out) - start))
            portion = out[start : start + length]
            rng.shuffle(portion)
            out[start : start + length] = portion

    elif strategy == 5:
        # Randomise ~70% of genes
        indices = rng.sample(range(len(out)), max(1, int(len(out) * 0.7)))
        for i in indices:
            out[i] = rng.randrange(0, arr_len)

    else:
        # Strategy 6: enforce complete diversity when pool is large enough
        if arr_len >= len(out):
            all_indices = list(range(arr_len))
            rng.shuffle(all_indices)
            out = all_indices[: len(out)]
        else:
            # Aggressive fallback for small pools
            for i in range(len(out)):
                if rng.random() < 0.6:
                    out[i] = rng.randrange(0, arr_len)

    return tuple(out)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def tournament_select(
    population: List[Individual],
    fitness_scores: List[Tuple[Individual, float]],
    rng: random.Random,
    k: int = 7,
) -> Individual:
    """Tournament selection with configurable tournament size.

    Args:
        population: Current population list.
        fitness_scores: Parallel list of (individual, fitness) pairs.
        rng: Seeded random instance.
        k: Number of contestants in each tournament.

    Returns:
        The individual with the lowest fitness among the tournament contestants.
    """
    tournament_indices = rng.sample(range(len(population)), min(k, len(population)))
    tournament = [(population[i], fitness_scores[i][1]) for i in tournament_indices]
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


def elitism(
    population: List[Individual],
    fitnesses: List[float],
    k: int,
) -> List[Individual]:
    """Select the *k* best (lowest-fitness) individuals for elite preservation.

    Args:
        population: Current population list.
        fitnesses: Parallel fitness values (lower is better).
        k: Number of elites to return.

    Returns:
        List of the *k* best individuals.
    """
    idxs = sorted(range(len(population)), key=lambda i: fitnesses[i])[:k]
    return [population[i] for i in idxs]


# ---------------------------------------------------------------------------
# Diversity injection
# ---------------------------------------------------------------------------


def inject_diversity(
    population: List[Individual],
    rng: random.Random,
    replacement_fraction: float,
    make_individual: Callable[[], Individual],
) -> List[Individual]:
    """Replace a fraction of the population with freshly generated individuals.

    Operates in-place and returns the same list for convenience.

    Args:
        population: Current population (mutated in-place).
        rng: Seeded random instance used to pick replacement positions.
        replacement_fraction: Fraction of the population to replace (0–1).
        make_individual: Zero-argument callable that produces a new individual.

    Returns:
        The updated population list.
    """
    replacement_count = int(len(population) * replacement_fraction)
    for _ in range(replacement_count):
        idx = rng.randrange(len(population))
        population[idx] = make_individual()
    return population


# ---------------------------------------------------------------------------
# Termination / early-stop thresholds
# ---------------------------------------------------------------------------


def compute_general_stop_threshold(
    num_players: int,
    num_rounds: int,
    min_runtime: float,
) -> float:
    """Return the minimum elapsed time before a non-perfect early stop is allowed.

    Args:
        num_players: Number of players in the tournament.
        num_rounds: Number of rounds in the schedule.
        min_runtime: Configured minimum runtime (may be 0.0).

    Returns:
        Minimum elapsed seconds required before the main loop may exit on a
        non-Range-0 convergence condition.
    """
    if os.environ.get("E2E_TEST") in ("1", "true", "True"):
        return 0.5
    if num_players <= 8 and num_rounds <= 8:
        return max(1.0, min_runtime or 0.0)
    return min_runtime or 5.0


def compute_perfect_stop_threshold(
    num_players: int,
    num_rounds: int,
    min_runtime: float,
    max_runtime: float,
) -> float:
    """Return the minimum elapsed time before a perfect (Range-0) stop is confirmed.

    Args:
        num_players: Number of players in the tournament.
        num_rounds: Number of rounds in the schedule.
        min_runtime: Configured minimum runtime (may be 0.0).
        max_runtime: Configured maximum runtime budget.

    Returns:
        Minimum elapsed seconds to confirm a perfect result before stopping.
    """
    if os.environ.get("E2E_TEST") in ("1", "true", "True"):
        return 0.5
    if num_players <= 8 and num_rounds <= 8:
        return max(1.0, min_runtime or 0.0)
    return min(30.0, max_runtime * 0.8)


def run_evolution_loop(
    scheduler,
    *,
    rng,
    now,
    start_time,
    verbose,
    progress_callback,
    absolute_min_runtime,
    perfect_stop_runtime,
    printer,
    invoke_progress,
):
    """Run the main GA evolution loop extracted from GeneticPickleballScheduler.

    Takes the scheduler as a duck-typed context for fitness/decode/operator
    access and returns (best_individual, best_fitness, generations_run).
    Seeded runs with an injected clock are fully reproducible, which is the
    equivalence check used when this code was moved out of generate_schedule.
    """
    # Initialize tracking
    best_individual = None
    best_fitness = float("inf")
    best_ranges = None
    generations_run = 0
    generations_without_improvement = 0

    # Track if we've logged progress recently
    last_progress_log = 0

    # Initialize population
    population = [scheduler._random_individual(rng) for _ in range(scheduler.population_size)]
    current_ranges = {"games": 0, "partners": 0, "opponents": 0, "courts": 0}

    # MAIN EVOLUTION LOOP - GUARANTEED TO RUN FOR MINIMUM TIME
    while True:
        elapsed = now() - start_time

        # CRITICAL: Check if we can stop
        can_stop = False
        stop_reason = None

        # Check for Range 0 achievement
        if best_individual is not None:
            schedule = scheduler._decode_cached(best_individual)
            metrics = scheduler._evaluate_metrics_cached(schedule, best_individual)
            signature = scheduler._get_duplicate_signature_cached(best_individual, schedule)
            avoidable_duplicate_rounds = avoidable_duplicate_rounds_from_signature(
                signature, scheduler.minimum_duplicate_rounds
            )
            current_ranges = {
                "games": metrics["games_range"],
                "partners": metrics["partners_range"],
                "opponents": metrics["opponents_range"],
                "courts": metrics["courts_range"],
            }

            total_range = sum(current_ranges.values())

            # Log progress every 15 seconds (much less frequent)
            if elapsed - last_progress_log >= 15.0:
                printer(
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
                    "avoidable_duplicate_rounds": avoidable_duplicate_rounds,
                    "range0_possible": scheduler.range0_possible,
                    "max_time": scheduler.max_runtime,
                }
                try:
                    invoke_progress(progress_callback, progress_data)
                except Exception as e:
                    printer(f"Progress callback failed: {e}")

            # Check stopping conditions - RUN FOR FULL TIME FOR QUALITY
            if total_range == 0 and avoidable_duplicate_rounds == 0:
                if elapsed >= perfect_stop_runtime:
                    can_stop = True
                    stop_reason = "PERFECT TOTAL RANGE ACHIEVED! 🎉"
                else:
                    can_stop = False
            elif scheduler.range0_possible and total_range > 0:
                # Continue trying for Range 0 but respect time limits
                if elapsed >= scheduler.max_runtime:
                    can_stop = True
                    stop_reason = (
                        f"Max runtime {scheduler.max_runtime}s reached - "
                        f"Range 0 not achieved (current: {total_range})"
                    )
            elif elapsed >= scheduler.max_runtime:
                can_stop = True
                stop_reason = f"Max runtime {scheduler.max_runtime}s reached - FORCING STOP"
            elif elapsed < absolute_min_runtime:
                # Continue until minimum runtime unless Range 0
                can_stop = False
                # Minimal logging every 5 seconds only
                if int(elapsed) % 5 == 0 and elapsed > 0 and int(elapsed) != int(elapsed - 0.1):
                    printer(f"⏳ Optimizing... {absolute_min_runtime - elapsed:.1f}s remaining")
            elif generations_run >= scheduler.max_generations and elapsed >= absolute_min_runtime:
                # Only stop for max generations AFTER minimum runtime
                can_stop = True
                stop_reason = f"Max generations {scheduler.max_generations} reached"
        else:
            # First generation
            if elapsed >= scheduler.max_runtime:
                can_stop = True
                stop_reason = "Max runtime reached (no valid solution found)"

        # EXIT CONDITION
        if can_stop:
            printer(f"🏁 Stopping: {stop_reason} after {elapsed:.1f}s and {generations_run} generations")
            break

        # EVOLUTION STEP
        # Evaluate current population with optimized batch processing
        fitness_scores = scheduler._evaluate_population_optimized(population)

        # Track best with convergence and quality-focused improvements
        fitness_improved = False
        for ind, fitness in fitness_scores:
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = ind
                fitness_improved = True

                # Only log significant improvements and limit frequency
                current_time = now()
                fitness_improvement = scheduler._last_logged_fitness - fitness
                time_since_log = current_time - scheduler._last_progress_time

                # Log if major improvement OR enough time passed
                if fitness_improvement > 1000000 or time_since_log > scheduler._progress_update_interval:

                    schedule = scheduler._decode_cached(ind)
                    metrics = scheduler._evaluate_metrics_cached(schedule, ind)
                    best_ranges = {
                        "games": metrics["games_range"],
                        "partners": metrics["partners_range"],
                        "opponents": metrics["opponents_range"],
                        "courts": metrics["courts_range"],
                    }
                    total_range = sum(best_ranges.values())

                    printer(f"📈 Gen {generations_run}: Range = {total_range}, Fitness = {fitness:.0f}")

                    scheduler._last_progress_time = current_time
                    scheduler._last_logged_fitness = fitness

        # Enhanced convergence tracking for quality
        if fitness_improved:
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Check for convergence with Range 0 priority
        if best_individual is not None:
            schedule = scheduler._decode_cached(best_individual)
            metrics = scheduler._evaluate_metrics_cached(schedule, best_individual)
            signature = scheduler._get_duplicate_signature_cached(best_individual, schedule)
            avoidable_duplicate_rounds = avoidable_duplicate_rounds_from_signature(
                signature, scheduler.minimum_duplicate_rounds
            )
            total_range = sum(
                [
                    metrics["games_range"],
                    metrics["partners_range"],
                    metrics["opponents_range"],
                    metrics["courts_range"],
                ]
            )

            # Quality-focused convergence logic
            if total_range == 0 and avoidable_duplicate_rounds == 0:  # Perfect result achieved
                if verbose:
                    printer(f"✅ Found optimal solution (Range 0) at generation {generations_run}")
                # DON'T STOP - let main loop handle timing constraints
                if elapsed >= perfect_stop_runtime and generations_without_improvement > 30:
                    if verbose:
                        printer("🏁 Range 0 confirmed stable after sufficient runtime - stopping")
                    break
            elif generations_without_improvement >= scheduler.convergence_patience:
                if verbose:
                    printer(f"Converged after {generations_without_improvement} generations without improvement")
                # CRITICAL: Don't stop unless we've used most of our runtime!
                # This ensures we run close to the full minute for quality
                if elapsed < scheduler.max_runtime * 0.8:
                    if verbose:
                        printer(f"Continuing search - only {elapsed:.1f}s of {scheduler.max_runtime}s used...")
                    # Reset convergence to give more time
                    generations_without_improvement = scheduler.convergence_patience - 50
                else:
                    printer("🏁 Convergence reached - stopping")
                    break
        fitness_scores.sort(key=lambda x: x[1])

        # Create next generation
        new_population = []

        # Elitism - keep best individuals
        elite_count = max(2, scheduler.population_size // 10)
        for ind, _ in fitness_scores[:elite_count]:
            new_population.append(ind)

        # Fill rest with crossover and mutation
        while len(new_population) < scheduler.population_size:
            # Tournament selection
            parent1 = scheduler._tournament_select(population, fitness_scores, rng, k=scheduler.tournament_size)
            parent2 = scheduler._tournament_select(population, fitness_scores, rng, k=scheduler.tournament_size)

            # Crossover
            if rng.random() < 0.9:
                child1, child2 = scheduler._crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1, parent2

            # SUPER aggressive mutation for Range 0
            child1 = scheduler._super_aggressive_mutate(child1, rng)
            child2 = scheduler._super_aggressive_mutate(child2, rng)

            new_population.extend([child1, child2])

        # Trim to population size
        population = new_population[: scheduler.population_size]

        # Inject diversity MORE frequently when pursuing Range 0
        diversity_interval = 15 if scheduler.range0_possible else 30
        if generations_run % diversity_interval == 0 and generations_run > 0:
            current_best_range = sum(
                [
                    current_ranges["games"],
                    current_ranges["partners"],
                    current_ranges["opponents"],
                    current_ranges["courts"],
                ]
            )
            printer(
                f"💉 Injecting MASSIVE diversity at generation {generations_run} "
                f"(current range: {current_best_range})"
            )
            replacement_fraction = 0.75 if (scheduler.range0_possible and current_best_range > 0) else 0.5
            inject_diversity(
                population,
                rng,
                replacement_fraction,
                make_individual=lambda: scheduler._random_individual(rng),
            )

        generations_run += 1

    return best_individual, best_fitness, generations_run
