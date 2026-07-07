"""Repair and partner-analysis helpers for the genetic scheduler."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Callable, cast

try:
    from src.algorithms.scheduler_metrics import GameTuple
    from src.algorithms.scheduler_reporting import sum_range_metrics
except ImportError:
    from algorithms.scheduler_metrics import GameTuple
    from algorithms.scheduler_reporting import sum_range_metrics

ScheduleMetricsEvaluator = Callable[[list[list[GameTuple]]], dict[str, int]]
RepairPrinter = Callable[[str], None]


def repair_invalid_schedule(
    schedule: list[list[GameTuple]],
    problematic_round_idx: int,
    arrangements: list[list[GameTuple]],
    printer: RepairPrinter,
) -> list[list[GameTuple]]:
    """Replace an invalid round when a conflict-free arrangement is available."""
    used_players: set[str] = set()
    for round_idx, round_games in enumerate(schedule):
        if round_idx != problematic_round_idx:
            for game in round_games:
                used_players.update(map(str, game))

    available_arrangements: list[tuple[int, list[GameTuple]]] = []
    for arrangement_index, arrangement in enumerate(arrangements):
        arrangement_players: set[str] = set()
        for game in arrangement:
            arrangement_players.update(map(str, game))

        if not (arrangement_players & used_players):
            available_arrangements.append((arrangement_index, arrangement))

    if available_arrangements:
        replacement_idx, replacement_arrangement = available_arrangements[0]
        schedule[problematic_round_idx] = replacement_arrangement
        printer(f"✅ Repaired round {problematic_round_idx + 1} with arrangement {replacement_idx}")
    else:
        printer(f"⚠️ No perfect replacement for round {problematic_round_idx + 1}, using fallback")

    return schedule


def count_partners(schedule: list[list[GameTuple]]) -> dict[str, dict[str, int]]:
    """Count partner pairings for each player in the schedule."""
    partner_count: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for round_games in schedule:
        for game in round_games:
            p1, p2, p3, p4 = map(str, game)
            partner_count[p1][p2] += 1
            partner_count[p2][p1] += 1
            partner_count[p3][p4] += 1
            partner_count[p4][p3] += 1

    return {player: dict(partners) for player, partners in partner_count.items()}


def repair_partner_imbalance(
    schedule: list[list[GameTuple]],
    max_time: float,
    *,
    evaluate_metrics: ScheduleMetricsEvaluator,
    printer: RepairPrinter,
    rng: random.Random | None = None,
    clock: Callable[[], float] | None = None,
) -> list[list[GameTuple]]:
    """Target partner-balance repairs with lightweight round-to-round swaps."""
    rand = rng if rng is not None else random
    now = clock if clock is not None else time.time
    start_time = now()
    best_schedule = [round_games[:] for round_games in schedule]
    current_metrics = evaluate_metrics(best_schedule)

    if current_metrics["partners_range"] == 0:
        return best_schedule

    printer(f"   🔧 Partner repair starting: range {current_metrics['partners_range']}")

    attempts = 0
    max_attempts = 200
    improvements = 0
    swap_strategies = [
        [(0, 0), (1, 1)],
        [(2, 2), (3, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
        [(0, 1)],
        [(2, 3)],
        [(0, 2)],
        [(1, 3)],
    ]

    while now() - start_time < max_time and attempts < max_attempts:
        attempts += 1
        test_schedule = [round_games[:] for round_games in best_schedule]

        r1_idx = rand.randint(0, len(test_schedule) - 1)
        r2_idx = rand.randint(0, len(test_schedule) - 1)
        if r1_idx == r2_idx or len(test_schedule[r1_idx]) == 0 or len(test_schedule[r2_idx]) == 0:
            continue

        g1_idx = rand.randint(0, len(test_schedule[r1_idx]) - 1)
        g2_idx = rand.randint(0, len(test_schedule[r2_idx]) - 1)

        game1 = list(test_schedule[r1_idx][g1_idx])
        game2 = list(test_schedule[r2_idx][g2_idx])
        strategy = swap_strategies[attempts % len(swap_strategies)]

        for pos1, pos2 in strategy:
            if pos1 < len(game1) and pos2 < len(game2):
                game1[pos1], game2[pos2] = game2[pos2], game1[pos1]

        test_schedule[r1_idx][g1_idx] = cast(GameTuple, tuple(game1))
        test_schedule[r2_idx][g2_idx] = cast(GameTuple, tuple(game2))

        test_metrics = evaluate_metrics(test_schedule)
        current_best_metrics = evaluate_metrics(best_schedule)
        is_improvement = test_metrics["partners_range"] < current_best_metrics["partners_range"] or (
            test_metrics["partners_range"] == current_best_metrics["partners_range"]
            and test_metrics["opponents_range"] <= current_best_metrics["opponents_range"]
            and test_metrics["games_range"] <= current_best_metrics["games_range"]
            and test_metrics["courts_range"] <= current_best_metrics["courts_range"]
        )

        if is_improvement and test_metrics["violations"] == 0:
            best_schedule = test_schedule
            improvements += 1
            printer(
                f"   🎯 Improvement #{improvements}: "
                f"P:{test_metrics['partners_range']} "
                f"O:{test_metrics['opponents_range']} "
                f"(attempt {attempts})"
            )
            if test_metrics["partners_range"] == 0:
                printer(f"   🎉 Partner balance achieved after {attempts} attempts!")
                break

    final_metrics = evaluate_metrics(best_schedule)
    final_total = sum_range_metrics(final_metrics)
    printer(f"   🏁 Partner repair complete: {attempts} attempts, {improvements} improvements")
    printer(f"      Partners Range: {final_metrics['partners_range']} | Total Range: {final_total}")
    printer(
        f"      Breakdown - G:{final_metrics['games_range']} "
        f"P:{final_metrics['partners_range']} "
        f"O:{final_metrics['opponents_range']} "
        f"C:{final_metrics['courts_range']}"
    )

    return best_schedule


def repair_opponent_imbalance(
    schedule: list[list[GameTuple]],
    max_time: float,
    *,
    evaluate_metrics: ScheduleMetricsEvaluator,
    printer: RepairPrinter,
) -> list[list[GameTuple]]:
    """Placeholder opponent repair hook kept separate from the GA class."""
    _ = (max_time, evaluate_metrics, printer)
    return schedule


def apply_targeted_repairs(
    schedule: list[list[GameTuple]],
    max_time: float,
    *,
    evaluate_metrics: ScheduleMetricsEvaluator,
    printer: RepairPrinter,
    partner_repair: Callable[[list[list[GameTuple]], float], list[list[GameTuple]]],
    opponent_repair: Callable[[list[list[GameTuple]], float], list[list[GameTuple]]],
    clock: Callable[[], float] | None = None,
) -> list[list[GameTuple]] | None:
    """Apply targeted local repairs and report the resulting fairness ranges."""
    now = clock if clock is not None else time.time
    start_time = now()
    current_schedule = [round_games[:] for round_games in schedule]
    current_metrics = evaluate_metrics(current_schedule)

    printer(
        "   Starting repair: "
        f"G:{current_metrics['games_range']} "
        f"P:{current_metrics['partners_range']} "
        f"O:{current_metrics['opponents_range']} "
        f"C:{current_metrics['courts_range']}"
    )

    if current_metrics["partners_range"] > 0:
        current_schedule = partner_repair(current_schedule, max_time * 0.8)
        current_metrics = evaluate_metrics(current_schedule)

    if current_metrics["opponents_range"] > 0 and now() - start_time < max_time:
        remaining_time = max_time - (now() - start_time)
        current_schedule = opponent_repair(current_schedule, remaining_time)
        current_metrics = evaluate_metrics(current_schedule)

    final_total = sum_range_metrics(current_metrics)
    if final_total == 0:
        printer("   🎉 REPAIR SUCCESS: Achieved Range 0!")
        return current_schedule

    printer(f"   ⚠️ REPAIR PARTIAL: Reduced to total range {final_total}")
    printer(
        f"      Breakdown - G:{current_metrics['games_range']} "
        f"P:{current_metrics['partners_range']} "
        f"O:{current_metrics['opponents_range']} "
        f"C:{current_metrics['courts_range']}"
    )
    return current_schedule
