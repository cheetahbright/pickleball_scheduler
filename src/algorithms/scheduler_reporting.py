"""Reporting and result-formatting helpers for the genetic scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable, Mapping

GameTuple = tuple[str, str, str, str]

RANGE_METRIC_KEYS = (
    "games_range",
    "partners_range",
    "opponents_range",
    "courts_range",
)


@dataclass
class Game:
    """Legacy game object shape returned to the Streamlit app and older tests."""

    team1: list[str]
    team2: list[str]
    court: int


def sum_range_metrics(metrics: Mapping[str, int]) -> int:
    """Return the scheduler's total range from the standard component metrics."""
    return sum(int(metrics[key]) for key in RANGE_METRIC_KEYS)


def format_schedule(schedule: list[list[GameTuple]]) -> list[dict[str, Any]]:
    """Format the raw tuple schedule into the legacy round/game payload."""
    rounds: list[dict[str, Any]] = []
    for round_index, round_games in enumerate(schedule, start=1):
        games: list[Game] = []
        for court_index, game in enumerate(round_games, start=1):
            p1, p2, p3, p4 = map(str, game)
            games.append(Game([p1, p2], [p3, p4], court_index))
        rounds.append({"round": round_index, "games": games})
    return rounds


def build_scheduler_result(
    *,
    formatted_schedule: list[dict[str, Any]],
    best_fitness: float,
    generations_run: int,
    elapsed_seconds: float,
    final_metrics: Mapping[str, int],
    optimal_total_range: int,
    duplicate_rounds: int,
    avoidable_duplicate_rounds: int,
    minimum_duplicate_rounds: int,
    max_unique_rounds: int,
    fitness_details: Mapping[str, int] | None,
    algorithm: str = "Genetic Algorithm",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the backward-compatible scheduler result payload."""
    raw_metrics = {key: int(final_metrics[key]) for key in RANGE_METRIC_KEYS}
    total_range = sum_range_metrics(raw_metrics)
    score = 100 - min(100, total_range * 10)

    result = {
        "schedule": formatted_schedule,
        "success": total_range == 0 and avoidable_duplicate_rounds == 0,
        "fitness": best_fitness,
        "score": score,
        "fitness_score": score,
        "generations": generations_run,
        "time_seconds": elapsed_seconds,
        "algorithm": algorithm,
        "total_range": total_range,
        "optimal_total_range": optimal_total_range,
        "duplicate_rounds": duplicate_rounds,
        "avoidable_duplicate_rounds": avoidable_duplicate_rounds,
        "minimum_duplicate_rounds": minimum_duplicate_rounds,
        "max_unique_rounds": max_unique_rounds,
        "games_range": raw_metrics["games_range"],
        "partners_range": raw_metrics["partners_range"],
        "opponents_range": raw_metrics["opponents_range"],
        "courts_range": raw_metrics["courts_range"],
        "fitness_details": dict(fitness_details) if fitness_details else None,
        "raw_metrics": raw_metrics,
        "metrics": dict(raw_metrics),
    }
    if seed is not None:
        result["seed"] = seed
    return result


def build_scheduler_failure_result(
    *,
    elapsed_seconds: float,
    generations_run: int,
    optimal_total_range: int,
    minimum_duplicate_rounds: int,
    max_unique_rounds: int,
    error: str = "No valid solution found",
    algorithm: str = "Genetic Algorithm",
) -> dict[str, Any]:
    """Build a dict-shaped failure result that matches the maintained scheduler contract."""
    return {
        "schedule": [],
        "success": False,
        "error": error,
        "fitness": None,
        "score": 0,
        "fitness_score": 0,
        "generations": generations_run,
        "time_seconds": elapsed_seconds,
        "algorithm": algorithm,
        "total_range": None,
        "optimal_total_range": optimal_total_range,
        "duplicate_rounds": 0,
        "avoidable_duplicate_rounds": 0,
        "minimum_duplicate_rounds": minimum_duplicate_rounds,
        "max_unique_rounds": max_unique_rounds,
        "games_range": None,
        "partners_range": None,
        "opponents_range": None,
        "courts_range": None,
        "fitness_details": None,
        "raw_metrics": {},
        "metrics": {},
    }


def print_scheduler_start_banner(
    printer: Callable[[str], None],
    *,
    absolute_min_runtime: float,
    perfect_stop_runtime: float,
    max_runtime: float,
    partner_range0_possible: bool,
    optimal_partner_range: int,
    range0_possible: bool,
    optimal_total_range: int,
    max_unique_rounds: int,
    minimum_duplicate_rounds: int,
    seed: int,
) -> None:
    """Print the runtime/feasibility banner for a scheduler run."""
    printer(f"🕐 GA Starting with ENFORCED minimum runtime: {absolute_min_runtime}s")
    printer(f"   Perfect-stop runtime floor: {perfect_stop_runtime}s")
    printer(f"   Maximum runtime: {max_runtime}s")
    printer(f"   Partner Range 0 possible: {partner_range0_possible}")
    printer(f"   Optimal partner range: {optimal_partner_range}")
    printer(f"   Overall Range 0 possible: {range0_possible}")
    printer(f"   Theoretical total range minimum: {optimal_total_range}")
    printer(f"   Canonical round patterns available: {max_unique_rounds}")
    printer(f"   Minimum duplicate rounds required: {minimum_duplicate_rounds}")
    printer(f"   Random seed: {seed} (different each run for variety)")


def print_scheduler_final_report(
    printer: Callable[[str], None],
    *,
    elapsed_seconds: float,
    final_metrics: Mapping[str, int],
    duplicate_rounds: int,
    avoidable_duplicate_rounds: int,
    minimum_duplicate_rounds: int,
    partner_range0_possible: bool,
    optimal_partner_range: int,
    optimal_total_range: int,
    num_players: int,
    max_unique_rounds: int,
) -> None:
    """Print the final report and optimality assessment for a scheduler run."""
    total_range = sum_range_metrics(final_metrics)

    printer(f"\n📊 FINAL RESULTS after {elapsed_seconds:.1f}s:")
    printer(f"   Games Range: {final_metrics['games_range']}")
    printer(f"   Partners Range: {final_metrics['partners_range']}")
    printer(f"   Opponents Range: {final_metrics['opponents_range']}")
    printer(f"   Courts Range: {final_metrics['courts_range']}")
    printer(f"   Total Range: {total_range}")
    printer(
        "   Duplicate Rounds: "
        f"{duplicate_rounds} "
        f"(minimum expected: {minimum_duplicate_rounds}, "
        f"avoidable excess: {avoidable_duplicate_rounds})"
    )

    if partner_range0_possible and int(final_metrics["partners_range"]) > optimal_partner_range:
        printer("\n⚠️ WARNING: Partner Range 0 was mathematically possible but NOT achieved!")
        printer("   This schedule is SUBOPTIMAL and should have continued optimizing!")
    elif total_range == optimal_total_range:
        printer(f"\n✅ OPTIMAL: Achieved the theoretical minimum for {num_players} players!")
        if minimum_duplicate_rounds > 0:
            printer(
                "   Round repeats are expected here because only "
                f"{max_unique_rounds} canonical round patterns exist."
            )
    elif total_range > optimal_total_range:
        printer(f"\n⚠️ SUBOPTIMAL: Could be improved further (target Range {optimal_total_range})")
    else:
        printer("\n🏆 PERFECT: Exceeded the current theoretical estimate for this player count!")


def invoke_progress_callback(
    callback: Callable[..., object],
    progress_data: Mapping[str, Any],
) -> None:
    """Invoke either the modern dict callback or the legacy 3-arg callback shape."""
    try:
        callback_signature = signature(callback)
    except (TypeError, ValueError):
        callback(progress_data)
        return

    positional_params = [
        parameter
        for parameter in callback_signature.parameters.values()
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in callback_signature.parameters.values()
    )

    if has_varargs or len(positional_params) <= 1:
        callback(progress_data)
        return

    if len(positional_params) >= 4:
        callback(
            int(progress_data["generation"]),
            float(progress_data["best_fitness"]),
            int(progress_data["total_range"]),
            float(progress_data["elapsed_time"]),
        )
        return

    callback(
        int(progress_data["generation"]),
        float(progress_data["best_fitness"]),
        float(progress_data["elapsed_time"]),
    )
