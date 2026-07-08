#!/usr/bin/env python3
"""Stress-test sweep: same generation path the Main Scheduler button uses,
run across many player/round/constraint combinations to surface breakage
that a single manual run wouldn't catch.
"""

from __future__ import annotations

import random
import traceback
from typing import Any, Callable, Sequence


def _random_constraint_pairs(players: Sequence[str], count: int, rng: random.Random) -> list[tuple[str, str]]:
    """Pick `count` distinct unordered player pairs at random."""
    if count <= 0 or len(players) < 2:
        return []
    all_pairs = [(players[i], players[j]) for i in range(len(players)) for j in range(i + 1, len(players))]
    rng.shuffle(all_pairs)
    return all_pairs[:count]


def run_stress_test(
    scheduler_cls: Callable[..., Any],
    validate_schedule_integrity_fn: Callable[[Any, list[str]], list[str]],
    *,
    player_counts: Sequence[int],
    round_counts: Sequence[int],
    max_time: float,
    num_pair_constraints: int,
    num_oppose_constraints: int,
    trials_per_combo: int = 1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run the real generation path across a grid of configurations.

    Mirrors render_main_scheduler_tab's generation block exactly (same
    scheduler kwargs, same constraint wiring, same validation call) so a
    failure found here is a failure the GUI would actually hit.
    """
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []

    for num_players in player_counts:
        players = [f"Player{i + 1}" for i in range(num_players)]
        courts = max(1, num_players // 4)

        for num_rounds in round_counts:
            for trial in range(trials_per_combo):
                pair_constraints = _random_constraint_pairs(players, num_pair_constraints, rng)
                oppose_constraints = _random_constraint_pairs(players, num_oppose_constraints, rng)
                trial_seed = rng.randrange(1_000_000)

                row: dict[str, Any] = {
                    "players": num_players,
                    "rounds": num_rounds,
                    "courts": courts,
                    "pair_constraints": len(pair_constraints),
                    "oppose_constraints": len(oppose_constraints),
                    "trial": trial,
                    "seed": trial_seed,
                    "status": "ok",
                    "total_range": None,
                    "errors": [],
                }

                try:
                    scheduler = scheduler_cls(
                        players=players,
                        num_courts=courts,
                        num_rounds=num_rounds,
                        use_desktop_params=True,
                        max_runtime=max_time,
                        max_generations=max(2000, int(max_time * 70)),
                    )

                    if pair_constraints or oppose_constraints:
                        scheduler.add_constraints(
                            pair_constraints=pair_constraints,
                            oppose_constraints=oppose_constraints,
                        )

                    result = scheduler.generate_schedule(max_time=max_time, seed=trial_seed)

                    if not (isinstance(result, dict) and "schedule" in result):
                        row["status"] = "no_schedule"
                        results.append(row)
                        continue

                    schedule_data = result["schedule"]
                    schedule_errors = validate_schedule_integrity_fn(schedule_data, players)
                    row["total_range"] = result.get("total_range")

                    if schedule_errors:
                        row["status"] = "invalid_schedule"
                        row["errors"] = schedule_errors
                    else:
                        row["status"] = "ok"

                except Exception as exc:  # noqa: BLE001 - deliberately broad; this IS the crash detector
                    row["status"] = "exception"
                    row["errors"] = [f"{type(exc).__name__}: {exc}"]
                    row["traceback"] = traceback.format_exc()

                results.append(row)

    return results


def summarize_stress_test(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count outcomes by status for a quick pass/fail readout."""
    summary = {"total": len(results), "ok": 0, "invalid_schedule": 0, "exception": 0, "no_schedule": 0}
    for row in results:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return summary
