#!/usr/bin/env python3
"""Stress-test sweep: same generation path the Main Scheduler button uses,
run across many player/court/round/constraint combinations to surface
breakage that a single manual run wouldn't catch.
"""

from __future__ import annotations

import random
import traceback
from itertools import product
from typing import Any, Callable, Sequence

# How many courts a sweep tries for each player count (see stress_test_court_counts).
COURT_MODES = ("max", "one_fewer", "all")

# Runs that found a problem. A "constraint_warning" run broke constraints that
# no round of its size can meet, so it is reported without failing the sweep.
FAILURE_STATUSES = ("invalid_schedule", "constraint_violation", "unbalanced_games", "exception", "no_schedule")
WARNING_STATUSES = ("constraint_warning",)

# Placements _round_can_meet_constraints tries before giving up - far more
# than the few random constraints of a sweep need.
_MAX_ROUND_SEARCH_STEPS = 100_000


class _RoundSearchGaveUp(Exception):
    pass


def _random_constraint_pairs(players: Sequence[str], count: int, rng: random.Random) -> list[tuple[str, str]]:
    """Pick `count` distinct unordered player pairs at random."""
    if count <= 0 or len(players) < 2:
        return []
    all_pairs = [(players[i], players[j]) for i in range(len(players)) for j in range(i + 1, len(players))]
    rng.shuffle(all_pairs)
    return all_pairs[:count]


def stress_test_court_counts(num_players: int, court_mode: str = "max") -> list[int]:
    """The court counts a sweep runs num_players on.

    "max" fills as many courts as the players can (players // 4); "one_fewer"
    uses one court less, so more players sit out each round; "all" runs every
    count from 1 up to the max.
    """
    if court_mode not in COURT_MODES:
        raise ValueError(f"Unknown court mode: {court_mode!r}")
    max_courts = max(1, num_players // 4)
    if court_mode == "all":
        return list(range(1, max_courts + 1))
    if court_mode == "one_fewer":
        return [max(1, max_courts - 1)]
    return [max_courts]


def count_stress_test_runs(
    player_counts: Sequence[int], round_counts: Sequence[int], trials_per_combo: int, court_mode: str = "max"
) -> int:
    """How many generations a sweep with these settings runs."""
    court_counts = sum(len(stress_test_court_counts(num_players, court_mode)) for num_players in player_counts)
    return court_counts * len(round_counts) * trials_per_combo


def _round_can_meet_constraints(
    players: Sequence[str],
    courts: int,
    pair_constraints: Sequence[tuple[str, str]],
    oppose_constraints: Sequence[tuple[str, str]],
    max_steps: int = _MAX_ROUND_SEARCH_STEPS,
) -> bool | None:
    """Whether some round of `courts` games breaks none of the constraints.

    If one does, every round of a schedule can, so any constraint a schedule
    breaks was avoidable. Only players named in a constraint matter - anyone
    else fits in any game - so each of those is placed in turn: sitting out,
    joining a game, or starting the next one, as long as every game can still
    be split into two teams that break nothing. None if the search gives up
    after max_steps placements.
    """
    roster = set(players)
    no_pair = {frozenset(pair) for pair in pair_constraints if len(set(pair)) == 2 and set(pair) <= roster}
    no_oppose = {frozenset(pair) for pair in oppose_constraints if len(set(pair)) == 2 and set(pair) <= roster}
    named = {name for pair in no_pair | no_oppose for name in pair}
    constrained = [name for name in players if name in named]
    games: list[list[str]] = []
    steps = 0

    def splittable(game: list[str]) -> bool:
        # Two teams of two; players named in no constraint fill the open seats.
        first, *others = game
        for partner in (None, *others):
            team_a = {first} if partner is None else {first, partner}
            team_b = set(others) - team_a
            if len(team_b) > 2:
                continue
            partners = [frozenset(team) for team in (team_a, team_b) if len(team) == 2]
            opponents = [frozenset((a, b)) for a in team_a for b in team_b]
            if not any(pair in no_pair for pair in partners) and not any(pair in no_oppose for pair in opponents):
                return True
        return False

    def place(index: int, sit_outs_left: int) -> bool:
        nonlocal steps
        steps += 1
        if steps > max_steps:
            raise _RoundSearchGaveUp
        if index == len(constrained):
            return True
        name = constrained[index]
        for game in games:
            if len(game) < 4:
                game.append(name)
                if splittable(game) and place(index + 1, sit_outs_left):
                    return True
                game.pop()
        if len(games) < courts:
            games.append([name])
            if place(index + 1, sit_outs_left):
                return True
            games.pop()
        return sit_outs_left > 0 and place(index + 1, sit_outs_left - 1)

    try:
        return place(0, len(players) - 4 * courts)
    except _RoundSearchGaveUp:
        return None


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
    court_mode: str = "max",
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the real generation path across a grid of configurations.

    Builds and runs the scheduler the way the Main Scheduler's Generate does
    (same scheduler kwargs, same constraint wiring, same validation call), on
    the court counts court_mode picks, so a failure found here is a failure
    the GUI would actually hit. Each run is judged on more than that
    validation, though: a schedule fails if it breaks a constraint that some
    round could have met, or if a run without constraints splits the games
    less evenly than is always possible. Constraints that no round can meet
    make a "constraint_warning" run instead.

    progress_callback, if given, is called as (completed, total) after every
    trial - total_runs is known upfront (it's just the grid size), so the UI
    can show real progress instead of an indeterminate spinner for sweeps
    that can take a very long time.

    Each row is appended to `results` (a new list if None) as soon as its run
    ends, so a caller whose sweep is cut short still holds the finished rows.
    """
    rng = random.Random(seed)
    if results is None:
        results = []
    total_runs = count_stress_test_runs(player_counts, round_counts, trials_per_combo, court_mode)

    for num_players in player_counts:
        players = [f"Player{i + 1}" for i in range(num_players)]

        for courts, num_rounds in product(stress_test_court_counts(num_players, court_mode), round_counts):
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
                    # generations == 0 marks a constructive-fast-path row (see
                    # constructive_scheduler.py) - distinguishing it from a GA
                    # row (generations > 0) is the whole point of surfacing
                    # this column in the stress-test view.
                    "generations": None,
                    "time_seconds": None,
                    "constraint_violations": None,
                    "games_range": None,
                    "errors": [],
                }

                try:
                    scheduler = scheduler_cls(
                        players=players,
                        num_courts=courts,
                        num_rounds=num_rounds,
                        use_desktop_params=True,
                        max_runtime=max_time,
                    )

                    if pair_constraints or oppose_constraints:
                        scheduler.add_constraints(
                            pair_constraints=pair_constraints,
                            oppose_constraints=oppose_constraints,
                        )

                    result = scheduler.generate_schedule(max_time=max_time, seed=trial_seed)

                    # A scheduler that finds nothing returns an empty schedule,
                    # which would otherwise pass validation.
                    if not (isinstance(result, dict) and result.get("schedule")):
                        row["status"] = "no_schedule"
                        if isinstance(result, dict) and result.get("error"):
                            row["errors"] = [str(result["error"])]
                        results.append(row)
                        if progress_callback is not None:
                            progress_callback(len(results), total_runs)
                        continue

                    schedule_data = result["schedule"]
                    schedule_errors = validate_schedule_integrity_fn(schedule_data, players)
                    row["total_range"] = result.get("total_range")
                    row["generations"] = result.get("generations")
                    row["time_seconds"] = result.get("time_seconds")
                    constraint_violations = int(result.get("constraint_violations") or 0)
                    row["constraint_violations"] = constraint_violations
                    games_range = result.get("games_range")
                    row["games_range"] = games_range
                    # The game slots divide evenly among the players, or within one.
                    game_slots = 4 * sum(len(round_data.get("games", [])) for round_data in schedule_data)
                    games_range_floor = 0 if game_slots % num_players == 0 else 1

                    if schedule_errors:
                        row["status"] = "invalid_schedule"
                        row["errors"] = schedule_errors
                    elif constraint_violations:
                        broken = f"{constraint_violations} game(s) break a 'won't pair' or 'won't oppose' constraint"
                        can_meet = _round_can_meet_constraints(players, courts, pair_constraints, oppose_constraints)
                        if can_meet is False:
                            row["status"] = "constraint_warning"
                            # Only that some break is unavoidable is known: how
                            # few a schedule can break also depends on how its
                            # rounds must differ, so the count isn't judged.
                            row["errors"] = [
                                f"{broken}, but no round of {courts} game(s) can meet them all, so some always will. "
                                "Whether this schedule breaks more of them than it has to isn't checked."
                            ]
                        else:
                            row["status"] = "constraint_violation"
                            reason = (
                                f"though a round of {courts} game(s) can meet them all"
                                if can_meet
                                else "and the check for whether a round can meet them all gave up"
                            )
                            row["errors"] = [f"{broken}, {reason}."]
                    elif (
                        not (pair_constraints or oppose_constraints)
                        and isinstance(games_range, int)
                        and games_range > games_range_floor
                    ):
                        row["status"] = "unbalanced_games"
                        row["errors"] = [
                            f"Games played per player differ by up to {games_range}, though without constraints "
                            f"they can always differ by at most {games_range_floor}."
                        ]
                    else:
                        row["status"] = "ok"

                except Exception as exc:  # noqa: BLE001 - deliberately broad; this IS the crash detector
                    row["status"] = "exception"
                    row["errors"] = [f"{type(exc).__name__}: {exc}"]
                    row["traceback"] = traceback.format_exc()

                results.append(row)
                if progress_callback is not None:
                    progress_callback(len(results), total_runs)

    return results


def summarize_stress_test(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count outcomes by status for a quick pass/fail readout."""
    summary = {"total": len(results), "ok": 0}
    summary.update({status: 0 for status in WARNING_STATUSES + FAILURE_STATUSES})
    for row in results:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return summary
