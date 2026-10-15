"""Plan one schedule around players who arrive late or leave early.

Each player is present from a first round to a last round (1-based,
inclusive; the whole session unless told otherwise), so the roster changes
from round to round. The rounds are cut into maximal runs of consecutive
rounds with the same roster - segments - and each segment is scheduled by
the genetic scheduler for exactly the players present, then spliced after
the segments before it. A round therefore only ever holds players who are
there, with at most the courts they can fill.

Planned alone, every segment would be fair only among itself and the
sit-outs could pile onto the same players from one segment to the next. So
each segment is told what the earlier ones did: its scheduler gets minus
every present player's sit-outs so far as baseline_games, which makes the
games range it puts first the range of everyone's sit-outs to date, and the
sit-outs of the rounds before it as preceding_sit_outs, so a sit-out streak
does not run across the boundary needlessly.

Each segment is planned in turn, without looking ahead, so the plan is fair
at every step rather than guaranteed best over the whole session. When the
scheduler finds the best schedule for a segment (never below the minimum
min_games_range_with_offsets proves), the players present at the end of it
have sat out as evenly as its rounds allow, given what came before. Sit-outs
are counted, not weighed by how long someone was there: a late arrival
starts at 0 and is first in line to sit out, and a player about to leave is
not given extra sit-outs to even things out before going. So after early
departures the range over everyone, leavers included, can trail the best
any plan could reach (by 2 at most in the scenarios tried), while the
players who stay are as even as the segments allow.

This module has no Streamlit dependency; the scheduler class is passed in.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

try:
    from src.algorithms.scheduler_reporting import invoke_progress_callback
    from src.utils.roster_rounds import constraint_pairs_for_roster, sit_outs_by_round, splice_schedules
    from src.utils.schedule_shapes import extract_game_teams, games_in_round
except ImportError:
    from algorithms.scheduler_reporting import invoke_progress_callback
    from utils.roster_rounds import constraint_pairs_for_roster, sit_outs_by_round, splice_schedules
    from utils.schedule_shapes import extract_game_teams, games_in_round

# A game needs four players, so no round can run with fewer.
PLAYERS_PER_COURT = 4

# The least time a segment is given when max_time can afford it (see segment_budgets).
MIN_SEGMENT_SECONDS = 2.0

RANGE_KEYS = ("games_range", "partners_range", "opponents_range", "courts_range")

# Who is here when: a (first round, last round) pair per player, 1-based and
# inclusive. A player left out is here for every round, and so is a side left None.
Attendance = Mapping[str, Sequence[int | None]]


class AttendanceError(ValueError):
    """The attendance can't be planned; ``problems`` lists everything wrong with it."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class AttendanceSegment:
    """A run of consecutive rounds that all have the same roster."""

    first_round: int
    last_round: int
    players: tuple[str, ...]

    @property
    def num_rounds(self) -> int:
        return self.last_round - self.first_round + 1


def _is_whole_number(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _round_list(rounds: Sequence[int]) -> str:
    """Round numbers as runs: [1, 2, 3, 7] -> "1-3, 7"."""
    runs: list[list[int]] = []
    for number in rounds:
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in runs)


def _read_window(name: str, window: object, num_rounds: int, problems: list[str]) -> tuple[int, int] | None:
    """One player's (first round, last round), or None - with what is wrong added to problems."""
    if isinstance(window, (str, bytes)) or not isinstance(window, Sequence) or len(window) != 2:
        problems.append(f"{name!r} needs a (first round, last round) pair, not {window!r}")
        return None
    first, last = window
    if not all(value is None or _is_whole_number(value) for value in (first, last)):
        problems.append(f"{name!r} needs whole round numbers, not {first!r} and {last!r}")
        return None
    first = 1 if first is None else int(first)
    last = num_rounds if last is None else int(last)
    valid = True
    if first < 1 or last > num_rounds or first > num_rounds or last < 1:
        problems.append(f"{name!r} is present for rounds {first}-{last}, outside rounds 1-{num_rounds}")
        valid = False
    if first > last:
        problems.append(f"{name!r} arrives in round {first}, after leaving in round {last}")
        valid = False
    return (first, last) if valid else None


def normalize_attendance(
    players: Sequence[str], num_rounds: int, attendance: Attendance | None = None
) -> dict[str, tuple[int, int]]:
    """Every player's (first round, last round), checked.

    Raises AttendanceError naming every problem found: a player who arrives
    after leaving or is present outside rounds 1..num_rounds, a name that is
    not a player, and rounds with fewer than four players present (a game
    needs four).
    """
    problems: list[str] = []
    if num_rounds < 1:
        problems.append(f"num_rounds must be at least 1, not {num_rounds}")
    names = [str(player) for player in players]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"player name(s) given more than once: {', '.join(repr(name) for name in duplicates)}")

    windows = {name: (1, num_rounds) for name in names}
    for raw_name, window in (attendance or {}).items():
        name = str(raw_name)
        if name not in windows:
            problems.append(f"{name!r} is not one of the players")
            continue
        read = _read_window(name, window, num_rounds, problems)
        if read is not None:
            windows[name] = read

    if not problems:
        short = [
            (number, present)
            for number in range(1, num_rounds + 1)
            if (present := sum(first <= number <= last for first, last in windows.values())) < PLAYERS_PER_COURT
        ]
        if short:
            counts = ", ".join(str(present) for _number, present in short)
            problems.append(
                f"fewer than {PLAYERS_PER_COURT} players are present in round(s) "
                f"{_round_list([number for number, _present in short])} ({counts} present) - a game needs "
                f"{PLAYERS_PER_COURT}"
            )
    if problems:
        raise AttendanceError(problems)
    return windows


def rosters_by_round(players: Sequence[str], num_rounds: int, attendance: Attendance | None = None) -> list[list[str]]:
    """Who is present in each round, in the order of players. Raises AttendanceError like normalize_attendance."""
    windows = normalize_attendance(players, num_rounds, attendance)
    names = [str(player) for player in players]
    return [
        [name for name in names if windows[name][0] <= number <= windows[name][1]]
        for number in range(1, num_rounds + 1)
    ]


def segment_rosters(rosters: Sequence[Sequence[str]]) -> list[AttendanceSegment]:
    """Cut the rounds into maximal runs of consecutive rounds whose rosters hold the same players."""
    segments: list[AttendanceSegment] = []
    for number, roster in enumerate(rosters, start=1):
        if segments and frozenset(segments[-1].players) == frozenset(roster):
            segments[-1] = replace(segments[-1], last_round=number)
        else:
            segments.append(AttendanceSegment(number, number, tuple(str(name) for name in roster)))
    return segments


def segment_budgets(max_time: float, segments: Sequence[AttendanceSegment]) -> list[float]:
    """Split max_time (seconds) across segments in proportion to rounds x players.

    Each segment gets at least MIN_SEGMENT_SECONDS (or an even share of
    max_time, if that is less), so a short segment is not starved; the rest is
    shared out by size. The shares add up to max_time, and are fixed by the
    segments alone - never by how long earlier ones took - so a replay of the
    plan gives every segment the budget it had. A single segment gets all of
    max_time. A segment ends sooner than its budget when its search reaches
    the best possible schedule.
    """
    if len(segments) == 1:
        return [float(max_time)]
    floor = min(MIN_SEGMENT_SECONDS, max_time / len(segments))
    spare = max_time - floor * len(segments)
    sizes = [segment.num_rounds * len(segment.players) for segment in segments]
    total = sum(sizes)
    return [floor + spare * size / total for size in sizes]


def segment_seed(master_seed: int, index: int) -> int:
    """The seed of segment ``index`` (0-based): the master seed for the first, derived from it for the rest."""
    if index == 0:
        return master_seed
    return random.Random(f"attendance-plan:{master_seed}:{index}").randrange(1_000_000)


def cumulative_sit_outs(schedule: Sequence[Any], roster_by_round: Sequence[Sequence[str]]) -> dict[str, int]:
    """How many rounds each player sat out while on the roster. Everyone on any roster is included."""
    counts = Counter({str(name): 0 for roster in roster_by_round for name in roster})
    for sitters in sit_outs_by_round(schedule, roster_by_round):
        counts.update(sitters)
    return dict(counts)


def sit_out_baseline(
    schedule: Sequence[Any], roster_by_round: Sequence[Sequence[str]], roster: Sequence[str]
) -> dict[str, int]:
    """The baseline_games that makes the next rounds balance sit-outs over all of schedule too.

    Minus the rounds each player of roster sat out while on roster_by_round's
    rosters (schedule and roster_by_round run in step, like sit_outs_by_round's
    arguments); a player who has not sat out, or was not there, is left out, so
    nothing to balance gives {}. Both the attendance planner's later segments and
    a mid-session replan hand this to their scheduler.
    """
    sat_out = cumulative_sit_outs(schedule, roster_by_round)
    return {str(name): -sat_out[str(name)] for name in roster if sat_out.get(str(name))}


def attendance_summary(schedule: Sequence[Any], roster_by_round: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """One row per player for showing who was here when: player, first_round, last_round,
    rounds_present, games_played and sit_outs (rounds sat out while present).

    Rounds are counted by position, 1 to len(schedule). Rows follow the order players first appear in a roster.
    """
    games: Counter[str] = Counter()
    for round_data in schedule:
        for game in games_in_round(round_data):
            team1, team2, _court = extract_game_teams(game)
            games.update(str(player) for player in [*team1, *team2])
    present_in: dict[str, list[int]] = {}
    for number, roster in enumerate(roster_by_round, start=1):
        for name in roster:
            present_in.setdefault(str(name), []).append(number)
    sit_outs = cumulative_sit_outs(schedule, roster_by_round)
    return [
        {
            "player": name,
            "first_round": rounds[0],
            "last_round": rounds[-1],
            "rounds_present": len(rounds),
            "games_played": games[name],
            "sit_outs": sit_outs[name],
        }
        for name, rounds in present_in.items()
    ]


def _plan_progress(
    progress_callback: Callable[..., object],
    *,
    elapsed_before: float,
    max_time: float,
    segment_number: int,
    segment_count: int,
) -> Callable[[Mapping[str, Any]], None]:
    """A segment's progress callback that reports in terms of the whole plan: the time since the plan
    started, out of max_time, with which segment of how many it is."""

    def on_progress(progress_data: Mapping[str, Any]) -> None:
        invoke_progress_callback(
            progress_callback,
            {
                **progress_data,
                "elapsed_time": elapsed_before + float(progress_data.get("elapsed_time", 0.0)),
                "max_time": max_time,
                "segment": segment_number,
                "segments": segment_count,
            },
        )

    return on_progress


def _plan_failure(
    roster_by_round: list[list[str]],
    master_seed: int,
    segments_done: list[dict[str, Any]],
    warnings: list[str],
    error: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "schedule": [],
        "roster_by_round": roster_by_round,
        "error": error,
        "constraint_violations": sum(segment["constraint_violations"] for segment in segments_done),
        "round_search_budget_reached": any(segment["round_search_budget_reached"] for segment in segments_done),
        "seed": master_seed,
        "generations": sum(segment["generations"] for segment in segments_done),
        "segments": segments_done,
        "warnings": warnings,
    }


def plan_attendance_schedule(
    players: Sequence[str],
    num_rounds: int,
    courts: int,
    attendance: Attendance | None,
    scheduler_cls: Callable[..., Any],
    *,
    constraints: Mapping[str, Sequence[Sequence[str]]] | None = None,
    max_time: float,
    seed: int | None = None,
    progress_callback: Callable[..., object] | None = None,
    generations: int | Sequence[int] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Plan num_rounds rounds in which every round uses exactly the players present.

    attendance maps a player to the (first round, last round) they are here
    for (see Attendance); AttendanceError is raised if it can't be planned.
    Each segment of constant roster is scheduled by scheduler_cls - the
    genetic scheduler's constructor and generate_schedule - with
    min(courts, players present // 4) courts, the saved constraints that
    name two players of that roster (constraints is {"do_not_pair": [...],
    "do_not_oppose": [...]}), and what the earlier segments did (see the
    module docstring). max_time is shared out by segment_budgets and the
    seeds are derived from seed (see segment_seed), so with a fake clock the
    whole plan reproduces. generations, one number for every segment or one
    per segment, replays a plan exactly as it does a single run (see
    GeneticPickleballScheduler.generate_schedule). progress_callback gets the
    scheduler's progress dicts with the elapsed time and budget of the whole
    plan, and "segment" and "segments" added. clock replaces time.time in
    every run.

    Returns a dict shaped like generate_schedule's, with:
      success: every segment's own success (a perfect schedule for it);
      schedule: rounds 1..num_rounds, spliced as a mid-session replan does;
      roster_by_round: who was present in each round;
      constraint_violations: summed; round_search_budget_reached: any;
      seed, generations (summed), time_seconds (summed), replayable (all);
      segments: per segment first_round, last_round, player_count,
        courts_used, generations, ranges (games_range there counts the
        sit-outs of earlier segments too), seed, success, time_seconds,
        budget_seconds, constraint_violations, round_search_budget_reached;
      warnings: for example segments that run fewer courts than asked.
    With a single segment (everyone there for every round) the result is
    that of one scheduler run with the same seed, plus these keys.
    If a segment finds no schedule at all, the result is unsuccessful with
    an empty schedule and an "error".
    """
    if int(courts) < 1:
        raise ValueError("courts must be at least 1")
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    roster_by_round = rosters_by_round(players, num_rounds, attendance)
    segments = segment_rosters(roster_by_round)
    budgets = segment_budgets(max_time, segments)
    if generations is None or isinstance(generations, int):
        replay_generations: list[int | None] = [generations] * len(segments)
    else:
        replay_generations = list(generations)
        if len(replay_generations) != len(segments):
            raise ValueError(f"{len(replay_generations)} generation counts given for {len(segments)} segments")
    master_seed = int(seed) if seed is not None else int(time.time() * 1000) % 1000000
    saved = constraints or {}

    warnings: list[str] = []
    schedule: list[Any] = []
    segment_results: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        roster = list(segment.players)
        courts_used = min(int(courts), len(roster) // PLAYERS_PER_COURT)
        if courts_used < courts:
            warnings.append(
                f"Round(s) {_round_list(range(segment.first_round, segment.last_round + 1))}: only {len(roster)} "
                f"players are present, which fill {courts_used} court(s) instead of {courts}."
            )

        scheduler = scheduler_cls(
            players=roster,
            num_courts=courts_used,
            num_rounds=segment.num_rounds,
            use_desktop_params=True,
            max_runtime=budgets[index],
        )
        pairs = [tuple(pair) for pair in constraint_pairs_for_roster(saved.get("do_not_pair", ()), roster)]
        opposes = [tuple(pair) for pair in constraint_pairs_for_roster(saved.get("do_not_oppose", ()), roster)]
        if pairs or opposes:
            scheduler.add_constraints(pair_constraints=pairs, oppose_constraints=opposes)
        if schedule:
            earlier_rosters = roster_by_round[: len(schedule)]
            scheduler.preceding_sit_outs = sit_outs_by_round(schedule, earlier_rosters)
            baseline = sit_out_baseline(schedule, earlier_rosters, roster)
            if baseline:
                scheduler.baseline_games = baseline

        run_options: dict[str, Any] = {}
        if clock is not None:
            run_options["clock"] = clock
        if len(segments) == 1:
            run_options["progress_callback"] = progress_callback
        elif progress_callback is not None:
            run_options["progress_callback"] = _plan_progress(
                progress_callback,
                elapsed_before=sum(done["time_seconds"] for done in segment_results),
                max_time=max_time,
                segment_number=index + 1,
                segment_count=len(segments),
            )
        result = scheduler.generate_schedule(
            max_time=budgets[index],
            seed=segment_seed(master_seed, index),
            generations=replay_generations[index],
            **run_options,
        )
        if not isinstance(result, dict):
            result = {}

        segment_results.append(
            {
                "first_round": segment.first_round,
                "last_round": segment.last_round,
                "player_count": len(roster),
                "courts_used": courts_used,
                "generations": int(result.get("generations") or 0),
                "ranges": {key: result.get(key) for key in RANGE_KEYS},
                "seed": segment_seed(master_seed, index),
                "success": bool(result.get("success")),
                "time_seconds": float(result.get("time_seconds") or 0.0),
                "budget_seconds": budgets[index],
                "constraint_violations": int(result.get("constraint_violations") or 0),
                "round_search_budget_reached": bool(result.get("round_search_budget_reached")),
            }
        )
        new_rounds = result.get("schedule")
        if not new_rounds or len(new_rounds) != segment.num_rounds:
            first, last = segment.first_round, segment.last_round
            return _plan_failure(
                roster_by_round,
                master_seed,
                segment_results,
                warnings,
                f"No schedule could be found for rounds {first}-{last} ({len(roster)} players present).",
            )
        results.append(result)
        schedule = splice_schedules(schedule, new_rounds)

    plan: dict[str, Any] = dict(results[0]) if len(results) == 1 else {}
    plan.update(
        success=all(segment["success"] for segment in segment_results),
        schedule=schedule,
        roster_by_round=roster_by_round,
        constraint_violations=sum(segment["constraint_violations"] for segment in segment_results),
        round_search_budget_reached=any(segment["round_search_budget_reached"] for segment in segment_results),
        seed=master_seed,
        generations=sum(segment["generations"] for segment in segment_results),
        time_seconds=sum(segment["time_seconds"] for segment in segment_results),
        replayable=all(bool(result.get("replayable", False)) for result in results),
        algorithm=results[0].get("algorithm", "Genetic Algorithm"),
        segments=segment_results,
        warnings=warnings,
    )
    return plan
