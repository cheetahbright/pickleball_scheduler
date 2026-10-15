"""Pure helpers for a schedule whose roster changes from round to round.

A mid-session replan and the attendance planner (src/algorithms/attendance_planner.py)
both stitch rounds planned for different rosters into one schedule. These
helpers hold what they share, apart from the Streamlit app that first needed
them, so the planner can use them without importing it.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    from src.utils.schedule_shapes import extract_game_teams, games_in_round
except ImportError:
    from utils.schedule_shapes import extract_game_teams, games_in_round


def pair_in_roster(pair: Sequence[str], roster: set[str]) -> bool:
    """Whether ``pair`` is two players who are both in ``roster``."""
    return len(pair) == 2 and pair[0] in roster and pair[1] in roster


def constraint_pairs_for_roster(constraint_pairs: Sequence[Sequence[str]], players: Sequence[str]) -> list[list[str]]:
    """Return only the saved pairs whose two players are both on ``players``.

    Saved constraints outlive any one roster (a regular who is away this
    week keeps theirs), so everything that acts on a specific roster - the
    scheduler, the constraint counts and warnings, the settings recorded in
    history - must use this filtered view rather than the full saved list.
    """
    roster = set(players)
    return [list(pair) for pair in constraint_pairs if pair_in_roster(pair, roster)]


def sit_outs_by_round(rounds: Sequence[Any], roster_by_round: Sequence[Sequence[str]]) -> list[set[str]]:
    """Who sat out each round: its roster minus everyone in one of its games.

    rounds and roster_by_round run in step (a mid-session replan's locked
    rounds and their rosters), so a player who left before a round is not
    counted as sitting it out, and one who joined later is not either.
    """
    sit_outs: list[set[str]] = []
    for round_data, roster in zip(rounds, roster_by_round):
        on_court: set[str] = set()
        for game in games_in_round(round_data):
            team1, team2, _court = extract_game_teams(game)
            on_court.update(str(player) for player in team1)
            on_court.update(str(player) for player in team2)
        sit_outs.append({str(player) for player in roster} - on_court)
    return sit_outs


def splice_schedules(locked_rounds: list, new_rounds: list) -> list:
    """Combine already-played rounds with freshly generated remaining rounds.

    New rounds are renumbered to continue directly after the locked ones, so
    a replan of rounds 4-6 (after 3 played) ends up numbered 4, 5, 6 - not
    restarting at 1.
    """
    spliced = list(locked_rounds)
    start_round = len(locked_rounds) + 1
    for i, round_data in enumerate(new_rounds):
        if hasattr(round_data, "get"):
            renumbered = dict(round_data)
            renumbered["round"] = start_round + i
        else:
            renumbered = round_data
        spliced.append(renumbered)
    return spliced
