"""Constructive (non-search) schedules for exactly 4 * num_courts players,
built from fixed, verified round designs in microseconds instead of relying on
the GA to discover that balance by search - which, at 12 and more players, the
GA's 30-second budget does not reliably do.

constructive_designs.DESIGNS has one design for every (num_courts,
num_rounds) with 2-5 courts (8-20 players) and 1-20 rounds - every round count
the app offers. Each design:

- puts every player in every round, and gives every player the same number of
  distinct partners, opponents and courts (games, partners, opponents and
  courts range all 0);
- never repeats a round;
- spreads partners as widely as possible: every player partners
  min(num_rounds, 4m - 1) different people (m = num_courts), and no pair
  partners more than ceil(num_rounds / (4m - 1)) times;
- does the same for opponents - min(2 * num_rounds, 4m - 1) distinct each, no
  pair meeting more than ceil(2 * num_rounds / (4m - 1)) times - except at 8
  players for 2-4 rounds and 12 players for 5 rounds, where exhaustive
  searches proved that ideal cannot coexist with range 0; there every player
  gets one fewer distinct opponent (and some pair meets twice), the best
  possible.

Range 0 on its own is a weak guarantee: the offset-table designs these replace
met it while giving every player only num_courts distinct partners (3
different partners in 8 rounds at 12 players), and padded longer schedules by
replaying rounds.

The designs for up to 16 rounds came from CP-SAT models and subsets of whist
tournaments (round sets in which every pair partners once and opposes twice).
For 17-20 rounds, 2-4 courts append a shorter design to a full whist
tournament; 5 courts use subsets and an extension of a cyclic 20-player whist
tournament. tests/unit/test_constructive_scheduler.py re-verifies every entry
with the scheduler's own metrics.

Deliberately does not cover num_courts=1 (4 players): the GA already solves
that instantly, and the arrangement pool there also has 3 distinct
team-pairings the existing duplicate-round bookkeeping depends on.
"""

from __future__ import annotations

from typing import Sequence

try:
    from src.algorithms.constructive_designs import DESIGNS
except ImportError:
    from algorithms.constructive_designs import DESIGNS

GameTuple = tuple[str, str, str, str]
Round = list[GameTuple]
Schedule = list[Round]


def _decode(num_courts: int, num_rounds: int) -> list[list[tuple[int, ...]]] | None:
    """The design for (num_courts, num_rounds) as rounds of player-index games, or None.

    Also None for a malformed entry - it must never produce a broken schedule
    advertised as perfect; the caller falls back to the GA instead.
    """
    encoded = DESIGNS.get((num_courts, num_rounds))
    if encoded is None:
        return None
    try:
        rounds = [
            [tuple(int(player, 36) for player in game) for game in round_text.split(".")]
            for round_text in encoded.split()
        ]
    except ValueError:
        return None
    every_player = list(range(4 * num_courts))
    well_formed = len(rounds) == num_rounds and all(
        len(round_games) == num_courts
        and all(len(game) == 4 for game in round_games)
        and sorted(player for game in round_games for player in game) == every_player
        for round_games in rounds
    )
    return rounds if well_formed else None


def build_perfect_schedule(names: Sequence[str], num_courts: int, num_rounds: int) -> Schedule | None:
    """Return the verified design (see the module docstring) for
    len(names) == 4 * num_courts players, or None if there is none for this
    (num_courts, num_rounds).

    names[i] plays as the design's player i, so callers choose who lands where
    through the order of `names`. Each round lists its games in court order.
    """
    if len(names) != 4 * num_courts:
        return None
    design = _decode(num_courts, num_rounds)
    if design is None:
        return None
    return [[(names[a], names[b], names[c], names[d]) for a, b, c, d in round_games] for round_games in design]
