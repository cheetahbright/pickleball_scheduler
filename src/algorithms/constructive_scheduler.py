"""Constructive (non-search) perfect schedules for specific "friendly" player
counts, verified to achieve games_range = partners_range = opponents_range =
courts_range = 0 in microseconds instead of relying on the GA to discover
that balance by search - which, at 12 and 16 players, the GA's 30-second
budget does not reliably do (it plateaus on opponents_range).

Design: split num_players = 4 * num_courts players into four groups A, B, C,
D of size m = num_courts, indexed by Z_m. Round r is defined by an offset
triple (f, g, h) in Z_m^3 and a court shift k:

    game at position x in Z_m: (A_x, B_{x+f}) vs (C_{x+g}, D_{x+h})
    placed at court (x + k) mod m

The offset triples for one verified round set are chosen so that every one of
the ten values derived from them - {f}, {h-g}, {g}, {h}, {g-f}, {h-f}, {k},
{k-f}, {k-g}, {k-h} - covers all of Z_m exactly once each. That coverage is
what forces every player's distinct partner/opponent/court count to be
identical across the group, which is exactly what "range 0" requires.

_OFFSET_TABLES is keyed by (num_courts, num_rounds); only combinations
hand-verified here are present (see tests/unit/test_constructive_scheduler.py,
which re-checks the coverage property directly for every entry). A request
for more rounds than any verified entry for that num_courts can pad by
cycling through the largest verified round set for that num_courts: replaying
an already-used round cannot increase any player's *distinct* partner,
opponent, or court count (they've already had that exact partner/opponent/
court), so range stays 0 - only the separately-tracked duplicate-round count
grows. A request for fewer rounds than every verified entry for that
num_courts returns None, since an arbitrary truncated prefix of a verified
round set is not guaranteed to preserve the coverage property.
"""

from __future__ import annotations

from typing import Sequence

GameTuple = tuple[str, str, str, str]
Round = list[GameTuple]
Schedule = list[Round]

# (num_courts, num_rounds) -> (offset triples, explicit court shifts or None to
# derive them as (f+g+h) % m). Hand-verified to cover Z_m in all ten derived
# senses (see module docstring and test_constructive_scheduler.py).
# Deliberately does not include num_courts=1 (4 players): the GA already
# solves that instantly, and the arrangement pool there also has 3 distinct
# team-pairings the existing duplicate-round bookkeeping depends on, which a
# single fixed pairing repeated every round would not exercise.
_OFFSET_TABLES: dict[tuple[int, int], tuple[list[tuple[int, int, int]], list[int] | None]] = {
    (2, 8): (
        [(f, g, h) for f in range(2) for g in range(2) for h in range(2)],
        None,
    ),
    (3, 8): (
        [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2)],
        [0, 2, 1, 0, 2, 1, 0, 2],
    ),
    (4, 8): (
        [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (0, 1, 2), (1, 3, 0), (2, 0, 3), (3, 2, 0)],
        [0, 2, 3, 1, 3, 0, 2, 1],
    ),
    (5, 8): (
        [(1, 4, 0), (2, 1, 4), (4, 1, 1), (4, 0, 4), (2, 4, 2), (3, 3, 4), (1, 3, 0), (0, 2, 3)],
        [0, 2, 1, 3, 3, 0, 4, 0],
    ),
    (3, 6): (
        [(1, 2, 0), (1, 1, 0), (2, 0, 2), (1, 1, 2), (0, 1, 1), (0, 2, 2)],
        [0, 2, 1, 1, 2, 1],
    ),
    (4, 6): (
        [(2, 2, 2), (3, 3, 3), (0, 2, 1), (1, 0, 1), (2, 1, 0), (0, 1, 3)],
        [2, 1, 3, 2, 3, 0],
    ),
}


def supported_court_counts() -> tuple[int, ...]:
    """Court counts this module has at least one verified design for."""
    return tuple(sorted({num_courts for num_courts, _num_rounds in _OFFSET_TABLES}))


def _derive_offset_sets(m: int, triples: list[tuple[int, int, int]], ks: list[int]) -> dict[str, set[int]]:
    return {
        "f": {f for f, _, _ in triples},
        "h-g": {(h - g) % m for _, g, h in triples},
        "g": {g for _, g, _ in triples},
        "h": {h for _, _, h in triples},
        "g-f": {(g - f) % m for f, g, _ in triples},
        "h-f": {(h - f) % m for f, _, h in triples},
        "k": set(ks),
        "k-f": {(k - f) % m for (f, _, _), k in zip(triples, ks)},
        "k-g": {(k - g) % m for (_, g, _), k in zip(triples, ks)},
        "k-h": {(k - h) % m for (_, _, h), k in zip(triples, ks)},
    }


def _coverage_ok(m: int, triples: list[tuple[int, int, int]], ks: list[int]) -> bool:
    full = set(range(m))
    return all(s == full for s in _derive_offset_sets(m, triples, ks).values())


def _resolve_ks(m: int, triples: list[tuple[int, int, int]], explicit_ks: list[int] | None) -> list[int]:
    return explicit_ks if explicit_ks is not None else [(f + g + h) % m for f, g, h in triples]


def _best_base_table(
    num_courts: int, num_rounds: int
) -> tuple[list[tuple[int, int, int]], list[int] | None] | None:
    """The largest verified round set for this num_courts that is no longer
    than num_rounds, so padding to num_rounds maximizes distinct rounds
    before any repeat is needed. None if no verified table for num_courts is
    short enough to pad up to num_rounds."""
    candidates = [
        (table_rounds, entry) for (m, table_rounds), entry in _OFFSET_TABLES.items() if m == num_courts
    ]
    candidates = [item for item in candidates if item[0] <= num_rounds]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_perfect_schedule(names: Sequence[str], num_courts: int, num_rounds: int) -> Schedule | None:
    """Return a games_range=partners_range=opponents_range=courts_range=0
    schedule for len(names) == 4 * num_courts players, or None if no verified
    design here can produce exactly num_rounds rounds for this num_courts.

    Group assignment is positional: names[0:m] is group A, names[m:2m] group
    B, names[2m:3m] group C, names[3m:4m] group D (m = num_courts) - callers
    control which real players land in which group via the order of `names`.
    """
    m = num_courts
    if len(names) != 4 * m:
        return None

    table_entry = _OFFSET_TABLES.get((m, num_rounds)) or _best_base_table(m, num_rounds)
    if table_entry is None:
        return None

    triples, explicit_ks = table_entry
    ks = _resolve_ks(m, triples, explicit_ks)
    if not _coverage_ok(m, triples, ks):
        # Defensive: a corrupted table entry must never silently produce an
        # imbalanced schedule advertised as "perfect". Covered by
        # test_constructive_scheduler.py so this should never trip in practice.
        return None

    base_len = len(triples)
    if base_len != num_rounds:
        # Requested more rounds than this verified round set has - pad by
        # cycling through it (see module docstring for why this preserves
        # range 0).
        triples = [triples[i % base_len] for i in range(num_rounds)]
        ks = [ks[i % base_len] for i in range(num_rounds)]

    schedule: Schedule = []
    for (f, g, h), k in zip(triples, ks):
        court_to_game: dict[int, GameTuple] = {}
        for x in range(m):
            a = names[0 * m + x]
            b = names[1 * m + (x + f) % m]
            c = names[2 * m + (x + g) % m]
            d = names[3 * m + (x + h) % m]
            court = (x + k) % m
            court_to_game[court] = (a, b, c, d)
        schedule.append([court_to_game[court] for court in range(m)])
    return schedule
