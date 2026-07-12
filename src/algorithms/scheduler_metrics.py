"""Metric and duplicate-round helpers for the genetic scheduler."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence, cast

GameTuple = tuple[str, str, str, str]
RawSchedule = list[list[GameTuple]]


def normalize_schedule(schedule: Sequence[Sequence[object]] | Sequence[dict[str, Any]]) -> RawSchedule:
    """Normalize either formatted or raw schedules into tuple-based rounds."""
    if schedule and isinstance(schedule[0], dict):
        raw_schedule: RawSchedule = []
        for round_data in cast(list[dict[str, Any]], schedule):
            round_games: list[GameTuple] = []
            games = round_data.get("games", [])
            for game in games:
                if hasattr(game, "team1") and hasattr(game, "team2"):
                    team1 = [str(player) for player in game.team1]
                    team2 = [str(player) for player in game.team2]
                    if len(team1) >= 2 and len(team2) >= 2:
                        round_games.append((team1[0], team1[1], team2[0], team2[1]))
            raw_schedule.append(round_games)
        return raw_schedule

    return cast(RawSchedule, schedule)


def round_signature(round_games: Sequence[Sequence[object]]) -> tuple[tuple[tuple[str, str], tuple[str, str]], ...]:
    """Build a canonical signature for a round regardless of game/team ordering."""
    normalized_games: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for game in round_games:
        p1, p2, p3, p4 = map(str, game)
        team1 = tuple(sorted([p1, p2]))
        team2 = tuple(sorted([p3, p4]))
        normalized_games.append(cast(tuple[tuple[str, str], tuple[str, str]], tuple(sorted([team1, team2]))))
    return tuple(sorted(normalized_games))


def _metric_range(values: Iterable[int]) -> int:
    data = list(values)
    if not data:
        return 0
    return max(data) - min(data)


def count_duplicate_rounds(schedule: Sequence[Sequence[Sequence[object]]]) -> int:
    """Count repeated round patterns in a schedule."""
    signatures = [round_signature(round_games) for round_games in schedule]
    return len(signatures) - len(set(signatures))


def count_avoidable_duplicate_rounds(
    schedule: Sequence[Sequence[Sequence[object]]],
    minimum_duplicate_rounds: int,
) -> int:
    """Count duplicate rounds beyond the mathematically unavoidable minimum."""
    duplicate_rounds = count_duplicate_rounds(schedule)
    return max(0, duplicate_rounds - minimum_duplicate_rounds)


def avoidable_duplicate_rounds_from_signature(
    signature: Sequence[object],
    minimum_duplicate_rounds: int,
) -> int:
    """Same arithmetic as count_avoidable_duplicate_rounds, but for a
    signature sequence already computed per round (e.g. via a cache), rather
    than a raw schedule that still needs signatures computed from scratch."""
    duplicate_rounds = len(signature) - len(set(signature))
    return max(0, duplicate_rounds - minimum_duplicate_rounds)


def precompute_arrangement_stats(
    arrangements: Sequence[Sequence[GameTuple]],
    *,
    num_courts: int,
    do_not_pair_map: Mapping[str, set[str]],
    do_not_oppose_map: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Summarize each pool arrangement once so per-individual evaluation is a merge.

    Every term in evaluate_schedule_metrics (including all violation terms) is
    computed within a single round, so a round's contribution depends only on
    the arrangement itself - never on the other rounds of the schedule.
    """
    stats: list[dict[str, Any]] = []
    for round_games in arrangements:
        partner_of: dict[str, str] = {}
        opponents_of: dict[str, tuple[str, str]] = {}
        court_of: dict[str, int] = {}
        violations = 0

        if len(round_games) != num_courts:
            violations += abs(num_courts - len(round_games))

        round_players: set[str] = set()
        for court_index, game in enumerate(round_games, start=1):
            p1, p2, p3, p4 = map(str, game)
            game_players = [p1, p2, p3, p4]

            if len(set(game_players)) != 4:
                violations += 1000

            game_player_set = set(game_players)
            overlap = game_player_set & round_players
            if overlap:
                violations += 1000 * len(overlap)
            round_players.update(game_player_set)

            if p2 in do_not_pair_map[p1] or p1 in do_not_pair_map[p2]:
                violations += 1
            if p4 in do_not_pair_map[p3] or p3 in do_not_pair_map[p4]:
                violations += 1
            for a, b in ((p1, p3), (p1, p4), (p2, p3), (p2, p4)):
                if b in do_not_oppose_map[a] or a in do_not_oppose_map[b]:
                    violations += 1

            partner_of[p1] = p2
            partner_of[p2] = p1
            partner_of[p3] = p4
            partner_of[p4] = p3
            opponents_of[p1] = (p3, p4)
            opponents_of[p2] = (p3, p4)
            opponents_of[p3] = (p1, p2)
            opponents_of[p4] = (p1, p2)
            for player in game_players:
                court_of[player] = court_index

        stats.append(
            {
                "partner_of": partner_of,
                "opponents_of": opponents_of,
                "court_of": court_of,
                "violations": violations,
            }
        )
    return stats


def evaluate_metrics_from_arrangement_stats(
    round_stats: Sequence[Mapping[str, Any]],
    player_names: Sequence[str],
) -> dict[str, int]:
    """Combine precomputed per-arrangement summaries into schedule metrics.

    Produces results identical to evaluate_schedule_metrics for any schedule
    decoded from the arrangement pool (locked in by the maintained equivalence
    test in tests/unit/test_scheduler_metrics.py).
    """
    games: dict[str, int] = {player: 0 for player in player_names}
    partners: dict[str, set[str]] = {player: set() for player in player_names}
    opponents: dict[str, set[str]] = {player: set() for player in player_names}
    courts: dict[str, set[int]] = {player: set() for player in player_names}
    violations = 0

    for stat in round_stats:
        violations += stat["violations"]
        opponents_of = stat["opponents_of"]
        court_of = stat["court_of"]
        for player, partner in stat["partner_of"].items():
            games[player] += 1
            partners[player].add(partner)
            opponents[player].update(opponents_of[player])
            courts[player].add(court_of[player])

    return {
        "games_range": _metric_range(games[player] for player in player_names),
        "partners_range": _metric_range(len(partners[player]) for player in player_names),
        "opponents_range": _metric_range(len(opponents[player]) for player in player_names),
        "courts_range": _metric_range(len(courts[player]) for player in player_names),
        "violations": violations,
    }


def evaluate_schedule_metrics(
    schedule: Sequence[Sequence[object]] | Sequence[dict[str, Any]],
    *,
    num_courts: int,
    player_names: Sequence[str],
    do_not_pair_map: Mapping[str, set[str]],
    do_not_oppose_map: Mapping[str, set[str]],
) -> dict[str, int]:
    """Evaluate fairness metrics from raw or formatted schedules."""
    raw_schedule = normalize_schedule(schedule)

    games_played: Counter[str] = Counter()
    partners: dict[str, Counter[str]] = defaultdict(Counter)
    opponents: dict[str, Counter[str]] = defaultdict(Counter)
    courts: dict[str, Counter[int]] = defaultdict(Counter)
    violations = 0

    for round_games in raw_schedule:
        if len(round_games) != num_courts:
            violations += abs(num_courts - len(round_games))

        round_players: set[str] = set()

        for court_index, game in enumerate(round_games, start=1):
            p1, p2, p3, p4 = map(str, game)
            game_players = [p1, p2, p3, p4]

            if len(set(game_players)) != 4:
                violations += 1000

            game_player_set = set(game_players)
            overlap = game_player_set & round_players
            if overlap:
                violations += 1000 * len(overlap)

            round_players.update(game_player_set)

            if p2 in do_not_pair_map[p1] or p1 in do_not_pair_map[p2]:
                violations += 1
            if p4 in do_not_pair_map[p3] or p3 in do_not_pair_map[p4]:
                violations += 1

            for a, b in ((p1, p3), (p1, p4), (p2, p3), (p2, p4)):
                if b in do_not_oppose_map[a] or a in do_not_oppose_map[b]:
                    violations += 1

            for player in (p1, p2, p3, p4):
                games_played[player] += 1
                courts[player][court_index] += 1

            for a, b in ((p1, p2), (p3, p4)):
                partners[a][b] += 1
                partners[b][a] += 1

            for a in (p1, p2):
                for b in (p3, p4):
                    opponents[a][b] += 1
                    opponents[b][a] += 1

    return {
        "games_range": _metric_range(games_played[player] for player in player_names),
        "partners_range": _metric_range(len(partners[player]) for player in player_names),
        "opponents_range": _metric_range(len(opponents[player]) for player in player_names),
        "courts_range": _metric_range(
            sum(1 for count in courts[player].values() if count > 0) for player in player_names
        ),
        "violations": violations,
    }
