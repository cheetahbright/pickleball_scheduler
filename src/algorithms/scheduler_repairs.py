"""Repair and partner-analysis helpers for the genetic scheduler."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from typing import Callable, Sequence, cast

try:
    from src.algorithms.scheduler_metrics import GameTuple, round_signature
    from src.algorithms.scheduler_reporting import sum_range_metrics
except ImportError:
    from algorithms.scheduler_metrics import GameTuple, round_signature
    from algorithms.scheduler_reporting import sum_range_metrics

ScheduleMetricsEvaluator = Callable[[list[list[GameTuple]]], dict[str, int]]
RepairPrinter = Callable[[str], None]
# Per player: how often they partnered/opposed each other player, and played each court.
PairingTallies = tuple[defaultdict[str, dict[object, int]], ...]


def games_range_floor(schedule: Sequence[Sequence[GameTuple]], num_players: int) -> int:
    """The smallest games range possible: 0 if the game slots divide evenly among the players, else 1."""
    slots = sum(len(round_games) * 4 for round_games in schedule)
    return 0 if num_players and slots % num_players == 0 else 1


def _tally_game(tallies: PairingTallies, game: Sequence[object], court: int, step: int) -> None:
    """Add (step 1) or remove (step -1) one game's partners, opponents and
    court, tallied the way evaluate_schedule_metrics tallies them."""
    partners, opponents, courts = tallies
    p1, p2, p3, p4 = map(str, game)
    pairings: list[tuple[defaultdict[str, dict[object, int]], str, object]] = [
        (partners, a, b) for a, b in ((p1, p2), (p2, p1), (p3, p4), (p4, p3))
    ]
    pairings += [(opponents, a, b) for x in (p1, p2) for y in (p3, p4) for a, b in ((x, y), (y, x))]
    pairings += [(courts, player, court) for player in (p1, p2, p3, p4)]
    for table, player, other in pairings:
        seen = table[player]
        count = seen.get(other, 0) + step
        if count:
            seen[other] = count
        else:
            del seen[other]


def repair_games_balance(
    schedule: list[list[GameTuple]],
    players: Sequence[str],
    max_time: float,
    *,
    evaluate_metrics: ScheduleMetricsEvaluator,
    printer: RepairPrinter,
    clock: Callable[[], float] | None = None,
) -> list[list[GameTuple]]:
    """Move games from the most- to the least-played players until the games range hits its floor.

    The partner and opponent repairs only swap players between games, which
    never changes how many games anyone plays, so nothing else can fix a
    games imbalance once the search has settled on one. Each move puts a
    least-played player who sits out a round into the slot of a most-played
    player in that round - so every round stays valid - choosing, among all
    such moves, the one that leaves the fewest constraint violations and the
    smallest total range. Moves that add a violation or a duplicate round
    are never taken.

    All the candidate moves leave the same games range, so they are ranked
    on running partner/opponent/court tallies rather than a full
    re-evaluation of the schedule per candidate: large rosters need a dozen
    or more moves, and full re-evaluations ran out of time after a few.
    Violations are counted per round - every violation term is within one
    round - and only for the moves actually considered.
    """
    now = clock if clock is not None else time.time
    start_time = now()
    current = [list(round_games) for round_games in schedule]
    floor = games_range_floor(current, len(players))
    counts = Counter({str(player): 0 for player in players})
    counts.update(str(player) for round_games in current for game in round_games for player in game)
    if max(counts.values()) - min(counts.values()) <= floor:
        return current

    violations = evaluate_metrics(current)["violations"]
    round_violations: dict[tuple[GameTuple, ...], int] = {}

    def violations_in(round_games: list[GameTuple]) -> int:
        key = tuple(round_games)
        if key not in round_violations:
            round_violations[key] = evaluate_metrics([list(round_games)])["violations"]
        return round_violations[key]

    signatures = Counter(round_signature(round_games) for round_games in current)
    duplicates = len(current) - len(signatures)
    duplicates_now = duplicates
    tallies: PairingTallies = (defaultdict(dict), defaultdict(dict), defaultdict(dict))
    for round_games in current:
        for court, game in enumerate(round_games, start=1):
            _tally_game(tallies, game, court, 1)
    sizes = [{str(player): len(table[str(player)]) for player in players} for table in tallies]

    def retally(old_game: GameTuple, new_game: GameTuple, court: int) -> None:
        _tally_game(tallies, old_game, court, -1)
        _tally_game(tallies, new_game, court, 1)
        touched = [player for player in {*map(str, old_game), *map(str, new_game)} if player in sizes[0]]
        for table, size in zip(tallies, sizes):
            for player in touched:
                size[player] = len(table[player])

    def spread_after(old_game: GameTuple, new_game: GameTuple, court: int) -> int:
        """The partner + opponent + court ranges with new_game in old_game's place."""
        retally(old_game, new_game, court)
        spread = sum(max(size.values(), default=0) - min(size.values(), default=0) for size in sizes)
        retally(new_game, old_game, court)
        return spread

    printer(f"   🔧 Games-balance repair starting: games range {max(counts.values()) - min(counts.values())}")
    moves = 0
    while max(counts.values()) - min(counts.values()) > floor and now() - start_time < max_time:
        most, least = max(counts.values()), min(counts.values())
        over = {player for player, count in counts.items() if count == most}
        under = [player for player, count in counts.items() if count == least]
        candidates: list[tuple[tuple[int, int], int, int, GameTuple, str, str]] = []
        for r_idx, round_games in enumerate(current):
            playing = {str(player) for game in round_games for player in game}
            sitters = [player for player in under if player not in playing]
            if not sitters:
                continue
            round_violations_now = violations_in(round_games)
            for g_idx, game in enumerate(round_games):
                for pos, player in enumerate(game):
                    if str(player) not in over:
                        continue
                    for sitter in sitters:
                        new_game = list(game)
                        new_game[pos] = sitter
                        candidate_game = cast(GameTuple, tuple(new_game))
                        # Only a round that already breaks a constraint can lose
                        # a violation; anywhere else a move adds one or none,
                        # which is checked below for the moves actually reached.
                        added = 0
                        if round_violations_now:
                            candidate_round = round_games[:g_idx] + [candidate_game] + round_games[g_idx + 1 :]
                            added = violations_in(candidate_round) - round_violations_now
                            if added > 0:
                                continue
                        key = (violations + added, spread_after(game, candidate_game, g_idx + 1))
                        candidates.append((key, r_idx, g_idx, candidate_game, str(player), sitter))

        # A stable sort, so among equally good moves the first one found wins.
        best = None
        for _, r_idx, g_idx, candidate_game, moved_from, moved_to in sorted(candidates, key=lambda c: c[0]):
            round_games = current[r_idx]
            candidate_round = round_games[:g_idx] + [candidate_game] + round_games[g_idx + 1 :]
            if violations_in(candidate_round) > violations_in(round_games):
                continue
            old_signature, new_signature = round_signature(round_games), round_signature(candidate_round)
            # +1 if the new round repeats one in the schedule, -1 if the old round was itself a repeat.
            added_duplicates = (1 if signatures[new_signature] else 0) - (1 if signatures[old_signature] > 1 else 0)
            if duplicates_now + added_duplicates > duplicates:
                continue
            best = (r_idx, g_idx, candidate_round, old_signature, new_signature, added_duplicates, moved_from, moved_to)
            break
        if best is None:
            break
        r_idx, g_idx, candidate_round, old_signature, new_signature, added_duplicates, moved_from, moved_to = best
        violations += violations_in(candidate_round) - violations_in(current[r_idx])
        signatures[old_signature] -= 1
        signatures[new_signature] += 1
        duplicates_now += added_duplicates
        retally(current[r_idx][g_idx], candidate_round[g_idx], g_idx + 1)
        current[r_idx] = candidate_round
        counts[moved_from] -= 1
        counts[moved_to] += 1
        moves += 1

    printer(
        f"   🏁 Games-balance repair complete: {moves} move(s), games range "
        f"{max(counts.values()) - min(counts.values())} (floor {floor})"
    )
    return current


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
    # Moves that keep every range where it was: still taken, so the search
    # can walk across a plateau, but not counted or logged as improvements.
    neutral_moves = 0
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
        is_improvement = test_metrics["partners_range"] < current_metrics["partners_range"] or (
            test_metrics["partners_range"] == current_metrics["partners_range"]
            and test_metrics["opponents_range"] <= current_metrics["opponents_range"]
            and test_metrics["games_range"] <= current_metrics["games_range"]
            and test_metrics["courts_range"] <= current_metrics["courts_range"]
        )

        if is_improvement and test_metrics["violations"] == 0:
            lowers_a_range = any(
                test_metrics[key] < current_metrics[key]
                for key in ("partners_range", "opponents_range", "games_range", "courts_range")
            )
            best_schedule = test_schedule
            current_metrics = test_metrics
            if not lowers_a_range:
                neutral_moves += 1
                continue
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

    final_metrics = current_metrics
    final_total = sum_range_metrics(final_metrics)
    printer(
        f"   🏁 Partner repair complete: {attempts} attempts, {improvements} improvements, "
        f"{neutral_moves} neutral moves"
    )
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
    rng: random.Random | None = None,
    clock: Callable[[], float] | None = None,
) -> list[list[GameTuple]]:
    """Target opponent-balance repairs with the same lightweight round-to-round
    swaps as repair_partner_imbalance, but scored on opponents_range.

    A move is only accepted when it strictly improves opponents_range without
    regressing games_range, partners_range, or courts_range - so this cannot
    undo work repair_partner_imbalance already did (it normally runs after
    partner repair, once partners_range is already at its achievable minimum)."""
    rand = rng if rng is not None else random
    now = clock if clock is not None else time.time
    start_time = now()
    best_schedule = [round_games[:] for round_games in schedule]
    current_metrics = evaluate_metrics(best_schedule)

    if current_metrics["opponents_range"] == 0:
        return best_schedule

    printer(f"   🔧 Opponent repair starting: range {current_metrics['opponents_range']}")

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
        is_improvement = test_metrics["opponents_range"] < current_metrics["opponents_range"] and (
            test_metrics["games_range"] <= current_metrics["games_range"]
            and test_metrics["partners_range"] <= current_metrics["partners_range"]
            and test_metrics["courts_range"] <= current_metrics["courts_range"]
        )

        if is_improvement and test_metrics["violations"] == 0:
            best_schedule = test_schedule
            current_metrics = test_metrics
            improvements += 1
            printer(
                f"   🎯 Improvement #{improvements}: "
                f"O:{test_metrics['opponents_range']} "
                f"P:{test_metrics['partners_range']} "
                f"(attempt {attempts})"
            )
            if test_metrics["opponents_range"] == 0:
                printer(f"   🎉 Opponent balance achieved after {attempts} attempts!")
                break

    final_metrics = current_metrics
    final_total = sum_range_metrics(final_metrics)
    printer(f"   🏁 Opponent repair complete: {attempts} attempts, {improvements} improvements")
    printer(f"      Opponents Range: {final_metrics['opponents_range']} | Total Range: {final_total}")
    printer(
        f"      Breakdown - G:{final_metrics['games_range']} "
        f"P:{final_metrics['partners_range']} "
        f"O:{final_metrics['opponents_range']} "
        f"C:{final_metrics['courts_range']}"
    )

    return best_schedule


def apply_targeted_repairs(
    schedule: list[list[GameTuple]],
    max_time: float,
    *,
    evaluate_metrics: ScheduleMetricsEvaluator,
    printer: RepairPrinter,
    partner_repair: Callable[[list[list[GameTuple]], float], list[list[GameTuple]]],
    opponent_repair: Callable[[list[list[GameTuple]], float], list[list[GameTuple]]],
    clock: Callable[[], float] | None = None,
    games_repair: Callable[[list[list[GameTuple]], float], list[list[GameTuple]]] | None = None,
    num_players: int | None = None,
    games_max_time: float | None = None,
) -> list[list[GameTuple]] | None:
    """Apply targeted local repairs and report the resulting fairness ranges.

    Games balance goes first (it is the top-priority objective, and the
    partner/opponent swaps that follow preserve game counts). It gets
    games_max_time (default: half of max_time), and the swaps' max_time
    starts once it is done, so a long games repair cannot starve them.
    """
    now = clock if clock is not None else time.time
    start_time = now()
    current_schedule = [round_games[:] for round_games in schedule]
    current_metrics = evaluate_metrics(current_schedule)
    starting_total = sum_range_metrics(current_metrics)

    printer(
        "   Starting repair: "
        f"G:{current_metrics['games_range']} "
        f"P:{current_metrics['partners_range']} "
        f"O:{current_metrics['opponents_range']} "
        f"C:{current_metrics['courts_range']}"
    )

    if (
        games_repair is not None
        and num_players
        and current_metrics["games_range"] > games_range_floor(current_schedule, num_players)
    ):
        current_schedule = games_repair(current_schedule, max_time * 0.5 if games_max_time is None else games_max_time)
        current_metrics = evaluate_metrics(current_schedule)
        start_time = now()

    if current_metrics["partners_range"] > 0 and now() - start_time < max_time:
        current_schedule = partner_repair(current_schedule, max(0.0, max_time * 0.8 - (now() - start_time)))
        current_metrics = evaluate_metrics(current_schedule)

    if current_metrics["opponents_range"] > 0 and now() - start_time < max_time:
        remaining_time = max_time - (now() - start_time)
        current_schedule = opponent_repair(current_schedule, remaining_time)
        current_metrics = evaluate_metrics(current_schedule)

    final_total = sum_range_metrics(current_metrics)
    if final_total == 0:
        printer("   🎉 REPAIR SUCCESS: Achieved Range 0!")
        return current_schedule

    if final_total < starting_total:
        printer(f"   ⚠️ REPAIR PARTIAL: Reduced to total range {final_total}")
    else:
        printer(f"   ⚠️ REPAIR PARTIAL: No improvement - still total range {final_total}")
    printer(
        f"      Breakdown - G:{current_metrics['games_range']} "
        f"P:{current_metrics['partners_range']} "
        f"O:{current_metrics['opponents_range']} "
        f"C:{current_metrics['courts_range']}"
    )
    return current_schedule


def sitout_streak_key(sitting_out: Sequence[int]) -> tuple[int, int]:
    """Score how rounds in this order bunch up sit-outs; lower is better.

    sitting_out holds, per round, a bit mask of the players who sit it out.
    Returns (most rounds in a row any one player sits out, how many times a
    player sits out two rounds in a row), compared in that order.
    """
    longest = back_to_back = 0
    # runs[k]: the players who sat out each of the last k + 1 rounds; each
    # entry is a subset of the one before, so the list ends at the first empty one.
    runs: list[int] = []
    for mask in sitting_out:
        runs = [mask] + [run & mask for run in runs]
        while runs and not runs[-1]:
            runs.pop()
        longest = max(longest, len(runs))
        if len(runs) > 1:
            back_to_back += bin(runs[1]).count("1")
    return longest, back_to_back


def _exact_round_order(sitting_out: Sequence[int], bound: tuple[int, int], node_limit: int) -> list[int] | None:
    """Depth-first branch and bound over round orders: the first order found
    whose sitout_streak_key beats bound, then better ones, until the search
    is exhausted or has visited node_limit partial orders. Returns the best
    order found, or None if none beat bound."""
    best: list = [bound, None]
    nodes = 0

    def extend(order: list[int], runs: list[int], longest: int, back_to_back: int) -> None:
        nonlocal nodes
        nodes += 1
        # Both parts of the key only grow as rounds are added.
        if nodes > node_limit or (longest, back_to_back) >= best[0]:
            return
        if len(order) == len(sitting_out):
            best[0], best[1] = (longest, back_to_back), order[:]
            return
        tried: set[int] = set()
        for index, mask in enumerate(sitting_out):
            if index in order or mask in tried:
                continue
            tried.add(mask)  # rounds with the same sitters are interchangeable
            next_runs = [mask] + [run & mask for run in runs]
            while next_runs and not next_runs[-1]:
                next_runs.pop()
            order.append(index)
            extend(
                order,
                next_runs,
                max(longest, len(next_runs)),
                back_to_back + (bin(next_runs[1]).count("1") if len(next_runs) > 1 else 0),
            )
            order.pop()

    extend([], [], 0, 0)
    return best[1]


def order_rounds_for_rest(
    schedule: list[list[GameTuple]], players: Sequence[str], node_limit: int = 100_000
) -> list[list[GameTuple]]:
    """Reorder whole rounds so nobody sits out more rounds in a row than needed.

    Every fitness term, the partner-repetition tiebreak, constraint
    violations and duplicate rounds all ignore round order, and each round
    keeps its games and courts, so this changes none of them: it is the
    last tiebreak, applied once the schedule is chosen. Orders are ranked by
    sitout_streak_key. Starting from the order given, a local search moves
    one round to another position or swaps two rounds while that strictly
    improves the key; then, if someone still sits out twice in a row, a
    branch and bound capped at node_limit partial orders looks for a strictly
    better order. Both are deterministic, and ties keep the earlier order.
    """
    bits = {str(player): 1 << index for index, player in enumerate(players)}
    everyone = (1 << len(bits)) - 1
    sitting_out: list[int] = []
    for round_games in schedule:
        playing = 0
        for game in round_games:
            for player in game:
                playing |= bits.get(str(player), 0)
        sitting_out.append(everyone & ~playing)

    order = list(range(len(schedule)))
    best = sitout_streak_key(sitting_out)
    # No one sitting out two rounds in a row means no streak can get shorter.
    improved = best[1] > 0
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(len(order)):
                if i == j:
                    continue
                moved = order[:]
                moved.insert(j, moved.pop(i))
                candidates = [moved]
                if i < j:
                    swapped = order[:]
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    candidates.append(swapped)
                for candidate in candidates:
                    key = sitout_streak_key([sitting_out[index] for index in candidate])
                    if key < best:
                        order, best, improved = candidate, key, True
    if best[1] > 0:
        order = _exact_round_order(sitting_out, best, node_limit) or order
    return [list(schedule[index]) for index in order]
