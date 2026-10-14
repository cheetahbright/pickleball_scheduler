"""Pure schedule analytics helpers shared by the Streamlit app and tests."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Any

try:
    from src.utils.schedule_shapes import extract_game_teams
except ImportError:
    from utils.schedule_shapes import extract_game_teams


def serialize_schedule_for_json(schedule):
    """Convert schedule rounds into a JSON-serializable round/game structure.

    Takes a list of round dicts (``{"round": n, "games": [...]}``, games as
    Game objects or dicts) or a scheduler result holding one under "schedule".
    """
    if isinstance(schedule, dict) and "schedule" in schedule:
        schedule_list = schedule["schedule"]
    else:
        schedule_list = schedule

    serializable_schedule: list[dict[str, Any]] = []

    for round_data in schedule_list:
        round_games = []
        for game in round_data.get("games", []):
            if hasattr(game, "team1"):
                team1, team2, court = extract_game_teams(game)
                game_dict = {
                    "team1": [str(player) for player in team1],
                    "team2": [str(player) for player in team2],
                    "court": court,
                }
            else:
                game_dict = game
            round_games.append(game_dict)

        # Numbered by position, as the screen, the CSV and score entry
        # number rounds: an older history row can hold gaps in its stored
        # numbers, and a replan of one would repeat a number.
        serializable_schedule.append(
            {
                "round": len(serializable_schedule) + 1,
                "games": round_games,
            }
        )

    return serializable_schedule


def compute_team_skill_balance(schedule: list[dict], skills: dict[str, int]) -> list[dict[str, Any]]:
    """Per-game team skill totals and imbalance, for players with a rated skill.

    A team's total sums only its rated players, which counts an unrated one
    as 0, so a game with an unrated player has an imbalance of None: the gap
    between partial totals would measure the missing rating, not the teams.
    A game where every player on both sides is unrated is skipped entirely -
    there is nothing to report. Does not feed into schedule generation; this
    is a post-hoc analytics view only (see SkillRatingManager docstring).
    """
    rows: list[dict[str, Any]] = []

    for round_num, round_data in enumerate(schedule, 1):
        for game in round_data.get("games", []):
            raw_team1, raw_team2, court = extract_game_teams(game)
            team1 = [str(p) for p in raw_team1]
            team2 = [str(p) for p in raw_team2]

            team1_ratings = [skills[p] for p in team1 if p in skills]
            team2_ratings = [skills[p] for p in team2 if p in skills]
            if not team1_ratings and not team2_ratings:
                continue

            team1_total = sum(team1_ratings)
            team2_total = sum(team2_ratings)
            fully_rated = len(team1_ratings) == len(team1) and len(team2_ratings) == len(team2)
            rows.append(
                {
                    "round_num": round_num,
                    "court": court,
                    "team1": team1,
                    "team2": team2,
                    "team1_skill_total": team1_total,
                    "team2_skill_total": team2_total,
                    "imbalance": abs(team1_total - team2_total) if fully_rated else None,
                    "fully_rated": fully_rated,
                }
            )

    return rows


def build_pairing_matrices(schedule: list[dict], players: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    """Return (partner_counts, opponent_counts) as NxN matrices in `players` order.

    partner_counts[i][j] is how many times players[i] and players[j] were on
    the same team; opponent_counts[i][j] is how many times they faced each
    other. Both matrices are symmetric with a zero diagonal.
    """
    index_by_player = {str(player): i for i, player in enumerate(players)}
    size = len(players)
    partner_counts = [[0] * size for _ in range(size)]
    opponent_counts = [[0] * size for _ in range(size)]

    for round_data in schedule:
        for game in round_data.get("games", []):
            raw_team1, raw_team2, _court = extract_game_teams(game)
            team1 = [str(p) for p in raw_team1]
            team2 = [str(p) for p in raw_team2]

            for team, matrix in ((team1, partner_counts), (team2, partner_counts)):
                for a in range(len(team)):
                    for b in range(a + 1, len(team)):
                        i, j = index_by_player.get(team[a]), index_by_player.get(team[b])
                        if i is None or j is None:
                            continue
                        matrix[i][j] += 1
                        matrix[j][i] += 1

            for p1 in team1:
                for p2 in team2:
                    i, j = index_by_player.get(p1), index_by_player.get(p2)
                    if i is None or j is None:
                        continue
                    opponent_counts[i][j] += 1
                    opponent_counts[j][i] += 1

    return partner_counts, opponent_counts


def calculate_fairness_metrics(
    schedule: list[dict],
    num_players: int | None = None,
    num_rounds: int | None = None,
    num_courts: int | None = None,
    players: list[str] | None = None,
    roster_by_round: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Calculate fairness metrics for a schedule, and how they compare to what's achievable.

    Pass ``players`` (the roster) so players who never get a game still count:
    built only from the games, the stats would leave them out and report a
    perfectly even split.

    The games can only be split evenly when the game slots divide by the
    number of players, otherwise some player must play one game more
    (``games_range_floor``). ``total_range_floor`` is the proven lower bound
    on the total range (see
    ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums): it adds the
    ranges forced when there are fewer slots than players. A schedule is
    reported as mathematically optimal only when its total range reaches it.
    ``most_partnered`` is the most times any one pair of players are
    partners - the ranges count only how many different partners each player
    has, so they can't show a pair teaming up again and again.

    Pass ``roster_by_round`` (each round's roster, as a mid-session replan
    records it) to measure each player against the rounds they were on the
    roster for, so leaving early or joining late doesn't count as unfair (see
    _per_round_roster_ranges). The ranges and the floor then come from those
    rosters and each round's own court count instead of the other arguments,
    and ``total_range_floor`` is the games-range floor, the only bound
    proven when the roster or court count changes between rounds.
    """
    if not schedule:
        return {}

    if roster_by_round is not None and len(roster_by_round) != len(schedule):
        roster_by_round = None
    rostered = [player for roster in roster_by_round or [] for player in roster]
    player_stats: dict[str, dict[str, Any]] = {
        str(player): {"games_played": 0, "partners": set(), "opponents": set(), "courts_used": set()}
        for player in [*(players or []), *rostered]
    }
    times_partnered: Counter[tuple[str, str]] = Counter()

    for round_data in schedule:
        games = round_data.get("games", [])
        for game in games:
            team1, team2, court = extract_game_teams(game)

            for team in [team1, team2]:
                if len(team) == 2:
                    first, second = sorted(str(player) for player in team)
                    times_partnered[(first, second)] += 1
                for player in team:
                    player_name = str(player)
                    if player_name not in player_stats:
                        player_stats[player_name] = {
                            "games_played": 0,
                            "partners": set(),
                            "opponents": set(),
                            "courts_used": set(),
                        }

                    player_stats[player_name]["games_played"] += 1
                    player_stats[player_name]["courts_used"].add(court)

                    for partner in team:
                        if str(partner) != player_name:
                            player_stats[player_name]["partners"].add(str(partner))

                    other_team = team2 if team == team1 else team1
                    for opponent in other_team:
                        player_stats[player_name]["opponents"].add(str(opponent))

    games_played: list[int] = [stats["games_played"] for stats in player_stats.values()]
    partners_count: list[int] = [len(stats["partners"]) for stats in player_stats.values()]
    opponents_count: list[int] = [len(stats["opponents"]) for stats in player_stats.values()]
    courts_count: list[int] = [len(stats["courts_used"]) for stats in player_stats.values()]

    games_range: float = max(games_played) - min(games_played) if games_played else 0
    partners_range: float = max(partners_count) - min(partners_count) if partners_count else 0
    opponents_range: float = max(opponents_count) - min(opponents_count) if opponents_count else 0
    courts_range: float = max(courts_count) - min(courts_count) if courts_count else 0
    total_range = games_range + partners_range + opponents_range + courts_range

    games_range_floor: float | None = None
    total_range_floor: float | None = None
    if roster_by_round is not None:
        ranges, floor = _per_round_roster_ranges(schedule, player_stats, roster_by_round)
        games_range, partners_range, opponents_range, courts_range = map(_plain_number, ranges)
        total_range = _plain_number(sum(ranges, Fraction(0)))
        total_floor = floor
        rosters = {frozenset(map(str, roster)) for roster in roster_by_round}
        court_counts = {len(round_data.get("games", [])) for round_data in schedule}
        if len(rosters) == 1 and len(court_counts) == 1:
            # One roster and court count throughout: the plain proven bound holds.
            (roster,), (courts,) = rosters, court_counts
            total_floor = max(floor, Fraction(_proven_total_range_floor(len(roster), courts, len(schedule))))
        games_range_floor = _plain_number(floor)
        total_range_floor = _plain_number(total_floor)
        fairness_score = _plain_number(max(Fraction(0), 10 - (sum(ranges, Fraction(0)) - total_floor)))
    elif num_players and num_rounds and num_courts:
        games_range_floor = 0 if (num_rounds * num_courts * 4) % num_players == 0 else 1
        total_range_floor = max(games_range_floor, _proven_total_range_floor(num_players, num_courts, num_rounds))
        fairness_score = max(0, 10 - (total_range - total_range_floor))
    else:
        fairness_score = max(0, 10 - total_range)

    return {
        "games_range": games_range,
        "partners_range": partners_range,
        "opponents_range": opponents_range,
        "courts_range": courts_range,
        "total_range": total_range,
        "player_stats": player_stats,
        "total_players": len(player_stats),
        "total_rounds": len(schedule),
        "overall_fairness": fairness_score,
        "games_range_floor": games_range_floor,
        "total_range_floor": total_range_floor,
        "is_mathematically_optimal": (total_range <= total_range_floor if total_range_floor is not None else None),
        "most_partnered": max(times_partnered.values(), default=0),
    }


def _per_round_roster_ranges(
    schedule: list[dict], player_stats: dict[str, dict[str, Any]], roster_by_round: list[list[str]]
) -> tuple[list[Fraction], Fraction]:
    """The games, partners, opponents and courts ranges, and the games-range
    floor, with each player measured against the rounds they were on the roster for.

    A round with c courts and n players on its roster gives each of them an
    even 4c/n games. Over a player's rounds those add up to their share of
    games, of partners (one per game, up to the other players rostered with
    them), of opponents (two per game, up to the same limit) and of courts
    (one per game, up to the courts in use in those rounds). Each range is
    the spread of the players' counts less their shares, so a player who
    left early or joined late isn't short of anything. With one roster for
    every round all shares are equal, and the ranges are the plain ones.

    Players on no round's roster (dropped by a replan of every round) are left out.
    """
    rounds = []
    for round_data, listed in zip(schedule, roster_by_round):
        games = [extract_game_teams(game) for game in round_data.get("games", [])]
        on_court = {str(player) for team1, team2, _court in games for player in (*team1, *team2)}
        rounds.append(({str(player) for player in listed} | on_court, len(games), {court for *_teams, court in games}))

    shares: list[Fraction] = []
    differences: list[list[Fraction]] = [[], [], [], []]
    for name, stats in player_stats.items():
        share = Fraction(0)
        rostered_with: set[str] = set()
        courts_in_use: set[Any] = set()
        for roster, courts, court_labels in rounds:
            if name in roster:
                share += Fraction(4 * courts, len(roster))
                rostered_with |= roster
                courts_in_use |= court_labels
        if not rostered_with:
            continue
        others = Fraction(len(rostered_with) - 1)
        shares.append(share)
        differences[0].append(stats["games_played"] - share)
        differences[1].append(len(stats["partners"]) - min(share, others))
        differences[2].append(len(stats["opponents"]) - min(2 * share, others))
        differences[3].append(len(stats["courts_used"]) - min(share, Fraction(len(courts_in_use))))
    ranges = [max(values) - min(values) if values else Fraction(0) for values in differences]

    # The floor: every place in a game is part of someone's share, so the
    # players' games less their shares add up to 0, and each is a whole
    # number away from the fraction its share leaves. Sorted by those
    # fractions, the tightest split takes them in turn from the one position
    # whose sum is 0 - with one roster throughout, that's 0 if the share is a
    # whole number of games and 1 otherwise, as without rosters.
    fractions = sorted(-share % 1 for share in shares)
    if not fractions:
        return ranges, Fraction(0)
    start = int(-sum(fractions, Fraction(0))) % len(fractions)
    floor = fractions[-1] - fractions[0] if start == 0 else fractions[start - 1] + 1 - fractions[start]
    return ranges, floor


def _plain_number(value: Fraction) -> float:
    """A whole number as an int, anything else as a float."""
    return int(value) if value.denominator == 1 else float(value)


def _proven_total_range_floor(num_players: int, num_courts: int, num_rounds: int) -> int:
    """The feasibility analyzer's proven lower bound on the total range (0 if unavailable)."""
    try:
        try:
            from src.utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
        except ImportError:
            from utils.feasibility_analyzer import ScheduleFeasibilityAnalyzer
        minimums = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(num_players, num_courts, num_rounds)
        return int(minimums["total_theoretical_min"])
    except Exception:
        return 0
