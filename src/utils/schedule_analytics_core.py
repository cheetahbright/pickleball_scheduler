"""Pure schedule analytics helpers shared by the Streamlit app and tests."""

from __future__ import annotations

from typing import Any

try:
    from src.utils.schedule_shapes import extract_game_teams
except ImportError:
    from utils.schedule_shapes import extract_game_teams


def serialize_schedule_for_json(schedule):
    """Convert schedule data into a JSON-serializable round/game structure."""
    if isinstance(schedule, dict) and "schedule" in schedule:
        schedule_list = schedule["schedule"]
    else:
        schedule_list = schedule

    serializable_schedule = []

    if schedule_list and hasattr(schedule_list[0], "team1"):
        games_per_round = 4
        for i in range(0, len(schedule_list), games_per_round):
            round_games = []
            round_num = (i // games_per_round) + 1
            for j in range(i, min(i + games_per_round, len(schedule_list))):
                game = schedule_list[j]
                round_games.append(
                    {
                        "team1": [str(player) for player in game.team1],
                        "team2": [str(player) for player in game.team2],
                        "court": game.court,
                    }
                )

            serializable_schedule.append({"round": round_num, "games": round_games})
    else:
        for round_data in schedule_list:
            if hasattr(round_data, "get"):
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

                serializable_schedule.append(
                    {
                        "round": round_data.get("round", len(serializable_schedule) + 1),
                        "games": round_games,
                    }
                )
            else:
                team1, team2, court = extract_game_teams(round_data)
                serializable_schedule.append(
                    {
                        "round": len(serializable_schedule) + 1,
                        "games": [
                            {
                                "team1": [str(player) for player in team1],
                                "team2": [str(player) for player in team2],
                                "court": court,
                            }
                        ],
                    }
                )

    return serializable_schedule


def compute_team_skill_balance(schedule: list[dict], skills: dict[str, int]) -> list[dict[str, Any]]:
    """Per-game team skill totals and imbalance, for players with a rated skill.

    Unrated players are excluded from their team's sum (not treated as 0 -
    that would make an unrated team look artificially weaker). A game where
    every player on both sides is unrated is skipped entirely - there is
    nothing to report. Does not feed into schedule generation; this is a
    post-hoc analytics view only (see SkillRatingManager docstring).
    """
    rows: list[dict[str, Any]] = []

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", []) if hasattr(round_data, "get") else [round_data]
        for game in games:
            raw_team1, raw_team2, court = extract_game_teams(game)
            team1 = [str(p) for p in raw_team1]
            team2 = [str(p) for p in raw_team2]

            team1_ratings = [skills[p] for p in team1 if p in skills]
            team2_ratings = [skills[p] for p in team2 if p in skills]
            if not team1_ratings and not team2_ratings:
                continue

            team1_total = sum(team1_ratings)
            team2_total = sum(team2_ratings)
            rows.append(
                {
                    "round_num": round_num,
                    "court": court,
                    "team1": team1,
                    "team2": team2,
                    "team1_skill_total": team1_total,
                    "team2_skill_total": team2_total,
                    "imbalance": abs(team1_total - team2_total),
                    "fully_rated": len(team1_ratings) == len(team1) and len(team2_ratings) == len(team2),
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
        games = round_data.get("games", []) if hasattr(round_data, "get") else []
        for game in games:
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
) -> dict[str, Any]:
    """Calculate fairness metrics and theoretical optimality for a schedule."""
    if not schedule:
        return {}

    player_stats: dict[str, dict[str, Any]] = {}

    for round_data in schedule:
        games = round_data.get("games", [])
        for game in games:
            team1, team2, court = extract_game_teams(game)

            for team in [team1, team2]:
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

    games_range = max(games_played) - min(games_played) if games_played else 0
    partners_range = max(partners_count) - min(partners_count) if partners_count else 0
    opponents_range = max(opponents_count) - min(opponents_count) if opponents_count else 0
    courts_range = max(courts_count) - min(courts_count) if courts_count else 0
    total_range = games_range + partners_range + opponents_range + courts_range

    if num_players and num_rounds and num_courts:
        total_slots = num_rounds * num_courts * 4
        total_partnerships = (num_players * (num_players - 1)) // 2
        partnership_slots = total_slots // 2

        partner_range0_possible = (
            partnership_slots % total_partnerships == 0 and partnership_slots >= total_partnerships
        )

        games_balanced = total_slots % num_players == 0
        courts_balanced = total_slots % (num_players * num_courts) == 0

        theoretical_partner_range = 0 if partner_range0_possible else 1
        theoretical_games_range = 0 if games_balanced else 1
        theoretical_courts_range = 0 if courts_balanced else 1
        theoretical_opponents_range = 0 if partner_range0_possible else 1

        theoretical_total_range = (
            theoretical_partner_range + theoretical_games_range + theoretical_courts_range + theoretical_opponents_range
        )

        if theoretical_total_range == 0:
            fairness_score = max(0, 10 - total_range)
        else:
            excess_range = total_range - theoretical_total_range
            fairness_score = 10 if excess_range <= 0 else max(0, 10 - excess_range * 2)
    else:
        fairness_score = max(0, 10 - total_range)
        theoretical_total_range = None

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
        "theoretical_optimum": theoretical_total_range,
        "is_mathematically_optimal": (
            total_range == theoretical_total_range if theoretical_total_range is not None else None
        ),
    }
