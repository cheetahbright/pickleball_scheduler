"""Pure schedule analytics helpers shared by the Streamlit app and tests."""

from typing import Any, Dict, List, Optional


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
                        game_dict = {
                            "team1": [str(player) for player in game.team1],
                            "team2": [str(player) for player in game.team2],
                            "court": game.court,
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
                serializable_schedule.append(
                    {
                        "round": len(serializable_schedule) + 1,
                        "games": [
                            {
                                "team1": [str(player) for player in round_data.team1],
                                "team2": [str(player) for player in round_data.team2],
                                "court": round_data.court,
                            }
                        ],
                    }
                )

    return serializable_schedule


def calculate_fairness_metrics(
    schedule: List[Dict],
    num_players: Optional[int] = None,
    num_rounds: Optional[int] = None,
    num_courts: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate fairness metrics and theoretical optimality for a schedule."""
    if not schedule:
        return {}

    player_stats = {}

    for round_data in schedule:
        games = round_data.get("games", [])
        for game in games:
            if hasattr(game, "team1"):
                team1 = game.team1
                team2 = game.team2
                court = game.court
            else:
                team1 = game.get("team1", [])
                team2 = game.get("team2", [])
                court = game.get("court", 1)

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

    games_played = [stats["games_played"] for stats in player_stats.values()]
    partners_count = [len(stats["partners"]) for stats in player_stats.values()]
    opponents_count = [len(stats["opponents"]) for stats in player_stats.values()]
    courts_count = [len(stats["courts_used"]) for stats in player_stats.values()]

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
