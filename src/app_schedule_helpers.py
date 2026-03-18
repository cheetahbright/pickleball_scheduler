#!/usr/bin/env python3
"""Schedule validation and display helpers extracted from the Streamlit app."""

from __future__ import annotations

from typing import Any, Iterable, List


def validate_schedule_integrity(schedule, all_players=None):
    """Comprehensive validation of schedule integrity."""
    _ = all_players
    errors = []

    for round_num, round_data in enumerate(schedule, 1):
        round_playing_players: List[str] = []
        games = round_data.get("games", [])

        for game in games:
            if hasattr(game, "team1"):
                round_playing_players.extend([str(p) for p in game.team1])
                round_playing_players.extend([str(p) for p in game.team2])

                team1_players = [str(p) for p in game.team1]
                team2_players = [str(p) for p in game.team2]
                overlap = set(team1_players) & set(team2_players)
                if overlap:
                    errors.append(
                        f"Round {round_num}, Court {game.court}: " f"Player(s) {list(overlap)} on both teams!"
                    )

                if len(team1_players) != len(set(team1_players)):
                    duplicates = [p for p in team1_players if team1_players.count(p) > 1]
                    errors.append(f"Round {round_num}, Court {game.court}: " f"Duplicate in Team 1: {duplicates}")

                if len(team2_players) != len(set(team2_players)):
                    duplicates = [p for p in team2_players if team2_players.count(p) > 1]
                    errors.append(f"Round {round_num}, Court {game.court}: " f"Duplicate in Team 2: {duplicates}")
            else:
                round_playing_players.extend([str(p) for p in game["team1"]])
                round_playing_players.extend([str(p) for p in game["team2"]])

        player_counts = {}
        for player in round_playing_players:
            player_counts[player] = player_counts.get(player, 0) + 1

        for player, count in player_counts.items():
            if count > 1:
                errors.append(f"Round {round_num}: Player '{player}' " f"appears {count} times (IMPOSSIBLE!)")

    return errors


def display_enhanced_schedule(schedule, st_module, pd_module, all_players=None):
    """Display schedule with enhanced formatting and error checking."""
    schedule_errors = validate_schedule_integrity(schedule, all_players)

    if schedule_errors:
        st_module.error("🚨 **CRITICAL SCHEDULE ERRORS DETECTED!**")
        for error in schedule_errors:
            st_module.error(f"❌ {error}")
        st_module.error("**This schedule is INVALID and should not be used!**")
        return

    for round_num, round_data in enumerate(schedule, 1):
        st_module.subheader(f"🎾 Round {round_num}")

        games_data = []
        playing_players = set()
        games = round_data.get("games", [])

        for game in games:
            if hasattr(game, "team1"):
                team1 = " & ".join([str(p) for p in game.team1])
                team2 = " & ".join([str(p) for p in game.team2])
                court = game.court
                players_in_game: Iterable[Any] = list(game.team1) + list(game.team2)
            else:
                team1 = " & ".join([str(p) for p in game["team1"]])
                team2 = " & ".join([str(p) for p in game["team2"]])
                court = game["court"]
                players_in_game = game["team1"] + game["team2"]

            for player in players_in_game:
                if str(player) in playing_players:
                    st_module.error(
                        f"🚨 **DUPLICATE PLAYER ERROR in Round {round_num}**: " f"{player} appears multiple times!"
                    )
                playing_players.add(str(player))

            games_data.append(
                {
                    "Court": f"Court {court}",
                    "Team 1": team1,
                    "vs": "vs",
                    "Team 2": team2,
                }
            )

        if games_data:
            df = pd_module.DataFrame(games_data)
            st_module.table(df)

            if all_players:
                sitting_out = [str(p) for p in all_players if str(p) not in playing_players]
                if sitting_out:
                    st_module.info(f"🪑 **Sitting out this round:** {', '.join(sitting_out)}")
                else:
                    st_module.info("🎾 **All players are playing this round**")


def schedule_to_csv(schedule):
    """Convert schedule to CSV format."""
    csv_lines = ["Round,Court,Team1_Player1,Team1_Player2,Team2_Player1,Team2_Player2"]

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", [])
        for game in games:
            if hasattr(game, "team1"):
                team1 = [str(p) for p in game.team1]
                team2 = [str(p) for p in game.team2]
                court = game.court
            else:
                team1 = [str(p) for p in game["team1"]]
                team2 = [str(p) for p in game["team2"]]
                court = game["court"]

            line = f"{round_num},{court},{team1[0]},{team1[1]},{team2[0]},{team2[1]}"
            csv_lines.append(line)

    return "\n".join(csv_lines)
