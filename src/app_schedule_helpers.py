#!/usr/bin/env python3
"""Schedule validation and display helpers extracted from the Streamlit app."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, List

try:
    from src.utils.schedule_shapes import extract_game_teams
except ImportError:
    from utils.schedule_shapes import extract_game_teams


def compute_round_times(num_rounds: int, start_time: str, end_time: str) -> List[str]:
    """Evenly divide a scheduling window into a display time per round.

    Returns an empty list - meaning "omit times" - whenever the inputs can't
    produce a sensible schedule: a non-positive round count, unparseable
    "HH:MM" strings, or a window that doesn't move forward (including
    midnight-crossing windows, which are not supported).
    """
    if num_rounds < 1:
        return []

    try:
        start = datetime.strptime(start_time, "%H:%M")
        end = datetime.strptime(end_time, "%H:%M")
    except (TypeError, ValueError):
        return []

    total_minutes = (end - start).total_seconds() / 60
    if total_minutes <= 0:
        return []

    interval = total_minutes / num_rounds
    return [(start + timedelta(minutes=interval * i)).strftime("%I:%M %p").lstrip("0") for i in range(num_rounds)]


def validate_schedule_integrity(schedule: Iterable[Any], all_players: Any = None) -> List[str]:
    """Comprehensive validation of schedule integrity."""
    _ = all_players
    errors: List[str] = []

    for round_num, round_data in enumerate(schedule, 1):
        round_playing_players: List[str] = []
        games = round_data.get("games", [])

        for game in games:
            team1, team2, court = extract_game_teams(game)
            team1_players = [str(p) for p in team1]
            team2_players = [str(p) for p in team2]
            round_playing_players.extend(team1_players)
            round_playing_players.extend(team2_players)

            # Duplicate/overlap checks only ever ran for attribute-style Game
            # objects historically - preserved here rather than silently
            # widened to dict-shaped games as part of this pure refactor.
            if hasattr(game, "team1"):
                overlap = set(team1_players) & set(team2_players)
                if overlap:
                    errors.append(f"Round {round_num}, Court {court}: " f"Player(s) {list(overlap)} on both teams!")

                if len(team1_players) != len(set(team1_players)):
                    duplicates = [p for p in team1_players if team1_players.count(p) > 1]
                    errors.append(f"Round {round_num}, Court {court}: " f"Duplicate in Team 1: {duplicates}")

                if len(team2_players) != len(set(team2_players)):
                    duplicates = [p for p in team2_players if team2_players.count(p) > 1]
                    errors.append(f"Round {round_num}, Court {court}: " f"Duplicate in Team 2: {duplicates}")

        player_counts: dict[str, int] = {}
        for player in round_playing_players:
            player_counts[player] = player_counts.get(player, 0) + 1

        for player, count in player_counts.items():
            if count > 1:
                errors.append(f"Round {round_num}: Player '{player}' " f"appears {count} times (IMPOSSIBLE!)")

    return errors


def display_enhanced_schedule(
    schedule: Iterable[Any],
    st_module: Any,
    pd_module: Any,
    all_players: Any = None,
    round_times: Any = None,
) -> None:
    """Display schedule with enhanced formatting and error checking."""
    schedule_errors = validate_schedule_integrity(schedule, all_players)

    if schedule_errors:
        st_module.error("🚨 **CRITICAL SCHEDULE ERRORS DETECTED!**")
        for error in schedule_errors:
            st_module.error(f"❌ {error}")
        st_module.error("**This schedule is INVALID and should not be used!**")
        return

    for round_num, round_data in enumerate(schedule, 1):
        round_time = round_times[round_num - 1] if round_times and round_num - 1 < len(round_times) else None
        heading = f"🎾 Round {round_num}" + (f" — {round_time}" if round_time else "")
        st_module.subheader(heading)

        games_data = []
        playing_players = set()
        games = round_data.get("games", [])

        for game in games:
            raw_team1, raw_team2, court = extract_game_teams(game)
            team1 = " & ".join([str(p) for p in raw_team1])
            team2 = " & ".join([str(p) for p in raw_team2])
            players_in_game: Iterable[Any] = list(raw_team1) + list(raw_team2)

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


def _extract_game_fields(game):
    """Return (team1, team2, court) as plain strings/values for either game shape."""
    if hasattr(game, "team1"):
        team1 = [str(p) for p in game.team1]
        team2 = [str(p) for p in game.team2]
        court = game.court
    else:
        team1 = [str(p) for p in game["team1"]]
        team2 = [str(p) for p in game["team2"]]
        court = game["court"]
    return team1, team2, court


def list_games_for_scoring(schedule) -> List[dict]:
    """Flatten a schedule into one row per game with round/court/team info,
    for the score-entry UI. Order matches display_enhanced_schedule."""
    games_for_scoring = []
    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", []) if hasattr(round_data, "get") else [round_data]
        for game in games:
            team1, team2, court = _extract_game_fields(game)
            games_for_scoring.append(
                {
                    "round_num": round_num,
                    "court": court,
                    "team1": team1,
                    "team2": team2,
                }
            )
    return games_for_scoring


def blank_score_sheet_csv(games: List[dict]) -> str:
    """Render a fillable CSV template - round,court,team1,team2,team1_score,team2_score -
    from list_games_for_scoring(schedule) output, for a scorekeeper to fill in and re-import."""
    lines = ["round,court,team1,team2,team1_score,team2_score"]
    for game in games:
        team1 = " & ".join(game["team1"])
        team2 = " & ".join(game["team2"])
        lines.append(f"{game['round_num']},{game['court']},{team1},{team2},,")
    return "\n".join(lines)


def schedule_to_text(schedule: Iterable[Any], all_players: Any = None, round_times: Any = None) -> str:
    """Render a schedule as plain text suitable for pasting into chat apps."""
    round_blocks: List[str] = []

    for round_num, round_data in enumerate(schedule, 1):
        round_time = round_times[round_num - 1] if round_times and round_num - 1 < len(round_times) else None
        heading = f"ROUND {round_num}" + (f" - {round_time}" if round_time else "")
        lines: List[str] = [heading]
        playing_players: set[str] = set()
        games = round_data.get("games", [])

        for game in games:
            team1, team2, court = _extract_game_fields(game)
            playing_players.update(team1)
            playing_players.update(team2)
            lines.append(f"Court {court}: {' & '.join(team1)} vs {' & '.join(team2)}")

        if all_players:
            sitting_out = [str(p) for p in all_players if str(p) not in playing_players]
            if sitting_out:
                lines.append(f"Sitting out: {', '.join(sitting_out)}")

        round_blocks.append("\n".join(lines))

    return "\n\n".join(round_blocks)


def schedule_to_player_text(schedule: Iterable[Any], all_players: Iterable[Any]) -> str:
    """Render a per-player view: one line per round showing court/partner/opponents."""
    player_lines: dict[str, list[str]] = {str(p): [] for p in all_players}

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", [])
        players_seen_this_round: set[str] = set()

        for game in games:
            team1, team2, court = _extract_game_fields(game)

            for player in team1:
                partner = [p for p in team1 if p != player]
                partner_text = f" with {partner[0]}" if partner else ""
                player_lines.setdefault(player, []).append(
                    f"R{round_num}: Court {court}{partner_text} vs {' & '.join(team2)}"
                )
                players_seen_this_round.add(player)

            for player in team2:
                partner = [p for p in team2 if p != player]
                partner_text = f" with {partner[0]}" if partner else ""
                player_lines.setdefault(player, []).append(
                    f"R{round_num}: Court {court}{partner_text} vs {' & '.join(team1)}"
                )
                players_seen_this_round.add(player)

        for player in all_players:
            player = str(player)
            if player not in players_seen_this_round:
                player_lines.setdefault(player, []).append(f"R{round_num}: sitting out")

    return "\n\n".join(f"{player}\n" + "\n".join(lines) for player, lines in player_lines.items())


def schedule_to_csv(schedule: Iterable[Any]) -> str:
    """Convert schedule to CSV format."""
    csv_lines: List[str] = ["Round,Court,Team1_Player1,Team1_Player2,Team2_Player1,Team2_Player2"]

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", [])
        for game in games:
            raw_team1, raw_team2, court = extract_game_teams(game)
            team1 = [str(p) for p in raw_team1]
            team2 = [str(p) for p in raw_team2]

            line = f"{round_num},{court},{team1[0]},{team1[1]},{team2[0]},{team2[1]}"
            csv_lines.append(line)

    return "\n".join(csv_lines)


def schedule_to_xlsx(schedule, all_players=None, round_times=None, scores=None) -> bytes:
    """Render a schedule as a formatted .xlsx workbook, returned as bytes.

    One "Schedule" sheet with a row per game (round, court, teams), reusing
    list_games_for_scoring for consistent dict/attr game-shape handling. An
    optional "Scores" sheet is added when `scores` (HistoryManager.get_game_scores
    output) is provided and non-empty.
    """
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.styles import Font

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Schedule"

    header_font = Font(bold=True)
    headers = ["Round", "Court", "Team 1", "Team 2", "Sitting Out"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = header_font

    games = list_games_for_scoring(schedule)
    round_playing: dict[int, set] = {}
    for game in games:
        round_playing.setdefault(game["round_num"], set()).update(game["team1"])
        round_playing.setdefault(game["round_num"], set()).update(game["team2"])

    for game in games:
        sitting_out = ""
        if all_players:
            playing = round_playing.get(game["round_num"], set())
            sitting_out = ", ".join(str(p) for p in all_players if str(p) not in playing)
        sheet.append(
            [
                game["round_num"],
                game["court"],
                " & ".join(game["team1"]),
                " & ".join(game["team2"]),
                sitting_out,
            ]
        )

    for column_cells in sheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

    if scores:
        scores_sheet = workbook.create_sheet("Scores")
        score_headers = ["Round", "Court", "Team 1", "Team 2", "Team 1 Score", "Team 2 Score"]
        scores_sheet.append(score_headers)
        for cell in scores_sheet[1]:
            cell.font = header_font
        for row in scores:
            scores_sheet.append(
                [
                    row["round_num"],
                    row["court"],
                    " & ".join(row["team1_players"]),
                    " & ".join(row["team2_players"]),
                    row.get("team1_score"),
                    row.get("team2_score"),
                ]
            )

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
