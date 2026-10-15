#!/usr/bin/env python3
"""Schedule validation and display helpers extracted from the Streamlit app."""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta
from typing import Any, Iterable, List

try:
    from src.utils.schedule_shapes import extract_game_teams
except ImportError:
    from utils.schedule_shapes import extract_game_teams

_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")
_FORMULA_TRIGGER = "[" + re.escape("".join(_FORMULA_TRIGGER_CHARS)) + "]"
# Where a spreadsheet can start a cell within one value: at its start, and after
# a ";", tab or line break in it. csv.writer (like st.dataframe's own CSV
# download) quotes only a field holding a comma, quote or line break, but Excel
# in a decimal-comma locale splits a CSV on ";" and honours quotes only at the
# start of a field, and a tab-separated reading splits on tabs.
_CELL_START = r"(?:^|(?<=[;\t\r\n]))"
_FORMULA_AT_CELL_START = re.compile(f"{_CELL_START}(?={_FORMULA_TRIGGER})")
_FORMULA_AFTER_APOSTROPHES_AT_CELL_START = re.compile(f"{_CELL_START}(?='*{_FORMULA_TRIGGER})")


def _excel_safe(value: str) -> str:
    """Neutralize spreadsheet formula injection (CWE-1236) in exported cells: an
    apostrophe goes before a formula character wherever a cell could start."""
    return _FORMULA_AT_CELL_START.sub("'", value)


def excel_safe(value: str) -> str:
    """_excel_safe for names shown in st.dataframe, whose toolbar has its own
    "Download as CSV" that writes cells (and the index and column headers)
    exactly as shown.

    Unlike _excel_safe it never gives two names the same text: "'=x" gets an
    apostrophe too, or it would show the same as "=x" - and the pairing
    matrix, whose columns are the players, can't have two columns with one name.
    """
    return _FORMULA_AFTER_APOSTROPHES_AT_CELL_START.sub("'", value)


# What opens or closes inline Markdown in Streamlit - emphasis, code, strikethrough,
# links and images, autolinks, LaTeX ($), :color[...] directives, table cells - and
# "&" before a letter or "#", which could start an entity like "&#58;". Other
# punctuation is left alone, so everyday names like "Mary-Jane O'Brien" come out
# unchanged: a widget's label is also its aria-label.
_MARKDOWN_INLINE = re.compile(r"([\\`*_~\[\]<>$:|]|&(?=[#A-Za-z]))")
# A heading, list item or setext underline marker, which only counts at the start
# of a line - for an ordered list ("1. Bob"), the "." or ")" after its number.
_MARKDOWN_LINE_START = re.compile(r"^([ \t]*(?:\d{1,9}(?=[.)](?:[ \t]|$)))?)([#+=.)-])", re.MULTILINE)
# The spots where GFM turns plain text into a link: "://", "www." and "@".
_AUTOLINK_BREAKS = re.compile(r"(?<=:)(?=//)|(?<=www)(?=\.)|(?=@)", re.IGNORECASE)
_WORD_JOINER = "\u2060"


def escape_markdown(text: Any) -> str:
    """Make user-supplied text (player and preset names) display literally where Streamlit renders Markdown.

    st.markdown/st.write, alerts, captions, st.table cells and widget and
    expander labels all parse Markdown, so a name could otherwise load a
    remote image, add a link, or - two "$" on one line - turn into LaTeX.
    Backslash escapes don't stop GFM linking a bare URL, "www." domain or
    e-mail address (it matches the text after escapes are resolved), so an
    invisible word joiner goes into those spots too.
    """
    text = _AUTOLINK_BREAKS.sub(_WORD_JOINER, str(text))
    text = _MARKDOWN_INLINE.sub(r"\\\1", text)
    return _MARKDOWN_LINE_START.sub(r"\1\\\2", text)


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


def validate_schedule_integrity(
    schedule: Iterable[Any], all_players: Any = None, roster_by_round: Any = None
) -> List[str]:
    """List what makes a schedule impossible to play, or [] if nothing does.

    Checks that every team has two players, that nobody plays twice in one
    round and, when all_players is given, that everyone playing is on it (the
    callers pass everyone in the session, so a replan's joiners and leavers
    are on it too). With roster_by_round (each round's roster, as a replan or
    an arrivals-and-departures plan records it), a player must also be on the
    roster of the round they play in: someone who has not arrived yet, or has
    left, is not playable. Rounds it has no roster for fall back to
    all_players. Courts and empty rounds are not checked.
    """
    roster = {str(p) for p in all_players} if all_players else None
    errors: List[str] = []

    for round_num, round_data in enumerate(schedule, 1):
        round_playing_players: List[str] = []
        games = round_data.get("games", [])
        round_roster = (
            {str(p) for p in roster_by_round[round_num - 1]}
            if roster_by_round and round_num - 1 < len(roster_by_round)
            else None
        )

        for game in games:
            team1, team2, court = extract_game_teams(game)
            team1_players = [str(p) for p in team1]
            team2_players = [str(p) for p in team2]
            round_playing_players.extend(team1_players)
            round_playing_players.extend(team2_players)

            for team_number, team_players in ((1, team1_players), (2, team2_players)):
                if len(team_players) != 2:
                    errors.append(
                        f"Round {round_num}, Court {court}: Team {team_number} has {len(team_players)} player(s), not 2"
                    )
            for player in team1_players + team2_players:
                if roster is not None and player not in roster:
                    errors.append(f"Round {round_num}, Court {court}: Player '{player}' is not in the player list")
                elif round_roster is not None and player not in round_roster:
                    errors.append(f"Round {round_num}, Court {court}: Player '{player}' is not on this round's roster")

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


def _round_roster(all_players: Any, roster_by_round: Any, round_num: int) -> Any:
    """Return the roster to compare "sitting out" against for a given round.

    Mid-session replan splices old-roster rounds with new-roster rounds under
    one schedule (see splice_schedules in utils/roster_rounds.py), so a single
    flat all_players list is wrong for either half: it would show players who
    already left as still sitting out, and players who haven't joined yet as
    already sitting out. roster_by_round, when supplied, is a list aligned to
    schedule rounds giving the actual roster in effect for each one.
    """
    if roster_by_round and round_num - 1 < len(roster_by_round):
        return roster_by_round[round_num - 1]
    return all_players


def display_enhanced_schedule(
    schedule: Iterable[Any],
    st_module: Any,
    pd_module: Any,
    all_players: Any = None,
    round_times: Any = None,
    roster_by_round: Any = None,
) -> None:
    """Display schedule with enhanced formatting and error checking."""
    schedule_errors = validate_schedule_integrity(schedule, all_players, roster_by_round)

    if schedule_errors:
        st_module.error("🚨 **CRITICAL SCHEDULE ERRORS DETECTED!**")
        for error in schedule_errors:
            st_module.error(f"❌ {escape_markdown(error)}")
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
            # st.table renders its cells as Markdown.
            team1 = " & ".join([escape_markdown(p) for p in raw_team1])
            team2 = " & ".join([escape_markdown(p) for p in raw_team2])
            players_in_game: Iterable[Any] = list(raw_team1) + list(raw_team2)

            for player in players_in_game:
                if str(player) in playing_players:
                    st_module.error(
                        f"🚨 **DUPLICATE PLAYER ERROR in Round {round_num}**: "
                        f"{escape_markdown(player)} appears multiple times!"
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

            round_roster = _round_roster(all_players, roster_by_round, round_num)
            if round_roster:
                sitting_out = [str(p) for p in round_roster if str(p) not in playing_players]
                if sitting_out:
                    st_module.info(
                        f"🪑 **Sitting out this round:** {', '.join(escape_markdown(p) for p in sitting_out)}"
                    )
                else:
                    st_module.info("🎾 **All players are playing this round**")


def _extract_game_fields(game):
    """Return (team1, team2, court) as plain strings for either game shape."""
    raw_team1, raw_team2, court = extract_game_teams(game)
    return [str(p) for p in raw_team1], [str(p) for p in raw_team2], court


def list_games_for_scoring(schedule) -> List[dict]:
    """Flatten a schedule into one row per game with round/court/team info,
    for the score-entry UI. Order matches display_enhanced_schedule."""
    games_for_scoring = []
    for round_num, round_data in enumerate(schedule, 1):
        for game in round_data.get("games", []):
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


def recorded_result(row) -> tuple[int, int] | None:
    """The (team1, team2) result a game_scores row records, or None if it has none.

    A row missing a score is no result, and neither is 0-0: an older Save
    Scores wrote 0-0 for every unplayed game (see history._result_row_condition).
    Nor is a score that isn't a number (SQLite keeps 'abc' as text in the
    INTEGER column): a hand-edited or badly restored row mustn't raise here,
    on every run, and stop the Leaderboard tab rendering.
    """
    if not row:
        return None
    scores = (row.get("team1_score"), row.get("team2_score"))
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in scores):
        return None
    if scores == (0, 0):
        return None
    return scores


def blank_score_sheet_csv(games: List[dict]) -> str:
    """Render a fillable CSV template - round,court,team1,team2,team1_score,team2_score -
    from list_games_for_scoring(schedule) output, for a scorekeeper to fill in and re-import."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["round", "court", "team1", "team2", "team1_score", "team2_score"])
    for game in games:
        team1 = _excel_safe(" & ".join(game["team1"]))
        team2 = _excel_safe(" & ".join(game["team2"]))
        writer.writerow([game["round_num"], game["court"], team1, team2, "", ""])
    return buffer.getvalue().rstrip("\n")


def schedule_to_text(
    schedule: Iterable[Any],
    all_players: Any = None,
    round_times: Any = None,
    roster_by_round: Any = None,
) -> str:
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

        round_roster = _round_roster(all_players, roster_by_round, round_num)
        if round_roster:
            sitting_out = [str(p) for p in round_roster if str(p) not in playing_players]
            if sitting_out:
                lines.append(f"Sitting out: {', '.join(sitting_out)}")

        round_blocks.append("\n".join(lines))

    return "\n\n".join(round_blocks)


def schedule_to_player_text(
    schedule: Iterable[Any],
    all_players: Iterable[Any],
    roster_by_round: Any = None,
) -> str:
    """Render a per-player view: one line per round showing court/partner/opponents."""
    player_lines: dict[str, list[str]] = {str(p): [] for p in all_players}

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", [])
        players_seen_this_round: set[str] = set()

        for game in games:
            team1, team2, court = _extract_game_fields(game)

            for team, opponents in ((team1, team2), (team2, team1)):
                for player in team:
                    partner = [p for p in team if p != player]
                    partner_text = f" with {partner[0]}" if partner else ""
                    player_lines.setdefault(player, []).append(
                        f"R{round_num}: Court {court}{partner_text} vs {' & '.join(opponents)}"
                    )
                    players_seen_this_round.add(player)

        # A player only "sits out" a round they were actually eligible for -
        # skip anyone outside that round's roster segment (e.g. a player who
        # left, or hasn't joined yet, across a mid-session replan).
        round_roster = _round_roster(all_players, roster_by_round, round_num)
        round_roster_players = {str(p) for p in round_roster} if round_roster else None
        for player in all_players:
            player = str(player)
            if player in players_seen_this_round:
                continue
            if round_roster_players is not None and player not in round_roster_players:
                continue
            player_lines.setdefault(player, []).append(f"R{round_num}: sitting out")

    return "\n\n".join(f"{player}\n" + "\n".join(lines) for player, lines in player_lines.items())


def schedule_to_csv(schedule: Iterable[Any]) -> str:
    """Convert schedule to CSV format."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Round", "Court", "Team1_Player1", "Team1_Player2", "Team2_Player1", "Team2_Player2"])

    for round_num, round_data in enumerate(schedule, 1):
        games = round_data.get("games", [])
        for game in games:
            raw_team1, raw_team2, court = extract_game_teams(game)
            team1 = [_excel_safe(str(p)) for p in raw_team1]
            team2 = [_excel_safe(str(p)) for p in raw_team2]

            writer.writerow([round_num, court, team1[0], team1[1], team2[0], team2[1]])

    return buffer.getvalue().rstrip("\n")


def schedule_to_xlsx(schedule, all_players=None, round_times=None, scores=None, roster_by_round=None) -> bytes:
    """Render a schedule as a formatted .xlsx workbook, returned as bytes.

    One "Schedule" sheet with a row per game (round, court, teams), reusing
    list_games_for_scoring for consistent dict/attr game-shape handling, and a
    "Time" column after "Round" when `round_times` are given. A "Scores" sheet
    is added when `scores` (HistoryManager.get_game_scores output) records at
    least one result.
    """
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Font

    def text_cell(value: str) -> str:
        # openpyxl raises on most control characters (a pasted backspace, say) -
        # which would stop the rest of the page rendering - and they don't show
        # on screen anyway. Dropped first, so they can't hide a formula character.
        # Only the start is guarded: unlike a CSV row, a workbook cell is never
        # split, so an apostrophe after a ";" in it would only show.
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        return "'" + value if value[:1] in _FORMULA_TRIGGER_CHARS else value

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Schedule"

    header_font = Font(bold=True)
    headers = ["Round", "Court", "Team 1", "Team 2", "Sitting Out"]
    if round_times:
        headers.insert(1, "Time")
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
        round_roster = _round_roster(all_players, roster_by_round, game["round_num"])
        if round_roster:
            playing = round_playing.get(game["round_num"], set())
            sitting_out = text_cell(", ".join(str(p) for p in round_roster if str(p) not in playing))
        game_row = [
            game["round_num"],
            game["court"],
            text_cell(" & ".join(game["team1"])),
            text_cell(" & ".join(game["team2"])),
            sitting_out,
        ]
        if round_times:
            round_index = game["round_num"] - 1
            game_row.insert(1, round_times[round_index] if round_index < len(round_times) else "")
        sheet.append(game_row)

    for column_cells in sheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

    results = [row for row in scores or [] if recorded_result(row) is not None]
    if results:
        scores_sheet = workbook.create_sheet("Scores")
        score_headers = ["Round", "Court", "Team 1", "Team 2", "Team 1 Score", "Team 2 Score"]
        scores_sheet.append(score_headers)
        for cell in scores_sheet[1]:
            cell.font = header_font
        for row in results:
            # Read as the score entry reads them (app_tabs._recorded_for_game): a
            # damaged row's team cell may hold anything JSON can (None, a number,
            # a string), and mustn't raise here, on every run, and stop the page
            # rendering. Anything but a list of players is written as no players.
            team1, team2 = (
                " & ".join(map(str, team)) if isinstance(team, list) else ""
                for team in (row.get("team1_players"), row.get("team2_players"))
            )
            scores_sheet.append(
                [
                    row["round_num"],
                    row["court"],
                    text_cell(team1),
                    text_cell(team2),
                    row.get("team1_score"),
                    row.get("team2_score"),
                ]
            )

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
