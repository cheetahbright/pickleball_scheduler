#!/usr/bin/env python3
"""The Main tab's "Arrivals and departures" control: who arrives late or leaves early.

Everyone is assumed to stay for all the rounds; the people listed here are
present for part of them, and the attendance planner
(src/algorithms/attendance_planner.py) schedules each round with only the
people who are there. This module holds what sits between the widgets and the
planner:

- pure helpers that turn the widget values into the planner's attendance
  (clamped, so a stale value can never crash a run), describe the rounds that
  result, and read an attendance back from a saved schedule;
- the widget state bookkeeping, following src/widget_state.py;
- render_attendance_controls and render_attendance_summary, which draw them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from src.algorithms.attendance_planner import (
        PLAYERS_PER_COURT,
        AttendanceError,
        attendance_summary,
        rosters_by_round,
        segment_rosters,
    )
    from src.app_schedule_helpers import escape_markdown
    from src.widget_state import (
        forget_widget_state,
        remember_widget_choice,
        seed_widget_choice,
        seed_widget_from_source,
    )
except ImportError:
    from algorithms.attendance_planner import (
        PLAYERS_PER_COURT,
        AttendanceError,
        attendance_summary,
        rosters_by_round,
        segment_rosters,
    )
    from app_schedule_helpers import escape_markdown
    from widget_state import (
        forget_widget_state,
        remember_widget_choice,
        seed_widget_choice,
        seed_widget_from_source,
    )

# Who is here when: a (first round, last round) pair for each player listed.
Windows = dict[str, tuple[int, int]]

EXPANDER_LABEL = "🕒 Arrivals and departures"

# The two multiselects. The number inputs under them are keyed by player name
# (never by position), with prefixes no other key starts with.
ARRIVES_SELECT_KEY = "attendance_arrives_late"
LEAVES_SELECT_KEY = "attendance_leaves_early"
ARRIVES_AT_KEY_PREFIX = "attendance_arrives_at::"
LEAVES_AFTER_KEY_PREFIX = "attendance_leaves_after::"

# Set once someone has been listed in this session, so the expander stays open (see render_attendance_controls).
EXPANDER_OPENED_KEY = "attendance_expander_opened"

# The arrivals and departures the schedule on screen was planned from,
# {name: [first round, last round]}, or None for any other schedule.
CURRENT_ATTENDANCE_KEY = "current_attendance"

# What a plan's sit-outs can honestly be said to be (see attendance_planner's docstring).
SIT_OUT_CAVEAT = (
    "Each stretch of rounds with the same people is balanced on its own, carrying on from the sit-outs before "
    "it, so the spread over the whole session is not guaranteed to be the smallest possible."
)
REPLAY_NOTE = (
    "🎲 A schedule planned around arrivals and departures can't be replayed exactly from a seed and generation "
    "count: each stretch of rounds is searched on its own within the time limit."
)
QUALITY_NOTE = (
    "ℹ️ Arrivals and departures are set, so quality is judged per stretch: each run of rounds with the same "
    "players is scheduled and balanced on its own. The estimates for one fixed roster don't apply and aren't shown."
)
# What the short column headings of the who-was-here-when table stand for.
TABLE_KEY = "From and To are the first and last round a player was here for, Rounds how many rounds that is."
# Shown in the Mid-session replan of a schedule planned around arrivals and departures.
REPLAN_NOTE = (
    "This schedule was planned around arrivals and departures. A replan gives all the remaining rounds the one "
    "list of players below, which starts as the players planned for the first of them, so the arrivals and "
    "departures planned for later rounds are not kept: replan again when a late arriver turns up or someone leaves."
)
# How the planner words a round without enough players (normalize_attendance), and what the user can do about it.
SHORT_ROUND_PROBLEM = f"fewer than {PLAYERS_PER_COURT} players are present"
SHORT_ROUND_HINT = "Have someone arrive sooner or leave later, or take them off the lists."
# Why the fairness score of a schedule whose rosters change (History, Analytics) is not a verdict on it.
FAIRNESS_NOTE = (
    "Fairness is scored against the rounds each player was here for. It reads low whenever players were here for "
    "different numbers of rounds (arrivals, departures, a replan), so it is not a verdict on the schedule."
)
# The algorithm History records for a schedule planned around arrivals and departures.
ATTENDANCE_ALGORITHM = "attendance"


def arrival_key(name: str) -> str:
    return f"{ARRIVES_AT_KEY_PREFIX}{name}"


def departure_key(name: str) -> str:
    return f"{LEAVES_AFTER_KEY_PREFIX}{name}"


def clamp_round(value: Any, low: int, high: int, default: int) -> int:
    """value as a round number within [low, high]; default (clamped too) when it isn't a whole number."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        value = default
    return max(low, min(high, int(value)))


def selected_players(selection: Any, players: Sequence[str]) -> list[str]:
    """The names a multiselect returned that are players, once each and in the order picked."""
    if not isinstance(selection, (list, tuple)):
        return []
    roster = set(players)
    return list(dict.fromkeys(name for name in selection if isinstance(name, str) and name in roster))


def build_attendance(
    players: Sequence[str],
    num_rounds: int,
    arrivals: Mapping[str, Any],
    departures: Mapping[str, Any],
) -> Windows:
    """The (first round, last round) of each listed player, in roster order.

    arrivals maps the players who arrive late to the round they arrive at,
    and departures those who leave early to the round they leave after. Both
    are clamped (an arrival to 2..num_rounds, a departure to 1..num_rounds-1),
    so a value left over from a longer session can't break a shorter one. A
    player in both lists is present between the two; one in neither is left
    out, which the planner reads as here for every round. A name that isn't
    a player is ignored, and with fewer than two rounds nobody can arrive
    late or leave early.
    """
    if num_rounds < 2:
        return {}
    windows: Windows = {}
    for name in players:
        if name not in arrivals and name not in departures:
            continue
        first = clamp_round(arrivals[name], 2, num_rounds, 2) if name in arrivals else 1
        last = clamp_round(departures[name], 1, num_rounds - 1, num_rounds - 1) if name in departures else num_rounds
        windows[name] = (first, last)
    return windows


def attendance_error_message(error: AttendanceError) -> str:
    """The red message for an attendance that can't be planned, one line per problem.

    The problems name players, so they are escaped like every other name
    shown. When a round is short of players it ends with what to change.
    Generate refuses with this same message.
    """
    problems = [escape_markdown(problem) for problem in error.problems]
    hint = SHORT_ROUND_HINT if any(problem.startswith(SHORT_ROUND_PROBLEM) for problem in problems) else ""
    if len(problems) == 1:
        return f"❌ These arrivals and departures can't be scheduled: {problems[0]}" + (f". {hint}" if hint else "")
    message = "❌ These arrivals and departures can't be scheduled:\n" + "\n".join(
        f"- {problem}" for problem in problems
    )
    return message + (f"\n\n{hint}" if hint else "")


def _round_label(first_round: int, last_round: int) -> str:
    return f"Round {first_round}" if first_round == last_round else f"Rounds {first_round}-{last_round}"


def headcount_lines(rosters: Sequence[Sequence[str]], courts: int) -> list[str]:
    """One line per stretch of rounds with the same players: how many are there, the courts they fill, who sits out.

    ``courts`` is the most courts to use; a stretch with too few players for
    them uses fewer, as the planner does.
    """
    lines = []
    for segment in segment_rosters(rosters):
        present = len(segment.players)
        courts_used = min(int(courts), present // PLAYERS_PER_COURT)
        parts = [
            f"{present} player{'s' if present != 1 else ''}",
            f"{courts_used} court{'s' if courts_used != 1 else ''}",
        ]
        sitting_out = present - courts_used * PLAYERS_PER_COURT
        if sitting_out:
            each_round = " each round" if segment.num_rounds > 1 else ""
            parts.append(f"{sitting_out} {'sits' if sitting_out == 1 else 'sit'} out{each_round}")
        lines.append(f"{_round_label(segment.first_round, segment.last_round)}: {', '.join(parts)}")
    return lines


@dataclass(frozen=True)
class AttendanceChoice:
    """What the Arrivals and departures control holds this run.

    windows holds the players listed (empty: everyone is here for every
    round, and Generate takes its usual path). rosters is who is present in
    each round, or None when nothing is listed or the attendance can't be
    planned, in which case error is the message to show.
    """

    windows: Windows
    rosters: list[list[str]] | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.windows)


def evaluate_attendance(players: Sequence[str], num_rounds: int, windows: Windows) -> AttendanceChoice:
    """Check the attendance with the planner's own rules, keeping its message if it can't be planned."""
    if not windows:
        return AttendanceChoice({})
    try:
        return AttendanceChoice(windows, rosters=rosters_by_round(players, num_rounds, windows))
    except AttendanceError as error:
        return AttendanceChoice(windows, error=attendance_error_message(error))


def attendance_for_history(windows: Windows) -> dict[str, list[int]]:
    """The attendance as a history setting: {name: [first round, last round]}."""
    return {name: [first, last] for name, (first, last) in windows.items()}


def load_attendance_from_history_entry(entry: Mapping[str, Any], json_module: Any, num_rounds: int) -> dict | None:
    """The arrivals and departures a schedule was planned from, or None if it has none (or they don't fit)."""
    try:
        settings = json_module.loads(entry.get("settings_data") or "{}")
    except (TypeError, ValueError):
        return None
    saved = settings.get("attendance") if isinstance(settings, dict) else None
    if not isinstance(saved, dict) or not saved:
        return None
    attendance: dict[str, list[int]] = {}
    for name, window in saved.items():
        if (
            not isinstance(window, list)
            or len(window) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in window)
            or not 1 <= window[0] <= window[1] <= num_rounds
        ):
            return None
        attendance[str(name)] = [window[0], window[1]]
    return attendance


def sit_out_balance_sentence(summary: Sequence[Mapping[str, Any]]) -> str:
    """One honest sentence on how evenly the plan shared out the sit-outs."""
    counts = [int(row["sit_outs"]) for row in summary]
    if not counts:
        return ""
    low, high = min(counts), max(counts)
    if high == 0:
        return "Nobody sits out: every player plays every round they are here."
    spread = f"Every player sits out {low} time(s)" if low == high else f"Sit-outs run from {low} to {high} per player"
    return f"{spread}. {SIT_OUT_CAVEAT}"


def rosters_vary(roster_by_round: Sequence[Sequence[str]] | None) -> bool:
    """Whether the people present differ from one round to another (an attendance plan, a replan)."""
    if not roster_by_round:
        return False
    return len({frozenset(str(name) for name in roster) for roster in roster_by_round}) > 1


def summary_table_rows(
    summary: Sequence[Mapping[str, Any]], players: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """attendance_summary's rows as a table of who was here when (names escaped: st.table renders Markdown).

    The rows follow ``players`` (the roster order) where it names them; the
    summary itself lists a late arriver after the people there from round 1.
    The columns are short, so the table fits a phone: From and To are the
    first and last round the player was here, Rounds how many of them.
    """
    order = {str(name): position for position, name in enumerate(players or [])}
    ranked = sorted(summary, key=lambda row: order.get(str(row["player"]), len(order)))  # stable: ties keep order
    return [
        {
            "Player": escape_markdown(row["player"]),
            "From": row["first_round"],
            "To": row["last_round"],
            "Rounds": row["rounds_present"],
            "Games": row["games_played"],
            "Sit-outs": row["sit_outs"],
        }
        for row in ranked
    ]


# --- Widget state -------------------------------------------------------------------------------

# A session_state entry of a player's number input, or the bookkeeping kept for it
# (src/widget_state.py prefixes its entries with "_...:"): the widget key, then the name.
_PLAYER_ENTRY = re.compile(
    r"^(?:_[a-z_]+:)?("
    + "|".join(re.escape(prefix) for prefix in (ARRIVES_AT_KEY_PREFIX, LEAVES_AFTER_KEY_PREFIX))
    + r")(.*)$",
    re.DOTALL,
)


def _player_entry(state_key: str) -> tuple[str, str] | None:
    """(widget key, player name) for a session_state entry that belongs to a player's number input."""
    match = _PLAYER_ENTRY.match(state_key)
    return (match.group(1) + match.group(2), match.group(2)) if match else None


def remember_attendance_choices(session_state: Any) -> None:
    """Record the round each player's number input holds as this run starts.

    Call it early in the Main tab with the other remember_* calls, before a
    handler can st.rerun(): the rerun drops the state of every widget not
    drawn yet, but a choice recorded here is put back when the widget is
    drawn again (see seed_widget_choice).
    """
    for state_key in list(session_state.keys()):
        if isinstance(state_key, str) and state_key.startswith((ARRIVES_AT_KEY_PREFIX, LEAVES_AFTER_KEY_PREFIX)):
            remember_widget_choice(session_state, state_key)


def sync_attendance_state(session_state: Any, players: Sequence[str]) -> None:
    """Make the two multiselects' state fit the roster, before they are drawn.

    A player removed from the roster is dropped from the selections, and
    everything kept for their number inputs is forgotten, so a player added
    later under that name starts afresh. The selections are seeded rather
    than left to the widgets, so one an st.rerun() dropped comes back as the
    browser still shows it (see seed_widget_from_source).
    """
    roster = set(players)
    for key in (ARRIVES_SELECT_KEY, LEAVES_SELECT_KEY):
        seed_widget_from_source(session_state, key, [])
        selected = session_state[key]
        kept = [name for name in selected if name in roster] if isinstance(selected, list) else []
        if kept != selected:
            session_state[key] = kept
    stale_keys = set()
    for state_key in list(session_state.keys()):
        entry = _player_entry(state_key) if isinstance(state_key, str) else None
        if entry is not None and entry[1] not in roster:
            stale_keys.add(entry[0])
    for key in stale_keys:
        forget_widget_state(session_state, key)


# --- Rendering ----------------------------------------------------------------------------------


def render_attendance_controls(
    st_module: Any, players: Sequence[str], num_rounds: int, courts: int
) -> AttendanceChoice:
    """Draw the Arrivals and departures expander and return what it holds.

    It starts closed and opens when someone is first listed. From then on it
    is drawn open as far as the page is concerned, so it stays as the user
    leaves it: one that was drawn closed the moment the last person was
    removed would close under the cursor that removed them. Under it is the
    live feedback: the headcount of each stretch of rounds with the same
    players, or a red message naming every problem (Generate refuses with
    that message).

    With a single round nobody can arrive late or leave early, so the two
    selections are drawn disabled and ignored: they keep their place on the
    page and what was listed comes back once there are two rounds again.
    """
    state = st_module.session_state
    sync_attendance_state(state, players)
    listed = bool(state.get(ARRIVES_SELECT_KEY) or state.get(LEAVES_SELECT_KEY))
    if listed:
        state[EXPANDER_OPENED_KEY] = True
    usable = num_rounds >= 2

    arrivals: dict[str, Any] = {}
    departures: dict[str, Any] = {}
    with st_module.expander(EXPANDER_LABEL, expanded=bool(state.get(EXPANDER_OPENED_KEY))):
        if usable:
            st_module.caption(
                f"Everyone is assumed to stay for all {num_rounds} rounds. List the people who arrive after round 1 "
                "or leave before the last round; each round is then scheduled with only the people who are there."
            )
        else:
            ignored = " Anyone listed below is ignored until there are." if listed else ""
            st_module.caption(
                f"Needs at least 2 rounds: nobody can arrive after round 1 or leave before the last.{ignored}"
            )
        arrives_column, leaves_column = st_module.columns(2)
        with arrives_column:
            late = st_module.multiselect(
                "Arrives late:",
                options=list(players),
                key=ARRIVES_SELECT_KEY,
                placeholder="Everyone is here from round 1",
                disabled=not usable,
            )
            for name in selected_players(late, players) if usable else []:
                seed_widget_choice(state, arrival_key(name), 2, 2, num_rounds)
                arrivals[name] = st_module.number_input(
                    f"{escape_markdown(name)} arrives at round",
                    min_value=2,
                    max_value=num_rounds,
                    step=1,
                    key=arrival_key(name),
                )
        with leaves_column:
            early = st_module.multiselect(
                "Leaves early:",
                options=list(players),
                key=LEAVES_SELECT_KEY,
                placeholder="Everyone stays to the last round",
                disabled=not usable,
            )
            for name in selected_players(early, players) if usable else []:
                seed_widget_choice(state, departure_key(name), num_rounds - 1, 1, num_rounds - 1)
                departures[name] = st_module.number_input(
                    f"{escape_markdown(name)} leaves after round",
                    min_value=1,
                    max_value=num_rounds - 1,
                    step=1,
                    key=departure_key(name),
                )

        choice = evaluate_attendance(players, num_rounds, build_attendance(players, num_rounds, arrivals, departures))
        if choice.error:
            st_module.error(choice.error)
        elif choice.rosters:
            st_module.markdown("\n".join(f"- {line}" for line in headcount_lines(choice.rosters, courts)))
    return choice


def render_attendance_summary(
    st_module: Any,
    schedule: Sequence[Any],
    roster_by_round: Sequence[Sequence[str]] | None,
    players: Sequence[str] | None = None,
) -> None:
    """Show who was here when, for a schedule planned around arrivals and departures.

    The fairness metrics and the theoretical optimum assume one roster, so
    this replaces them: each player's first and last round, rounds present,
    games and sit-outs, with one sentence on how evenly the sit-outs fell.
    The players are the table's row headers (no number column beside them to
    take up room on a phone), listed in the order of ``players``.
    """
    if not roster_by_round or len(roster_by_round) != len(schedule):
        return
    summary = attendance_summary(schedule, roster_by_round)
    if not summary:
        return
    st_module.markdown("**Who was here when**")
    st_module.caption(TABLE_KEY)
    st_module.table(pd.DataFrame(summary_table_rows(summary, players)).set_index("Player"))
    st_module.caption(sit_out_balance_sentence(summary))
    st_module.caption(REPLAY_NOTE)
