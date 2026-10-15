#!/usr/bin/env python3
"""Scheduler page and generation flow extracted from the Streamlit app."""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from src.app_schedule_helpers import escape_markdown
except ImportError:
    from app_schedule_helpers import escape_markdown

try:
    from src.algorithms.attendance_planner import sit_out_baseline
except ImportError:
    from algorithms.attendance_planner import sit_out_baseline

try:
    from src.utils.roster_rounds import constraint_pairs_for_roster, pair_in_roster, sit_outs_by_round, splice_schedules
except ImportError:
    from utils.roster_rounds import constraint_pairs_for_roster, pair_in_roster, sit_outs_by_round, splice_schedules

try:
    from src.algorithms.attendance_planner import AttendanceError, plan_attendance_schedule
    from src.app_attendance import (
        CURRENT_ATTENDANCE_KEY,
        QUALITY_NOTE,
        REPLAN_NOTE,
        attendance_error_message,
        attendance_for_history,
        clamp_round,
        remember_attendance_choices,
        render_attendance_controls,
        render_attendance_summary,
    )
except ImportError:
    from algorithms.attendance_planner import AttendanceError, plan_attendance_schedule
    from app_attendance import (
        CURRENT_ATTENDANCE_KEY,
        QUALITY_NOTE,
        REPLAN_NOTE,
        attendance_error_message,
        attendance_for_history,
        clamp_round,
        remember_attendance_choices,
        render_attendance_controls,
        render_attendance_summary,
    )

try:
    from src.widget_state import (
        apply_deferred_widget_values,
        defer_widget_value,
        forget_widget_state,
        rebase_line_edits,
        remember_seeded_widget_values,
        remember_widget_choice,
        seed_widget_choice,
        seed_widget_from_source,
    )
except ImportError:
    from widget_state import (
        apply_deferred_widget_values,
        defer_widget_value,
        forget_widget_state,
        rebase_line_edits,
        remember_seeded_widget_values,
        remember_widget_choice,
        seed_widget_choice,
        seed_widget_from_source,
    )

normalize_config_constraints = import_module_with_fallback("app_managers").normalize_config_constraints
# Read as features.<FLAG> at call time, never copied, so a patched flag takes effect.
features = import_module_with_fallback("features")

logger = logging.getLogger(__name__)


# Deliberately generic placeholder names for the onboarding demo - distinct from
# the protected default roster in docs/summaries/PLAYER_NAMES_POLICY.md.
SAMPLE_PLAYER_NAMES = ["Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper"]

# Widget keys owned by render_main_scheduler_tab.
PLAYER_PRESET_SELECT_KEY = "player_preset_select"
PLAYERS_INPUT_KEY = "players_input"
QUICK_ADD_KEY = "quick_add"
PRESET_NAME_KEY = "preset_name"
NUM_ROUNDS_KEY = "num_rounds_input"
COURTS_KEY = "courts_available"
MAX_TIME_KEY = "max_generation_time"
# Quick Remove buttons are keyed by name (see render_main_scheduler_tab).
QUICK_REMOVE_KEY_PREFIX = "quick_remove::"

# The preset dropdown's own choices, listed ahead of the saved presets. A
# saved preset can't share one of these names: picking it would load the
# built-in choice instead.
RESERVED_PRESET_NAMES = ("Custom", "Default Names")

# The history save that failed for the schedule on screen, kept so it can be
# retried (see save_schedule_to_history).
UNSAVED_SCHEDULE_KEY = "unsaved_schedule"

# The schedule the mid-session replan's inputs were last set up for (see
# _render_mid_session_replan).
REPLAN_INPUTS_SCHEDULE_KEY = "_replan_inputs_schedule"

# Largest roster the scheduler accepts - 18 full courts, well beyond any real
# session (the Stress Test tab stops at 40 players). The constraint pickers
# offer every possible pair, which grows with the square of the roster (2,556
# options each at 72), so a pasted membership list of thousands of names would
# otherwise stall every rerun and run the shared server process out of memory.
# A preset or backup saved under the old limit of 200 can hold more: it is
# kept as it is and refused, with too_many_players_message, when it is picked.
MAX_PLAYERS = 72


def is_reserved_preset_name(name: str) -> bool:
    """Whether ``name`` is, ignoring case and outer spaces, one of RESERVED_PRESET_NAMES."""
    return name.strip().casefold() in {reserved.casefold() for reserved in RESERVED_PRESET_NAMES}


def reserved_preset_name_message(name: str) -> str:
    return (
        f"❌ '{escape_markdown(name.strip())}' is one of the preset list's built-in choices "
        f"({', '.join(RESERVED_PRESET_NAMES)}), so a preset can't use it. Choose a different name."
    )


def too_many_players_message(player_count: int) -> str:
    """Why a roster over MAX_PLAYERS is refused - shared by every place one can be entered or saved."""
    return f"{player_count} players is more than the {MAX_PLAYERS} this scheduler supports."


def set_player_preset(session_state: Any, preset: str) -> None:
    """Switch the active preset from outside the selectbox widget itself.

    Callers are button handlers that run after the selectbox has already been
    instantiated this run, so the widget key can only be updated via a
    deferred write - see src/widget_state.py for why.
    """
    session_state.selected_player_preset = preset
    defer_widget_value(session_state, PLAYER_PRESET_SELECT_KEY, preset)


def parse_players_text(players_text: str) -> list[str]:
    """Normalize textarea input into a clean player list."""
    return [line.strip() for line in players_text.splitlines() if line.strip()]


def split_schedule_at_round(schedule: list, played_rounds: int) -> tuple[list, int]:
    """Split a schedule into (locked_rounds, remaining_round_count) for a mid-session replan.

    locked_rounds are the rounds already played and must never be touched;
    remaining_round_count is how many fresh rounds need generating to finish
    out the original session length. played_rounds is clamped to
    [0, len(schedule)] so an out-of-range value can't lose or invent rounds.
    """
    played_rounds = max(0, min(played_rounds, len(schedule)))
    return list(schedule[:played_rounds]), len(schedule) - played_rounds


def add_player_to_text(players_text: str, new_player: str) -> str:
    """Append a player to the textarea content if it is not already present."""
    current_players = parse_players_text(players_text)
    normalized_player = new_player.strip()
    if normalized_player and normalized_player not in current_players:
        current_players.append(normalized_player)
    return "\n".join(current_players)


def remove_player_from_text(players_text: str, player_to_remove: str) -> str:
    """Remove a player from the textarea content."""
    current_players = parse_players_text(players_text)
    return "\n".join([player for player in current_players if player != player_to_remove])


def _constructive_design_available(num_players: int, num_courts: int, num_rounds: int) -> bool:
    """Whether the constructive fast path (constructive_scheduler.py) has a
    verified design for this exact configuration - producing a range-0
    schedule instantly, without running the GA search at all."""
    if num_players != num_courts * 4:
        return False
    build_perfect_schedule = import_module_with_fallback("algorithms.constructive_scheduler").build_perfect_schedule
    probe_names = [str(i) for i in range(num_players)]
    return build_perfect_schedule(probe_names, num_courts, num_rounds) is not None


def build_feasibility_notes(num_players: int, num_courts: int, num_rounds: int, num_constraints: int = 0) -> list[str]:
    """Advisory notes about whether this configuration can reach a perfectly fair schedule.

    Runs before generation so users are not surprised by a mediocre score after
    waiting up to the full max-time budget for a combination that was never
    going to reach range 0. It warns only about what is proven (see
    ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums): games that
    can't divide evenly, or players who can't get a game at all. Anything else
    is left to the search. num_constraints is the number of do-not-pair/
    do-not-oppose constraints active for this roster: any of them turns off
    the instant constructive schedule, so the search runs instead.
    """
    ScheduleFeasibilityAnalyzer = import_module_with_fallback("utils.feasibility_analyzer").ScheduleFeasibilityAnalyzer

    feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(num_players, num_courts, num_rounds)
    notes: list[str] = []

    if feasibility["range_0_possible"]:
        if not num_constraints and _constructive_design_available(num_players, num_courts, num_rounds):
            notes.append(f"✅ A perfectly fair schedule (range 0) is guaranteed instant for {num_players} players.")
        else:
            notes.append(
                f"Nothing rules out a perfectly fair schedule (range 0) for {num_players} players - the search "
                "looks for one for up to the max generation time, and stops early if it finds one."
            )
        if num_constraints:
            notes.append(
                f"With {num_constraints} active constraint(s) the schedule has to be searched for: that can take "
                "up to the max generation time, and the constraints may keep it from reaching range 0."
            )
        return notes

    slots = num_rounds * num_courts * 4
    if feasibility["total_theoretical_min"] > feasibility["min_games_range"]:
        notes.append(
            f"⚠️ {num_rounds} round(s) on {num_courts} court(s) give {slots} places in games for "
            f"{num_players} players, so {num_players - slots} can't play at all - perfect fairness (range 0) is "
            f"not possible, and the total range can't go below {feasibility['total_theoretical_min']}."
        )
        return notes

    notes.append(
        f"⚠️ {num_rounds} round(s) on {num_courts} court(s) give {slots} places in games, which can't divide "
        f"evenly among {num_players} players - some play one game more, so perfect fairness (range 0) is not "
        "possible and the total range can't go below 1."
    )

    # Courts are the user's fixed physical resource, not something that
    # scales with the candidate player count - a suggestion is only useful
    # if it's achievable with the courts actually available, so a candidate
    # must fill every court.
    even_counts = [
        candidate
        for candidate in (num_players + delta for delta in (-4, -3, -2, -1, 1, 2, 3, 4))
        if candidate >= num_courts * 4 and slots % candidate == 0
    ]
    if even_counts:
        counts_text = " or ".join(str(c) for c in even_counts[:3])
        notes.append(f"With {counts_text} players (same courts and rounds), the games would divide evenly.")

    return notes


def suggest_max_generation_time(num_players: int, num_constraints: int) -> int:
    """Suggest a starting "Max generation time" scaled to this configuration's
    complexity, rather than a flat default regardless of player count or
    constraint pressure.

    Starts from the same per-player-count runtime tiers
    ScheduleFeasibilityAnalyzer.get_optimal_parameters already uses internally
    for the GA's own tuning, then adds a bump proportional to constraint
    density: do_not_pair/do_not_oppose constraints shrink the feasible
    arrangement pool add_constraints() has to rebuild and search
    (see genetic_scheduler.py), so a heavily constrained group needs more
    time to find a good schedule than an unconstrained one of the same size.
    The result is capped to the widget's own 10-300s range.
    """
    ScheduleFeasibilityAnalyzer = import_module_with_fallback("utils.feasibility_analyzer").ScheduleFeasibilityAnalyzer
    base_runtime = ScheduleFeasibilityAnalyzer.get_optimal_parameters(num_players)["max_runtime"]

    constraint_density = num_constraints / max(1, num_players)
    constraint_bump = min(60.0, constraint_density * 40.0)

    return int(min(300, max(10, round(base_runtime + constraint_bump))))


def constraint_pressure_warnings(
    players: list[str],
    pair_constraints: list,
    oppose_constraints: list,
) -> list[str]:
    """Flag players constrained against so many others that scheduling gets tight."""
    warnings: list[str] = []
    total_players = len(players)
    if total_players < 2:
        return warnings

    constrained_counts = {player: 0 for player in players}
    for pair in list(pair_constraints) + list(oppose_constraints):
        if len(pair) != 2:
            continue
        first, second = pair
        if first in constrained_counts:
            constrained_counts[first] += 1
        if second in constrained_counts:
            constrained_counts[second] += 1

    # Floor of 2 (not 1) so a single, completely ordinary constraint doesn't
    # trigger this at the smallest supported group sizes (4-6 players),
    # where (total_players - 1) // 2 would otherwise degenerate to 1.
    threshold = max(2, (total_players - 1) // 2)
    for player, count in constrained_counts.items():
        if count >= threshold:
            warnings.append(
                f"{player} is constrained against {count} of {total_players - 1} other players - "
                "this may make scheduling difficult."
            )

    return warnings


# The round search's time budget is the max runtime clamped to this range (see
# GeneticPickleballScheduler._search_arrangements), and the Max generation time
# widget stops at the top of it.
ROUND_SEARCH_MIN_SECONDS = 20
MAX_GENERATION_TIME_SECONDS = 300


def constraint_violation_warning(
    violations: int, budget_reached: bool, *, replan: bool = False, max_time: float | None = None
) -> str:
    """The warning for a schedule that breaks a do-not-pair/do-not-oppose constraint.

    When no round in the search meets every constraint, the GA keeps the
    unconstrained arrangement pool (with penalties), and the result's
    round_search_budget_reached says how that search ended. False: it visited
    every arrangement and none works, so no schedule can meet the constraints
    with this roster and court count. True: it ran out of budget before it
    could tell, which proves nothing - a longer max runtime searches further,
    unless max_time is already at the widget's ceiling, when the only remedy
    left is a different roster, court count or constraint. Each needs its own
    wording: blaming the constraints for a search that merely ran out of time
    sends the user off loosening constraints that may be fine.
    """
    where = "the replanned rounds" if replan else "this schedule"
    prefix = f"⚠️ {violations} game(s) in {where} violate a requested 'do not pair' or 'do not oppose' constraint"
    scope = "with the remaining players and courts" if replan else "with these players and courts"
    remedy = "loosen a constraint or change the roster or court count"
    if not budget_reached:
        return (
            f"{prefix} - the round search checked every arrangement and none meets every constraint {scope}; {remedy}."
        )
    ran_out = f"{prefix} - the round search ran out of time before it could tell whether they can all be met {scope}"
    if max_time is not None and max_time >= MAX_GENERATION_TIME_SECONDS:
        return f"{ran_out}, even at the longest max runtime; {remedy}."
    if max_time is not None and max_time < ROUND_SEARCH_MIN_SECONDS:
        return (
            f"{ran_out}; a longer max runtime ({ROUND_SEARCH_MIN_SECONDS} s or more, up to "
            f"{MAX_GENERATION_TIME_SECONDS} s) searches further."
        )
    return f"{ran_out}; a longer max runtime (up to {MAX_GENERATION_TIME_SECONDS} s) searches further."


# Characters shown with no width at all, so a name holding one looks just like
# the name without it - pasted text often carries them.
_ZERO_WIDTH_CHARACTERS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


def _visible_name(name: str) -> str:
    """A name as it reads: an accent typed as a separate mark composed with its letter (NFC), and
    zero-width characters left out. Only for comparing names - they are kept as typed."""
    return unicodedata.normalize("NFC", name).translate(_ZERO_WIDTH_CHARACTERS)


def find_player_name_issues(players: list[str]) -> list[str]:
    """Detect duplicate and near-duplicate names before they reach the scheduler.

    An exact duplicate makes the GA treat one physical player as filling
    multiple schedule slots, which later surfaces as a misleading "algorithm
    bug" error (impossible player assignments) instead of a clear input
    problem. Case/spacing collisions ("Bob" vs "bob") are flagged separately
    since they are usually typos rather than intentional distinct players, and
    so are names that look identical but are typed differently ("José" with
    its accent typed separately, a pasted zero-width space).
    """
    issues: list[str] = []

    seen: set[str] = set()
    exact_dupes: set[str] = set()
    for name in players:
        if name in seen:
            exact_dupes.add(name)
        seen.add(name)
    for name in sorted(exact_dupes):
        issues.append(f"Duplicate player: {name}")

    normalized_groups: dict[str, list[str]] = {}
    for name in players:
        key = re.sub(r"\s+", " ", _visible_name(name).strip().lower())
        normalized_groups.setdefault(key, []).append(name)
    for variants in normalized_groups.values():
        distinct_variants = sorted(set(variants))
        if len({_visible_name(name) for name in distinct_variants}) < len(distinct_variants):
            issues.append(
                "Names that look the same but are typed with different characters (an accent typed "
                "separately, or an invisible character): " + ", ".join(distinct_variants)
            )
        elif len(distinct_variants) > 1:
            issues.append("Names that look the same but differ in case or spacing: " + ", ".join(distinct_variants))

    return issues


def determine_initial_player_preset(
    session_state: Any,
    config: Mapping[str, Any],
) -> str:
    """Choose a useful default preset for first render without overwriting user intent."""
    preset_names = list(config.get("player_presets", {}).keys())
    preset_options = [*RESERVED_PRESET_NAMES, *preset_names]

    custom_players = session_state.get("custom_players", "")
    has_custom_players = isinstance(custom_players, str) and bool(custom_players.strip())
    selection_initialized = bool(session_state.get("selected_player_preset_initialized", False))
    selected_preset = session_state.get("selected_player_preset")
    if (
        selection_initialized
        and isinstance(selected_preset, str)
        and selected_preset in preset_options
        and (selected_preset != "Custom" or has_custom_players)
    ):
        return selected_preset

    if has_custom_players:
        return "Custom"

    config_manager = session_state.get("config_manager")
    if config_manager is not None:
        default_names = config_manager.load_default_names()
        if isinstance(default_names, list) and default_names:
            return "Default Names"

    if preset_names:
        return preset_names[0]

    return "Custom"


def build_player_preset_options(
    session_state: Any,
    config: Mapping[str, Any],
) -> list[str]:
    """Return preset options ordered so the intended selection is first."""
    preset_names = list(config.get("player_presets", {}).keys())
    # A preset saved under a reserved name (before those were refused) is
    # listed once: the reserved choice is what picking it loads.
    preset_options = list(dict.fromkeys([*RESERVED_PRESET_NAMES, *preset_names]))
    initial_preset = determine_initial_player_preset(session_state, config)
    return [initial_preset, *[option for option in preset_options if option != initial_preset]]


def resolve_players_text_for_preset(
    session_state: Any,
    config: Mapping[str, Any],
    preset: str,
) -> str:
    """Resolve the player text to display for the current or fallback preset."""

    def load_default_names_text() -> str:
        config_manager = session_state.get("config_manager")
        if config_manager is None:
            return ""

        default_names = config_manager.load_default_names()
        if isinstance(default_names, list):
            return "\n".join(default_names)

        return ""

    if preset == "Default Names":
        return load_default_names_text()

    if preset != "Custom":
        preset_players = config.get("player_presets", {}).get(preset, [])
        if isinstance(preset_players, list):
            return "\n".join(preset_players)
        return ""

    custom_players = session_state.get("custom_players", "")
    if isinstance(custom_players, str) and custom_players.strip():
        return custom_players

    fallback_preset = determine_initial_player_preset(session_state, config)
    if fallback_preset == "Default Names":
        return load_default_names_text()

    if fallback_preset != "Custom":
        fallback_players = config.get("player_presets", {}).get(fallback_preset, [])
        if isinstance(fallback_players, list):
            return "\n".join(fallback_players)

    return ""


def build_constraint_pair_options(players: list[str], separator: str) -> dict[str, list[str]]:
    """Map each constraint multiselect label to its pair, in roster order.

    Callers map a selected label back to its pair through this dict rather
    than re-splitting the label on ``separator``, so a player name that
    itself contains " & " or " vs " still round-trips to the right pair.
    Labels are unique: names like "Tom & Jerry" + "Spike" and "Tom" +
    "Jerry & Spike" would both read "Tom & Jerry & Spike", so colliding
    labels quote each name instead.
    """
    pairs = [(p1, p2) for i, p1 in enumerate(players) for p2 in players[i + 1 :]]
    plain_counts = Counter(f"{p1}{separator}{p2}" for p1, p2 in pairs)
    options: dict[str, list[str]] = {}
    for index, (p1, p2) in enumerate(pairs):
        label = f"{p1}{separator}{p2}"
        if plain_counts[label] > 1:
            label = f'"{p1}"{separator}"{p2}"'
        if label in options:
            label = f"{label} (#{index + 1})"
        options[label] = [p1, p2]
    return options


def _fold_name(name: str) -> str:
    return re.sub(r"\s+", "", _visible_name(name)).casefold()


def find_constraint_name_issues(config: Mapping[str, Any], players: Sequence[str]) -> list[str]:
    """Flag constraint names that miss a roster player only by case or spacing.

    Constraints apply by exact name (constraint_pairs_for_roster), so a pair
    saved as "alice, bob" never constrains Alice and Bob - and since a pair
    naming someone off the roster is dropped without comment (they may just
    be away this week), nothing else would say so. Closer guesses are left
    alone: "Dan" may be a typo for Dana, or a different player who is away.
    """
    roster = set(players)
    by_folded: dict[str, str] = {}
    for name in players:
        by_folded.setdefault(_fold_name(name), name)

    issues: list[str] = []
    for pair in [*config["constraints"]["do_not_pair"], *config["constraints"]["do_not_oppose"]]:
        for name in pair:
            suggestion = None if name in roster else by_folded.get(_fold_name(name))
            if suggestion is not None:
                issues.append(
                    f"No player is named exactly '{name}' - did you mean '{suggestion}'? "
                    f"Constraints naming '{name}' won't be applied."
                )
    return list(dict.fromkeys(issues))


def build_constraint_multiselect_defaults(
    constraint_pairs: list[list[str]],
    players: list[str],
    separator: str,
    pair_options: Mapping[str, list[str]] | None = None,
) -> list[str]:
    """Format saved constraints into multiselect labels for the current player list.

    Labels come from the same mapping as the options, whatever order the
    pair was saved in: Streamlit raises if a default is not one of the
    options, so a pair saved as [Bob, Alice] (as quick-add, the Configuration
    tab or an import can store it) must still render as "Alice & Bob" when
    Alice is listed first. A caller that already has
    build_constraint_pair_options(players, separator) passes it as
    ``pair_options`` rather than having every pair built again.
    """
    if pair_options is None:
        pair_options = build_constraint_pair_options(players, separator)
    label_for_pair = {frozenset(pair): label for label, pair in pair_options.items()}
    labels = [
        label_for_pair[frozenset(pair)]
        for pair in constraint_pairs_for_roster(constraint_pairs, players)
        if frozenset(pair) in label_for_pair
    ]
    return list(dict.fromkeys(labels))


def merge_roster_constraint_selection(
    saved_pairs: Sequence[Sequence[str]],
    selected_labels: Sequence[str],
    label_to_pair: Mapping[str, list[str]],
    players: Sequence[str],
) -> list[list[str]]:
    """Combine the constraint multiselect's selection with the saved pairs it can't show.

    The multiselect only offers pairs where both players are on the current
    roster, so its value says nothing about any other saved constraint:
    those are kept unchanged. Saved pairs inside the roster are kept only
    while still selected, and newly selected pairs are appended. Saved order
    is preserved so an unchanged selection writes back an identical list.
    """
    roster = set(players)
    selected_pairs = [label_to_pair[label] for label in selected_labels if label in label_to_pair]
    selected_keys = {frozenset(pair) for pair in selected_pairs}

    merged: list[list[str]] = []
    kept_keys: set[frozenset[str]] = set()
    for pair in saved_pairs:
        if pair_in_roster(pair, roster):
            key = frozenset(pair)
            if key not in selected_keys or key in kept_keys:
                continue
            kept_keys.add(key)
        merged.append(list(pair))

    for pair in selected_pairs:
        key = frozenset(pair)
        if key not in kept_keys:
            kept_keys.add(key)
            merged.append(pair)
    return merged


def add_quick_constraint(
    config: Mapping[str, Any],
    player_a: str,
    player_b: str,
    constraint_key: str,
) -> tuple[bool, str]:
    """Add a single pair constraint to config in place, avoiding duplicates.

    The pair multiselect above scales as O(n^2) options (a 20-player group
    already means 190 entries to scroll through to find one pair). This
    lets a user find a specific pair by name via two searchable selectboxes
    instead. Returns (added, message) for the caller to surface as UI feedback.
    """
    if player_a == player_b:
        return False, "Pick two different players."

    existing_pairs = config["constraints"][constraint_key]
    already_present = any({player_a, player_b} == set(pair) for pair in existing_pairs if len(pair) == 2)
    if already_present:
        return False, f"{player_a} and {player_b} already have that constraint."

    existing_pairs.append([player_a, player_b])
    return True, f"Added: {player_a} & {player_b}"


def get_constraint_widget_keys(session_state: Any) -> tuple[str, str]:
    """Return the current multiselect widget keys for constraint inputs."""
    version = int(session_state.get("_constraint_widget_version", 0))
    return f"do_not_pair_{version}", f"do_not_oppose_{version}"


def sync_constraint_widget_state(
    session_state: Any,
    players: list[str],
    config: Mapping[str, Any],
    pair_options: Mapping[str, list[str]] | None = None,
    oppose_options: Mapping[str, list[str]] | None = None,
) -> None:
    """Seed constraint widget state from saved config when the UI needs a resync.

    Besides an explicit request (a Configuration-tab edit or save, an import,
    quick-add), a roster change resyncs too: the multiselects keep their old
    selection while it is still a valid subset of the new options, so a
    returning player's saved pairs would never appear selected - and the
    write-back would then treat them as deselected and drop them. Nothing is
    lost by resyncing, since every run already merges the current selection
    into config. ``pair_options`` and ``oppose_options`` are the " & " and
    " vs " build_constraint_pair_options results, when the caller has them.
    """
    should_sync_from_config = bool(session_state.pop("_sync_main_constraints_from_config", False))
    roster_signature = tuple(players)
    previous_roster = session_state.get("_constraint_widget_roster")
    if previous_roster is not None and tuple(previous_roster) != roster_signature:
        should_sync_from_config = True
    session_state["_constraint_widget_roster"] = roster_signature
    do_not_pair_key, do_not_oppose_key = get_constraint_widget_keys(session_state)
    if pair_options is None:
        pair_options = build_constraint_pair_options(players, " & ")
    if oppose_options is None:
        oppose_options = build_constraint_pair_options(players, " vs ")

    current_pair_selection = session_state.get(do_not_pair_key)
    if (
        should_sync_from_config
        or not isinstance(current_pair_selection, list)
        or any(selection not in pair_options for selection in current_pair_selection)
    ):
        session_state[do_not_pair_key] = build_constraint_multiselect_defaults(
            config["constraints"]["do_not_pair"],
            players,
            " & ",
            pair_options,
        )

    current_oppose_selection = session_state.get(do_not_oppose_key)
    if (
        should_sync_from_config
        or not isinstance(current_oppose_selection, list)
        or any(selection not in oppose_options for selection in current_oppose_selection)
    ):
        session_state[do_not_oppose_key] = build_constraint_multiselect_defaults(
            config["constraints"]["do_not_oppose"],
            players,
            " vs ",
            oppose_options,
        )


def history_unavailable_message(exc: BaseException) -> str:
    """Explain a schedule history database that couldn't be opened, and what still works."""
    if isinstance(exc, sqlite3.DatabaseError) and not isinstance(exc, sqlite3.OperationalError):
        cause = (
            "The database file looks damaged. It has not been changed: restore it from a backup, or move it "
            "aside to start a new, empty history."
        )
    else:
        cause = (
            "Another program may be using the database file, or it may be read-only. The app tries again "
            "each time the page updates."
        )
    unavailable = "History, score entry and the Leaderboard are" if features.SCORE_TRACKING else "History is"
    return (
        f"❌ Couldn't open the schedule history ({exc}). {cause} Until then you can still generate "
        f"schedules, but {unavailable} unavailable."
    )


def unsaved_schedule(session_state: Any) -> dict | None:
    """The failed history save of the schedule on screen, or None if it is saved (or none is shown).

    Matched to the schedule object itself, so it never applies to a schedule
    generated or loaded since.
    """
    unsaved = session_state.get(UNSAVED_SCHEDULE_KEY)
    if (
        unsaved is None
        or session_state.get("current_schedule_id") is not None
        or unsaved["schedule"] is not session_state.get("current_schedule")
    ):
        return None
    return unsaved


def save_schedule_to_history(session_state: Any, schedule, players, settings, algorithm) -> int | None:
    """Save the schedule on screen to history and point score entry at it.

    Returns the new history id, or None if it couldn't be saved (including
    when the history database couldn't be opened at all). A failed save is
    kept under UNSAVED_SCHEDULE_KEY so it can be retried: until it is saved,
    the schedule's scores can't be entered.
    """
    history_manager = session_state.get("history_manager")
    schedule_id = (
        history_manager.save_schedule(schedule, players, settings, algorithm=algorithm)
        if history_manager is not None
        else None
    )
    session_state.current_schedule_id = schedule_id
    if schedule_id is None:
        session_state[UNSAVED_SCHEDULE_KEY] = {
            "schedule": schedule,
            "players": players,
            "settings": settings,
            "algorithm": algorithm,
        }
    else:
        session_state.pop(UNSAVED_SCHEDULE_KEY, None)
    return schedule_id


# Warnings and errors left for the next run, as (level, text) pairs: see queue_status_notice.
STATUS_NOTICES_KEY = "_status_notices"


def queue_status_notice(session_state: Any, level: str, text: str) -> None:
    """Show text as a "warning" or an "error" at the top of the page on the next run.

    For a handler that calls st.rerun(): the rerun clears whatever the
    handler drew itself. These show above the tabs, under the success
    message it leaves in global_status_message.
    """
    session_state.setdefault(STATUS_NOTICES_KEY, []).append((level, text))


def _cloud_backup_before_tabs(st_module) -> None:
    """Restore a fresh start from the encrypted cloud copy, if cloud backup is set up (see app_remote_backup)."""
    try:
        notice = import_module_with_fallback("app_remote_backup").before_tabs(st_module)
        if notice:
            queue_status_notice(st_module.session_state, "warning", notice)
    except Exception:
        logger.exception("Cloud backup: the start-of-run step failed")


def _cloud_backup_after_tabs(st_module) -> None:
    """Ask for a cloud backup if this run changed the saved data; it is sent in the background."""
    try:
        import_module_with_fallback("app_remote_backup").after_tabs(st_module)
    except Exception:
        logger.exception("Cloud backup: the end-of-run step failed")


def render_enhanced_scheduler_page(
    st_module,
    history_manager_cls,
    config_manager_cls,
    logout_fn,
    main_scheduler_tab_fn,
    analytics_tab_fn,
    history_tab_fn,
    configuration_tab_fn,
    player_management_tab_fn,
    stress_test_tab_fn=None,
    leaderboard_tab_fn=None,
    skill_manager_cls=None,
    elo_manager_cls=None,
):
    """Render the top-level scheduler page with tabs."""
    history_error = None
    if "history_manager" not in st_module.session_state:
        try:
            st_module.session_state.history_manager = history_manager_cls()
        except (sqlite3.Error, OSError) as exc:
            # Scheduling doesn't need the history, so keep the rest of the app
            # usable. Nothing is stored, so the next run tries again - another
            # program's lock on the file is often brief.
            logger.exception("Could not open the schedule history database")
            history_error = exc
    if "config_manager" not in st_module.session_state:
        st_module.session_state.config_manager = config_manager_cls()
    if skill_manager_cls is not None and "skill_manager" not in st_module.session_state:
        st_module.session_state.skill_manager = skill_manager_cls()
    if features.SCORE_TRACKING and elo_manager_cls is not None and "elo_manager" not in st_module.session_state:
        st_module.session_state.elo_manager = elo_manager_cls()
    _cloud_backup_before_tabs(st_module)  # before the configuration is loaded: a restore changes it
    # Notices above the tabs render into slots that exist on every run, even
    # when empty. st.tabs is identified by its position on the page, so a
    # banner that appears on one run and not the next shifts the tabs, which
    # remounts them and snaps the user back to the first tab - e.g. right
    # after "Save Configuration" or "Import CSV" on a later tab.
    repair_notices = st_module.container()
    if history_error is not None:
        repair_notices.error(history_unavailable_message(history_error))
    if "app_config" not in st_module.session_state:
        CONFIG_REPAIR_MESSAGES_KEY = import_module_with_fallback("app_managers").CONFIG_REPAIR_MESSAGES_KEY

        loaded_config = st_module.session_state.config_manager.load_config()
        repair_messages = loaded_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
        st_module.session_state.app_config = loaded_config
        for message in repair_messages or []:
            repair_notices.warning(f"⚠️ {escape_markdown(message)}")

    col1, col2 = st_module.columns([1, 0.2])
    with col1:
        st_module.title("🎾 Pickleball Scheduler")
    with col2:
        # No logout_fn when no password is set: there is no login page to go to.
        if logout_fn is not None and st_module.button("Logout"):
            logout_fn()

    st_module.markdown("*Complete scheduling with analytics and history*")

    status_slot = st_module.empty()
    global_status_message = st_module.session_state.pop("global_status_message", None)
    status_notices = st_module.session_state.pop(STATUS_NOTICES_KEY, None)
    if status_notices:
        status_slot = status_slot.container()  # an empty slot holds just one element
    if isinstance(global_status_message, str) and global_status_message:
        status_slot.success(global_status_message)
    for level, text in status_notices or []:
        (status_slot.error if level == "error" else status_slot.warning)(text)
    interrupted_run_slot = st_module.empty()
    # Drawn before the tabs, so they don't jump while each run draws them.
    interrupted_run = _interrupted_run_marker(st_module.session_state)
    if interrupted_run is not None:
        _render_interrupted_run_notice(st_module, interrupted_run_slot, interrupted_run)

    tab_labels = [
        "🎯 Main Scheduler",
        "📊 Analytics",
        "📅 History",
        "⚙️ Configuration",
        "👥 Player Management",
    ]
    tab_fns = [
        main_scheduler_tab_fn,
        analytics_tab_fn,
        history_tab_fn,
        configuration_tab_fn,
        player_management_tab_fn,
    ]
    if features.SCORE_TRACKING and leaderboard_tab_fn is not None:
        tab_labels.append("🏆 Leaderboard")
        tab_fns.append(leaderboard_tab_fn)
    if stress_test_tab_fn is not None:
        tab_labels.append("🧪 Stress Test")
        tab_fns.append(stress_test_tab_fn)

    tabs = st_module.tabs(tab_labels)

    queued_runs: list = []
    st_module.session_state[QUEUED_SCHEDULER_RUNS_KEY] = queued_runs
    for tab, tab_fn in zip(tabs, tab_fns):
        with tab:
            tab_fn()
    st_module.session_state.pop(QUEUED_SCHEDULER_RUNS_KEY, None)

    # The notice goes once a tab has changed the schedule on screen, or when
    # a Generate or replan starts below and so replaces the interrupted one.
    if interrupted_run is not None and (
        queued_runs or _interrupted_run_marker(st_module.session_state) is not interrupted_run
    ):
        interrupted_run_slot.empty()
    # A save on this run (or on the run a st.rerun cut short) may have replaced
    # a config file it couldn't read; say so, naming the kept copy, in the
    # notice slot above the tabs. A queued run that saves reruns the page, so
    # its notice shows on that rerun.
    for message in st_module.session_state.config_manager.pop_notices():
        repair_notices.warning(f"⚠️ {escape_markdown(message)}")
    for run in queued_runs:
        run()
    _cloud_backup_after_tabs(st_module)


# Held in session_state while a Generate or replan run works. The GA runs on
# the script thread, and Streamlit stops that run - when the user changes a
# widget, clicks a button, presses Stop or the connection drops - by raising a
# BaseException out of the next st call (the progress callback), which none of
# the `except Exception` handlers here see, so nothing gets saved. The marker
# outlives such a run, which lets the next run say what happened.
SCHEDULER_RUN_MARKER_KEY = "_scheduler_run_in_progress"
DISMISS_INTERRUPTED_RUN_KEY = "dismiss_interrupted_run"
# The list render_enhanced_scheduler_page starts queued Generate and replan
# runs from, once every tab is drawn (see _start_after_the_tabs).
QUEUED_SCHEDULER_RUNS_KEY = "_queued_scheduler_runs"
# How the last Generate went, for the rerun that draws its schedule.
GENERATION_OUTCOME_KEY = "_generation_outcome"


def _run_cancel_caption() -> str:
    """What changing the page, pressing Stop or losing the connection does to a Generate or replan run."""
    changes = "a setting, a button, a score" if features.SCORE_TRACKING else "a setting, a button"
    return (
        f"⏳ Changing anything on the page while this runs ({changes}), pressing Stop or "
        "losing the connection cancels it and leaves the schedule as it was."
    )


_INTERRUPTED_RUN_MESSAGES = {
    "generation": (
        "⚠️ The last schedule generation was interrupted before it finished, so no new schedule was saved and "
        "any schedule shown is still the previous one. This happens when something on the page is changed or "
        "clicked, Stop is pressed, or the connection drops while it runs. Click '🎯 Generate Schedule' to run "
        "it again."
    ),
    "replan": (
        "⚠️ The last mid-session replan was interrupted before it finished, so the remaining rounds were not "
        "changed. This happens when something on the page is changed or clicked, Stop is pressed, or the "
        "connection drops while it runs. Click '🔁 Replan remaining rounds' to run it again."
    ),
}


# How many generations (Generate, replan, a Stress Test sweep) are running in
# this server process, across every session. They are threads of one process
# sharing the processor, so with two running each searches about half the
# generations it would alone within its time limit.
_running_generations = 0
_running_generations_lock = threading.Lock()
_SHARED_SERVER_NOTICE = (
    "ℹ️ Another schedule is being generated on this server right now. They share the processor, so this run may "
    "search fewer generations than usual in its time limit."
)


@contextmanager
def counted_generation(st_module) -> Iterator[None]:
    """Count a generation as running while the block runs, first saying so on
    the page if another one already is."""
    global _running_generations
    with _running_generations_lock:
        already_running = _running_generations
        _running_generations += 1
    try:
        if already_running:
            st_module.info(_SHARED_SERVER_NOTICE)
        yield
    finally:
        # Also when Streamlit stops the run part-way: the generation stops with it.
        with _running_generations_lock:
            _running_generations -= 1


def _start_scheduler_run(session_state: Any, kind: str) -> dict:
    marker = {"kind": kind, "schedule": session_state.get("current_schedule")}
    session_state[SCHEDULER_RUN_MARKER_KEY] = marker
    return marker


def _end_scheduler_run(session_state: Any, marker: dict) -> None:
    # Only this run's own marker: a run that ends late must not clear the
    # marker of the run that replaced it.
    if session_state.get(SCHEDULER_RUN_MARKER_KEY) is marker:
        session_state.pop(SCHEDULER_RUN_MARKER_KEY, None)


@contextmanager
def _scheduler_run(session_state: Any, kind: str) -> Iterator[None]:
    """Mark a Generate or replan run in progress, clearing the mark when it
    ends in any way except being interrupted by Streamlit."""
    marker = _start_scheduler_run(session_state, kind)
    try:
        yield
    except Exception:
        _end_scheduler_run(session_state, marker)
        raise
    _end_scheduler_run(session_state, marker)


def _interrupted_run_marker(session_state: Any) -> dict | None:
    """The marker a Generate or replan left when Streamlit interrupted it, kept
    until the schedule on screen changes (a new run, a History load) or the
    user dismisses the notice about it."""
    marker = session_state.get(SCHEDULER_RUN_MARKER_KEY)
    if not isinstance(marker, dict):
        return None
    if session_state.get("current_schedule") is not marker.get("schedule"):
        session_state.pop(SCHEDULER_RUN_MARKER_KEY, None)
        return None
    return marker


def _render_interrupted_run_notice(st_module, slot, marker: dict) -> None:
    notice = slot.container()
    notice.warning(_INTERRUPTED_RUN_MESSAGES[marker["kind"]])
    notice.button(
        "Dismiss",
        key=DISMISS_INTERRUPTED_RUN_KEY,
        on_click=st_module.session_state.pop,
        args=(SCHEDULER_RUN_MARKER_KEY, None),
    )


def _start_after_the_tabs(st_module, slot, run) -> None:
    """Start a Generate or replan run, drawing into slot, once every tab is drawn.

    Streamlit stops a run part-way when the user interacts, presses Stop or
    the connection drops, and the browser then clears everything that run had
    not drawn yet. A long run started inside the first tab took the schedule
    below it, every other tab and any unsaved score entry with it.
    """

    def run_in_slot():
        with slot.container():
            run()

    queued_runs = st_module.session_state.get(QUEUED_SCHEDULER_RUNS_KEY)
    if isinstance(queued_runs, list):
        queued_runs.append(run_in_slot)
    else:  # the tab drawn on its own, outside render_enhanced_scheduler_page
        run_in_slot()


def _render_generation_outcome(st_module, slot, warnings: list) -> None:
    """Show how the last Generate went, on the rerun that draws its schedule."""
    # Left collapsed, the box would hide its warnings behind a bare "Done".
    label = "⚠️ Done - see the warnings below" if warnings else "✅ Done"
    with slot.container(), st_module.status(label, state="complete", expanded=bool(warnings)):
        st_module.success("✅ Schedule generated!")
        for warning in warnings:
            st_module.warning(warning)


def _generation_progress(progress_data, max_time, replay_generations=None) -> tuple[float, str]:
    """The progress bar fraction and status line for one GA progress update.

    A run is bounded by max_time, so its progress is the time used. A replay
    ignores the clock (see generate_schedule), so its progress is the
    generations done out of replay_generations.
    """
    elapsed = float(progress_data.get("elapsed_time", 0.0))
    generation = progress_data.get("generation", 0)
    total_range = progress_data.get("total_range", "?")
    if replay_generations:
        return (
            max(0.0, min(1.0, generation / replay_generations)),
            f"Replaying generation {generation} of {replay_generations} - best range {total_range} - "
            f"{elapsed:.0f}s elapsed",
        )
    budget = float(progress_data.get("max_time") or max_time or 1.0)
    fraction = max(0.0, min(1.0, elapsed / budget)) if budget else 0.0
    return fraction, f"Generation {generation} - best range {total_range} - {elapsed:.0f}s elapsed"


def _show_invalid_schedule(st_module, status_box, schedule_errors: list) -> None:
    """Report a generated schedule that can't be played (a player twice in a round, say) and fail the run."""
    st_module.error("🚨 **CRITICAL ERROR: Invalid schedule generated!**")
    st_module.error("**The algorithm has bugs that generated impossible player assignments!**")
    for error in schedule_errors:
        st_module.error(f"❌ {escape_markdown(error)}")
    st_module.error("**🛑 This indicates serious bugs in the scheduling algorithm!**")
    st_module.info("Please report this bug with the exact player configuration.")
    status_box.update(label="❌ Invalid schedule generated", state="error", expanded=True)


def _handle_generate_schedule_click(
    st_module,
    config,
    players,
    num_rounds,
    courts,
    max_time,
    generation_seed,
    scheduler_cls,
    schedule_analytics_cls,
    validate_schedule_integrity_fn,
    replay_generations=None,
) -> None:
    """Handle the "Generate Schedule" button: build the scheduler,
    generate and validate a schedule, and persist it to session_state/history.

    The run is bounded by max_time ("Max generation time"), not a generation
    count - unless replay_generations is given with generation_seed, which
    replays that earlier run exactly (see generate_schedule).

    It runs once every tab is drawn (see _start_after_the_tabs), so on every
    outcome the page already shows whatever schedule was on screen before -
    a failed attempt must never hide one from a previous successful
    generation. A new schedule is drawn by the rerun that ends a successful
    run, which also shows the outcome kept under GENERATION_OUTCOME_KEY.
    """
    generated = False
    status = st_module.status("Generating schedule...", expanded=True)
    with _scheduler_run(st_module.session_state, "generation"), status as status_box:
        cancel_caption = st_module.empty()
        cancel_caption.caption(_run_cancel_caption())
        progress_bar = st_module.progress(0)
        progress_text = st_module.empty()

        def _on_progress(progress_data, _bar=progress_bar, _text=progress_text, _max_time=max_time):
            try:
                fraction, text = _generation_progress(progress_data, _max_time, replay_generations)
                _bar.progress(fraction)
                _text.text(text)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

        import time as _time

        active_constraints = {
            "do_not_pair": constraint_pairs_for_roster(config["constraints"]["do_not_pair"], players),
            "do_not_oppose": constraint_pairs_for_roster(config["constraints"]["do_not_oppose"], players),
        }
        generation_started_at = _time.time()
        logger.info(
            "Generation start: players=%d rounds=%d courts=%d " "pair_constraints=%d oppose_constraints=%d seed=%s",
            len(players),
            num_rounds,
            courts,
            len(active_constraints["do_not_pair"]),
            len(active_constraints["do_not_oppose"]),
            generation_seed,
        )

        try:
            warnings = []
            if not st_module.session_state.config_manager.save_config(config):
                warnings.append("⚠️ Could not save configuration before generating - continuing anyway.")
                st_module.warning(warnings[-1])

            scheduler = scheduler_cls(
                players=players,
                num_courts=courts,
                num_rounds=num_rounds,
                use_desktop_params=True,
                max_runtime=max_time,
            )

            constraints = active_constraints
            if constraints["do_not_pair"] or constraints["do_not_oppose"]:
                scheduler.add_constraints(
                    pair_constraints=[
                        tuple(pair) if isinstance(pair, list) else pair for pair in constraints["do_not_pair"]
                    ],
                    oppose_constraints=[
                        tuple(pair) if isinstance(pair, list) else pair for pair in constraints["do_not_oppose"]
                    ],
                )

            with counted_generation(st_module):
                result = scheduler.generate_schedule(
                    max_time=max_time,
                    seed=generation_seed,
                    progress_callback=_on_progress,
                    generations=replay_generations,
                )

            if not (isinstance(result, dict) and "schedule" in result):
                logger.warning(
                    "Generation end: status=no_schedule elapsed=%.1fs",
                    _time.time() - generation_started_at,
                )
                st_module.error("❌ Could not generate schedule. Try adjusting parameters.")
                status_box.update(label="❌ Generation failed", state="error", expanded=True)
                return

            schedule_data = result["schedule"]
            schedule_errors = validate_schedule_integrity_fn(schedule_data, players)

            if schedule_errors:
                logger.error(
                    "Generation end: status=invalid_schedule elapsed=%.1fs errors=%d",
                    _time.time() - generation_started_at,
                    len(schedule_errors),
                )
                _show_invalid_schedule(st_module, status_box, schedule_errors)
                return

            progress_bar.progress(1.0)

            constraint_violations = result.get("constraint_violations", 0)
            if constraint_violations:
                warnings.append(
                    constraint_violation_warning(
                        constraint_violations,
                        bool(result.get("round_search_budget_reached", False)),
                        max_time=max_time,
                    )
                )

            metrics = schedule_analytics_cls.calculate_fairness_metrics(
                schedule_data,
                num_players=len(players),
                num_rounds=num_rounds,
                num_courts=courts,
                players=players,
            )

            logger.info(
                "Generation end: status=ok elapsed=%.1fs total_range=%s",
                _time.time() - generation_started_at,
                metrics.get("total_range") if isinstance(metrics, dict) else None,
            )

            compute_round_times = import_module_with_fallback("app_schedule_helpers").compute_round_times

            scheduling_config = config.get("scheduling", {})
            round_times = compute_round_times(
                num_rounds,
                scheduling_config.get("start_time"),
                scheduling_config.get("end_time"),
            )

            st_module.session_state.current_schedule = schedule_data
            st_module.session_state.current_players = players
            st_module.session_state.current_metrics = metrics
            st_module.session_state.current_seed = result.get("seed")
            # With the seed, the generation count replays this exact schedule -
            # unless a repair ran out of time (see generate_schedule).
            st_module.session_state.current_generations = (
                result.get("generations") if result.get("replayable") else None
            )
            st_module.session_state.current_round_times = round_times
            # A fresh generation uses one roster for every round - clear any
            # per-round roster segmentation left over from a previous
            # schedule's mid-session replan.
            st_module.session_state.current_roster_by_round = None
            st_module.session_state[CURRENT_ATTENDANCE_KEY] = None

            settings = {
                "num_rounds": num_rounds,
                "courts": courts,
                "max_time": max_time,
                "constraints": active_constraints,
                "seed": result.get("seed"),
            }
            # generations == 0 means the constructive fast path produced this
            # schedule directly, without running the GA search at all.
            algorithm = "constructive" if result.get("generations") == 0 else "genetic"
            # A failed save is reported (with a retry) above the schedule - see
            # _render_unsaved_schedule_notice.
            save_schedule_to_history(st_module.session_state, schedule_data, players, settings, algorithm)

            st_module.session_state[GENERATION_OUTCOME_KEY] = warnings
            generated = True

        except MemoryError:
            logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
            st_module.error("❌ Ran out of memory while generating the schedule. Try fewer players or rounds.")
            status_box.update(label="❌ Out of memory", state="error", expanded=True)
        except ValueError as exc:
            logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
            st_module.error(f"❌ Invalid input: {exc}. Check your players and constraints.")
            status_box.update(label="❌ Invalid input", state="error", expanded=True)
        except Exception:
            logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
            st_module.error("❌ Schedule generation failed unexpectedly. See the server logs for details.")
            status_box.update(label="❌ Generation error", state="error", expanded=True)
        finally:
            cancel_caption.empty()

    if generated:
        # Every tab was drawn with the previous schedule - redraw them all.
        st_module.rerun()


def _handle_attendance_generate_click(
    st_module,
    config,
    players,
    num_rounds,
    courts,
    max_time,
    generation_seed,
    scheduler_cls,
    validate_schedule_integrity_fn,
    windows,
) -> None:
    """Handle "Generate Schedule" when some players arrive late or leave early.

    The attendance planner (see src/algorithms/attendance_planner.py)
    schedules each stretch of rounds with the same players for exactly those
    players, up to ``courts`` courts, within max_time shared out between the
    stretches. Like _handle_generate_schedule_click it runs once every tab is
    drawn, leaves the schedule on screen alone unless it succeeds, and ends in
    the rerun that draws the new schedule and its outcome.

    What differs is what is kept. Each round is checked against its own
    roster; the fairness metrics and the theoretical optimum assume one
    roster, so current_metrics is left empty (the Main tab shows who was here
    when instead); and the seed and generation count can't replay a plan, so
    none is kept - a seed typed by the user only starts the search.
    """
    generated = False
    status = st_module.status("Generating schedule...", expanded=True)
    with _scheduler_run(st_module.session_state, "generation"), status as status_box:
        cancel_caption = st_module.empty()
        cancel_caption.caption(_run_cancel_caption())
        progress_bar = st_module.progress(0)
        progress_text = st_module.empty()

        def _on_progress(progress_data, _bar=progress_bar, _text=progress_text, _max_time=max_time):
            try:
                fraction, text = _generation_progress(progress_data, _max_time)
                segments = progress_data.get("segments") or 1
                if segments > 1:
                    text = f"Stretch {progress_data.get('segment')} of {segments} - {text}"
                _bar.progress(fraction)
                _text.text(text)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

        active_constraints = {
            "do_not_pair": constraint_pairs_for_roster(config["constraints"]["do_not_pair"], players),
            "do_not_oppose": constraint_pairs_for_roster(config["constraints"]["do_not_oppose"], players),
        }
        generation_started_at = time.time()
        logger.info(
            "Generation start: players=%d rounds=%d courts=%d pair_constraints=%d oppose_constraints=%d seed=%s "
            "attendance=%d",
            len(players),
            num_rounds,
            courts,
            len(active_constraints["do_not_pair"]),
            len(active_constraints["do_not_oppose"]),
            generation_seed,
            len(windows),
        )

        try:
            warnings = []
            if not st_module.session_state.config_manager.save_config(config):
                warnings.append("⚠️ Could not save configuration before generating - continuing anyway.")
                st_module.warning(warnings[-1])

            with counted_generation(st_module):
                result = plan_attendance_schedule(
                    players,
                    num_rounds,
                    courts,
                    windows,
                    scheduler_cls,
                    constraints=active_constraints,
                    max_time=max_time,
                    seed=generation_seed,
                    progress_callback=_on_progress,
                )

            schedule_data = result.get("schedule")
            if not schedule_data:
                logger.warning("Generation end: status=no_schedule elapsed=%.1fs", time.time() - generation_started_at)
                reason = result.get("error") or "Could not generate schedule. Try adjusting parameters."
                st_module.error(f"❌ {escape_markdown(reason)}")
                status_box.update(label="❌ Generation failed", state="error", expanded=True)
                return

            roster_by_round = [list(roster) for roster in result["roster_by_round"]]
            schedule_errors = validate_schedule_integrity_fn(schedule_data, players, roster_by_round=roster_by_round)
            if schedule_errors:
                logger.error(
                    "Generation end: status=invalid_schedule elapsed=%.1fs errors=%d",
                    time.time() - generation_started_at,
                    len(schedule_errors),
                )
                _show_invalid_schedule(st_module, status_box, schedule_errors)
                return

            progress_bar.progress(1.0)

            # Rounds that run fewer courts than asked, for want of players to fill them.
            warnings.extend(f"⚠️ {warning}" for warning in result.get("warnings", []))
            constraint_violations = result.get("constraint_violations", 0)
            if constraint_violations:
                warnings.append(
                    constraint_violation_warning(
                        constraint_violations,
                        bool(result.get("round_search_budget_reached", False)),
                        max_time=max_time,
                    )
                )

            logger.info(
                "Generation end: status=ok elapsed=%.1fs segments=%d",
                time.time() - generation_started_at,
                len(result.get("segments", [])),
            )

            compute_round_times = import_module_with_fallback("app_schedule_helpers").compute_round_times
            scheduling_config = config.get("scheduling", {})
            round_times = compute_round_times(
                num_rounds,
                scheduling_config.get("start_time"),
                scheduling_config.get("end_time"),
            )

            # Everyone who is there for at least one round (everyone listed is).
            attending = [player for player in players if any(player in roster for roster in roster_by_round)]
            attendance = attendance_for_history(windows)
            st_module.session_state.current_schedule = schedule_data
            st_module.session_state.current_players = attending
            st_module.session_state.current_metrics = {}
            st_module.session_state.current_seed = None
            st_module.session_state.current_generations = None
            st_module.session_state.current_round_times = round_times
            st_module.session_state.current_roster_by_round = roster_by_round
            st_module.session_state[CURRENT_ATTENDANCE_KEY] = attendance

            settings = {
                "num_rounds": num_rounds,
                "courts": courts,
                "max_time": max_time,
                "constraints": active_constraints,
                "seed": result.get("seed"),
                "roster_by_round": roster_by_round,
                "attendance": attendance,
            }
            # A failed save is reported (with a retry) above the schedule - see
            # _render_unsaved_schedule_notice.
            save_schedule_to_history(st_module.session_state, schedule_data, attending, settings, "attendance")

            st_module.session_state[GENERATION_OUTCOME_KEY] = warnings
            generated = True

        except AttendanceError as exc:
            logger.warning(
                "Generation end: status=invalid_attendance elapsed=%.1fs", time.time() - generation_started_at
            )
            st_module.error(attendance_error_message(exc))
            status_box.update(label="❌ Invalid arrivals and departures", state="error", expanded=True)
        except MemoryError:
            logger.exception("Generation end: status=exception elapsed=%.1fs", time.time() - generation_started_at)
            st_module.error("❌ Ran out of memory while generating the schedule. Try fewer players or rounds.")
            status_box.update(label="❌ Out of memory", state="error", expanded=True)
        except ValueError as exc:
            logger.exception("Generation end: status=exception elapsed=%.1fs", time.time() - generation_started_at)
            st_module.error(f"❌ Invalid input: {exc}. Check your players and constraints.")
            status_box.update(label="❌ Invalid input", state="error", expanded=True)
        except Exception:
            logger.exception("Generation end: status=exception elapsed=%.1fs", time.time() - generation_started_at)
            st_module.error("❌ Schedule generation failed unexpectedly. See the server logs for details.")
            status_box.update(label="❌ Generation error", state="error", expanded=True)
        finally:
            cancel_caption.empty()

    if generated:
        # Every tab was drawn with the previous schedule - redraw them all.
        st_module.rerun()


def render_main_scheduler_tab(
    st_module,
    datetime_cls,
    json_module,
    scheduler_cls,
    schedule_analytics_cls,
    validate_schedule_integrity_fn,
    display_enhanced_schedule_fn,
    schedule_to_csv_fn,
):
    """Render the main scheduling interface and schedule generation flow."""
    config = normalize_config_constraints(st_module.session_state.app_config)
    st_module.session_state.app_config = config
    # Values queued by this page's button handlers on the previous run, applied
    # here before any of the widgets they target are instantiated.
    apply_deferred_widget_values(st_module.session_state)
    # Before a handler below can st.rerun() and drop the state of the widgets
    # not drawn yet, here and on the later tabs, with any change the user made
    # to them along with the click - though the browser goes on showing it.
    remember_seeded_widget_values(st_module.session_state)
    for key in (COURTS_KEY, MAX_TIME_KEY):
        remember_widget_choice(st_module.session_state, key)
    remember_attendance_choices(st_module.session_state)

    st_module.subheader("👥 Players")

    col1, col2 = st_module.columns([2, 1])

    with col1:
        previous_preset = st_module.session_state.get("selected_player_preset")
        preset_options = build_player_preset_options(st_module.session_state, config)
        # build_player_preset_options reorders its list every render to put the
        # current selection first, purely for display. Without a stable key,
        # Streamlit can't reliably tell "the options were just reordered" apart
        # from "this is a new widget", and falls back to the first entry -
        # which is *last* render's selection, not what the user just clicked.
        # A key makes Streamlit track the actual chosen value directly, so a
        # second (or third...) preset switch keeps taking effect instead of
        # only the first one after a fresh session.
        preset = st_module.selectbox("Choose preset or custom:", preset_options, key=PLAYER_PRESET_SELECT_KEY)
        st_module.session_state.selected_player_preset = preset
        st_module.session_state.selected_player_preset_initialized = True
        default_players = resolve_players_text_for_preset(st_module.session_state, config, preset)

        # Picking a different preset replaces the roster outright, dropping any
        # edits made under the old one (and the old roster the seed below
        # compares against).
        if preset != previous_preset:
            forget_widget_state(st_module.session_state, PLAYERS_INPUT_KEY)
            st_module.session_state[PLAYERS_INPUT_KEY] = default_players
        # A saved preset's roster can also change while it stays selected (saved
        # in Player Management, restored from a backup). Follow it, carrying any
        # names added or removed here by hand over onto the new roster. Custom
        # and Default Names have nothing to follow: under Custom default_players
        # tracks the user's own typing, and the app never writes the default
        # names file (a failed read of it would only empty the box).
        if preset not in RESERVED_PRESET_NAMES:
            seed_widget_from_source(st_module.session_state, PLAYERS_INPUT_KEY, default_players, rebase_line_edits)
        # The box takes no value= (see src/widget_state.py), so its key must
        # be set - an st.rerun() before it renders would have dropped it.
        if PLAYERS_INPUT_KEY not in st_module.session_state:
            st_module.session_state[PLAYERS_INPUT_KEY] = default_players

        players_text = st_module.text_area("Player names (one per line):", height=150, key=PLAYERS_INPUT_KEY)

        if preset == "Custom":
            st_module.session_state.custom_players = players_text

    with col2:
        st_module.markdown("**Quick Actions**")

        new_player = st_module.text_input("Quick add player:", key=QUICK_ADD_KEY)
        preset_name = st_module.text_input("Preset name:", key=PRESET_NAME_KEY)

        if st_module.button("➕ Add Player"):
            if not new_player:
                st_module.warning("⚠️ Type a player name first.")
            else:
                updated_text = add_player_to_text(players_text, new_player)
                if updated_text != players_text:
                    st_module.session_state.custom_players = updated_text
                    set_player_preset(st_module.session_state, "Custom")
                    st_module.session_state.selected_player_preset_initialized = True
                    defer_widget_value(st_module.session_state, PLAYERS_INPUT_KEY, updated_text)
                    defer_widget_value(st_module.session_state, QUICK_ADD_KEY, "")
                    st_module.rerun()
                else:
                    st_module.warning(f"⚠️ '{escape_markdown(new_player.strip())}' is already in the player list.")

        if st_module.button("💾 Save as Preset"):
            if not preset_name.strip():
                st_module.warning("⚠️ Type a preset name first.")
            else:
                preset_players = parse_players_text(players_text)
                stripped_name = preset_name.strip()
                if is_reserved_preset_name(stripped_name):
                    st_module.error(reserved_preset_name_message(stripped_name))
                elif not preset_players:
                    st_module.warning("⚠️ Add at least one player before saving a preset.")
                elif len(preset_players) > MAX_PLAYERS:
                    st_module.error(
                        f"❌ {too_many_players_message(len(preset_players))} Remove some names before saving "
                        "the preset."
                    )
                elif stripped_name in config["player_presets"]:
                    st_module.error(
                        f"❌ A preset named '{escape_markdown(stripped_name)}' already exists. "
                        "Use a different name, or edit it from the Player Management tab."
                    )
                elif import_module_with_fallback("app_tabs").preset_saved_elsewhere(st_module, stripped_name):
                    st_module.error(
                        f"❌ A preset named '{escape_markdown(stripped_name)}' already exists - it was saved "
                        "from another tab or device. Reload the page to see it, or use a different name."
                    )
                else:
                    # Save a copy and adopt it only on success: a failed save used to
                    # leave the preset in memory, blocking a retry ("already exists")
                    # and letting the next unrelated save persist it anyway.
                    save_presets_change = import_module_with_fallback("app_tabs").save_presets_change
                    if save_presets_change(
                        st_module, lambda presets: presets.__setitem__(stripped_name, preset_players)
                    ):
                        config = st_module.session_state.app_config
                        st_module.success(f"Saved preset: {escape_markdown(stripped_name)}")
                    else:
                        st_module.error(f"❌ Failed to save preset: {escape_markdown(stripped_name)}")

    players = parse_players_text(players_text)

    def render_schedule_without_settings() -> None:
        # A roster this tab can't schedule skips everything built from it, but
        # the current schedule keeps showing - it comes from session_state, not
        # from this text box - and so do its downloads and replan (whose state
        # would be lost while not drawn). Its replan gets the time chosen under
        # 'Max generation time', which isn't drawn, else the suggested one.
        max_time = remember_widget_choice(st_module.session_state, MAX_TIME_KEY)
        if max_time is None:
            max_time = suggest_max_generation_time(len(st_module.session_state.get("current_players") or []), 0)
        _render_persistent_schedule(
            st_module,
            datetime_cls,
            json_module,
            scheduler_cls,
            schedule_analytics_cls,
            validate_schedule_integrity_fn,
            display_enhanced_schedule_fn,
            schedule_to_csv_fn,
            config,
            max_time,
        )

    if len(players) > MAX_PLAYERS:
        st_module.error(f"❌ {too_many_players_message(len(players))} Remove some names to continue.")
        render_schedule_without_settings()
        return

    if players:
        st_module.markdown("**Quick Remove Players:**")
        cols = st_module.columns(min(len(players), 4))
        occurrences: Counter[str] = Counter()
        for i, player in enumerate(players):
            occurrences[player] += 1
            with cols[i % 4]:
                # Keyed by name, not position: a click that arrives with an edit
                # to the box (or after it) must remove the player it showed, not
                # whoever has moved into that spot - and none if they're gone.
                remove_key = f"{QUICK_REMOVE_KEY_PREFIX}{occurrences[player]}::{player}"
                if st_module.button(f"❌ {escape_markdown(player)}", key=remove_key):
                    updated_text = remove_player_from_text(players_text, player)
                    st_module.session_state.custom_players = updated_text
                    set_player_preset(st_module.session_state, "Custom")
                    st_module.session_state.selected_player_preset_initialized = True
                    defer_widget_value(st_module.session_state, PLAYERS_INPUT_KEY, updated_text)
                    st_module.rerun()

    if not players and not st_module.session_state.get("current_schedule"):
        st_module.info("👋 New here? Try it out instantly with sample players.")
        if st_module.button("✨ Try with 8 sample players"):
            sample_players_text = "\n".join(SAMPLE_PLAYER_NAMES)
            st_module.session_state.custom_players = sample_players_text
            set_player_preset(st_module.session_state, "Custom")
            st_module.session_state.selected_player_preset_initialized = True
            defer_widget_value(st_module.session_state, PLAYERS_INPUT_KEY, sample_players_text)
            st_module.rerun()

    if len(players) < 4:
        st_module.warning("⚠️ Need at least 4 players")
        render_schedule_without_settings()
        return

    name_issues = find_player_name_issues(players)
    if name_issues:
        for issue in name_issues:
            st_module.error(f"❌ {escape_markdown(issue)}")
        render_schedule_without_settings()
        return

    st_module.info(f"📊 {len(players)} players")

    st_module.subheader("🚫 Player Constraints")

    col1, col2 = st_module.columns(2)
    # Every possible pair, built once per run and shared by everything below.
    pair_options = build_constraint_pair_options(players, " & ")
    oppose_options = build_constraint_pair_options(players, " vs ")
    sync_constraint_widget_state(st_module.session_state, players, config, pair_options, oppose_options)
    do_not_pair_key, do_not_oppose_key = get_constraint_widget_keys(st_module.session_state)
    preserve_saved_constraints = bool(st_module.session_state.pop("_preserve_saved_constraints_from_config", False))
    default_do_not_pair = build_constraint_multiselect_defaults(
        config["constraints"]["do_not_pair"],
        players,
        " & ",
        pair_options,
    )
    default_do_not_oppose = build_constraint_multiselect_defaults(
        config["constraints"]["do_not_oppose"],
        players,
        " vs ",
        oppose_options,
    )

    # The multiselects can only show pairs inside the current roster, so
    # their selection is merged into the saved lists rather than replacing
    # them - otherwise every constraint involving someone not on today's
    # roster would be dropped here and then written to disk by the next save.
    # Their selection is seeded by sync_constraint_widget_state, so they take
    # no default= (see src/widget_state.py).
    with col1:
        st_module.markdown("**Who should NOT play together:**")
        do_not_pair = st_module.multiselect(
            "Select player pairs that should not be on the same team:",
            options=list(pair_options),
            key=do_not_pair_key,
        )

        if preserve_saved_constraints and not do_not_pair and default_do_not_pair:
            do_not_pair = default_do_not_pair
        config["constraints"]["do_not_pair"] = merge_roster_constraint_selection(
            config["constraints"]["do_not_pair"], do_not_pair, pair_options, players
        )

    with col2:
        st_module.markdown("**Who should NOT play against each other:**")
        do_not_oppose = st_module.multiselect(
            "Select player pairs that should not be opponents:",
            options=list(oppose_options),
            key=do_not_oppose_key,
        )

        if preserve_saved_constraints and not do_not_oppose and default_do_not_oppose:
            do_not_oppose = default_do_not_oppose
        config["constraints"]["do_not_oppose"] = merge_roster_constraint_selection(
            config["constraints"]["do_not_oppose"], do_not_oppose, oppose_options, players
        )

    active_pair_constraints = constraint_pairs_for_roster(config["constraints"]["do_not_pair"], players)
    active_oppose_constraints = constraint_pairs_for_roster(config["constraints"]["do_not_oppose"], players)
    total_constraints = len(active_pair_constraints) + len(active_oppose_constraints)
    if total_constraints > 0:
        st_module.info(f"🎯 Active constraints: {total_constraints} total")
    for issue in find_constraint_name_issues(config, players):
        st_module.warning(f"⚠️ {escape_markdown(issue)}")

    with st_module.expander("🔎 Find a specific pair instead of scrolling"):
        quick_col1, quick_col2, quick_col3, quick_col4 = st_module.columns([2, 2, 2, 1])
        with quick_col1:
            quick_player_a = st_module.selectbox("Player A", options=players, key="quick_constraint_player_a")
        with quick_col2:
            # Default to a different player than Player A's default (index 0)
            # so the two dropdowns don't start identical, guaranteeing a
            # "same player twice" rejection on first use.
            quick_player_b = st_module.selectbox(
                "Player B",
                options=players,
                index=1 if len(players) > 1 else 0,
                key="quick_constraint_player_b",
            )
        with quick_col3:
            quick_constraint_type = st_module.selectbox(
                "Constraint",
                options=["Should not play together", "Should not oppose each other"],
                key="quick_constraint_type",
            )
        with quick_col4:
            st_module.markdown("&nbsp;")
            if st_module.button("➕ Add"):
                constraint_key = (
                    "do_not_pair" if quick_constraint_type == "Should not play together" else "do_not_oppose"
                )
                added, message = add_quick_constraint(config, quick_player_a, quick_player_b, constraint_key)
                if added:
                    if st_module.session_state.config_manager.save_config(config):
                        st_module.session_state.global_status_message = f"✅ {escape_markdown(message)}"
                    else:
                        # Kept for this session, like the pairs ticked above, which Generate saves.
                        queue_status_notice(
                            st_module.session_state,
                            "error",
                            f"❌ {escape_markdown(message)}, but the configuration couldn't be saved, so the "
                            "constraint applies to this session only until a later save works (Generate saves it "
                            "too).",
                        )
                    st_module.session_state.app_config = config
                    st_module.session_state._constraint_widget_version = (
                        int(st_module.session_state.get("_constraint_widget_version", 0)) + 1
                    )
                    st_module.session_state._sync_main_constraints_from_config = True
                    st_module.session_state._preserve_saved_constraints_from_config = True
                    st_module.rerun()
                else:
                    st_module.warning(escape_markdown(message))

    st_module.subheader("⚙️ Schedule Settings")
    col1, col2, col3 = st_module.columns(3)

    with col1:
        # Keyed so History > Load can set it to the loaded schedule's round
        # count; re-seeded when the configured default changes.
        seed_widget_from_source(st_module.session_state, NUM_ROUNDS_KEY, config["scheduling"]["default_rounds"])
        num_rounds = st_module.number_input("Number of rounds:", min_value=1, max_value=20, key=NUM_ROUNDS_KEY)

    # Courts and time follow their suggested values (all the courts this
    # roster fills, a time for its size and constraints) until the user picks
    # their own, which is then kept as the roster changes: these widgets used
    # to start over at the suggestion, and Generate used it, on every change
    # to the roster or constraints, and after Quick Add or Quick Remove.
    with col2:
        max_courts = max(1, len(players) // 4)
        seed_widget_choice(st_module.session_state, COURTS_KEY, max_courts, 1, max_courts)
        courts = st_module.number_input("Courts available:", min_value=1, max_value=max_courts, key=COURTS_KEY)

    with col3:
        suggested_max_time = suggest_max_generation_time(len(players), total_constraints)
        seed_widget_choice(st_module.session_state, MAX_TIME_KEY, suggested_max_time, 10, MAX_GENERATION_TIME_SECONDS)
        max_time = st_module.number_input(
            "Max generation time (seconds):",
            min_value=10,
            max_value=MAX_GENERATION_TIME_SECONDS,
            key=MAX_TIME_KEY,
            help=(
                f"{suggested_max_time} seconds is suggested for {len(players)} players and {total_constraints} "
                "constraint(s) - harder setups (more players, more do-not-pair/oppose constraints) suggest more "
                "time. Adjust as needed."
            ),
        )

    attendance = render_attendance_controls(st_module, players, num_rounds, courts)

    # With players arriving late or leaving early the roster differs from round
    # to round, which the single-roster estimates know nothing about.
    feasibility_notes = (
        [QUALITY_NOTE]
        if attendance.active
        else build_feasibility_notes(len(players), courts, num_rounds, total_constraints)
    )
    # build_feasibility_notes exists specifically so a user isn't surprised
    # after waiting up to the full max-time budget for a result that was
    # never going to reach range 0 - defeats the purpose if the one note
    # that matters is hidden behind a collapsed, neutrally-labeled expander
    # by default. Auto-expand only when there's actually a warning to see.
    has_feasibility_warning = any(note.startswith("⚠️") for note in feasibility_notes)
    with st_module.expander("📐 Expected quality for this configuration", expanded=has_feasibility_warning):
        for note in feasibility_notes:
            st_module.write(note)

    for warning_text in constraint_pressure_warnings(
        players,
        active_pair_constraints,
        active_oppose_constraints,
    ):
        st_module.warning(f"⚠️ {escape_markdown(warning_text)}")

    with st_module.expander("🎲 Advanced: reproducible generation"):
        seed_text = st_module.text_input(
            "Random seed (optional):",
            key="generation_seed",
            help=(
                "Reuse the seed shown under a generated schedule. With its generation count too, that run "
                "is replayed exactly. A seed alone sets where the search starts, but the search stops on "
                "Max generation time, so on a busier computer it can end with a different schedule."
            ),
        )
        generations_text = st_module.text_input(
            "Generations (optional):",
            key="generation_count",
            help=(
                "The generation count shown with the seed. The run then does exactly that many "
                "generations, however long they take, instead of stopping on Max generation time."
            ),
        )
        if attendance.active:
            st_module.caption(
                "Arrivals and departures are set: a seed here only starts the search for each stretch of rounds, "
                "Generations is ignored, and exact replay is not guaranteed."
            )
    generation_seed = None
    seed_text = (seed_text or "").strip()
    if seed_text:
        try:
            generation_seed = int(seed_text)
        except ValueError:
            st_module.warning("Seed must be a whole number - ignoring it for this run.")
    replay_generations = None
    generations_text = (generations_text or "").strip()
    if generations_text:
        # Parsed like the seed: str.isdigit() also accepts characters such as
        # "²" that int() rejects.
        try:
            parsed_generations = int(generations_text)
        except ValueError:
            parsed_generations = 0
        if parsed_generations < 1:
            st_module.warning("Generations must be a whole number of at least 1 - ignoring it for this run.")
        elif generation_seed is None:
            st_module.warning("Generations only replay a run together with its seed - ignoring them for this run.")
        elif attendance.active:
            st_module.info(
                "🎲 Arrivals and departures are set, so Generations is ignored for this run: a schedule planned "
                "around them can't be replayed from a seed and a generation count."
            )
        else:
            replay_generations = parsed_generations
            # Both inputs keep their values, so say so wherever Generate is:
            # a leftover count would otherwise silently replace the time limit.
            st_module.info(
                f"🎲 Generate will replay seed {generation_seed} for exactly {replay_generations} generations, "
                "however long they take - Max generation time doesn't apply. Clear Generations under "
                "'🎲 Advanced: reproducible generation' to search normally."
            )

    if st_module.session_state.get("current_schedule"):
        st_module.caption(
            "ℹ️ You already have a schedule below. Generating again replaces it entirely, "
            "including any rounds already played - use 'Mid-session replan' instead if you just "
            "want to adjust the remaining rounds."
        )

    generate_clicked = st_module.button("🎯 Generate Schedule", type="primary")
    # On every run, so the schedule below never shifts: a running Generate
    # draws here, and so does its outcome on the rerun that follows it.
    generation_slot = st_module.empty()
    if generate_clicked and attendance.error:
        # Refused before anything runs: the same message the Arrivals and
        # departures control shows, and the genetic algorithm never starts.
        generation_slot.error(attendance.error)
    elif generate_clicked and attendance.active:
        _start_after_the_tabs(
            st_module,
            generation_slot,
            lambda: _handle_attendance_generate_click(
                st_module,
                config,
                players,
                num_rounds,
                courts,
                max_time,
                generation_seed,
                scheduler_cls,
                validate_schedule_integrity_fn,
                attendance.windows,
            ),
        )
    elif generate_clicked:
        _start_after_the_tabs(
            st_module,
            generation_slot,
            lambda: _handle_generate_schedule_click(
                st_module,
                config,
                players,
                num_rounds,
                courts,
                max_time,
                generation_seed,
                scheduler_cls,
                schedule_analytics_cls,
                validate_schedule_integrity_fn,
                replay_generations,
            ),
        )
    else:
        generation_outcome = st_module.session_state.pop(GENERATION_OUTCOME_KEY, None)
        if isinstance(generation_outcome, list):
            _render_generation_outcome(st_module, generation_slot, generation_outcome)

    _render_persistent_schedule(
        st_module,
        datetime_cls,
        json_module,
        scheduler_cls,
        schedule_analytics_cls,
        validate_schedule_integrity_fn,
        display_enhanced_schedule_fn,
        schedule_to_csv_fn,
        config,
        max_time,
    )


def _render_persistent_schedule(
    st_module,
    datetime_cls,
    json_module,
    scheduler_cls,
    schedule_analytics_cls,
    validate_schedule_integrity_fn,
    display_enhanced_schedule_fn,
    schedule_to_csv_fn,
    config,
    max_time,
):
    """Re-render the last generated schedule from session_state on every run.

    Streamlit reruns the whole script on any widget interaction, so without this
    the schedule would only ever appear for the single run where Generate was
    clicked and vanish the instant the user touched anything else.
    """
    schedule_data = st_module.session_state.get("current_schedule")
    if not schedule_data:
        return

    players = st_module.session_state.get("current_players", [])
    metrics = st_module.session_state.get("current_metrics", {})
    seed = st_module.session_state.get("current_seed")
    generations = st_module.session_state.get("current_generations")
    round_times = st_module.session_state.get("current_round_times") or None
    roster_by_round = st_module.session_state.get("current_roster_by_round")

    if seed is not None and generations == 0:
        # The instant constructive design: no search, so the seed alone replays it.
        st_module.caption(
            f"🎲 Seed: {seed} - with the same players and settings, enter it under '🎲 Advanced: reproducible "
            "generation' to get this exact schedule again."
        )
    elif seed is not None and generations is not None:
        st_module.caption(
            f"🎲 Seed: {seed} · Generations: {generations} - with the same players and settings, enter both "
            "under '🎲 Advanced: reproducible generation' to get this exact schedule again."
        )
    elif seed is not None:
        st_module.caption(
            f"🎲 Seed: {seed} - the search stops on a time limit, so the same seed can end with a different "
            "schedule, especially on a busier computer."
        )

    _render_unsaved_schedule_notice(st_module)
    _render_metric_summary(st_module, metrics, players)
    _render_optimality_status(st_module, metrics, players)
    if st_module.session_state.get(CURRENT_ATTENDANCE_KEY):
        render_attendance_summary(st_module, schedule_data, roster_by_round, players)

    display_enhanced_schedule_fn(schedule_data, players, round_times=round_times, roster_by_round=roster_by_round)

    # display_enhanced_schedule_fn already showed the critical-error banner
    # above for an invalid schedule - exports must not offer to download
    # something the app has just told the user not to use. It checks each round
    # against its own roster, so this must too.
    if roster_by_round:
        schedule_errors = validate_schedule_integrity_fn(schedule_data, players, roster_by_round=roster_by_round)
    else:
        schedule_errors = validate_schedule_integrity_fn(schedule_data, players)
    if not schedule_errors:
        _render_download_buttons(
            st_module,
            datetime_cls,
            json_module,
            schedule_analytics_cls,
            schedule_data,
            players,
            round_times,
            schedule_to_csv_fn,
            roster_by_round,
        )

    _render_mid_session_replan(st_module, scheduler_cls, schedule_data, players, config, max_time)


def _render_unsaved_schedule_notice(st_module):
    """Offer to retry a failed history save of the schedule on screen - until it
    is saved, it can't be loaded again (nor, with score tracking on, can its scores be entered)."""
    unsaved = unsaved_schedule(st_module.session_state)
    if unsaved is None:
        return
    consequence = (
        "its scores can't be entered and it can't be loaded again later"
        if features.SCORE_TRACKING
        else "it can't be loaded again later"
    )
    st_module.warning(f"⚠️ This schedule isn't saved to history yet, so {consequence}.")
    if st_module.button("💾 Retry saving to history"):
        if save_schedule_to_history(st_module.session_state, **unsaved) is None:
            st_module.error(
                "❌ Still couldn't save this schedule to history. If this keeps happening, the history "
                "database may be damaged or read-only - see the server logs."
            )
        else:
            st_module.session_state.global_status_message = "✅ Schedule saved to history."
            st_module.rerun()


def _scored_rounds_message(scored_rounds, total_rounds: int) -> str:
    rounds = ", ".join(map(str, scored_rounds))
    if not features.SCORE_TRACKING:
        # Only data saved while scores were tracked (or restored from a backup) can trip this.
        # There is no Leaderboard tab to clear them on, so the way on is to keep those rounds as played.
        # 'Rounds already played' stops one short of the last round, so a result there can't be kept.
        if max(scored_rounds) >= total_rounds:
            return (
                f"❌ Round(s) {rounds} already have recorded results, including the last round, so this "
                "schedule can't be replanned. Generate a new schedule instead."
            )
        return (
            f"❌ Round(s) {rounds} already have recorded results, so they can't be replanned. "
            f"Set 'Rounds already played' to at least {max(scored_rounds)}."
        )
    return (
        f"❌ Round(s) {rounds} already have recorded scores, so they can't be "
        f"replanned. Set 'Rounds already played' to at least {max(scored_rounds)}, or clear those scores "
        "on the Leaderboard tab first."
    )


def _replan_not_saved_message(result, total_rounds: int) -> str:
    """Explain why update_schedule_after_replan refused or failed (see ReplanSaveResult)."""
    if result.status == "scored":
        return _scored_rounds_message(result.scored_rounds, total_rounds)
    if result.status == "missing":
        return (
            "❌ This schedule was deleted from History, so a replan of it can't be saved. Restore it "
            "from History > Recently deleted, or generate a new schedule."
        )
    if result.status == "changed":
        return (
            "❌ This schedule was changed in another session since you opened it, so nothing was "
            "replanned. Load it again from History and replan from there."
        )
    return "❌ Could not save the replanned rounds to history, so the schedule was left unchanged. Please try again."


def find_replan_name_issues(remaining_players: list[str], locked_roster_by_round: Sequence[Sequence[str]]) -> list[str]:
    """find_player_name_issues for a replan's roster, which also flags a
    newcomer whose name differs from a player of the rounds kept only in case
    or spacing: one person would otherwise be two players of the schedule, in
    its history and standings.

    Variants already among the rounds kept (saved before this check existed)
    aren't flagged again, so they can't block every later replan.
    """
    issues = find_player_name_issues(remaining_players)
    if issues:
        return issues
    locked_players = sorted({player for roster in locked_roster_by_round for player in roster})
    known_issues = set(find_player_name_issues(locked_players))
    return [
        f"{issue} - counting the players of the rounds already played. Use the same spelling for the same player."
        for issue in find_player_name_issues(sorted(set(locked_players) | set(remaining_players)))
        if issue not in known_issues
    ]


def replan_default_roster(
    players: Sequence[str],
    tracked_rosters: Sequence[Sequence[str]] | None,
    planned_around_attendance: bool,
    played_rounds: int,
) -> list[str]:
    """The players a replan's roster box starts with: who is here for the rounds it will replan.

    ``players`` is everyone the schedule has had, leavers and late arrivers
    included, so once a replan (or a plan) has recorded each round's roster
    the box starts from one of those instead. A schedule that was only
    replanned is replanned for whoever was there at the end, its last round's
    roster. One planned around arrivals and departures has a different roster
    in nearly every round, so the box starts with the roster of the first
    round to replan (the one after ``played_rounds``): the last round's would
    hold a late arriver who is not here yet and miss someone who still is.
    """
    if not tracked_rosters:
        return list(players)
    if planned_around_attendance:
        return list(tracked_rosters[max(0, min(played_rounds, len(tracked_rosters) - 1))])
    return list(tracked_rosters[-1])


def _render_mid_session_replan(st_module, scheduler_cls, schedule_data, players, config, max_time):
    """Let the user regenerate only the not-yet-played rounds, optionally with a
    changed player list (someone left early / showed up late) or court count -
    without losing the rounds already played.

    The replanned rounds look back at the locked rounds in two ways, both about
    who sat out. How many rounds each remaining player has sat out so far (while
    on the roster) goes to the scheduler as baseline_games, as the attendance
    planner does between its segments, so the range of everyone's sit-outs to
    date is kept as low as the new rounds allow instead of the new sit-outs
    piling onto the same players. That bounds the range; it does not promise
    that every new sit-out goes to a player at the minimum, since when the
    rounds kept were already uneven a player above it can still sit out while
    one at it plays. A newcomer counts as having sat out none, so is among the
    first to sit out. And the new rounds' order carries on from who sat out
    each locked round, so a sit-out streak does not run on across the boundary
    unnecessarily. They do not account for who already partnered or opposed
    whom in the locked rounds: that would require the scheduler's fitness
    function to accept pre-existing partner/opponent counts, which it does
    not currently support (baseline_games offsets the games count only).

    A replan runs for up to max_time, the "Max generation time" the user set
    under Schedule Settings.
    """
    total_rounds = len(schedule_data)
    if total_rounds < 2:
        return

    with st_module.expander("🔁 Mid-session replan"):
        st_module.caption(
            "Regenerate the rounds that haven't been played yet - useful if a player "
            "leaves early or joins late. Rounds already played are never changed. The "
            "replanned rounds are fair among themselves and even out how many rounds each "
            "player has sat out so far, counting the rounds you keep (a newcomer starts at "
            "none, so is among the first to sit out), but they are not aware of who already "
            "partnered or opposed whom in those rounds. A replan takes up to the 'Max "
            "generation time' set under Schedule Settings above."
        )

        # Each round's roster, once a replan has changed it. Who is playing
        # now is then its last round's roster: `players` is everyone the
        # schedule has had, including anyone who left.
        tracked_rosters = st_module.session_state.get("current_roster_by_round")
        if not tracked_rosters or len(tracked_rosters) != total_rounds:
            tracked_rosters = None
        planned_around_attendance = bool(tracked_rosters and st_module.session_state.get(CURRENT_ATTENDANCE_KEY))
        if planned_around_attendance:
            st_module.caption(REPLAN_NOTE)
        current_courts = max(1, len(schedule_data[-1].get("games", [])))
        # Set the inputs up for each new schedule on screen (Generate, History
        # load), keeping in-progress edits otherwise. Keyed on the schedule
        # itself, not its roster: a new schedule of the same players must not
        # inherit a "Rounds already played" count that would lock rounds
        # nobody has played. A replan sets them for the schedule it makes
        # (see _run_replan).
        if st_module.session_state.get(REPLAN_INPUTS_SCHEDULE_KEY) is not schedule_data:
            st_module.session_state[REPLAN_INPUTS_SCHEDULE_KEY] = schedule_data
            for key in ("replan_played_rounds", "replan_players_input", "replan_courts"):
                forget_widget_state(st_module.session_state, key)
        # Seeded, not assigned: an st.rerun() before they render (Quick Add,
        # say) drops their state, and this puts back what they showed.
        seed_widget_from_source(st_module.session_state, "replan_played_rounds", 0)
        # Who starts the box depends on the rounds played for a plan with
        # arrivals and departures, so it follows that input - carrying over
        # names the user added or took out - until they type their own list.
        roster_text = "\n".join(
            replan_default_roster(
                players,
                tracked_rosters,
                planned_around_attendance,
                clamp_round(st_module.session_state.get("replan_played_rounds"), 0, total_rounds - 1, 0),
            )
        )
        seed_widget_from_source(
            st_module.session_state,
            "replan_players_input",
            roster_text,
            rebase=rebase_line_edits if planned_around_attendance else None,
        )
        seed_widget_from_source(st_module.session_state, "replan_courts", current_courts)

        # Set through session_state above, so none takes value= (see src/widget_state.py).
        played_rounds = st_module.number_input(
            "Rounds already played:",
            min_value=0,
            max_value=total_rounds - 1,
            key="replan_played_rounds",
        )
        remaining_players_text = st_module.text_area(
            "Players for the remaining rounds (edit to reflect who's still here):",
            key="replan_players_input",
        )
        # A schedule saved under the old, higher MAX_PLAYERS (loaded from History)
        # can use more courts than the limit allows. Its own count stays in range:
        # a box whose maximum is below its seeded value resets to 1.
        requested_courts = st_module.number_input(
            "Courts for remaining rounds:",
            min_value=1,
            max_value=max(MAX_PLAYERS // 4, current_courts),
            key="replan_courts",
            help=(
                "The courts this schedule uses now. Raise it if late arrivals let you open another court - "
                "each court needs 4 players."
            ),
        )

        if st_module.button("🔁 Replan remaining rounds"):
            remaining_players = parse_players_text(remaining_players_text)
            if len(remaining_players) < 4:
                st_module.warning("⚠️ Need at least 4 players for the remaining rounds.")
                return
            if len(remaining_players) > MAX_PLAYERS:
                st_module.warning(f"⚠️ {too_many_players_message(len(remaining_players))}")
                return

            # Track which roster was actually in effect for each round, so
            # "sitting out" can be computed per-round instead of against the
            # union of everyone - which would otherwise show departed players
            # as still sitting out in every future round, and incoming players
            # as already sitting out in rounds before they joined. Reuse the
            # previous replan's per-round tracking for the still-locked
            # portion (a second replan must not flatten it back to `players`,
            # which is only correct for a schedule that's never been
            # replanned before).
            if tracked_rosters:
                locked_roster_by_round = tracked_rosters[:played_rounds]
            else:
                locked_roster_by_round = [players] * played_rounds
            # A name given twice would put one person on court twice.
            name_issues = find_replan_name_issues(remaining_players, locked_roster_by_round)
            if name_issues:
                for issue in name_issues:
                    st_module.error(f"❌ {escape_markdown(issue)}")
                return

            # Rounds being replaced must not hold recorded results: their games
            # are about to change, and the scores would otherwise be credited to
            # the wrong teams (or silently dropped).
            schedule_id = st_module.session_state.get("current_schedule_id")
            history_manager = st_module.session_state.get("history_manager")
            # (Checked again, atomically, when the replan is saved - this early
            # check just avoids running the scheduler for a replan that can't be.)
            if schedule_id is not None and history_manager is not None:
                scored_rounds = history_manager.rounds_with_results(schedule_id, played_rounds + 1)
                if scored_rounds is None:
                    recorded = "scores" if features.SCORE_TRACKING else "results"
                    st_module.error(
                        f"❌ Could not check this schedule's recorded {recorded}, so nothing was replanned."
                    )
                    return
                if scored_rounds:
                    st_module.error(_scored_rounds_message(scored_rounds, total_rounds))
                    return

            locked_rounds, remaining_round_count = split_schedule_at_round(schedule_data, played_rounds)
            courts = requested_courts
            # Shown again above the tabs once the replan reruns the page.
            replan_warnings: list[str] = []
            # The remaining roster may be too small for the courts asked for
            # (players left without replacements) - reduce courts to fit
            # rather than letting the scheduler constructor raise, which
            # would otherwise surface as an opaque "failed unexpectedly".
            max_courts_for_remaining = len(remaining_players) // 4
            if max_courts_for_remaining < courts:
                replan_warnings.append(
                    f"⚠️ Only {len(remaining_players)} players remain, which supports "
                    f"{max_courts_for_remaining} court(s) instead of {courts} - "
                    "using the reduced court count for the remaining rounds."
                )
                st_module.warning(replan_warnings[-1])
                courts = max_courts_for_remaining

            constraints = config["constraints"]
            pair_constraints = [
                tuple(pair) for pair in constraint_pairs_for_roster(constraints["do_not_pair"], remaining_players)
            ]
            oppose_constraints = [
                tuple(pair) for pair in constraint_pairs_for_roster(constraints["do_not_oppose"], remaining_players)
            ]

            # The replan starts once every tab is drawn, and draws here.
            replan_slot = st_module.empty()

            def _run_replan():
                run_marker = _start_scheduler_run(st_module.session_state, "replan")
                with st_module.status("Replanning remaining rounds...", expanded=True) as status_box:
                    cancel_caption = st_module.empty()
                    cancel_caption.caption(_run_cancel_caption())
                    progress_bar = st_module.progress(0)
                    progress_text = st_module.empty()

                    def _on_progress(progress_data, _bar=progress_bar, _text=progress_text, _max_time=max_time):
                        try:
                            elapsed = float(progress_data.get("elapsed_time", 0.0))
                            budget = float(progress_data.get("max_time") or _max_time or 1.0)
                            fraction = max(0.0, min(1.0, elapsed / budget)) if budget else 0.0
                            _bar.progress(fraction)
                            generation = progress_data.get("generation", 0)
                            total_range = progress_data.get("total_range", "?")
                            _text.text(f"Generation {generation} - best range {total_range} - {elapsed:.0f}s elapsed")
                        except Exception:
                            logger.debug("Progress callback failed", exc_info=True)

                    try:
                        scheduler = scheduler_cls(
                            players=remaining_players,
                            num_courts=courts,
                            num_rounds=remaining_round_count,
                            use_desktop_params=True,
                            max_runtime=max_time,
                        )
                        if pair_constraints or oppose_constraints:
                            scheduler.add_constraints(
                                pair_constraints=pair_constraints,
                                oppose_constraints=oppose_constraints,
                            )
                        # Who sat out each locked round, so the new rounds are
                        # ordered to carry on from the last one: a player who
                        # sat out the last round played is not sat out again
                        # in the first replanned round when another order of
                        # the new rounds avoids it (first-review finding 56).
                        scheduler.preceding_sit_outs = sit_outs_by_round(locked_rounds, locked_roster_by_round)
                        # And how many each remaining player has sat out so
                        # far while on the roster, so the range of everyone's
                        # sit-outs to date is kept as low as the new rounds
                        # allow rather than piling onto the same players (as
                        # the attendance planner does between its segments).
                        # Nothing is set when nobody has sat out, which
                        # leaves the scheduler exactly as it was.
                        baseline = sit_out_baseline(locked_rounds, locked_roster_by_round, remaining_players)
                        if baseline:
                            scheduler.baseline_games = baseline
                        with counted_generation(st_module):
                            result = scheduler.generate_schedule(max_time=max_time, progress_callback=_on_progress)
                    except Exception:
                        logger.exception("Mid-session replan failed")
                        st_module.error("❌ Replan failed unexpectedly. See the server logs for details.")
                        status_box.update(label="❌ Replan failed", state="error", expanded=True)
                        _end_scheduler_run(st_module.session_state, run_marker)
                        return
                    finally:
                        cancel_caption.empty()

                    if not (isinstance(result, dict) and "schedule" in result):
                        st_module.error("❌ Could not replan the remaining rounds. Try adjusting the player list.")
                        status_box.update(label="❌ Replan failed", state="error", expanded=True)
                        _end_scheduler_run(st_module.session_state, run_marker)
                        return

                    status_box.update(label="✅ Replan complete", state="complete")

                replan_constraint_violations = result.get("constraint_violations", 0)
                if replan_constraint_violations:
                    replan_warnings.append(
                        constraint_violation_warning(
                            replan_constraint_violations,
                            bool(result.get("round_search_budget_reached", False)),
                            replan=True,
                            max_time=max_time,
                        )
                    )
                    st_module.warning(replan_warnings[-1])

                roster_by_round = locked_roster_by_round + [remaining_players] * remaining_round_count
                replanned_schedule = splice_schedules(locked_rounds, result["schedule"])
                # Everyone on a round's roster, not everyone the session ever
                # had: a name left out of every round kept (a typo corrected by
                # replanning from the same round again) is no longer a player.
                all_players = sorted({player for roster in roster_by_round for player in roster})

                # Save to history before showing it: a replan that lives only in
                # this session is lost on reload, and scores entered against it
                # would end up credited to the original teams.
                if schedule_id is not None and history_manager is not None:
                    saved = history_manager.update_schedule_after_replan(
                        schedule_id, schedule_data, replanned_schedule, all_players, played_rounds + 1, roster_by_round
                    )
                    if not saved.saved:
                        st_module.error(_replan_not_saved_message(saved, total_rounds))
                        _end_scheduler_run(st_module.session_state, run_marker)
                        return

                unsaved = unsaved_schedule(st_module.session_state)  # read before current_schedule changes below
                st_module.session_state.current_roster_by_round = roster_by_round
                st_module.session_state.current_schedule = replanned_schedule
                st_module.session_state.current_players = all_players
                # The arrivals and departures it was planned from no longer
                # describe it: every remaining round now has the one roster.
                st_module.session_state[CURRENT_ATTENDANCE_KEY] = None
                # The metrics and seed shown belong to the pre-replan schedule;
                # after splicing they no longer describe what's on screen. (The
                # replan's own seed can't replay it either: the generator would
                # schedule every round for the whole roster with it.)
                st_module.session_state.current_metrics = {}
                st_module.session_state.current_seed = None
                st_module.session_state.current_generations = None
                if unsaved is not None:
                    # The schedule never made it into history: save the replanned
                    # one now, or keep it as what a retry will save.
                    settings = dict(
                        unsaved["settings"],
                        roster_by_round=[list(roster) for roster in roster_by_round],
                        replanned_from_round=played_rounds + 1,
                    )
                    settings.pop("attendance", None)
                    save_schedule_to_history(
                        st_module.session_state, replanned_schedule, all_players, settings, unsaved["algorithm"]
                    )
                # Keep the inputs as entered for the replanned schedule, so the
                # rounds this replan kept still count as played. The roster box
                # shows the names as scheduled (blank lines and outer spaces
                # dropped), and the courts the ones actually used (capped above).
                st_module.session_state[REPLAN_INPUTS_SCHEDULE_KEY] = replanned_schedule
                defer_widget_value(st_module.session_state, "replan_players_input", "\n".join(remaining_players))
                defer_widget_value(st_module.session_state, "replan_courts", courts)
                st_module.session_state.global_status_message = (
                    f"✅ Replanned rounds {played_rounds + 1}-{total_rounds}."
                )
                for warning in replan_warnings:
                    queue_status_notice(st_module.session_state, "warning", warning)
                _end_scheduler_run(st_module.session_state, run_marker)
                st_module.rerun()

            _start_after_the_tabs(st_module, replan_slot, _run_replan)


def _render_metric_summary(st_module, metrics, players):
    """Render the top-level fairness summary metrics."""
    if not metrics:
        # Schedules loaded from history have no freshly computed metrics -
        # showing all-zero placeholders would read as a terrible schedule.
        return
    most_partnered = _most_partnered(metrics)
    # Metrics with no repeat count (from before it was measured) keep the
    # five-metric row rather than show a placeholder.
    col1, col2, col3, col4, col5, *repeat_column = st_module.columns(6 if most_partnered else 5)
    with col1:
        fairness_score = metrics.get("overall_fairness", 0)
        is_optimal = metrics.get("is_mathematically_optimal", False)
        fairness_label = "Fairness Score"
        if is_optimal is True:
            fairness_label += " ✅"
        elif is_optimal is False and metrics.get("total_range_floor") is not None:
            fairness_label += " ⚠️"
        st_module.metric(fairness_label, f"{fairness_score:.1f}/10")
    with col2:
        st_module.metric("Games Range", metrics.get("games_range", 0))
    with col3:
        st_module.metric("Partners Range", metrics.get("partners_range", 0))
    with col4:
        st_module.metric("Opponents Range", metrics.get("opponents_range", 0))
    with col5:
        st_module.metric("Courts Range", metrics.get("courts_range", 0))
    if most_partnered and repeat_column:
        with repeat_column[0]:
            st_module.metric(
                "Partner Repeats",
                f"{most_partnered}x",
                help=_partner_repeat_help(_partner_repeat_floor(metrics, players)),
            )


def _most_partnered(metrics) -> int | None:
    """The most times any one pair partners, or None when the metrics don't
    say (they predate the count, it is damaged, or no pair ever partnered)."""
    value = metrics.get("most_partnered")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _partner_repeat_floor(metrics, players) -> int | None:
    """The fewest times the most-partnering pair can partner: a player's games
    are spread over len(players) - 1 possible partners, so the player with the
    most games must partner someone at least ceil(most games / (len(players)
    - 1)) times; with nobody sitting out that is ceil(rounds / (len(players)
    - 1)). None when the metrics don't say how many rounds were played."""
    rounds = metrics.get("total_rounds")
    if not rounds or len(players) < 2:
        return None
    games_played = [stats.get("games_played", 0) for stats in (metrics.get("player_stats") or {}).values()]
    most_games = max(games_played, default=rounds)
    return math.ceil(most_games / (len(players) - 1))


def _partner_repeat_help(floor: int | None) -> str:
    """The "Partner Repeats" tooltip: what the number is, why the ranges
    beside it don't show it, and the fewest repeats the games played force."""
    help_text = (
        "The most times any one pair of players are partners. The Partners Range counts "
        "different partners, not repeats, so a pair teaming up again and again doesn't show in it."
    )
    if floor is not None:
        help_text += (
            f" At least {floor}x is unavoidable here: the player with the most games must "
            "partner someone that often."
        )
    return help_text


def _partner_repeat_note(metrics, players) -> str:
    """A factual aside for the balance banners when some pair partners more
    often than an even spread would need: the ranges count each player's
    different partners, so a pair teaming up again and again doesn't show in
    them. The floor is _partner_repeat_floor, the same one the "Partner
    Repeats" metric states."""
    most_partnered = _most_partnered(metrics)
    floor = _partner_repeat_floor(metrics, players)
    if most_partnered is None or floor is None or most_partnered <= floor:
        return ""
    return (
        " The balance counts different partners, not repeats: some pairs are partners "
        f"{most_partnered} times in {metrics['total_rounds']} rounds."
    )


def _render_optimality_status(st_module, metrics, players):
    """Say how balanced the schedule is - claiming "optimal" only when it's proven.

    games_range_floor and total_range_floor are proven lower bounds (see
    calculate_fairness_metrics), so a schedule at total_range_floor is
    optimal. Above it, nothing says how much lower is actually possible.
    """
    games_floor = metrics.get("games_range_floor")
    if games_floor is None:
        return

    total_range = metrics.get("total_range", 0)
    games_range = metrics.get("games_range", 0)
    floor = metrics.get("total_range_floor")
    if floor is None:
        floor = games_floor

    if total_range <= floor:
        if floor == 0:
            message = (
                "✅ **Perfectly balanced:** every player has the same number of games, partners, "
                "opponents and courts."
            )
        elif floor == games_floor:
            message = (
                f"✅ **As balanced as possible:** the games can't divide evenly among {len(players)} "
                "players, so some play one more; partners, opponents and courts are even."
            )
        else:
            message = (
                f"✅ **As balanced as possible:** there are fewer places in the games than the {len(players)} "
                "players, so some can't play at all - no schedule can be more even than this."
            )
        st_module.success(message + _partner_repeat_note(metrics, players))
        return

    if games_range > games_floor:
        evenly = "an even split is possible" if games_floor == 0 else "the smallest possible difference is 1"
        st_module.warning(f"⚠️ **Uneven games:** some players play {games_range} more game(s) than others; {evenly}.")

    if floor == 0:
        bound = "Nothing rules out a perfectly fair schedule here, though one may not exist"
    else:
        bound = f"No schedule for this configuration can go below {floor}, though that may not be reachable"
    st_module.info(
        f"Total range {total_range}. {bound} - generating again, or allowing more time, may improve it."
        + _partner_repeat_note(metrics, players)
    )


def _render_download_buttons(
    st_module,
    datetime_cls,
    json_module,
    schedule_analytics_cls,
    schedule_data,
    players,
    round_times,
    schedule_to_csv_fn,
    roster_by_round=None,
):
    """Render CSV/JSON/text/xlsx download buttons and copy-friendly views for the schedule."""
    _app_schedule_helpers = import_module_with_fallback("app_schedule_helpers")
    schedule_to_player_text = _app_schedule_helpers.schedule_to_player_text
    schedule_to_text = _app_schedule_helpers.schedule_to_text
    schedule_to_xlsx = _app_schedule_helpers.schedule_to_xlsx

    col1, col2, col3, col4 = st_module.columns(4)
    with col1:
        csv_data = schedule_to_csv_fn(schedule_data)
        st_module.download_button(
            "📥 Download CSV",
            csv_data,
            f"enhanced_schedule_{datetime_cls.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )
    with col2:
        serializable_schedule = schedule_analytics_cls.serialize_schedule_for_json(schedule_data)
        json_data = json_module.dumps(serializable_schedule, indent=2)
        st_module.download_button(
            "📄 Download JSON",
            json_data,
            f"schedule_data_{datetime_cls.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json",
        )
    with col3:
        text_data = schedule_to_text(schedule_data, players, round_times, roster_by_round=roster_by_round)
        st_module.download_button(
            "📋 Download Text",
            text_data,
            f"schedule_{datetime_cls.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
        )
    with col4:
        # With score tracking on, the results recorded so far for this schedule
        # go in as a Scores sheet; off, the workbook has the schedule alone.
        schedule_id = st_module.session_state.get("current_schedule_id")
        history_manager = st_module.session_state.get("history_manager")
        scores = (
            history_manager.get_game_scores(schedule_id)
            if features.SCORE_TRACKING and schedule_id is not None and history_manager is not None
            else None
        )
        xlsx_data = schedule_to_xlsx(
            schedule_data, players, round_times, scores=scores, roster_by_round=roster_by_round
        )
        st_module.download_button(
            "📊 Download Excel",
            xlsx_data,
            f"schedule_{datetime_cls.now().strftime('%Y%m%d_%H%M')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st_module.expander("📋 Copy as text"):
        st_module.code(text_data)

    if players:
        with st_module.expander("👤 Per-player schedule"):
            st_module.code(schedule_to_player_text(schedule_data, players, roster_by_round=roster_by_round))
