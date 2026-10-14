#!/usr/bin/env python3
"""Tab renderers extracted from the main Streamlit app."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime
from datetime import time as time_type
from datetime import tzinfo
from pathlib import Path
from typing import Any, Hashable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from src.app_schedule_helpers import escape_markdown, excel_safe, recorded_result, validate_schedule_integrity
except ImportError:
    from app_schedule_helpers import escape_markdown, excel_safe, recorded_result, validate_schedule_integrity

try:
    from src.managers.history import stored_time
except ImportError:
    from managers.history import stored_time

try:
    from src.app_scheduler_flow import (
        MAX_PLAYERS,
        NUM_ROUNDS_KEY,
        PLAYERS_INPUT_KEY,
        PRESET_NAME_KEY,
        RESERVED_PRESET_NAMES,
        counted_generation,
        find_constraint_name_issues,
        find_player_name_issues,
        is_reserved_preset_name,
        parse_players_text,
        queue_status_notice,
        reserved_preset_name_message,
        set_player_preset,
        too_many_players_message,
        unsaved_schedule,
    )
except ImportError:
    from app_scheduler_flow import (
        MAX_PLAYERS,
        NUM_ROUNDS_KEY,
        PLAYERS_INPUT_KEY,
        PRESET_NAME_KEY,
        RESERVED_PRESET_NAMES,
        counted_generation,
        find_constraint_name_issues,
        find_player_name_issues,
        is_reserved_preset_name,
        parse_players_text,
        queue_status_notice,
        reserved_preset_name_message,
        set_player_preset,
        too_many_players_message,
        unsaved_schedule,
    )

try:
    from src.widget_state import (
        apply_deferred_widget_values,
        defer_widget_value,
        forget_widget_state,
        move_widget_state,
        rebase_line_edits,
        seed_widget_from_source,
    )
except ImportError:
    from widget_state import (
        apply_deferred_widget_values,
        defer_widget_value,
        forget_widget_state,
        move_widget_state,
        rebase_line_edits,
        seed_widget_from_source,
    )

_app_managers = import_module_with_fallback("app_managers")
normalize_config_constraints = _app_managers.normalize_config_constraints
normalize_constraint_pairs = _app_managers.normalize_constraint_pairs
parse_constraint_text = _app_managers.parse_constraint_text
serialize_constraint_pairs = _app_managers.serialize_constraint_pairs
export_config_json = _app_managers.export_config_json
import_config_json = _app_managers.import_config_json
CONFIG_REPAIR_MESSAGES_KEY = _app_managers.CONFIG_REPAIR_MESSAGES_KEY
_backup = import_module_with_fallback("managers.backup")

logger = logging.getLogger(__name__)

# Shown when the history database couldn't be opened at all (see
# render_enhanced_scheduler_page), or when one read from it failed - which
# must never look like an empty history.
HISTORY_UNAVAILABLE_MESSAGE = "❌ The schedule history couldn't be opened - see the message at the top of the page."
_READ_FAILED_HINT = (
    "Nothing has been deleted. This usually clears up once another program stops using the database file; "
    "if it keeps happening, the file may be damaged - see the server logs."
)


def infer_players_from_schedule(schedule_data: list[dict[str, Any]]) -> list[str]:
    """Infer players from serialized schedule data while preserving first-seen order."""
    players: list[str] = []

    for round_data in schedule_data:
        games = round_data.get("games", []) if isinstance(round_data, dict) else []
        for game in games:
            if not isinstance(game, dict):
                continue
            for team_key in ("team1", "team2"):
                for player in game.get(team_key, []):
                    normalized_player = str(player)
                    if normalized_player not in players:
                        players.append(normalized_player)

    return players


def load_schedule_and_players_from_history_entry(entry: dict[str, Any], json_module):
    """Load schedule data and restore players from a history entry."""
    try:
        schedule_data = json_module.loads(entry["schedule_data"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Corrupt schedule data in history entry: {exc}") from exc
    players: list[str] = []

    raw_settings = entry.get("settings_data")
    if isinstance(raw_settings, str) and raw_settings:
        try:
            parsed_settings = json_module.loads(raw_settings)
        except (TypeError, ValueError):
            parsed_settings = {}
        if isinstance(parsed_settings, dict):
            stored_players = parsed_settings.get("players", [])
            if isinstance(stored_players, list):
                players = [str(player) for player in stored_players if str(player).strip()]

    if not players:
        players = infer_players_from_schedule(schedule_data)

    return schedule_data, players


def load_roster_by_round_from_history_entry(entry: dict[str, Any], json_module, num_rounds: int):
    """The per-round rosters a replanned schedule saved, or None if it has none (or they don't fit)."""
    try:
        settings = json_module.loads(entry.get("settings_data") or "{}")
    except (TypeError, ValueError):
        return None
    roster_by_round = settings.get("roster_by_round") if isinstance(settings, dict) else None
    if (
        isinstance(roster_by_round, list)
        and len(roster_by_round) == num_rounds
        and all(isinstance(roster, list) for roster in roster_by_round)
    ):
        return [[str(player) for player in roster] for roster in roster_by_round]
    return None


def _show_loaded_schedule_on_main_tab(session_state, players: list[str], num_rounds: int) -> None:
    """Point the Main tab's player list and round count at a schedule loaded from History.

    Otherwise the Main tab keeps showing whatever it had (after a reload, the
    default roster and round count) above the loaded schedule, and Generate
    would schedule that group instead. The player list is left alone when it
    already names the same players, and when the loaded players would fail the
    Main tab's own checks (a replan can save "Hal" and "hal" together): the
    schedule shows either way, but such a list would replace the user's with
    one the Main tab only answers with an error.
    """
    passes_main_tab_checks = len(players) >= 4 and not find_player_name_issues(players)
    if passes_main_tab_checks and set(parse_players_text(session_state.get(PLAYERS_INPUT_KEY) or "")) != set(players):
        # A saved preset naming exactly these players is picked over Custom,
        # which would replace the user's unsaved Custom list.
        presets = session_state.get("app_config", {}).get("player_presets", {})
        preset = next(
            (name for name, names in presets.items() if isinstance(names, list) and sorted(names) == sorted(players)),
            "Custom",
        )
        if preset == "Custom":
            players_text = "\n".join(players)
            session_state.custom_players = players_text
        else:
            players_text = "\n".join(presets[preset])
        set_player_preset(session_state, preset)
        session_state.selected_player_preset_initialized = True
        defer_widget_value(session_state, PLAYERS_INPUT_KEY, players_text)
    defer_widget_value(session_state, NUM_ROUNDS_KEY, num_rounds)


# The History table key a row was ticked under on the previous run, if any.
_HISTORY_TICKED_TABLE_KEY = "_history_ticked_table_key"


def _history_table_key(entries: list[dict[str, Any]]) -> str:
    """Key the History table by the schedules it lists, in order.

    Its row selection is a position in that list, and the browser keeps it for
    as long as the key stays the same - deleting the key from session_state
    doesn't clear it. So any change to the list (a delete, an undo or restore,
    a schedule saved by this or another session) gives the table a new key,
    which the browser shows as a new, unselected table, instead of leaving the
    old row number to pick out whichever schedule moved into that row.
    """
    ids = json.dumps([entry.get("id") for entry in entries])
    return f"history_table_{hashlib.sha1(ids.encode('utf-8'), usedforsecurity=False).hexdigest()[:8]}"


def _selected_history_row_index(selection_event, row_count: int) -> int | None:
    """Extract the selected row index from a st.dataframe selection event, if any.

    An index past the end of the list is ignored rather than raising.
    """
    try:
        rows = selection_event.selection.rows
    except AttributeError:
        return None
    if not rows:
        return None
    row_index = rows[0]
    if row_index >= row_count:
        return None
    return row_index


# How many schedules the History list and its "Recently deleted" list show at
# first, and how many more each of their "Show older" buttons adds.
_HISTORY_PAGE_SIZE = 20
_DELETED_PAGE_SIZE = 10
_HISTORY_SHOWN_KEY = "_history_shown_count"
_DELETED_SHOWN_KEY = "_history_deleted_shown_count"
# A warning about the schedule just loaded, carried across the Load's rerun.
_HISTORY_LOAD_WARNING_KEY = "_history_load_warning"


def _shown_count(session_state, key: str, page_size: int) -> int:
    return int(session_state.get(key, page_size))


def _show_older(session_state, key: str, page_size: int) -> None:
    """A "Show older" button's on_click: list one more page.

    A callback rather than st.rerun(), so the longer list is drawn on the
    click's own run, and the tabs after History are drawn on it too.
    """
    session_state[key] = _shown_count(session_state, key, page_size) + page_size
    # This session changed the list itself: no "list changed" note.
    session_state.pop(_HISTORY_TICKED_TABLE_KEY, None)


def _viewer_time_zone(st_module) -> tzinfo | None:
    """The time zone of the viewer's browser, or None (the server's own) when it isn't known."""
    try:
        name = st_module.context.timezone
        return ZoneInfo(name) if isinstance(name, str) and name else None
    except (AttributeError, OSError, ValueError, ZoneInfoNotFoundError):
        return None


def _history_time(timestamp, time_zone: tzinfo | None) -> str:
    """A stored timestamp as "YYYY-MM-DD HH:MM" in time_zone (see history.stored_time)."""
    try:
        return stored_time(timestamp, time_zone).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return str(timestamp)[:16].replace("T", " ")


_ELO_NOT_UPDATED_WARNING = (
    "⚠️ Couldn't update the ELO ratings, so they don't include this change yet. Use '🔄 Recompute ELO ratings' "
    "on the Leaderboard tab to try again."
)


def _update_elo_ratings(session_state, history_manager) -> None:
    """Recompute the ELO ratings after this session changed which scores count, before the caller's st.rerun().

    Otherwise the Leaderboard's ELO column lags behind its win/loss counts,
    and keeps a deleted schedule's games, until someone recomputes by hand.
    Replaying even years of weekly games takes milliseconds.
    """
    elo_manager = session_state.get("elo_manager")
    if elo_manager is None:
        return
    try:
        elo_manager.recompute_from_history(history_manager)
    except Exception:  # whatever went wrong, the change itself is saved, and the caller still has to say so
        logger.exception("Could not update the ELO ratings")
        queue_status_notice(session_state, "warning", _ELO_NOT_UPDATED_WARNING)


def _import_errors_markdown(errors: list[str]) -> str:
    """A CSV import's errors as one alert, a list of them if there are several, rather than a box per row."""
    if len(errors) == 1:
        return escape_markdown(errors[0])
    return "\n".join(f"- {escape_markdown(error)}" for error in errors)


def render_history_tab(st_module, pd_module, json_module):
    """Render the schedule history tab."""
    st_module.subheader("📅 Schedule History")

    history_manager = st_module.session_state.get("history_manager")
    if history_manager is None:
        st_module.error(HISTORY_UNAVAILABLE_MESSAGE)
        return

    time_zone = _viewer_time_zone(st_module)
    load_warning = st_module.session_state.pop(_HISTORY_LOAD_WARNING_KEY, None)

    st_module.markdown("### 📚 Recent Schedules")
    shown = _shown_count(st_module.session_state, _HISTORY_SHOWN_KEY, _HISTORY_PAGE_SIZE)
    # One more than is shown, to tell whether there are older ones.
    recent = history_manager.get_recent_schedules(shown + 1)
    ticked_table_key = st_module.session_state.pop(_HISTORY_TICKED_TABLE_KEY, None)

    if recent is None:
        st_module.error(f"❌ Couldn't read the schedule history, so it isn't shown. {_READ_FAILED_HINT}")
    elif recent:
        has_older = len(recent) > shown
        recent = recent[:shown]
        history_data = []
        for entry in recent:
            history_data.append(
                {
                    "Date": _history_time(entry["timestamp"], time_zone),
                    "Players": entry["num_players"],
                    "Rounds": entry["num_rounds"],
                    "Algorithm": entry["algorithm"],
                    "Fairness": f"{entry['fairness_score']:.1f}/10",
                }
            )

        df = pd_module.DataFrame(history_data)
        table_key = _history_table_key(recent)
        selection_event = st_module.dataframe(
            df,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=table_key,
        )
        selected_index = _selected_history_row_index(selection_event, len(recent))
        selected_entry = recent[selected_index] if selected_index is not None else None
        if selected_entry is not None:
            st_module.session_state[_HISTORY_TICKED_TABLE_KEY] = table_key
        # Drawn even when empty, so what's below keeps its place on the page as
        # these notes come and go: the browser draws an element that has moved
        # as a new one, which closes an open "Recently deleted".
        with st_module.container():
            if selected_entry is None and ticked_table_key not in (None, table_key):
                # A row was ticked, but the list has changed since (another session
                # saved a schedule, say), so the table is a new, unselected one and
                # a Load Selected or Delete Selected click made just now was dropped.
                st_module.info("The list of schedules changed, so nothing is selected now. Select the schedule again.")
            if has_older:
                total = history_manager.count_schedules()
                st_module.caption(
                    f"Showing the {shown} newest of {total} schedules."
                    if total is not None
                    else f"Showing the {shown} newest schedules. There are older ones."
                )
                st_module.button(
                    "⬇️ Show older schedules",
                    key="history_show_older",
                    on_click=_show_older,
                    args=(st_module.session_state, _HISTORY_SHOWN_KEY, _HISTORY_PAGE_SIZE),
                )

        col1, col2 = st_module.columns(2)
        with col1:
            load_label = "🔄 Load Selected Schedule" if selected_entry is not None else "🔄 Load Latest Schedule"
            if st_module.button(load_label):
                try:
                    target_entry = selected_entry if selected_entry is not None else recent[0]
                    schedule_data, players = load_schedule_and_players_from_history_entry(target_entry, json_module)
                    # Older versions could save a schedule that books a player
                    # twice in one round. It still loads, so it can be looked at,
                    # but not with a plain success.
                    schedule_errors = validate_schedule_integrity(schedule_data, players)
                    st_module.session_state.current_schedule = schedule_data
                    st_module.session_state.current_players = players
                    # Point score entry at the loaded schedule - otherwise the
                    # Leaderboard tab would keep attaching scores to whichever
                    # schedule was generated last.
                    st_module.session_state.current_schedule_id = target_entry.get("id")
                    # Re-read its scores from the database: this session may have
                    # shown them before another session saved new ones.
                    refresh_score_widgets(st_module.session_state, target_entry.get("id"))
                    # Reset companion state computed for the previously shown
                    # schedule so the loaded one doesn't display someone
                    # else's metrics or seed.
                    st_module.session_state.current_metrics = {}
                    st_module.session_state.current_seed = None
                    # Round times aren't stored with a schedule; work them out
                    # from the configured session window, as Generate does.
                    compute_round_times = import_module_with_fallback("app_schedule_helpers").compute_round_times
                    scheduling_config = st_module.session_state.get("app_config", {}).get("scheduling", {})
                    st_module.session_state.current_round_times = compute_round_times(
                        len(schedule_data), scheduling_config.get("start_time"), scheduling_config.get("end_time")
                    )
                    # Restore a replanned schedule's per-round rosters (so "sitting
                    # out" stays right), and otherwise clear any left over from
                    # whatever schedule was on screen before.
                    st_module.session_state.current_roster_by_round = load_roster_by_round_from_history_entry(
                        target_entry, json_module, len(schedule_data)
                    )
                    _show_loaded_schedule_on_main_tab(st_module.session_state, players, len(schedule_data))
                    if schedule_errors:
                        st_module.session_state[_HISTORY_LOAD_WARNING_KEY] = (
                            f"⚠️ Schedule loaded, but it isn't valid: {len(schedule_errors)} problem(s), such as "
                            "a player in two places in the same round (listed on the Main Scheduler tab). It was "
                            "probably made by an older version of the app. Don't play or score it - generate a "
                            "new schedule instead."
                        )
                    else:
                        st_module.session_state.global_status_message = "✅ Schedule loaded!"
                    logger.info(
                        "History load: rounds=%d players=%d problems=%d",
                        len(schedule_data),
                        len(players),
                        len(schedule_errors),
                    )
                    st_module.rerun()
                except Exception:
                    logger.exception("History load failed")
                    st_module.error("Failed to load the selected schedule. See the server logs for details.")

        with col2:
            if selected_entry is not None and st_module.button("🗑️ Delete Selected Schedule"):
                if history_manager.delete_schedule(selected_entry["id"]):
                    _update_elo_ratings(st_module.session_state, history_manager)
                    st_module.session_state.last_deleted_id = selected_entry["id"]
                    st_module.session_state.global_status_message = "🗑️ Schedule deleted."
                    # This session changed the list itself: no "list changed" note.
                    st_module.session_state.pop(_HISTORY_TICKED_TABLE_KEY, None)
                    st_module.rerun()
                else:
                    st_module.error("Failed to delete schedule.")
    else:
        st_module.info("No schedules in history yet. Generate some schedules to build history.")

    last_deleted_id = st_module.session_state.get("last_deleted_id")
    # None too once it's purged, or while it can't be read: Undo would fail.
    last_deleted = history_manager.get_schedule(last_deleted_id) if last_deleted_id is not None else None
    if last_deleted is not None and last_deleted.get("deleted_at") is None:
        # Restored since, from Recently deleted or by another session: there's
        # nothing left to undo.
        st_module.session_state.last_deleted_id = last_deleted = None
    # Drawn even when empty, like the notes under the table.
    with st_module.container():
        if load_warning:
            st_module.warning(load_warning)
        if last_deleted is not None and st_module.button("↩️ Undo Delete"):
            if history_manager.restore_schedule(last_deleted_id):
                _update_elo_ratings(st_module.session_state, history_manager)
                st_module.session_state.last_deleted_id = None
                st_module.session_state.global_status_message = "↩️ Schedule restored."
                st_module.session_state.pop(_HISTORY_TICKED_TABLE_KEY, None)
                st_module.rerun()
            else:
                st_module.error("Failed to restore schedule.")

    deleted_shown = _shown_count(st_module.session_state, _DELETED_SHOWN_KEY, _DELETED_PAGE_SIZE)
    deleted = history_manager.get_deleted_schedules(deleted_shown + 1)
    if deleted is None:
        st_module.error(f"❌ Couldn't read the recently deleted schedules. {_READ_FAILED_HINT}")
    elif deleted:
        has_older_deleted = len(deleted) > deleted_shown
        deleted = deleted[:deleted_shown]
        deleted_total = history_manager.count_schedules(deleted=True) if has_older_deleted else len(deleted)
        with st_module.expander("🗑️ Recently deleted"):
            for entry in deleted:
                col1, col2 = st_module.columns([3, 1])
                with col1:
                    st_module.write(f"{_history_time(entry['timestamp'], time_zone)} — {entry['num_players']} players")
                with col2:
                    if st_module.button("↩️ Restore", key=f"restore_{entry['id']}"):
                        if history_manager.restore_schedule(entry["id"]):
                            _update_elo_ratings(st_module.session_state, history_manager)
                            st_module.session_state.global_status_message = "↩️ Schedule restored."
                            st_module.session_state.pop(_HISTORY_TICKED_TABLE_KEY, None)
                            st_module.rerun()
            if has_older_deleted:
                st_module.caption(
                    f"Showing the {deleted_shown} most recent of {deleted_total} deletions."
                    if deleted_total is not None
                    else f"Showing the {deleted_shown} most recent deletions. There are older ones."
                )
                st_module.button(
                    "⬇️ Show older deletions",
                    key="history_show_older_deletions",
                    on_click=_show_older,
                    args=(st_module.session_state, _DELETED_SHOWN_KEY, _DELETED_PAGE_SIZE),
                )

            st_module.markdown("---")
            st_module.caption("Permanently remove deletions older than this many days. This cannot be undone.")
            purge_col1, purge_col2 = st_module.columns([2, 1])
            with purge_col1:
                purge_days = st_module.number_input(
                    "Purge deletions older than (days):",
                    min_value=1,
                    max_value=365,
                    value=30,
                    key="purge_older_than_days",
                )
            with purge_col2:
                st_module.markdown("&nbsp;")
                purge_pending = st_module.session_state.get("_purge_confirm_pending", False)
                if not purge_pending:
                    if st_module.button("🧹 Purge Now"):
                        st_module.session_state._purge_confirm_pending = True
                        st_module.rerun()
                else:
                    purgeable_count = history_manager.count_purgeable(older_than_days=purge_days)
                    # Purge takes the oldest deletions and the list shows the
                    # newest, so the unlisted ones are the first it takes.
                    unlisted = (
                        None if deleted_total is None else min(purgeable_count, max(deleted_total - len(deleted), 0))
                    )
                    if unlisted is None:
                        unlisted_note = " Some of them may not be listed above."
                    elif unlisted == 1:
                        unlisted_note = " 1 of them isn't listed above: use ⬇️ Show older deletions to see it."
                    elif unlisted:
                        unlisted_note = (
                            f" {unlisted} of them aren't listed above: use ⬇️ Show older deletions to see them."
                        )
                    else:
                        unlisted_note = ""
                    st_module.warning(
                        f"⚠️ This will permanently remove {purgeable_count} deletion(s) older than "
                        f"{purge_days} days.{unlisted_note} This cannot be undone."
                    )
                    confirm_col, cancel_col = st_module.columns(2)
                    with confirm_col:
                        if st_module.button("✅ Confirm Purge"):
                            purged_count = history_manager.purge_deleted(older_than_days=purge_days)
                            st_module.session_state._purge_confirm_pending = False
                            st_module.session_state.global_status_message = (
                                f"🧹 Purged {purged_count} deletion(s) older than {purge_days} days."
                            )
                            st_module.rerun()
                    with cancel_col:
                        if st_module.button("Cancel"):
                            st_module.session_state._purge_confirm_pending = False
                            st_module.rerun()

    st_module.markdown("### 🤝 Weekly Partner History")
    partner_history = history_manager.get_weekly_partners(tz=time_zone)

    if partner_history is None:
        st_module.error(f"❌ Couldn't read the weekly partner history. {_READ_FAILED_HINT}")
    elif not partner_history:
        st_module.info("No partner history yet. Generate and play some schedules to build history.")
    else:
        for week, pairs in sorted(partner_history.items(), reverse=True)[:4]:
            with st_module.expander(f"Week of {week}"):
                partner_counts: dict[str, set[str]] = {}
                for p1, p2 in pairs:
                    for player in [p1, p2]:
                        if player not in partner_counts:
                            partner_counts[player] = set()
                        partner_counts[player].add(p1 if player == p2 else p2)

                for player, partners in sorted(partner_counts.items()):
                    partner_names = ", ".join(escape_markdown(partner) for partner in sorted(partners))
                    st_module.write(f"**{escape_markdown(player)}**: {partner_names}")


_PAIRING_HEATMAP_CACHE_KEY = "_pairing_heatmap_cache"


def _render_pairing_heatmap(st_module, pd_module, schedule_analytics_cls, schedule, players):
    """Render a partner/opponent count heatmap, toggled by the user, plus the
    same matrix as a plain dataframe underneath - a non-visual equivalent
    that doesn't depend on interpreting a colorscale, available regardless
    of whether plotly is installed."""
    if not players:
        return

    view = st_module.radio("Pairing view:", ["Partners", "Opponents"], key="pairing_heatmap_view")
    partner_counts, opponent_counts = schedule_analytics_cls.build_pairing_matrices(schedule, players)
    matrix = partner_counts if view == "Partners" else opponent_counts

    # st.tabs runs this tab on every rerun, including a click on any other
    # tab, so reuse the last figure while its inputs are unchanged instead of
    # rebuilding a label for every cell each time.
    heatmap_inputs = (view, tuple(players), tuple(map(tuple, matrix)))
    cached_heatmap = st_module.session_state.get(_PAIRING_HEATMAP_CACHE_KEY)
    if isinstance(cached_heatmap, tuple) and cached_heatmap[0] == heatmap_inputs:
        heatmap = cached_heatmap[1]
    else:
        heatmap = schedule_analytics_cls.create_pairing_heatmap(matrix, players, f"{view} Heatmap")
        st_module.session_state[_PAIRING_HEATMAP_CACHE_KEY] = (heatmap_inputs, heatmap)
    if heatmap:
        st_module.plotly_chart(heatmap, width="stretch")

    labels = [excel_safe(player) for player in players]
    df = pd_module.DataFrame(matrix, index=labels, columns=labels)
    st_module.dataframe(df, width="stretch")


def _render_pairing_matrix_fallback(st_module, pd_module, schedule_analytics_cls, schedule, players):
    """Render the pairing matrix as a plain table when plotly is unavailable."""
    if not players:
        return

    st_module.markdown("### 🤝 Pairing Counts")
    view = st_module.radio("Pairing view:", ["Partners", "Opponents"], key="pairing_matrix_view")
    partner_counts, opponent_counts = schedule_analytics_cls.build_pairing_matrices(schedule, players)
    matrix = partner_counts if view == "Partners" else opponent_counts

    labels = [excel_safe(player) for player in players]
    df = pd_module.DataFrame(matrix, index=labels, columns=labels)
    st_module.dataframe(df, width="stretch")


def render_analytics_tab(st_module, pd_module, schedule_analytics_cls, has_plotly):
    """Render the analytics and visualization tab."""
    st_module.subheader("📊 Schedule Analytics")

    if "current_schedule" not in st_module.session_state:
        st_module.info("Generate a schedule first to see analytics")
        return

    schedule = st_module.session_state.current_schedule
    players = st_module.session_state.get("current_players", [])

    if players and schedule:
        num_players = len(players)
        num_rounds = len(schedule)
        first_round_games = schedule[0].get("games", []) if schedule else []
        num_courts = len(first_round_games) if first_round_games else 1

        # After a replan, each player is measured against the rounds they were
        # there for - leaving early or joining late isn't unfair - as History does.
        metrics = schedule_analytics_cls.calculate_fairness_metrics(
            schedule,
            num_players,
            num_rounds,
            num_courts,
            players=players,
            roster_by_round=st_module.session_state.get("current_roster_by_round"),
        )
    else:
        metrics = schedule_analytics_cls.calculate_fairness_metrics(schedule)

    st_module.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st_module.columns(4)

    with col1:
        st_module.metric("Total Players", metrics.get("total_players", 0))
    with col2:
        st_module.metric("Total Rounds", metrics.get("total_rounds", 0))
    with col3:
        st_module.metric("Overall Fairness", f"{metrics.get('overall_fairness', 0):.1f}/10")
    with col4:
        total_games = sum(len(round_data.get("games", [])) for round_data in schedule)
        st_module.metric("Total Games", total_games)

    if has_plotly:
        st_module.markdown("### 🎯 Fairness Analysis")

        col1, col2 = st_module.columns(2)

        with col1:
            player_chart = schedule_analytics_cls.create_player_stats_chart(metrics)
            if player_chart:
                st_module.plotly_chart(player_chart, width="stretch")

        with col2:
            _render_pairing_heatmap(st_module, pd_module, schedule_analytics_cls, schedule, players)
    else:
        st_module.info("📊 Install plotly for interactive fairness charts: `pip install plotly`")
        _render_pairing_matrix_fallback(st_module, pd_module, schedule_analytics_cls, schedule, players)

    st_module.markdown("### 👥 Player Details")
    if metrics.get("player_stats"):
        stats_data = []
        for player, stats in metrics["player_stats"].items():
            stats_data.append(
                {
                    "Player": excel_safe(player),
                    "Games": stats["games_played"],
                    "Partners": len(stats["partners"]),
                    "Opponents": len(stats["opponents"]),
                    "Courts": len(stats["courts_used"]),
                    "Partner List": excel_safe(", ".join(sorted(stats["partners"]))),
                    "Opponent List": excel_safe(", ".join(sorted(stats["opponents"]))),
                }
            )

        df = pd_module.DataFrame(stats_data)
        st_module.dataframe(df, width="stretch")

    _render_skill_balance(st_module, pd_module, schedule)


def _render_skill_balance(st_module, pd_module, schedule):
    """Show per-game team skill totals, if any players have a rated skill (#124)."""
    skill_manager = st_module.session_state.get("skill_manager")
    if skill_manager is None:
        return

    skills = skill_manager.load_skills()
    if not skills:
        return

    compute_team_skill_balance = import_module_with_fallback("utils.schedule_analytics_core").compute_team_skill_balance
    balance_rows = compute_team_skill_balance(schedule, skills)
    if not balance_rows:
        return

    st_module.markdown("### ⚖️ Team Skill Balance")
    st_module.caption("Team skill totals from your saved ratings - not used during schedule generation.")

    balance_data = [
        {
            "Round": row["round_num"],
            "Court": row["court"],
            "Team 1": excel_safe(" & ".join(row["team1"])),
            "Team 1 Skill": row["team1_skill_total"],
            "Team 2": excel_safe(" & ".join(row["team2"])),
            "Team 2 Skill": row["team2_skill_total"],
            # Text throughout: Streamlit can't send a column mixing numbers and "—" as is, and logs a
            # traceback converting it on every run, while a nullable number column shows a grey "None"
            # (even through a Styler's na_rep). Ratings run 1-5, so a doubles imbalance is one digit
            # and sorts as text the same as it would as a number.
            "Imbalance": "—" if row["imbalance"] is None else str(row["imbalance"]),
            "Fully Rated": row["fully_rated"],
        }
        for row in balance_rows
    ]
    df = pd_module.DataFrame(balance_data)
    st_module.dataframe(df, width="stretch")

    rated_rows = [row for row in balance_rows if row["imbalance"] is not None]
    most_imbalanced = max(rated_rows, key=lambda row: row["imbalance"], default=None)
    if most_imbalanced is not None and most_imbalanced["imbalance"] > 0:
        st_module.info(
            f"Most imbalanced: Round {most_imbalanced['round_num']}, Court {most_imbalanced['court']} "
            f"(skill difference of {most_imbalanced['imbalance']})"
        )


_CONFIG_WIDGET_KEYS = (
    "do_not_pair_input",
    "do_not_oppose_input",
    "cfg_default_rounds",
    "cfg_start_time",
    "cfg_end_time",
)


def _reset_config_widgets_to_saved(session_state) -> None:
    """Make every config-derived widget show the saved config on the next run, dropping unsaved edits.

    For a restore: it should replace the settings, not keep (or rebase) edits
    made before it. Forgetting each widget's source fingerprint makes
    seed_widget_from_source reassign it from the saved config - an assignment
    a real browser applies, unlike deleting the widget's key.
    """
    keys = [
        key
        for key in list(session_state.keys())
        if isinstance(key, str) and (key.startswith("weight_") or key.startswith(_PRESET_WIDGET_KEY_PREFIX))
    ]
    for key in keys + list(_CONFIG_WIDGET_KEYS):
        forget_widget_state(session_state, key)


def _constraint_line_key(line: str) -> Hashable:
    pairs = normalize_constraint_pairs([line])
    return frozenset(pairs[0]) if pairs else line.strip()


def _unreadable_constraint_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not normalize_constraint_pairs([line])]


def _rebase_constraint_edits(previous_source: str, edited: str, new_source: str) -> str:
    # Lines compare by the pair they name, so a pending "Carl,Dan" isn't kept
    # beside the "Carl, Dan" the Main tab just added (and then outliving that
    # pair's removal there).
    return rebase_line_edits(previous_source, edited, new_source, line_key=_constraint_line_key)


def render_configuration_tab(st_module):
    """Render the configuration and constraints tab.

    The widgets edit a private copy (``edited``), never the shared
    session config: every tab renders on every run, so writing widget values
    into app_config would let an abandoned edit here ride along with any
    unrelated save (Generate, a preset save) and reach disk. Only "Save
    Configuration" commits the copy, and only once the save succeeds.
    """
    st_module.subheader("⚙️ Configuration")

    config = normalize_config_constraints(st_module.session_state.app_config)
    st_module.session_state.app_config = config
    edited = deepcopy(config)

    st_module.markdown("### 🎚️ Objective Weights")
    st_module.info("Adjust the importance of different scheduling objectives")

    for obj_name, obj_data in config["objectives"].items():
        col1, col2 = st_module.columns([3, 1])

        with col1:
            slider_key = f"weight_{obj_name}"
            # Track the saved weight (an import, another tab's save) without
            # discarding an edit in progress on unrelated reruns.
            seed_widget_from_source(st_module.session_state, slider_key, obj_data["weight"])
            new_weight = st_module.slider(
                obj_name.replace("_", " ").title(),
                min_value=obj_data["min"],
                max_value=obj_data["max"],
                step=100,
                key=slider_key,
            )
            edited["objectives"][obj_name]["weight"] = new_weight

        with col2:
            st_module.metric("Current", new_weight)

    st_module.markdown("### 🎯 Player Constraints")

    col1, col2 = st_module.columns(2)

    with col1:
        st_module.markdown("**Do Not Pair Together**")
        do_not_pair_text = serialize_constraint_pairs(config["constraints"].get("do_not_pair", []))
        # st.tabs() runs every tab's body on every rerun, not just the visible
        # one - so this keyed widget goes stale whenever do_not_pair changes
        # elsewhere (Main Scheduler quick-add, a loaded history entry), and the
        # write-back below would then silently revert that change.
        seed_widget_from_source(
            st_module.session_state, "do_not_pair_input", do_not_pair_text, _rebase_constraint_edits
        )
        do_not_pair = st_module.text_area(
            "Players who should not be paired (format: Player1,Player2):",
            height=100,
            key="do_not_pair_input",
        )
        # Re-parse only text the user actually edited, and even then keep
        # unchanged lines as their saved pairs: the "A, B" format is ambiguous
        # for names containing ",", "&" or " vs " (see parse_constraint_text).
        if do_not_pair != do_not_pair_text:
            edited["constraints"]["do_not_pair"] = parse_constraint_text(
                do_not_pair, config["constraints"].get("do_not_pair", [])
            )

    with col2:
        st_module.markdown("**Do Not Oppose Each Other**")
        do_not_oppose_text = serialize_constraint_pairs(config["constraints"].get("do_not_oppose", []))
        seed_widget_from_source(
            st_module.session_state, "do_not_oppose_input", do_not_oppose_text, _rebase_constraint_edits
        )
        do_not_oppose = st_module.text_area(
            "Players who should not oppose each other (format: Player1,Player2):",
            height=100,
            key="do_not_oppose_input",
        )
        if do_not_oppose != do_not_oppose_text:
            edited["constraints"]["do_not_oppose"] = parse_constraint_text(
                do_not_oppose, config["constraints"].get("do_not_oppose", [])
            )

    # Checked against the Main tab's roster, and shown for saved pairs too:
    # Save reruns, so a warning only for unsaved text would never be seen.
    roster = parse_players_text(st_module.session_state.get(PLAYERS_INPUT_KEY) or "")
    for issue in find_constraint_name_issues(edited, roster):
        st_module.warning(f"⚠️ {escape_markdown(issue)}")
    # Parsing drops these, so without a word here Save would lose them silently.
    unreadable_lines = _unreadable_constraint_lines(do_not_pair) + _unreadable_constraint_lines(do_not_oppose)
    for line in unreadable_lines:
        st_module.warning(f"⚠️ '{line}' isn't a pair of players (format: Player1,Player2), so it won't be saved.")

    st_module.markdown("### ⏰ Scheduling Preferences")

    col1, col2, col3 = st_module.columns(3)

    with col1:
        saved_rounds = config["scheduling"]["default_rounds"]
        seed_widget_from_source(st_module.session_state, "cfg_default_rounds", saved_rounds)
        default_rounds = st_module.number_input("Default rounds:", min_value=1, max_value=20, key="cfg_default_rounds")
        edited["scheduling"]["default_rounds"] = default_rounds

    with col2:
        saved_start = time_type.fromisoformat(config["scheduling"]["start_time"])
        seed_widget_from_source(st_module.session_state, "cfg_start_time", saved_start)
        selected_start_time = st_module.time_input("Default start time:", key="cfg_start_time")
        if selected_start_time is not None:
            edited["scheduling"]["start_time"] = selected_start_time.strftime("%H:%M")

    with col3:
        saved_end = time_type.fromisoformat(config["scheduling"]["end_time"])
        seed_widget_from_source(st_module.session_state, "cfg_end_time", saved_end)
        selected_end_time = st_module.time_input("Default end time:", key="cfg_end_time")
        if selected_end_time is not None:
            edited["scheduling"]["end_time"] = selected_end_time.strftime("%H:%M")

    if edited != config:
        st_module.caption("You have unsaved changes - click Save Configuration to keep them.")
    if st_module.button("💾 Save Configuration"):
        if st_module.session_state.config_manager.save_config(edited):
            st_module.session_state.app_config = edited
            st_module.session_state._constraint_widget_version = (
                int(st_module.session_state.get("_constraint_widget_version", 0)) + 1
            )
            st_module.session_state._sync_main_constraints_from_config = True
            st_module.session_state._preserve_saved_constraints_from_config = True
            # Show the pairs as saved. Rebased as a pending edit, the typed
            # text would re-add every line the save rewrote ("Carl,Dan" is
            # saved as "Carl, Dan") or dropped, and a later save would write
            # such a line back after the pair was removed elsewhere.
            for key in ("do_not_pair_input", "do_not_oppose_input"):
                forget_widget_state(st_module.session_state, key)
            message = "✅ Configuration saved!"
            if unreadable_lines:
                # The warning above is gone after the rerun, and so are the lines.
                skipped = ", ".join(f"'{line}'" for line in unreadable_lines)
                message += f" Not a pair of players, so left out: {skipped}."
            st_module.session_state.global_status_message = message
            st_module.rerun()
        else:
            st_module.error("❌ Failed to save configuration.")

    st_module.markdown("### 💼 Backup / Restore")
    config_manager = st_module.session_state.config_manager
    session_view = deepcopy(config)  # the download runs in another thread, while later reruns may edit config

    def export_config() -> str:
        # Built when clicked, as a save would write it: this session's copy
        # lacks what other sessions saved since it loaded, and restoring such
        # a backup (a replacing save) would delete their presets and
        # constraints. Its own unsaved Main-tab constraint ticks stay in.
        return export_config_json(config_manager.merge_with_saved(session_view))

    st_module.download_button(
        "⬇️ Download config",
        data=export_config,
        file_name="app_config.json",
        mime="application/json",
    )
    uploaded = st_module.file_uploader("⬆️ Upload config", type=["json"], key="config_upload")
    # file_uploader keeps returning the same UploadedFile on every rerun for
    # as long as it stays selected in the widget - without this guard, any
    # unrelated button click elsewhere on the page would re-run the import
    # (and re-save) indefinitely. file_id changes only on a genuinely new
    # upload, so this fires exactly once per distinct file.
    if uploaded is not None and st_module.session_state.get("_last_config_upload_id") != uploaded.file_id:
        st_module.session_state._last_config_upload_id = uploaded.file_id
        imported, messages = import_config_json(
            uploaded.getvalue(), st_module.session_state.config_manager.default_config
        )
        for message in messages:
            st_module.warning(escape_markdown(message))
        if messages:
            # A repaired-with-defaults or totally-unparseable result is not
            # the config the user meant to import - don't save it or claim
            # success, so they can fix the file and re-upload instead.
            st_module.error("❌ Import failed - configuration was not changed. Fix the issue(s) above and re-upload.")
        elif st_module.session_state.config_manager.save_config(imported, replace=True):
            _adopt_replacing_config(st_module.session_state, imported)
            st_module.session_state.global_status_message = "✅ Configuration imported!"
            st_module.rerun()
        else:
            st_module.error("❌ Failed to save imported configuration.")

    _render_full_backup(st_module, export_config)


def _adopt_replacing_config(session_state, config) -> None:
    """Make a config that replaced the saved one (an upload, a full restore) this session's, in every widget."""
    session_state.app_config = config
    session_state._constraint_widget_version = int(session_state.get("_constraint_widget_version", 0)) + 1
    session_state._sync_main_constraints_from_config = True
    session_state._preserve_saved_constraints_from_config = True
    _reset_config_widgets_to_saved(session_state)


_FULL_BACKUP_UPLOAD_VERSION_KEY = "_full_backup_upload_version"
_FULL_BACKUP_CHECK_KEY = "_full_backup_check"
_SKILL_RATING_WIDGET_KEY_PREFIX = "skill_rating_"


def _full_backup_data_paths(session_state) -> dict[str, Path]:
    """Where each JSON file of a full backup is saved, as this session's managers resolve it."""
    skill_manager = session_state.get("skill_manager") or _app_managers.SkillRatingManager()
    elo_manager = session_state.get("elo_manager") or _app_managers.EloRatingManager()
    return {
        _backup.CONFIG_NAME: session_state.config_manager.config_path,
        _backup.SKILLS_NAME: skill_manager.skills_path,
        _backup.ELO_NAME: elo_manager.ratings_path,
    }


def _adopt_restored_data(session_state) -> bool:
    """After a full restore, show the restored data in place of this session's views of what it replaced.

    Returns whether the schedule on screen had to be let go from score entry:
    the restored history doesn't hold it under its id (that id may now name
    another schedule, or none).
    """
    restored_config = session_state.config_manager.load_config()
    # Shown when this session first loaded the config; a restored config has passed the import checks.
    restored_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
    _adopt_replacing_config(session_state, restored_config)
    for key in list(session_state.keys()):
        if isinstance(key, str) and key.startswith(_SKILL_RATING_WIDGET_KEY_PREFIX):
            forget_widget_state(session_state, key)
    schedule_id = session_state.get("current_schedule_id")
    schedule = session_state.get("current_schedule")
    if schedule_id is None or not schedule:
        return False
    if not session_state.history_manager.holds_schedule(schedule_id, schedule):
        session_state.current_schedule_id = None
        return True
    refresh_score_widgets(session_state, schedule_id)  # reseed the score boxes from the restored scores
    return False


def _render_full_backup(st_module, export_config) -> None:
    """Download, or restore, everything the app saves as one file.

    The hosted app keeps its data in files that a reboot deletes; this is how
    the data survives one: download before the reboot, restore after it.
    """
    st_module.markdown("#### 🗄️ Full backup")
    history_manager = st_module.session_state.get("history_manager")
    if history_manager is None:
        st_module.caption("Full backup and restore aren't available while the schedule history can't be opened.")
        return
    data_paths = _full_backup_data_paths(st_module.session_state)
    default_config = st_module.session_state.config_manager.default_config
    st_module.caption(
        "One file with everything the app saves: schedule history and scores, configuration, skill ratings and "
        "ELO ratings. Rebooting the hosted app deletes all of it, so download a full backup before a reboot and "
        "restore it afterwards."
    )
    st_module.download_button(
        "⬇️ Download full backup",
        # Built when clicked (in another thread), from the saved data as it is then.
        data=lambda: _backup.make_full_backup(history_manager, data_paths, export_config(), default_config),
        file_name=f"pickleball_full_backup_{datetime.now():%Y-%m-%d}.zip",
        mime="application/zip",
    )
    kept_copy = _backup.latest_kept_copy(history_manager.db_path.parent)
    if kept_copy is not None:
        st_module.download_button(
            "⬇️ Download the data the last restore replaced",
            data=kept_copy.read_bytes,
            file_name=kept_copy.name,
            mime="application/zip",
        )

    # A new key after each restore empties the uploader, so the restored file isn't offered again.
    upload_version = int(st_module.session_state.get(_FULL_BACKUP_UPLOAD_VERSION_KEY, 0))
    uploaded = st_module.file_uploader(
        "⬆️ Restore full backup", type=["zip"], key=f"full_backup_upload_{upload_version}"
    )
    if uploaded is None:
        return
    check = st_module.session_state.get(_FULL_BACKUP_CHECK_KEY)
    if not check or check["file_id"] != uploaded.file_id:
        # Once per file, not on every rerun: every tab renders on every run.
        try:
            lines = _backup.read_full_backup(uploaded.getvalue(), default_config).describe()
            check = {"file_id": uploaded.file_id, "lines": lines}
        except _backup.BackupError as exc:
            check = {"file_id": uploaded.file_id, "problem": str(exc)}
        st_module.session_state[_FULL_BACKUP_CHECK_KEY] = check
    if "problem" in check:
        st_module.error(f"❌ {escape_markdown(check['problem'])} Nothing was changed.")
        return

    st_module.markdown("\n".join(f"- {escape_markdown(line)}" for line in check["lines"]))
    st_module.warning(
        "⚠️ Restoring replaces the saved schedule history, scores, configuration and ratings with this "
        "backup's, for everyone using the app. A copy of the current data is kept first."
    )
    confirmed = st_module.checkbox(
        "Replace the current data with this backup", key=f"full_backup_confirm_{uploaded.file_id}"
    )
    if st_module.button("♻️ Restore full backup", disabled=not confirmed, type="primary"):
        try:
            # Checked again, as restored: the earlier check only describes the file.
            backup = _backup.read_full_backup(uploaded.getvalue(), default_config)
            kept = _backup.restore_full_backup(backup, history_manager, data_paths)
        except _backup.BackupError as exc:
            st_module.error(f"❌ {escape_markdown(str(exc))}")
            return
        schedule_let_go = _adopt_restored_data(st_module.session_state)
        st_module.session_state[_FULL_BACKUP_UPLOAD_VERSION_KEY] = upload_version + 1
        st_module.session_state.pop(_FULL_BACKUP_CHECK_KEY, None)
        st_module.session_state.global_status_message = (
            f"✅ Full backup restored. The data it replaced is kept as {escape_markdown(kept.name)}."
        )
        if schedule_let_go:
            st_module.session_state.global_status_message += (
                " The schedule on screen isn't in the restored history, so its scores can't be entered: load a "
                "schedule from History, or generate a new one."
            )
        st_module.rerun()


# A skill rating choice that saves nothing, so a player nobody rated stays unrated.
_UNRATED = "Unrated"


def _preset_widget_key(kind: str, preset_name: str) -> str:
    """The key of one preset's widget of this kind ("text", "save", "delete", ...).

    Preset names are free text, so their widgets get a namespace no other
    widget's key starts with: keyed "preset_<name>", a preset named "name"
    took the Main tab's "Preset name:" box key and crashed this tab, and every
    tab drawn after it, in every session.
    """
    return f"pm_preset_{kind}::{preset_name}"


_PRESET_WIDGET_KEY_PREFIX = _preset_widget_key("text", "")
# Preset text boxes were keyed "<this><name>" before; the state of a session
# that outlives that change is carried over to the new keys.
_LEGACY_PRESET_WIDGET_KEY_PREFIX = "preset_"


def _prune_stale_preset_widget_state(session_state, presets) -> None:
    """Drop preset text box state whose preset no longer exists in config, after moving any under a legacy key."""
    legacy_keys = [
        key
        for key in list(session_state.keys())
        if key.startswith(_LEGACY_PRESET_WIDGET_KEY_PREFIX) and key != PRESET_NAME_KEY
    ]
    for key in legacy_keys:
        move_widget_state(session_state, key, _preset_widget_key("text", key[len(_LEGACY_PRESET_WIDGET_KEY_PREFIX) :]))
    stale_keys = [
        key
        for key in list(session_state.keys())
        if key.startswith(_PRESET_WIDGET_KEY_PREFIX) and key[len(_PRESET_WIDGET_KEY_PREFIX) :] not in presets
    ]
    for key in stale_keys:
        forget_widget_state(session_state, key)


def save_presets_change(st_module, change) -> bool:
    """Apply one preset change to a copy of the config, save it, and adopt it only if the save worked.

    Preset textareas never write into the shared app_config while rendering,
    so an abandoned edit can't reach disk through some other save; and a
    failed save leaves the in-memory config exactly as it was.
    """
    updated = deepcopy(st_module.session_state.app_config)
    change(updated.setdefault("player_presets", {}))
    if not st_module.session_state.config_manager.save_config(updated):
        return False
    st_module.session_state.app_config = updated
    return True


def preset_saved_elsewhere(st_module, name) -> bool:
    """Whether a preset with this name is saved on disk, e.g. added by another session since this one loaded.

    This session's config copy lacks such a preset, so adding one under the
    same name would look like a new entry to the save's merge and replace
    that roster without a word.
    """
    saved = st_module.session_state.config_manager.read_saved_config()
    return saved is not None and name in saved.get("player_presets", {})


def render_player_management_tab(st_module, player_manager_cls):
    """Render the advanced player management tab."""
    # Values queued by this tab's button handlers on the previous run, applied
    # before the widgets they target are instantiated.
    apply_deferred_widget_values(st_module.session_state)

    st_module.subheader("👥 Advanced Player Management")

    substitution_enabled = st_module.checkbox(
        "Enable player substitutions",
        value=st_module.session_state.get("substitution_enabled", False),
    )

    if substitution_enabled:
        player_manager_cls.create_substitution_interface(st_module)

        st_module.markdown("### ℹ️ How Substitutions Work")
        substitution_round = st_module.session_state.get("substitution_round", 4)
        st_module.info(
            "\n".join(
                [
                    f"- **First Half Players**: Available for rounds 1-{substitution_round}",
                    f"- **Second Half Players**: Available for rounds {substitution_round + 1}-end",
                    "- **Overlap**: Players in both lists can play throughout",
                    "- **Scheduling**: Algorithm ensures smooth transitions",
                ]
            )
        )

    st_module.markdown("### 📋 Player Preset Management")

    config = st_module.session_state.app_config
    presets = config.get("player_presets", {})

    _prune_stale_preset_widget_state(st_module.session_state, presets)

    for preset_name, players in list(presets.items()):
        preset_label = escape_markdown(preset_name)
        with st_module.expander(f"Preset: {preset_label} ({len(players)} players)"):
            col1, col2 = st_module.columns([3, 1])

            with col1:
                if preset_name in RESERVED_PRESET_NAMES:
                    # Saved before such names were refused. Kept as it is, but
                    # say why the Main tab can't load it.
                    st_module.warning(
                        f"⚠️ '{preset_label}' is also one of the Main tab's built-in choices, so picking it there "
                        "loads that choice, not this preset. Add these players as a new preset under another name."
                    )
                preset_text = "\n".join(players)
                widget_key = _preset_widget_key("text", preset_name)
                # If the preset was overwritten elsewhere (e.g. "Save as Preset"
                # on the Main Scheduler tab) this textarea would keep showing -
                # and write back - the old roster.
                seed_widget_from_source(st_module.session_state, widget_key, preset_text, rebase_line_edits)
                players_text = st_module.text_area(f"Players in {preset_label}:", key=widget_key)

            with col2:
                if st_module.button("💾 Save", key=_preset_widget_key("save", preset_name)):
                    roster = [p.strip() for p in players_text.split("\n") if p.strip()]
                    if len(roster) > MAX_PLAYERS:
                        st_module.error(
                            f"❌ {too_many_players_message(len(roster))} Remove some names before saving the preset."
                        )
                    elif save_presets_change(st_module, lambda presets: presets.__setitem__(preset_name, roster)):
                        st_module.session_state.global_status_message = f"✅ Saved preset: {preset_label}"
                        st_module.rerun()
                    else:
                        st_module.error(f"❌ Failed to save preset: {preset_label}")
                delete_pending_key = f"_delete_preset_confirm_pending_{preset_name}"
                if not st_module.session_state.get(delete_pending_key, False):
                    if st_module.button("🗑️ Delete", key=_preset_widget_key("delete", preset_name)):
                        st_module.session_state[delete_pending_key] = True
                        st_module.rerun()
                else:
                    st_module.warning(f"⚠️ Delete preset '{preset_label}'? This cannot be undone.")
                    if st_module.button("✅ Confirm Delete", key=_preset_widget_key("confirm_delete", preset_name)):
                        if save_presets_change(st_module, lambda presets: presets.pop(preset_name, None)):
                            if st_module.session_state.get("selected_player_preset") == preset_name:
                                set_player_preset(st_module.session_state, "Custom")
                            st_module.session_state.pop(delete_pending_key, None)
                            st_module.rerun()
                        else:
                            st_module.error(f"❌ Failed to delete preset: {preset_label}")
                    if st_module.button("Cancel", key=_preset_widget_key("cancel_delete", preset_name)):
                        st_module.session_state.pop(delete_pending_key, None)
                        st_module.rerun()

    st_module.markdown("### ➕ Add New Preset")
    col1, col2 = st_module.columns([2, 1])

    with col1:
        new_preset_name = st_module.text_input("New preset name:", key="new_preset_name")
        new_preset_players = st_module.text_area("Players (one per line):", key="new_preset_players")

    with col2:
        if st_module.button("Add Preset"):
            new_preset_name = new_preset_name.strip()
            if not new_preset_name or not new_preset_players:
                st_module.warning("⚠️ Enter both a preset name and at least one player before adding.")
            elif is_reserved_preset_name(new_preset_name):
                st_module.error(reserved_preset_name_message(new_preset_name))
            elif new_preset_name in config["player_presets"]:
                st_module.error(
                    f"❌ A preset named '{escape_markdown(new_preset_name)}' already exists. "
                    "Edit it above instead, or choose a different name."
                )
            elif preset_saved_elsewhere(st_module, new_preset_name):
                st_module.error(
                    f"❌ A preset named '{escape_markdown(new_preset_name)}' already exists - it was saved from another "
                    "tab or device. Reload the page to see it, or choose a different name."
                )
            else:
                roster = [p.strip() for p in new_preset_players.split("\n") if p.strip()]
                if len(roster) > MAX_PLAYERS:
                    st_module.error(
                        f"❌ {too_many_players_message(len(roster))} Remove some names before saving the preset."
                    )
                elif save_presets_change(st_module, lambda presets: presets.__setitem__(new_preset_name, roster)):
                    st_module.session_state.global_status_message = (
                        f"✅ Added preset: {escape_markdown(new_preset_name)}"
                    )
                    defer_widget_value(st_module.session_state, "new_preset_name", "")
                    defer_widget_value(st_module.session_state, "new_preset_players", "")
                    st_module.rerun()
                else:
                    st_module.error(f"❌ Failed to add preset: {escape_markdown(new_preset_name)}")

    skill_manager = st_module.session_state.get("skill_manager")
    if skill_manager is not None:
        st_module.markdown("### 🎯 Skill Ratings")
        st_module.info(
            "Rate each player 1 (beginner) to 5 (advanced) to see team-balance "
            "info in Analytics. Players left Unrated aren't saved. Ratings don't "
            "affect schedule generation itself."
        )

        saved_skills, skills_problem = skill_manager.read_skills()
        # Save replaces the file, so it must not act on a problem the user
        # hasn't seen: one that arose after the page was drawn shows only now.
        problem_shown_before = st_module.session_state.get("_skill_ratings_problem_shown")
        if skills_problem:
            st_module.warning(f"⚠️ {escape_markdown(skills_problem)}")
        st_module.session_state["_skill_ratings_problem_shown"] = skills_problem
        skills = saved_skills or {}

        current_players = st_module.session_state.get("current_players") or []
        rateable_players = sorted(set(current_players) | set(skills))

        if rateable_players:
            rating_options = [_UNRATED, *range(skill_manager.MIN_RATING, skill_manager.MAX_RATING + 1)]
            new_ratings = {}
            for player in rateable_players:
                slider_key = f"skill_rating_{player}"
                if saved_skills is not None or slider_key not in st_module.session_state:
                    # Track the saved rating, keeping a rating the user is
                    # changing. While the file can't be read, sliders already
                    # shown keep their ratings, and a new one is tracked at the
                    # Unrated it starts at: once the file is readable it picks up
                    # the saved rating - unless the user has moved it meanwhile.
                    seed_widget_from_source(
                        st_module.session_state,
                        slider_key,
                        skills.get(player, _UNRATED),
                        lambda _saved_before, edited, _saved_now: edited,
                    )
                rating = st_module.select_slider(
                    escape_markdown(player),
                    options=rating_options,
                    key=slider_key,
                )
                # Only ratings someone set are saved. Save replaces the whole
                # file, so moving a player back to Unrated clears their rating.
                if rating != _UNRATED:
                    new_ratings[player] = rating

            if st_module.button("💾 Save Skill Ratings"):
                if skills_problem and skills_problem != problem_shown_before:
                    st_module.error(
                        "❌ Skill ratings not saved: the warning above is new since the page was drawn. "
                        "Click 💾 Save Skill Ratings again to save anyway."
                    )
                elif skill_manager.save_skills(new_ratings):
                    st_module.session_state.global_status_message = "✅ Skill ratings saved!"
                    st_module.rerun()
                else:
                    st_module.error("❌ Could not save skill ratings. Please try again.")
        else:
            st_module.info("Generate a schedule first, or add players, to rate their skill.")


def _score_widget_generation(session_state, schedule_id) -> int:
    return int(session_state.get(f"_score_widget_gen_{schedule_id}", 0))


def _score_seed_key(schedule_id, generation: int) -> str:
    return f"_score_seed_{schedule_id}_{generation}"


def refresh_score_widgets(session_state, schedule_id) -> None:
    """Give one schedule's score inputs new keys so they re-read the database.

    Deleting a keyed widget's session_state entry is not enough: the browser
    still holds the old value and sends it back on the next interaction, so
    the widget would keep showing (and a later Save would re-write) its stale
    value. A new key is a new widget, which the Leaderboard tab seeds with
    the stored score again.
    """
    generation_key = f"_score_widget_gen_{schedule_id}"
    session_state[generation_key] = _score_widget_generation(session_state, schedule_id) + 1
    prefixes = tuple(f"{name}_{schedule_id}_" for name in ("score_team1", "score_team2", "score_remove", "_score_seed"))
    for state_key in [key for key in list(session_state.keys()) if key.startswith(prefixes)]:
        del session_state[state_key]


def _recorded_for_game(row, game) -> bool:
    """Whether a game_scores row names this game's teams, in the same team1/team2 order."""
    recorded = [sorted(map(str, row.get(team) or [])) for team in ("team1_players", "team2_players")]
    return recorded == [sorted(game["team1"]), sorted(game["team2"])]


def _score_game_id(game) -> str:
    """Identify one game for its score inputs: its slot plus a digest of its teams."""
    teams = json.dumps([list(game["team1"]), list(game["team2"])])
    digest = hashlib.sha1(teams.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"r{game['round_num']}c{game['court']}_{digest}"


def _score_widget_keys(schedule_id, generation: int, game_id: str) -> tuple[str, str, str]:
    """One game's input keys: its team1 and team2 scores, and the box to remove a score stored for other players."""
    suffix = f"{schedule_id}_{generation}_{game_id}"
    return (f"score_team1_{suffix}", f"score_team2_{suffix}", f"score_remove_{suffix}")


def _game_label(game) -> str:
    return f"Round {game['round_num']}, Court {game['court']}"


def _changed_scores(score_inputs):
    """Split score entries into the edits to save and the games whose entry can't be saved.

    Only games whose inputs differ from what they started with this render
    generation are saved - unedited games are left alone, so a score never
    entered isn't stored as a 0-0 tie, and a stale session can't overwrite
    a score another session saved since. An edit needs both scores (and not
    0-0, which no game ends on); clearing both removes a saved score, and so
    does leaving the game blank with its remove box ticked (for a score stored
    for other players, which the inputs don't show).
    """
    to_save = []
    invalid_games = []
    for game, scores, seed, remove_stored in score_inputs:
        if tuple(scores) == tuple(seed):
            if remove_stored:
                to_save.append((game, (None, None)))
            continue
        team1_score, team2_score = scores
        if (team1_score is None) != (team2_score is None) or (team1_score == 0 and team2_score == 0):
            invalid_games.append(_game_label(game))
            continue
        to_save.append((game, (team1_score, team2_score)))
    return to_save, invalid_games


def _scores_not_saved_message(result, to_save) -> str:
    """Explain why save_game_scores refused or failed (see ScoreSaveResult)."""
    if result.status == "changed":
        stale_games = ", ".join(f"Round {round_num}, Court {court}" for round_num, court in result.stale_games)
        return (
            "❌ Nothing saved. This schedule was changed in another session since you opened it, and these "
            f"games no longer have the players shown here: {stale_games}. Load it again from History, then "
            "enter the scores again."
        )
    if result.status == "missing":
        return (
            "❌ Nothing saved. This schedule was deleted from History. Restore it from History > Recently "
            "deleted to record its scores."
        )
    return (
        "❌ Could not save score(s) for: " + ", ".join(_game_label(game) for game, _ in to_save) + ". Please try again."
    )


def render_leaderboard_tab(st_module, pd_module):
    """Render score entry for the current schedule plus the all-time win/loss leaderboard."""
    st_module.subheader("🏆 Leaderboard")

    history_manager = st_module.session_state.get("history_manager")
    if history_manager is None:
        st_module.error(HISTORY_UNAVAILABLE_MESSAGE)
        return
    schedule = st_module.session_state.get("current_schedule")
    schedule_id = st_module.session_state.get("current_schedule_id")

    if schedule and schedule_id is not None:
        stored_scores = history_manager.get_game_scores(schedule_id)
        # While the saved scores can't be read, score entry is read-only rather
        # than hidden: an input that isn't drawn loses whatever was typed in it.
        scores_unreadable = stored_scores is None
        if scores_unreadable:
            st_module.error(
                "❌ Couldn't read this schedule's saved scores, so scores can't be entered or saved right now "
                f"(anything already typed is kept). {_READ_FAILED_HINT}"
            )
        # Saving refuses a schedule deleted from History; say so before any
        # scores are typed, not only when Save is pressed.
        schedule_entry = history_manager.get_schedule(schedule_id)
        if schedule_entry is not None and schedule_entry.get("deleted_at"):
            st_module.warning(
                "⚠️ This schedule was deleted from History, so its scores can't be saved. Restore it first, "
                "with ↩️ Undo Delete or from Recently deleted on the History tab, then enter its scores."
            )
        elif schedule_entry is None and stored_scores is not None:
            # Not there, while the database reads fine: purged after it was deleted.
            st_module.warning(
                "⚠️ This schedule is no longer in History (it was deleted and then purged), so its scores "
                "can't be saved."
            )
        _schedule_helpers = import_module_with_fallback("app_schedule_helpers")
        list_games_for_scoring = _schedule_helpers.list_games_for_scoring
        blank_score_sheet_csv = _schedule_helpers.blank_score_sheet_csv
        games = list_games_for_scoring(schedule)
        existing_scores = {(row["round_num"], row["court"]): row for row in stored_scores or []}

        with st_module.expander("📋 Bulk import / export scores"):
            st_module.download_button(
                "⬇️ Download blank score sheet",
                # With a byte-order mark, or Excel reads it as Windows-1252, garbling
                # accented names, and the import then can't match their team cells.
                data=blank_score_sheet_csv(games).encode("utf-8-sig"),
                file_name="score_sheet.csv",
                mime="text/csv",
            )
            csv_text = st_module.text_area(
                "Paste filled-in CSV (round,court,team1,team2,team1_score,team2_score)",
                key="score_csv_import",
                disabled=scores_unreadable,
            )
            if st_module.button("📥 Import CSV", disabled=scores_unreadable):
                if not csv_text:
                    st_module.warning("⚠️ Paste CSV data first.")
                else:
                    result = history_manager.import_scores_csv(schedule_id, games, csv_text)
                    # Rerun only when something was imported: nothing else changed otherwise.
                    if result["applied"]:
                        message = f"✅ Imported {result['applied']} score(s)."
                        if result["errors"]:
                            message += f" Skipped {len(result['errors'])} row(s):"
                        st_module.session_state.global_status_message = message
                        # The rerun clears what this run draws, so the skipped rows are shown on the next.
                        if result["errors"]:
                            queue_status_notice(
                                st_module.session_state, "error", _import_errors_markdown(result["errors"])
                            )
                        _update_elo_ratings(st_module.session_state, history_manager)
                        # Re-key the score inputs so they show the imported scores
                        # instead of their pre-import values.
                        refresh_score_widgets(st_module.session_state, schedule_id)
                        st_module.rerun()
                    elif result["errors"]:
                        st_module.error(_import_errors_markdown(result["errors"]))

        with st_module.expander("📝 Enter scores for the current schedule", expanded=True):
            st_module.caption("Leave a game blank until it's played. Clear both boxes to remove a saved score.")
            generation = _score_widget_generation(st_module.session_state, schedule_id)
            # What each input started from this generation, so Save can tell
            # which games the user actually edited (see _changed_scores).
            seeds = st_module.session_state.setdefault(_score_seed_key(schedule_id, generation), {})
            game_ids = [_score_game_id(game) for game in games]
            if any(
                key not in st_module.session_state
                for game_id in game_ids
                if game_id in seeds
                for key in _score_widget_keys(schedule_id, generation, game_id)[:2]
            ):
                # Streamlit drops an input's session_state when a run ends
                # without drawing it (a button on an earlier tab called
                # st.rerun()). Seeding it again would show the stored score
                # while the browser's form still holds - and Save would send -
                # whatever was typed into it before; with new keys the browser
                # drops that too, so Save sends what the boxes show.
                refresh_score_widgets(st_module.session_state, schedule_id)
                generation = _score_widget_generation(st_module.session_state, schedule_id)
                seeds = st_module.session_state.setdefault(_score_seed_key(schedule_id, generation), {})
            score_inputs = []
            # A form, so typing a score doesn't rerun the whole app (every tab)
            # for each box - only Save does. Enter (Go on a phone keyboard)
            # doesn't submit it, since after a game's first box that would try
            # to save the game half-entered.
            with st_module.form(f"score_entry_{schedule_id}", enter_to_submit=False, border=False):
                for game, game_id in zip(games, game_ids):
                    if scores_unreadable and game_id not in seeds:
                        # Not drawn before, so nothing is typed in it - and blank
                        # boxes would pass for a game that hasn't been played.
                        continue
                    # Keyed by the game's teams as well as its slot, so a game a
                    # replan changed gets a fresh input (seeded from the database)
                    # rather than the one of the game it replaced.
                    team1_label = " & ".join(escape_markdown(player) for player in game["team1"])
                    team2_label = " & ".join(escape_markdown(player) for player in game["team2"])
                    team1_key, team2_key, remove_key = _score_widget_keys(schedule_id, generation, game_id)

                    st_module.markdown(
                        f"**Round {game['round_num']}, Court {game['court']}**: {team1_label} vs {team2_label}"
                    )
                    row = existing_scores.get((game["round_num"], game["court"]))
                    recorded = recorded_result(row)
                    remove_stored = False
                    if row and recorded is not None and not _recorded_for_game(row, game):
                        # Saved for a game a replan replaced: it isn't this game's
                        # score, so it isn't shown as one.
                        stored = f"{recorded[0]}-{recorded[1]}"
                        st_module.caption(
                            f"⚠️ The {stored} stored for this court was recorded for other players, so it isn't "
                            "shown here, though it still counts in the standings. If another session replanned "
                            "this schedule, load it again from History first; otherwise saving this game's score "
                            "replaces it."
                        )
                        remove_stored = st_module.checkbox(
                            f"Remove the stored {stored} when saving", key=remove_key, disabled=scores_unreadable
                        )
                        recorded = None
                    elif recorded is not None and min(recorded) < 0:
                        # Imported before negative scores were refused. A score box
                        # can't show it: it drops a value below its minimum of 0.
                        stored = f"{recorded[0]} to {recorded[1]}"
                        st_module.caption(
                            f"⚠️ The score stored for this game, {stored}, can't be right, so it isn't shown here, "
                            "though it still counts in the standings. Saving this game's score replaces it."
                        )
                        remove_stored = st_module.checkbox(
                            f"Remove the stored {stored} when saving", key=remove_key, disabled=scores_unreadable
                        )
                        recorded = None
                    seed = seeds.setdefault(game_id, recorded or (None, None))
                    for key, value in zip((team1_key, team2_key), seed):
                        # Seeded through session_state, not value=: an input whose
                        # default isn't None can't be emptied in the browser (it snaps
                        # back to the default), and emptying is how a score is removed.
                        if key not in st_module.session_state:
                            st_module.session_state[key] = value
                    col1, col2 = st_module.columns(2)
                    with col1:
                        team1_score = st_module.number_input(
                            team1_label,
                            min_value=0,
                            step=1,
                            value=None,
                            placeholder="not played",
                            key=team1_key,
                            disabled=scores_unreadable,
                        )
                    with col2:
                        team2_score = st_module.number_input(
                            team2_label,
                            min_value=0,
                            step=1,
                            value=None,
                            placeholder="not played",
                            key=team2_key,
                            disabled=scores_unreadable,
                        )
                    score_inputs.append((game, (team1_score, team2_score), seed, remove_stored))

                if st_module.form_submit_button("💾 Save Scores", disabled=scores_unreadable):
                    to_save, invalid_games = _changed_scores(score_inputs)
                    if invalid_games:
                        st_module.error(
                            "❌ Nothing saved. Enter both scores (a game can't end 0-0), or clear both to "
                            "remove a score, for: " + ", ".join(invalid_games) + "."
                        )
                    elif not to_save:
                        st_module.info("No score changes to save.")
                    else:
                        save_result = history_manager.save_game_scores(
                            schedule_id, [(game, *scores) for game, scores in to_save]
                        )
                        if save_result.saved:
                            _update_elo_ratings(st_module.session_state, history_manager)
                            refresh_score_widgets(st_module.session_state, schedule_id)
                            st_module.session_state.global_status_message = f"✅ Saved {len(to_save)} score(s)."
                            st_module.rerun()
                        else:
                            st_module.error(_scores_not_saved_message(save_result, to_save))
    elif schedule and unsaved_schedule(st_module.session_state) is not None:
        st_module.warning(
            "⚠️ This schedule isn't saved to history yet, so its scores can't be entered. Use "
            "'💾 Retry saving to history' above the schedule on the Main Scheduler tab."
        )
    else:
        st_module.info("Generate a schedule first to enter scores for it.")

    st_module.markdown("### 📊 All-Time Standings")

    leaderboard = history_manager.get_leaderboard()
    elo_manager = st_module.session_state.get("elo_manager")
    if elo_manager is not None:
        # While the scores can't be read, a recompute can only fail.
        if st_module.button("🔄 Recompute ELO ratings", disabled=leaderboard is None):
            try:
                elo_manager.recompute_from_history(history_manager)
            except RuntimeError:
                st_module.error("❌ Couldn't recompute the ELO ratings, so they were left unchanged. Please try again.")
            else:
                st_module.session_state.global_status_message = "✅ ELO ratings recomputed!"
                st_module.rerun()

    if leaderboard is None:
        st_module.error(f"❌ Couldn't read the recorded scores, so the standings aren't shown. {_READ_FAILED_HINT}")
        return
    if not leaderboard:
        st_module.info("No scores recorded yet. Enter scores above to build the leaderboard.")
        return

    elo_ratings = elo_manager.load_ratings() if elo_manager is not None else {}

    leaderboard_data = []
    for entry in leaderboard:
        row = {"Player": excel_safe(entry["player"])}
        if elo_ratings:
            row["ELO"] = round(elo_ratings.get(entry["player"], 1000))
        row["Wins"] = entry["wins"]
        row["Losses"] = entry["losses"]
        row["Games Played"] = entry["games_played"]
        row["Win Rate"] = f"{entry['win_rate'] * 100:.0f}%"
        leaderboard_data.append(row)

    df = pd_module.DataFrame(leaderboard_data)
    st_module.dataframe(df, width="stretch")


# Set while a Stress Test sweep runs. Streamlit stops a sweep part-way - when
# something on the page is changed or clicked, Stop is pressed or the
# connection drops - by raising out of its next progress update, so the key
# outlives such a sweep and lets the next run say its results are partial.
STRESS_TEST_RUN_KEY = "stress_test_run_in_progress"
# Sweeps that can take longer than this need confirming first: while one runs,
# every other generation on the server gets less of the processor.
STRESS_TEST_CONFIRM_SECONDS = 600
_STRESS_TEST_COURT_MODES = {
    "max": "As many as the players fill (players ÷ 4)",
    "one_fewer": "One fewer, so more players sit out",
    "all": "Every count from 1 up to players ÷ 4",
}
_STRESS_TEST_CANCEL_CAPTION = (
    "⏳ Changing anything on the page while the sweep runs, pressing Stop or losing the connection stops it. "
    "The runs it finished are kept."
)


def _format_duration(seconds: float) -> str:
    """Roughly how long `seconds` is, e.g. '45 s', '12 min' or '3.5 h'."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def render_stress_test_tab(st_module, scheduler_cls, validate_schedule_integrity_fn, pd_module):
    """Sweep player/court/round/constraint combinations through the real
    generation path so bugs that only show up under specific configurations
    (not just the one you happen to click 'Generate' with) surface here first.
    """
    try:
        from src.app_stress_test import (
            FAILURE_STATUSES,
            WARNING_STATUSES,
            count_stress_test_runs,
            run_stress_test,
            summarize_stress_test,
        )
    except ImportError:
        from app_stress_test import (
            FAILURE_STATUSES,
            WARNING_STATUSES,
            count_stress_test_runs,
            run_stress_test,
            summarize_stress_test,
        )

    st_module.subheader("🧪 Stress Test")
    st_module.caption(
        "Runs the same generation code the Main Scheduler button uses, across many "
        "player counts, court counts, round counts, and randomized won't-play constraints, to find "
        "configurations that break."
    )

    col1, col2 = st_module.columns(2)
    with col1:
        min_players = st_module.number_input("Min players", min_value=4, max_value=40, value=4, step=1)
        max_players = st_module.number_input("Max players", min_value=4, max_value=40, value=16, step=1)
        player_step = st_module.number_input("Player step", min_value=1, max_value=10, value=2, step=1)
        min_rounds = st_module.number_input("Min rounds", min_value=1, max_value=20, value=2, step=1)
        max_rounds = st_module.number_input("Max rounds", min_value=1, max_value=20, value=8, step=1)

    with col2:
        max_time = st_module.number_input(
            "Max generation time per trial (seconds)", min_value=1, max_value=60, value=3, step=1
        )
        num_pair_constraints = st_module.number_input(
            "Random 'won't pair' constraints per trial", min_value=0, max_value=10, value=1, step=1
        )
        num_oppose_constraints = st_module.number_input(
            "Random 'won't oppose' constraints per trial", min_value=0, max_value=10, value=1, step=1
        )
        trials_per_combo = st_module.number_input(
            "Trials per combination (different random seed each)", min_value=1, max_value=10, value=1, step=1
        )
        court_mode = st_module.selectbox(
            "Courts for each player count",
            options=list(_STRESS_TEST_COURT_MODES),
            format_func=_STRESS_TEST_COURT_MODES.get,
        )

    player_counts = list(range(int(min_players), int(max_players) + 1, int(player_step)))
    round_counts = list(range(int(min_rounds), int(max_rounds) + 1))
    total_runs = count_stress_test_runs(player_counts, round_counts, int(trials_per_combo), court_mode)
    # Each run searches for up to max_time, then may spend a tenth of that (at
    # least a quarter second) plus half a second repairing the schedule. The
    # rest of the extra second covers setting the run up.
    longest_seconds = total_runs * (1.1 * float(max_time) + 1)
    st_module.info(
        f"This will run {total_runs} generation(s), taking up to about {_format_duration(longest_seconds)} "
        f"(each searches for up to {int(max_time)} s, many stop sooner)."
    )
    confirmed = True
    if longest_seconds > STRESS_TEST_CONFIRM_SECONDS:
        confirmed = st_module.checkbox(
            f"Run it anyway: this sweep can take up to about {_format_duration(longest_seconds)}, and while it "
            "runs, schedule generation is slower for everyone using this server"
        )

    current_params = (
        tuple(player_counts),
        tuple(round_counts),
        float(max_time),
        int(num_pair_constraints),
        int(num_oppose_constraints),
        int(trials_per_combo),
        court_mode,
    )

    run_clicked = st_module.button("▶️ Run Stress Test", type="primary")
    sweep_area = st_module.container()
    # A container drawn again keeps showing its old contents until they are
    # redrawn, so the previous results would stay on screen while a sweep
    # runs. On the run that starts one they go in a placeholder instead, left
    # empty until the sweep ends.
    results_placeholder = st_module.empty() if run_clicked else None

    if run_clicked:
        with sweep_area:
            if total_runs == 0:
                # An inverted or empty range (e.g. Min players > Max players)
                # yields zero combinations - running anyway would silently wipe
                # any previously-displayed valid sweep results with nothing.
                st_module.warning(
                    "⚠️ This configuration produces 0 runs - check that Min players/rounds "
                    "aren't set higher than Max players/rounds."
                )
            elif not confirmed:
                st_module.warning("⚠️ Tick the box above to confirm this long sweep, then click Run again.")
            else:
                cancel_caption = st_module.empty()
                cancel_caption.caption(_STRESS_TEST_CANCEL_CAPTION)
                progress_bar = st_module.progress(0)
                progress_text = st_module.empty()

                def _on_stress_progress(completed, total, _bar=progress_bar, _text=progress_text):
                    fraction = completed / total if total else 1.0
                    _bar.progress(min(1.0, fraction))
                    _text.text(f"{completed} of {total} runs complete")

                # run_stress_test adds each row here as its run ends, so a
                # sweep stopped part-way keeps the runs it finished.
                finished_runs: list = []
                st_module.session_state.stress_test_results = finished_runs
                st_module.session_state.stress_test_params = current_params
                st_module.session_state[STRESS_TEST_RUN_KEY] = {"total": total_runs}
                with counted_generation(st_module):
                    run_stress_test(
                        scheduler_cls,
                        validate_schedule_integrity_fn,
                        player_counts=player_counts,
                        round_counts=round_counts,
                        max_time=float(max_time),
                        num_pair_constraints=int(num_pair_constraints),
                        num_oppose_constraints=int(num_oppose_constraints),
                        trials_per_combo=int(trials_per_combo),
                        court_mode=court_mode,
                        progress_callback=_on_stress_progress,
                        results=finished_runs,
                    )
                st_module.session_state.pop(STRESS_TEST_RUN_KEY, None)
                cancel_caption.empty()

    results = st_module.session_state.get("stress_test_results")
    interrupted_run = st_module.session_state.get(STRESS_TEST_RUN_KEY)
    if interrupted_run and len(results or []) >= interrupted_run["total"]:
        interrupted_run = None  # stopped after its last run, so nothing is missing
    results_area = st_module.container() if results_placeholder is None else results_placeholder.container()
    with results_area:
        if interrupted_run:
            finished = len(results or [])
            shown = ", so the results below are only the runs it finished" if finished else ""
            st_module.warning(
                f"⚠️ The last sweep was interrupted after {finished} of {interrupted_run['total']} runs{shown}. "
                "This happens when something on the page is changed or clicked, Stop is pressed, or the "
                "connection drops while it runs. Click '▶️ Run Stress Test' to run it again."
            )
        if not results:
            return

        if st_module.session_state.get("stress_test_params") != current_params:
            st_module.warning(
                "⚠️ These results are from a previous configuration. Click "
                "'▶️ Run Stress Test' to refresh with the settings above."
            )

        summary = summarize_stress_test(results)
        failures = sum(summary[status] for status in FAILURE_STATUSES)
        warnings = sum(summary[status] for status in WARNING_STATUSES)

        if failures:
            st_module.error(f"🚨 {failures} of {summary['total']} runs failed.")
        elif not interrupted_run:
            st_module.success(f"✅ All {summary['total']} runs passed.")
        if warnings:
            st_module.warning(
                f"⚠️ {warnings} of {summary['total']} runs broke 'won't pair' or 'won't oppose' constraints "
                "that no round of their size can meet, so some breaks were unavoidable. They don't count as failures, "
                "and whether they broke more than they had to isn't checked."
            )

        cols = st_module.columns(4)
        cols[0].metric("Total", summary["total"])
        cols[1].metric("OK", summary["ok"])
        cols[2].metric("Warnings", warnings)
        cols[3].metric("Failed", failures)

        table_rows = [
            {
                "players": r["players"],
                "rounds": r["rounds"],
                "courts": r["courts"],
                "pair_constraints": r["pair_constraints"],
                "oppose_constraints": r["oppose_constraints"],
                "seed": r["seed"],
                "status": r["status"],
                "constraint_violations": r.get("constraint_violations"),
                "games_range": r.get("games_range"),
                "total_range": r["total_range"],
                "generations": r["generations"],
                "time_seconds": r["time_seconds"],
            }
            for r in results
        ]
        st_module.dataframe(pd_module.DataFrame(table_rows), width="stretch")

        for heading, statuses in (
            ("### ❌ Failure details", FAILURE_STATUSES),
            ("### ⚠️ Warning details", WARNING_STATUSES),
        ):
            detail_rows = [r for r in results if r["status"] in statuses]
            if not detail_rows:
                continue
            st_module.markdown(heading)
            for r in detail_rows:
                label = (
                    f"{r['players']} players, {r['courts']} court(s), {r['rounds']} rounds, seed {r['seed']} "
                    f"({r['pair_constraints']} pair / {r['oppose_constraints']} oppose constraints) - {r['status']}"
                )
                with st_module.expander(label):
                    for err in r["errors"]:
                        st_module.write(f"- {err}")
                    if r.get("traceback"):
                        st_module.code(r["traceback"])
