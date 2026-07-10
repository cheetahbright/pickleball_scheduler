#!/usr/bin/env python3
"""Tab renderers extracted from the main Streamlit app."""

from __future__ import annotations

import logging
from datetime import time as time_type
from typing import Any

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

_app_managers = import_module_with_fallback("app_managers")
normalize_config_constraints = _app_managers.normalize_config_constraints
normalize_constraint_pairs = _app_managers.normalize_constraint_pairs
serialize_constraint_pairs = _app_managers.serialize_constraint_pairs
export_config_json = _app_managers.export_config_json
import_config_json = _app_managers.import_config_json

logger = logging.getLogger(__name__)


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


def _selected_history_row_index(selection_event, row_count: int) -> int | None:
    """Extract the selected row index from a st.dataframe selection event, if any.

    The selection is tied to the dataframe's persistent widget key, while
    ``recent`` is recomputed fresh every run - a delete (or any action that
    shrinks the list) can leave a stale positional index pointing past the
    end of the new list.
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


def render_history_tab(st_module, pd_module, json_module):
    """Render the schedule history tab."""
    st_module.subheader("📅 Schedule History")

    history_manager = st_module.session_state.history_manager

    st_module.markdown("### 📚 Recent Schedules")
    recent = history_manager.get_recent_schedules(20)

    if recent:
        history_data = []
        for entry in recent:
            history_data.append(
                {
                    "Date": entry["timestamp"][:16].replace("T", " "),
                    "Players": entry["num_players"],
                    "Rounds": entry["num_rounds"],
                    "Algorithm": entry["algorithm"],
                    "Fairness": f"{entry['fairness_score']:.1f}/10",
                }
            )

        df = pd_module.DataFrame(history_data)
        selection_event = st_module.dataframe(
            df,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="history_table",
        )
        selected_index = _selected_history_row_index(selection_event, len(recent))
        selected_entry = recent[selected_index] if selected_index is not None else None

        col1, col2 = st_module.columns(2)
        with col1:
            load_label = "🔄 Load Selected Schedule" if selected_entry is not None else "🔄 Load Latest Schedule"
            if st_module.button(load_label):
                try:
                    target_entry = selected_entry if selected_entry is not None else recent[0]
                    schedule_data, players = load_schedule_and_players_from_history_entry(target_entry, json_module)
                    st_module.session_state.current_schedule = schedule_data
                    st_module.session_state.current_players = players
                    # Point score entry at the loaded schedule - otherwise the
                    # Leaderboard tab would keep attaching scores to whichever
                    # schedule was generated last.
                    st_module.session_state.current_schedule_id = target_entry.get("id")
                    # Reset companion state computed for the previously shown
                    # schedule so the loaded one doesn't display someone
                    # else's metrics, seed, or round times.
                    st_module.session_state.current_metrics = {}
                    st_module.session_state.current_seed = None
                    st_module.session_state.current_round_times = None
                    st_module.session_state.global_status_message = "✅ Schedule loaded!"
                    logger.info("History load: rounds=%d players=%d", len(schedule_data), len(players))
                    st_module.rerun()
                except Exception as exc:
                    logger.exception("History load failed")
                    st_module.error(f"Failed to load schedule: {exc}")

        with col2:
            if selected_entry is not None and st_module.button("🗑️ Delete Selected Schedule"):
                if history_manager.delete_schedule(selected_entry["id"]):
                    st_module.session_state.last_deleted_id = selected_entry["id"]
                    st_module.session_state.global_status_message = "🗑️ Schedule deleted."
                    st_module.rerun()
                else:
                    st_module.error("Failed to delete schedule.")
    else:
        st_module.info("No schedules in history yet. Generate some schedules to build history.")

    last_deleted_id = st_module.session_state.get("last_deleted_id")
    if last_deleted_id is not None and st_module.button("↩️ Undo Delete"):
        if history_manager.restore_schedule(last_deleted_id):
            st_module.session_state.last_deleted_id = None
            st_module.session_state.global_status_message = "↩️ Schedule restored."
            st_module.rerun()
        else:
            st_module.error("Failed to restore schedule.")

    deleted = history_manager.get_deleted_schedules(10)
    if deleted:
        with st_module.expander("🗑️ Recently deleted"):
            for entry in deleted:
                col1, col2 = st_module.columns([3, 1])
                with col1:
                    st_module.write(f"{entry['timestamp'][:16].replace('T', ' ')} — {entry['num_players']} players")
                with col2:
                    if st_module.button("↩️ Restore", key=f"restore_{entry['id']}"):
                        if history_manager.restore_schedule(entry["id"]):
                            st_module.session_state.global_status_message = "↩️ Schedule restored."
                            st_module.rerun()

    st_module.markdown("### 🤝 Weekly Partner History")
    partner_history = history_manager.get_weekly_partners()

    if partner_history:
        for week, pairs in sorted(partner_history.items(), reverse=True)[:4]:
            with st_module.expander(f"Week of {week}"):
                partner_counts = {}
                for p1, p2 in pairs:
                    for player in [p1, p2]:
                        if player not in partner_counts:
                            partner_counts[player] = set()
                        partner_counts[player].add(p1 if player == p2 else p2)

                for player, partners in sorted(partner_counts.items()):
                    st_module.write(f"**{player}**: {', '.join(sorted(partners))}")


def _render_pairing_heatmap(st_module, schedule_analytics_cls, schedule, players):
    """Render a partner/opponent count heatmap, toggled by the user."""
    if not players:
        return

    view = st_module.radio("Pairing view:", ["Partners", "Opponents"], key="pairing_heatmap_view")
    partner_counts, opponent_counts = schedule_analytics_cls.build_pairing_matrices(schedule, players)
    matrix = partner_counts if view == "Partners" else opponent_counts

    heatmap = schedule_analytics_cls.create_pairing_heatmap(matrix, players, f"{view} Heatmap")
    if heatmap:
        st_module.plotly_chart(heatmap, width="stretch")


def _render_pairing_matrix_fallback(st_module, pd_module, schedule_analytics_cls, schedule, players):
    """Render the pairing matrix as a plain table when plotly is unavailable."""
    if not players:
        return

    st_module.markdown("### 🤝 Pairing Counts")
    view = st_module.radio("Pairing view:", ["Partners", "Opponents"], key="pairing_matrix_view")
    partner_counts, opponent_counts = schedule_analytics_cls.build_pairing_matrices(schedule, players)
    matrix = partner_counts if view == "Partners" else opponent_counts

    df = pd_module.DataFrame(matrix, index=players, columns=players)
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

        metrics = schedule_analytics_cls.calculate_fairness_metrics(schedule, num_players, num_rounds, num_courts)
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
        total_games = sum(stats["games_played"] for stats in metrics.get("player_stats", {}).values())
        st_module.metric("Total Games", total_games)

    if has_plotly:
        st_module.markdown("### 🎯 Fairness Analysis")

        col1, col2 = st_module.columns(2)

        with col1:
            player_chart = schedule_analytics_cls.create_player_stats_chart(metrics)
            if player_chart:
                st_module.plotly_chart(player_chart, width="stretch")

        with col2:
            _render_pairing_heatmap(st_module, schedule_analytics_cls, schedule, players)
    else:
        st_module.info("📊 Install plotly for interactive fairness charts: `pip install plotly`")
        _render_pairing_matrix_fallback(st_module, pd_module, schedule_analytics_cls, schedule, players)

    st_module.markdown("### 👥 Player Details")
    if metrics.get("player_stats"):
        stats_data = []
        for player, stats in metrics["player_stats"].items():
            stats_data.append(
                {
                    "Player": player,
                    "Games": stats["games_played"],
                    "Partners": len(stats["partners"]),
                    "Opponents": len(stats["opponents"]),
                    "Courts": len(stats["courts_used"]),
                    "Partner List": ", ".join(sorted(stats["partners"])),
                    "Opponent List": ", ".join(sorted(stats["opponents"])),
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
            "Team 1": " & ".join(row["team1"]),
            "Team 1 Skill": row["team1_skill_total"],
            "Team 2": " & ".join(row["team2"]),
            "Team 2 Skill": row["team2_skill_total"],
            "Imbalance": row["imbalance"],
            "Fully Rated": row["fully_rated"],
        }
        for row in balance_rows
    ]
    df = pd_module.DataFrame(balance_data)
    st_module.dataframe(df, width="stretch")

    most_imbalanced = max(balance_rows, key=lambda row: row["imbalance"])
    if most_imbalanced["imbalance"] > 0:
        st_module.info(
            f"Most imbalanced: Round {most_imbalanced['round_num']}, Court {most_imbalanced['court']} "
            f"(skill difference of {most_imbalanced['imbalance']})"
        )


def _clear_configuration_widget_state(session_state) -> None:
    """Drop keyed weight-slider widget state so sliders re-seed from config.

    Keyed widgets ignore their ``value=`` argument once the key exists in
    session_state, so after importing a config the weight sliders would keep
    showing the old values - and this tab writes the displayed values
    straight back into the config, silently reverting the import. (The
    do_not_pair_input/do_not_oppose_input textareas handle this themselves
    via a seed fingerprint, since they can also go stale outside of import -
    see the "_do_not_pair_seed" check in render_configuration_tab.)
    """
    stale_keys = [key for key in list(session_state.keys()) if key.startswith("weight_")]
    for state_key in stale_keys:
        session_state.pop(state_key, None)


def render_configuration_tab(st_module):
    """Render the configuration and constraints tab."""
    st_module.subheader("⚙️ Configuration")

    config = normalize_config_constraints(st_module.session_state.app_config)
    st_module.session_state.app_config = config

    st_module.markdown("### 🎚️ Objective Weights")
    st_module.info("Adjust the importance of different scheduling objectives")

    objectives = config["objectives"]

    for obj_name, obj_data in objectives.items():
        col1, col2 = st_module.columns([3, 1])

        with col1:
            new_weight = st_module.slider(
                obj_name.replace("_", " ").title(),
                min_value=obj_data["min"],
                max_value=obj_data["max"],
                value=obj_data["weight"],
                step=100,
                key=f"weight_{obj_name}",
            )
            objectives[obj_name]["weight"] = new_weight

        with col2:
            st_module.metric("Current", new_weight)

    st_module.markdown("### 🎯 Player Constraints")

    col1, col2 = st_module.columns(2)

    with col1:
        st_module.markdown("**Do Not Pair Together**")
        do_not_pair_text = serialize_constraint_pairs(config["constraints"].get("do_not_pair", []))
        # st.tabs() runs every tab's body on every rerun, not just the visible
        # one - so once this keyed widget exists, it ignores value= even when
        # do_not_pair changed elsewhere (Main Scheduler quick-add, a loaded
        # history entry). Re-seed it when the config-side value has changed
        # since the last render; otherwise this tab's own write-back below
        # keeps the seed in lockstep with in-progress typed edits.
        if st_module.session_state.get("_do_not_pair_seed") != do_not_pair_text:
            st_module.session_state.do_not_pair_input = do_not_pair_text
        st_module.session_state._do_not_pair_seed = do_not_pair_text
        do_not_pair = st_module.text_area(
            "Players who should not be paired (format: Player1,Player2):",
            value=do_not_pair_text,
            height=100,
            key="do_not_pair_input",
        )
        config["constraints"]["do_not_pair"] = normalize_constraint_pairs(do_not_pair)

    with col2:
        st_module.markdown("**Do Not Oppose Each Other**")
        do_not_oppose_text = serialize_constraint_pairs(config["constraints"].get("do_not_oppose", []))
        if st_module.session_state.get("_do_not_oppose_seed") != do_not_oppose_text:
            st_module.session_state.do_not_oppose_input = do_not_oppose_text
        st_module.session_state._do_not_oppose_seed = do_not_oppose_text
        do_not_oppose = st_module.text_area(
            "Players who should not oppose each other (format: Player1,Player2):",
            value=do_not_oppose_text,
            height=100,
            key="do_not_oppose_input",
        )
        config["constraints"]["do_not_oppose"] = normalize_constraint_pairs(do_not_oppose)

    st_module.markdown("### ⏰ Scheduling Preferences")

    col1, col2, col3 = st_module.columns(3)

    with col1:
        default_rounds = st_module.number_input(
            "Default rounds:",
            min_value=1,
            max_value=20,
            value=config["scheduling"]["default_rounds"],
        )
        config["scheduling"]["default_rounds"] = default_rounds

    with col2:
        selected_start_time = st_module.time_input(
            "Default start time:",
            value=time_type.fromisoformat(config["scheduling"]["start_time"]),
        )
        if selected_start_time is not None:
            config["scheduling"]["start_time"] = selected_start_time.strftime("%H:%M")

    with col3:
        selected_end_time = st_module.time_input(
            "Default end time:",
            value=time_type.fromisoformat(config["scheduling"]["end_time"]),
        )
        if selected_end_time is not None:
            config["scheduling"]["end_time"] = selected_end_time.strftime("%H:%M")

    if st_module.button("💾 Save Configuration"):
        saved = st_module.session_state.config_manager.save_config(config)
        st_module.session_state.app_config = config
        if saved:
            st_module.session_state._constraint_widget_version = (
                int(st_module.session_state.get("_constraint_widget_version", 0)) + 1
            )
            st_module.session_state._sync_main_constraints_from_config = True
            st_module.session_state._preserve_saved_constraints_from_config = True
            st_module.session_state.global_status_message = "✅ Configuration saved!"
            st_module.rerun()

    st_module.markdown("### 💼 Backup / Restore")
    st_module.download_button(
        "⬇️ Download config",
        data=export_config_json(config),
        file_name="app_config.json",
        mime="application/json",
    )
    uploaded = st_module.file_uploader("⬆️ Upload config", type=["json"], key="config_upload")
    if uploaded is not None:
        imported, messages = import_config_json(
            uploaded.getvalue(), st_module.session_state.config_manager.default_config
        )
        for message in messages:
            st_module.warning(message)
        if st_module.session_state.config_manager.save_config(imported):
            st_module.session_state.app_config = imported
            st_module.session_state._constraint_widget_version = (
                int(st_module.session_state.get("_constraint_widget_version", 0)) + 1
            )
            st_module.session_state._sync_main_constraints_from_config = True
            st_module.session_state._preserve_saved_constraints_from_config = True
            _clear_configuration_widget_state(st_module.session_state)
            st_module.session_state.global_status_message = "✅ Configuration imported!"
            st_module.rerun()


_PRESET_WIDGET_KEY_PREFIX = "preset_"
_PRESET_SEED_KEY_PREFIX = "_preset_seed_"
# Widget keys that share the "preset_" prefix but belong to other widgets.
_PRESET_WIDGET_RESERVED_KEYS = {"preset_name"}


def _prune_stale_preset_widget_state(session_state, presets) -> None:
    """Drop preset_* widget and seed keys whose preset no longer exists in config."""
    stale_keys = [
        key
        for key in list(session_state.keys())
        if (
            key.startswith(_PRESET_WIDGET_KEY_PREFIX)
            and key not in _PRESET_WIDGET_RESERVED_KEYS
            and key[len(_PRESET_WIDGET_KEY_PREFIX) :] not in presets
        )
        or (key.startswith(_PRESET_SEED_KEY_PREFIX) and key[len(_PRESET_SEED_KEY_PREFIX) :] not in presets)
    ]
    for key in stale_keys:
        del session_state[key]


def render_player_management_tab(st_module, player_manager_cls):
    """Render the advanced player management tab."""
    st_module.subheader("👥 Advanced Player Management")

    substitution_enabled = st_module.checkbox(
        "Enable player substitutions",
        value=st_module.session_state.get("substitution_enabled", False),
    )

    if substitution_enabled:
        player_manager_cls.create_substitution_interface()

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
        with st_module.expander(f"Preset: {preset_name} ({len(players)} players)"):
            col1, col2 = st_module.columns([3, 1])

            with col1:
                preset_text = "\n".join(players)
                widget_key = f"{_PRESET_WIDGET_KEY_PREFIX}{preset_name}"
                seed_key = f"{_PRESET_SEED_KEY_PREFIX}{preset_name}"
                # Keyed widgets ignore value= once the key exists, so if the
                # preset was overwritten elsewhere (e.g. "Save as Preset" on
                # the Main Scheduler tab) the textarea would keep showing -
                # and write back - the old roster. Re-seed it when the
                # config-side value changed since the last render.
                if st_module.session_state.get(seed_key) != preset_text and widget_key in st_module.session_state:
                    st_module.session_state[widget_key] = preset_text
                st_module.session_state[seed_key] = preset_text
                players_text = st_module.text_area(
                    f"Players in {preset_name}:",
                    value=preset_text,
                    key=widget_key,
                )
                config["player_presets"][preset_name] = [p.strip() for p in players_text.split("\n") if p.strip()]

            with col2:
                if st_module.button("💾 Save", key=f"save_preset_{preset_name}"):
                    if st_module.session_state.config_manager.save_config(config):
                        st_module.session_state.global_status_message = f"✅ Saved preset: {preset_name}"
                        st_module.rerun()
                if st_module.button("🗑️ Delete", key=f"delete_{preset_name}"):
                    del config["player_presets"][preset_name]
                    if st_module.session_state.config_manager.save_config(config):
                        if st_module.session_state.get("selected_player_preset") == preset_name:
                            st_module.session_state.selected_player_preset = "Custom"
                        st_module.rerun()

    st_module.markdown("### ➕ Add New Preset")
    col1, col2 = st_module.columns([2, 1])

    with col1:
        if st_module.session_state.pop("_pending_clear_new_preset_inputs", False):
            st_module.session_state["new_preset_name"] = ""
            st_module.session_state["new_preset_players"] = ""
        new_preset_name = st_module.text_input("New preset name:", key="new_preset_name")
        new_preset_players = st_module.text_area("Players (one per line):", key="new_preset_players")

    with col2:
        if st_module.button("Add Preset") and new_preset_name and new_preset_players:
            config["player_presets"][new_preset_name] = [p.strip() for p in new_preset_players.split("\n") if p.strip()]
            if st_module.session_state.config_manager.save_config(config):
                st_module.session_state.global_status_message = f"✅ Added preset: {new_preset_name}"
                st_module.session_state._pending_clear_new_preset_inputs = True
                st_module.rerun()

    skill_manager = st_module.session_state.get("skill_manager")
    if skill_manager is not None:
        st_module.markdown("### 🎯 Skill Ratings")
        st_module.info(
            "Rate each player 1 (beginner) to 5 (advanced) to see team-balance "
            "info in Analytics. Ratings don't affect schedule generation itself."
        )

        current_players = st_module.session_state.get("current_players") or []
        rateable_players = sorted(set(current_players) | set(skill_manager.load_skills().keys()))

        if rateable_players:
            skills = skill_manager.load_skills()
            new_ratings = {}
            for player in rateable_players:
                new_ratings[player] = st_module.slider(
                    player,
                    min_value=skill_manager.MIN_RATING,
                    max_value=skill_manager.MAX_RATING,
                    value=skills.get(player, 3),
                    key=f"skill_rating_{player}",
                )

            if st_module.button("💾 Save Skill Ratings"):
                if skill_manager.save_skills(new_ratings):
                    st_module.session_state.global_status_message = "✅ Skill ratings saved!"
                    st_module.rerun()
        else:
            st_module.info("Generate a schedule first, or add players, to rate their skill.")


def _clear_score_widget_state(session_state, schedule_id) -> None:
    """Drop the keyed score-entry widget state for one schedule.

    Keyed widgets ignore their ``value=`` argument once the key exists in
    session_state, so after a bulk CSV import the number_inputs would keep
    showing their pre-import values instead of the imported scores.
    """
    prefixes = (f"score_team1_{schedule_id}_", f"score_team2_{schedule_id}_")
    for state_key in [key for key in list(session_state.keys()) if key.startswith(prefixes)]:
        del session_state[state_key]


def render_leaderboard_tab(st_module, pd_module):
    """Render score entry for the current schedule plus the all-time win/loss leaderboard."""
    st_module.subheader("🏆 Leaderboard")

    history_manager = st_module.session_state.history_manager
    schedule = st_module.session_state.get("current_schedule")
    schedule_id = st_module.session_state.get("current_schedule_id")

    if schedule and schedule_id is not None:
        _schedule_helpers = import_module_with_fallback("app_schedule_helpers")
        list_games_for_scoring = _schedule_helpers.list_games_for_scoring
        blank_score_sheet_csv = _schedule_helpers.blank_score_sheet_csv
        games = list_games_for_scoring(schedule)
        existing_scores = {
            (row["round_num"], row["court"]): row for row in history_manager.get_game_scores(schedule_id)
        }

        with st_module.expander("📋 Bulk import / export scores"):
            st_module.download_button(
                "⬇️ Download blank score sheet",
                data=blank_score_sheet_csv(games),
                file_name="score_sheet.csv",
                mime="text/csv",
            )
            csv_text = st_module.text_area(
                "Paste filled-in CSV (round,court,team1,team2,team1_score,team2_score)",
                key="score_csv_import",
            )
            if st_module.button("📥 Import CSV") and csv_text:
                result = history_manager.import_scores_csv(schedule_id, games, csv_text)
                if result["applied"]:
                    st_module.session_state.global_status_message = f"✅ Imported {result['applied']} score(s)."
                for error in result["errors"]:
                    st_module.error(error)
                if result["applied"]:
                    # The score number_inputs are keyed widgets, so Streamlit
                    # would keep showing their pre-import session values and a
                    # later "Save Scores" would overwrite the imported scores
                    # with them. Drop the keys so they re-seed from the DB.
                    _clear_score_widget_state(st_module.session_state, schedule_id)
                    st_module.rerun()

        with st_module.expander("📝 Enter scores for the current schedule", expanded=True):
            score_inputs = {}
            for i, game in enumerate(games):
                existing = existing_scores.get((game["round_num"], game["court"]))
                team1_label = " & ".join(game["team1"])
                team2_label = " & ".join(game["team2"])

                st_module.markdown(
                    f"**Round {game['round_num']}, Court {game['court']}**: {team1_label} vs {team2_label}"
                )
                col1, col2 = st_module.columns(2)
                with col1:
                    team1_score = st_module.number_input(
                        team1_label,
                        min_value=0,
                        value=existing["team1_score"] if existing and existing["team1_score"] is not None else 0,
                        key=f"score_team1_{schedule_id}_{i}",
                    )
                with col2:
                    team2_score = st_module.number_input(
                        team2_label,
                        min_value=0,
                        value=existing["team2_score"] if existing and existing["team2_score"] is not None else 0,
                        key=f"score_team2_{schedule_id}_{i}",
                    )
                score_inputs[i] = (game, team1_score, team2_score)

            if st_module.button("💾 Save Scores"):
                for game, team1_score, team2_score in score_inputs.values():
                    history_manager.save_game_score(
                        schedule_id,
                        game["round_num"],
                        game["court"],
                        game["team1"],
                        game["team2"],
                        team1_score,
                        team2_score,
                    )
                st_module.session_state.global_status_message = "✅ Scores saved!"
                st_module.rerun()
    else:
        st_module.info("Generate a schedule first to enter scores for it.")

    st_module.markdown("### 📊 All-Time Standings")

    elo_manager = st_module.session_state.get("elo_manager")
    if elo_manager is not None:
        if st_module.button("🔄 Recompute ELO ratings"):
            elo_manager.recompute_from_history(history_manager)
            st_module.rerun()

    leaderboard = history_manager.get_leaderboard()
    if not leaderboard:
        st_module.info("No scores recorded yet. Enter scores above to build the leaderboard.")
        return

    elo_ratings = elo_manager.load_ratings() if elo_manager is not None else {}

    leaderboard_data = []
    for entry in leaderboard:
        row = {"Player": entry["player"]}
        if elo_ratings:
            row["ELO"] = round(elo_ratings.get(entry["player"], 1000))
        row["Wins"] = entry["wins"]
        row["Losses"] = entry["losses"]
        row["Games Played"] = entry["games_played"]
        row["Win Rate"] = f"{entry['win_rate'] * 100:.0f}%"
        leaderboard_data.append(row)

    df = pd_module.DataFrame(leaderboard_data)
    st_module.dataframe(df, width="stretch")


def render_stress_test_tab(st_module, scheduler_cls, validate_schedule_integrity_fn, pd_module):
    """Sweep player/round/constraint combinations through the real generation
    path so bugs that only show up under specific configurations (not just
    the one you happen to click 'Generate' with) surface here first.
    """
    try:
        from src.app_stress_test import run_stress_test, summarize_stress_test
    except ImportError:
        from app_stress_test import run_stress_test, summarize_stress_test

    st_module.subheader("🧪 Stress Test")
    st_module.caption(
        "Runs the same generation code the Main Scheduler button uses, across many "
        "player counts, round counts, and randomized won't-play constraints, to find "
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

    player_counts = list(range(int(min_players), int(max_players) + 1, int(player_step)))
    round_counts = list(range(int(min_rounds), int(max_rounds) + 1))
    total_runs = len(player_counts) * len(round_counts) * int(trials_per_combo)
    st_module.info(f"This will run {total_runs} generation(s). Larger sweeps take longer.")

    if st_module.button("▶️ Run Stress Test", type="primary"):
        with st_module.spinner(f"Running {total_runs} generation(s)..."):
            results = run_stress_test(
                scheduler_cls,
                validate_schedule_integrity_fn,
                player_counts=player_counts,
                round_counts=round_counts,
                max_time=float(max_time),
                num_pair_constraints=int(num_pair_constraints),
                num_oppose_constraints=int(num_oppose_constraints),
                trials_per_combo=int(trials_per_combo),
            )
        st_module.session_state.stress_test_results = results

    results = st_module.session_state.get("stress_test_results")
    if not results:
        return

    summary = summarize_stress_test(results)
    failures = summary["invalid_schedule"] + summary["exception"] + summary["no_schedule"]

    if failures == 0:
        st_module.success(f"✅ All {summary['total']} runs passed.")
    else:
        st_module.error(f"🚨 {failures} of {summary['total']} runs failed.")

    cols = st_module.columns(4)
    cols[0].metric("Total", summary["total"])
    cols[1].metric("OK", summary["ok"])
    cols[2].metric("Invalid schedule", summary["invalid_schedule"])
    cols[3].metric("Exceptions", summary["exception"] + summary["no_schedule"])

    table_rows = [
        {
            "players": r["players"],
            "rounds": r["rounds"],
            "courts": r["courts"],
            "pair_constraints": r["pair_constraints"],
            "oppose_constraints": r["oppose_constraints"],
            "seed": r["seed"],
            "status": r["status"],
            "total_range": r["total_range"],
        }
        for r in results
    ]
    st_module.dataframe(pd_module.DataFrame(table_rows), width="stretch")

    failing_rows = [r for r in results if r["status"] != "ok"]
    if failing_rows:
        st_module.markdown("### ❌ Failure details")
        for r in failing_rows:
            label = (
                f"{r['players']} players, {r['rounds']} rounds, seed {r['seed']} "
                f"({r['pair_constraints']} pair / {r['oppose_constraints']} oppose constraints) - {r['status']}"
            )
            with st_module.expander(label):
                for err in r["errors"]:
                    st_module.write(f"- {err}")
                if r.get("traceback"):
                    st_module.code(r["traceback"])
