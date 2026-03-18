#!/usr/bin/env python3
"""Tab renderers extracted from the main Streamlit app."""

from __future__ import annotations

from datetime import time as time_type
from typing import Any


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
    schedule_data = json_module.loads(entry["schedule_data"])
    players: list[str] = []

    raw_settings = entry.get("settings_data")
    if isinstance(raw_settings, str) and raw_settings:
        try:
            parsed_settings = json_module.loads(raw_settings)
        except Exception:
            parsed_settings = {}
        if isinstance(parsed_settings, dict):
            stored_players = parsed_settings.get("players", [])
            if isinstance(stored_players, list):
                players = [str(player) for player in stored_players if str(player).strip()]

    if not players:
        players = infer_players_from_schedule(schedule_data)

    return schedule_data, players


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
        st_module.dataframe(df, width="stretch")

        if st_module.button("🔄 Load Latest Schedule"):
            try:
                latest = recent[0]
                schedule_data, players = load_schedule_and_players_from_history_entry(latest, json_module)
                st_module.session_state.current_schedule = schedule_data
                st_module.session_state.current_players = players
                st_module.session_state.global_status_message = "✅ Latest schedule loaded!"
                st_module.rerun()
            except Exception as exc:
                st_module.error(f"Failed to load schedule: {exc}")
    else:
        st_module.info("No schedules in history yet. Generate some schedules to build history.")

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


def render_analytics_tab(st_module, pd_module, schedule_analytics_cls, has_plotly):
    """Render the analytics and visualization tab."""
    st_module.subheader("📊 Schedule Analytics")

    if "current_schedule" not in st_module.session_state:
        st_module.info("Generate a schedule first to see analytics")
        return

    schedule = st_module.session_state.current_schedule
    players = st_module.session_state.current_players

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
    else:
        st_module.info("📊 Install plotly for interactive fairness charts: `pip install plotly`")

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


def render_configuration_tab(st_module):
    """Render the configuration and constraints tab."""
    st_module.subheader("⚙️ Configuration")

    config = st_module.session_state.app_config

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
        do_not_pair = st_module.text_area(
            "Players who should not be paired (format: Player1,Player2):",
            value="\n".join(config["constraints"].get("do_not_pair", [])),
            height=100,
            key="do_not_pair_input",
        )
        config["constraints"]["do_not_pair"] = [line.strip() for line in do_not_pair.split("\n") if line.strip()]

    with col2:
        st_module.markdown("**Do Not Oppose Each Other**")
        do_not_oppose = st_module.text_area(
            "Players who should not oppose each other:",
            value="\n".join(config["constraints"].get("do_not_oppose", [])),
            height=100,
            key="do_not_oppose_input",
        )
        config["constraints"]["do_not_oppose"] = [line.strip() for line in do_not_oppose.split("\n") if line.strip()]

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
        st_module.session_state.config_manager.save_config(config)
        st_module.session_state.app_config = config
        st_module.success("✅ Configuration saved!")


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

    for preset_name, players in presets.items():
        with st_module.expander(f"Preset: {preset_name} ({len(players)} players)"):
            col1, col2 = st_module.columns([3, 1])

            with col1:
                players_text = st_module.text_area(
                    f"Players in {preset_name}:",
                    value="\n".join(players),
                    key=f"preset_{preset_name}",
                )
                config["player_presets"][preset_name] = [p.strip() for p in players_text.split("\n") if p.strip()]

            with col2:
                if st_module.button("🗑️ Delete", key=f"delete_{preset_name}"):
                    del config["player_presets"][preset_name]
                    st_module.session_state.config_manager.save_config(config)
                    st_module.rerun()

    st_module.markdown("### ➕ Add New Preset")
    col1, col2 = st_module.columns([2, 1])

    with col1:
        new_preset_name = st_module.text_input("New preset name:")
        new_preset_players = st_module.text_area("Players (one per line):")

    with col2:
        if st_module.button("Add Preset") and new_preset_name and new_preset_players:
            config["player_presets"][new_preset_name] = [p.strip() for p in new_preset_players.split("\n") if p.strip()]
            st_module.session_state.config_manager.save_config(config)
            st_module.success(f"✅ Added preset: {new_preset_name}")
            st_module.rerun()
