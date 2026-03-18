#!/usr/bin/env python3
"""Scheduler page and generation flow extracted from the Streamlit app."""

from __future__ import annotations

from typing import Any, Mapping


def parse_players_text(players_text: str) -> list[str]:
    """Normalize textarea input into a clean player list."""
    return [line.strip() for line in players_text.splitlines() if line.strip()]


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


def determine_initial_player_preset(
    session_state: Any,
    config: Mapping[str, Any],
) -> str:
    """Choose a useful default preset for first render without overwriting user intent."""
    preset_names = list(config.get("player_presets", {}).keys())
    preset_options = ["Custom", "Default Names", *preset_names]

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
    preset_options = ["Custom", "Default Names", *preset_names]
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
):
    """Render the top-level scheduler page with tabs."""
    if "history_manager" not in st_module.session_state:
        st_module.session_state.history_manager = history_manager_cls()
    if "config_manager" not in st_module.session_state:
        st_module.session_state.config_manager = config_manager_cls()
    if "app_config" not in st_module.session_state:
        st_module.session_state.app_config = st_module.session_state.config_manager.load_config()

    col1, col2 = st_module.columns([1, 0.2])
    with col1:
        st_module.title("🎾 Pickleball Scheduler")
    with col2:
        if st_module.button("Logout"):
            logout_fn()

    st_module.markdown("*Complete scheduling with analytics and history*")

    global_status_message = st_module.session_state.pop("global_status_message", None)
    if isinstance(global_status_message, str) and global_status_message:
        st_module.success(global_status_message)

    tab1, tab2, tab3, tab4, tab5 = st_module.tabs(
        [
            "🎯 Main Scheduler",
            "📊 Analytics",
            "📅 History",
            "⚙️ Configuration",
            "👥 Player Management",
        ]
    )

    with tab1:
        main_scheduler_tab_fn()

    with tab2:
        analytics_tab_fn()

    with tab3:
        history_tab_fn()

    with tab4:
        configuration_tab_fn()

    with tab5:
        player_management_tab_fn()


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
    config = st_module.session_state.app_config
    pending_players_text = st_module.session_state.pop("_pending_players_input", None)
    if isinstance(pending_players_text, str):
        st_module.session_state.players_input = pending_players_text

    st_module.subheader("👥 Players")

    col1, col2 = st_module.columns([2, 1])

    with col1:
        preset_options = build_player_preset_options(st_module.session_state, config)
        preset = st_module.selectbox("Choose preset or custom:", preset_options)
        st_module.session_state.selected_player_preset = preset
        st_module.session_state.selected_player_preset_initialized = True
        default_players = resolve_players_text_for_preset(st_module.session_state, config, preset)

        players_text = st_module.text_area(
            "Player names (one per line):",
            value=default_players,
            height=150,
            key="players_input",
        )

        if preset == "Custom":
            st_module.session_state.custom_players = players_text

    with col2:
        st_module.markdown("**Quick Actions**")

        new_player = st_module.text_input("Quick add player:", key="quick_add")
        preset_name = st_module.text_input("Preset name:", key="preset_name")

        if st_module.button("➕ Add Player") and new_player:
            updated_text = add_player_to_text(players_text, new_player)
            if updated_text != players_text:
                st_module.session_state.custom_players = updated_text
                st_module.session_state.selected_player_preset = "Custom"
                st_module.session_state.selected_player_preset_initialized = True
                st_module.session_state._pending_players_input = updated_text
                st_module.rerun()

        if st_module.button("💾 Save as Preset") and preset_name.strip():
            preset_players = parse_players_text(players_text)
            if preset_players:
                config["player_presets"][preset_name.strip()] = preset_players
                st_module.session_state.config_manager.save_config(config)
                st_module.success(f"Saved preset: {preset_name.strip()}")

    players = parse_players_text(players_text)

    if players:
        st_module.markdown("**Quick Remove Players:**")
        cols = st_module.columns(min(len(players), 4))
        for i, player in enumerate(players):
            with cols[i % 4]:
                if st_module.button(f"❌ {player}", key=f"remove_{i}"):
                    updated_text = remove_player_from_text(players_text, player)
                    st_module.session_state.custom_players = updated_text
                    st_module.session_state.selected_player_preset = "Custom"
                    st_module.session_state.selected_player_preset_initialized = True
                    st_module.session_state._pending_players_input = updated_text
                    st_module.rerun()

    if len(players) < 4:
        st_module.warning("⚠️ Need at least 4 players")
        return

    st_module.info(f"📊 {len(players)} players")

    st_module.subheader("🚫 Player Constraints")

    col1, col2 = st_module.columns(2)

    with col1:
        st_module.markdown("**Who should NOT play together:**")
        do_not_pair = st_module.multiselect(
            "Select player pairs that should not be on the same team:",
            options=[f"{p1} & {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]],
            default=[
                f"{pair[0]} & {pair[1]}"
                for pair in config["constraints"]["do_not_pair"]
                if pair[0] in players and pair[1] in players
            ],
            key="do_not_pair",
        )

        config["constraints"]["do_not_pair"] = [pair.split(" & ") for pair in do_not_pair]

    with col2:
        st_module.markdown("**Who should NOT play against each other:**")
        do_not_oppose = st_module.multiselect(
            "Select player pairs that should not be opponents:",
            options=[f"{p1} vs {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]],
            default=[
                f"{pair[0]} vs {pair[1]}"
                for pair in config["constraints"]["do_not_oppose"]
                if pair[0] in players and pair[1] in players
            ],
            key="do_not_oppose",
        )

        config["constraints"]["do_not_oppose"] = [pair.split(" vs ") for pair in do_not_oppose]

    total_constraints = len(config["constraints"]["do_not_pair"]) + len(config["constraints"]["do_not_oppose"])
    if total_constraints > 0:
        st_module.info(f"🎯 Active constraints: {total_constraints} total")

    st_module.subheader("⚙️ Schedule Settings")
    col1, col2, col3 = st_module.columns(3)

    with col1:
        num_rounds = st_module.number_input(
            "Number of rounds:",
            min_value=1,
            max_value=20,
            value=config["scheduling"]["default_rounds"],
        )

    with col2:
        courts = max(1, len(players) // 4)
        st_module.metric("Courts needed", courts)

    with col3:
        max_time = st_module.number_input("Max generation time (seconds):", min_value=10, max_value=300, value=60)

    total_constraints = len(config["constraints"]["do_not_pair"]) + len(config["constraints"]["do_not_oppose"])
    if total_constraints > 0:
        st_module.info(f"🎯 Active constraints: {total_constraints} total")

    if st_module.button("🎯 Generate Enhanced Schedule", type="primary"):
        with st_module.spinner("Creating optimized schedule..."):
            try:
                st_module.session_state.config_manager.save_config(config)

                scheduler = scheduler_cls(
                    players=players,
                    num_courts=courts,
                    num_rounds=num_rounds,
                    use_desktop_params=True,
                    max_runtime=max_time,
                    max_generations=max(2000, int(max_time * 70)),
                )

                constraints = config["constraints"]
                if constraints["do_not_pair"] or constraints["do_not_oppose"]:
                    scheduler.add_constraints(
                        pair_constraints=[
                            tuple(pair) if isinstance(pair, list) else pair for pair in constraints["do_not_pair"]
                        ],
                        oppose_constraints=[
                            tuple(pair) if isinstance(pair, list) else pair for pair in constraints["do_not_oppose"]
                        ],
                    )

                result = scheduler.generate_schedule(max_time=max_time)

                if not (isinstance(result, dict) and "schedule" in result):
                    st_module.error("❌ Could not generate schedule. Try adjusting parameters.")
                    return

                schedule_data = result["schedule"]
                schedule_errors = validate_schedule_integrity_fn(schedule_data, players)

                if schedule_errors:
                    st_module.error("🚨 **CRITICAL ERROR: Invalid schedule generated!**")
                    st_module.error("**The algorithm has bugs that generated impossible player assignments!**")
                    for error in schedule_errors:
                        st_module.error(f"❌ {error}")
                    st_module.error("**🛑 This indicates serious bugs in the scheduling algorithm!**")
                    st_module.info("Please report this bug with the exact player configuration.")
                    return

                st_module.success("✅ Enhanced schedule generated!")

                st_module.session_state.current_schedule = schedule_data
                st_module.session_state.current_players = players

                metrics = schedule_analytics_cls.calculate_fairness_metrics(
                    schedule_data,
                    num_players=len(players),
                    num_rounds=num_rounds,
                    num_courts=courts,
                )

                _render_metric_summary(st_module, metrics, players)
                _render_optimality_status(st_module, metrics, players)

                display_enhanced_schedule_fn(schedule_data, players)

                settings = {
                    "num_rounds": num_rounds,
                    "max_time": max_time,
                    "constraints": config["constraints"],
                }
                st_module.session_state.history_manager.save_schedule(schedule_data, players, settings)

                _render_download_buttons(
                    st_module,
                    datetime_cls,
                    json_module,
                    schedule_analytics_cls,
                    schedule_data,
                    schedule_to_csv_fn,
                )

            except Exception as exc:
                st_module.error(f"❌ Error: {str(exc)}")


def _render_metric_summary(st_module, metrics, players):
    """Render the top-level fairness summary metrics."""
    col1, col2, col3, col4, col5 = st_module.columns(5)
    with col1:
        fairness_score = metrics.get("overall_fairness", 0)
        is_optimal = metrics.get("is_mathematically_optimal", False)
        fairness_label = "Fairness Score"
        if is_optimal is True:
            fairness_label += " ✅"
        elif is_optimal is False and metrics.get("theoretical_optimum") is not None:
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


def _render_optimality_status(st_module, metrics, players):
    """Render optimality messaging for the generated schedule."""
    if metrics.get("theoretical_optimum") is None:
        return

    total_range = metrics.get("total_range", 0)
    theoretical_optimum = metrics.get("theoretical_optimum")
    is_optimal = metrics.get("is_mathematically_optimal", False)

    if is_optimal:
        st_module.success(
            "✅ **Mathematically Optimal!** Achieved the theoretical minimum "
            f"range of {theoretical_optimum} for {len(players)} players."
        )
        return

    excess = total_range - theoretical_optimum
    st_module.info(
        "⚠️ **Near Optimal:** "
        f"Current range {total_range}, theoretical minimum {theoretical_optimum} "
        f"(excess: +{excess}). This is good performance!"
    )


def _render_download_buttons(
    st_module,
    datetime_cls,
    json_module,
    schedule_analytics_cls,
    schedule_data,
    schedule_to_csv_fn,
):
    """Render CSV and JSON download buttons for the generated schedule."""
    col1, col2 = st_module.columns(2)
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
