#!/usr/bin/env python3
"""Scheduler page and generation flow extracted from the Streamlit app."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

normalize_config_constraints = import_module_with_fallback("app_managers").normalize_config_constraints

logger = logging.getLogger(__name__)


# Deliberately generic placeholder names for the onboarding demo - distinct from
# the protected default roster in docs/summaries/PLAYER_NAMES_POLICY.md.
SAMPLE_PLAYER_NAMES = ["Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper"]


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


def splice_schedules(locked_rounds: list, new_rounds: list) -> list:
    """Combine already-played rounds with freshly generated remaining rounds.

    New rounds are renumbered to continue directly after the locked ones, so
    a replan of rounds 4-6 (after 3 played) ends up numbered 4, 5, 6 - not
    restarting at 1.
    """
    spliced = list(locked_rounds)
    start_round = len(locked_rounds) + 1
    for i, round_data in enumerate(new_rounds):
        if hasattr(round_data, "get"):
            renumbered = dict(round_data)
            renumbered["round"] = start_round + i
        else:
            renumbered = round_data
        spliced.append(renumbered)
    return spliced


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


def build_feasibility_notes(num_players: int, num_courts: int, num_rounds: int) -> list[str]:
    """Advisory notes about whether this configuration can reach a perfectly fair schedule.

    Runs before generation so users are not surprised by a mediocre score after
    waiting up to the full max-time budget for a combination that was never
    going to reach range 0.
    """
    ScheduleFeasibilityAnalyzer = import_module_with_fallback("utils.feasibility_analyzer").ScheduleFeasibilityAnalyzer

    feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(num_players, num_courts, num_rounds)
    notes: list[str] = []

    if feasibility["range_0_possible"]:
        notes.append(f"✅ A perfectly fair schedule (range 0) is mathematically possible for {num_players} players.")
        return notes

    notes.append(
        f"⚠️ With {num_players} players and {num_courts} court(s), perfect fairness (range 0) is not "
        f"mathematically possible for {num_rounds} rounds - expect a minimum total range of "
        f"{feasibility['total_theoretical_min']}."
    )

    better_counts = []
    for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
        candidate = num_players + delta
        if candidate < 4:
            continue
        candidate_feasibility = ScheduleFeasibilityAnalyzer.calculate_theoretical_minimums(candidate, None, num_rounds)
        if candidate_feasibility["range_0_possible"]:
            better_counts.append(candidate)

    if better_counts:
        better_counts = sorted(set(better_counts))[:3]
        counts_text = " or ".join(str(c) for c in better_counts)
        notes.append(f"With {counts_text} players (same round count), range 0 becomes possible.")

    return notes


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

    threshold = max(1, (total_players - 1) // 2)
    for player, count in constrained_counts.items():
        if count >= threshold:
            warnings.append(
                f"{player} is constrained against {count} of {total_players - 1} other players - "
                "this may make scheduling difficult."
            )

    return warnings


def find_player_name_issues(players: list[str]) -> list[str]:
    """Detect duplicate and near-duplicate names before they reach the scheduler.

    An exact duplicate makes the GA treat one physical player as filling
    multiple schedule slots, which later surfaces as a misleading "algorithm
    bug" error (impossible player assignments) instead of a clear input
    problem. Case/spacing collisions ("Bob" vs "bob") are flagged separately
    since they are usually typos rather than intentional distinct players.
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
        key = re.sub(r"\s+", " ", name.strip().lower())
        normalized_groups.setdefault(key, []).append(name)
    for variants in normalized_groups.values():
        distinct_variants = sorted(set(variants))
        if len(distinct_variants) > 1:
            issues.append("Names that look the same but differ in case or spacing: " + ", ".join(distinct_variants))

    return issues


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


def build_constraint_multiselect_defaults(
    constraint_pairs: list[list[str]],
    players: list[str],
    separator: str,
) -> list[str]:
    """Format saved constraints into multiselect labels for the current player list."""
    return [
        f"{first}{separator}{second}" for first, second in constraint_pairs if first in players and second in players
    ]


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
) -> None:
    """Seed constraint widget state from saved config when the UI needs a resync."""
    should_sync_from_config = bool(session_state.pop("_sync_main_constraints_from_config", False))
    do_not_pair_key, do_not_oppose_key = get_constraint_widget_keys(session_state)
    pair_options = [f"{p1} & {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]]
    oppose_options = [f"{p1} vs {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]]

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
        )


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
    if "history_manager" not in st_module.session_state:
        st_module.session_state.history_manager = history_manager_cls()
    if "config_manager" not in st_module.session_state:
        st_module.session_state.config_manager = config_manager_cls()
    if skill_manager_cls is not None and "skill_manager" not in st_module.session_state:
        st_module.session_state.skill_manager = skill_manager_cls()
    if elo_manager_cls is not None and "elo_manager" not in st_module.session_state:
        st_module.session_state.elo_manager = elo_manager_cls()
    if "app_config" not in st_module.session_state:
        CONFIG_REPAIR_MESSAGES_KEY = import_module_with_fallback("app_managers").CONFIG_REPAIR_MESSAGES_KEY

        loaded_config = st_module.session_state.config_manager.load_config()
        repair_messages = loaded_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
        st_module.session_state.app_config = loaded_config
        for message in repair_messages or []:
            st_module.warning(f"⚠️ {message}")

    ui_config = st_module.session_state.app_config.setdefault("ui", {"theme": "light"})

    col1, col2, col3 = st_module.columns([1, 0.3, 0.2])
    with col1:
        st_module.title("🎾 Pickleball Scheduler")
    with col2:
        dark_mode = st_module.checkbox("🌙 Dark mode", value=ui_config.get("theme") == "dark")
        new_theme = "dark" if dark_mode else "light"
        if new_theme != ui_config.get("theme"):
            ui_config["theme"] = new_theme
            st_module.session_state.config_manager.save_config(st_module.session_state.app_config)
            st_module.rerun()
    with col3:
        if st_module.button("Logout"):
            logout_fn()

    inject_theme_css = import_module_with_fallback("theme_styles").inject_theme_css
    inject_theme_css(st_module, ui_config.get("theme", "light"))

    st_module.markdown("*Complete scheduling with analytics and history*")

    global_status_message = st_module.session_state.pop("global_status_message", None)
    if isinstance(global_status_message, str) and global_status_message:
        st_module.success(global_status_message)

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
    if leaderboard_tab_fn is not None:
        tab_labels.append("🏆 Leaderboard")
        tab_fns.append(leaderboard_tab_fn)
    if stress_test_tab_fn is not None:
        tab_labels.append("🧪 Stress Test")
        tab_fns.append(stress_test_tab_fn)

    tabs = st_module.tabs(tab_labels)

    for tab, tab_fn in zip(tabs, tab_fns):
        with tab:
            tab_fn()


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
    pending_players_text = st_module.session_state.pop("_pending_players_input", None)
    if isinstance(pending_players_text, str):
        st_module.session_state.players_input = pending_players_text

    st_module.subheader("👥 Players")

    col1, col2 = st_module.columns([2, 1])

    with col1:
        previous_preset = st_module.session_state.get("selected_player_preset")
        preset_options = build_player_preset_options(st_module.session_state, config)
        preset = st_module.selectbox("Choose preset or custom:", preset_options)
        st_module.session_state.selected_player_preset = preset
        st_module.session_state.selected_player_preset_initialized = True
        default_players = resolve_players_text_for_preset(st_module.session_state, config, preset)

        # Once a widget has a `key`, Streamlit ignores `value=` on every rerun
        # after the first (session_state[key] takes precedence) - so switching
        # presets would otherwise leave the textarea showing the old preset's
        # players. Re-seed it explicitly, but only when the preset selection
        # itself just changed, so it doesn't clobber in-progress manual edits
        # under "Custom" on every unrelated rerun.
        if preset != previous_preset:
            st_module.session_state.players_input = default_players

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

        if st_module.session_state.pop("_pending_clear_quick_add", False):
            st_module.session_state["quick_add"] = ""
        new_player = st_module.text_input("Quick add player:", key="quick_add")
        preset_name = st_module.text_input("Preset name:", key="preset_name")

        if st_module.button("➕ Add Player") and new_player:
            updated_text = add_player_to_text(players_text, new_player)
            if updated_text != players_text:
                st_module.session_state.custom_players = updated_text
                st_module.session_state.selected_player_preset = "Custom"
                st_module.session_state.selected_player_preset_initialized = True
                st_module.session_state._pending_players_input = updated_text
                st_module.session_state._pending_clear_quick_add = True
                st_module.rerun()

        if st_module.button("💾 Save as Preset") and preset_name.strip():
            preset_players = parse_players_text(players_text)
            if preset_players:
                config["player_presets"][preset_name.strip()] = preset_players
                if st_module.session_state.config_manager.save_config(config):
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

    if not players and not st_module.session_state.get("current_schedule"):
        st_module.info("👋 New here? Try it out instantly with sample players.")
        if st_module.button("✨ Try with 8 sample players"):
            sample_players_text = "\n".join(SAMPLE_PLAYER_NAMES)
            st_module.session_state.custom_players = sample_players_text
            st_module.session_state.selected_player_preset = "Custom"
            st_module.session_state.selected_player_preset_initialized = True
            st_module.session_state._pending_players_input = sample_players_text
            st_module.rerun()

    if len(players) < 4:
        st_module.warning("⚠️ Need at least 4 players")
        return

    name_issues = find_player_name_issues(players)
    if name_issues:
        for issue in name_issues:
            st_module.error(f"❌ {issue}")
        return

    st_module.info(f"📊 {len(players)} players")

    st_module.subheader("🚫 Player Constraints")

    col1, col2 = st_module.columns(2)
    sync_constraint_widget_state(st_module.session_state, players, config)
    do_not_pair_key, do_not_oppose_key = get_constraint_widget_keys(st_module.session_state)
    preserve_saved_constraints = bool(st_module.session_state.pop("_preserve_saved_constraints_from_config", False))
    default_do_not_pair = build_constraint_multiselect_defaults(
        config["constraints"]["do_not_pair"],
        players,
        " & ",
    )
    default_do_not_oppose = build_constraint_multiselect_defaults(
        config["constraints"]["do_not_oppose"],
        players,
        " vs ",
    )

    with col1:
        st_module.markdown("**Who should NOT play together:**")
        do_not_pair = st_module.multiselect(
            "Select player pairs that should not be on the same team:",
            options=[f"{p1} & {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]],
            default=default_do_not_pair,
            key=do_not_pair_key,
        )

        if preserve_saved_constraints and not do_not_pair and default_do_not_pair:
            do_not_pair = default_do_not_pair
        config["constraints"]["do_not_pair"] = [pair.split(" & ") for pair in do_not_pair]

    with col2:
        st_module.markdown("**Who should NOT play against each other:**")
        do_not_oppose = st_module.multiselect(
            "Select player pairs that should not be opponents:",
            options=[f"{p1} vs {p2}" for i, p1 in enumerate(players) for p2 in players[i + 1 :]],
            default=default_do_not_oppose,
            key=do_not_oppose_key,
        )

        if preserve_saved_constraints and not do_not_oppose and default_do_not_oppose:
            do_not_oppose = default_do_not_oppose
        config["constraints"]["do_not_oppose"] = [pair.split(" vs ") for pair in do_not_oppose]

    total_constraints = len(config["constraints"]["do_not_pair"]) + len(config["constraints"]["do_not_oppose"])
    if total_constraints > 0:
        st_module.info(f"🎯 Active constraints: {total_constraints} total")

    with st_module.expander("🔎 Find a specific pair instead of scrolling"):
        quick_col1, quick_col2, quick_col3, quick_col4 = st_module.columns([2, 2, 2, 1])
        with quick_col1:
            quick_player_a = st_module.selectbox("Player A", options=players, key="quick_constraint_player_a")
        with quick_col2:
            quick_player_b = st_module.selectbox("Player B", options=players, key="quick_constraint_player_b")
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
                    st_module.session_state.config_manager.save_config(config)
                    st_module.session_state.app_config = config
                    st_module.session_state._constraint_widget_version = (
                        int(st_module.session_state.get("_constraint_widget_version", 0)) + 1
                    )
                    st_module.session_state._sync_main_constraints_from_config = True
                    st_module.session_state._preserve_saved_constraints_from_config = True
                    st_module.session_state.global_status_message = f"✅ {message}"
                    st_module.rerun()
                else:
                    st_module.warning(message)

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
        max_courts = max(1, len(players) // 4)
        courts = st_module.number_input(
            "Courts available:",
            min_value=1,
            max_value=max_courts,
            value=max_courts,
        )

    with col3:
        max_time = st_module.number_input("Max generation time (seconds):", min_value=10, max_value=300, value=60)

    with st_module.expander("📐 Expected quality for this configuration"):
        for note in build_feasibility_notes(len(players), courts, num_rounds):
            st_module.write(note)

    for warning_text in constraint_pressure_warnings(
        players,
        config["constraints"]["do_not_pair"],
        config["constraints"]["do_not_oppose"],
    ):
        st_module.warning(f"⚠️ {warning_text}")

    with st_module.expander("🎲 Advanced: reproducible generation"):
        seed_text = st_module.text_input(
            "Random seed (optional):",
            key="generation_seed",
            help="Reuse the seed shown under a generated schedule to replay that generation run.",
        )
    generation_seed = None
    seed_text = (seed_text or "").strip()
    if seed_text:
        try:
            generation_seed = int(seed_text)
        except ValueError:
            st_module.warning("Seed must be a whole number - ignoring it for this run.")

    if st_module.button("🎯 Generate Enhanced Schedule", type="primary"):
        with st_module.status("Generating schedule...", expanded=True) as status_box:
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

            import time as _time

            generation_started_at = _time.time()
            logger.info(
                "Generation start: players=%d rounds=%d courts=%d " "pair_constraints=%d oppose_constraints=%d seed=%s",
                len(players),
                num_rounds,
                courts,
                len(config["constraints"]["do_not_pair"]),
                len(config["constraints"]["do_not_oppose"]),
                generation_seed,
            )

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

                result = scheduler.generate_schedule(
                    max_time=max_time,
                    seed=generation_seed,
                    progress_callback=_on_progress,
                )

                if not (isinstance(result, dict) and "schedule" in result):
                    logger.warning(
                        "Generation end: status=no_schedule elapsed=%.1fs",
                        _time.time() - generation_started_at,
                    )
                    st_module.error("❌ Could not generate schedule. Try adjusting parameters.")
                    status_box.update(label="❌ Generation failed", state="error")
                    return

                schedule_data = result["schedule"]
                schedule_errors = validate_schedule_integrity_fn(schedule_data, players)

                if schedule_errors:
                    logger.error(
                        "Generation end: status=invalid_schedule elapsed=%.1fs errors=%d",
                        _time.time() - generation_started_at,
                        len(schedule_errors),
                    )
                    st_module.error("🚨 **CRITICAL ERROR: Invalid schedule generated!**")
                    st_module.error("**The algorithm has bugs that generated impossible player assignments!**")
                    for error in schedule_errors:
                        st_module.error(f"❌ {error}")
                    st_module.error("**🛑 This indicates serious bugs in the scheduling algorithm!**")
                    st_module.info("Please report this bug with the exact player configuration.")
                    status_box.update(label="❌ Invalid schedule generated", state="error")
                    return

                progress_bar.progress(1.0)
                status_box.update(label="✅ Done", state="complete")
                st_module.success("✅ Enhanced schedule generated!")

                metrics = schedule_analytics_cls.calculate_fairness_metrics(
                    schedule_data,
                    num_players=len(players),
                    num_rounds=num_rounds,
                    num_courts=courts,
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
                st_module.session_state.current_round_times = round_times

                settings = {
                    "num_rounds": num_rounds,
                    "courts": courts,
                    "max_time": max_time,
                    "constraints": config["constraints"],
                    "seed": result.get("seed"),
                }
                st_module.session_state.current_schedule_id = st_module.session_state.history_manager.save_schedule(
                    schedule_data, players, settings
                )

            except MemoryError:
                logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
                st_module.error("❌ Ran out of memory while generating the schedule. Try fewer players or rounds.")
                status_box.update(label="❌ Out of memory", state="error")
            except ValueError as exc:
                logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
                st_module.error(f"❌ Invalid input: {exc}. Check your players and constraints.")
                status_box.update(label="❌ Invalid input", state="error")
            except Exception as exc:
                logger.exception("Generation end: status=exception elapsed=%.1fs", _time.time() - generation_started_at)
                st_module.error(f"❌ Error ({type(exc).__name__}): {str(exc)}")
                status_box.update(label="❌ Generation error", state="error")

    _render_persistent_schedule(
        st_module,
        datetime_cls,
        json_module,
        scheduler_cls,
        schedule_analytics_cls,
        display_enhanced_schedule_fn,
        schedule_to_csv_fn,
    )


def _render_persistent_schedule(
    st_module,
    datetime_cls,
    json_module,
    scheduler_cls,
    schedule_analytics_cls,
    display_enhanced_schedule_fn,
    schedule_to_csv_fn,
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
    round_times = st_module.session_state.get("current_round_times") or None

    if seed is not None:
        st_module.caption(
            f"🎲 Seed: {seed} - enter it under '🎲 Advanced: reproducible " "generation' to replay this generation run."
        )

    _render_metric_summary(st_module, metrics, players)
    _render_optimality_status(st_module, metrics, players)

    display_enhanced_schedule_fn(schedule_data, players, round_times=round_times)

    _render_download_buttons(
        st_module,
        datetime_cls,
        json_module,
        schedule_analytics_cls,
        schedule_data,
        players,
        round_times,
        schedule_to_csv_fn,
    )

    _render_mid_session_replan(st_module, scheduler_cls, schedule_data, players)


def _render_mid_session_replan(st_module, scheduler_cls, schedule_data, players):
    """Let the user regenerate only the not-yet-played rounds, optionally with a
    changed player list (someone left early / showed up late) - without losing
    the rounds already played.

    The replanned rounds are optimized for fairness among themselves only; they
    do not account for who already played together in the locked rounds, since
    that would require the scheduler's fitness function to accept pre-existing
    partner/opponent/game counts, which it does not currently support.
    """
    total_rounds = len(schedule_data)
    if total_rounds < 2:
        return

    with st_module.expander("🔁 Mid-session replan"):
        st_module.caption(
            "Regenerate the rounds that haven't been played yet - useful if a player "
            "leaves early or joins late. Rounds already played are never changed. The "
            "replanned rounds are freshly fair among themselves, not aware of who "
            "already played together in the rounds you keep."
        )

        played_rounds = st_module.number_input(
            "Rounds already played:",
            min_value=0,
            max_value=total_rounds - 1,
            value=0,
            key="replan_played_rounds",
        )
        # Keyed widgets ignore value= once the key exists in session_state, so
        # without this the textarea would keep showing the roster of whatever
        # schedule was on screen when it first rendered. Re-seed it whenever
        # the current schedule's players change (new generation, history load,
        # completed replan) while preserving in-progress edits otherwise.
        roster_text = "\n".join(players)
        if st_module.session_state.get("_replan_players_seed") != roster_text:
            st_module.session_state.replan_players_input = roster_text
            st_module.session_state._replan_players_seed = roster_text
        remaining_players_text = st_module.text_area(
            "Players for the remaining rounds (edit to reflect who's still here):",
            value=roster_text,
            key="replan_players_input",
        )

        if st_module.button("🔁 Replan remaining rounds"):
            remaining_players = parse_players_text(remaining_players_text)
            if len(remaining_players) < 4:
                st_module.warning("⚠️ Need at least 4 players for the remaining rounds.")
                return

            locked_rounds, remaining_round_count = split_schedule_at_round(schedule_data, played_rounds)
            courts = len(locked_rounds[0].get("games", [])) if locked_rounds else 1

            try:
                scheduler = scheduler_cls(
                    players=remaining_players,
                    num_courts=courts,
                    num_rounds=remaining_round_count,
                    use_desktop_params=True,
                )
                result = scheduler.generate_schedule()
            except Exception as exc:
                logger.exception("Mid-session replan failed")
                st_module.error(f"❌ Replan failed: {exc}")
                return

            if not (isinstance(result, dict) and "schedule" in result):
                st_module.error("❌ Could not replan the remaining rounds. Try adjusting the player list.")
                return

            st_module.session_state.current_schedule = splice_schedules(locked_rounds, result["schedule"])
            st_module.session_state.current_players = sorted(set(players) | set(remaining_players))
            # The metrics and seed shown belong to the pre-replan schedule;
            # after splicing they no longer describe what's on screen.
            st_module.session_state.current_metrics = {}
            st_module.session_state.current_seed = None
            st_module.session_state.global_status_message = f"✅ Replanned rounds {played_rounds + 1}-{total_rounds}."
            st_module.rerun()


def _render_metric_summary(st_module, metrics, players):
    """Render the top-level fairness summary metrics."""
    if not metrics:
        # Schedules loaded from history have no freshly computed metrics -
        # showing all-zero placeholders would read as a terrible schedule.
        return
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
    players,
    round_times,
    schedule_to_csv_fn,
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
        text_data = schedule_to_text(schedule_data, players, round_times)
        st_module.download_button(
            "📋 Download Text",
            text_data,
            f"schedule_{datetime_cls.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
        )
    with col4:
        xlsx_data = schedule_to_xlsx(schedule_data, players, round_times)
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
            st_module.code(schedule_to_player_text(schedule_data, players))
