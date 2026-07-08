#!/usr/bin/env python3
"""Session state keys used throughout the Streamlit app.

Instead of sprinkling magic strings throughout the codebase, all session_state
keys are defined here for easy discovery, refactoring, and documentation.
"""

# User authentication & profile
AUTHENTICATED = "authenticated"
AUTH_TIMESTAMP = "auth_timestamp"
SIMPLE_AUTH = "simple_auth"
USER = "user"
USER_ROLE = "user_role"

# Player management
CURRENT_PLAYERS = "current_players"
CUSTOM_PLAYERS = "custom_players"
SELECTED_PLAYER_PRESET = "selected_player_preset"
SELECTED_PLAYER_PRESET_INITIALIZED = "selected_player_preset_initialized"
NEW_PRESET_NAME = "new_preset_name"
NEW_PRESET_PLAYERS = "new_preset_players"
QUICK_ADD = "quick_add"
FIRST_HALF_PLAYERS = "first_half_players"
SECOND_HALF_PLAYERS = "second_half_players"
SUBSTITUTION_ENABLED = "substitution_enabled"
SUBSTITUTION_ROUND = "substitution_round"

# Schedule generation & display
CURRENT_SCHEDULE = "current_schedule"
CURRENT_METRICS = "current_metrics"
CURRENT_ROUND_TIMES = "current_round_times"
CURRENT_SEED = "current_seed"
SEED = "seed"

# Configuration & managers
APP_CONFIG = "app_config"
CONFIG_MANAGER = "config_manager"
HISTORY_MANAGER = "history_manager"

# UI state & persistence
GLOBAL_STATUS_MESSAGE = "global_status_message"
STRESS_TEST_RESULTS = "stress_test_results"

# Internal state (prefixed with _ to indicate implementation detail)
_CONSTRAINT_WIDGET_VERSION = "_constraint_widget_version"
_PENDING_CLEAR_NEW_PRESET_INPUTS = "_pending_clear_new_preset_inputs"
_PENDING_CLEAR_QUICK_ADD = "_pending_clear_quick_add"
_PENDING_PLAYERS_INPUT = "_pending_players_input"
_PRESERVE_SAVED_CONSTRAINTS_FROM_CONFIG = "_preserve_saved_constraints_from_config"
_SYNC_MAIN_CONSTRAINTS_FROM_CONFIG = "_sync_main_constraints_from_config"
