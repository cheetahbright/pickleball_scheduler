#!/usr/bin/env python3
"""Compatibility facade for the former monolithic app_managers.py.

Manager logic now lives in focused modules under src/managers/ (history,
config, skill_rating, elo_rating, player), split by concern the same way
main_app.py already documents its own facade role - but this module
intentionally keeps the legacy app_managers import surface stable for
main_app.py, app_scheduler_flow.py, app_tabs.py, and existing tests.

json/os/sqlite3 are re-imported here (not just re-exported) so that
`patch.object(app_managers_module.sqlite3, "connect")`-style test mocking
keeps working: since these are singleton modules in sys.modules, patching
sqlite3.connect here patches the exact same module object src/managers/
history.py's own `import sqlite3` sees.
"""

from __future__ import annotations

import json
import os
import sqlite3

try:
    from src.managers.config import (
        CONFIG_REPAIR_MESSAGES_KEY,
        CONFIG_SCHEMA_VERSION_KEY,
        CURRENT_CONFIG_SCHEMA_VERSION,
        MAX_CONFIG_UPLOAD_BYTES,
        ConfigurationManager,
        export_config_json,
        import_config_json,
        migrate_config,
        normalize_config_constraints,
        normalize_constraint_pairs,
        serialize_constraint_pairs,
        validate_config_shape,
    )
    from src.managers.elo_rating import EloRatingManager
    from src.managers.history import HistoryManager
    from src.managers.player import PlayerManager, validate_player_names
    from src.managers.skill_rating import SkillRatingManager
except ImportError:
    from managers.config import (
        CONFIG_REPAIR_MESSAGES_KEY,
        CONFIG_SCHEMA_VERSION_KEY,
        CURRENT_CONFIG_SCHEMA_VERSION,
        MAX_CONFIG_UPLOAD_BYTES,
        ConfigurationManager,
        export_config_json,
        import_config_json,
        migrate_config,
        normalize_config_constraints,
        normalize_constraint_pairs,
        serialize_constraint_pairs,
        validate_config_shape,
    )
    from managers.elo_rating import EloRatingManager
    from managers.history import HistoryManager
    from managers.player import PlayerManager, validate_player_names
    from managers.skill_rating import SkillRatingManager

__all__ = [
    "CONFIG_REPAIR_MESSAGES_KEY",
    "CONFIG_SCHEMA_VERSION_KEY",
    "CURRENT_CONFIG_SCHEMA_VERSION",
    "MAX_CONFIG_UPLOAD_BYTES",
    "ConfigurationManager",
    "EloRatingManager",
    "HistoryManager",
    "PlayerManager",
    "SkillRatingManager",
    "export_config_json",
    "import_config_json",
    "json",
    "migrate_config",
    "normalize_config_constraints",
    "normalize_constraint_pairs",
    "os",
    "serialize_constraint_pairs",
    "sqlite3",
    "validate_config_shape",
    "validate_player_names",
]
