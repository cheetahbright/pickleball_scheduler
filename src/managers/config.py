#!/usr/bin/env python3
"""Config-schema constraints normalization, validation, migration, and
ConfigurationManager (app_config.json persistence)."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.managers._paths import _resolve_default_names_path, _resolve_storage_path, save_json
except ImportError:
    from managers._paths import _resolve_default_names_path, _resolve_storage_path, save_json

logger = logging.getLogger(__name__)


def _parse_constraint_pair(raw_constraint: Any) -> list[str] | None:
    """Normalize a single stored constraint into a two-player pair."""
    if isinstance(raw_constraint, (list, tuple)) and len(raw_constraint) >= 2:
        first, second = raw_constraint[0], raw_constraint[1]
    elif isinstance(raw_constraint, str):
        stripped_value = raw_constraint.strip()
        if not stripped_value:
            return None

        parts: list[str] | None = None
        for pattern, flags in (
            (r"\s+vs\s+", re.IGNORECASE),
            (r"\s*&\s*", 0),
            (r"\s*,\s*", 0),
        ):
            split_parts = re.split(pattern, stripped_value, maxsplit=1, flags=flags)
            if len(split_parts) == 2:
                parts = split_parts
                break

        if parts is None:
            return None

        first, second = parts[0], parts[1]
    else:
        return None

    normalized_first = str(first).strip()
    normalized_second = str(second).strip()
    if not normalized_first or not normalized_second or normalized_first == normalized_second:
        return None

    return [normalized_first, normalized_second]


def normalize_constraint_pairs(raw_constraints: Any) -> list[list[str]]:
    """Normalize legacy or mixed constraint values into a consistent pair list."""
    if isinstance(raw_constraints, str):
        items = raw_constraints.splitlines()
    elif isinstance(raw_constraints, list):
        items = raw_constraints
    elif isinstance(raw_constraints, tuple):
        items = list(raw_constraints)
    else:
        return []

    normalized_pairs: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for raw_constraint in items:
        parsed_pair = _parse_constraint_pair(raw_constraint)
        if parsed_pair is None:
            continue

        # Sorted so "Alice,Bob" and "Bob,Alice" collapse to the same key -
        # the pair is semantically unordered (do_not_pair/do_not_oppose
        # constraints have no direction), so keeping both as distinct
        # entries would silently duplicate the same constraint.
        pair_key = tuple(sorted((parsed_pair[0], parsed_pair[1])))
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        normalized_pairs.append(parsed_pair)

    return normalized_pairs


def serialize_constraint_pairs(raw_constraints: Any, separator: str = ", ") -> str:
    """Render stored constraint pairs for text-area editing."""
    return "\n".join(f"{first}{separator}{second}" for first, second in normalize_constraint_pairs(raw_constraints))


def normalize_config_constraints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure constraint settings use the pair structure expected by the scheduler UI."""
    constraints = config.setdefault("constraints", {})
    constraints["do_not_pair"] = normalize_constraint_pairs(constraints.get("do_not_pair", []))
    constraints["do_not_oppose"] = normalize_constraint_pairs(constraints.get("do_not_oppose", []))
    return config


def _is_valid_objectives(objectives: Any) -> bool:
    """An objectives section is a non-empty dict of {weight, min, max} int entries."""
    if not isinstance(objectives, dict) or not objectives:
        return False
    for entry in objectives.values():
        if not isinstance(entry, dict):
            return False
        for key in ("weight", "min", "max"):
            if not isinstance(entry.get(key), int) or isinstance(entry.get(key), bool):
                return False
    return True


def _is_valid_scheduling(scheduling: Any) -> bool:
    """A scheduling section has an in-range round count and parseable HH:MM times."""
    if not isinstance(scheduling, dict):
        return False

    default_rounds = scheduling.get("default_rounds")
    if not isinstance(default_rounds, int) or isinstance(default_rounds, bool) or not (1 <= default_rounds <= 20):
        return False

    for key in ("start_time", "end_time"):
        value = scheduling.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
            return False
        hours, minutes = value.split(":")
        if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
            return False

    return True


def _is_valid_player_presets(player_presets: Any) -> bool:
    """player_presets must be dict[str, list[str]] (an empty dict is valid - no presets saved)."""
    if not isinstance(player_presets, dict):
        return False
    for name, players in player_presets.items():
        if not isinstance(name, str) or not isinstance(players, list):
            return False
        if not all(isinstance(player, str) for player in players):
            return False
    return True


_CONFIG_SECTION_VALIDATORS = {
    "objectives": _is_valid_objectives,
    "scheduling": _is_valid_scheduling,
    "player_presets": _is_valid_player_presets,
}

CONFIG_REPAIR_MESSAGES_KEY = "_config_repair_messages"
CONFIG_SCHEMA_VERSION_KEY = "schema_version"

# Bump this and add a new v(N-1)->vN migration function below whenever the
# config shape changes in a way that would otherwise silently drop user data
# on validate_config_shape's default-reset repair path.
CURRENT_CONFIG_SCHEMA_VERSION = 3


def _migrate_v1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
    """v1 configs need no shape change to reach v2 - the version bump alone
    is the migration (v1 predated an unused, since-removed constraint list)."""
    return config


def _migrate_v2_to_v3(config: Dict[str, Any]) -> Dict[str, Any]:
    """v2 configs need no shape change to reach v3 - the version bump alone
    is the migration (v2 predated an unused, since-removed 'ui'/theme section)."""
    return config


# Keyed by the version being migrated FROM (i.e. _CONFIG_MIGRATIONS[1] takes a
# v1 config and returns a v2 config). Applied in order until the config
# reaches CURRENT_CONFIG_SCHEMA_VERSION.
_CONFIG_MIGRATIONS: Dict[int, Any] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], list[int]]:
    """Migrate a config forward to CURRENT_CONFIG_SCHEMA_VERSION in place of resetting it.

    Configs written before schema_version existed are treated as v1. Returns
    (migrated_config, versions_migrated_from) - the second is empty if no
    migration was needed. A config from an unknown future version is left
    untouched (not migrated backward, not reset) with a warning logged by the
    caller; a config stuck on a version with no registered migration path
    also stops there rather than raising.
    """
    version = config.get(CONFIG_SCHEMA_VERSION_KEY, 1)
    if not isinstance(version, int):
        version = 1

    applied: list[int] = []
    while version < CURRENT_CONFIG_SCHEMA_VERSION:
        migration = _CONFIG_MIGRATIONS.get(version)
        if migration is None:
            break
        config = migration(config)
        applied.append(version)
        version += 1

    config[CONFIG_SCHEMA_VERSION_KEY] = version
    return config, applied


def validate_config_shape(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Repair any top-level config section that doesn't match its expected shape.

    app_config.json can be hand-edited, synced, or corrupted; without this, a
    wrong type (e.g. "objectives": "high") would propagate until some tab
    crashed with an opaque TypeError. Constraints are already covered by
    normalize_config_constraints and are not re-checked here. Returns a new
    dict; on any repairs, CONFIG_REPAIR_MESSAGES_KEY holds one human-readable
    message per repaired section for the caller to surface and then discard -
    it must never be persisted.
    """
    repaired = deepcopy(config)
    repaired_sections = []

    for section, is_valid in _CONFIG_SECTION_VALIDATORS.items():
        if not is_valid(repaired.get(section)):
            repaired[section] = deepcopy(default_config[section])
            repaired_sections.append(section)

    if repaired_sections:
        repaired[CONFIG_REPAIR_MESSAGES_KEY] = [
            f"Config issue fixed: '{section}' was invalid and reset to defaults." for section in repaired_sections
        ]

    return repaired


MAX_CONFIG_UPLOAD_BYTES = 1_000_000  # 1 MB - a hand-edited app_config.json is a few KB


def export_config_json(config: Dict[str, Any]) -> str:
    """Serialize config to pretty JSON for download, stripping the repair-messages channel."""
    exportable = deepcopy(config)
    exportable.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
    return json.dumps(exportable, indent=2)


def import_config_json(raw_bytes: bytes, default_config: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Parse and repair an uploaded config file. Never raises.

    Returns (config, messages). `messages` is empty on a clean import; on any
    problem (invalid JSON, oversized file, wrong shape) it holds human-readable
    reasons and `config` falls back to validate_config_shape's repair-and-warn
    path (or, for unparseable input, a full default copy).
    """
    if len(raw_bytes) > MAX_CONFIG_UPLOAD_BYTES:
        return deepcopy(default_config), [
            f"File too large ({len(raw_bytes)} bytes) - must be under {MAX_CONFIG_UPLOAD_BYTES} bytes."
        ]

    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        return deepcopy(default_config), [f"Not valid JSON: {exc}"]
    except RecursionError:
        # A deeply-nested payload (e.g. thousands of nested arrays) is only a
        # few KB and sails under MAX_CONFIG_UPLOAD_BYTES, but json.loads still
        # blows Python's recursion limit decoding it - this is a shape
        # problem, not a size problem, so it gets its own message.
        return deepcopy(default_config), ["JSON is too deeply nested to parse."]

    if not isinstance(parsed, dict):
        return deepcopy(default_config), ["Uploaded file must contain a JSON object."]

    normalized = normalize_config_constraints(parsed)
    repaired = validate_config_shape(normalized, default_config)
    messages = repaired.pop(CONFIG_REPAIR_MESSAGES_KEY, [])
    return repaired, messages


class ConfigurationManager:
    """Manage application configuration and constraints."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        default_names_path: str | Path | None = None,
    ):
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else _resolve_storage_path(
                "PICKLEBALL_APP_CONFIG",
                "app_config.json",
            )
        )
        self.default_names_path = (
            Path(default_names_path) if default_names_path is not None else _resolve_default_names_path()
        )
        self.config_path.parent.mkdir(exist_ok=True)
        self.default_config = {
            CONFIG_SCHEMA_VERSION_KEY: CURRENT_CONFIG_SCHEMA_VERSION,
            "objectives": {
                "equal_games": {"weight": 5000, "min": 1000, "max": 10000},
                "equal_variety": {"weight": 4000, "min": 1000, "max": 10000},
                "constraints": {"weight": 6000, "min": 2000, "max": 10000},
            },
            "constraints": {"do_not_pair": [], "do_not_oppose": []},
            "player_presets": {
                "Regular Group": [
                    "Teresa",
                    "Katie",
                    "Chris",
                    "Lisa",
                    "Mary",
                    "Michelle",
                    "Beth",
                    "Betsi",
                    "Kat",
                    "Nancy",
                    "Laura",
                    "Denise",
                    "LizBeth",
                    "MichelleM",
                    "BethL",
                ],
                "12 Player League": [
                    "Alice",
                    "Bob",
                    "Charlie",
                    "Diana",
                    "Eve",
                    "Frank",
                    "Grace",
                    "Henry",
                    "Ivy",
                    "Jack",
                    "Kate",
                    "Liam",
                ],
                "16 Player Tournament": [
                    "P1",
                    "P2",
                    "P3",
                    "P4",
                    "P5",
                    "P6",
                    "P7",
                    "P8",
                    "P9",
                    "P10",
                    "P11",
                    "P12",
                    "P13",
                    "P14",
                    "P15",
                    "P16",
                ],
            },
            "scheduling": {
                "default_rounds": 8,
                "start_time": "14:00",
                "end_time": "16:00",
                "game_duration_minutes": 15,
            },
        }

    def load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    raw_config = json.load(f)
                # Migrate on the raw on-disk shape - self._merge_configs would otherwise
                # backfill schema_version from self.default_config, masking an older version.
                migrated, applied_from = migrate_config(deepcopy(raw_config))
                if applied_from:
                    logger.info(
                        "Config migrated: schema_version %s -> %s", applied_from[0], migrated[CONFIG_SCHEMA_VERSION_KEY]
                    )
                elif migrated[CONFIG_SCHEMA_VERSION_KEY] > CURRENT_CONFIG_SCHEMA_VERSION:
                    logger.warning(
                        "Config schema_version %s is newer than this app supports (%s) - using as-is",
                        migrated[CONFIG_SCHEMA_VERSION_KEY],
                        CURRENT_CONFIG_SCHEMA_VERSION,
                    )
                loaded = normalize_config_constraints(self._merge_configs(self.default_config, migrated))
            else:
                loaded = normalize_config_constraints(deepcopy(self.default_config))
        except Exception:
            logger.exception("Failed to load app_config.json - falling back to defaults")
            loaded = normalize_config_constraints(deepcopy(self.default_config))

        repaired = validate_config_shape(loaded, self.default_config)
        repair_messages = repaired.get(CONFIG_REPAIR_MESSAGES_KEY)
        if repair_messages:
            logger.warning("Config repaired on load: %s", "; ".join(repair_messages))
        return repaired

    def save_config(self, config: Dict) -> bool:
        """Save configuration to file. Returns True on success, False on failure.

        Callers are responsible for surfacing a False return to the user -
        this manager has no UI dependency of its own."""
        try:
            normalized_config = normalize_config_constraints(deepcopy(config))
            normalized_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
            normalized_config[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
        except Exception:
            logger.exception("Failed to normalize configuration")
            return False
        return save_json(self.config_path, normalized_config, "app configuration")

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        result = deepcopy(default)
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def load_default_names(self) -> List[str]:
        """Load default player names from file."""
        try:
            if self.default_names_path.exists():
                with open(self.default_names_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load default player names")
            return []
