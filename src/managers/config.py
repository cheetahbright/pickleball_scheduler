#!/usr/bin/env python3
"""Config-schema constraints normalization, validation, migration, and
ConfigurationManager (app_config.json persistence)."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, TypeGuard

try:
    from src.managers._paths import (
        _resolve_default_names_path,
        _resolve_storage_path,
        back_up_unreadable,
        load_json_value,
        read_json,
        save_json,
    )
except ImportError:
    from managers._paths import (
        _resolve_default_names_path,
        _resolve_storage_path,
        back_up_unreadable,
        load_json_value,
        read_json,
        save_json,
    )

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

    if not isinstance(first, str) or not isinstance(second, str):
        return None  # a null or a number isn't a player name - str() would invent "None" or "7"
    normalized_first = first.strip()
    normalized_second = second.strip()
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
        first, second = sorted((parsed_pair[0], parsed_pair[1]))
        pair_key = (first, second)
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        normalized_pairs.append(parsed_pair)

    return normalized_pairs


def serialize_constraint_pairs(raw_constraints: Any, separator: str = ", ") -> str:
    """Render stored constraint pairs for text-area editing."""
    return "\n".join(f"{first}{separator}{second}" for first, second in normalize_constraint_pairs(raw_constraints))


def parse_constraint_text(text: str, saved_pairs: Any, separator: str = ", ") -> list[list[str]]:
    """Parse edited constraint text, keeping lines that still match a saved pair exactly.

    The "A, B" text format is ambiguous for names containing ",", "&" or
    " vs ", so a line left exactly as serialize_constraint_pairs wrote it
    maps back to its saved pair instead of being re-split; only new or
    changed lines go through the heuristic parser.
    """
    saved_by_line = {
        f"{first}{separator}{second}": [first, second] for first, second in normalize_constraint_pairs(saved_pairs)
    }
    return normalize_constraint_pairs([saved_by_line.get(line.strip(), line) for line in text.splitlines()])


def _unkept_constraint_entries(raw_constraints: Any) -> tuple[int, int]:
    """Count the stored entries normalize_constraint_pairs can't keep as written: (dropped, shortened).

    An entry that isn't a pair of two different names is dropped; one naming
    more than two players keeps only the first two. Blank lines, duplicates,
    and a blank, null or repeated name after the pair lose nothing, so they aren't
    counted.
    """
    if isinstance(raw_constraints, str):
        items: list[Any] = raw_constraints.splitlines()
    elif isinstance(raw_constraints, (list, tuple)):
        items = list(raw_constraints)
    else:
        return 0, 0
    dropped = shortened = 0
    for item in items:
        if isinstance(item, str) and not item.strip():
            continue
        pair = _parse_constraint_pair(item)
        if pair is None:
            dropped += 1
        elif isinstance(item, (list, tuple)) and any(
            extra is not None and not (isinstance(extra, str) and extra.strip() in ("", *pair)) for extra in item[2:]
        ):
            shortened += 1
    return dropped, shortened


def normalize_config_constraints(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure constraint settings use the pair structure expected by the scheduler UI.

    A "constraints" section that isn't an object (null, a list, a string) is
    replaced with an empty one, and constraint entries that can't be kept as
    written are dropped or shortened; each repair is reported through
    CONFIG_REPAIR_MESSAGES_KEY.
    """
    constraints = config.get("constraints", {})
    if not isinstance(constraints, dict):
        _add_repair_message(config, "Config issue fixed: 'constraints' was invalid and was cleared.")
        constraints = {}
    config["constraints"] = constraints
    for key in ("do_not_pair", "do_not_oppose"):
        raw_constraints = constraints.get(key, [])
        if raw_constraints is not None and not isinstance(raw_constraints, (str, list, tuple)):
            _add_repair_message(config, f"Config issue fixed: '{key}' wasn't a list of pairs and was cleared.")
        dropped, shortened = _unkept_constraint_entries(raw_constraints)
        if dropped:
            _add_repair_message(config, f"Config issue fixed: '{key}' had {dropped} invalid pair(s) removed.")
        if shortened:
            _add_repair_message(
                config, f"Config issue fixed: '{key}' had {shortened} pair(s) with more than two names cut to two."
            )
        constraints[key] = normalize_constraint_pairs(raw_constraints)
    return config


# Largest integer a browser number can hold exactly; st.slider rejects larger.
_MAX_SAFE_INTEGER = 2**53 - 1


def _is_valid_objectives(objectives: Any) -> bool:
    """An objectives section is a non-empty dict of entries a weight slider can render."""
    if not isinstance(objectives, dict) or not objectives:
        return False
    return all(_slider_renders(entry) for entry in objectives.values())


def _slider_renders(entry: Any) -> bool:
    """Whether st.slider (streamlit 1.55) can render this {weight, min, max} entry.

    It widens its bounds to take in the value, so a weight outside [min, max]
    or min > max renders fine; it only raises when the adjusted bounds
    collapse to one value, or a bound isn't a browser-safe integer. This
    mirrors that adjustment (streamlit/elements/widgets/slider.py) rather
    than demanding a tidy range, so configs the app always showed fine
    aren't "repaired" away.
    """
    if not isinstance(entry, dict):
        return False
    weight, low, high = entry.get("weight"), entry.get("min"), entry.get("max")
    if not (_is_strict_int(weight) and _is_strict_int(low) and _is_strict_int(high)):
        return False
    low = min(low, high)
    high = max(low, high)
    low, high = min(weight, low), max(weight, high)
    return low != high and abs(low) <= _MAX_SAFE_INTEGER and abs(high) <= _MAX_SAFE_INTEGER


def _repair_objectives(objectives: Dict[str, Any], defaults: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Reset just the objective entries a slider can't render, keeping the rest."""
    repaired: Dict[str, Any] = {}
    messages: list[str] = []
    for name, entry in objectives.items():
        if _slider_renders(entry):
            repaired[name] = entry
        elif name in defaults:
            repaired[name] = deepcopy(defaults[name])
            messages.append(f"Config issue fixed: objective '{name}' had an invalid range and was reset.")
        else:
            messages.append(f"Config issue fixed: objective '{name}' had an invalid range and was removed.")
    return repaired, messages


def _is_strict_int(value: Any) -> TypeGuard[int]:
    """An int that isn't a bool (JSON true/false would otherwise pass as 1/0)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_scheduling(scheduling: Any) -> bool:
    """A scheduling section has an in-range round count and parseable HH:MM times."""
    if not isinstance(scheduling, dict):
        return False

    default_rounds = scheduling.get("default_rounds")
    if not isinstance(default_rounds, int) or isinstance(default_rounds, bool) or not (1 <= default_rounds <= 20):
        return False

    for key in ("start_time", "end_time"):
        value = scheduling.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}", value):
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


def _repair_player_presets(player_presets: Dict[str, Any]) -> tuple[Dict[str, list[str]], list[str]]:
    """Repair a player_presets dict entry by entry instead of resetting every preset.

    A preset whose roster isn't a list is dropped; non-text or blank names
    inside a roster are dropped. Each repair gets its own message.
    """
    repaired: Dict[str, list[str]] = {}
    messages: list[str] = []
    for name, players in player_presets.items():
        if not isinstance(players, list):
            messages.append(f"Config issue fixed: preset '{name}' wasn't a list of players and was removed.")
            continue
        names = [player.strip() for player in players if isinstance(player, str) and player.strip()]
        if len(names) != len(players):
            messages.append(
                f"Config issue fixed: preset '{name}' had {len(players) - len(names)} invalid name(s) removed."
            )
        repaired[str(name)] = names
    return repaired, messages


_CONFIG_SECTION_VALIDATORS = {
    "objectives": _is_valid_objectives,
    "scheduling": _is_valid_scheduling,
    "player_presets": _is_valid_player_presets,
}

CONFIG_REPAIR_MESSAGES_KEY = "_config_repair_messages"


def _add_repair_message(config: Dict[str, Any], message: str) -> None:
    config.setdefault(CONFIG_REPAIR_MESSAGES_KEY, []).append(message)


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
    recorded = repaired.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
    messages = [message for message in recorded if isinstance(message, str)] if isinstance(recorded, list) else []

    presets = repaired.get("player_presets")
    if isinstance(presets, dict) and not _is_valid_player_presets(presets):
        repaired["player_presets"], preset_messages = _repair_player_presets(presets)
        messages.extend(preset_messages)

    objectives = repaired.get("objectives")
    if isinstance(objectives, dict) and objectives and not _is_valid_objectives(objectives):
        repaired["objectives"], objective_messages = _repair_objectives(objectives, default_config["objectives"])
        messages.extend(objective_messages)

    for section, is_valid in _CONFIG_SECTION_VALIDATORS.items():
        if not is_valid(repaired.get(section)):
            repaired[section] = deepcopy(default_config[section])
            messages.append(f"Config issue fixed: '{section}' was invalid and reset to defaults.")

    if messages:
        repaired[CONFIG_REPAIR_MESSAGES_KEY] = messages

    return repaired


MAX_CONFIG_UPLOAD_BYTES = 1_000_000  # 1 MB - a hand-edited app_config.json is a few KB
MAX_CONFIG_NESTING_DEPTH = 20  # a real app_config.json nests about 4 levels deep


def _nesting_exceeds(value: Any, max_depth: int) -> bool:
    """Return True if dicts/lists in value nest more than max_depth levels deep.

    Walks with an explicit stack rather than recursion, so it is safe on the
    very payloads it exists to reject.
    """
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if isinstance(current, dict):
            children = list(current.values())
        elif isinstance(current, list):
            children = current
        else:
            continue
        if depth > max_depth:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False


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
        parsed = json.loads(raw_bytes)  # detects a BOM or UTF-16 itself, as read_json does
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
    # json.loads accepts nesting hundreds of levels deep (thousands on Python
    # 3.12+), but the repair steps below deepcopy recursively and run out of
    # stack after a few hundred.
    if _nesting_exceeds(parsed, MAX_CONFIG_NESTING_DEPTH):
        return deepcopy(default_config), ["JSON is too deeply nested to parse."]
    parsed.pop(CONFIG_REPAIR_MESSAGES_KEY, None)  # internal channel, never trusted from an upload

    normalized = normalize_config_constraints(parsed)
    repaired = validate_config_shape(normalized, default_config)
    messages = repaired.pop(CONFIG_REPAIR_MESSAGES_KEY, [])
    return repaired, messages


# Sections whose entries belong to the user: the defaults seed them only when
# the section is missing entirely, and are never merged back in on load - a
# deleted built-in preset must stay deleted.
USER_OWNED_SECTIONS = frozenset({"player_presets"})

# Sections whose entries are independent of each other (a preset, an
# objective), so concurrent edits merge entry by entry. Other sections - e.g.
# scheduling, whose start and end times only make sense together - are
# replaced as a whole by whichever session changed them.
ENTRY_MERGED_SECTIONS = frozenset({"player_presets", "objectives"})

# The constraints section's lists hold independent, unordered pairs, so
# concurrent edits merge pair by pair (see _merge_constraints).
PAIR_MERGED_LISTS = ("do_not_pair", "do_not_oppose")

_MISSING = object()


def _merge_section_entries(base: Dict, mine: Dict, theirs: Dict) -> Dict:
    """Three-way merge one config section entry by entry (a preset, an objective, a setting).

    Starts from `theirs` (what is on disk now) and applies only the entries
    `mine` changed, added, or removed relative to `base` (what this session
    last loaded or saved), so two sessions editing different entries of the
    same section both keep their edits.
    """
    result = deepcopy(theirs)
    for key in list(base) + [key for key in mine if key not in base]:
        mine_value = mine.get(key, _MISSING)
        if mine_value == base.get(key, _MISSING):
            continue
        if mine_value is _MISSING:
            result.pop(key, None)
        else:
            result[key] = deepcopy(mine_value)
    return result


def _merge_pair_lists(base: list, mine: list, theirs: list) -> list:
    """Three-way merge one constraint list pair by pair (a pair's name order doesn't matter).

    Keeps `theirs` minus the pairs `mine` removed relative to `base`, then
    appends the pairs `mine` added, so two sessions adding or removing
    different pairs both keep their edits.
    """
    base_pairs = {frozenset(pair) for pair in base}
    removed = base_pairs - {frozenset(pair) for pair in mine}
    result = [deepcopy(pair) for pair in theirs if frozenset(pair) not in removed]
    present = {frozenset(pair) for pair in result}
    for pair in mine:
        key = frozenset(pair)
        if key not in base_pairs and key not in present:
            result.append(deepcopy(pair))
            present.add(key)
    return result


def _merge_constraints(base: Dict, mine: Dict, theirs: Dict) -> Dict:
    """Three-way merge the constraints section, each of PAIR_MERGED_LISTS pair by pair."""
    result = _merge_section_entries(base, mine, theirs)
    for list_key in PAIR_MERGED_LISTS:
        lists = (base.get(list_key, []), mine.get(list_key, []), theirs.get(list_key, []))
        if all(isinstance(pairs, list) for pairs in lists):
            result[list_key] = _merge_pair_lists(*lists)
    return result


def _normalized_for_save(config: Dict) -> Dict:
    """A copy of `config` in the shape save_config writes (may raise on a malformed config)."""
    normalized_config = normalize_config_constraints(deepcopy(config))
    normalized_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
    normalized_config[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
    return normalized_config


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
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot of what this instance last loaded/saved, for detecting a
        # concurrent save from another session at save time (see save_config).
        self._loaded_snapshot: Dict | None = None
        # Messages about a save for the UI to show once (see pop_notices), and
        # the ones the save in progress will add if its write succeeds.
        self._notices: list[str] = []
        self._pending_notices: list[str] = []
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
            },
        }

    def pop_notices(self) -> list[str]:
        """Return the messages about this session's saves that the user hasn't seen yet, and forget them."""
        notices, self._notices = self._notices, []
        return notices

    def load_config(self) -> Dict:
        """Load configuration from file.

        An unreadable file (bad JSON, not a JSON object) falls back to the
        defaults, but never silently: the file is copied aside first and a
        repair message names the copy, since the next save will replace
        app_config.json.
        """
        unreadable_message = None
        try:
            if self.config_path.exists():
                raw_config = read_json(self.config_path)
                if not isinstance(raw_config, dict):
                    raise ValueError("the file doesn't contain a JSON object")
                raw_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)  # internal channel, never trusted from disk
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
        except Exception as exc:
            logger.exception("Failed to load app_config.json - falling back to defaults")
            loaded = normalize_config_constraints(deepcopy(self.default_config))
            backup = back_up_unreadable(self.config_path)
            kept = f" Your file was kept as {backup.name}." if backup else ""
            unreadable_message = (
                f"Couldn't read {self.config_path.name} ({exc}), so the default settings are in use "
                f"and the next save will replace it.{kept}"
            )

        repaired = validate_config_shape(loaded, self.default_config)
        if unreadable_message:
            repaired.setdefault(CONFIG_REPAIR_MESSAGES_KEY, []).insert(0, unreadable_message)
        repair_messages = repaired.get(CONFIG_REPAIR_MESSAGES_KEY)
        if repair_messages:
            logger.warning("Config repaired on load: %s", "; ".join(repair_messages))

        snapshot = deepcopy(repaired)
        snapshot.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
        self._loaded_snapshot = snapshot

        return repaired

    def save_config(self, config: Dict, replace: bool = False) -> bool:
        """Save configuration to file. Returns True on success, False on failure.

        By default the save is merged with what other sessions saved since this
        one loaded (see _merge_concurrent_edits). ``replace=True`` writes the
        config exactly as given - for restoring a backup, which should replace
        the settings rather than blend into them.

        Callers are responsible for surfacing a False return to the user -
        this manager has no UI dependency of its own. A save that replaced a
        file it couldn't read, or rewrote constraint entries it couldn't keep,
        leaves a message for pop_notices."""
        try:
            normalized_config = _normalized_for_save(config)
        except Exception:
            logger.exception("Failed to normalize configuration")
            return False
        self._pending_notices = []
        to_write = normalized_config if replace else self._merge_concurrent_edits(normalized_config)
        if not save_json(self.config_path, to_write, "app configuration"):
            return False  # the baseline is untouched, so a retry merges exactly as this attempt did
        self._notices.extend(self._pending_notices)
        # The next save's baseline is what this session just submitted. (It
        # used to end up as whatever was on disk before this write, so
        # reverting an edit looked like "no change" and was skipped, and a
        # section edited once kept looking "changed" and overwrote other
        # sessions on every later save.)
        self._loaded_snapshot = deepcopy(normalized_config)
        return True

    def read_saved_config(self) -> Dict | None:
        """The config saved on disk right now, other sessions' saves included - or None if there's no readable file.

        A session loads its config once, so its own copy lacks what other
        sessions saved since. Unlike load_config this never moves this
        session's merge baseline, so it's safe for read-only checks such as a
        duplicate-name check.
        """
        disk = self._read_disk_for_merge(saving=False)
        return None if disk is None else disk[0]

    def merge_with_saved(self, config: Dict) -> Dict:
        """What save_config(config) would write right now, without writing it or moving this session's baseline.

        That is this session's edits merged into what other sessions saved
        since it loaded (or `config` itself when there's no readable file) -
        for a backup download, which should hold both.
        """
        return self._merge_concurrent_edits(_normalized_for_save(config), saving=False)

    def _read_disk_for_merge(self, saving: bool = True) -> tuple[Dict, set[str], list[str]] | None:
        """What is on disk now, for merging - or None if there's nothing trustworthy to merge with.

        Unlike load_config this never falls back to the defaults (merging
        against the defaults would drop every entry this session didn't touch)
        and never moves this session's baseline. Also returns the sections that
        were missing or invalid on disk - load would have filled them in from
        the defaults, so the merge must not treat them as another session's
        edit either - and the repair messages for the constraint entries it
        couldn't keep as written. ``saving=False`` is for read-only callers: an
        unreadable file is only logged, since no save is about to replace it.
        """
        try:
            if not self.config_path.exists():
                return None
            raw_config = read_json(self.config_path)
            if not isinstance(raw_config, dict):
                raise ValueError("the file doesn't contain a JSON object")
            raw_config.pop(CONFIG_REPAIR_MESSAGES_KEY, None)  # internal channel, never trusted from disk
            migrated, _ = migrate_config(deepcopy(raw_config))
            # Sections missing from the file (filled in from the defaults) or
            # not even an object are no other session's edit.
            unreliable = {key for key in self.default_config if key not in migrated}
            if not isinstance(migrated.get("constraints", {}), dict):
                unreliable.add("constraints")
            with_defaults = normalize_config_constraints(self._merge_configs(self.default_config, migrated))
            constraint_repairs = with_defaults.pop(CONFIG_REPAIR_MESSAGES_KEY, [])
        except Exception as exc:
            if not saving:
                logger.warning("Could not read the saved config - going by this session's copy", exc_info=True)
                return None
            logger.exception("Could not read the current config to merge with - saving this session's view")
            backup = back_up_unreadable(self.config_path)  # this save is about to replace it
            kept = f" Your file was kept as {backup.name}." if backup else ""
            self._pending_notices.append(
                f"Couldn't read {self.config_path.name} ({exc}) when saving, so it was replaced "
                f"with this session's settings.{kept}"
            )
            return None
        unreliable |= {
            section
            for section, is_valid in _CONFIG_SECTION_VALIDATORS.items()
            if not is_valid(with_defaults.get(section))
        }
        unreliable.discard(CONFIG_SCHEMA_VERSION_KEY)
        current = validate_config_shape(with_defaults, self.default_config)
        current.pop(CONFIG_REPAIR_MESSAGES_KEY, None)
        return current, unreliable, constraint_repairs

    def _merge_concurrent_edits(self, config: Dict, saving: bool = True) -> Dict:
        """Merge this save against whatever is currently on disk, so a
        concurrent session's edit to a section THIS session never touched
        (e.g. Player Management saving player_presets while this session
        only edited Configuration's objectives) isn't silently discarded by
        a blind whole-file overwrite - each browser session holds its own
        in-memory config copy and its own ConfigurationManager instance, so
        two sessions can easily save moments apart without either knowing
        about the other's change.

        Only what THIS session actually changed since it last loaded/saved
        (via this instance) wins; everything else defers to whatever is
        currently on disk. ENTRY_MERGED_SECTIONS merge entry by entry (see
        _merge_section_entries) and constraints pair by pair (see
        _merge_constraints), so two sessions editing different presets,
        objectives or constraint pairs both keep their edits; if both changed
        the same entry - or the same other section - this session's edit wins.
        """
        if self._loaded_snapshot is None:
            return config  # never loaded via this instance - nothing to diff against

        baseline = self._loaded_snapshot
        disk = self._read_disk_for_merge(saving)
        if disk is None:
            return config
        current_on_disk, unreliable_sections, constraint_repairs = disk

        merged = deepcopy(current_on_disk)
        # Whether the file's (repaired) constraints carry into what is written,
        # as they are or merged with this session's pair edits.
        constraints_from_disk = True
        for key in list(baseline) + [key for key in config if key not in baseline]:
            mine = config.get(key, _MISSING)
            base = baseline.get(key, _MISSING)
            if key in unreliable_sections and mine is not _MISSING:
                merged[key] = deepcopy(mine)
                constraints_from_disk &= key != "constraints"
                continue
            if mine == base:
                continue
            theirs = merged.get(key, _MISSING)
            all_dicts = all(isinstance(value, dict) for value in (mine, base, theirs))
            if key in ENTRY_MERGED_SECTIONS and all_dicts:
                merged[key] = _merge_section_entries(base, mine, theirs)
            elif key == "constraints" and all_dicts:
                merged[key] = _merge_constraints(base, mine, theirs)
            elif mine is _MISSING:
                merged.pop(key, None)
                constraints_from_disk &= key != "constraints"
            else:
                merged[key] = deepcopy(mine)
                constraints_from_disk &= key != "constraints"
        if saving and constraints_from_disk:
            # The file's constraints are written back as repaired (e.g. a hand
            # edit made while this session was open), so what the repair
            # dropped is this save's to report.
            self._pending_notices.extend(
                f"While saving {self.config_path.name}: {message}" for message in constraint_repairs
            )
        return merged

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults (USER_OWNED_SECTIONS are taken as-is)."""
        result = deepcopy(default)
        for key, value in user.items():
            if key in USER_OWNED_SECTIONS:
                result[key] = deepcopy(value)
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def load_default_names(self) -> List[str]:
        """Load default player names from file.

        Non-string or blank entries in a malformed file are dropped rather
        than passed through raw - a caller (e.g. selecting the "Default
        Names" preset) that assumes every element is a usable name string
        would otherwise crash on the first non-string element.
        """
        data = load_json_value(self.default_names_path, list, [], "default player names")
        names = [name for name in data if isinstance(name, str) and name.strip()]
        if len(names) != len(data):
            logger.warning(
                "Ignored %d entries in %s that aren't player names", len(data) - len(names), self.default_names_path
            )
        return names
