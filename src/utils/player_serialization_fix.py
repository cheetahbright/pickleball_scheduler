#!/usr/bin/env python3
"""
PLAYER OBJECT SERIALIZATION FIX
==============================

Fix Streamlit Player object serialization issues by ensuring all Player objects
are converted to strings before being passed to pandas DataFrames or Streamlit components.
"""


def safe_player_to_string(player):
    """Safely convert Player object to string representation."""
    if hasattr(player, "name"):
        return str(player.name)
    elif hasattr(player, "__str__"):
        return str(player)
    else:
        return repr(player)


def safe_player_list_to_strings(players):
    """Convert list of Player objects to list of strings."""
    if not players:
        return []
    return [safe_player_to_string(p) for p in players]


def sanitize_dataframe_data(data):
    """Sanitize data for DataFrame creation by converting Player objects to strings."""
    if isinstance(data, dict):
        # Handle dictionary data
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, list):
                sanitized[key] = [safe_player_to_string(item) if hasattr(item, "name") else item for item in value]
            elif hasattr(value, "name"):  # Player object
                sanitized[key] = safe_player_to_string(value)
            else:
                sanitized[key] = value
        return sanitized
    elif isinstance(data, list):
        # Handle list of dictionaries
        return [
            (
                sanitize_dataframe_data(item)
                if isinstance(item, dict)
                else safe_player_to_string(item) if hasattr(item, "name") else item
            )
            for item in data
        ]
    else:
        return data


def serialize_player_data(players):
    """Serialize player data for safe storage and transmission.

    Args:
        players: List of player names or Player objects

    Returns:
        List of serialized player data (strings)
    """
    if not players:
        return []

    # Convert all player objects to strings
    serialized = safe_player_list_to_strings(players)

    # Ensure all items are strings and not empty
    result = []
    for player in serialized:
        if player and str(player).strip():
            result.append(str(player).strip())

    return result


if __name__ == "__main__":
    print("🔧 Player Object Serialization Fix Module")
    print("This module provides utilities to fix Streamlit Player object serialization issues.")
