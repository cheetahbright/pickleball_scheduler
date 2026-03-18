#!/usr/bin/env python3
"""Smart Settings Utilities
Only update timestamps when settings actually change
"""


def save_settings_smart(settings_data, settings_file_path):
    """Smart settings saver that only updates timestamp when settings actually change"""
    import hashlib
    import json
    from datetime import datetime, timedelta
    from pathlib import Path

    settings_file = Path(settings_file_path)
    hash_file = settings_file.parent / ".settings_hash"

    # Calculate hash of current settings (without timestamp)
    settings_copy = settings_data.copy()
    if "timestamp" in settings_copy:
        del settings_copy["timestamp"]

    current_hash = hashlib.md5(json.dumps(settings_copy, sort_keys=True).encode()).hexdigest()

    # Read last hash
    last_hash = None
    if hash_file.exists():
        try:
            with open(hash_file) as f:
                last_hash = f.read().strip()
        except OSError:
            pass

    # Only update timestamp if settings actually changed
    if current_hash != last_hash:
        # Read existing timestamp if present to avoid rare equality on fast successive writes
        existing_ts = None
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    existing_ts = existing_data.get("timestamp")
            except Exception:
                # If the file is unreadable or invalid JSON, proceed without comparing timestamps
                existing_ts = None

        new_ts = datetime.now().isoformat()
        # Extremely fast consecutive calls can, in rare cases, yield identical isoformat strings
        if existing_ts == new_ts:
            new_ts = (datetime.now() + timedelta(microseconds=1)).isoformat()

        settings_data["timestamp"] = new_ts

        # Create parent directories if they don't exist
        settings_file.parent.mkdir(parents=True, exist_ok=True)

        # Save settings
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2)

        # Update hash
        with open(hash_file, "w") as f:
            f.write(current_hash)

        return True  # Settings were updated

    return False  # No update needed
