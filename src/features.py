#!/usr/bin/env python3
"""Feature flags: switches for features that are built but hidden for now.

A flag is a module-level boolean that is off by default. To turn one on, set
the environment variable PICKLEBALL_<FLAG> to 1, true or yes (any letter case)
before the server starts - PICKLEBALL_SCORE_TRACKING for SCORE_TRACKING. The
environment is read once, when this module is first imported.
docs/ARCHITECTURE.md ("Feature flags") says what each flag hides and how to
bring it back.

Read a flag when it is used, through the module, so a test can patch it:

    features = import_module_with_fallback("features")
    if features.SCORE_TRACKING:
        ...

Do not copy it with ``from features import SCORE_TRACKING``: that freezes the
value at import time.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes"})


def _env_flag(name: str) -> bool:
    """Whether PICKLEBALL_<name> is set to 1, true or yes (any letter case, outer spaces ignored)."""
    return os.environ.get(f"PICKLEBALL_{name}", "").strip().lower() in _TRUE_VALUES


# Draws the Objective Weights sliders in the Configuration tab. The weights are still stored, validated, merged and
# backed up either way, but no scheduler reads them yet.
OBJECTIVE_WEIGHTS_UI: bool = _env_flag("OBJECTIVE_WEIGHTS_UI")

# Win/loss score tracking: the Leaderboard tab with its score entry, CSV score import and standings, the ELO
# ratings, and the recorded scores in the Excel download. The code stays; with this off the app doesn't show or
# use it.
SCORE_TRACKING: bool = _env_flag("SCORE_TRACKING")
