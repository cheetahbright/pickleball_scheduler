#!/usr/bin/env python3
"""Persistent per-player ELO ratings (#145) that evolve from recorded scores."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from src.managers._paths import _resolve_storage_path, load_json_value, save_json
except ImportError:
    from managers._paths import _resolve_storage_path, load_json_value, save_json

_rating_elo = import_module_with_fallback("rating_elo")


class EloRatingManager:
    """Store per-player ELO ratings (#145) that evolve from recorded scores.

    Unlike SkillRatingManager's manually-set 1-5 ratings, these are never
    edited directly - they only change via recompute_from_history(), which
    replays every recorded game in chronological order through rating_elo's
    pure math. Like SkillRatingManager, this is deliberately kept OUT of the
    genetic scheduler's fitness function (see #124/#126 discussion) - it is
    an analytics/ranking view only, not a scheduling input.
    """

    DEFAULT_RATING = 1000

    def __init__(self, ratings_path: str | Path | None = None):
        self.ratings_path = (
            Path(ratings_path)
            if ratings_path is not None
            else _resolve_storage_path("PICKLEBALL_ELO_FILE", "player_elo.json")
        )

    def load_ratings(self) -> Dict[str, float]:
        """Return {player_name: rating}. Missing/corrupted file yields an empty dict."""
        raw = load_json_value(self.ratings_path, dict, {}, "ELO ratings")
        return {
            str(player): float(rating)
            for player, rating in raw.items()
            if isinstance(rating, (int, float)) and not isinstance(rating, bool)
        }

    def save_ratings(self, ratings: Dict[str, float]) -> bool:
        """Persist the full ratings dict, overwriting whatever was there before."""
        return save_json(self.ratings_path, ratings, "ELO ratings")

    def recompute_from_history(self, history_manager) -> Dict[str, float]:
        """Replay every recorded score (oldest first) and persist the result.

        Starts from an empty ratings dict every time - not whatever is
        currently saved - so the output only ever depends on the recorded
        score history itself. That's what makes recompute reproducible:
        running it twice against the same history yields identical ratings.

        Raises RuntimeError if the recomputed ratings could not be
        persisted, rather than silently discarding a failed write - a
        caller must not treat this as having succeeded.
        """
        games = history_manager.get_all_scored_games()
        ratings: Dict[str, float] = {}
        for game in games:
            ratings = _rating_elo.apply_game(
                ratings,
                game["team1_players"],
                game["team2_players"],
                game["team1_score"],
                game["team2_score"],
            )
        if not self.save_ratings(ratings):
            raise RuntimeError("Failed to persist recomputed ELO ratings")
        return ratings
