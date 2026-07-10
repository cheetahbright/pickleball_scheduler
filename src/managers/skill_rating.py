#!/usr/bin/env python3
"""Manually-set per-player skill ratings, used for post-hoc team-balance analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    from src.managers._paths import _resolve_storage_path, load_json_value, save_json
except ImportError:
    from managers._paths import _resolve_storage_path, load_json_value, save_json


class SkillRatingManager:
    """Store per-player skill ratings (1-5) for team-balance analytics.

    Deliberately does not feed into the genetic scheduler's fitness function -
    that would require dedicated algorithm design and verification work
    (see #124/#126 discussion). This gives visibility into how skill-balanced
    a generated schedule happens to be, as a first step.
    """

    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, skills_path: str | Path | None = None):
        self.skills_path = (
            Path(skills_path)
            if skills_path is not None
            else _resolve_storage_path("PICKLEBALL_SKILLS_FILE", "player_skills.json")
        )

    def load_skills(self) -> Dict[str, int]:
        """Return {player_name: rating}. Missing/corrupted file yields an empty dict."""
        raw = load_json_value(self.skills_path, dict, {}, "player skills")
        return {
            str(player): rating
            for player, rating in raw.items()
            if isinstance(rating, int) and self.MIN_RATING <= rating <= self.MAX_RATING
        }

    def save_skills(self, skills: Dict[str, int]) -> bool:
        """Persist the full skills dict, overwriting whatever was there before."""
        return save_json(self.skills_path, skills, "player skills")

    def set_skill(self, player: str, rating: int) -> bool:
        """Set one player's rating and persist immediately."""
        if not (self.MIN_RATING <= rating <= self.MAX_RATING):
            raise ValueError(f"Rating must be between {self.MIN_RATING} and {self.MAX_RATING}")
        skills = self.load_skills()
        skills[player] = rating
        return self.save_skills(skills)
