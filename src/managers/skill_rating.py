#!/usr/bin/env python3
"""Manually-set per-player skill ratings, used for post-hoc team-balance analytics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

try:
    from src.managers._paths import _resolve_storage_path
except ImportError:
    from managers._paths import _resolve_storage_path

logger = logging.getLogger(__name__)


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
        try:
            if self.skills_path.exists():
                with open(self.skills_path, "r", encoding="utf-8") as f:
                    skills = json.load(f)
                if isinstance(skills, dict):
                    return {
                        str(player): rating
                        for player, rating in skills.items()
                        if isinstance(rating, int) and self.MIN_RATING <= rating <= self.MAX_RATING
                    }
            return {}
        except Exception:
            logger.exception("Failed to load player skills")
            return {}

    def save_skills(self, skills: Dict[str, int]) -> bool:
        """Persist the full skills dict, overwriting whatever was there before."""
        try:
            self.skills_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.skills_path, "w", encoding="utf-8") as f:
                json.dump(skills, f, indent=2)
            return True
        except Exception:
            logger.exception("Failed to save player skills")
            return False

    def set_skill(self, player: str, rating: int) -> bool:
        """Set one player's rating and persist immediately."""
        if not (self.MIN_RATING <= rating <= self.MAX_RATING):
            raise ValueError(f"Rating must be between {self.MIN_RATING} and {self.MAX_RATING}")
        skills = self.load_skills()
        skills[player] = rating
        return self.save_skills(skills)
