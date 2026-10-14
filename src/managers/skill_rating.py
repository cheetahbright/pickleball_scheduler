#!/usr/bin/env python3
"""Manually-set per-player skill ratings, used for post-hoc team-balance analytics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

try:
    from src.managers._paths import _resolve_storage_path, back_up_unreadable, read_json, save_json
except ImportError:
    from managers._paths import _resolve_storage_path, back_up_unreadable, read_json, save_json

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
        self._lasting_lock = False  # see _read_json_with_retry
        self._logged_problem: str | None = None  # see _report

    def load_skills(self) -> Dict[str, int]:
        """Return {player_name: rating}. A missing or unreadable file yields an empty dict (see read_skills)."""
        return self.read_skills()[0] or {}

    def read_skills(self) -> tuple[Dict[str, int] | None, str | None]:
        """Return ``(ratings, problem)``, where ``problem`` is a message for the user or None.

        ``ratings`` is None when player_skills.json exists but can't be read
        (bad JSON, not a JSON object, still locked). A file the next save
        would lose data from - unreadable, or holding ratings that aren't
        whole numbers from 1 to 5 - is copied aside first, and ``problem``
        names the copy. A missing file just means nothing is rated yet:
        ``({}, None)``.
        """
        try:
            raw = self._read_json_with_retry()
            if not isinstance(raw, dict):
                raise ValueError("the file doesn't contain a JSON object")
        except FileNotFoundError:
            return self._report({}, None)
        except Exception as exc:
            kept = "" if isinstance(exc, PermissionError) else self._kept_note()  # a locked file can't be copied
            problem = (
                f"Couldn't read {self.skills_path.name} ({exc}), so saved ratings can't be shown "
                f"and the next save will replace it.{kept}"
            )
            return self._report(None, problem, exc)

        skills: Dict[str, int] = {}
        ignored = []
        for player, rating in raw.items():
            if isinstance(rating, float) and rating.is_integer():
                rating = int(rating)  # 4.0 from a hand edit means 4
            is_whole_number = isinstance(rating, int) and not isinstance(rating, bool)  # JSON true isn't a 1
            if is_whole_number and self.MIN_RATING <= rating <= self.MAX_RATING:
                skills[str(player)] = rating
            else:
                ignored.append(str(player))
        if not ignored:
            return self._report(skills, None)
        return self._report(
            skills,
            f"Ignored saved ratings in {self.skills_path.name} that aren't whole numbers from 1 to 5 "
            f"({', '.join(sorted(ignored))}); the next save will leave those players unrated "
            f"unless you first rate the ones that have a slider.{self._kept_note()}",
        )

    def _report(
        self, ratings: Dict[str, int] | None, problem: str | None, error: Exception | None = None
    ) -> tuple[Dict[str, int] | None, str | None]:
        """Return ``(ratings, problem)``, logging each problem once: the page re-reads the file on every rerun."""
        if problem and problem != self._logged_problem:
            logger.warning("%s (%s)", problem, self.skills_path, exc_info=error)
        self._logged_problem = problem
        return ratings, problem

    def _read_json_with_retry(self) -> object:
        """Parse the skills file, retrying while another save holds it mid-replace (Windows).

        A file still locked after every retry is held by something slower than
        a save (a sync client, an editor). The page reads the file on every
        rerun, so until a read gets through again, later reads try only once.
        """
        retry = not self._lasting_lock
        self._lasting_lock = False
        try:
            return read_json(self.skills_path, retry)
        except PermissionError:
            self._lasting_lock = True
            raise

    def _kept_note(self) -> str:
        backup = back_up_unreadable(self.skills_path)
        return f" Your file was kept as {backup.name}." if backup else ""

    def save_skills(self, skills: Dict[str, int]) -> bool:
        """Persist the full skills dict, overwriting whatever was there before."""
        return save_json(self.skills_path, skills, "player skills")
