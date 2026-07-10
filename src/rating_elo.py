#!/usr/bin/env python3
"""Pure ELO rating math for cross-week skill tracking (#145).

Like SkillRatingManager's static 1-5 ratings (#124), this is deliberately
NOT wired into the genetic scheduler's fitness function - see that class's
docstring in app_managers.py for the same rationale (#124/#126 discussion:
feeding a rating into scheduling would need dedicated algorithm design and
verification work). This module only computes the math; EloRatingManager
(app_managers.py) persists it and replays recorded scores to keep it
reproducible.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

DEFAULT_RATING = 1000.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the standard ELO win probability for A against B.

    expected_score(a, b) + expected_score(b, a) == 1 for any a, b - a always
    "expects" the complement of what B expects.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_pair(rating_a: float, rating_b: float, score_a: float, k: float = 32) -> Tuple[float, float]:
    """Return (new_rating_a, new_rating_b) after one head-to-head result.

    score_a is A's actual outcome: 1.0 (win), 0.0 (loss), or 0.5 (tie). B's
    outcome is always 1 - score_a, so the two ratings always move by equal
    and opposite amounts (zero-sum). Ties nudge each rating toward the
    other's value rather than counting as a win for either side.
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)
    return new_rating_a, new_rating_b


def apply_game(
    ratings: Dict[str, float],
    team1: List[str],
    team2: List[str],
    team1_score: float,
    team2_score: float,
    k: float = 32,
) -> Dict[str, float]:
    """Replay one doubles game and return a NEW ratings dict (input untouched).

    Each team's average rating stands in for a single ELO "player" when
    computing the expected outcome; the resulting team-level delta is then
    applied equally to both players on that side, which is the standard
    team-ELO approximation the feature was scoped to ("distribute the delta
    to both players on a side" - see #145). Missing players default to
    DEFAULT_RATING. A tied score (team1_score == team2_score) is scored as
    0.5/0.5 for both teams, consistent with standard ELO tie handling - it
    is not double-counted as a win for either side.
    """
    team1_ratings = [ratings.get(player, DEFAULT_RATING) for player in team1]
    team2_ratings = [ratings.get(player, DEFAULT_RATING) for player in team2]
    team1_avg = sum(team1_ratings) / len(team1_ratings)
    team2_avg = sum(team2_ratings) / len(team2_ratings)

    if team1_score > team2_score:
        team1_outcome = 1.0
    elif team1_score < team2_score:
        team1_outcome = 0.0
    else:
        team1_outcome = 0.5

    new_team1_avg, new_team2_avg = update_pair(team1_avg, team2_avg, team1_outcome, k)
    team1_delta = new_team1_avg - team1_avg
    team2_delta = new_team2_avg - team2_avg

    updated = dict(ratings)
    for player in team1:
        updated[player] = ratings.get(player, DEFAULT_RATING) + team1_delta
    for player in team2:
        updated[player] = ratings.get(player, DEFAULT_RATING) + team2_delta
    return updated
