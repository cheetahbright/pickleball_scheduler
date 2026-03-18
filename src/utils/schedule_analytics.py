"""Compatibility helpers for older analytics imports."""

from src.main_app import ScheduleAnalytics
from src.utils.schedule_analytics_core import calculate_fairness_metrics, serialize_schedule_for_json

__all__ = [
    "ScheduleAnalytics",
    "calculate_fairness_metrics",
    "serialize_schedule_for_json",
]
