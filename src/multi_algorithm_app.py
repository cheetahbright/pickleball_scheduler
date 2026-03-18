"""Compatibility wrapper for the old multi-algorithm app entry point."""

from src.main_app import (
    analytics_tab,
    configuration_tab,
    display_enhanced_schedule,
    enhanced_scheduler_page,
    history_tab,
    main,
    main_scheduler_tab,
    player_management_tab,
    schedule_to_csv,
    validate_schedule_integrity,
)

__all__ = [
    "analytics_tab",
    "configuration_tab",
    "display_enhanced_schedule",
    "enhanced_scheduler_page",
    "history_tab",
    "main",
    "main_scheduler_tab",
    "player_management_tab",
    "schedule_to_csv",
    "validate_schedule_integrity",
]
