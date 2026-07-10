"""Stateful managers, split by concern from the former monolithic app_managers.py.

src/app_managers.py remains the stable import surface (a compatibility
facade re-exporting everything below) for existing callers/tests.
"""
