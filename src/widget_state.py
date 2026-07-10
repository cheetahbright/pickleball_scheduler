#!/usr/bin/env python3
"""Helpers for driving keyed Streamlit widgets from application state.

Streamlit imposes two rules on widgets that have an explicit ``key``, and
both of them have bitten this app repeatedly:

1. Once ``key`` exists in ``session_state``, the widget's ``value=`` (or
   ``index=``) argument is ignored on every subsequent rerun -
   ``session_state[key]`` wins. So a widget whose displayed value should
   track something external (a preset, a config value, a loaded schedule)
   silently keeps showing stale data unless that key is re-seeded.

2. ``session_state[key]`` cannot be assigned once the widget owning that
   key has already been instantiated during the current script run -
   Streamlit raises ``StreamlitAPIException``. Button handlers run *after*
   the widgets above them, so they can never write those keys directly.

This module provides one primitive per rule, so the fix lives in one place
instead of being rediscovered (and mis-implemented) at each call site.
"""

from __future__ import annotations

from typing import Any

# Namespaces for our bookkeeping entries. They live in session_state
# alongside real widget keys, so they're prefixed to avoid collisions.
_DEFERRED_PREFIX = "_deferred_widget_value:"
_SOURCE_FINGERPRINT_PREFIX = "_widget_source_fingerprint:"


def defer_widget_value(session_state: Any, key: str, value: Any) -> None:
    """Queue ``value`` to become widget ``key``'s value on the next rerun.

    Use this from code that runs *after* the widget was instantiated - button
    handlers, mostly - where assigning ``session_state[key]`` directly would
    raise ``StreamlitAPIException`` (rule 2 above). The caller is expected to
    trigger a rerun; ``apply_deferred_widget_values`` then applies the queued
    value at the top of the next run, before the widget is created.
    """
    session_state[_DEFERRED_PREFIX + key] = value


def apply_deferred_widget_values(session_state: Any) -> None:
    """Apply any values queued by ``defer_widget_value`` and clear the queue.

    Must be called before the widgets in question are instantiated - i.e. at
    the top of the render function that owns them. Safe to call more than
    once per run: applied entries are removed as they're consumed.
    """
    deferred_keys = [key for key in list(session_state.keys()) if key.startswith(_DEFERRED_PREFIX)]
    for deferred_key in deferred_keys:
        widget_key = deferred_key[len(_DEFERRED_PREFIX) :]
        session_state[widget_key] = session_state.pop(deferred_key)


def seed_widget_from_source(session_state: Any, key: str, value: Any) -> bool:
    """Re-seed widget ``key`` when its upstream source value has changed.

    Solves rule 1 without clobbering the user's in-progress edits: the widget
    is only reset when ``value`` differs from what the source produced on the
    previous run. If the user typed into the widget but the source is
    unchanged, their text survives the rerun.

    Call this *before* instantiating the widget, and still pass ``value`` as
    the widget's ``value=`` so the very first render (when no key exists yet)
    is seeded correctly. Returns True if the widget was re-seeded.
    """
    fingerprint_key = _SOURCE_FINGERPRINT_PREFIX + key
    source_changed = session_state.get(fingerprint_key) != value
    if source_changed:
        session_state[key] = value
    session_state[fingerprint_key] = value
    return source_changed


def forget_widget_state(session_state: Any, key: str) -> None:
    """Drop widget ``key``'s value along with any bookkeeping kept for it.

    Use when the thing a widget represents no longer exists (a deleted preset,
    a schedule that's been replaced), so a later widget reusing that key
    doesn't inherit stale state or a stale fingerprint.
    """
    session_state.pop(key, None)
    session_state.pop(_SOURCE_FINGERPRINT_PREFIX + key, None)
    session_state.pop(_DEFERRED_PREFIX + key, None)
