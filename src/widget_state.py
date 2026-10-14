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

A widget seeded through ``session_state`` is created without ``value=``:
Streamlit warns ("created with a default value but also had its value set
via the Session State API") whenever a widget gets both. The seed_* helpers
below always leave the key set before the widget renders, so it needs none.
They set it by assignment, which the browser then shows, so where an
st.rerun() dropped a widget's state they put back the value the browser
still has rather than a default (see remember_seeded_widget_values).
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

# Namespaces for our bookkeeping entries. They live in session_state
# alongside real widget keys, so they're prefixed to avoid collisions.
_DEFERRED_PREFIX = "_deferred_widget_value:"
_SOURCE_FINGERPRINT_PREFIX = "_widget_source_fingerprint:"
_LAST_VALUE_PREFIX = "_widget_last_value:"
_CHOICE_PREFIX = "_widget_choice:"
_SEEDED_VALUE_PREFIX = "_widget_seeded_value:"
_SUGGESTED_VALUE_PREFIX = "_widget_suggested_value:"


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


def seed_widget_from_source(
    session_state: Any,
    key: str,
    value: Any,
    rebase: Callable[[Any, Any, Any], Any] | None = None,
) -> bool:
    """Re-seed widget ``key`` when its upstream source value has changed.

    Solves rule 1 without clobbering the user's in-progress edits: the widget
    is only reset when ``value`` differs from what the source produced on the
    previous run. If the user typed into the widget but the source is
    unchanged, their text survives the rerun.

    If the source changes while the user has an edit pending, the edit is
    normally discarded. Pass ``rebase(previous_source, edited, new_source)``
    to carry it over instead (see rebase_line_edits).

    The widget's value is also kept under a private key, because an
    st.rerun() before the widget renders (a handler earlier on the page)
    drops its state, while the browser keeps showing - and sends back with
    the next click - what the user had there. That kept value is put back
    then, not the source's, and a rebased edit is carried over from it if
    the source changed in that same run. remember_seeded_widget_values keeps
    it current as each run starts.

    Call this *before* instantiating the widget, and create the widget
    without ``value=``: the key is always set here, including on the first
    render and after an st.rerun() dropped it. Returns True if the widget was
    re-seeded.
    """
    fingerprint_key = _SOURCE_FINGERPRINT_PREFIX + key
    last_value_key = _LAST_VALUE_PREFIX + key
    previous = session_state.get(fingerprint_key)
    source_changed = previous != value
    if source_changed:
        current = session_state.get(key, session_state.get(last_value_key))
        has_pending_edit = previous is not None and current is not None and current != previous
        session_state[key] = rebase(previous, current, value) if rebase and has_pending_edit else value
    elif key not in session_state:
        session_state[key] = session_state.get(last_value_key, value)
    session_state[fingerprint_key] = value
    session_state[last_value_key] = session_state[key]
    return source_changed


def remember_seeded_widget_values(session_state: Any) -> None:
    """Keep the value each seed_widget_from_source widget has as this run starts.

    Call it early in each run, before any handler that can st.rerun() (the
    app does at the top of the Main tab, the first one drawn): the rerun
    drops the state of every widget not drawn yet in that run, including a
    value the user changed along with the click, which the browser keeps
    showing. seed_widget_from_source then puts this value back.
    """
    for state_key in list(session_state.keys()):
        if isinstance(state_key, str) and state_key.startswith(_SOURCE_FINGERPRINT_PREFIX):
            key = state_key[len(_SOURCE_FINGERPRINT_PREFIX) :]
            if key in session_state:
                session_state[_LAST_VALUE_PREFIX + key] = session_state[key]


def rebase_line_edits(
    previous_source: str,
    edited: str,
    new_source: str,
    line_key: Callable[[str], Hashable] = str.strip,
) -> str:
    """Re-apply a user's line edits (relative to ``previous_source``) on top of ``new_source``.

    For one-item-per-line text boxes (constraint pairs, preset rosters):
    lines the user removed are dropped from the new text, and lines they
    added are appended once. Non-blank lines compare by ``line_key``, their
    stripped text by default.
    """

    def lines(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip()]

    previous_keys = {line_key(line) for line in lines(previous_source)}
    edited_lines = lines(edited)
    removed = previous_keys - {line_key(line) for line in edited_lines}
    added = [line for line in edited_lines if line_key(line) not in previous_keys]
    kept = [line for line in lines(new_source) if line_key(line) not in removed]
    kept_keys = {line_key(line) for line in kept}
    return "\n".join(kept + [line for line in added if line_key(line) not in kept_keys])


def remember_widget_choice(session_state: Any, key: str) -> Any:
    """Record a value the user gave seed_widget_choice's widget ``key``, and return their choice (None if none).

    seed_widget_choice calls this itself. Call it earlier in the run as well
    where an st.rerun() can come before the widget renders: the rerun drops
    the widget's state, including a value changed along with the click that
    caused it. It also reads the choice while the widget isn't drawn.
    """
    current = session_state.get(key)
    if current is not None and current != session_state.get(_SEEDED_VALUE_PREFIX + key):
        if current == session_state.get(_SUGGESTED_VALUE_PREFIX + key):
            # Set back to what was suggested: follow the suggestion again.
            session_state.pop(_CHOICE_PREFIX + key, None)
        else:
            session_state[_CHOICE_PREFIX + key] = current
    return session_state.get(_CHOICE_PREFIX + key)


def seed_widget_choice(session_state: Any, key: str, suggested: Any, min_value: Any, max_value: Any) -> None:
    """Seed a keyed number widget whose default depends on other input, keeping the user's own choice.

    The widget follows ``suggested`` until the user sets a value of their
    own, and keeps that from then on, through later changes to ``suggested``
    and through an st.rerun() that drops the widget's state. Either way the
    value is clamped to [min_value, max_value], which may change between runs
    too; a choice clamped by a narrower range comes back once the range
    allows it again. Setting the widget back to the value it suggested makes
    it follow ``suggested`` again. Call this before the widget, and create the
    widget without ``value=``.
    """
    choice = remember_widget_choice(session_state, key)
    suggested = max(min_value, min(max_value, suggested))
    value = suggested if choice is None else max(min_value, min(max_value, choice))
    if session_state.get(key) != value:
        session_state[key] = value
    session_state[_SEEDED_VALUE_PREFIX + key] = value
    session_state[_SUGGESTED_VALUE_PREFIX + key] = suggested


def move_widget_state(session_state: Any, old_key: str, new_key: str) -> None:
    """Carry widget state from ``old_key`` over to ``new_key``, for a widget whose key changed.

    Moves the value and the bookkeeping seed_widget_from_source keeps for
    it, unless ``new_key`` already has a value; ``old_key``'s state is
    dropped either way.
    """
    if old_key in session_state and new_key not in session_state:
        for prefix in ("", _SOURCE_FINGERPRINT_PREFIX, _LAST_VALUE_PREFIX):
            if prefix + old_key in session_state:
                session_state[prefix + new_key] = session_state[prefix + old_key]
    forget_widget_state(session_state, old_key)


def forget_widget_state(session_state: Any, key: str) -> None:
    """Drop widget ``key``'s value along with any bookkeeping kept for it.

    Use when the thing a widget represents no longer exists (a deleted preset,
    a schedule that's been replaced), so a later widget reusing that key
    doesn't inherit stale state or a stale fingerprint.
    """
    session_state.pop(key, None)
    session_state.pop(_SOURCE_FINGERPRINT_PREFIX + key, None)
    session_state.pop(_LAST_VALUE_PREFIX + key, None)
    session_state.pop(_DEFERRED_PREFIX + key, None)
    session_state.pop(_CHOICE_PREFIX + key, None)
    session_state.pop(_SEEDED_VALUE_PREFIX + key, None)
    session_state.pop(_SUGGESTED_VALUE_PREFIX + key, None)
