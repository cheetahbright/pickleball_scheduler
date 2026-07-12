#!/usr/bin/env python3
"""Simple Streamlit auth helpers with optional secrets/env configuration."""

import hmac
import os
import threading
import time

import streamlit as st

try:
    from streamlit.errors import StreamlitSecretNotFoundError
except ImportError:  # pragma: no cover - defensive: some tests replace
    # sys.modules["streamlit"] with a bare stand-in lacking real submodules.
    # In that context _configured_password()'s except clause below simply
    # never matches (st.secrets is always mocked per-test anyway), so a
    # local placeholder is a safe fallback rather than crashing at import.
    class StreamlitSecretNotFoundError(Exception):  # type: ignore[no-redef]
        pass

PASSWORD_ENV_VAR = "PICKLEBALL_APP_PASSWORD"
PASSWORD_SECRET_KEY = "app_password"
TRUE_VALUES = {"1", "true", "yes"}

MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_SECONDS = 600

# Failed-login timestamps for the single shared app password, tracked process-wide
# rather than in st.session_state. session_state is scoped per browser session, so
# an attacker could reset the lockout just by opening a new tab/incognito window.
# Streamlit Community Cloud runs one process per app, so module-level state is shared
# across every session; the lock guards concurrent ScriptRunner threads. A process
# restart clears this, which is acceptable - it is not an attacker-triggerable reset.
_failed_attempts_lock = threading.Lock()
_failed_attempts: list[float] = []


def _configured_password() -> str | None:
    """Return the configured password from secrets or environment."""
    try:
        secrets = getattr(st, "secrets", None)
        if secrets and hasattr(secrets, "get"):
            secret_password = secrets.get(PASSWORD_SECRET_KEY)
            if isinstance(secret_password, str) and secret_password:
                return secret_password
    except StreamlitSecretNotFoundError:
        # Streamlit raises this both when no secrets.toml exists at all and
        # when one exists but fails to parse - it does not distinguish the
        # two. Either way, falling through to the env var (and ultimately to
        # "auth not configured") is the correct behavior. Any OTHER
        # exception here is a real bug in this code, not an absence of
        # configuration, and must not be silently treated as "no password".
        pass

    env_password = os.environ.get(PASSWORD_ENV_VAR)
    if isinstance(env_password, str) and env_password:
        return env_password

    return None


def _is_e2e_mode() -> bool:
    """Allow browser automation to bypass login intentionally."""
    return os.environ.get("E2E_TEST", "").lower() in TRUE_VALUES


def _recent_failed_attempts() -> list[float]:
    """Return failed-login timestamps still inside the rate-limit window, pruning stale ones."""
    cutoff = time.time() - LOGIN_ATTEMPT_WINDOW_SECONDS
    with _failed_attempts_lock:
        _failed_attempts[:] = [attempt for attempt in _failed_attempts if attempt > cutoff]
        return list(_failed_attempts)


def _record_failed_attempt() -> None:
    """Record one failed login against the process-wide rate-limit window."""
    cutoff = time.time() - LOGIN_ATTEMPT_WINDOW_SECONDS
    now = time.time()
    with _failed_attempts_lock:
        _failed_attempts[:] = [attempt for attempt in _failed_attempts if attempt > cutoff]
        _failed_attempts.append(now)


def _reset_failed_attempts() -> None:
    """Clear the failed-login window after a successful login."""
    with _failed_attempts_lock:
        _failed_attempts.clear()


def simple_auth():
    """Return True when auth is satisfied or disabled by configuration."""
    if _is_e2e_mode() and _configured_password() is not None:
        raise RuntimeError(
            "E2E_TEST auth bypass is enabled while an app password is configured - "
            "refusing to start with authentication disabled. Unset E2E_TEST in any "
            "environment where PICKLEBALL_APP_PASSWORD or the app_password secret is set."
        )

    if _is_e2e_mode():
        st.session_state.simple_auth = True
        return True

    if st.session_state.get("simple_auth", False):
        return True

    configured_password = _configured_password()
    if configured_password is None:
        st.session_state.simple_auth = True
        return True

    st.title("🎾 Pickleball Scheduler")
    st.markdown("### Quick Login")

    if len(_recent_failed_attempts()) >= MAX_LOGIN_ATTEMPTS:
        st.error(
            f"🚫 Too many failed login attempts. Try again in up to " f"{LOGIN_ATTEMPT_WINDOW_SECONDS // 60} minutes."
        )
        return False

    password = st.text_input("Password:", type="password")

    if st.button("Login"):
        if hmac.compare_digest(password.encode("utf-8"), configured_password.encode("utf-8")):
            st.session_state.simple_auth = True
            _reset_failed_attempts()
            st.success("✅ Welcome!")
            st.rerun()
        else:
            _record_failed_attempt()
            st.error("❌ Wrong password")

    return False


def logout():
    """Simple logout"""
    st.session_state.simple_auth = False
    st.rerun()
