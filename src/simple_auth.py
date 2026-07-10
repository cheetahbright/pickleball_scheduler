#!/usr/bin/env python3
"""Simple Streamlit auth helpers with optional secrets/env configuration."""

import hmac
import os
import time

import streamlit as st

PASSWORD_ENV_VAR = "PICKLEBALL_APP_PASSWORD"
PASSWORD_SECRET_KEY = "app_password"
TRUE_VALUES = {"1", "true", "yes"}

MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_SECONDS = 600


def _configured_password() -> str | None:
    """Return the configured password from secrets or environment."""
    try:
        secrets = getattr(st, "secrets", None)
        if secrets and hasattr(secrets, "get"):
            secret_password = secrets.get(PASSWORD_SECRET_KEY)
            if isinstance(secret_password, str) and secret_password:
                return secret_password
    except Exception:
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
    attempts = st.session_state.get("simple_auth_failed_attempts", [])
    cutoff = time.time() - LOGIN_ATTEMPT_WINDOW_SECONDS
    recent = [attempt for attempt in attempts if attempt > cutoff]
    st.session_state.simple_auth_failed_attempts = recent
    return recent


def simple_auth():
    """Return True when auth is satisfied or disabled by configuration."""
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
            st.session_state.simple_auth_failed_attempts = []
            st.success("✅ Welcome!")
            st.rerun()
        else:
            st.session_state.simple_auth_failed_attempts = _recent_failed_attempts() + [time.time()]
            st.error("❌ Wrong password")

    return False


def logout():
    """Simple logout"""
    st.session_state.simple_auth = False
    st.rerun()
