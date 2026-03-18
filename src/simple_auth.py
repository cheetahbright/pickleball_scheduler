#!/usr/bin/env python3
"""Simple Streamlit auth helpers with optional secrets/env configuration."""

import os

import streamlit as st

PASSWORD_ENV_VAR = "PICKLEBALL_APP_PASSWORD"
PASSWORD_SECRET_KEY = "app_password"
TRUE_VALUES = {"1", "true", "yes"}


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

    password = st.text_input("Password:", type="password")

    if st.button("Login"):
        if password == configured_password:
            st.session_state.simple_auth = True
            st.success("✅ Welcome!")
            st.rerun()
        else:
            st.error("❌ Wrong password")

    return False


def logout():
    """Simple logout"""
    st.session_state.simple_auth = False
    st.rerun()
