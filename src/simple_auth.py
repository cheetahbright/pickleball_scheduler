#!/usr/bin/env python3
"""Simple Streamlit auth helpers with optional secrets/env configuration."""

import hashlib
import hmac
import logging
import os
import threading
import time

import streamlit as st

try:
    from src._compat import import_module_with_fallback
except ImportError:
    from _compat import import_module_with_fallback

try:
    from streamlit.errors import StreamlitSecretNotFoundError
except ImportError:  # pragma: no cover - defensive: some tests replace
    # sys.modules["streamlit"] with a bare stand-in lacking real submodules.
    # In that context _configured_password()'s except clause below simply
    # never matches (st.secrets is always mocked per-test anyway), so a
    # local placeholder is a safe fallback rather than crashing at import.
    class StreamlitSecretNotFoundError(Exception):  # type: ignore[no-redef]
        pass


# Read as features.<FLAG> at call time, never copied, so a patched flag takes effect.
features = import_module_with_fallback("features")

PASSWORD_ENV_VAR = "PICKLEBALL_APP_PASSWORD"
PASSWORD_SECRET_KEY = "app_password"
TRUE_VALUES = {"1", "true", "yes"}

logger = logging.getLogger(__name__)

MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_SECONDS = 600
MAX_LOGIN_DELAY_SECONDS = 30

# This session's failed-login timestamps: MAX_LOGIN_ATTEMPTS of them inside the
# window close the login form for this session only.
FAILED_ATTEMPTS_KEY = "simple_auth_failed_attempts"

# A session records which password it logged in with, as a keyed hash rather
# than the password itself. Changing the password then signs out sessions that
# are already open - e.g. an intruder's, after a leak - not just new visitors.
# The hash key is random per session and kept in the session, not in this
# module: when any source file changes under a running server, Streamlit
# re-imports every app module, and a new key would sign everyone out.
PASSWORD_FINGERPRINT_KEY = "simple_auth_password_fingerprint"
PASSWORD_FINGERPRINT_SALT_KEY = "simple_auth_password_fingerprint_salt"

# Set while the "Log out?" dialog is open (see logout()).
LOGOUT_PENDING_KEY = "simple_auth_logout_pending"

# Login-attempt timestamps from every session, tracked process-wide as well as per
# session, since a new tab or incognito window starts a fresh st.session_state.
# Streamlit Community Cloud runs one process per app, so module-level state is
# shared across every session; the lock guards concurrent ScriptRunner threads.
# Past MAX_LOGIN_ATTEMPTS in the window, every submitted password waits before it
# is checked (see _login_delay). The wait is capped, so no visitor can use this to
# lock the organizer out: the right password always gets in, at worst
# MAX_LOGIN_DELAY_SECONDS late. The cap also means this is no server-wide limit:
# someone holding many sessions open at once gets one guess per session per wait,
# so only a long, random password stands up to that. A process restart clears
# this, which is acceptable - it is not an attacker-triggerable reset.
_failed_attempts_lock = threading.Lock()
_failed_attempts: list[float] = []


class AuthConfigError(RuntimeError):
    """The password settings can't be read, so the app must not open."""


def _configured_password() -> str | None:
    """Return the configured password from secrets or environment.

    Raises AuthConfigError when the secrets can't say whether a password is
    set, rather than treating that as "no password" and opening the app.
    """
    try:
        secrets = getattr(st, "secrets", None)
        if secrets and hasattr(secrets, "get"):
            secret_password = secrets.get(PASSWORD_SECRET_KEY)
            if secret_password is None and f"\ufeff{PASSWORD_SECRET_KEY}" in secrets:
                # Streamlit reads a UTF-8 byte-order mark (Windows PowerShell 5.1's
                # -Encoding UTF8 writes one) as part of the first key.
                raise AuthConfigError(
                    "The secrets file (secrets.toml) starts with a byte-order mark, so its app_password can't be "
                    "read. Save it as UTF-8 without a BOM and reload."
                )
            if isinstance(secret_password, (int, float)) and not isinstance(secret_password, bool):
                # An unquoted PIN (app_password = 482913) is a number in TOML.
                return str(secret_password)
            if isinstance(secret_password, str):
                if secret_password:
                    return secret_password
            elif secret_password is not None:
                raise AuthConfigError(
                    'The app_password secret must be text in quotes, for example app_password = "your password".'
                )
    except StreamlitSecretNotFoundError as exc:
        # Streamlit raises this both when no secrets.toml exists and when one
        # fails to parse; only a parse failure has the TOML error as its cause.
        # No file means no password there, so fall through to the env var. A
        # file that doesn't parse may hold one, so the app stays closed. Any
        # OTHER exception here is a real bug in this code, not an absence of
        # configuration, and must not be silently treated as "no password".
        if exc.__cause__ is not None:
            logger.error("Could not parse the Streamlit secrets: %s", exc)
            raise AuthConfigError(
                "The secrets file (secrets.toml) could not be read, so the app can't tell whether a password "
                "is set. Fix the file and reload."
            ) from exc
    except UnicodeDecodeError as exc:
        # Streamlit reads secrets.toml as UTF-8 and doesn't catch this. When the
        # file is re-saved as, say, UTF-16 while the app runs, Streamlit refuses
        # new connections, but sessions already open would show the traceback.
        logger.error("Could not decode the Streamlit secrets: %s", exc)
        raise AuthConfigError(
            "The secrets file (secrets.toml) could not be read as UTF-8 text, so the app can't tell whether a "
            "password is set. Save it as UTF-8 and reload."
        ) from exc

    env_password = os.environ.get(PASSWORD_ENV_VAR)
    if isinstance(env_password, str) and env_password:
        return env_password

    return None


def _is_e2e_mode() -> bool:
    """Allow browser automation to bypass login intentionally."""
    return os.environ.get("E2E_TEST", "").lower() in TRUE_VALUES


def _session_failed_attempts() -> list[float]:
    """Return this session's failed-login timestamps still inside the window, pruning stale ones."""
    cutoff = time.time() - LOGIN_ATTEMPT_WINDOW_SECONDS
    stored = st.session_state.get(FAILED_ATTEMPTS_KEY)
    attempts = [attempt for attempt in stored if attempt > cutoff] if isinstance(stored, list) else []
    st.session_state[FAILED_ATTEMPTS_KEY] = attempts
    return attempts


def _record_failed_attempt() -> int:
    """Count one login attempt as failed in the process-wide window; return how many it now holds.

    Called before the password is checked, so attempts made at the same time each
    count; a successful login clears the window (_reset_failed_attempts).
    """
    cutoff = time.time() - LOGIN_ATTEMPT_WINDOW_SECONDS
    now = time.time()
    with _failed_attempts_lock:
        _failed_attempts[:] = [attempt for attempt in _failed_attempts if attempt > cutoff]
        _failed_attempts.append(now)
        return len(_failed_attempts)


def _login_delay(recent_attempts: int) -> float:
    """Seconds a submitted password waits before it is checked: none for the first
    MAX_LOGIN_ATTEMPTS failed attempts in the window, then doubling up to MAX_LOGIN_DELAY_SECONDS."""
    extra_attempts = recent_attempts - MAX_LOGIN_ATTEMPTS
    if extra_attempts <= 0:
        return 0.0
    return float(min(MAX_LOGIN_DELAY_SECONDS, 2 ** min(extra_attempts, 8)))


def _reset_failed_attempts() -> None:
    """Clear the failed-login window after a successful login."""
    with _failed_attempts_lock:
        _failed_attempts.clear()


def _password_fingerprint(password: str) -> str:
    """Return a keyed hash that identifies password without storing it."""
    if not isinstance(st.session_state.get(PASSWORD_FINGERPRINT_SALT_KEY), bytes):
        st.session_state[PASSWORD_FINGERPRINT_SALT_KEY] = os.urandom(32)
    salt = st.session_state[PASSWORD_FINGERPRINT_SALT_KEY]
    return hmac.new(salt, password.encode("utf-8"), hashlib.sha256).hexdigest()


def simple_auth():
    """Return True when auth is satisfied or disabled by configuration."""
    try:
        configured_password = _configured_password()
    except AuthConfigError as exc:
        # Fail closed: settings that can't be read must not open the app.
        st.title("🎾 Pickleball Scheduler")
        st.error(f"🚫 {exc}")
        return False

    if _is_e2e_mode() and configured_password is not None:
        raise RuntimeError(
            "E2E_TEST auth bypass is enabled while an app password is configured - "
            "refusing to start with authentication disabled. Unset E2E_TEST in any "
            "environment where PICKLEBALL_APP_PASSWORD or the app_password secret is set."
        )

    if _is_e2e_mode():
        st.session_state.simple_auth = True
        return True

    if configured_password is None:
        st.session_state.simple_auth = True
        return True

    if st.session_state.get("simple_auth", False):
        stored_fingerprint = st.session_state.get(PASSWORD_FINGERPRINT_KEY)
        if isinstance(stored_fingerprint, str) and hmac.compare_digest(
            stored_fingerprint, _password_fingerprint(configured_password)
        ):
            if st.session_state.get(LOGOUT_PENDING_KEY, False):
                _open_logout_confirmation()
            return True
        # The password has changed since this session logged in.
        st.session_state.simple_auth = False

    st.title("🎾 Pickleball Scheduler")
    st.markdown("### Quick Login")

    # Only this session's own failures close its form, so no visitor can lock
    # anyone else out; a guesser who opens new sessions meets the wait below.
    if len(_session_failed_attempts()) >= MAX_LOGIN_ATTEMPTS:
        st.error(
            f"🚫 Too many failed login attempts. Try again in up to " f"{LOGIN_ATTEMPT_WINDOW_SECONDS // 60} minutes."
        )
        return False

    # A form, so Enter (Go on a phone keyboard) submits; "current-password"
    # lets password managers fill the saved password instead of offering a new one.
    with st.form("login_form"):
        password = st.text_input("Password:", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not password:
            # Not a guess, so it doesn't count toward the lockout.
            st.warning("Enter the password, then tap Login.")
        else:
            # Wait BEFORE checking, right password or not: delaying only the
            # "Wrong password" answer would let a guesser who never waits for it
            # learn of a right guess at once.
            delay = _login_delay(_record_failed_attempt())
            if delay:
                time.sleep(delay)
            if hmac.compare_digest(password.encode("utf-8"), configured_password.encode("utf-8")):
                st.session_state.simple_auth = True
                st.session_state[PASSWORD_FINGERPRINT_KEY] = _password_fingerprint(configured_password)
                st.session_state[LOGOUT_PENDING_KEY] = False
                st.session_state[FAILED_ATTEMPTS_KEY] = []
                _reset_failed_attempts()
                st.success("✅ Welcome!")
                st.rerun()
            else:
                st.session_state[FAILED_ATTEMPTS_KEY] = [*_session_failed_attempts(), time.time()]
                st.error("❌ Wrong password")

    return False


def _confirm_logout() -> None:
    st.session_state[LOGOUT_PENDING_KEY] = False
    st.session_state.simple_auth = False


def _cancel_logout() -> None:
    st.session_state[LOGOUT_PENDING_KEY] = False


def _logout_confirmation() -> None:
    """Body of the "Log out?" dialog."""
    if not st.session_state.get(LOGOUT_PENDING_KEY, False):
        # A button below was pressed and its callback has acted: close the dialog.
        st.rerun()
    unsaved = "typed scores, replan changes" if features.SCORE_TRACKING else "replan changes"
    st.write(f"Anything not saved yet will be lost: {unsaved} and unsaved Configuration edits.")
    st.button("Log out", type="primary", on_click=_confirm_logout)
    st.button("Cancel", on_click=_cancel_logout)


def _open_logout_confirmation() -> None:
    st.dialog("Log out?", on_dismiss=_cancel_logout)(_logout_confirmation)()


def auth_enabled() -> bool:
    """Whether visitors must log in, so there is a login page to log out to."""
    return not _is_e2e_mode() and _configured_password() is not None


def logout():
    """Ask before logging out.

    Logging out renders only the login page, so Streamlit drops the state of
    every main-page widget, including input not saved yet. The dialog opens
    over the page, which keeps rendering, so nothing is dropped unless the
    user confirms. While it is pending, simple_auth() reopens it on every full
    rerun; its buttons act in on_click callbacks, which run first whether the
    click reruns only the dialog (a browser) or the whole script (AppTest).
    """
    if not auth_enabled():
        # No login page to log out to; the page shows no Logout button then.
        return
    if not st.session_state.get(LOGOUT_PENDING_KEY, False):
        st.session_state[LOGOUT_PENDING_KEY] = True
        _open_logout_confirmation()
