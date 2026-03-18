"""Compatibility wrapper for the legacy ``security.headers`` import path."""

from src.security_headers import SecurityHeaders, headers

__all__ = ["SecurityHeaders", "headers"]
