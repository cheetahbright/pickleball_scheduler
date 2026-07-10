"""Security compatibility package for legacy imports."""

from .input_sanitizer import InputSanitizer, SecureJSONLoader, SecurityError

__all__ = [
    "InputSanitizer",
    "SecureJSONLoader",
    "SecurityError",
]
