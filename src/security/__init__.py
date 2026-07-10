"""Security compatibility package for legacy imports."""

from .input_sanitizer import InputSanitizer, SecurityError

__all__ = [
    "InputSanitizer",
    "SecurityError",
]
