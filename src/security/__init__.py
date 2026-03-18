"""Security compatibility package for legacy imports."""

from .headers import SecurityHeaders, headers
from .input_sanitizer import InputSanitizer, SecureJSONLoader, SecurityError

__all__ = [
    "InputSanitizer",
    "SecureJSONLoader",
    "SecurityError",
    "SecurityHeaders",
    "headers",
]
