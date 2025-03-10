"""Utility functions."""

from typing import Any, Callable, Optional


def if_ok(fn: Callable, string: str) -> Optional[Any]:
    """Try to apply a function to a string, return None if it fails.

    Args:
        fn: Function to apply
        string: String to apply the function to

    Returns:
        Result of fn(string) or None if an exception occurs
    """
    try:
        return fn(string)
    except (ValueError, NameError, TypeError):
        return None


def is_numeric(string):
    """Check if a string can be converted to a numeric type.

    Args:
        string: String to check

    Returns:
        True if the string can be converted to int, float, or complex
    """
    # Special case for empty strings
    if not string:
        return False

    # Try to convert to numeric types
    return (
        if_ok(int, string) is not None
        or if_ok(float, string) is not None
        or if_ok(complex, string) is not None
    )
