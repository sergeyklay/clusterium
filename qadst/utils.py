"""Utility functions."""

from typing import Any, Callable, Optional, Union


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


def is_numeric(string: str) -> bool:
    """Check if a string can be converted to a numeric type.

    Args:
        string: String to check

    Returns:
        True if the string can be converted to int, float, or complex
    """
    # Special case for numeric types
    if isinstance(string, (int, float, complex)):
        return True

    # Special case for empty strings or non-string values
    if not string or not isinstance(string, str) or string.strip() == "":
        return False

    # Try to convert to numeric types
    return (
        if_ok(int, string) is not None
        or if_ok(float, string) is not None
        or if_ok(complex, string) is not None
    )


def to_numeric(string: Any) -> Optional[Union[int, float, complex]]:
    """Convert a string to the most appropriate numeric type (int, float, complex).

    Args:
        string: Input value to convert. If already a numeric type, it is returned as-is.

    Returns:
        The converted numeric value (int, float, complex) if possible, otherwise None.
    """
    # Check if the input is already a numeric type
    if isinstance(string, (int, float, complex)):
        return string

    # Proceed only if the input is a string
    if not isinstance(string, str):
        return None

    # Check in order: int, float, complex
    for converter in (int, float, complex):
        result = if_ok(converter, string)
        if result is not None:
            return result

    return None
