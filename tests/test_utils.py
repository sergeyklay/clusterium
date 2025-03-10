"""Test the utils module."""

import pytest

from qadst.utils import if_ok, is_numeric, to_numeric


@pytest.mark.parametrize(
    "fn, string, expected_result",
    [
        # Integer conversion
        (int, "123", 123),
        (int, "-123", -123),
        (int, "0", 0),
        # Float conversion
        (float, "123.45", 123.45),
        (float, "-123.45", -123.45),
        (float, "0.0", 0.0),
        (float, ".5", 0.5),
        (float, "1e6", 1000000.0),
        (float, "-1e6", -1000000.0),
        # Complex conversion
        (complex, "1+2j", 1 + 2j),
        (complex, "-1-2j", -1 - 2j),
        # Invalid conversions
        (int, "123.45", None),
        (int, "abc", None),
        (int, "", None),
        (int, "1e6", None),
        (float, "abc", None),
        (float, "", None),
        (complex, "abc", None),
        (complex, "", None),
    ],
)
def test_if_ok(fn, string, expected_result):
    """Test the if_ok method with various inputs."""
    result = if_ok(fn, string)
    assert result == expected_result


@pytest.mark.parametrize(
    "string, expected_result",
    [
        # Numeric strings
        ("123", True),
        ("-123", True),
        ("0", True),
        ("123.45", True),
        ("-123.45", True),
        ("0.0", True),
        (".5", True),
        ("1e6", True),
        ("-1e6", True),
        ("1+2j", True),
        ("-1-2j", True),
        ("inf", True),
        ("-inf", True),
        ("nan", True),
        # Non-numeric strings
        ("abc", False),
        ("", False),
        ("123abc", False),
        ("abc123", False),
        ("12.34.56", False),
        ("12,345", False),
        ("$123", False),
        ("None", False),
        ("True", False),
        ("False", False),
    ],
)
def test_is_numeric(string, expected_result):
    """Test the is_numeric method with various inputs."""
    result = is_numeric(string)
    assert result == expected_result


@pytest.mark.parametrize(
    "string, expected_result",
    [
        # Regular numeric strings
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("10", 10),
        ("100", 100),
        ("-1", -1),
        ("-2", -2),
        # Floating point numbers
        ("0.0", 0.0),
        ("0.1", 0.1),
        ("1.0", 1.0),
        ("1.1", 1.1),
        ("1.2", 1.2),
        ("2.0", 2.0),
        ("10.5", 10.5),
        ("-1.0", -1.0),
        ("-1.1", -1.1),
        # Scientific notation
        ("1e6", 1000000.0),
        ("1.0e6", 1000000.0),
        # Floating point numbers with non-numeric components
        ("a.0", None),
        ("0.a", None),
        ("a.b", None),
        # Non-numeric strings
        ("a", None),
        ("abc", None),
        ("", None),
        ("None", None),
        ("True", None),
        ("False", None),
        # Edge cases
        ("1.0.0", None),
        ("1,000", None),
    ],
)
def test_to_numeric(string, expected_result):
    """Test the to_numeric method with various inputs."""
    result = to_numeric(string)
    assert result == expected_result
