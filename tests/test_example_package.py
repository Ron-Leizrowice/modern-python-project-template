"""Test the example_package module."""

import importlib.util

import pytest


def test_import() -> None:
    if importlib.util.find_spec("example_package") is None:
        pytest.fail("Failed to find example_package module")
