"""Tests for CLI entrypoints."""

from click.testing import CliRunner

from modern_python_template.cli import hello_world


def test_hello_world() -> None:
    """Test the hello_world function."""
    runner = CliRunner()
    result = runner.invoke(hello_world)

    assert result.exit_code == 0
    assert result.output == "Hello, World!\n"
