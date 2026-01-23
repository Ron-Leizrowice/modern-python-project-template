"""Define CLI entrypoints for the project.

IMPORTANT: Be sure to register your CLI entrypoints in the `[project.scripts]` section of `pyproject.toml`.
"""

import click


@click.command()
def hello_world() -> None:
    """Print a friendly greeting."""
    click.echo("Hello, World!")
