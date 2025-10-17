# Repository Guidelines

## Environment Setup

Use `uv` to manage dependencies; run `uv sync` from the repo root to create the virtual environment with the `dev` and `lint` groups enabled. Activate shells with `uv run <command>` so tooling stays in sync with `uv.lock`. Target Python 3.10+; the ruff config is tuned for Python 3.13 syntax, so keep uv pinned to a modern interpreter.

## Coding Style & Naming Conventions

- Follow the `ruff.toml` defaults: 4-space indentation, 120-character max lines, and double-quoted strings.
- Prefer fast-fail, do not add fallbacks or exception handling.
- See CODESTYLE.md for detailed Python style guidelines.

## Running Code

Since the python environment is managed by `uv`, run scripts and modules with `uv run <command>`. For example, to execute the training script, use:

```zsh
uv run python -m llm_training_example.train --help
```

## Code Quality

Whenever you are finished making substantial changes to the code, ensure you have run the formatter, linter, and type-checker. You can do this by running the following commands from the root of the repository:

```zsh
uv run ruff format .
uv run ruff check . -- fix
uv run pyreyfly check .
```

Note that once you finished fixing any issues arising from the linter or type-checker, you should re-run the commands to ensure that all issues have been resolved, and that no new issues have been introduced.
