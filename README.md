# Python 2025 Template • Opinionated Guide

This is my opinionated guide to python development. It should cover most of the major tools and libraries you will need, as well provide reasoning as to why they are currently the best in class. I will also try and address popular alternatives, and explain when they make sense or why I do not reccomend them.

## Configuration

As of 2025, the standard for all python project configuration is a `pyproject.toml`. Almost all python tooling is now configurable in this file, and is the de-facto standard. Seperate config files should only be introduced when:

a. The tool does not support configuration via pyproject.toml
b. The config section would be too long and make the pyproject.toml cumbersome to navigate (e.g. ruff.toml)

## Project management: `uv`

Many of my opinions and suggestions are contestable, this is not. If you are working with python in 2025, you should be using `uv`. No ifs, no buts. I don't care if you think a `requirements.txt` is enough, or if you are used to `conda`, or if you think `poetry` already solves all your dependency management problems. Stop whatever you are doing, and migrate to `uv` right now.

Per the Astral website, `uv` is: "An extremely fast Python package and project manager, written in Rust.". Let me break down for you what that means and why you should care:

- Environment management makes it trivially easy to swap between different python builds and environments in different projects, without risking your global system python which is often required for your operating system to function.
- Package management ensures you can easily add, remove, and update dependencies while ensuring compatibility with your CPU architecture, operating system, python version, and critically - other dependencies.
- Project management makes it easy to run, build, publish, or bundle your project - and can even handle versioning for you.
- It's fast, REALLY fast. 10-100x faster than `pip`. If you use `conda`, the difference will be earth-shattering.
- "Written in Rust" may not sound like a feature, but it helps ensure it runs fast, fewer bugs are introduced, and security vulnerabilities are much rarer.

If you've written Rust before, then `uv` is the python equivalent to `cargo`. It replaces `virtualenv`, `pyenv`, `pip`, `pipx`, `pip-tools`, `poetry`, `conda`, `twine`.

How to get started:

1. Install `uv` in your preferred manner: <https://docs.astral.sh/uv/getting-started/installation/>
2. Install the latest version of python: `uv python install 3.13`
3. Install the python environment with: `uv sync`
4. Run a script: `uv run hello-world`

It's that easy! For a more in depth guide, check out the documentation at <https://docs.astral.sh/uv/getting-started/>

## Linter & Formatter: `ruff`

Much like `uv` (and made by the same team!), `ruff` is a fast, extensible linter and formatter for Python. It's use is also non-negotiable. It doesn't really have any direct equivalents to compare to but was designed to be a drop-in replacement for `flake8` and `black` and it's much faster than both. It should also replace `isort`, `pydocstyle`, `pyupgrade`, `autoflake`, and any other code formatting and linting tools.

I have enabled most of the `ruff` rules by default, with a few sensible exceptions. You may find various rules irritating, or require too much refactoring in an existing codebase. I recommend disabling them individually in the `ruff.toml` rather than disabling the entire rule set, as most of the rules have solid reasoning behind their existence.

### Philosophy of Formatters

Many python developers, especially data-scientists or those coming from academia, do not use or appreciate formatters. Fixed line lengths, pedantic import order, and other rules are often seen as unnecessary or even harmful to productivity. However here are just a few of the many benefits:

- Readability and consistency: all code will be laid out roughly similar regardless of where it is or who wrote it, making it easier to read and understand.
- Maintainability: consistent code formatting makes it easier to read diffs in git commits/PRs and makes it easier to review changes.
- Reduced cognitive load: you can focus on the logic of the code, instead of thinking about how it should be formatted.

### Philosophy of Linters

The case for linters is even stronger than formatters. Linters can catch bugs and enforce best practices that are not possible to enforce with formatters alone. Here are just a few of the many benefits:

- Catching bugs: linters can catch bugs without the need for testing.
- Enforcing best practices: linters can enforce best practices that are not possible to enforce with formatters alone.
- Improved code quality: linters can improve code quality by enforcing best practices and catching bugs.
- Improved collaboration: linters can help ensure that code is consistent across different developers and teams.

## Static Type Checking: `basedpyright` or `pyright` (`ty`?)

When you first start learning python, one of the most common selling points is its simple and flexible type system. In reality, its one of pythons biggest downsides, and huge amounts of time and engineering work have been invested into solving this. In 2025 I argue that its worth devoting the effort to coerce python into a pseudo-statically typed language. The first step to achieving this is `ruff` which will force you to include type hints for function signatures at very least. The second is liberal use of type hints for variables, constants, class attributes, and other objects - ideally even using typed classes via `dataclasses`, `msgspec`, or `pydantic`. Thirdly, and most importantly, you need a static type checker to catch type errors at compile time.

### `mypy` vs `basedpyright`

`mypy` is the OG static type checker for python. It (probably) still the most widely used and is still actively maintained. It is still an excellent choice, but suffers from some major issues, many resulting from the choice to write it in pure python, such as its slow performance and lack of type inference. `pyright` was developed by microsoft as an alternative in typescript making it significantly faster and more powerful than `mypy`.

`basedpyright` is a fork of `pyright` that adds numerous new features, along with opinionated defaults that make it easier to use without extensive setup. It also functions as a language server which allows you to use it and see the errors in realtime in your IDE. There is also a `basedmypy` but it is no longer maintained, and its own developers reccomend using `basedpyright` or `ty` (see next section) instead.

The main advantage of `mypy` over `pyright` is the plugin system which provides better type support for third-party libraries and frameworks, like `pydantic`. If speed isn't a concern, and you are running into false-positive type errors with `pyright` then `mypy` is a good choice too.

### `pyrefly` and `ty`

However, there are two new static type checkers you should be aware of, and I encourage you to use - at least as type-checkers if not as language servers. Both are written in rust, which makes them significantly faster.

`ty` is developed by Astral (also the developers of `uv` and `ruff`). It is extremely fast, and looks to be very promising, but even by the teams own admission it is still far from being ready for use in production-grade projects, and its language server is still very immature.

`pyrefly` is developed by facebook/meta, and is much more mature than `ty`. It is significantly slower than `ty`, although both are 10-100x faster than `mypy` or `pyright` anyway. `pyrefly` has experimental support for third-party libraries and frameworks, like `pydantic`, and is being rapidly developed. It will likely be a fully fledged competitor to `mypy` and `pyright` in the very near future.

### Conclusion

If you are working on a production grade project, where code will be deployed and you need strong type-safety guarantees, then `basedpyright` would be my first choice. If your requirements are more relaxed, then I would recommend `pyrefly`. In either case, I would include `ty` as a warning-only type-check in your CI/pre-commit and wait for it to grow and improve.

## Dependency Checker: `deptry`

Not much to say here, you will likely want a tool to help you catch unused dependencies, and `deptry` is the best option available. It's fast (also written in Rust, you may sense a pattern emerging), its accurate, and easy to configure. You can use `fawltydeps` or `py-unused-deps` if you prefer, but they are less accurate or slower.
