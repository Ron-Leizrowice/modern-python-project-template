# Modern Python Code Style Guide

**Target:** Python 3.13+

## Typing

- Use built‑in generics for concrete collections: `list[int]`, `dict[str, Any]`, `tuple[int, ...]`, `set[str]`.
- Import abstract types from `collections.abc`: `Iterable`, `Mapping`, `Sequence`, `Callable`.
- Use unions with `|` and optionals with `| None` instead of `Union` / `Optional`.
- Use modern type parameters and `type` aliases (PEP 695).
- Mark intentional overrides with `@override` (Python ≥ 3.12).
- Prefer structural typing with `Protocol` over nominal ABCs when it improves flexibility.
- On 3.13+, no `from __future__ import annotations` is needed; add it only if supporting older versions.
- Use `typing_extensions` only to backport newer typing features.
- Keep `Any` rare; prefer precise types, type variables, or protocols.
- Use `assert` for type narrowing.

## Syntax & Patterns

- Prefer `match` over long `if`/`elif` chains when it clarifies intent.
- Use assignment expressions (`:=`) judiciously for clear, single‑pass logic.
- Prefer f‑strings for formatting.
- Use comprehensions and generator expressions when they stay readable.
- Use context managers (`contextlib`, `ExitStack`) for resources.
- Use `asyncio` for I/O‑bound work; avoid blocking calls in async paths.
- Use `StrEnum` for fixed string variants.
- Replace magic numbers/strings with named constants.
- Avoid legacy pre‑3.10 idioms.

## Example

```python
from collections.abc import Iterable, Sequence, Mapping, Callable
from typing import Any, Protocol, override
from enum import StrEnum

# Type alias (PEP 695)
type UserId = int

# StrEnum for variants
class Status(StrEnum):
    OK = "ok"
    ERROR = "error"

# Structural typing
class Reader(Protocol):
    def read(self, n: int) -> bytes: ...

# Generic function with modern type params
def head[T](xs: Sequence[T]) -> T:
    return xs[0]

# Built-in generics + unions
def parse(data: bytes | str) -> dict[str, Any]:
    return {"len": len(data), "status": Status.OK}

# Constants, not magic values
TIMEOUT_SECONDS = 30

# Async for I/O-bound work
async def fetch(reader: Reader) -> bytes:
    # placeholder for async I/O
    return await some_async_io(reader)

class Base:
    def value(self) -> int: ...

class Child(Base):
    @override
    def value(self) -> int:
        return 1

def apply(xs: Iterable[int], f: Callable[[int], int]) -> list[int]:
    return [f(x) for x in xs]

def describe(status: Status) -> str:
    match status:
        case Status.OK:
            return f"ok in {TIMEOUT_SECONDS}s"
        case Status.ERROR:
            return "error"
```
