"""Main package entrypoint for executing scripts."""

from __future__ import annotations

import asyncio
from time import perf_counter

from rich import print as pprint

from example_package.json_loading_demo import (
    DATASET_CONFIGS,
    JsonLoader,
    TypedDecoder,
    benchmark_json_loading,
    benchmark_typed_decoding,
)
from example_package.mandelbrot import ExecutionMode, time_mandelbrot


def hello_world() -> None:
    """Print a greeting message."""
    pprint("Hello, world!")


def json_loading_performance_demo() -> None:
    """Demonstrate JSON loading performance across multiple parser backends."""

    results = asyncio.run(benchmark_json_loading())

    for config in DATASET_CONFIGS:
        backend_timings = results.get(config)
        if backend_timings is None:
            continue

        pprint(f"Dataset rows={config.rows:,}, tags per record={config.tag_count}")
        for backend in JsonLoader:
            elapsed = backend_timings.get(backend)
            if elapsed is None:
                continue
            pprint(f"  {backend.value}: {elapsed:.3f} seconds")
        pprint("")


def typed_decoding_demo() -> None:
    """Demonstrate typed decoding performance from JSON files on disk."""

    results = asyncio.run(benchmark_typed_decoding())

    for config in DATASET_CONFIGS:
        decoder_timings = results.get(config)
        if decoder_timings is None:
            continue

        pprint(
            f"Typed decoding rows={config.rows:,}, tags per record={config.tag_count}",
        )
        for decoder in TypedDecoder:
            elapsed = decoder_timings.get(decoder)
            if elapsed is None:
                continue
            pprint(f"  {decoder.value}: {elapsed:.3f} seconds")
        pprint("")


def mandelbrot_performance_demo() -> None:
    """Demonstrate Mandelbrot set performance across multiple execution backends."""

    for mode in ExecutionMode:
        t0 = perf_counter()
        _ = time_mandelbrot(mode)
        pprint(f"Elapsed time ({mode.value}): {perf_counter() - t0:.3f} seconds")


if __name__ == "__main__":
    pprint("Start your projects main entrypoint here")
