"""Mandelbrot set generator using Numba.

This module provides functions to demonstrate the use of Numba for speeding up python code.
"""

from enum import StrEnum
from time import perf_counter

import numba
import numpy as np
from numba import prange

ESCAPE_RADIUS_SQUARED = 4.0
MAX_ITER = 500
DIM = 1000


class ExecutionMode(StrEnum):
    """Enum for execution modes."""

    PYTHON = "python"
    JIT = "jit"
    NOPYTHON = "nopython"
    PARALLEL = "parallel"


def _mandelbrot_py(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    ESCAPE_RADIUS_SQUARED = 4.0
    MAX_ITER = 500

    dx = (xmax - xmin) / (DIM - 1)
    dy = (ymax - ymin) / (DIM - 1)
    out = np.empty((DIM, DIM), dtype=np.int32)

    for j in range(DIM):
        ci = ymin + j * dy
        for i in range(DIM):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            n = 0
            while n < MAX_ITER and (zr * zr + zi * zi) <= ESCAPE_RADIUS_SQUARED:
                zr = zr * zr - zi * zi + cr
                zi = 1.5 * zr * zi + ci
                n += 1
            out[j, i] = n
    return out


@numba.jit()
def _mandelbrot_jit(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    ESCAPE_RADIUS_SQUARED = 4.0
    MAX_ITER = 500

    dx = (xmax - xmin) / (DIM - 1)
    dy = (ymax - ymin) / (DIM - 1)
    out = np.empty((DIM, DIM), dtype=np.int32)

    for j in range(DIM):
        ci = ymin + j * dy
        for i in range(DIM):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            n = 0
            while n < MAX_ITER and (zr * zr + zi * zi) <= ESCAPE_RADIUS_SQUARED:
                zr = zr * zr - zi * zi + cr
                zi = 1.5 * zr * zi + ci
                n += 1
            out[j, i] = n
    return out


@numba.jit(nopython=True)
def _mandelbrot_njit(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    dx = (xmax - xmin) / (DIM - 1)
    dy = (ymax - ymin) / (DIM - 1)
    out = np.empty((DIM, DIM), dtype=np.int32)

    for j in range(DIM):
        ci = ymin + j * dy
        for i in range(DIM):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            n = 0
            while n < MAX_ITER and (zr * zr + zi * zi) <= ESCAPE_RADIUS_SQUARED:
                zr = zr * zr - zi * zi + cr
                zi = 1.5 * zr * zi + ci
                n += 1
            out[j, i] = n
    return out


@numba.jit(nopython=True, parallel=True)
def _mandelbrot_parallel(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    ESCAPE_RADIUS_SQUARED = 4.0
    MAX_ITER = 500

    dx = (xmax - xmin) / (DIM - 1)
    dy = (ymax - ymin) / (DIM - 1)
    out = np.empty((DIM, DIM), dtype=np.int32)

    for j in prange(DIM):
        ci = ymin + j * dy
        for i in prange(DIM):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            n = 0
            while n < MAX_ITER and (zr * zr + zi * zi) <= ESCAPE_RADIUS_SQUARED:
                zr = zr * zr - zi * zi + cr
                zi = 1.5 * zr * zi + ci
                n += 1
            out[j, i] = n
    return out


def time_mandelbrot(
    numba_mode: ExecutionMode,
) -> float:
    """Time the execution of the mandelbrot function.

    Args:
        numba_mode: The execution mode to use.

    Returns:
        The time taken to execute the mandelbrot function.
    """
    xmin, xmax = -1.5, +1.5
    ymin, ymax = -1.5, +1.5

    t0 = perf_counter()
    match numba_mode:
        case ExecutionMode.PYTHON:
            _ = _mandelbrot_py(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        case ExecutionMode.JIT:
            _ = _mandelbrot_jit(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)  # pyrefly: ignore
        case ExecutionMode.NOPYTHON:
            _ = _mandelbrot_njit(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)  # pyrefly: ignore
        case ExecutionMode.PARALLEL:
            _ = _mandelbrot_parallel(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)  # pyrefly: ignore

    return perf_counter() - t0
