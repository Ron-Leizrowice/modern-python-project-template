"""Asynchronous JSON loading benchmark across multiple parsers."""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import aiofiles
import msgspec
import orjson
from msgspec import Struct
from pydantic import BaseModel, TypeAdapter

DATASET_FILENAME_PREFIX = "json_loading_demo_dataset"
RNG_SEED = 42
REPEATS = 3


class JsonLoader(StrEnum):
    JSON = "json"
    ORJSON = "orjson"


class TypedDecoder(StrEnum):
    PYDANTIC = "pydantic"
    MSGSPEC = "msgspec"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for the synthetic dataset written to a temporary file."""

    rows: int
    tag_count: int

    def filename(self) -> str:
        """Generate the filename for the given dataset configuration."""
        return f"{DATASET_FILENAME_PREFIX}_rows{self.rows}_tags{self.tag_count}.json"


DATASET_CONFIGS: tuple[DatasetConfig, ...] = (
    DatasetConfig(rows=1_000, tag_count=8),
    DatasetConfig(rows=10_000, tag_count=8),
    DatasetConfig(rows=100_000, tag_count=8),
    DatasetConfig(rows=1_000_000, tag_count=8),
)


class PydanticMetrics(BaseModel):
    count: int
    ratio: float
    flag: bool


class PydanticRecord(BaseModel):
    id: int
    name: str
    value: float
    active: bool
    tags: list[str]
    metrics: PydanticMetrics


class MsgspecMetrics(Struct):
    count: int
    ratio: float
    flag: bool


class MsgspecRecord(Struct):
    id: int
    name: str
    value: float
    active: bool
    tags: list[str]
    metrics: MsgspecMetrics


PYDANTIC_ADAPTER = TypeAdapter(list[PydanticRecord])
MSGSPEC_JSON_DECODER = msgspec.json.Decoder(type=list[MsgspecRecord])


async def _ensure_dataset(
    config: DatasetConfig,
) -> Path:
    """Create a dataset on disk if it does not already exist for this config."""
    path = Path(tempfile.gettempdir()) / config.filename()

    if Path(path).exists():
        return path

    Path(path.parent).mkdir(parents=True, exist_ok=True)

    # Derive a deterministic RNG seed based on the config so each file differs.
    seed = RNG_SEED ^ config.rows ^ (config.tag_count << 16)
    rng = random.Random(seed)

    records = [
        {
            "id": idx,
            "name": f"item-{idx}",
            "value": rng.random(),
            "active": rng.choice([True, False]),
            "tags": [f"tag-{rng.randint(0, 1_000)}" for _ in range(config.tag_count)],
            "metrics": {
                "count": rng.randint(0, 10_000),
                "ratio": rng.random(),
                "flag": rng.choice([True, False]),
            },
        }
        for idx in range(config.rows)
    ]

    payload = json.dumps(records, separators=(",", ":"))
    async with aiofiles.open(path, "w") as f:
        await f.write(payload)

    return path


async def _read_dataset_bytes(config: DatasetConfig) -> bytes:
    """Ensure a dataset exists and return its raw bytes from disk."""
    path = await _ensure_dataset(config)
    async with aiofiles.open(path, "rb") as file:
        return await file.read()


async def time_json_loading(
    backend: JsonLoader,
    *,
    config: DatasetConfig,
) -> float:
    """Measure the average parse time for the dataset with the requested backend."""
    payload = await _read_dataset_bytes(config)

    elapsed = 0.0
    for _ in range(REPEATS):
        t0 = perf_counter()
        match backend:
            case JsonLoader.JSON:
                _ = json.loads(payload.decode("utf-8"))
            case JsonLoader.ORJSON:
                _ = orjson.loads(payload)
        elapsed += perf_counter() - t0

    return elapsed / REPEATS


async def time_typed_decoding(
    decoder: TypedDecoder,
    *,
    config: DatasetConfig,
) -> float:
    """Measure the average parse time using orjson and typed decoders."""
    payload = await _read_dataset_bytes(config)

    elapsed = 0.0
    for _ in range(REPEATS):
        t0 = perf_counter()
        match decoder:
            case TypedDecoder.PYDANTIC:
                raw = orjson.loads(payload)
                _ = PYDANTIC_ADAPTER.validate_python(raw)
            case TypedDecoder.MSGSPEC:
                _ = MSGSPEC_JSON_DECODER.decode(payload)
        elapsed += perf_counter() - t0

    return elapsed / REPEATS


async def benchmark_json_loading() -> dict[DatasetConfig, dict[JsonLoader, float]]:
    """Benchmark all requested configs/backends and return their average timings."""

    results: dict[DatasetConfig, dict[JsonLoader, float]] = {}

    for config in DATASET_CONFIGS:
        config_results: dict[JsonLoader, float] = {}
        for backend in JsonLoader:
            config_results[backend] = await time_json_loading(
                backend,
                config=config,
            )
        results[config] = config_results

    return results


async def benchmark_typed_decoding() -> dict[DatasetConfig, dict[TypedDecoder, float]]:
    """Benchmark typed decoding using orjson with pydantic and msgspec."""

    results: dict[DatasetConfig, dict[TypedDecoder, float]] = {}

    for config in DATASET_CONFIGS:
        config_results: dict[TypedDecoder, float] = {}
        for decoder in TypedDecoder:
            config_results[decoder] = await time_typed_decoding(
                decoder,
                config=config,
            )
        results[config] = config_results

    return results
