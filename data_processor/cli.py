"""Main package entrypoint for executing scripts."""

import asyncio

from dotenv import load_dotenv

from data_processor.classifier import measure_classification_accuracy
from data_processor.data import raw_data
from data_processor.extractor import measure_extraction_accuracy
from data_processor.paths import ENV_PATH

if not load_dotenv(ENV_PATH):
    raise FileNotFoundError(f"Could not load .env file at {ENV_PATH}")


def benchmark_classification() -> None:
    """Main entrypoint function."""
    asyncio.run(measure_classification_accuracy(raw_data))


def benchmark_extraction() -> None:
    """Main entrypoint function."""
    asyncio.run(measure_extraction_accuracy(raw_data))


if __name__ == "__main__":
    benchmark_classification()
