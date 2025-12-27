import asyncio

import polars as pl
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelSettings

from data_processor.data import DATASET, DisruptionInfo
from data_processor.paths import ENV_PATH

if not load_dotenv(ENV_PATH):
    raise FileNotFoundError(f"Could not load .env file at {ENV_PATH}")

EXTRACTOR_AGENT = Agent(
    model="openai:gpt-5-mini",
    system_prompt="""
Extract structured information about disruptions from news text.

Return a JSON object with the following fields:
- duration_days: integer, number of days the disruption lasts
- entities: list of strings, names of affected entities
- date: string, date of the disruption in ISO 8601 format (YYYY-MM-DD)
- site: string, location of the disruption
""".strip(),
    model_settings=ModelSettings(temperature=0.0, seed=42),
)


async def extract_disruption_info(text: str) -> DisruptionInfo:
    """Extract disruption information from the given text."""
    result = await EXTRACTOR_AGENT.run(
        user_prompt=text,
        output_type=DisruptionInfo,
    )
    return result.output


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_entities(entities: list[str]) -> set[str]:
    return {_normalize_text(entity) for entity in entities}


async def measure_extraction_accuracy(data: pl.DataFrame) -> pl.DataFrame:
    """Extract disruption info for tagged entries and print field-level accuracy."""
    disruption_entries = [entry for entry in DATASET if entry.tag == "disruption"]
    tasks: list[asyncio.Task[DisruptionInfo]] = []

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(extract_disruption_info(entry.text)) for entry in disruption_entries]

    predicted = [t.result() for t in tasks]
    actual = [entry.info for entry in disruption_entries]

    duration_match = [pred.duration_days == truth.duration_days for pred, truth in zip(predicted, actual, strict=True)]
    entities_match = [
        _normalize_entities(pred.entities) == _normalize_entities(truth.entities)
        for pred, truth in zip(predicted, actual, strict=True)
    ]
    date_match = [pred.date.date() == truth.date.date() for pred, truth in zip(predicted, actual, strict=True)]
    site_match = [
        _normalize_text(pred.site) == _normalize_text(truth.site) for pred, truth in zip(predicted, actual, strict=True)
    ]
    all_match = [
        dm and em and dtm and sm
        for dm, em, dtm, sm in zip(duration_match, entities_match, date_match, site_match, strict=True)
    ]

    total = len(disruption_entries)
    duration_accuracy = sum(duration_match) / total if total else 0.0
    entities_accuracy = sum(entities_match) / total if total else 0.0
    date_accuracy = sum(date_match) / total if total else 0.0
    site_accuracy = sum(site_match) / total if total else 0.0
    all_accuracy = sum(all_match) / total if total else 0.0

    print(f"Samples: {total}")
    print(f"Duration accuracy: {duration_accuracy:.2%}")
    print(f"Entities accuracy: {entities_accuracy:.2%}")
    print(f"Date accuracy: {date_accuracy:.2%}")
    print(f"Site accuracy: {site_accuracy:.2%}")
    print(f"All-fields accuracy: {all_accuracy:.2%}")

    disruption_data = data.filter(pl.col("tag") == "disruption")
    pred_date = [pred.date.date() for pred in predicted]
    actual_date = [truth.date.date() for truth in actual]
    return disruption_data.with_columns(
        pl.Series("pred_duration_days", [pred.duration_days for pred in predicted]),
        pl.Series("actual_duration_days", [truth.duration_days for truth in actual]),
        pl.Series("pred_entities", [pred.entities for pred in predicted]),
        pl.Series("actual_entities", [truth.entities for truth in actual]),
        pl.Series("pred_date", pred_date),
        pl.Series("actual_date", actual_date),
        pl.Series("pred_site", [pred.site for pred in predicted]),
        pl.Series("actual_site", [truth.site for truth in actual]),
        pl.Series("match_duration", duration_match),
        pl.Series("match_entities", entities_match),
        pl.Series("match_date", date_match),
        pl.Series("match_site", site_match),
        pl.Series("match_all", all_match),
    )
