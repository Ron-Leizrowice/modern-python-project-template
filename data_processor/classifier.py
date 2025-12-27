import asyncio

import polars as pl
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelSettings

from data_processor.data import DATASET
from data_processor.paths import ENV_PATH
from data_processor.utils import confusion_from_pairs, precision_recall_f1

if not load_dotenv(ENV_PATH):
    raise FileNotFoundError(f"Could not load .env file at {ENV_PATH}")

CLASSIFIER_AGENT = Agent(
    model="openai:gpt-5-mini",
    system_prompt="""
Decide if the sample of text contains mention of a real-world disruption.

Return: True if it does, false otherwise.
""".strip(),
    model_settings=ModelSettings(temperature=0.0, seed=42),
)


async def _classify_sample(text: str) -> bool:
    result = await CLASSIFIER_AGENT.run(
        user_prompt=text,
        output_type=bool,
    )
    return result.output


async def measure_classification_accuracy(data: pl.DataFrame) -> pl.DataFrame:
    """Classify each entry and return a dataframe with predictions plus metrics printed."""
    tasks: list[asyncio.Task[bool]] = []

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_classify_sample(entry.text)) for entry in DATASET]

    pred_col = [t.result() for t in tasks]
    actual_col = [entry.tag == "disruption" for entry in DATASET]
    conf = confusion_from_pairs(list(zip(pred_col, actual_col, strict=True)))
    precision, recall, f1 = precision_recall_f1(conf)

    print(f"TP={conf.tp} FP={conf.fp} FN={conf.fn} TN={conf.tn}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")

    return data.with_columns(
        pl.Series("predicted_disruption", pred_col),
        pl.Series("actual_disruption", actual_col),
    )
