import json
from datetime import datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, field_validator, model_validator

from data_processor.paths import DATA_PATH

raw_data = pl.read_csv(DATA_PATH)


class DisruptionInfo(BaseModel):
    duration_days: int
    entities: list[str]
    date: datetime
    site: str


class NewsEntry(BaseModel):
    text: str
    tag: Literal["disruption", "clean"]
    info: DisruptionInfo | None

    @field_validator("info", mode="before")
    @classmethod
    def _parse_info(cls, v: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @model_validator(mode="after")
    def _validate_info(self) -> "NewsEntry":
        if self.tag == "disruption" and self.info is None:
            raise ValueError("info must be provided for disruption entries")
        if self.tag == "clean" and self.info is not None:
            raise ValueError("info must be None for clean entries")
        return self


def pre_process_data(data: pl.DataFrame) -> list[NewsEntry]:
    """Pre-process the raw data into validated NewsEntry instances."""
    entries: list[NewsEntry] = []
    for row in data.iter_rows(named=True):
        entry = NewsEntry.model_validate(row)
        entries.append(entry)
    return entries


DATASET = pre_process_data(raw_data)
