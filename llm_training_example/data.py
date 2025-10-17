import functools
import hashlib
import json
import math
import os
from collections.abc import Generator
from pathlib import Path
from typing import cast

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import TextInput

from llm_training_example.constants import CACHE_DIR, SEED

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

_TOKENIZED_CACHE_DIR = CACHE_DIR / "tokenized_datasets"
_TOKENIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_data() -> DatasetDict:
    """Load and cache the textbooks dataset."""
    return cast("DatasetDict", load_dataset("open-phi/textbooks"))


def _cache_path(*, split_name: str, tokenizer: PreTrainedTokenizerFast, fingerprint: str) -> Path:
    cache_key = {
        "split": split_name,
        "seq_len": tokenizer.model_max_length,
        "fingerprint": fingerprint,
        "tokenizer": tokenizer.vocab_size,
    }
    digest = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()
    return _TOKENIZED_CACHE_DIR / f"{split_name}_{tokenizer.model_max_length}_{digest}.pt"


def _batched(seq: list[TextInput], size: int) -> Generator[list[TextInput]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _tokenize_and_chunk_texts(
    texts: list[str],
    *,
    tokenizer: PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,
) -> TensorDataset:
    seq_len = tokenizer.model_max_length

    encoded_data: list[torch.Tensor] = []
    total_batches = math.ceil(len(texts) / batch_size) if texts else 0

    for batch in tqdm(  # type: ignore[not-iterable]
        _batched(texts, batch_size),
        total=total_batches,
        desc=f"Tokenizing {split_name}",
        unit="batch",
    ):
        # encode the batch of strings
        encodings: BatchEncoding = tokenizer.batch_encode_plus(
            batch, padding=False, truncation=False, add_special_tokens=True
        )

        # process each encoded item in the batch
        tokenized_input_ids = cast("list[list[int]]", encodings["input_ids"])
        for input_ids in tokenized_input_ids:
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            if len(input_ids_tensor) < seq_len:
                continue  # skip short sequences
            # split into max_length chunks
            input_ids_batch = torch.split(input_ids_tensor, seq_len)

            for sub_input_ids in input_ids_batch:
                if len(sub_input_ids) < seq_len:
                    continue
                if len(sub_input_ids) == seq_len:
                    encoded_data.append(sub_input_ids)
    dataset = torch.stack(encoded_data)
    return TensorDataset(dataset)


def _load_data(
    split: str,
    *,
    tokenizer: PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int = 64,
) -> TensorDataset:
    data_split = get_data()[split]
    fingerprint = getattr(data_split, "_fingerprint", split_name)

    cache_path = _cache_path(
        split_name=split_name,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
    )

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert isinstance(data, TensorDataset)
        return data

    texts = cast("list[str]", data_split["markdown"])
    data = _tokenize_and_chunk_texts(
        texts,
        tokenizer=tokenizer,
        split_name=split_name,
        batch_size=batch_size,
    )

    torch.save(data, cache_path)
    return data


def prepare_data(
    tokenizer: PreTrainedTokenizerFast,
    *,
    batch_size: int,
) -> DataLoader:
    """Prepare the training data loader."""
    train_dataset = _load_data(
        split="train",
        tokenizer=tokenizer,
        split_name="train",
    )
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
        generator=g,
    )
