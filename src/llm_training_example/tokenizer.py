import os
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llm_training_example.constants import BASE_TOKENIZER_PATH, CACHE_DIR
from llm_training_example.data import get_data

os.environ["TOKENIZERS_PARALLELISM"] = "1"


def _get_training_corpus() -> list[str]:
    raw_text_path = CACHE_DIR / "raw_text.md"
    if raw_text_path.exists():
        return raw_text_path.read_text(encoding="utf-8").splitlines()
    data: list[str] = get_data().unique("markdown")["train"]
    with raw_text_path.open("w", encoding="utf-8") as f:
        f.writelines(item.strip() + "\n" for item in data)
    return data


def _train_tokenizer(vocab_size: int, seq_len: int, path: Path) -> PreTrainedTokenizerFast:
    corpus = _get_training_corpus()

    base_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH, use_fast=True)

    tokenizer: PreTrainedTokenizerFast = base_tokenizer.train_new_from_iterator(corpus, vocab_size=vocab_size)
    tokenizer.model_max_length = seq_len

    tokenizer.save_pretrained(path)

    return tokenizer


def get_tokenizer(seq_len: int, vocab_size: int) -> PreTrainedTokenizerFast:
    """Load or train a tokenizer with the desired vocab_size."""
    tokenizer_path = CACHE_DIR / f"tokenizer_{vocab_size}"

    if tokenizer_path.exists():
        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    return _train_tokenizer(vocab_size=vocab_size, seq_len=seq_len, path=tokenizer_path)
