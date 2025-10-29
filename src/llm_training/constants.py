import random
from pathlib import Path

import numpy as np
import torch

LIB_DIR = Path(__file__).parent

CACHE_DIR = LIB_DIR / ".cache"
WANDB_LOGGING_DIR = LIB_DIR / ".wandb"

CACHE_DIR.mkdir(exist_ok=True)
WANDB_LOGGING_DIR.mkdir(exist_ok=True)

BASE_TOKENIZER_PATH = LIB_DIR / "base_tokenizer"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device("cuda")
