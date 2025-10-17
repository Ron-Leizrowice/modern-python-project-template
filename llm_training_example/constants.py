import random
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

BASE_TOKENIZER_PATH = CACHE_DIR / "base_tokenizer"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
