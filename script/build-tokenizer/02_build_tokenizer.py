"""
02 - Build the Bilingual BPE Tokenizer (Arabic + English)
=========================================================
"""

from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cleaning import build_normalizer

# =========================================================
# CONFIGURATION
# =========================================================

CURRENT_WD = Path(__file__).parent
CLEAN_PATHS = [
    CURRENT_WD / "output" / "phase1_train_clean.parquet",
    CURRENT_WD / "output" / "phase2_train_clean.parquet",
]

OUTPUT_PATH = CURRENT_WD / "tokenizer.json"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                  "[AR]", "[EN]",
                  "[SYS]", "[USER]", "[ASST]"]