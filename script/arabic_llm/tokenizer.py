"""
tokenizer.py -- loads the bilingual BPE tokenizer trained before in 02
"""

from pathlib import Path
from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"

class Tokenizer:

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    SEP = 4
    AR = 5
    EN = 6
    SYS = 7
    USER = 8
    ASST = 9

    _SPECIAL_TOKENS = {  # noqa: RUF012
        "[PAD]": PAD, "[UNK]": UNK, "[BOS]": BOS, "[EOS]": EOS, "[SEP]": SEP,
        "[AR]": AR, "[EN]": EN, "[SYS]": SYS, "[USER]": USER, "[ASST]": ASST,
    }
def __init__(self, backing_tokenizer: BackingTokenizer):
    self._tokenizer = backing_tokenizer

@classmethod
def from_file(cls, path=DEFAULT_TOKENIZER_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {path}."
        )

    return cls(
        BackingTokenizer.from_file( str(path) )
    )
