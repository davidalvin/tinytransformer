import os
import sys
from transformers import PreTrainedTokenizerFast
from tinytransformer.config import config as C


# 🔹 NEW: hard fail early if the resolved path isn't accessible
if not C.TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer file missing: {C.TOKENIZER_PATH}\n"
                            f"Resolved from config module located at {C.__file__}\n"
                            f"sys.executable={sys.executable}\n"
                            f"cwd={os.getcwd()}")
# Load trained tokenizer from file using transformers wrapper
_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(C.TOKENIZER_PATH))

def encode(text: str) -> list[int]:
    """Encode text into list of token IDs."""
    return _tokenizer.encode(text)

def decode(tokens: list[int]) -> str:
    """Decode list of token IDs back to string, removing special tokens and cleanup artifacts."""
    return _tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )