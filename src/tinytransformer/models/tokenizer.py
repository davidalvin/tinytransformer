from transformers import PreTrainedTokenizerFast
from tinytransformer.config.config import TOKENIZER_PATH

# Load trained tokenizer from file using transformers wrapper
_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

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