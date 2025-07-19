import numpy as np
import os
from tinytransformer.config.config import TOKENS_PATH, VOCAB_SIZE

def test_tokens_npy():
    assert os.path.exists(TOKENS_PATH), f"❌ File not found: {TOKENS_PATH}"

    try:
        tokens = np.load(TOKENS_PATH, mmap_mode="r")
    except ValueError as e:
        raise AssertionError(
            f"❌ Failed to memory-map tokens.npy. Probably saved with pickled data.\n{e}"
        )

    assert isinstance(tokens, np.memmap), "❌ tokens.npy is not memory-mapped"
    assert tokens.dtype in (np.uint16, np.int32, np.int64), f"❌ Unexpected dtype: {tokens.dtype}"
    assert tokens.ndim == 1, f"❌ Expected 1D array, got shape {tokens.shape}"
    assert len(tokens) > 0, "❌ tokens.npy is empty"
    assert tokens.max() < VOCAB_SIZE, f"❌ Found token ID >= VOCAB_SIZE ({VOCAB_SIZE})"
    assert tokens.min() >= 0, "❌ Found negative token ID"

    print("✅ tokens.npy passed")
