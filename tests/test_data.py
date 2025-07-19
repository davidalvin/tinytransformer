import torch
import numpy as np
from tinytransformer.data.data import get_batch, load_memmap
from tinytransformer.config.config import VOCAB_SIZE  # <-- Import vocab size

def test_get_batch():
    batch_size = 4
    context_length = 16
    device = "cpu"

    tokens = load_memmap()
    assert len(tokens) > 0
    assert tokens.dtype in (np.uint8, np.int16, np.uint16, np.int32, np.int64)

    X, Y = get_batch("train", batch_size, context_length, device=device)

    assert X.shape == (batch_size, context_length), f"X shape wrong: {X.shape}"
    assert Y.shape == (batch_size, context_length), f"Y shape wrong: {Y.shape}"

    assert X.dtype == torch.long, f"X dtype wrong: {X.dtype}"
    assert Y.dtype == torch.long, f"Y dtype wrong: {Y.dtype}"

    assert torch.all((X >= 0) & (X < VOCAB_SIZE)), f"X has out-of-range token IDs (max={X.max()})"
    assert torch.all((Y >= 0) & (Y < VOCAB_SIZE)), f"Y has out-of-range token IDs (max={Y.max()})"
