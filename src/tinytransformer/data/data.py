import torch
import numpy as np
from tinytransformer.config.config import TOKENS_PATH, TRAIN_SPLIT_FRACTION

# Global cache for memory-mapped token array
_cached_tokens = None

# -----------------------------------------------
# Load memmapped token array (read-only, lazy)
# -----------------------------------------------
def load_memmap():
    global _cached_tokens
    if _cached_tokens is None:
        _cached_tokens = np.load(TOKENS_PATH, mmap_mode="r")
    return _cached_tokens

# -----------------------------------------------
# Return a random batch (X, Y) of shape (B, T)
# - split: "train" or "val"
# - device: "cpu", "cuda", or "mps"
# -----------------------------------------------
def get_batch(split, batch_size, context_length, device="cpu"):
    tokens = load_memmap()
    split_idx = int(len(tokens) * TRAIN_SPLIT_FRACTION)

    if split == "train":
        data = tokens[:split_idx]
    elif split == "val":
        data = tokens[split_idx:]
    else:
        raise ValueError(f"Invalid split: {split}")

    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    X = np.stack([data[i : i + context_length] for i in starts])
    Y = np.stack([data[i + 1 : i + 1 + context_length] for i in starts])

    return torch.from_numpy(X).to(device).long(), torch.from_numpy(Y).to(device).long()
