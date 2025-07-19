"""
data/save_tokens.py
-------------------
Stream-encode TinyStories with your BPE tokenizer and write a **proper
`.npy` file that can be memory-mapped** later with
`np.load(..., mmap_mode="r")`.

Key change ⬇️
=============
Use **numpy.lib.format.open_memmap** instead of bare `np.memmap`.
`open_memmap` writes the NumPy header (`\x93NUMPY …`) *and* returns a
mem-mappable array you can fill in chunks.
"""

import numpy as np
from numpy.lib.format import open_memmap
from tinytransformer.models.tokenizer import encode
from tinytransformer.config.config import DATA_PATH, TOKENS_PATH


def count_lines(path: str) -> int:
    """Fast line-count helper."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def save_tokenized_file_chunked(
    source: str = DATA_PATH,
    out_path: str = TOKENS_PATH,
    dtype: np.dtype = np.uint16,
):
    # ---------- 1st pass: count tokens ----------
    print("🔍 Counting stories …")
    total_lines = count_lines(source)
    print(f"🧠 Found {total_lines} lines. Counting tokens …")

    total_tokens = 0
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            total_tokens += len(encode(line.strip()))

    print(f"🧮 Total tokens = {total_tokens:,}")

    # ---------- allocate header-correct memmap ----------
    print(f"💾 Creating npy-compatible memmap at {out_path} …")
    arr = open_memmap(
        out_path,
        mode="w+",
        dtype=dtype,
        shape=(total_tokens,),
    )

    # ---------- 2nd pass: stream-encode & fill ----------
    idx = 0
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            tok = encode(line.strip())
            arr[idx : idx + len(tok)] = tok          # write slice
            idx += len(tok)

    arr.flush()
    print(f"✅ Saved {total_tokens:,} tokens to {out_path}")


if __name__ == "__main__":
    save_tokenized_file_chunked()
