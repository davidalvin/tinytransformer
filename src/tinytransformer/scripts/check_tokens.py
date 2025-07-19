# scripts/check_tokens_npy.py
import numpy as np
from tinytransformer.config.config import TOKENS_PATH

print(f"🔍 Checking file: {TOKENS_PATH}")

try:
    tokens = np.load(TOKENS_PATH, mmap_mode="r")
except ValueError as e:
    print("❌ Failed to load with mmap_mode='r'")
    print("Reason:", e)
    print("Trying again with allow_pickle=True...")

    tokens = np.load(TOKENS_PATH, allow_pickle=True)
    print("📎 dtype:", tokens.dtype)
    print("📏 shape:", tokens.shape)
    print("🧪 example:", tokens[0])
    print("✅ Success (but file is pickled — this is the problem!)")
    exit()

print("✅ File is safe for memory-mapping")
print("📎 dtype:", tokens.dtype)
print("📏 shape:", tokens.shape)
