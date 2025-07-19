# config/config.py
from pathlib import Path

BASE_DIR = Path("src/tinytransformer")

# ------------------------------------------------------------------
# Raw / processed data now live under repo‑root/data_external/
# ------------------------------------------------------------------
DATA_ROOT = Path("data_external/tinytransformer")
HF_CACHE = Path("data_external/hf_tinystories")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_STORIES = 100_000
JSONL_PATH = DATA_ROOT / "tinystories_raw.jsonl"
TOKENS_PATH = DATA_ROOT / "tokens.npy"
TXT_PATH    = DATA_ROOT / "tinystories.txt"   # alias
DATA_PATH   = TXT_PATH

# Tokenizer
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"
VOCAB_SIZE = 2048
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "<|endoftext|>"]

# Model / training
BLOCK_SIZE    = 128
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 10
NUM_STEPS     = 1_000
CHECKPOINT_PATH = "checkpoints/model.pt"
MODEL_PATH      = "tiny_transformer.pt"

# Architecture
D_MODEL   = 128
N_HEAD    = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 1024
TRAIN_SPLIT_FRACTION = 0.9
