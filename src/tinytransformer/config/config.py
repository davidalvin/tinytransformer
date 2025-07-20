from pathlib import Path

# ------------------------------------------------------------------ #
# Resolve paths relative to repo root                                #
# ------------------------------------------------------------------ #
# If this file is at: src/tinytransformer/config/config.py
# Then we want:
_THIS_DIR    = Path(__file__).resolve().parent            # → config/
PROJECT_ROOT = _THIS_DIR.parent                           # → tinytransformer/
REPO_ROOT    = PROJECT_ROOT.parent.parent                 # → your repo root (one above src/)


# ------------------------------------------------------------------ #
# Data paths                                                         #
# ------------------------------------------------------------------ #
DATA_ROOT = REPO_ROOT / "data_external" / "tinytransformer"
HF_CACHE  = REPO_ROOT / "data_external" / "hf_tinystories"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True,  exist_ok=True)

NUM_DATA_EXAMPLES = None                       # full dataset if None
JSONL_PATH  = DATA_ROOT / "tinystories_raw.jsonl"
TXT_PATH    = DATA_ROOT / "tinystories.txt"
TOKENS_PATH = DATA_ROOT / "tokens.npy"
DATA_PATH   = TXT_PATH                         # alias

# ------------------------------------------------------------------ #
# Tokenizer                                                          #
# ------------------------------------------------------------------ #
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"
VOCAB_SIZE     = 2048
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "<|endoftext|>"]

# ------------------------------------------------------------------ #
# Training & model                                                   #
# ------------------------------------------------------------------ #
BLOCK_SIZE    = 128
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS    = 10
NUM_STEPS     = 1_000

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "model.pt"
MODEL_PATH      = REPO_ROOT / "checkpoints" / "tiny_transformer.pt"

# ------------------------------------------------------------------ #
# Architecture                                                       #
# ------------------------------------------------------------------ #
D_MODEL     = 128
N_HEAD      = 4
NUM_LAYERS  = 2
MAX_SEQ_LEN = 1024
TRAIN_SPLIT_FRACTION = 0.9

# ------------------------------------------------------------------ #
# Decoding knobs (default)                                           #
# ------------------------------------------------------------------ #
TEMPERATURE = 1.0
TOP_P       = None        # nucleus disabled by default
