# config/config.py

BASE_DIR = "src/tinytransformer"

# Tokenizer
VOCAB_SIZE = 2048
TOKENIZER_PATH = f"{BASE_DIR}/models/tokenizer.json"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "<|endoftext|>"]

# Dataset
NUM_TRAIN_STORIES = 100000
JSONL_PATH = f"{BASE_DIR}/input/tinystories_raw.jsonl"
TOKENS_PATH = f"{BASE_DIR}/input/tokens.npy"
DATA_PATH = f"{BASE_DIR}/input/tinystories.txt"
TXT_PATH = f"{BASE_DIR}/input/tinystories.txt"
TRAIN_SPLIT_FRACTION = 0.9

# Model / training
BLOCK_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
CHECKPOINT_PATH = "checkpoints/model.pt"
MODEL_PATH = "tiny_transformer.pt"
NUM_STEPS = 1000

# Architecture
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 1024
