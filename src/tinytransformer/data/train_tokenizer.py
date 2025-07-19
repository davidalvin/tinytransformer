from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel           # splits on bytes
from tokenizers.decoders       import ByteLevel as DecBL  # ← NEW
from tokenizers.processors     import ByteLevel as ProcBL # ← NEW

from tinytransformer.config.config import VOCAB_SIZE, SPECIAL_TOKENS, DATA_PATH, TOKENIZER_PATH

print("Starting tokenizer...")

# ─── 1. Build & train ───────────────────────────────────────────────────────────
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size      = VOCAB_SIZE,
    special_tokens  = SPECIAL_TOKENS,
)

print(f"Training tokenizer on data at {DATA_PATH}...")
tokenizer.train([str(DATA_PATH)], trainer)

# ─── 2. Add decoder + post-processor for clean spaces ───────────────────────────
tokenizer.decoder        = DecBL()                   # byte-level decoder
tokenizer.post_processor = ProcBL(trim_offsets=True) # strips the “Ġ” artifacts

# ─── 3. Save ────────────────────────────────────────────────────────────────────
tokenizer.save(str(TOKENIZER_PATH))
print(f"✅ Saved tokenizer (with clean decode) to {TOKENIZER_PATH}")
