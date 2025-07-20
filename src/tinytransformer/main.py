"""
Entry‑point that trains and saves the model.
Supports CLI overrides (e.g., --lr, --num_layers, etc).
"""

import torch
from tinytransformer.config.cli import parse_and_apply
parse_and_apply()  # ← apply CLI overrides before loading config

from tinytransformer.config import config as C
from tinytransformer.data.data import get_batch
from tinytransformer.models.factory import build_model
from tinytransformer.training.train import train


def main(
    *,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    get_batch_fn = get_batch,
):
    print("🚀 Starting training …")

    model = build_model(
        "tiny",  # may make this cli at later stage
        vocab_size=C.VOCAB_SIZE,
        d_model=C.D_MODEL,
        nhead=C.N_HEAD,
        num_layers=C.NUM_LAYERS,
        max_seq_len=C.MAX_SEQ_LEN,
    )

    trained = train(
        model,
        num_steps=C.NUM_STEPS,
        batch_size=C.BATCH_SIZE,
        context_length=C.BLOCK_SIZE,
        lr=C.LEARNING_RATE,
        device=device,
        get_batch_fn=get_batch_fn,
    )

    print("💾 Saving model …")
    torch.save(trained.state_dict(), C.MODEL_PATH)
    print(f"✅ Model saved to {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
