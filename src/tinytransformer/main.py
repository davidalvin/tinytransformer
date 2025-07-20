"""
Entry-point that trains and saves the model.
Supports CLI overrides (e.g., --lr, --num_layers, etc).
"""

import torch
from tinytransformer.config.cli import parse_and_apply
from tinytransformer.config import config as C
from tinytransformer.data.data import get_batch
from tinytransformer.models.factory import build_model
from tinytransformer.training.train import train


def run_training(
    *,
    num_steps: int = C.NUM_STEPS,
    batch_size: int = C.BATCH_SIZE,
    context_length: int = C.BLOCK_SIZE,
    lr: float = C.LEARNING_RATE,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    get_batch_fn = get_batch,
    model_path = C.MODEL_PATH,
):
    """Core training logic that can be reused in tests or scripts."""
    print("🚀 Starting training …")

    model = build_model(
        "tiny",
        vocab_size=C.VOCAB_SIZE,
        d_model=C.D_MODEL,
        nhead=C.N_HEAD,
        num_layers=C.NUM_LAYERS,
        max_seq_len=C.MAX_SEQ_LEN,
    )

    trained = train(
        model,
        num_steps=num_steps,
        batch_size=batch_size,
        context_length=context_length,
        lr=lr,
        device=device,
        get_batch_fn=get_batch_fn,
    )

    print("💾 Saving model …")
    torch.save(trained.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")


def main():
    """CLI entry point that uses config defaults and CLI overrides."""
    parse_and_apply()
    run_training()


if __name__ == "__main__":
    main()
