"""
Entry‑point that trains and saves the model.
The default `get_batch_fn` is the real data loader, so CLI runs work
out of the box, while tests can still inject a dummy function.
"""

import torch
from tinytransformer.data.data import get_batch          # ← explicit import
from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.training.train import train
from tinytransformer.config.config import (
    NUM_STEPS,
    BATCH_SIZE,
    BLOCK_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
)


def main(
    *,
    num_steps: int = NUM_STEPS,
    batch_size: int = BATCH_SIZE,
    context_length: int = BLOCK_SIZE,
    lr: float = LEARNING_RATE,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    get_batch_fn=get_batch,                    # passes real get_batch - alterantive is to pass a dummy get_batch allows testing
    model_path: str = MODEL_PATH,
):
    print("🚀 Starting training …")

    model = TinyTransformerLM()

    trained = train(
        model,
        num_steps=num_steps,
        batch_size=batch_size,
        context_length=context_length,
        lr=lr,
        device=device,
        get_batch_fn=get_batch_fn,             # always a callable now
    )

    print("💾 Saving model …")
    torch.save(trained.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    main()
