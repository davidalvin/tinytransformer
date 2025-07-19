import torch

from tinytransformer.models.transformer import TinyTransformerLM  # fixed: use correct module name
from tinytransformer.training.train import train
from tinytransformer.config.config import NUM_STEPS, BATCH_SIZE, BLOCK_SIZE, LEARNING_RATE, MODEL_PATH  # added MODEL_PATH

def main():
    print("🚀 Starting training...")
    model = TinyTransformerLM()

    trained_model = train(
        model,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        context_length=BLOCK_SIZE,
        lr=LEARNING_RATE,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Saving model")
    # save model weights
    torch.save(trained_model.state_dict(), MODEL_PATH)  # <-- use config
    print(f"✅ model saved to {MODEL_PATH}")  # <-- use config


if __name__ == "__main__":
    main()
