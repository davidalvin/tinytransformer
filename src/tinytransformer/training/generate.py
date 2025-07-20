# generate.py

import torch
from tinytransformer.config.cli import parse_and_apply
parse_and_apply()  # Apply CLI overrides to config

from tinytransformer.config import config as C
from tinytransformer.models.factory import build_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the model architecture
    model = build_model("tiny")

    # Load trained weights into the model
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))

    print("✨ Model ready. Type your prompt (Ctrl+C to exit)\n")

    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            # Generate output using model's built-in generate method
            output = model.generate(
                prompt,
                max_tokens=100,
                context_length=C.BLOCK_SIZE,
                device=device,
                temperature=C.TEMPERATURE,
                top_p=C.TOP_P,
            )
            print("LM:", output)

    except KeyboardInterrupt:
        print("\n👋 Bye.")


if __name__ == "__main__":
    main()
