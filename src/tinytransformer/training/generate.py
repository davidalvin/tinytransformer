import torch

from tinytransformer.models.transformer import TinyTransformerLM                # fixed: use correct module name
from tinytransformer.models.tokenizer import encode, decode               # updated import
from tinytransformer.config.config import BLOCK_SIZE, TOKENIZER_PATH, MODEL_PATH      # added MODEL_PATH

@torch.no_grad()
def generate(model, prompt, max_tokens=100, context_length=BLOCK_SIZE, device="cpu"):
    model.eval()
    model.to(device)

    # Encode initial prompt to token IDs
    tokens = encode(prompt)
    tokens = tokens[-context_length:]  # truncate if longer than context window
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_tokens):
        # Crop to context length
        x_cond = x[:, -context_length:]  # (1, T)

        logits = model(x_cond)  # (1, T, vocab_size)
        next_logits = logits[0, -1]  # last token's logits: (vocab_size,)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # sample

        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)  # append to sequence

    return decode(x[0].tolist())

# 🔽 Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformerLM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # <-- use config

    print("�� TinyTransformer is ready. Press Ctrl+C to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            response = generate(model, user_input, max_tokens=100, device=device)
            print("LM:", response)
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
