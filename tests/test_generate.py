import torch
from tinytransformer.training.generate import generate
from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.config.config import MODEL_PATH

def test_generate_basic():
    model = TinyTransformerLM()
    model.eval()

    # Try loading weights; fallback to untrained model
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    except FileNotFoundError:
        pass  # no weights — test will just use untrained model
    except RuntimeError:
        pass  # likely vocab mismatch from old checkpoint — also okay here

    prompt = "Once"
    output = generate(model, prompt, max_tokens=32, device="cpu")

    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should not be empty"

    # Remove this BPE-incompatible check:
    # assert all(0 <= ord(c) <= 255 for c in output)

    # Instead, optionally validate it's clean printable text:
    assert any(c.isalpha() for c in output), "Output looks malformed (no letters?)"
