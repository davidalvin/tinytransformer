import torch
from tinytransformer.models.factory import build_model
from tinytransformer.config import config as C


def test_generate_basic():
    model = build_model("tiny")
    model.eval()

    try:
        model.load_state_dict(torch.load(C.MODEL_PATH, map_location="cpu"))
    except FileNotFoundError:
        pass
    except RuntimeError:
        pass

    prompt = "Once"
    output = model.generate(prompt, max_tokens=32, device="cpu")

    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should not be empty"
    assert any(c.isalpha() for c in output), "Output looks malformed (no letters?)"

def test_generate_exact_context_length():
    model = build_model("tiny")
    prompt = "a " * C.BLOCK_SIZE  # repeat token-like input
    output = model.generate(prompt, max_tokens=5, context_length=C.BLOCK_SIZE, device="cpu")
    assert isinstance(output, str)