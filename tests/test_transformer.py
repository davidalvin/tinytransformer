import pytest
import torch
from tinytransformer.models.transformer import TinyTransformerLM
from tinytransformer.data.data import get_batch
from tinytransformer.config.config import VOCAB_SIZE


def test_model_forward():
    model = TinyTransformerLM(vocab_size=VOCAB_SIZE)
    B, T = 4, 32
    x = torch.randint(0, VOCAB_SIZE, (B, T))  # random valid token IDs
    logits = model(x)

    assert logits.shape == (B, T, VOCAB_SIZE), f"Expected {(B, T, VOCAB_SIZE)}, got {logits.shape}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_forward_pass(device):
    model = TinyTransformerLM(vocab_size=VOCAB_SIZE)
    try:
        model = model.to(device)
        X, _ = get_batch("train", batch_size=2, context_length=16, device=device)
        output = model(X)

        assert output.shape[:2] == X.shape
    except RuntimeError as e:
        if device == "cuda" and not torch.cuda.is_available():
            # CUDA not available = expected failure
            pytest.skip("CUDA not available on this system")
        else:
            raise e
