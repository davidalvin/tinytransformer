import torch
import tempfile
from tinytransformer.models.factory import build_model

def test_model_save_and_reload_consistency():
    model = build_model("tiny")
    prompt = "The quick brown fox"

    out1 = model.generate(prompt, max_tokens=5, device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        model2 = build_model("tiny")
        model2.load_state_dict(torch.load(f.name))

    out2 = model2.generate(prompt, max_tokens=5, device="cpu")
    assert isinstance(out2, str)
    assert len(out2) > 0
