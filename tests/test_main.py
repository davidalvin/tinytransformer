import torch
from tinytransformer.main import run_training

def test_main_runs_one_step(tmp_path):
    def dummy_get_batch(split, batch_size, context_length, device):
        vocab_size = 2048
        X = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        Y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        return X, Y

    save_path = tmp_path / "tiny.pt"

    run_training(
        num_steps=1,
        batch_size=2,
        context_length=8,
        lr=1e-3,
        device="cpu",
        get_batch_fn=dummy_get_batch,
        model_path=str(save_path),
    )

    assert save_path.exists(), "Model file was not written"
